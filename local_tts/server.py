"""本地 CosyVoice3 HTTP 服务（T05）。

启动后会加载 `Fun-CosyVoice3-0.5B-2512`，缓存一个默认音色（官方示例的
`asset/zero_shot_prompt.wav` + 对应提示文本），然后对外提供：

    POST /synth   {"text": "..."}  ->  流式 16kHz / 16-bit / 单声道 小端 PCM
    GET  /health                    ->  模型是否就绪

模型内部是 24kHz。为了让下游只处理 16kHz，服务用 soxr 的**流式重采样器**
（`ResampleStream`）逐块转换，而不是每块独立重采样，避免块边界出现咔哒声。

接口与取消语义在 `services/tts_local.py` 的客户端侧实现；本服务按请求串行推理
（一把进程内锁），T05 只要求单路实时合成。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OUT_SAMPLE_RATE = 16000
DEFAULT_PORT = 50000
DEFAULT_SPK_ID = "assistant"
# 官方 CosyVoice3 示例用的参考音色与提示文本（asset/zero_shot_prompt.wav）。
DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
)
MAX_TEXT_CHARS = 500

SYNTH_LOCK = threading.Lock()


class SynthRequest(BaseModel):
    text: str


def cosyvoice_root() -> Path:
    """CosyVoice 官方仓库的位置，默认 ~/code/ai/CosyVoice（不入库）。"""
    return Path(os.environ.get("COSYVOICE_ROOT", Path.home() / "code/ai/CosyVoice")).expanduser()


def compute_voice_id(prompt_wav: str | Path, prompt_text: str) -> str:
    """参考音频内容 + 提示文本的 sha1 前 12 位，供 T06 缓存键使用。"""
    digest = hashlib.sha1(Path(prompt_wav).read_bytes() + prompt_text.encode("utf-8"))
    return digest.hexdigest()[:12]


def load_engine(
    model_dir: str | Path,
    prompt_wav: str | Path,
    prompt_text: str,
    fp16: bool,
    use_vllm: bool = False,
) -> Any:
    """加载官方 AutoModel，并把参考音色缓存成 spk2info（省掉每次请求的提示编码）。

    T05b：上游 `CosyVoice2Model.tts()` 在流式循环里会把 `self.token_hop_len`
    从 25 逐块翻倍到 100，且请求之间不复位。这里保存加载时的初始值，
    由 `iter_pcm()` 在每次请求开始前恢复。
    """
    root = cosyvoice_root()
    for path in (str(root / "third_party/Matcha-TTS"), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from cosyvoice.cli.cosyvoice import AutoModel

    engine = AutoModel(model_dir=str(model_dir), fp16=fp16, load_vllm=use_vllm)
    engine.add_zero_shot_spk(prompt_text, str(prompt_wav), DEFAULT_SPK_ID)

    engine.initial_token_hop_len = int(engine.model.token_hop_len)
    prompt_tokens = int(engine.frontend.spk2info[DEFAULT_SPK_ID]["llm_prompt_speech_token"].shape[1])
    engine.prompt_token_count = prompt_tokens
    engine.prompt_token_pad = (-prompt_tokens) % engine.initial_token_hop_len
    engine.voice_id = compute_voice_id(prompt_wav, prompt_text)
    install_cancel_hook(engine)
    print(
        f"reference voice: id={engine.voice_id} prompt_tokens={prompt_tokens} "
        f"pad={engine.prompt_token_pad} initial_hop={engine.initial_token_hop_len}",
        flush=True,
    )
    return engine


def install_cancel_hook(engine: Any) -> None:
    """包装上游 `engine.model.llm.inference`，让它能被当前请求的 stop Event 提前结束。

    上游 `llm_job()` 只是遍历这个生成器，结束后设置 `llm_end_dict[uuid]=True`；
    `tts()` 看到结束后会对剩余 token 做一次 finalize 就退出。因此这里让生成器
    提前 return，上游会自己正常收尾，不需要改 CosyVoice 源码。
    """
    llm = engine.model.llm
    if getattr(llm, "_cosyvoice_original_inference", None) is not None:
        return
    original = llm.inference
    llm._cosyvoice_original_inference = original

    def cancellable_inference(*args: Any, **kwargs: Any):
        iterator = original(*args, **kwargs)
        try:
            for token in iterator:
                stop_event = getattr(engine, "stop_event", None)
                if stop_event is not None and stop_event.is_set():
                    iterator.close()
                    return
                yield token
        finally:
            try:
                iterator.close()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响上游收尾
                pass

    llm.inference = cancellable_inference


def pcm16_bytes(samples: np.ndarray) -> bytes:
    """float32 [-1, 1] 波形 -> 16-bit 小端 PCM。"""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def iter_pcm(engine: Any, text: str, stop_event: threading.Event | None = None) -> Iterator[bytes]:
    """官方流式推理 -> 16kHz PCM 块。整个过程持锁，避免两路请求同时用一块 GPU。

    T05c：`stop_event` 由 HTTP 端在检测到客户端断开时置位；包装后的
    `llm.inference` 会据此提前结束，上游再做一次 finalize 收尾。
    """
    import soxr

    with SYNTH_LOCK:
        # T05c：每个请求一个停止开关，供包装后的 llm.inference 检查。
        engine.stop_event = stop_event if stop_event is not None else threading.Event()
        # T05b：复位上游在流式循环里被翻倍且不复位的实例属性。
        engine.model.token_hop_len = engine.initial_token_hop_len
        if engine.sample_rate == OUT_SAMPLE_RATE:
            resampler = None
        else:
            resampler = soxr.ResampleStream(
                engine.sample_rate, OUT_SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )

        iterator = engine.inference_zero_shot(
            text, "", "", zero_shot_spk_id=DEFAULT_SPK_ID, stream=True
        )
        try:
            for output in iterator:
                speech = output["tts_speech"].squeeze(0).float().cpu().numpy()
                samples = (
                    speech if resampler is None else resampler.resample_chunk(speech, last=False)
                )
                data = pcm16_bytes(samples)
                if data:
                    yield data

            if resampler is not None:
                tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                data = pcm16_bytes(tail)
                if data:
                    yield data
        except GeneratorExit:
            # 客户端取消/断开：先让 LLM 生成器提前结束（T05c），再把上游剩余
            # 的收尾工作跑完（不 yield）。T05b 的“跑完再放锁”保护保留，
            # 但正常情况下只剩一块音频的 flow/hift，几百毫秒内结束。
            engine.stop_event.set()
            cleanup_started = time.perf_counter()
            try:
                for _ in iterator:
                    pass
            except Exception:  # noqa: BLE001 - 收尾失败不应盖住取消语义
                pass
            finally:
                print(
                    f"cancel cleanup {((time.perf_counter() - cleanup_started) * 1000):.0f}ms",
                    flush=True,
                )
            raise


def create_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.perf_counter()
        engine = load_engine(
            args.model_dir, args.prompt_wav, args.prompt_text, args.fp16, args.vllm
        )
        # 预热：首轮推理会触发 CUDA kernel autotune，不计入压测。
        for _ in iter_pcm(engine, "您好。"):
            pass
        app.state.engine = engine
        app.state.model_sample_rate = engine.sample_rate
        app.state.voice_id = engine.voice_id
        app.state.initial_token_hop_len = engine.initial_token_hop_len
        app.state.prompt_token_count = engine.prompt_token_count
        app.state.prompt_token_pad = engine.prompt_token_pad
        app.state.load_s = time.perf_counter() - started
        print(
            f"local tts ready: model={args.model_dir} sr={engine.sample_rate} "
            f"load={app.state.load_s:.1f}s",
            flush=True,
        )
        try:
            yield
        finally:
            app.state.engine = None

    app = FastAPI(title="local-cosyvoice3-tts", lifespan=lifespan)

    async def _watch_disconnect(request: Request, stop_event: threading.Event) -> None:
        """客户端断开时立刻置位停止开关，而不是等下一次写 socket 失败。"""
        while not stop_event.is_set():
            if await request.is_disconnected():
                stop_event.set()
                return
            await asyncio.sleep(0.02)

    async def _stream_with_cancel(
        request: Request, sync_iterator: Iterator[bytes], stop_event: threading.Event
    ):
        """把同步推理线程的产出转成异步流，同时监视客户端断开。

        推理跑在单独线程里（保持上游的线程/CUDA 上下文稳定），用无界队列交接；
        断开后置位 stop_event，推理线程会在当前块算完后收尾，锁随之释放。
        """
        chunks: queue.Queue = queue.Queue()
        sentinel = object()

        def worker() -> None:
            try:
                for chunk in sync_iterator:
                    chunks.put(chunk)
            except BaseException as exc:  # noqa: BLE001 - 交给消费端再抛
                chunks.put(exc)
            finally:
                chunks.put(sentinel)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        watcher = asyncio.create_task(_watch_disconnect(request, stop_event))
        loop = asyncio.get_running_loop()
        finished_normally = False
        try:
            while True:
                item = await loop.run_in_executor(None, chunks.get)
                if item is sentinel:
                    finished_normally = True
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            was_cancelled = not finished_normally
            cleanup_started = time.perf_counter()
            stop_event.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            await asyncio.to_thread(worker_thread.join, 5.0)
            if was_cancelled:
                print(
                    f"cancel cleanup {((time.perf_counter() - cleanup_started) * 1000):.0f}ms",
                    flush=True,
                )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if app.state.engine is not None else "loading",
            "sample_rate": OUT_SAMPLE_RATE,
            "model_sample_rate": getattr(app.state, "model_sample_rate", None),
            "load_s": round(getattr(app.state, "load_s", 0.0), 2),
            "fp16": args.fp16,
            "voice_id": getattr(app.state, "voice_id", None),
            "initial_token_hop_len": getattr(app.state, "initial_token_hop_len", None),
            "prompt_token_count": getattr(app.state, "prompt_token_count", None),
            "prompt_token_pad": getattr(app.state, "prompt_token_pad", None),
            "vllm": args.vllm,
        }

    @app.post("/synth")
    async def synth(payload: SynthRequest, request: Request) -> StreamingResponse:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=413, detail=f"text longer than {MAX_TEXT_CHARS} chars")
        engine = getattr(app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="model not ready")
        stop_event = threading.Event()
        return StreamingResponse(
            _stream_with_cancel(request, iter_pcm(engine, text, stop_event), stop_event),
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(OUT_SAMPLE_RATE)},
        )

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = cosyvoice_root()
    parser = argparse.ArgumentParser(description="本地 CosyVoice3 TTS 服务")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "COSYVOICE_MODEL_DIR", str(root / "pretrained_models/Fun-CosyVoice3-0.5B")
        ),
        help="模型目录（默认 <CosyVoice>/pretrained_models/Fun-CosyVoice3-0.5B）",
    )
    parser.add_argument(
        "--prompt-wav",
        default=os.environ.get("COSYVOICE_PROMPT_WAV", str(root / "asset/zero_shot_prompt.wav")),
        help="参考音色 wav",
    )
    parser.add_argument(
        "--prompt-text",
        default=os.environ.get("COSYVOICE_PROMPT_TEXT", DEFAULT_PROMPT_TEXT),
        help="参考音色的提示文本",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="半精度推理（默认开；T05 实测比 fp32 快约 30%%）",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="用 vLLM 跑 LLM 部分（需要单独的 cosyvoice_vllm 环境，见 README）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

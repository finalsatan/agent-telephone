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
import hashlib
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
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
    print(
        f"reference voice: id={engine.voice_id} prompt_tokens={prompt_tokens} "
        f"pad={engine.prompt_token_pad} initial_hop={engine.initial_token_hop_len}",
        flush=True,
    )
    return engine


def pcm16_bytes(samples: np.ndarray) -> bytes:
    """float32 [-1, 1] 波形 -> 16-bit 小端 PCM。"""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def iter_pcm(engine: Any, text: str) -> Iterator[bytes]:
    """官方流式推理 -> 16kHz PCM 块。整个过程持锁，避免两路请求同时用一块 GPU。"""
    import soxr

    with SYNTH_LOCK:
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
            # 客户端取消/断开：不能立刻丢下上游生成器——它内部的 LLM 线程还在跑，
            # 与下一请求并发使用同一模型会触发 NaN 并永久卡住（T05b 实测）。
            # 这里把剩余部分跑完（不再 yield），让上游线程收尾后再释放锁。
            try:
                for _ in iterator:
                    pass
            except Exception:  # noqa: BLE001 - 收尾失败不应盖住取消语义
                pass
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
    def synth(request: SynthRequest) -> StreamingResponse:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=413, detail=f"text longer than {MAX_TEXT_CHARS} chars")
        engine = getattr(app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="model not ready")
        return StreamingResponse(
            iter_pcm(engine, text),
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

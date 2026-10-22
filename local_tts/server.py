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


def load_engine(
    model_dir: str | Path,
    prompt_wav: str | Path,
    prompt_text: str,
    fp16: bool,
) -> Any:
    """加载官方 AutoModel，并把参考音色缓存成 spk2info（省掉每次请求的提示编码）。"""
    root = cosyvoice_root()
    for path in (str(root / "third_party/Matcha-TTS"), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from cosyvoice.cli.cosyvoice import AutoModel

    engine = AutoModel(model_dir=str(model_dir), fp16=fp16)
    engine.add_zero_shot_spk(prompt_text, str(prompt_wav), DEFAULT_SPK_ID)
    return engine


def pcm16_bytes(samples: np.ndarray) -> bytes:
    """float32 [-1, 1] 波形 -> 16-bit 小端 PCM。"""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def iter_pcm(engine: Any, text: str) -> Iterator[bytes]:
    """官方流式推理 -> 16kHz PCM 块。整个过程持锁，避免两路请求同时用一块 GPU。"""
    import soxr

    with SYNTH_LOCK:
        if engine.sample_rate == OUT_SAMPLE_RATE:
            resampler = None
        else:
            resampler = soxr.ResampleStream(
                engine.sample_rate, OUT_SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )

        for output in engine.inference_zero_shot(
            text, "", "", zero_shot_spk_id=DEFAULT_SPK_ID, stream=True
        ):
            speech = output["tts_speech"].squeeze(0).float().cpu().numpy()
            samples = speech if resampler is None else resampler.resample_chunk(speech, last=False)
            data = pcm16_bytes(samples)
            if data:
                yield data

        if resampler is not None:
            tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
            data = pcm16_bytes(tail)
            if data:
                yield data


def create_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.perf_counter()
        engine = load_engine(args.model_dir, args.prompt_wav, args.prompt_text, args.fp16)
        # 预热：首轮推理会触发 CUDA kernel autotune，不计入压测。
        for _ in iter_pcm(engine, "您好。"):
            pass
        app.state.engine = engine
        app.state.model_sample_rate = engine.sample_rate
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

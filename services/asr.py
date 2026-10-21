"""ASR 服务：硅基流动的 OpenAI 兼容语音转写接口（T03）。

接口形态是**整段上传**（`POST {base_url}/audio/transcriptions`，multipart 字段
`file` + `model`），没有 partial（设计文档 16.1 已核实），因此一轮的识别耗时全部
落在关键路径上。

- `transcribe(pcm16k)`：把 16kHz 单声道 16-bit PCM 封成 WAV 上传，返回 `AsrResult`
  （`text` / `latency_ms` / `audio_ms`）。
- `load_wav_pcm()`：把任意 wav 读成 16kHz 单声道 16-bit PCM，供 bench 与 live 测试
  共用，避免各写一份。
- 重试策略与 T02 一致：429 与 5xx 指数退避重试最多 2 次，其它 4xx 不重试。
- 号码规范化不在这里做：`core/normalize.py` 负责（设计文档第 4 节）。
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import wave
from typing import Any

import httpx

from core.types import AsrResult

API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_S = 0.5
DEFAULT_SAMPLE_RATE = 16000
TRANSCRIPTIONS_PATH = "/audio/transcriptions"


class AsrError(RuntimeError):
    """ASR 调用失败：重试用尽、响应不可用或缺少密钥。"""


# --------------------------------------------------------------------------- #
# 音频封装与读取
# --------------------------------------------------------------------------- #


def pcm_to_wav(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """把 16-bit 单声道 PCM 封成 WAV 字节串（接口要求上传音频文件而非裸 PCM）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buf.getvalue()


def pcm_duration_ms(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """16-bit 单声道 PCM 的时长（毫秒）。"""
    return len(pcm) / (2 * sample_rate) * 1000.0


def load_wav_pcm(path: str | os.PathLike[str], sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """读取 wav 并转成 16kHz 单声道 16-bit PCM。

    多声道取平均，采样率不同则用线性插值重采样（够用于测试与压测；正式链路拿到的
    本来就是 16k 单声道，不会走到重采样分支）。
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != sample_rate and len(mono) > 0:
        n_out = max(1, int(round(len(mono) * sample_rate / sr)))
        mono = np.interp(
            np.linspace(0.0, len(mono) - 1, n_out),
            np.arange(len(mono), dtype="float64"),
            mono,
        )
    pcm = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2")
    return pcm.tobytes()


# --------------------------------------------------------------------------- #
# HTTP 辅助（与 services/llm.py 的同类逻辑保持一致，不跨模块复用以免耦合）
# --------------------------------------------------------------------------- #


class _HttpStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _HttpStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return isinstance(exc, httpx.TransportError)


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #


class AsrClient:
    """实现 0.3 的 ASR 接口（`async transcribe(pcm16k) -> AsrResult`）。

    api_key 缺省从环境变量 SILICONFLOW_API_KEY 读取；密钥不得写进代码或配置。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY_S,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise AsrError(
                f"missing API key: set the {API_KEY_ENV} environment variable "
                "(keys must never be hardcoded or committed)"
            )
        self.model = model
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        # 整段上传是单次请求，httpx 的 read timeout 就等于整次调用的上限（没有流式 chunk）。
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "AsrClient":
        """用 config.yaml 的内容构造（tools/ 与 tests/live/ 用）；kwargs 优先。"""
        asr_cfg = config.get("asr") or {}
        sf_cfg = config.get("siliconflow") or {}
        audio_cfg = config.get("audio") or {}
        kwargs.setdefault("base_url", sf_cfg.get("base_url", ""))
        kwargs.setdefault("model", asr_cfg.get("model", ""))
        kwargs.setdefault("sample_rate", audio_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
        return cls(**kwargs)

    async def transcribe(self, pcm16k: bytes) -> AsrResult:
        """整段识别。空音频直接返回空结果，不发请求（省配额，也不需要重试）。"""
        audio_ms = pcm_duration_ms(pcm16k, self.sample_rate)
        if not pcm16k:
            return AsrResult(text="", latency_ms=0.0, audio_ms=0.0)

        wav = pcm_to_wav(pcm16k, self.sample_rate)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = await self._client.post(
                    TRANSCRIPTIONS_PATH,
                    files={"file": ("audio.wav", wav, "audio/wav")},
                    data={"model": self.model},
                )
            except httpx.TransportError as exc:  # 连接失败、读超时
                last_exc = exc
            else:
                if response.status_code >= 400:
                    last_exc = _HttpStatusError(response.status_code, response.text[:500])
                else:
                    text = _parse_transcript(response.text)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    return AsrResult(text=text, latency_ms=latency_ms, audio_ms=audio_ms)

            if attempt < self.max_retries and _is_retryable(last_exc):
                await asyncio.sleep(self.retry_base_delay * (2**attempt))
                continue
            break

        raise AsrError(f"transcription failed after {self.max_retries + 1} attempt(s): {last_exc}")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsrClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


def _parse_transcript(body: str) -> str:
    """从响应体里取 `text`。字段缺失或类型不对说明接口返回异常，按失败处理。"""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AsrError(f"transcription response is not JSON: {body[:200]}") from exc
    if not isinstance(payload, dict):
        raise AsrError(f"transcription response is not an object: {body[:200]}")
    text = payload.get("text")
    if not isinstance(text, str):
        raise AsrError(f"transcription response has no text field: {body[:200]}")
    return text.strip()

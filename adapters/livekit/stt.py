"""T16：ASR 的 LiveKit 适配层（薄封装）。

LiveKit 的 `stt.STT` 抽象对非流式后端有两种接法：

1. `stt.StreamAdapter(stt=<非流式 STT>, vad=<VAD>)` —— 按 VAD 的 END_OF_SPEECH 切段，
   每段调一次 `recognize()`；
2. 只实现 `_recognize_impl()`，让 `Agent.default.stt_node()` 在发现
   `capabilities.streaming == False` 时自动用 `StreamAdapter` 包一层。

用第 2 种：少一层自写代码，行为与框架一致（`voice/agent.py:510`）。
`StreamAdapter` 会把一轮里所有发声段的 `FINAL_TRANSCRIPT` 依次交上来，框架自己把它们
累加成整轮文本，所以“一轮多段发声不丢后半句”是框架原生行为（见 notes 1.6）。

这里只做两件事：把 `services.asr.AsrClient` 的整段识别包成 `SpeechEvent`，
以及把 T10 格式的 `asr_request` / `asr_final` / `error` 写进同一份 `EventLogger`。

**识别失败或转写为空时不能抛异常**：`StreamAdapterWrapper._recognize()` 里没有 try，
抛出去会让整条 STT 流结束。这里返回空文本的 `SpeechEvent`（框架会忽略它），
由 `LiveKitCall` 监听 `user_transcription_timeout` 播兜底话术——与 T15 的
`AsrFailedFrame` 路径同义（见 notes 4.x）。
"""

from __future__ import annotations

from typing import Any

from livekit import rtc
from livekit.agents import stt

from core.events import EventLogger, monotonic_ms

DEFAULT_SAMPLE_RATE = 16000


class SiliconFlowSTT(stt.STT):
    """非流式 STT：一段音频 → 一次 `services.asr.AsrClient.transcribe()`。"""

    def __init__(
        self,
        asr: Any,
        events: EventLogger,
        *,
        model: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._asr = asr
        self._event_log = events
        self._model = model or "unknown"
        self._sample_rate = int(sample_rate)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "siliconflow"

    async def _recognize_impl(
        self,
        buffer: Any,
        *,
        language: Any = None,
        conn_options: Any = None,
    ) -> stt.SpeechEvent:
        pcm = _to_pcm(buffer)
        audio_ms = len(pcm) / 2 / self._sample_rate * 1000.0
        self._event_log.log("asr_request", audio_ms=round(audio_ms, 2), start_ms=None, end_ms=None)
        started = monotonic_ms()
        try:
            result = await self._asr.transcribe(pcm)
        except Exception as exc:  # noqa: BLE001 - 单段失败不该让整条链路挂掉
            self._event_log.log(
                "error", stage="asr", reason=type(exc).__name__, message=str(exc)[:200]
            )
            return _empty_event()

        latency = getattr(result, "latency_ms", None)
        if latency is None:
            latency = monotonic_ms() - started
        text = (getattr(result, "text", "") or "").strip()
        self._event_log.log(
            "asr_final",
            text=text,
            latency_ms=latency,
            audio_ms=getattr(result, "audio_ms", round(audio_ms, 2)),
        )
        if not text:
            # 空转写：返回空文本事件（框架会忽略），由 user_transcription_timeout 兜底。
            self._event_log.log("error", stage="asr", reason="empty_transcript")
            return _empty_event()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="zh", text=text)],
        )


def _to_pcm(buffer: Any) -> bytes:
    if isinstance(buffer, rtc.AudioFrame):
        return bytes(buffer.data)
    if isinstance(buffer, list):
        if not buffer:
            return b""
        return bytes(rtc.combine_audio_frames(buffer).data)
    raise TypeError(f"unsupported audio buffer: {type(buffer)!r}")


def _empty_event() -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="zh", text="")],
    )

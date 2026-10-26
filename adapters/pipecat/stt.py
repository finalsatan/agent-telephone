"""T15：ASR 的 Pipecat 适配层（薄封装）。

Pipecat 的 `SegmentedSTTService` 已经实现了本项目需要的“整段识别”流程：靠 VAD 的
`VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` 切出一段发声，拼成一段音频
后交给 `run_stt()`（硅基流动的 ASR 就是整段上传，没有 partial，见设计文档 3.1）。
因此这里只做两件事：

1. 把 `run_stt()` 收到的裸 PCM 转交给 `services.asr.AsrClient`；
2. 把 T10 格式的 `asr_request` / `asr_final` 事件写进同一份 `EventLogger`。

`wants_wav_segments=False`：`AsrClient.transcribe()` 自己会把裸 PCM 封成 WAV 再上传，
拿到的应该是裸 PCM 而不是 Pipecat 默认给的 WAV 容器。
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.settings import STTSettings

from core.events import EventLogger, monotonic_ms


class SiliconFlowSTTService(SegmentedSTTService):
    """`SegmentedSTTService` + 硅基流动 ASR（`services.asr.AsrClient`）。"""

    def __init__(
        self,
        asr: Any,
        events: EventLogger,
        *,
        model: str | None = None,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> None:
        # 不要把音频继续往下游送：对话处理器只看转写文本。
        kwargs.setdefault("audio_passthrough", False)
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._asr = asr
        self._events = events
        # `self.sample_rate` 要等 Pipecat setup 后才有值；这里留一份构造时的值，
        # 供单元测试直接调 run_stt()（未 setup）时算时长。
        self._init_sample_rate = int(sample_rate)
        self._settings = STTSettings(model=model)

    @property
    def _pcm_sample_rate(self) -> int:
        return int(self.sample_rate or self._init_sample_rate)

    @property
    def wants_wav_segments(self) -> bool:
        # AsrClient 要裸 PCM，自己封 WAV。
        return False

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """把一段裸 PCM 交给 ASR，产出 `TranscriptionFrame`。"""
        audio_ms = len(audio) / 2 / self._pcm_sample_rate * 1000.0
        self._events.log("asr_request", audio_ms=round(audio_ms, 2), start_ms=None, end_ms=None)
        started = monotonic_ms()
        try:
            result = await self._asr.transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - 单段识别失败不该让整条链路挂掉
            self._events.log(
                "error", stage="asr", reason=type(exc).__name__, message=str(exc)[:200]
            )
            return
        # `transcribe()` 的 latency_ms 从请求发出算起，包含重试与退避（T03b）。
        latency = getattr(result, "latency_ms", None)
        if latency is None:
            latency = monotonic_ms() - started
        self._events.log(
            "asr_final",
            text=result.text,
            latency_ms=latency,
            audio_ms=getattr(result, "audio_ms", round(audio_ms, 2)),
        )
        if not (result.text or "").strip():
            return
        yield TranscriptionFrame(
            text=result.text,
            user_id="",
            timestamp=str(time.time()),
        )

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

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import (
    DataFrame, SystemFrame, Frame, TranscriptionFrame, VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.settings import STTSettings

from core.events import EventLogger, monotonic_ms


@dataclass
class AsrFailedFrame(DataFrame):
    """一段识别失败 / 转写为空（T15b）。

    分段 STT 在 ASR 报错或返回空文本时不会产出 `TranscriptionFrame`，下游的对话
    处理器就永远等不到这一轮。用这个帧通知它播兜底话术（与 `CallSession` 的
    `_reply_with_fallback` 一致）。`reason` 只作诊断用，"""

    reason: str = "asr_failed"


@dataclass
class AsrSegmentStartedFrame(SystemFrame):
    """一段发声的编号；在转写与轮次结束之前登记。"""

    segment_id: int = 0


@dataclass
class AsrSegmentFinishedFrame(SystemFrame):
    """该段的终态；SystemFrame 避免被抢话清掉数据队列后永远等待。"""

    segment_id: int = 0
    text: str = ""
    failed: bool = False


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
        # VAD-start IDs remain open until VAD-stop. Only IDs paired with audio
        # actually queued for SegmentedSTTService are available to run_stt().
        self._open_segment_ids: deque[int] = deque()
        self._segment_ids: deque[int] = deque()
        self._next_segment_id = 0
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

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, VADUserStartedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self._next_segment_id += 1
            self._open_segment_ids.append(self._next_segment_id)
            # SystemFrame：必须先于轮次结束广播抵达下游，转写则在后台按段返回。
            await self.push_frame(AsrSegmentStartedFrame(segment_id=self._next_segment_id))
        await super().process_frame(frame, direction)

    async def _handle_user_stopped_speaking(self, frame) -> None:
        """Pair every VAD start with a queued transcript or an explicit failure.

        Pipecat 1.13's SegmentedSTTService returns early when unusable, before
        queueing audio for run_stt(). Keep the ID open until this decision so a
        dropped segment cannot consume the next segment's ID.
        """
        self._user_speaking = False
        segment_id = self._open_segment_ids.popleft() if self._open_segment_ids else None
        if not self.is_usable:
            self._audio_buffer.clear()
            if segment_id is not None:
                await self.push_frame(
                    AsrSegmentFinishedFrame(segment_id=segment_id, failed=True)
                )
            return

        pcm = bytes(self._audio_buffer) + self._trailing_silence()
        self._audio_buffer.clear()
        if not pcm:
            if segment_id is not None:
                await self.push_frame(
                    AsrSegmentFinishedFrame(segment_id=segment_id, failed=True)
                )
            return

        self._record_stt_audio_usage(pcm)
        await self.emit_stt_usage_metrics()
        audio = pcm_to_wav(pcm, self.sample_rate) if self.wants_wav_segments else pcm
        if segment_id is not None:
            self._segment_ids.append(segment_id)
        await self._segment_queue.put(audio)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        segment_id = self._segment_ids.popleft() if self._segment_ids else None
        text = ""
        failed = False
        try:
            async for frame in self._transcribe(audio):
                if isinstance(frame, TranscriptionFrame):
                    text = frame.text
                elif isinstance(frame, AsrFailedFrame):
                    failed = True
                yield frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - release the barrier on segment errors
            failed = True
            self._events.log(
                "error", stage="asr", reason=type(exc).__name__, message=str(exc)[:200]
            )
        if segment_id is not None:
            yield AsrSegmentFinishedFrame(segment_id=segment_id, text=text, failed=failed)

    async def _transcribe(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
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
            yield AsrFailedFrame(reason=type(exc).__name__)
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
            yield AsrFailedFrame(reason="empty_transcript")
            return
        yield TranscriptionFrame(
            text=result.text,
            user_id="",
            timestamp=str(time.time()),
        )

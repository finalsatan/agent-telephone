"""T16：LiveKit 适配用的音频输入/输出。

LiveKit 的 `AgentSession` 允许把 `session.input.audio` / `session.output.audio` 换成
自己的实现，此时**不需要连 livekit-server**（`voice/agent_session.py:877`）。这样
管线级测试与线上跑的是同一份代码，测试里不用起服务、也不用改接线。

- `QueueAudioInput`：外部按实时节奏把 PCM 推进队列，`AgentSession` 从这里取帧。
  浏览器路径（房间里的来电者音轨）和场景回放（`tools/lk_client.py`）都推到它上面。
- `PacedAudioSink`：按实时节奏播出，并按框架的播放事件语义上报进度
  （`playback_started` / `playback_progressed` / `playback_finished`）。
  框架用这些事件算出“这一句实际播到哪”，我们用它维护历史与 `interrupted` 事件
  （见 notes 1.4 与 5 节）。

**为什么自己按实时节奏播**：`PacedAudioSink` 收帧时不等待（框架可以一次把整句推完），
由内部任务按帧时长 sleep 后逐帧写出。这样 `playback_finished` 的
`playback_position` 就是真正播出的秒数，抢话时框架能给出准确的“播到哪”。
房间路径把它写进 `rtc.AudioSource`，测试路径只记录。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from livekit import rtc
from livekit.agents.voice import io

SAMPLE_RATE = 16000
# 帧长与 T13 的浏览器采集一致（20ms）。
FRAME_MS = 20
# 帧上携带“这段音频要说的话”的 key（tts_node 写入，sink 读出）。
TEXT_KEY = "lk_text"


def frames_from_pcm(pcm: bytes, *, sample_rate: int = SAMPLE_RATE, frame_ms: int = FRAME_MS):
    """把任意长度的 16-bit 单声道 PCM 切成等长帧（最后不足一帧的补零）。"""
    frame_bytes = int(sample_rate * frame_ms / 1000) * 2
    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + bytes(frame_bytes - len(chunk))
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(chunk) // 2,
        )


class QueueAudioInput(io.AudioInput):
    """外部推入 PCM，`AgentSession` 逐帧取走。"""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE, label: str = "queue") -> None:
        super().__init__(label=label)
        self._sample_rate = int(sample_rate)
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def push_pcm(self, pcm: bytes) -> None:
        """推入一段 PCM（任意长度）；调用方负责按实时节奏推。"""
        if self._closed:
            return
        for frame in frames_from_pcm(pcm, sample_rate=self._sample_rate):
            self._queue.put_nowait(frame)

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._closed:
            self._queue.put_nowait(frame)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    def __aiter__(self) -> "QueueAudioInput":
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


class PacedAudioSink(io.AudioOutput):
    """按实时节奏播出的 sink；每段（`flush()` 之间）单独上报播放进度。"""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        label: str = "paced",
        write: Callable[[rtc.AudioFrame], Awaitable[None]] | None = None,
        on_play_start: Callable[[str], None] | None = None,
        on_play_end: Callable[[str, float, bool], None] | None = None,
        on_frame: Callable[[rtc.AudioFrame], None] | None = None,
    ) -> None:
        super().__init__(
            label=label,
            capabilities=io.AudioOutputCapabilities(pause=False),
            sample_rate=sample_rate,
        )
        self._write = write
        self._on_play_start = on_play_start
        # (text, played_seconds, interrupted) —— 段结束时回调。
        self._on_play_end = on_play_end
        self._on_frame = on_frame
        self._queue: deque[rtc.AudioFrame] = deque()
        self._wake = asyncio.Event()
        self._flush_pending = False
        self._clear_requested = False
        self._segment_open = False
        self._playing = False
        self._played_s = 0.0
        self._started_at: float | None = None
        self._next_at: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        # 当前段要播的文本（来自帧的 `userdata["lk_text"]`，由 tts_node 打上）。
        self._segment_text = ""

    # ---- 生命周期 -------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="paced-audio-sink")

    async def aclose(self) -> None:
        self._closed = True
        self._wake.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ---- AudioOutput 接口 ----------------------------------------------

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self._segment_open = True
        self._queue.append(frame)
        self._wake.set()

    def flush(self) -> None:
        super().flush()
        self._flush_pending = True
        self._wake.set()

    def clear_buffer(self) -> None:
        self._queue.clear()
        self._clear_requested = True
        self._wake.set()

    # ---- 内部 -----------------------------------------------------------

    async def _run(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            await self._drain()

    async def _drain(self) -> None:
        while not self._closed:
            if self._clear_requested:
                self._clear_requested = False
                self._queue.clear()
                self._flush_pending = False
                self._finish_segment(interrupted=True)
                continue
            if self._queue:
                frame = self._queue.popleft()
                await self._emit(frame)
                continue
            if self._flush_pending:
                self._flush_pending = False
                self._finish_segment(interrupted=False)
                continue
            return

    async def _emit(self, frame: rtc.AudioFrame) -> None:
        if not self._playing:
            self._playing = True
            self._segment_text = str(frame.userdata.get(TEXT_KEY, ""))
            self._started_at = time.time()
            self._next_at = self._started_at
            self.on_playback_started(created_at=self._started_at)
            if self._on_play_start is not None:
                self._on_play_start(self._segment_text)
        # 实时节奏：每帧按自己的时长推进。
        now = time.time()
        if self._next_at is not None and self._next_at > now:
            await asyncio.sleep(self._next_at - now)
        self._next_at = (self._next_at or time.time()) + frame.duration

        if self._write is not None:
            await self._write(frame)
        if self._on_frame is not None:
            self._on_frame(frame)
        if self._started_at is not None:
            self.on_playback_progressed(
                started_at=self._started_at, offset=self._played_s, duration=frame.duration
            )
        self._played_s += frame.duration

    def _finish_segment(self, *, interrupted: bool) -> None:
        if not self._segment_open:
            return
        self._segment_open = False
        played, self._played_s = self._played_s, 0.0
        text, self._segment_text = self._segment_text, ""
        self._playing = False
        self._started_at = None
        self._next_at = None
        self.on_playback_finished(playback_position=played, interrupted=interrupted)
        if self._on_play_end is not None:
            self._on_play_end(text, played, interrupted)


def audio_source_sink(
    source: rtc.AudioSource,
    **kwargs: Any,
) -> PacedAudioSink:
    """把 sink 的音频写进 LiveKit 的 `rtc.AudioSource`（房间路径）。"""

    async def _write(frame: rtc.AudioFrame) -> None:
        await source.capture_frame(frame)

    return PacedAudioSink(write=_write, label="room", **kwargs)


def pcm_ms(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / 2 / sample_rate * 1000.0


__all__ = [
    "FRAME_MS",
    "SAMPLE_RATE",
    "TEXT_KEY",
    "PacedAudioSink",
    "QueueAudioInput",
    "audio_source_sink",
    "frames_from_pcm",
    "pcm_ms",
]

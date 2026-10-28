"""T16：LiveKit 适配用的音频输入/输出。

LiveKit 的 `AgentSession` 允许把 `session.input.audio` / `session.output.audio` 换成
自己的实现，此时**不需要连 livekit-server**（`voice/agent_session.py:877`）。这样
管线级测试与线上跑的是同一份代码，测试里不用起服务、也不用改接线。

- `QueueAudioInput`：外部按实时节奏把 PCM 推进队列，`AgentSession` 从这里取帧。
  浏览器路径（房间里的来电者音轨）和场景回放（`tools/lk_client.py`）都推到它上面。
- `PacedAudioSink`：按实时节奏播出，并按框架的播放事件语义上报进度
  （`playback_started` / `playback_progressed` / `playback_finished`）。
  框架用这些事件算出“这一句实际播到哪”，我们用它维护历史与 `interrupted` 事件
  （见 notes 1.4 与 5 节）。**段边界用队列里的标记表示**（`flush()` 放一个
  `_SEGMENT_END`），这样即使两段的帧都在队列里也不会合并成一段，
  每次 `flush()` 都能对上框架的一次 `playback_finished`（notes 第 1.7 节）。

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

# 段结束标记：`flush()` 往队列里放一个，`_drain` 取到就结束当前段。
#
# 为什么不用布尔值：框架把 `flush()` / `clear_buffer()` 之间算作一段，并要求每段各报一次
# `playback_finished`（`AudioOutput.on_playback_finished()` 的 docstring 与
# `wait_for_playout()` 的计数，`livekit/agents/voice/io.py:249,265-269,278-296,322`）。
# 布尔值只能表示“有一个待处理的边界”，两段的帧一旦都在队列里就会合并成一段、少报一次。
# 1.8.9 的 `AgentActivity` 按段串行转发（`voice/agent_activity.py:3875-3877`），
# 正常路径上摸不到；但实现必须守住这条契约（notes 第 1.7 节）。
_SEGMENT_END = object()


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
        on_clear: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            label=label,
            capabilities=io.AudioOutputCapabilities(pause=False),
            sample_rate=sample_rate,
        )
        # 可换：房间路径把音频写进 `rtc.AudioSource`（见 session.py 的 room 模式）。
        self.write = write
        self._on_play_start = on_play_start
        # (text, played_seconds, interrupted) —— 段结束时回调。
        self._on_play_end = on_play_end
        self._on_frame = on_frame
        # `clear_buffer()` 时清下游已排队音频的钩子：房间模式设成 `AudioSource.clear_queue`
        # （livekit 1.1.21 有这个方法，见 notes 第 4 节），否则打断后还会多播一段。
        self.on_clear = on_clear
        self._queue: deque[Any] = deque()
        self._wake = asyncio.Event()
        # `clear_buffer()` 递增：正在 `_emit` 里 sleep 的那一帧据此丢弃，不再写出、不计时长。
        self._generation = 0
        # `clear_buffer()` 置位：`_drain` 据此把当前段按“被打断”结束。
        self._interrupt_requested = False
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
        self._queue.append(frame)
        self._wake.set()

    def flush(self) -> None:
        super().flush()
        # 段边界：之后进来的帧排在标记后面，天然属于下一段。
        self._queue.append(_SEGMENT_END)
        self._wake.set()

    def clear_buffer(self) -> None:
        # 同步清空队列（连同段结束标记）：已经推给 sink 的音频不再播出。
        # 之后推进来的帧属于新的一段，不能被这里连带清掉（第 32 次评审第 7 条）。
        self._queue.clear()
        self._generation += 1
        self._interrupt_requested = True
        if self.on_clear is not None:
            self.on_clear()
        self._wake.set()

    # ---- 内部 -----------------------------------------------------------

    async def _run(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            await self._drain()

    async def _drain(self) -> None:
        while not self._closed:
            if self._interrupt_requested:
                self._interrupt_requested = False
                self._finish_segment(interrupted=True)
                continue
            if self._queue:
                item = self._queue.popleft()
                if item is _SEGMENT_END:
                    self._finish_segment(interrupted=False)
                else:
                    await self._emit(item)
                continue
            return

    async def _emit(self, frame: rtc.AudioFrame) -> None:
        generation = self._generation
        # 实时节奏：每帧按自己的时长推进。
        now = time.time()
        if self._next_at is not None and self._next_at > now:
            await asyncio.sleep(self._next_at - now)
        if generation != self._generation:
            # sleep 期间被打断（`clear_buffer()`）：这一帧丢掉，不写出也不计时长。
            return
        if not self._playing:
            self._playing = True
            self._segment_text = str(frame.userdata.get(TEXT_KEY, ""))
            self._started_at = time.time()
            self._next_at = self._started_at
            self.on_playback_started(created_at=self._started_at)
            if self._on_play_start is not None:
                self._on_play_start(self._segment_text)
        self._next_at = (self._next_at or time.time()) + frame.duration

        if self.write is not None:
            await self.write(frame)
        if self._on_frame is not None:
            self._on_frame(frame)
        if self._started_at is not None:
            self.on_playback_progressed(
                started_at=self._started_at, offset=self._played_s, duration=frame.duration
            )
        self._played_s += frame.duration

    def _finish_segment(self, *, interrupted: bool) -> None:
        # “还有没有没结束的段”以框架的计数为准：`capture_frame()` 的第一帧自增、
        # `on_playback_finished()` 自减，与 `wait_for_playout()` 读的是同一对计数
        # （`voice/io.py:265-301`）。
        #
        # 不能用“capture 时置位、finish 时清位”的布尔值：两段的帧都在队列里时，
        # 处理前一段的段结束标记会把标志清掉，后一段就再也发不出 `playback_finished`。
        if self._pending_playback_count <= 0:
            return
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

    return PacedAudioSink(
        write=_write, label="room", on_clear=source.clear_queue, **kwargs
    )


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

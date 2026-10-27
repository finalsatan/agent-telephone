"""浏览器端待播队列的排队模型（T13b，T15b 起两套适配器共用）。

服务端不等浏览器真正播完：音频一经送出就进入浏览器队列，`play_end` 也在整句送完时
就写。因此“AI 是不是还在说话”不能看 `play_start` / `play_end`，要按送出顺序估算每句
在浏览器里的播放区间：

- 新一句从前一句队列结束（或当前时刻，取大者）开始；
- 每送出一块音频就把这一句的结束时刻往后推；
- 尾音抢话时找出当前还在播的那一句，以及它后面已送出但从未开播的句子。

T13 的 `RealtimeCall` 与 T15 的 `PipecatCall` 用同一个实现，保证两套链路的
`interrupted(tail=True, unplayed=...)` 行为一致。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.events import monotonic_ms


@dataclass
class PlaybackSegment:
    """一句 AI 音频在浏览器里的预计播放区间。"""

    text: str
    start_ms: float
    end_ms: float


class PlaybackTracker:
    """按送出顺序估算浏览器播放进度；线程/事件循环内单线程使用即可。"""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic_ms,
        prune_ms: float = 500.0,
    ) -> None:
        self._clock = clock
        self._prune_ms = float(prune_ms)
        self._segments: list[PlaybackSegment] = []
        self._current: PlaybackSegment | None = None
        self._until_ms = 0.0

    @property
    def until_ms(self) -> float:
        """队列里最后一块音频的预计结束时刻。"""
        return self._until_ms

    def playing(self) -> bool:
        """浏览器里还有没播完的音频。"""
        return self._clock() < self._until_ms

    def on_play_start(self, text: str) -> PlaybackSegment:
        """新一句开始送出：从前一句队列结束（或当前时刻）起算。"""
        start = max(self._clock(), self._until_ms)
        segment = PlaybackSegment(text=str(text or ""), start_ms=start, end_ms=start)
        self._segments.append(segment)
        self._current = segment
        self._prune()
        return segment

    def on_audio(self, pcm_ms: float, text: str = "") -> None:
        """送出一块音频：把它加到当前这一句的结束时刻上。"""
        now = self._clock()
        segment = self._current
        if segment is None or (text and segment.text != text):
            start = max(now, self._until_ms)
            segment = PlaybackSegment(text=str(text or ""), start_ms=start, end_ms=start)
            self._current = segment
            self._segments.append(segment)
        if segment.end_ms < now:
            segment.end_ms = now
        segment.end_ms += float(pcm_ms)
        self._until_ms = max(self._until_ms, segment.end_ms)

    def current_playing(self) -> PlaybackSegment | None:
        """当前（或下一句待播）的播放区间；都已经播完则返回 None。"""
        now = self._clock()
        for segment in self._segments:
            if segment.end_ms > now:
                return segment
        return None

    def interrupt(self) -> tuple[str, float | None, list[str]]:
        """尾音抢话：返回 `(被打断的句子, 已播毫秒数, 未开播的句子)` 并清空队列。"""
        now = self._clock()
        segment = self.current_playing()
        text = segment.text if segment is not None else ""
        played_ms: float | None = None
        unplayed: list[str] = []
        if segment is not None:
            played_ms = max(0.0, min(now, segment.end_ms) - segment.start_ms)
            position = self._segments.index(segment)
            unplayed = [item.text for item in self._segments[position + 1 :] if item.text]
        self.reset()
        return text, played_ms, unplayed

    def reset(self) -> None:
        self._segments.clear()
        self._current = None
        self._until_ms = 0.0

    def _prune(self) -> None:
        cutoff = self._clock() - self._prune_ms
        self._segments = [segment for segment in self._segments if segment.end_ms > cutoff]

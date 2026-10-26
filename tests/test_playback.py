"""T15b：浏览器待播队列模型（`core.playback.PlaybackTracker`）的单测。

级联链路（T13b）与 Pipecat 链路（T15b）共用这套模型；这里用可手动推进的假时钟，
覆盖“浏览器还在播时抢话”的几种情形：单句尾音、两句之间、已播完。
"""

from __future__ import annotations

from core.playback import PlaybackTracker


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, ms: float) -> None:
        self.value += ms


def test_playing_until_audio_drains() -> None:
    clock = FakeClock()
    tracker = PlaybackTracker(clock=clock)
    tracker.on_play_start("你好")
    tracker.on_audio(500.0, "你好")
    assert tracker.playing()
    assert tracker.until_ms == 1_500.0
    clock.advance(400.0)
    assert tracker.current_playing() is not None
    clock.advance(200.0)
    assert not tracker.playing()
    assert tracker.current_playing() is None


def test_new_sentence_starts_after_previous_queue() -> None:
    clock = FakeClock()
    tracker = PlaybackTracker(clock=clock)
    tracker.on_play_start("第一句")
    tracker.on_audio(1_000.0, "第一句")
    # 第二句在第一句还没播完时送出：从第一句队列结束处接着排。
    tracker.on_play_start("第二句")
    tracker.on_audio(500.0, "第二句")
    assert tracker.until_ms == 2_500.0
    clock.advance(1_200.0)  # 第一句已播完，正在播第二句
    current = tracker.current_playing()
    assert current is not None and current.text == "第二句"


def test_interrupt_reports_playing_and_unplayed() -> None:
    clock = FakeClock()
    tracker = PlaybackTracker(clock=clock)
    tracker.on_play_start("第一句")
    tracker.on_audio(1_000.0, "第一句")
    tracker.on_play_start("第二句")
    tracker.on_audio(1_000.0, "第二句")
    clock.advance(1_200.0)  # 第一句播完 200ms，进入第二句 200ms
    text, played_ms, unplayed = tracker.interrupt()
    assert text == "第二句"
    assert played_ms == 200.0
    assert unplayed == []
    assert not tracker.playing()


def test_interrupt_marks_later_sentences_unplayed() -> None:
    clock = FakeClock()
    tracker = PlaybackTracker(clock=clock)
    tracker.on_play_start("第一句")
    tracker.on_audio(1_000.0, "第一句")
    tracker.on_play_start("第二句")
    tracker.on_audio(1_000.0, "第二句")
    tracker.on_play_start("第三句")
    tracker.on_audio(1_000.0, "第三句")
    clock.advance(100.0)  # 还在播第一句
    text, played_ms, unplayed = tracker.interrupt()
    assert text == "第一句"
    assert played_ms == 100.0
    assert unplayed == ["第二句", "第三句"]


def test_interrupt_after_drain_returns_nothing() -> None:
    clock = FakeClock()
    tracker = PlaybackTracker(clock=clock)
    tracker.on_play_start("你好")
    tracker.on_audio(300.0, "你好")
    clock.advance(500.0)
    assert tracker.current_playing() is None
    text, played_ms, unplayed = tracker.interrupt()
    assert (text, played_ms, unplayed) == ("", None, [])


def test_reset_clears_queue() -> None:
    clock = FakeClock()
    tracker = PlaybackTracker(clock=clock)
    tracker.on_play_start("你好")
    tracker.on_audio(1_000.0, "你好")
    tracker.reset()
    assert not tracker.playing()
    assert tracker.current_playing() is None

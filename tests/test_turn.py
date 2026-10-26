"""T09 验收：VAD 与轮次判断。

大部分用例用 fake VAD / fake Smart Turn 驱动状态机，不加载模型；最后一条用
`scenarios/asr_02.wav` 跑真实模型（本地 CPU，不联网）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.turn import (
    FRAME_BYTES,
    FRAME_MS,
    SPEECH_END,
    SPEECH_START,
    TURN_END,
    SmartTurn,
    TurnDetector,
    TurnEvent,
    segment_wav,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "scenarios" / "asr_02.wav"

SPEECH = 3000  # 非零样本即可被 fake VAD 判为语音


def make_pcm(pattern: list[bool]) -> bytes:
    """按帧构造 PCM：True 为语音（非零），False 为纯静音。"""
    out = bytearray()
    for is_speech in pattern:
        if is_speech:
            out += np.full(512, SPEECH, dtype=np.int16).tobytes()
        else:
            out += bytes(FRAME_BYTES)
    return bytes(out)


def speech_frames(ms: int) -> list[bool]:
    return [True] * (ms // FRAME_MS)


def silence_frames(ms: int) -> list[bool]:
    return [False] * (ms // FRAME_MS)


def kinds(events: list[TurnEvent]) -> list[str]:
    return [event.kind for event in events]


class FakeVad:
    """非零样本即判为语音；prob 固定 0.0 / 0.9。"""

    def __init__(self) -> None:
        self.reset_calls = 0

    def speech_probability(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16)
        return 0.9 if np.any(samples) else 0.0

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSmartTurn:
    """按预设顺序返回 (是否说完, 概率)，并记录每次送进来的 PCM。"""

    def __init__(self, results: list[tuple[bool, float]] | None = None) -> None:
        self.results = list(results or [(True, 0.95)])
        self.calls: list[bytes] = []

    def predict(self, pcm: bytes) -> tuple[bool, float]:
        self.calls.append(pcm)
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]


def make_detector(**kwargs) -> TurnDetector:
    kwargs.setdefault("vad", FakeVad())
    kwargs.setdefault("smart_turn", FakeSmartTurn())
    return TurnDetector(**kwargs)


def test_pure_silence_emits_nothing() -> None:
    detector = make_detector()
    events = detector.feed(make_pcm(silence_frames(2000)))

    assert events == []
    assert detector.in_speech is False
    assert detector.turn_active is False


def test_speech_then_silence_emits_speech_end_and_turn_end() -> None:
    vad = FakeVad()
    smart = FakeSmartTurn([(True, 0.97)])
    detector = make_detector(vad=vad, smart_turn=smart)

    pcm = make_pcm(speech_frames(1024) + silence_frames(1000))
    events = detector.feed(pcm)

    assert kinds(events) == [SPEECH_START, SPEECH_END, TURN_END]
    assert events[0].t_ms == 0
    assert events[0].speech_ms == FRAME_MS
    # 静音累计到 224ms（7 帧）才 >= 200ms；事件 t 取达到阈值的那一帧的起点。
    assert events[1].t_ms == 1024 + 192
    assert events[1].speech_ms == 1024
    assert events[2].t_ms == events[1].t_ms
    assert events[2].reason == "smart_turn"
    assert events[2].probability == pytest.approx(0.97)
    assert len(smart.calls) == 1
    assert detector.turn_active is False


def test_short_pause_inside_speech_does_not_end_turn() -> None:
    """“嗯……我想……”式停顿：96ms < 200ms，不应出现 speech_end / turn_end。"""
    detector = make_detector()
    pattern = speech_frames(1024) + silence_frames(96) + speech_frames(512) + silence_frames(1000)

    events = detector.feed(make_pcm(pattern))

    assert kinds(events) == [SPEECH_START, SPEECH_END, TURN_END]
    assert events[0].t_ms == 0
    # 停顿期间没有第二个 speech_start，也没有 turn_end。
    assert events[1].t_ms > 1024 + 96 + 512
    assert events[1].speech_ms == 1024 + 512


def test_pause_over_speech_end_but_incomplete_turn_resumes_without_turn_end() -> None:
    """停顿超过 200ms 会发 speech_end，但 Smart Turn 判为没说完就不结束轮次。"""
    smart = FakeSmartTurn([(False, 0.2), (True, 0.93)])
    detector = make_detector(smart_turn=smart)
    pattern = (
        speech_frames(1024) + silence_frames(416) + speech_frames(512) + silence_frames(1000)
    )

    events = detector.feed(make_pcm(pattern))

    assert kinds(events) == [SPEECH_START, SPEECH_END, SPEECH_START, SPEECH_END, TURN_END]
    first_end, resume = events[1], events[2]
    assert first_end.t_ms == 1024 + 192
    assert resume.t_ms == 1024 + 416
    # 停顿期间没有 turn_end；整轮只结束一次，且由第二次 Smart Turn 判定。
    assert events[-1].reason == "smart_turn"
    assert events[-1].probability == pytest.approx(0.93)
    assert len(smart.calls) == 2
    # 第二段音频包含停顿前的语音（同一轮），不是从重新开口处截断。
    assert len(smart.calls[1]) > len(speech_frames(512)) * FRAME_BYTES


def test_incomplete_turn_falls_back_after_stop_secs() -> None:
    detector = make_detector(
        smart_turn=FakeSmartTurn([(False, 0.3)]),
        stop_secs=1.0,
    )
    pattern = speech_frames(512) + silence_frames(1500)

    events = detector.feed(make_pcm(pattern))

    assert kinds(events) == [SPEECH_START, SPEECH_END, TURN_END]
    assert events[1].t_ms == 512 + 192
    # 静音累计到 1024ms（>= stop_secs=1.0）时兜底结束；事件 t 取该帧起点。
    assert events[2].t_ms == 512 + 992
    assert events[2].reason == "stop_secs"
    assert events[2].probability is None
    assert detector.turn_active is False


def test_smart_turn_threshold_requires_higher_probability() -> None:
    """T13c：模型判“说完”(0.8)但低于 smart_turn_threshold(0.85) 时不结束本轮。"""
    pattern = speech_frames(512) + silence_frames(416) + speech_frames(512) + silence_frames(1000)

    # 阈值 0.85：第一次 0.8 不够，继续同一轮；第二次 0.9 才结束。
    detector = make_detector(
        smart_turn=FakeSmartTurn([(True, 0.8), (True, 0.9)]),
        smart_turn_threshold=0.85,
    )
    events = detector.feed(make_pcm(pattern))
    assert kinds(events) == [SPEECH_START, SPEECH_END, SPEECH_START, SPEECH_END, TURN_END]
    assert events[-1].reason == "smart_turn"
    assert events[-1].probability == pytest.approx(0.9)

    # 默认阈值 0.5：第一次就结束，后半句成了新的一轮。
    detector = make_detector(smart_turn=FakeSmartTurn([(True, 0.8), (True, 0.9)]))
    events = detector.feed(make_pcm(pattern))
    assert kinds(events)[:3] == [SPEECH_START, SPEECH_END, TURN_END]
    assert events[2].probability == pytest.approx(0.8)


def test_smart_turn_threshold_never_loosens_model_decision() -> None:
    """模型自己判没说完(False)时，即使概率高于阈值也不能结束（阈值只会更严）。"""
    detector = make_detector(
        smart_turn=FakeSmartTurn([(False, 0.9)]),
        smart_turn_threshold=0.5,
        stop_secs=1.0,
    )
    events = detector.feed(make_pcm(speech_frames(512) + silence_frames(1500)))
    assert events[-1].reason == "stop_secs"
    assert detector.last_smart_turn_probability == pytest.approx(0.9)


def test_pre_roll_is_sent_to_smart_turn() -> None:
    smart = FakeSmartTurn([(True, 0.9)])
    detector = make_detector(smart_turn=smart)
    pattern = silence_frames(512) + speech_frames(512) + silence_frames(512)

    detector.feed(make_pcm(pattern))

    pcm = smart.calls[0]
    speech_bytes = len(speech_frames(512)) * FRAME_BYTES
    # 预卷最多 pre_speech_ms（500ms，取整为 15 帧）。
    assert len(pcm) > speech_bytes
    assert len(pcm) <= (15 + 512 // FRAME_MS + 7) * FRAME_BYTES
    # 预卷是静音：第一帧全零。
    assert pcm[:FRAME_BYTES] == bytes(FRAME_BYTES)


def test_feed_accepts_unaligned_chunks() -> None:
    pattern = speech_frames(1024) + silence_frames(1000)
    pcm = make_pcm(pattern)

    whole = make_detector().feed(pcm)
    chunked_detector = make_detector()
    chunked: list[TurnEvent] = []
    for offset in range(0, len(pcm), 100):
        chunked.extend(chunked_detector.feed(pcm[offset : offset + 100]))

    assert kinds(chunked) == kinds(whole)
    assert [event.t_ms for event in chunked] == [event.t_ms for event in whole]


def test_reset_clears_state() -> None:
    vad = FakeVad()
    detector = make_detector(vad=vad)
    detector.feed(make_pcm(speech_frames(512)))

    detector.reset()

    assert detector.in_speech is False
    assert detector.turn_active is False
    assert vad.reset_calls == 1
    events = detector.feed(make_pcm(silence_frames(512)))
    assert events == []


def test_segment_wav_returns_turn_boundaries() -> None:
    detector = make_detector()
    pattern = (
        silence_frames(320)
        + speech_frames(512)
        + silence_frames(1024)
        + speech_frames(512)
        + silence_frames(1280)
    )
    duration_ms = len(pattern) * FRAME_MS

    segments = segment_wav(make_pcm(pattern), detector=detector)

    assert len(segments) == 2
    first_start, first_end = segments[0]
    second_start, second_end = segments[1]
    # 每轮起点带 500ms 预卷，但不早于 0 / 上一轮结束。
    assert first_start == 0
    assert first_end == 320 + 512 + 192
    assert second_start == max(0, 320 + 512 + 1024 - 500)
    assert second_end == 320 + 512 + 1024 + 512 + 192
    assert second_end < duration_ms


def test_segment_wav_clamps_trailing_segment_to_audio_end() -> None:
    detector = make_detector()
    pcm = make_pcm(speech_frames(512))

    segments = segment_wav(pcm, detector=detector)

    assert segments == [(0.0, 512.0)]


def test_from_config_reads_vad_section() -> None:
    config = {
        "vad": {
            "threshold": 0.7,
            "smart_turn_threshold": 0.8,
            "speech_end_silence_ms": 320,
            "stop_secs": 2.5,
            "pre_speech_ms": 300,
            "max_segment_s": 6.0,
        }
    }
    detector = TurnDetector.from_config(config, vad=FakeVad(), smart_turn=FakeSmartTurn())

    assert detector.threshold == 0.7
    assert detector.smart_turn_threshold == 0.8
    assert detector.speech_end_silence_ms == 320
    assert detector.stop_secs == 2.5
    assert detector.pre_speech_ms == 300
    assert detector.max_segment_s == 6.0


def test_invalid_event_kind_rejected() -> None:
    with pytest.raises(ValueError):
        TurnEvent("vad_start", 0.0)


async def test_smart_turn_slow_switches_to_thread_offload() -> None:
    """T13b：Smart Turn 推理超过阈值后，feed_async 改到线程里执行。"""
    import threading
    import time as _time

    class SlowSmartTurn(FakeSmartTurn):
        def __init__(self) -> None:
            super().__init__([(True, 0.9)])
            self.thread_ids: list[int] = []

        def predict(self, pcm: bytes) -> tuple[bool, float]:
            self.thread_ids.append(threading.get_ident())
            _time.sleep(0.03)  # > 20ms 阈值
            return super().predict(pcm)

    smart = SlowSmartTurn()
    detector = make_detector(smart_turn=smart, smart_turn_offload_ms=20.0)
    pcm = make_pcm(speech_frames(64) + silence_frames(400))

    # 第一次仍在事件循环线程里跑，但会触发超阈值标记。
    await detector.feed_async(pcm)
    assert detector.last_smart_turn_ms is not None
    assert detector.last_smart_turn_ms > 20.0
    assert detector.smart_turn_offload is True
    assert smart.thread_ids[0] == threading.get_ident()

    # 第二次整帧处理改到线程里，predict 的线程 id 应不同于事件循环线程。
    await detector.feed_async(pcm)
    assert smart.thread_ids[1] != threading.get_ident()


@pytest.mark.skipif(not SAMPLE.exists(), reason=f"缺少样例音频 {SAMPLE}")
def test_real_models_on_sample_wav() -> None:
    """真实 Silero VAD + Smart Turn：一段完整的话 + 1s 静音应产出 turn_end。"""
    audio, sample_rate = sf.read(SAMPLE, dtype="int16")
    assert sample_rate == 16000

    detector = TurnDetector(vad=None, smart_turn=SmartTurn())
    pcm = audio.tobytes() + bytes(16000 * 2)  # 追加 1s 静音
    events = detector.feed(pcm)

    assert kinds(events).count(SPEECH_START) >= 1
    assert TURN_END in kinds(events)
    turn_end = events[-1]
    assert turn_end.kind == TURN_END
    assert turn_end.t_ms < 2517 + 1000 + 100
    assert turn_end.probability is not None and turn_end.probability > 0.5

    segments = segment_wav(audio.tobytes())
    assert len(segments) == 1
    start, end = segments[0]
    assert start == 0
    assert end >= len(audio) / 16 - 1

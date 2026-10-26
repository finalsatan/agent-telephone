"""T15 / T15b：Pipecat 适配层的单元测试（不联网、不加载 ONNX 模型）。

覆盖：

- `SiliconFlowSTTService.run_stt()`：调用 `services.asr`，写 `asr_request` / `asr_final`，
  产出 `TranscriptionFrame`；ASR 报错 / 空转写时产出 `AsrFailedFrame`（T15b）。
- `SiliconFlowTTSService.run_tts()`：预合成缓存命中不调网络、首包 0；未命中走 `synth()`，
  写事件；CONFIRM 合成失败时退回号码片段拼接（T15b）。
- `DialogProcessor`：一轮 LLM + Dialog 产出 `TTSSpeakFrame`、写事件；抢话时取消 LLM 并把
  文本顺延（T15b）；ASR 失败播兜底；号码没报完按超时处理；`confirm_lead` 拆句。
- `ConfiguredSmartTurn`：把 `smart_turn_threshold` 应用到概率上；记录 `turn_end` 原因。
- `build_vad_params`：`interrupt_min_speech_ms` 映射到 VAD 的 `start_secs`（T15b）。
- `PipecatCall._on_barge_in()`：浏览器还在播时记尾音抢话（T15b）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame

from adapters.pipecat.llm import DialogProcessor
from adapters.pipecat.stt import AsrFailedFrame, SiliconFlowSTTService
from adapters.pipecat.tts import SiliconFlowTTSService
from core.events import EventLogger, read_events
from core.phrases import PHRASES, PhraseMissing
from core.playback import PlaybackTracker
from core.types import AsrResult, ReplyChunk, TurnResult
from services.asr import AsrError

SAMPLE_RATE = 16000


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, ms: float) -> None:
        self.value += ms


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeAsr:
    def __init__(self, text: str = "我是小李", *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[bytes] = []

    async def transcribe(self, pcm: bytes) -> AsrResult:
        self.calls.append(pcm)
        if self.fail:
            raise AsrError("boom")
        return AsrResult(text=self.text, latency_ms=12.5, audio_ms=len(pcm) / 32.0)


class FakeTts:
    def __init__(self, *, chunks: int = 2, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.calls: list[str] = []

    async def synth(self, text, cancel):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("boom")
        yield b"\x00\x00" * 160
        for _ in range(max(0, self.chunks - 1)):
            if cancel.is_set():
                return
            await asyncio.sleep(0)
            yield b"\x00\x00" * 160


class FakeCache:
    def __init__(
        self, hits: dict[str, bytes] | None = None, *, phone_pcm: bytes | None = None
    ) -> None:
        self.hits = dict(hits or {})
        self.phone_pcm = phone_pcm

    def get(self, text: str) -> bytes | None:
        return self.hits.get(text)

    def render_phone(self, number: str) -> bytes:
        if self.phone_pcm is None:
            raise PhraseMissing(f"no fragments for {number}")
        return self.phone_pcm


class FakeLlm:
    def __init__(self, reply: str = "好的", slots_update: dict[str, str] | None = None) -> None:
        self.reply = reply
        self.slots_update = dict(slots_update or {})
        self.calls: list[str] = []

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.calls.append(user_text)
        yield ReplyChunk(kind="sentence", text=self.reply, t_ms=5.0)
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply=self.reply,
                slots_update=self.slots_update,
                first_token_ms=5.0,
                total_ms=10.0,
                parse_ok=True,
                first_sentence_ms=5.0,
            ),
            t_ms=10.0,
        )


class ScriptedLlm:
    """按用户文本返回槽位（模拟 LLM 的 slots_update）。"""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[str] = []

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.calls.append(user_text)
        reply, slots_update = self.responder(user_text)
        yield ReplyChunk(kind="sentence", text=reply, t_ms=5.0)
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply=reply,
                slots_update=slots_update,
                first_token_ms=5.0,
                total_ms=10.0,
                parse_ok=True,
                first_sentence_ms=5.0,
            ),
            t_ms=10.0,
        )


class BlockingLlm:
    """第一次产出前一直等，用来在 LLM 阶段注入抢话。"""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.started.set()
        await self.release.wait()
        yield ReplyChunk(kind="sentence", text="好的", t_ms=5.0)
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply="好的", slots_update={}, first_token_ms=5.0, total_ms=10.0, parse_ok=True
            ),
            t_ms=10.0,
        )


def make_config(confirm_lead: bool = False) -> dict:
    return {
        "owner": {"name": "张先生"},
        "dialog": {"max_asks": 2, "max_confirm_rounds": 2},
        "audio": {"sample_rate": SAMPLE_RATE},
        "vad": {"stop_secs": 2.0},
        "tts": {"confirm_lead": confirm_lead},
    }


def make_dialog(config: dict | None = None):
    from core.dialog import Dialog

    return Dialog.from_config(config or make_config())


def events_for(tmp_path: Path) -> EventLogger:
    return EventLogger(tmp_path / "call")


def capture_frames(processor: DialogProcessor) -> list:
    pushed: list = []

    async def _capture(frame, direction=None):
        pushed.append(frame)

    processor.push_frame = _capture  # type: ignore[method-assign]
    return pushed


def play_texts(events) -> list[str]:
    return [
        str(event["data"].get("text") or "")
        for event in events
        if event["type"] == "play_start"
    ]


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #


async def test_stt_logs_and_yields_transcription(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    service = SiliconFlowSTTService(FakeAsr("我是小李"), events, sample_rate=SAMPLE_RATE)
    frames = [frame async for frame in service.run_stt(b"\x01\x00" * 1600)]
    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "我是小李"
    types = [event["type"] for event in read_events(tmp_path / "call")]
    assert types == ["asr_request", "asr_final"]
    final = read_events(tmp_path / "call")[1]["data"]
    assert final["text"] == "我是小李"
    assert final["latency_ms"] == 12.5


async def test_stt_error_logs_and_emits_asr_failed(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    service = SiliconFlowSTTService(FakeAsr(fail=True), events, sample_rate=SAMPLE_RATE)
    frames = [frame async for frame in service.run_stt(b"\x01\x00" * 1600)]
    assert len(frames) == 1 and isinstance(frames[0], AsrFailedFrame)
    types = [event["type"] for event in read_events(tmp_path / "call")]
    assert types == ["asr_request", "error"]


async def test_stt_empty_transcript_emits_asr_failed(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    service = SiliconFlowSTTService(FakeAsr("   "), events, sample_rate=SAMPLE_RATE)
    frames = [frame async for frame in service.run_stt(b"\x01\x00" * 1600)]
    assert len(frames) == 1 and isinstance(frames[0], AsrFailedFrame)
    assert frames[0].reason == "empty_transcript"


# --------------------------------------------------------------------------- #
# TTS
# --------------------------------------------------------------------------- #


async def test_tts_cached_phrase_does_not_call_network(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    pcm = b"\x00\x00" * 320
    tts = FakeTts()
    service = SiliconFlowTTSService(
        tts, FakeCache({"你好": pcm}), events, sample_rate=SAMPLE_RATE
    )
    frames = [frame async for frame in service.run_tts("你好", "ctx")]
    assert frames and sum(len(frame.audio) for frame in frames) == len(pcm)
    assert tts.calls == []
    records = read_events(tmp_path / "call")
    assert [event["type"] for event in records] == [
        "tts_request",
        "tts_first_audio",
        "play_start",
        "play_end",
    ]
    assert records[0]["data"]["cached"] is True
    assert records[1]["data"]["latency_ms"] == 0.0


async def test_tts_synth_logs_first_audio_and_play_end(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    tts = FakeTts(chunks=3)
    service = SiliconFlowTTSService(tts, FakeCache(), events, sample_rate=SAMPLE_RATE)
    frames = [frame async for frame in service.run_tts("现合成的句子", "ctx")]
    assert len(frames) == 3
    types = [event["type"] for event in read_events(tmp_path / "call")]
    assert types[0] == "tts_request"
    assert types[1] == "tts_first_audio"
    assert types[2] == "play_start"
    assert types[-1] == "play_end"


async def test_tts_confirm_failure_falls_back_to_phone_fragments(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    pcm = b"\x00\x00" * 480
    service = SiliconFlowTTSService(
        FakeTts(fail=True), FakeCache(phone_pcm=pcm), events, sample_rate=SAMPLE_RATE
    )
    service._current_phone = "13800138000"  # noqa: SLF001 - 模拟 SpeakHintFrame 携带的号码
    confirm_text = "跟您确认一下：您是李明，回拨电话是幺三八，零零幺三，八零零零，对吗？"
    frames = [frame async for frame in service.run_tts(confirm_text, "ctx")]
    assert sum(len(frame.audio) for frame in frames) == len(pcm)
    records = read_events(tmp_path / "call")
    reasons = [e["data"].get("reason") for e in records if e["type"] == "error"]
    assert "confirm_synth_failed" in reasons
    played = play_texts(records)
    assert played and played[0].startswith(PHRASES["confirm_phone_prefix"])
    assert played[0].endswith(PHRASES["confirm_phone_suffix"])


# --------------------------------------------------------------------------- #
# DialogProcessor
# --------------------------------------------------------------------------- #


async def test_dialog_turn_emits_speak_frame_and_events(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    llm = FakeLlm(reply="好的", slots_update={"name": "小李"})
    processor = DialogProcessor(llm, make_dialog(), events)
    pushed = capture_frames(processor)
    processor._user_text.append("我是小李")  # noqa: SLF001 - 直接驱动内部状态
    await processor._run_turn()  # noqa: SLF001
    records = read_events(tmp_path / "call")
    types = [event["type"] for event in records]
    assert "llm_request" in types and "llm_done" in types
    assert "state_change" in types
    assert any(event["type"] == "slot_update" and event["data"]["slot"] == "name" for event in records)
    # 这一轮被追问或复述，至少要播一句；播放内容由 Dialog 决定，不在这里断言文本。
    assert pushed and isinstance(pushed[0], TTSSpeakFrame) and pushed[0].text
    assert processor._history[-1]["role"] == "assistant"  # noqa: SLF001
    assert processor.turn_count == 1


async def test_dialog_takeover_stops_responding(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    llm = FakeLlm()
    processor = DialogProcessor(llm, make_dialog(), events)
    processor.takeover()
    processor._user_text.append("喂")  # noqa: SLF001
    processor._schedule_turn()  # noqa: SLF001
    assert processor._turn_task is None  # noqa: SLF001 - 接管后不再起 LLM 任务
    assert llm.calls == []


async def test_dialog_llm_barge_in_carries_text_and_logs_interrupted(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    llm = BlockingLlm()
    processor = DialogProcessor(llm, make_dialog(), events)
    processor._user_text.append("前半句")  # noqa: SLF001
    task = asyncio.create_task(processor._run_turn())  # noqa: SLF001
    await asyncio.wait_for(llm.started.wait(), timeout=1.0)
    processor._cancel_turn()  # noqa: SLF001 - 模拟 InterruptionFrame
    llm.release.set()
    await asyncio.wait_for(task, timeout=1.0)

    # 文本没丢：顺延回 `_user_text`，等下一轮拼接。
    assert processor._user_text == ["前半句"]  # noqa: SLF001
    records = read_events(tmp_path / "call")
    interrupted = [e for e in records if e["type"] == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["data"]["stage"] == "llm"
    assert interrupted[0]["data"]["text"] == "前半句"


async def test_dialog_asr_failure_plays_repeat(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    pushed = capture_frames(processor)
    processor._asr_failed = True  # noqa: SLF001 - 模拟 AsrFailedFrame
    await processor._run_turn()  # noqa: SLF001
    records = read_events(tmp_path / "call")
    assert any(e["type"] == "error" and e["data"].get("reason") == "empty_transcript" for e in records)
    assert pushed and pushed[0].text == PHRASES["repeat"].format(owner="张先生")


async def test_dialog_partial_phone_counts_failed_ask(tmp_path: Path) -> None:
    def responder(text: str):
        if "姓李" in text:
            return "", {"name": "李明", "purpose": "想找张先生问快递"}
        return "", {}

    events = events_for(tmp_path)
    processor = DialogProcessor(ScriptedLlm(responder), make_dialog(), events)

    processor._user_text.append("我姓李，想问下快递。")  # noqa: SLF001
    await processor._run_turn()  # noqa: SLF001
    assert processor._dialog.asked_slot == "callback_number"  # noqa: SLF001

    processor._user_text.append("一三八零零一三")  # noqa: SLF001
    await processor._run_turn()  # noqa: SLF001

    records = read_events(tmp_path / "call")
    assert any(
        e["type"] == "error" and e["data"].get("reason") == "incomplete_phone_timeout"
        for e in records
    )
    assert processor._dialog.slots["callback_number"].value is None  # noqa: SLF001
    assert processor._dialog.attempts["callback_number"] == 2  # noqa: SLF001


async def test_dialog_confirm_lead_splits_into_two_frames(tmp_path: Path) -> None:
    def responder(text: str):
        return "", {"name": "李明", "purpose": "想约张先生谈合同"}

    events = events_for(tmp_path)
    config = make_config(confirm_lead=True)
    processor = DialogProcessor(ScriptedLlm(responder), make_dialog(config), events, config=config)
    pushed = capture_frames(processor)
    processor._user_text.append("我姓李，想约张先生谈合同，电话是一三八零零一三八零零零。")  # noqa: SLF001
    await processor._run_turn()  # noqa: SLF001

    assert len(pushed) == 2
    assert pushed[0].text == PHRASES["confirm_lead"]
    assert pushed[1].text.startswith("您是") or pushed[1].text.startswith("您想找")
    assert processor._history[-1]["content"] == pushed[0].text + pushed[1].text  # noqa: SLF001


async def test_dialog_confirm_fallback_updates_history(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    processor._history.append({"role": "assistant", "content": "好的，跟您确认一下：您是李明"})  # noqa: SLF001
    processor._confirm_pending = {  # noqa: SLF001
        "lead": "好的，",
        "text": "跟您确认一下：您是李明",
        "phone": "13800138000",
    }
    spoken = "您的回拨电话是幺三八，零零幺三八零零零，对吗？"
    processor.on_tts_fallback(spoken)
    assert processor._history[-1]["content"] == "好的，" + spoken  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Smart Turn 阈值 / VAD 阈值 / 尾音抢话
# --------------------------------------------------------------------------- #


def test_configured_smart_turn_applies_threshold(monkeypatch) -> None:
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    from adapters.pipecat.pipeline import ConfiguredSmartTurn

    # 用 __new__ 跳过父类加载 ONNX：_predict_endpoint 只需要 _configured_threshold。
    instance = ConfiguredSmartTurn.__new__(ConfiguredSmartTurn)
    instance._configured_threshold = 0.85

    for probability, expected in ((0.9, 1), (0.6, 0), (0.4, 0)):
        monkeypatch.setattr(
            LocalSmartTurnAnalyzerV3,
            "_predict_endpoint",
            lambda self, audio, p=probability: {
                "prediction": int(p >= 0.5),  # 父类的布尔预测按 0.5 判
                "probability": p,
            },
        )
        result = instance._predict_endpoint(np.zeros(10))
        assert result["prediction"] == expected


def test_build_vad_params_maps_interrupt_threshold() -> None:
    from adapters.pipecat.pipeline import build_vad_params

    params = build_vad_params({"threshold": 0.5, "speech_end_silence_ms": 600}, 300.0)
    assert params.start_secs == 0.3  # 连续说话 300ms 才判为开口 → 短噪声不打断
    assert params.stop_secs == 0.6
    assert params.confidence == 0.5
    # 页面把阈值改小/改大都要生效。
    assert build_vad_params({}, 0.0).start_secs == 0.0
    assert build_vad_params({}, 500.0).start_secs == 0.5


async def test_turn_end_reason_from_smart_turn(tmp_path: Path) -> None:
    from pipecat.frames.frames import UserStoppedSpeakingFrame

    from adapters.pipecat.pipeline import TurnEventLogger

    class FakeSmartTurn:
        last_reason = "stop_secs"

    events = events_for(tmp_path)
    logger = TurnEventLogger(events, smart_turn=FakeSmartTurn())
    await logger.process_frame(UserStoppedSpeakingFrame(), None)  # type: ignore[arg-type]
    records = read_events(tmp_path / "call")
    assert records[0]["type"] == "turn_end"
    assert records[0]["data"]["reason"] == "stop_secs"


def test_pipecat_barge_in_uses_playback_tracker(tmp_path: Path) -> None:
    from adapters.pipecat.pipeline import PipecatCall

    class FakeDialogProcessor:
        def __init__(self) -> None:
            self.marked: list[str] = []

        def mark_playing_interrupted(self, text: str) -> None:
            self.marked.append(text)

    clock = FakeClock()
    events = events_for(tmp_path)
    call = PipecatCall.__new__(PipecatCall)
    call.finished = False  # type: ignore[attr-defined]
    call._takeover = False  # type: ignore[attr-defined]
    call.playback = PlaybackTracker(clock=clock)  # type: ignore[attr-defined]
    call.events = events  # type: ignore[attr-defined]
    call.dialog_processor = FakeDialogProcessor()  # type: ignore[attr-defined]

    call.playback.on_play_start("您好")
    call.playback.on_audio(1000.0, "您好")
    clock.advance(200.0)
    PipecatCall._on_barge_in(call)

    records = read_events(tmp_path / "call")
    interrupted = [e for e in records if e["type"] == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["data"]["tail"] is True
    assert interrupted[0]["data"]["text"] == "您好"
    assert interrupted[0]["data"]["played_ms"] == 200.0
    assert call.dialog_processor.marked == ["您好"]  # type: ignore[attr-defined]


def test_pipecat_barge_in_ignored_when_nothing_playing(tmp_path: Path) -> None:
    from adapters.pipecat.pipeline import PipecatCall

    class FakeDialogProcessor:
        def mark_playing_interrupted(self, text: str) -> None:  # pragma: no cover
            raise AssertionError("不该在没播时标历史")

    events = events_for(tmp_path)
    call = PipecatCall.__new__(PipecatCall)
    call.finished = False  # type: ignore[attr-defined]
    call._takeover = False  # type: ignore[attr-defined]
    call.playback = PlaybackTracker()  # type: ignore[attr-defined]
    call.events = events  # type: ignore[attr-defined]
    call.dialog_processor = FakeDialogProcessor()  # type: ignore[attr-defined]
    PipecatCall._on_barge_in(call)
    assert read_events(tmp_path / "call") == []


def test_pipecat_forwards_tail_interrupted_with_unplayed() -> None:
    from adapters.pipecat.pipeline import PipecatCall

    sent: list[dict] = []
    call = PipecatCall.__new__(PipecatCall)
    call._send_json = sent.append  # type: ignore[attr-defined]
    call.playback = PlaybackTracker()  # type: ignore[attr-defined]
    call._on_event(
        {
            "type": "interrupted",
            "t": 1.0,
            "data": {"tail": True, "text": "第二句", "unplayed": ["第三句"]},
        }
    )
    ai = [message for message in sent if message.get("type") == "transcript"]
    assert len(ai) == 1
    assert ai[0]["interrupted"] is True
    assert ai[0]["text"] == "第二句"
    assert ai[0]["unplayed"] == ["第三句"]

    # LLM 阶段的抢话（stage="llm"）没有 AI 输出，事件里的 text 是来电者的话，不生成 AI 逐句。
    sent.clear()
    call._on_event(
        {"type": "interrupted", "t": 2.0, "data": {"stage": "llm", "text": "来电者的话"}}
    )
    assert [message for message in sent if message.get("type") == "transcript"] == []

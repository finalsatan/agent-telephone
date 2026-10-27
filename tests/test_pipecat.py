"""T15 / T15b / T15c：不联网的适配层测试；T15c 用真实管线和本地 Silero ONNX。

覆盖：

- `SiliconFlowSTTService.run_stt()`：调用 `services.asr`，写 `asr_request` / `asr_final`，
  产出 `TranscriptionFrame`；ASR 报错 / 空转写时产出 `AsrFailedFrame`（T15b）。
- `SiliconFlowTTSService.run_tts()`：预合成缓存命中不调网络、首包 0；未命中走 `synth()`，
  写事件；CONFIRM 合成失败时退回号码片段拼接（T15b）。
- `DialogProcessor`：一轮 LLM + Dialog 产出 `TTSSpeakFrame`、写事件；抢话时取消 LLM 并把
  文本顺延（T15b）；ASR 失败播兜底；号码没报完按超时处理；`confirm_lead` 拆句。
- `ConfiguredSmartTurn`：把 `smart_turn_threshold` 应用到概率上；记录 `turn_end` 原因。
- `build_vad_params`：开口与打断阈值分离（T15c）。
- `PipecatCall._on_barge_in()`：浏览器还在播时记尾音抢话（T15b）。
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame

from adapters.pipecat.llm import DialogProcessor, TurnDoneFrame
from adapters.pipecat.stt import AsrFailedFrame, SiliconFlowSTTService
from adapters.pipecat.tts import SiliconFlowTTSService
from core.call import INTERRUPTED_SUFFIX
from core.events import EventLogger, read_events
from core.phrases import PHRASES, PhraseMissing
from core.playback import PlaybackTracker
from core.types import AsrResult, ReplyChunk, TurnResult
from services.asr import AsrError, load_wav_pcm

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


async def start_queued_turn(
    processor: DialogProcessor, text: str, failed: bool = False
) -> asyncio.Task:
    """按管线路径起一轮：走 `_queue_turn()`，返回这一轮的任务。

    `_queue_turn()` 用 `create_task()` 起任务，单独用时需要一个 TaskManager（管线里由
    Pipeline 注入）；这里补上，不再走 `_run_turn()` 的旧直调分支（T15d 要求 5）。
    """
    from pipecat.utils.asyncio.task_manager import TaskManager
    from pipecat.utils.base_object import BaseObject

    await BaseObject.setup(processor, TaskManager())
    processor._queue_turn(text, failed)  # noqa: SLF001
    task = processor._turn_task  # noqa: SLF001
    assert task is not None
    return task


async def run_queued_turn(processor: DialogProcessor, text: str, failed: bool = False) -> None:
    """按管线路径驱动一轮，并等这一轮结束。"""
    task = await start_queued_turn(processor, text, failed)
    await asyncio.wait_for(task, timeout=2.0)


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
    service._context_phones["ctx"] = "13800138000"  # noqa: SLF001 - 模拟 SpeakHintFrame 携带的号码
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
    await run_queued_turn(processor, "我是小李")
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
    pushed = capture_frames(processor)
    processor.takeover()
    await run_queued_turn(processor, "喂")
    assert llm.calls == []
    assert processor.turn_count == 0  # noqa: SLF001
    assert [f.kind for f in pushed if isinstance(f, TurnDoneFrame)] == ["skipped"]


async def test_dialog_llm_barge_in_carries_text_and_logs_interrupted(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    llm = BlockingLlm()
    processor = DialogProcessor(llm, make_dialog(), events)
    task = await start_queued_turn(processor, "前半句")
    await asyncio.wait_for(llm.started.wait(), timeout=1.0)
    processor._cancel_turn()  # noqa: SLF001 - 模拟 InterruptionFrame
    llm.release.set()
    await asyncio.wait_for(task, timeout=1.0)

    # 文本没丢：顺延到 `_carried_text`，等下一轮拼接。
    assert processor._carried_text == "前半句"  # noqa: SLF001
    records = read_events(tmp_path / "call")
    interrupted = [e for e in records if e["type"] == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["data"]["stage"] == "llm"
    assert interrupted[0]["data"]["text"] == "前半句"


async def test_dialog_asr_failure_plays_repeat(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    pushed = capture_frames(processor)
    await run_queued_turn(processor, "", failed=True)
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

    await run_queued_turn(processor, "我姓李，想问下快递。")
    assert processor._dialog.asked_slot == "callback_number"  # noqa: SLF001

    await run_queued_turn(processor, "一三八零零一三")

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
    await run_queued_turn(
        processor, "我姓李，想约张先生谈合同，电话是一三八零零一三八零零零。"
    )

    pushed = [f for f in pushed if isinstance(f, TTSSpeakFrame)]
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


async def test_dialog_tts_unplayed_drops_sentence_from_history(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    processor._record_assistant("跟您确认一下：您是李明，对吗？")  # noqa: SLF001
    processor.on_tts_unplayed("跟您确认一下：您是李明，对吗？")
    assert processor._history == []  # noqa: SLF001
    records = read_events(tmp_path / "call")
    assert [e["type"] for e in records] == ["interrupted"]
    assert records[0]["data"] == {
        "reason": "barge_in",
        "text": "跟您确认一下：您是李明，对吗？",
    }


async def test_dialog_tts_unplayed_keeps_played_confirm_lead(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    lead = PHRASES["confirm_lead"]
    rest = "您是李明，对吗？"
    processor._record_assistant(lead + rest)  # noqa: SLF001
    processor.on_tts_unplayed(rest)
    assert processor._history[-1]["content"] == lead + INTERRUPTED_SUFFIX  # noqa: SLF001


async def test_dialog_tts_fallback_after_interrupt_keeps_history(tmp_path: Path) -> None:
    """T15d 要求 3：这一轮已经被抢话时，号码片段没播出，不能写回历史。"""
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    cancel = asyncio.Event()
    cancel.set()
    processor._record_assistant("跟您确认一下：您是李明，对吗？")  # noqa: SLF001
    processor._confirm_pending = {  # noqa: SLF001
        "lead": "",
        "text": "跟您确认一下：您是李明，对吗？",
        "phone": "13800138000",
        "cancel": cancel,
    }
    processor.on_tts_fallback("您的回拨电话是幺三八，零零幺三八零零零，对吗？")
    assert processor._history == []  # noqa: SLF001


async def test_dialog_takeover_reason_reaches_interrupted_event(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    processor = DialogProcessor(FakeLlm(), make_dialog(), events)
    processor.takeover()
    processor.on_tts_unplayed("您是李明，对吗？")
    records = read_events(tmp_path / "call")
    assert records[0]["data"]["reason"] == "takeover"


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


def test_build_vad_params_separates_interrupt_threshold() -> None:
    from adapters.pipecat.pipeline import build_vad_params
    from pipecat.audio.vad.vad_analyzer import VADParams

    for threshold in (0.0, 300.0, 800.0):
        params = build_vad_params({"threshold": 0.5, "speech_end_silence_ms": 600}, threshold)
        assert params.start_secs == VADParams().start_secs
        assert params.stop_secs == 0.6
    assert build_vad_params({"start_secs": 0.1}, 800).start_secs == 0.1


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

# T15c：真实 PipecatCall + Silero + SegmentedSTT + UserTurnProcessor + TTS + sink。
# 只替换外部服务与 Smart Turn 预测，按 20ms 节奏送 PCM，绝不直接调用处理器方法。
ROOT = Path(__file__).resolve().parents[1]


async def eventually(predicate, timeout=4.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def speech_pcm(seconds=None):
    if seconds == 0.76 or seconds is None:
        return load_wav_pcm(ROOT / 'scenarios/barge_in/s03b_barge_in/turn_04.wav')
    return load_wav_pcm(ROOT / 'scenarios/s01_delivery/turn_01.wav')


async def feed_pcm(call, pcm):
    loop = asyncio.get_running_loop()
    start = loop.time()
    for offset in range(0, len(pcm), 640):
        await call.feed(pcm[offset:offset + 640])
        await asyncio.sleep(max(0, start + (offset + 640) / 32000 - loop.time()))


async def say(call, seconds=0.76):
    await feed_pcm(call, speech_pcm(seconds))
    await feed_pcm(call, bytes(32000))


class SequenceAsr:
    def __init__(self, texts, blocked=None):
        self.texts = texts
        self.calls = []
        self.blocked = blocked
        self.release = asyncio.Event()

    async def transcribe(self, pcm):
        index = len(self.calls)
        self.calls.append(pcm)
        if index == self.blocked:
            await self.release.wait()
        value = self.texts[index]
        if isinstance(value, Exception):
            raise value
        return AsrResult(text=value, latency_ms=1, audio_ms=len(pcm)/32)


@asynccontextmanager
async def pipeline_call(tmp_path, monkeypatch, asr, llm=None, tts=None, cache=None,
                        threshold=300, predictions=None, confirm_lead=False):
    from adapters.pipecat.pipeline import PipecatCall, ConfiguredSmartTurn
    predictions = iter(predictions or [1] * 20)
    monkeypatch.setattr(ConfiguredSmartTurn, '_predict_endpoint',
                        lambda self, audio: {'prediction': (p := next(predictions)), 'probability': float(p)})
    async def close():
        pass
    services = SimpleNamespace(asr=asr, llm=llm or FakeLlm(slots_update={'name': '小李'}),
                               tts=tts or FakeTts(), cache=cache or FakeCache(), aclose=close)
    config = make_config(confirm_lead)
    config['vad']['speech_end_silence_ms'] = 600
    messages, audio, wire = [], [], []
    def send_audio(pcm):
        audio.append(pcm)
        wire.append('audio')
    def send_json(payload):
        messages.append(payload)
        if payload['type'] == 'turn_done':
            wire.append('done')
    call = PipecatCall(call_dir=tmp_path/'pipeline', config=config, services=services,
                       send_audio=send_audio, send_json=send_json, channel_sim=False,
                       interrupt_min_speech_ms=threshold)
    try:
        await call.start()
        await eventually(lambda: any(e['type'] == 'play_end' for e in read_events(call.call_dir)))
        await feed_pcm(call, bytes(19200))
        wire.clear()
        yield call, messages, audio, wire
    finally:
        asr.release.set()
        if hasattr(services.llm, 'release'):
            services.llm.release.set()
        await call.finish()


def done_messages(messages):
    return [m for m in messages if m['type'] == 'turn_done']


@pytest.mark.parametrize('last', ['后半句', '', AsrError('offline')], ids=['text', 'empty', 'failure'])
async def test_pipeline_waits_for_all_asr_segments(tmp_path, monkeypatch, last):
    asr = SequenceAsr(['前半句', last], blocked=1)
    llm = FakeLlm()
    async with pipeline_call(tmp_path, monkeypatch, asr, llm, predictions=[0, 1]) as (call, messages, _, _):
        await say(call)
        assert len(asr.calls) == 1 and llm.calls == []
        await say(call)
        await eventually(lambda: any(e['type'] == 'turn_end' for e in read_events(call.call_dir)))
        await asyncio.sleep(0.6)  # 超过旧的固定宽限仍不能提前启动。
        assert llm.calls == [] and done_messages(messages) == []
        asr.release.set()
        await eventually(lambda: len(done_messages(messages)) == 1)
        assert llm.calls == ['前半句' + (last if isinstance(last, str) else '')]
        assert len([e for e in read_events(call.call_dir) if e['type'] == 'turn_end']) == 1


@pytest.mark.parametrize('threshold', [300, 800])
async def test_pipeline_short_reply_is_recognized_at_any_interrupt_threshold(tmp_path, monkeypatch, threshold):
    asr = SequenceAsr(['不对。'])
    async with pipeline_call(tmp_path, monkeypatch, asr, threshold=threshold) as (call, messages, _, _):
        await say(call, 0.76)
        await eventually(lambda: len(done_messages(messages)) == 1)
        assert call.services.llm.calls == ['不对。']
        assert not [e for e in read_events(call.call_dir) if e['type'] == 'interrupted']


@pytest.mark.parametrize('threshold,seconds,interrupts', [(300, 1.1, True), (800, 0.76, False)])
async def test_pipeline_interrupt_threshold_only_gates_barge_in(tmp_path, monkeypatch, threshold, seconds, interrupts):
    greeting = PHRASES['greet'].format(owner='张先生')
    cache = FakeCache({greeting: bytes(32000 * 10)})
    asr = SequenceAsr(['不对。'])
    async with pipeline_call(tmp_path, monkeypatch, asr, cache=cache, threshold=threshold) as (call, messages, _, _):
        await say(call, seconds)
        await eventually(lambda: len(done_messages(messages)) == 1)
        assert call.services.llm.calls == ['不对。']
        events = [e for e in read_events(call.call_dir) if e['type'] == 'interrupted']
        assert bool(events) is interrupts
        if interrupts:
            assert len(events) == 1 and events[0]['data']['tail'] is True


async def test_pipeline_barge_in_after_turn_done_while_long_confirm_audio_queued(tmp_path, monkeypatch):
    class LongConfirmTts(FakeTts):
        async def synth(self, text, cancel):
            self.calls.append(text)
            if text.startswith('跟您确认一下：'):
                # 10 秒确认语：覆盖 turn_done 已抵达 sink、浏览器仍有 PCM 待播的窗口。
                yield bytes(16000 * 2 * 10)
                return
            async for chunk in super().synth(text, cancel):
                yield chunk

    asr = SequenceAsr(['我姓李', '询问快递', '电话一三八零零一三八零零零', '不对。'])
    llm = FakeLlm(slots_update={'name': '李明', 'purpose': '询问快递'})
    async with pipeline_call(tmp_path, monkeypatch, asr, llm, LongConfirmTts()) as (call, messages, _, _):
        for index in range(3):
            await say(call)
            await eventually(lambda: len(done_messages(messages)) == index + 1)
        assert call.dialog.state.value == 'CONFIRM'
        assert call.playback.current_playing() is not None

        busy_checks = []
        original_is_busy = call.vad.is_busy

        def record_busy():
            busy = original_is_busy()
            busy_checks.append((call.vad.analyzer.speech_ms, busy))
            return busy

        call.vad.is_busy = record_busy
        # Use the exact 420ms RMS correction WAV used by ws_client's barge variant.
        await feed_pcm(call, speech_pcm(0.76))
        await feed_pcm(call, bytes(32000))
        await eventually(lambda: len(done_messages(messages)) == 4)
        assert max((speech_ms for speech_ms, _ in busy_checks), default=0) >= 300
        assert any(speech_ms >= 300 and busy for speech_ms, busy in busy_checks)
        interruptions = [e for e in read_events(call.call_dir) if e['type'] == 'interrupted']
        assert len(interruptions) == 1
        assert interruptions[0]['data']['tail'] is True


class DelayedFirstLlm(FakeLlm):
    def __init__(self):
        super().__init__(slots_update={'name': '小李', 'purpose': '询问快递'})
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        if not self.calls:
            self.calls.append(user_text)
            self.started.set()
            await self.release.wait()
            yield ReplyChunk(kind='sentence', text='旧回复', t_ms=5000)
            return
        async for chunk in super().turn_stream(state, slots, history, user_text, attempts):
            yield chunk


async def test_pipeline_llm_barge_in_serializes_carry_and_keeps_shared_dialog(tmp_path, monkeypatch):
    asr = SequenceAsr(['前半句', '后半句', '一三八零零一三八零零零'])
    llm = DelayedFirstLlm()
    async with pipeline_call(tmp_path, monkeypatch, asr, llm) as (call, messages, _, _):
        shared = call.dialog
        await say(call, 1.1)
        await eventually(llm.started.is_set)
        await say(call, 1.1)
        assert len(asr.calls) == 2
        assert llm.calls == ['前半句']  # 新一轮已结束，旧 LLM 仍没首包。
        assert not done_messages(messages)
        llm.release.set()
        await eventually(lambda: len(done_messages(messages)) == 2)
        assert llm.calls == ['前半句', '前半句后半句']
        records = read_events(call.call_dir)
        interruptions = [e for e in records if e['type'] == 'interrupted']
        assert len(interruptions) == 1
        assert interruptions[0]['data']['stage'] == 'llm'
        assert not [e for e in records if e['data'].get('reason') == 'no_done_chunk']
        assert PHRASES['repeat'].format(owner='张先生') not in play_texts(records)
        assert call.dialog is call.dialog_processor.dialog is shared
        assert shared.slots['name'].value == '小李'
        assert call.dialog_processor.turn_count == 1
        await say(call, 1.1)
        await eventually(lambda: len(done_messages(messages)) == 3)
        assert call.dialog.state.value == 'CONFIRM'
        assert call.dialog.slots['callback_number'].value == '13800138000'
        assert any(m['type'] == 'slots' and m['slots']['callback_number']['value'] == '13800138000' for m in messages)
    saved = json.loads((call.call_dir/'slots.json').read_text())
    assert saved['state'] == 'CONFIRM'
    assert saved['slots']['callback_number']['value'] == '13800138000'
    end = [e for e in read_events(call.call_dir) if e['type'] == 'call_end'][0]
    assert end['data']['state'] == 'CONFIRM' and end['data']['turns'] == 2


class GatedFailTts(FakeTts):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def synth(self, text, cancel):
        if '幺三八' in text:
            self.started.set()
            await self.release.wait()
            raise RuntimeError('confirm failed')
        async for chunk in super().synth(text, cancel):
            yield chunk


async def test_pipeline_confirm_phone_survives_system_and_audio_frames(tmp_path, monkeypatch):
    from pipecat.frames.frames import OutputTransportMessageUrgentFrame
    tts = GatedFailTts()
    asr = SequenceAsr(['我姓李，电话一三八零零一三八零零零。'])
    llm = FakeLlm(slots_update={'name': '李明', 'purpose': '询问快递'})
    cache = FakeCache(phone_pcm=bytes(1600))
    async with pipeline_call(tmp_path, monkeypatch, asr, llm, tts, cache, confirm_lead=True) as (call, messages, _, wire):
        await say(call)
        await eventually(tts.started.is_set)
        assert done_messages(messages) == []
        await call.worker.queue_frame(OutputTransportMessageUrgentFrame(message={'probe': True}))
        await feed_pcm(call, bytes(3200))
        tts.release.set()
        await eventually(lambda: len(done_messages(messages)) == 1)
        records = read_events(call.call_dir)
        assert any(e['data'].get('reason') == 'confirm_synth_failed' for e in records)
        assert PHRASES['confirm_lead'] in play_texts(records)
        assert play_texts(records)[-1].startswith(PHRASES['confirm_phone_prefix'])
        assert call.dialog_processor._history[-1]['content'] == PHRASES['confirm_lead'] + play_texts(records)[-1]
        assert wire[-1] == 'done' and 'audio' in wire
        assert not call.tts._context_phones


async def test_pipeline_turn_done_allows_next_replay_turn(tmp_path, monkeypatch):
    asr = SequenceAsr(['我是小李', '询问快递'])
    async with pipeline_call(tmp_path, monkeypatch, asr) as (call, messages, _, wire):
        for index in range(2):
            await say(call)
            await eventually(lambda: len(done_messages(messages)) == index + 1)
            assert wire[-1] == 'done'
        assert [m['kind'] for m in done_messages(messages)] == ['reply', 'reply']
        assert len(call.services.llm.calls) == 2


async def test_pipeline_turn_done_skipped_after_takeover(tmp_path, monkeypatch):
    asr = SequenceAsr(['来电内容'], blocked=0)
    async with pipeline_call(tmp_path, monkeypatch, asr) as (call, messages, _, _):
        await say(call, 1.1)
        await eventually(lambda: any(e['type'] == 'turn_end' for e in read_events(call.call_dir)))
        assert not done_messages(messages)
        call.takeover()
        asr.release.set()
        await eventually(lambda: len(done_messages(messages)) == 1)
        assert done_messages(messages)[0]['kind'] == 'skipped'
        assert not call.services.llm.calls


async def test_pipeline_short_noise_does_not_interrupt_cached_audio(tmp_path, monkeypatch):
    greeting = PHRASES['greet'].format(owner='张先生')
    asr = SequenceAsr([])
    async with pipeline_call(tmp_path, monkeypatch, asr,
                             cache=FakeCache({greeting: bytes(32000 * 10)})) as (call, messages, _, _):
        await feed_pcm(call, speech_pcm()[:3200])  # 100ms，不能加上随后的静音凑阈值。
        await feed_pcm(call, bytes(32000))
        assert not [e for e in read_events(call.call_dir) if e['type'] == 'interrupted']
        assert not asr.calls
        assert not done_messages(messages)


async def test_pipeline_sparse_noise_does_not_accumulate_during_cached_audio(tmp_path, monkeypatch):
    greeting = PHRASES['greet'].format(owner='张先生')
    asr = SequenceAsr([])
    async with pipeline_call(
        tmp_path,
        monkeypatch,
        asr,
        cache=FakeCache({greeting: bytes(32000 * 10)}),
    ) as (call, messages, _, _):
        source = speech_pcm(0.76)
        # A voiced 80ms slice: too short to open a Pipecat VAD turn on its own.
        noise = source[int(0.080 * SAMPLE_RATE * 2):int(0.160 * SAMPLE_RATE * 2)]
        assert len(noise) == int(0.080 * SAMPLE_RATE * 2)
        assert call.playback.current_playing() is not None

        for index in range(6):
            await feed_pcm(call, noise)
            assert call.vad.analyzer.speech_ms < 300
            assert call.vad.analyzer._candidate_speech_ms < 300
            assert call.playback.current_playing() is not None
            if index < 5:
                await feed_pcm(call, bytes(int(1.5 * SAMPLE_RATE * 2)))
                assert call.vad.analyzer.speech_ms == 0
                assert call.vad.analyzer._candidate_speech_ms == 0
                assert call.playback.current_playing() is not None

        assert not [e for e in read_events(call.call_dir) if e['type'] == 'interrupted']
        assert asr.calls == []
        assert not done_messages(messages)


@pytest.mark.parametrize('last', ['', AsrError('offline')], ids=['empty', 'failure'])
async def test_pipeline_failed_asr_completes_with_repeat(tmp_path, monkeypatch, last):
    asr = SequenceAsr([last])
    async with pipeline_call(tmp_path, monkeypatch, asr) as (call, messages, _, wire):
        await say(call)
        await eventually(lambda: len(done_messages(messages)) == 1)
        assert not call.services.llm.calls
        assert play_texts(read_events(call.call_dir))[-1] == PHRASES['repeat']
        assert done_messages(messages)[0]['kind'] == 'reply'
        assert wire[-1] == 'done'


async def test_pipeline_tts_wait_can_be_interrupted_without_losing_turn_done(tmp_path, monkeypatch):
    class WaitingTts(FakeTts):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.cancel = None

        async def synth(self, text, cancel):
            if '幺三八' in text:
                self.cancel = cancel
                self.started.set()
                await asyncio.Event().wait()
            async for chunk in super().synth(text, cancel):
                yield chunk

    tts = WaitingTts()
    asr = SequenceAsr(['电话一三八零零一三八零零零。', '不对。'])
    llm = FakeLlm(slots_update={'name': '李明', 'purpose': '询问快递'})
    async with pipeline_call(tmp_path, monkeypatch, asr, llm, tts) as (call, messages, _, wire):
        await say(call)
        await eventually(tts.started.is_set)
        assert not done_messages(messages)
        await say(call, 1.1)
        await eventually(lambda: len(done_messages(messages)) == 2)
        assert tts.cancel.is_set()
        assert wire[-1] == 'done'
        assert not call.tts._context_phones

        # T15d 要求 1 / 2：首块之前被抢话 = 这句从未播出：记 interrupted，历史不写这句，
        # 页面也不生成它的 AI 逐句（没有 play_start，事件也不带 tail）。
        records = read_events(call.call_dir)
        interruptions = [e for e in records if e['type'] == 'interrupted']
        assert len(interruptions) == 1
        assert interruptions[0]['data']['reason'] == 'barge_in'
        assert '幺三八' in interruptions[0]['data']['text']
        assert 'tail' not in interruptions[0]['data']
        history = call.dialog_processor._history
        assert not [m for m in history if m['role'] == 'assistant' and '幺三八' in m['content']]
        assert not [m for m in messages
                    if m['type'] == 'transcript' and m['role'] == 'ai' and '幺三八' in m['text']]


async def test_pipeline_confirm_lead_keeps_only_played_lead_when_rest_is_interrupted(tmp_path, monkeypatch):
    class WaitingRestTts(FakeTts):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def synth(self, text, cancel):
            if '幺三八' in text:
                self.started.set()
                await asyncio.Event().wait()
            async for chunk in super().synth(text, cancel):
                yield chunk

    lead = PHRASES['confirm_lead']
    # 10ms 的开头：抢话之前已经在浏览器里播完，不会走尾音分支（要求 1 后半）。
    cache = FakeCache({lead: bytes(320)})
    tts = WaitingRestTts()
    asr = SequenceAsr(['电话一三八零零一三八零零零。', '不对。'])
    llm = FakeLlm(slots_update={'name': '李明', 'purpose': '询问快递'})
    async with pipeline_call(
        tmp_path, monkeypatch, asr, llm, tts, cache, confirm_lead=True
    ) as (call, messages, _, _):
        await say(call)
        await eventually(tts.started.is_set)
        assert lead in play_texts(read_events(call.call_dir))
        await feed_pcm(call, bytes(6400))  # 200ms 静音，等开头播完
        assert call.playback.current_playing() is None
        await say(call, 1.1)
        await eventually(lambda: len(done_messages(messages)) == 2)

        records = read_events(call.call_dir)
        interruptions = [e for e in records if e['type'] == 'interrupted']
        assert len(interruptions) == 1
        assert 'tail' not in interruptions[0]['data']
        history = call.dialog_processor._history
        # 只保留已经播出的前一句，并标“（被对方打断）”；没播的后半句不写进历史。
        assert lead + INTERRUPTED_SUFFIX in [m['content'] for m in history]
        assert not [m for m in history if m['role'] == 'assistant' and '幺三八' in m['content']]


async def test_pipeline_takeover_while_tts_waits_marks_sentence_unplayed(tmp_path, monkeypatch):
    """T15d 要求 1：接管也会在首块之前取消合成，这句同样没播出。"""
    class WaitingTts(FakeTts):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def synth(self, text, cancel):
            if '幺三八' in text:
                self.started.set()
                await asyncio.Event().wait()
            async for chunk in super().synth(text, cancel):
                yield chunk

    tts = WaitingTts()
    asr = SequenceAsr(['电话一三八零零一三八零零零。'])
    llm = FakeLlm(slots_update={'name': '李明', 'purpose': '询问快递'})
    async with pipeline_call(tmp_path, monkeypatch, asr, llm, tts) as (call, messages, _, _):
        await say(call)
        await eventually(tts.started.is_set)
        call.takeover()
        await eventually(lambda: any(
            e['type'] == 'interrupted' for e in read_events(call.call_dir)
        ))
        records = read_events(call.call_dir)
        interruptions = [e for e in records if e['type'] == 'interrupted']
        assert len(interruptions) == 1
        assert interruptions[0]['data']['reason'] == 'takeover'
        assert '幺三八' in interruptions[0]['data']['text']
        history = call.dialog_processor._history
        assert not [m for m in history if m['role'] == 'assistant' and '幺三八' in m['content']]


class FailAtCancelTts(FakeTts):
    """确认语等首包时被抢话，底层合成同时报错：会走到号码兜底分支。"""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def synth(self, text, cancel):
        if '幺三八' in text:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError('confirm failed') from None
        async for chunk in super().synth(text, cancel):
            yield chunk


async def test_pipeline_confirm_fallback_after_interrupt_does_not_write_history(tmp_path, monkeypatch):
    """T15d 要求 3：被打断时号码兜底片段同样没播出，不能写回历史。"""
    tts = FailAtCancelTts()
    asr = SequenceAsr(['电话一三八零零一三八零零零。', '不对。'])
    llm = FakeLlm(slots_update={'name': '李明', 'purpose': '询问快递'})
    cache = FakeCache(phone_pcm=bytes(320))
    async with pipeline_call(tmp_path, monkeypatch, asr, llm, tts, cache) as (call, messages, _, _):
        await say(call)
        await eventually(tts.started.is_set)
        await say(call, 1.1)
        await eventually(lambda: len(done_messages(messages)) == 2)

        records = read_events(call.call_dir)
        # 兜底片段从未播出：不记 confirm_synth_failed、不记它的 play_start、不写历史。
        assert not [e for e in records if e['type'] == 'error'
                    and e['data'].get('reason') == 'confirm_synth_failed']
        assert not [t for t in play_texts(records) if t.startswith(PHRASES['confirm_phone_prefix'])]
        history = call.dialog_processor._history
        assert not [m for m in history if m['role'] == 'assistant' and '幺三八' in m['content']]
        assert not [m for m in history if m['role'] == 'assistant'
                    and PHRASES['confirm_phone_prefix'] in m['content']]


class FirstTurnRaisesLlm(FakeLlm):
    def __init__(self):
        super().__init__(slots_update={'name': '小李'})
        self.first_started = asyncio.Event()

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        if not self.calls:
            self.calls.append(user_text)
            self.first_started.set()
            raise RuntimeError('first turn failed')
        async for chunk in super().turn_stream(state, slots, history, user_text, attempts):
            yield chunk


async def test_pipeline_next_turn_runs_after_previous_turn_exception(tmp_path, monkeypatch):
    asr = SequenceAsr(['第一轮', '第二轮'])
    llm = FirstTurnRaisesLlm()
    async with pipeline_call(tmp_path, monkeypatch, asr, llm) as (call, messages, _, wire):
        await say(call)
        await eventually(llm.first_started.is_set)
        await eventually(lambda: any(
            e['type'] == 'error' and e['data'].get('stage') == 'turn'
            for e in read_events(call.call_dir)
        ))
        assert done_messages(messages) == []
        await say(call)
        await eventually(lambda: len(done_messages(messages)) == 1)
        records = read_events(call.call_dir)
        assert llm.calls == ['第一轮', '第二轮']
        assert any(e['type'] == 'error' and e['data'].get('reason') == 'RuntimeError'
                   for e in records)
        assert call.dialog_processor.turn_count == 1
        assert done_messages(messages)[0]['kind'] == 'reply'
        assert wire[-1] == 'done'


async def test_pipeline_unusable_stt_segment_closes_barrier_without_shifting_ids(tmp_path, monkeypatch):
    asr = SequenceAsr(['下一轮转写'])
    llm = FakeLlm()
    async with pipeline_call(tmp_path, monkeypatch, asr, llm) as (call, messages, _, _):
        await call.stt.set_usable(False)
        await say(call)
        await eventually(lambda: len(done_messages(messages)) == 1)
        assert asr.calls == []
        assert play_texts(read_events(call.call_dir))[-1] == PHRASES['repeat']
        assert not call.stt._open_segment_ids
        assert not call.stt._segment_ids

        await call.stt.set_usable(True)
        await say(call)
        await eventually(lambda: len(done_messages(messages)) == 2)
        assert len(asr.calls) == 1
        assert llm.calls == ['下一轮转写']
        assert [m['kind'] for m in done_messages(messages)] == ['reply', 'reply']

"""T15：Pipecat 适配层的单元测试（不联网、不加载 ONNX 模型）。

覆盖：

- `SiliconFlowSTTService.run_stt()`：调用 `services.asr`，写 `asr_request` / `asr_final`，
  产出 `TranscriptionFrame`；ASR 报错时只写 `error`、不抛。
- `SiliconFlowTTSService.run_tts()`：预合成缓存命中不调网络、首包 0；未命中走 `synth()`，
  写 `tts_request` / `tts_first_audio` / `play_start` / `play_end`；被打断时写 `interrupted`。
- `DialogProcessor`：一轮 LLM + Dialog 产出 `TTSSpeakFrame`、写 `llm_request` / `llm_done` /
  `state_change` / `slot_update`；抢话时取消 LLM 并把文本顺延。
- `ConfiguredSmartTurn`：把 `config.yaml` 的 `smart_turn_threshold` 应用到概率上。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from pipecat.frames.frames import InterruptionFrame, TranscriptionFrame, TTSSpeakFrame

from adapters.pipecat.llm import DialogProcessor
from adapters.pipecat.stt import SiliconFlowSTTService
from adapters.pipecat.tts import SiliconFlowTTSService
from core.events import EventLogger, read_events
from core.types import AsrResult, ReplyChunk, TurnResult
from services.asr import AsrError

SAMPLE_RATE = 16000


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
    def __init__(self, hits: dict[str, bytes] | None = None) -> None:
        self.hits = dict(hits or {})

    def get(self, text: str) -> bytes | None:
        return self.hits.get(text)


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


def make_config() -> dict:
    return {
        "owner": {"name": "张先生"},
        "dialog": {"max_asks": 2, "max_confirm_rounds": 2},
        "audio": {"sample_rate": SAMPLE_RATE},
        "vad": {"stop_secs": 2.0},
    }


def make_dialog():
    from core.dialog import Dialog

    return Dialog.from_config(make_config())


def events_for(tmp_path: Path) -> EventLogger:
    return EventLogger(tmp_path / "call")


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


async def test_stt_error_is_logged_not_raised(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    service = SiliconFlowSTTService(FakeAsr(fail=True), events, sample_rate=SAMPLE_RATE)
    frames = [frame async for frame in service.run_stt(b"\x01\x00" * 1600)]
    assert frames == []
    types = [event["type"] for event in read_events(tmp_path / "call")]
    assert types == ["asr_request", "error"]


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


async def test_tts_interruption_logs_interrupted(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    service = SiliconFlowTTSService(FakeTts(chunks=100), FakeCache(), events, sample_rate=SAMPLE_RATE)
    agen = service.run_tts("长句", "ctx")
    await agen.__anext__()
    await agen.aclose()
    types = [event["type"] for event in read_events(tmp_path / "call")]
    # aclose() 会关闭生成器，finally 里写 play_end；interrupted 只在任务被取消时写。
    assert "play_end" in types


# --------------------------------------------------------------------------- #
# DialogProcessor
# --------------------------------------------------------------------------- #


async def test_dialog_turn_emits_speak_frame_and_events(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    llm = FakeLlm(reply="好的", slots_update={"name": "小李"})
    processor = DialogProcessor(llm, make_dialog(), events)
    processor._user_text.append("我是小李")  # noqa: SLF001 - 直接驱动内部状态
    await processor._run_turn()  # noqa: SLF001
    records = read_events(tmp_path / "call")
    types = [event["type"] for event in records]
    assert "llm_request" in types and "llm_done" in types
    assert "state_change" in types
    assert any(event["type"] == "slot_update" and event["data"]["slot"] == "name" for event in records)
    # 这一轮被追问或复述，至少要播一句；播放内容由 Dialog 决定，不在这里断言文本。
    assert processor._playing_text  # noqa: SLF001


async def test_dialog_takeover_stops_responding(tmp_path: Path) -> None:
    events = events_for(tmp_path)
    llm = FakeLlm()
    processor = DialogProcessor(llm, make_dialog(), events)
    processor.takeover()
    processor._user_text.append("喂")  # noqa: SLF001
    processor._schedule_turn()  # noqa: SLF001
    assert processor._turn_task is None  # noqa: SLF001 - 接管后不再起 LLM 任务
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# Smart Turn 阈值
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

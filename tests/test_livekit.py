"""T16：LiveKit 适配层测试。

不联网：ASR / LLM / TTS 都用 fake，Smart Turn 用脚本化的概率（任务书允许 mock
Smart Turn 的预测结果），VAD 用本地 Silero ONNX。

分两层：

- **单元**：`build_turn_handling` 的参数映射、`PacedAudioSink`（实时节奏、
  `playback_position`、被打断的位置）、`SiliconFlowSTT` 的事件与空转写处理。
- **管线级**：启动**真实**的 `AgentSession`（不连 livekit-server，音频走自定义
  input/output），按 20/32ms 帧喂真实场景 wav，覆盖要求 4 的每一项：
  一轮多段发声、打断阈值只管打断、AI 在播时抢话、LLM 阶段抢话、TTS 首块之前被抢话、
  `confirm_lead` 拆句、`turn_done` 语义、号码兜底、ASR 失败兜底、接管。
"""

from __future__ import annotations

import asyncio
import json
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from livekit.agents import llm as lk_llm

from adapters.livekit.audio import PacedAudioSink, QueueAudioInput
from adapters.livekit.dialog import CallAgent
from adapters.livekit.session import LiveKitCall, build_turn_handling
from adapters.livekit.stt import SiliconFlowSTT
from core.dialog import Dialog
from core.events import EventLogger, read_events
from core.history import INTERRUPTED_SUFFIX
from core.phrases import PHRASES
from core.turn import TurnDetector
from core.types import AsrResult, ReplyChunk, TurnResult

SAMPLE_RATE = 16000
FRAME_BYTES = SAMPLE_RATE // 50 * 2  # 20ms
REPO = Path(__file__).resolve().parents[1]
TURN1 = REPO / "scenarios" / "h01_aircon" / "turn_01.wav"
TURN2 = REPO / "scenarios" / "h01_aircon" / "turn_02.wav"
GREETING = PHRASES["greet"].format(owner="张先生")


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeAsr:
    """按段返回脚本化的转写；`None` 表示这一段识别失败。"""

    def __init__(self, texts: list[str | None] | None = None) -> None:
        self.texts = list(texts or [])
        self.calls = 0
        self.audio_ms: list[float] = []

    async def transcribe(self, pcm: bytes) -> AsrResult:
        self.calls += 1
        self.audio_ms.append(len(pcm) / 32.0)
        text = self.texts.pop(0) if self.texts else ""
        if text is None:
            raise RuntimeError("asr boom")
        return AsrResult(text=text, latency_ms=1.0, audio_ms=len(pcm) / 32.0)


class FakeLlm:
    """按轮返回 `(reply, slots_update)`；`delay` 模拟 LLM 首 token 延迟。"""

    def __init__(self, turns: list[tuple[str, dict[str, str]]] | None = None, *, delay: float = 0.0):
        self.turns = list(turns or [])
        self.delay = float(delay)
        self.requests: list[str] = []

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.requests.append(user_text)
        if self.delay:
            await asyncio.sleep(self.delay)
        reply, updates = self.turns.pop(0) if self.turns else ("好的。", {})
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply=reply,
                slots_update=updates,
                first_token_ms=1.0,
                total_ms=2.0,
                parse_ok=True,
            ),
        )


class FakeTts:
    """每字 `secs_per_char` 秒静音，用来控制播放时长。"""

    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        secs_per_char: float = 0.06,
        fail_for: str | None = None,
        first_chunk_delay: float = 0.0,
    ) -> None:
        self.secs_per_char = secs_per_char
        self.fail_for = fail_for
        # 模拟“首块音频之前”的合成耗时（T15d 的那一类抢话）。
        self.first_chunk_delay = float(first_chunk_delay)
        self.texts: list[str] = []

    async def synth(self, text: str, cancel: asyncio.Event):
        self.texts.append(text)
        if self.first_chunk_delay:
            await asyncio.sleep(self.first_chunk_delay)
            if cancel.is_set():
                return
        if self.fail_for is not None and self.fail_for in text:
            raise RuntimeError("tts boom")
        total = max(1, int(len(text) * self.secs_per_char * SAMPLE_RATE)) * 2
        for offset in range(0, total, FRAME_BYTES):
            if cancel.is_set():
                return
            yield bytes(min(FRAME_BYTES, total - offset))
            await asyncio.sleep(0)


class FakeCache:
    def __init__(self, phrases: dict[str, bytes] | None = None) -> None:
        self.phrases = dict(phrases or {})
        self.render_calls: list[str] = []

    def get(self, text: str) -> bytes | None:
        return self.phrases.get(text)

    def render_phone(self, number: str) -> bytes:
        self.render_calls.append(number)
        return bytes(SAMPLE_RATE // 10 * 2)


class FakeSmartTurn:
    """脚本化的 Smart Turn 概率；用完后固定返回最后一个值。"""

    def __init__(self, probabilities: list[float] | None = None, default: float = 0.9) -> None:
        self.probabilities = list(probabilities or [])
        self.default = default
        self.calls = 0

    def predict(self, pcm: bytes) -> tuple[bool, float]:
        self.calls += 1
        value = self.probabilities.pop(0) if self.probabilities else self.default
        return value >= 0.5, value


class FakeServices:
    def __init__(self, *, asr=None, llm=None, tts=None, cache=None) -> None:
        self.asr = asr or FakeAsr()
        self.llm = llm or FakeLlm()
        self.tts = tts or FakeTts()
        self.cache = cache or FakeCache()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        assert reader.getframerate() == SAMPLE_RATE
        assert reader.getnchannels() == 1 and reader.getsampwidth() == 2
        return reader.readframes(reader.getnframes())


def silence(seconds: float) -> bytes:
    return bytes(int(seconds * SAMPLE_RATE) * 2)


def scripted_llm(*turns: tuple[str, dict[str, str]], delay: float = 0.0) -> FakeLlm:
    return FakeLlm(list(turns), delay=delay)


def greeting_cache(*, seconds: float = 0.4) -> FakeCache:
    """把问候语放进预合成缓存：慢 TTS 的用例里只有回复才会现合成。"""
    return FakeCache({GREETING: bytes(int(seconds * SAMPLE_RATE) * 2)})


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class Harness:
    """驱动一通 LiveKit 电话（不连 livekit-server）。"""

    def __init__(
        self,
        tmp_path: Path,
        *,
        services: FakeServices | None = None,
        smart_turn: FakeSmartTurn | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sent: list[dict[str, Any]] = []
        self.services = services or FakeServices()
        self.config = {
            "vad": {
                "threshold": 0.5,
                "speech_end_silence_ms": 600,
                "smart_turn_threshold": 0.85,
                "stop_secs": 2.0,
                "pre_speech_ms": 500,
                "max_segment_s": 8.0,
            },
            "interrupt": {"min_speech_ms": 300},
            "owner": {"name": "张先生"},
            "asr": {"total_timeout_s": 2.0, "model": "fake-asr"},
            "llm": {"realtime_model": "fake-llm"},
            "tts": {"model": "fake-tts", "confirm_lead": False},
        }
        self.config.update(config or {})
        self.call = LiveKitCall(
            call_dir=tmp_path / "call",
            config=self.config,
            services=self.services,
            send_json=self.sent.append,
            send_audio=None,
            channel_sim=False,
            sample_rate=SAMPLE_RATE,
            detector=TurnDetector.from_config(self.config, smart_turn=smart_turn or FakeSmartTurn()),
            **kwargs,
        )

    @property
    def agent(self) -> CallAgent:
        return self.call.agent

    @property
    def dialog(self) -> Dialog:
        return self.call.dialog

    def events(self) -> list[dict[str, Any]]:
        return read_events(self.call.call_dir)

    def types(self) -> list[str]:
        return [event["type"] for event in self.events()]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [event["data"] for event in self.events() if event["type"] == kind]

    def sent_of(self, kind: str) -> list[dict[str, Any]]:
        return [item for item in self.sent if item.get("type") == kind]

    async def start(self) -> None:
        await self.call.start()

    async def feed_pcm(self, pcm: bytes, *, realtime: bool = False) -> None:
        for offset in range(0, len(pcm), FRAME_BYTES):
            frame = pcm[offset : offset + FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame = frame + bytes(FRAME_BYTES - len(frame))
            await self.call.feed(frame)
            await asyncio.sleep(0.02 if realtime else 0)

    async def feed_wav(
        self,
        path: Path = TURN1,
        *,
        realtime: bool = False,
        tail_silence_s: float = 1.0,
        lead_silence_s: float = 0.0,
    ) -> None:
        if lead_silence_s:
            await self.feed_pcm(silence(lead_silence_s), realtime=realtime)
        await self.feed_pcm(load_wav(path), realtime=realtime)
        if tail_silence_s:
            await self.feed_pcm(silence(tail_silence_s), realtime=realtime)

    async def wait_for(self, predicate, *, timeout: float = 15.0, what: str = "condition") -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"timeout waiting for {what}")

    async def wait_events(self, kind: str, count: int = 1, *, timeout: float = 15.0) -> None:
        await self.wait_for(
            lambda: len(self.of(kind)) >= count, timeout=timeout, what=f"{count} x {kind}"
        )

    async def finish(self) -> None:
        await self.call.finish()


@asynccontextmanager
async def running(harness: Harness):
    await harness.start()
    try:
        yield harness
    finally:
        if not harness.call.finished:
            await harness.finish()


# --------------------------------------------------------------------------- #
# 单元：轮次检测
# --------------------------------------------------------------------------- #


def test_turn_handling_uses_manual_turn_detection():
    """轮次由我们自己提交（非流式 ASR 的转写比 VAD 事件晚到，见 notes 1.3 / 5）。"""
    options = build_turn_handling(
        {
            "vad": {"speech_end_silence_ms": 600, "stop_secs": 2.0},
            "interrupt": {"min_speech_ms": 300},
        }
    )
    assert options["turn_detection"] == "manual"
    assert options["endpointing"]["min_delay"] == pytest.approx(0.6)
    assert options["endpointing"]["max_delay"] == pytest.approx(2.0)
    # 打断仍然由框架执行（我们按 TurnSpeechCounter 触发 session.interrupt()）。
    assert options["interruption"]["enabled"] is True
    assert options["interruption"]["mode"] == "vad"
    assert options["interruption"]["resume_false_interruption"] is False
    # 回复要先由 Dialog 决定状态，提前生成与之冲突。
    assert options["preemptive_generation"]["enabled"] is False


# --------------------------------------------------------------------------- #
# 单元：音频 sink / 输入
# --------------------------------------------------------------------------- #


async def test_paced_sink_reports_playback_position_and_interrupt():
    started: list[str] = []
    finished: list[tuple[str, float, bool]] = []
    sink = PacedAudioSink(
        sample_rate=SAMPLE_RATE,
        on_play_start=started.append,
        on_play_end=lambda text, played, interrupted: finished.append(
            (text, round(played, 3), interrupted)
        ),
    )
    sink.start()
    frame = _frame(bytes(FRAME_BYTES))
    frame.userdata["lk_text"] = "第一句"
    for _ in range(10):
        await sink.capture_frame(frame)
    sink.flush()
    await asyncio.sleep(0.4)
    # 第二段：播到一半被 clear_buffer 打断。
    frame.userdata["lk_text"] = "第二句"
    for _ in range(10):
        await sink.capture_frame(frame)
    await asyncio.sleep(0.1)
    sink.clear_buffer()
    await asyncio.sleep(0.1)
    await sink.aclose()

    assert started == ["第一句", "第二句"]
    assert finished[0][0] == "第一句" and finished[0][2] is False
    assert finished[0][1] == pytest.approx(0.2, abs=0.06)
    assert finished[1][0] == "第二句" and finished[1][2] is True
    assert 0 < finished[1][1] < 0.2


async def test_queue_audio_input_frames_and_close():
    source = QueueAudioInput(sample_rate=SAMPLE_RATE)
    source.push_pcm(bytes(FRAME_BYTES * 3))
    source.close()
    frames = [frame.data async for frame in source]
    assert len(frames) == 3
    # `AudioFrame.data` 是 int16 的 memoryview，len() 按采样数算，这里要比字节数。
    assert all(len(bytes(data)) == FRAME_BYTES for data in frames)


# --------------------------------------------------------------------------- #
# 单元：STT
# --------------------------------------------------------------------------- #


async def test_stt_logs_events_and_swallows_empty(tmp_path: Path):
    events = EventLogger(tmp_path)
    asr = FakeAsr(["你好", ""])
    adapter = SiliconFlowSTT(asr, events, model="fake-asr")
    frame = _frame(bytes(FRAME_BYTES * 5))

    first = await adapter._recognize_impl(frame)  # noqa: SLF001
    assert first.alternatives[0].text == "你好"
    second = await adapter._recognize_impl(frame)  # noqa: SLF001
    # 空转写返回空文本事件（框架会忽略），不抛异常。
    assert second.alternatives[0].text == ""
    assert [e["type"] for e in read_events(tmp_path)] == [
        "asr_request",
        "asr_final",
        "asr_request",
        "asr_final",
        "error",
    ]
    assert adapter.capabilities.streaming is False


async def test_stt_error_is_reported_not_raised(tmp_path: Path):
    events = EventLogger(tmp_path)
    adapter = SiliconFlowSTT(FakeAsr([None]), events)
    event = await adapter._recognize_impl(_frame(bytes(FRAME_BYTES)))  # noqa: SLF001
    assert event.alternatives[0].text == ""
    errors = [e["data"] for e in read_events(tmp_path) if e["type"] == "error"]
    assert errors[0]["reason"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# 管线级
# --------------------------------------------------------------------------- #


def test_pipeline_greeting_and_one_turn(tmp_path: Path):
    async def main() -> Harness:
        services = FakeServices(
            asr=FakeAsr(["你好，我找张先生"]),
            llm=scripted_llm(("好的。", {"name": "张先生"})),
        )
        harness = Harness(tmp_path, services=services)
        async with running(harness) as h:
            await h.wait_events("play_end")  # 问候语
            await h.feed_wav()
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
        return harness

    harness = asyncio.run(main())
    kinds = harness.types()
    # 问候语：状态推进 + 缓存/合成 + 播放，没有 turn_done。
    assert kinds[0] == "call_start"
    assert "state_change" in kinds
    assert harness.of("tts_request")[0]["text"] == GREETING
    assert harness.of("llm_request")[0]["text"] == "你好，我找张先生"
    assert [event["type"] for event in harness.events()][-1] == "call_end"
    assert harness.of("llm_done")[0]["parse_ok"] is True
    # AI 逐句发给页面 + turn_done。
    ai_lines = [item["text"] for item in harness.sent if item.get("role") == "ai"]
    assert GREETING in ai_lines
    done = harness.sent_of("turn_done")[0]
    assert done["kind"] == "reply" and done["interrupted"] is False
    # slots.json / 分轨 wav。
    slots = json.loads((harness.call.call_dir / "slots.json").read_text())
    assert slots["slots"]["name"]["value"] == "张先生"
    assert (harness.call.call_dir / "caller.wav").exists()
    assert harness.call.ai_pcm


def test_pipeline_multi_segment_turn_keeps_second_half(tmp_path: Path):
    """一轮里两次发声（Smart Turn 判第一次没说完）：LLM 拿到两段的合并文本。"""

    async def main() -> Harness:
        services = FakeServices(asr=FakeAsr(["我找张先生"]), llm=scripted_llm(("好的。", {})))
        # 第一段判“没说完”（< 0.85），第二段判“说完”。
        # 第一段判“没说完”（< 0.85），第二段判“说完”。必须按实时节奏喂：
        # 比实时快时 VAD 事件与 predict() 会交错，测的就不是真实时序了。
        harness = Harness(tmp_path, services=services, smart_turn=FakeSmartTurn([0.1, 0.95]))
        async with running(harness) as h:
            await h.feed_pcm(load_wav(TURN1), realtime=True)
            await h.feed_pcm(silence(0.7), realtime=True)  # > speech_end_silence_ms，触发第一次 EOU
            await h.feed_pcm(load_wav(TURN1), realtime=True)
            await h.feed_pcm(silence(1.2), realtime=True)
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done", timeout=25.0)
        return harness

    harness = asyncio.run(main())
    requests = harness.of("llm_request")
    assert len(requests) == 1
    assert requests[0]["text"] == "我找张先生"
    assert len(harness.of("asr_final")) == 1
    assert harness.services.asr.calls == 1
    wav_ms = len(load_wav(TURN1)) / 32
    assert 2 * wav_ms + 700 - 1000 <= harness.services.asr.audio_ms[0] <= 2 * wav_ms + 700 + 1000
    events = read_events(harness.call.call_dir)
    turn_end = next(e["t"] for e in events if e["type"] == "turn_end")
    llm_request = next(e["t"] for e in events if e["type"] == "llm_request")
    assert llm_request - turn_end < 1000


async def test_room_greeting_waits_for_subscription_and_has_timeout(tmp_path: Path):
    for subscribed in (True, False):
        ready = asyncio.Event()
        harness = Harness(tmp_path / str(subscribed), services=FakeServices(cache=greeting_cache()),
                          config={"livekit": {"greeting_subscription_timeout_s": 0.3}},
                          room_url="ws://test", room_token="test")

        class Publication:
            async def wait_for_subscription(self):
                await ready.wait()

        async def connect():
            harness.call._ai_publication = Publication()

        harness.call._connect_room = connect
        async with running(harness) as h:
            await asyncio.sleep(0.1)
            assert not h.of("play_start")
            if subscribed:
                ready.set()
            await h.wait_events("play_start")
            assert bool([e for e in h.of("error") if e.get("reason") == "greeting_subscription_timeout"]) is not subscribed


def test_pipeline_short_reply_recognised_regardless_of_interrupt_threshold(tmp_path: Path):
    """打断阈值只管打断，不管开口：0.8s 的短回答在两种阈值下都能被识别。"""

    async def main() -> tuple[Harness, Harness]:
        short = load_wav(TURN1)[: int(0.8 * SAMPLE_RATE) * 2]
        out: list[Harness] = []
        for name, min_speech in (("low", 300), ("high", 2000)):
            services = FakeServices(asr=FakeAsr(["对"]), llm=scripted_llm(("好的。", {})))
            harness = Harness(
                tmp_path / name,
                services=services,
                config={"interrupt": {"min_speech_ms": min_speech}},
            )
            async with running(harness) as h:
                await h.wait_events("play_end")  # 先等问候语播完
                await h.feed_pcm(short)
                await h.feed_pcm(silence(1.5))
                await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
            out.append(harness)
        return out[0], out[1]

    low, high = asyncio.run(main())
    for harness in (low, high):
        assert [data["text"] for data in harness.of("asr_final")] == ["对"]
        assert harness.of("llm_request")[0]["text"] == "对"
        assert harness.sent_of("turn_done")[0]["kind"] == "reply"


def test_pipeline_barge_in_while_playing_marks_tail(tmp_path: Path):
    async def main() -> Harness:
        services = FakeServices(
            asr=FakeAsr(["你好", "别说了"]),
            llm=scripted_llm(("请问您是哪位？", {})),
            # 回复要播得够久（7 字 × 0.4s），才有机会在播放期间抢话。
            tts=FakeTts(secs_per_char=0.4),
        )
        harness = Harness(tmp_path, services=services)
        async with running(harness) as h:
            await h.wait_events("play_end")  # 问候语播完
            await h.feed_pcm(load_wav(TURN1))
            await h.feed_pcm(silence(0.7))
            await h.wait_events("play_start", 2)  # 回复开始播
            await h.feed_pcm(load_wav(TURN1), realtime=True)  # AI 还在播时抢话
            await h.wait_for(
                lambda: [d for d in h.of("interrupted") if d.get("tail")], what="tail interrupt"
            )
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
        return harness

    harness = asyncio.run(main())
    tail = [data for data in harness.of("interrupted") if data.get("tail")]
    assert tail and tail[0]["reason"] == "barge_in"
    # 历史里最后一条 assistant 标成“被对方打断”。
    assert harness.agent.history[-1]["content"].endswith(INTERRUPTED_SUFFIX)
    # 页面拿到一条 interrupted=True 的 AI 逐句。
    assert any(item.get("role") == "ai" and item.get("interrupted") for item in harness.sent)
    assert harness.sent_of("turn_done")[0]["kind"] == "interrupted"


def test_pipeline_barge_in_during_llm_carries_text(tmp_path: Path):
    async def main() -> Harness:
        services = FakeServices(
            asr=FakeAsr(["你好", "别说了"]),
            llm=scripted_llm(("请问您是哪位？", {}), delay=1.0),
            tts=FakeTts(secs_per_char=0.4),
        )
        harness = Harness(tmp_path, services=services)
        async with running(harness) as h:
            await h.wait_events("play_end")
            await h.feed_pcm(load_wav(TURN1))
            await h.feed_pcm(silence(0.7))
            await h.wait_events("llm_request")
            # LLM 还在跑时抢话：只需要 VAD 听到声音（不需要等这一段说完）。
            await h.feed_pcm(load_wav(TURN1))
            await h.wait_for(
                lambda: [d for d in h.of("interrupted") if d.get("stage") == "llm"],
                what="llm-stage interrupt",
            )
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
        return harness

    harness = asyncio.run(main())
    llm_interrupts = [data for data in harness.of("interrupted") if data.get("stage") == "llm"]
    assert llm_interrupts[0]["text"] == "你好"
    # 这一轮没有播 AI 音频，也没写进历史（文本顺延到下一轮）。
    assert harness.of("play_start")[-1]["text"] == GREETING
    assert harness.agent._carried_text == "你好"  # noqa: SLF001
    # 历史里只有问候语（这一轮的回复没播出，也没记）。
    assert harness.agent.history == [{"role": "assistant", "content": GREETING}]
    done = harness.sent_of("turn_done")[0]
    assert done["kind"] == "interrupted" and done["text"] == "你好"


def test_pipeline_barge_in_before_first_tts_chunk_drops_sentence(tmp_path: Path):
    """TTS 首块音频之前被抢话：历史里不写没播出的句子（T15d）。"""

    async def main() -> Harness:
        services = FakeServices(
            asr=FakeAsr(["你好", "别说了"]),
            llm=scripted_llm(("请问您是哪位？", {})),
            # 首块音频要 3 秒才出来，抢话一定发生在它之前。
            tts=FakeTts(secs_per_char=0.4, first_chunk_delay=3.0),
            cache=greeting_cache(),
        )
        harness = Harness(tmp_path, services=services)
        async with running(harness) as h:
            await h.wait_events("play_end")
            await h.feed_pcm(load_wav(TURN1))
            await h.feed_pcm(silence(0.7))
            await h.wait_events("tts_request", 2)  # 回复的合成已经开始
            await h.feed_pcm(load_wav(TURN1))
            await h.feed_pcm(silence(0.7))
            await h.wait_for(
                lambda: [d for d in h.of("interrupted") if not d.get("tail")], what="interrupt"
            )
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
        return harness

    harness = asyncio.run(main())
    interrupts = [data for data in harness.of("interrupted") if not data.get("tail")]
    assert interrupts[0]["text"] == "请问您是哪位？"
    # 没播出的那句不在历史里（只剩问候语和来电者这一句）。
    assert {"role": "user", "content": "你好"} in harness.agent.history
    assert all(item["content"] != "请问您是哪位？" for item in harness.agent.history)
    assert harness.sent_of("turn_done")[0]["kind"] == "interrupted"


def test_pipeline_confirm_lead_keeps_only_played_first_sentence(tmp_path: Path):
    async def main() -> Harness:
        services = FakeServices(
            asr=FakeAsr(["你好", "别说了"]),
            llm=scripted_llm(("请问您是哪位？", {})),
            tts=FakeTts(secs_per_char=0.4),
            cache=greeting_cache(),
        )
        harness = Harness(tmp_path, services=services, config={"tts": {"confirm_lead": True}})
        # 直接把 Dialog 推到 CONFIRM，省掉三轮。
        harness.dialog.on_user_turn("小李", {"name": "小李"}, True)
        harness.dialog.on_user_turn("空调坏了", {"purpose": "空调坏了"}, True)
        harness.dialog.on_user_turn("13800138000", {"callback_number": "13800138000"}, True)
        assert harness.dialog.state.value == "CONFIRM"
        async with running(harness) as h:
            await h.wait_events("play_end")  # 问候语
            await h.feed_pcm(load_wav(TURN1))
            await h.feed_pcm(silence(0.7))
            await h.wait_events("play_start", 2)  # 第一句（预合成开头）开始播
            await h.feed_pcm(load_wav(TURN1), realtime=True)
            await h.wait_for(
                lambda: [d for d in h.of("interrupted") if d.get("tail")], what="tail interrupt"
            )
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
        return harness

    harness = asyncio.run(main())
    lead = PHRASES["confirm_lead"]
    assert harness.of("play_start")[-1]["text"] == lead
    # 前一句已播：历史只保留它并标“被对方打断”。
    assert harness.agent.history[-1]["content"] == lead + INTERRUPTED_SUFFIX


def test_pipeline_confirm_synth_failure_falls_back_to_phone_parts(tmp_path: Path):
    """CONFIRM 确认语合成失败、一块音频都没播出：退回号码片段拼接（T15b / T06b）。"""

    async def main() -> Harness:
        services = FakeServices(
            asr=FakeAsr(["你好"]),
            llm=scripted_llm(("好的。", {})),
            # 确认语合成报错。
            tts=FakeTts(fail_for="跟您确认一下"),
            cache=greeting_cache(),
        )
        harness = Harness(tmp_path, services=services)
        harness.dialog.on_user_turn("小李", {"name": "小李"}, True)
        harness.dialog.on_user_turn("空调坏了", {"purpose": "空调坏了"}, True)
        harness.dialog.on_user_turn("13800138000", {"callback_number": "13800138000"}, True)
        async with running(harness) as h:
            await h.wait_events("play_end")
            await h.feed_pcm(load_wav(TURN1))
            await h.feed_pcm(silence(0.7))
            await h.wait_for(
                lambda: bool(h.services.cache.render_calls), what="phone fallback"
            )
            await h.wait_for(lambda: h.sent_of("turn_done"), what="turn_done")
        return harness

    harness = asyncio.run(main())
    reasons = [data["reason"] for data in harness.of("error")]
    assert "confirm_synth_failed" in reasons
    assert harness.services.cache.render_calls == ["13800138000"]
    # 兜底播出的是号码片段拼接的文本，历史里换成它。
    spoken = [data["text"] for data in harness.of("play_start")][-1]
    assert "幺三八" in spoken and harness.agent.history[-1]["content"] == spoken


def test_pipeline_takeover_stops_replies_and_records_caller(tmp_path: Path):
    async def main() -> Harness:
        services = FakeServices(asr=FakeAsr(["你好"]), llm=scripted_llm(("好的。", {})))
        harness = Harness(tmp_path, services=services)
        async with running(harness) as h:
            await h.wait_events("play_end")
            h.call.takeover()
            await h.wait_events("takeover")
            await h.feed_wav()
            await asyncio.sleep(0.5)
        return harness

    harness = asyncio.run(main())
    assert harness.of("takeover")[0]["turns"] == 0
    assert harness.of("llm_request") == []
    assert harness.call.caller_pcm  # 接管后来电者音频照常录进 caller.wav
    # 接管后不再有 AI 回复，页面收到 skipped 的 turn_done。
    assert harness.sent_of("turn_done") == [] or all(
        item["kind"] == "skipped" for item in harness.sent_of("turn_done")
    )


def test_pipeline_asr_failure_plays_fallback(tmp_path: Path):
    async def main() -> Harness:
        services = FakeServices(asr=FakeAsr([None]), llm=scripted_llm(("好的。", {})))
        harness = Harness(tmp_path, services=services)
        async with running(harness) as h:
            await h.wait_events("play_end")
            await h.feed_wav()
            await h.wait_for(
                lambda: PHRASES["repeat"] in [d["text"] for d in h.of("tts_request")],
                what="fallback phrase",
            )
        return harness

    harness = asyncio.run(main())
    assert [data["reason"] for data in harness.of("error") if data["stage"] == "asr"]
    assert harness.of("llm_request") == []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _frame(pcm: bytes):
    from livekit import rtc

    return rtc.AudioFrame(
        data=pcm,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(pcm) // 2,
    )


def test_node_llm_and_tts_are_framework_types():
    from adapters.livekit.nodes import NodeLLM, NodeTTS

    assert isinstance(NodeLLM(), lk_llm.LLM)
    assert isinstance(NodeTTS(), __import__("livekit.agents.tts", fromlist=["TTS"]).TTS)

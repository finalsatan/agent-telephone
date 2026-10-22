"""T10 验收测试：用 fake ASR／LLM／TTS 跑完离线回放主流程。

覆盖：
- 3 轮场景的事件顺序、history 构造、slots.json；
- 轮次合并（“喂。”＋“你好，我找一下张先生。”只回复一次）；
- 追问号码时号码没报完（“一三八零零一三”＋“八零零零”）；
- 号码没报完但超时（按普通轮次处理、计一次追问失败）；
- CONFIRM 整句合成失败时退回 render_phone 拼接兜底；
- LlmTimeout 播放兜底话术并记 error；
- 模板缓存缺失时现合成并补写缓存。

全部用 FakeClock 推进时间，结果确定、不联网、不 sleep。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.call import CallSession
from core.dialog import ReplyHint, compose_confirm_text
from core.events import EventLogger, read_events
from core.phrases import PHRASES, PhraseCache, cache_texts
from core.types import AsrResult, ReplyChunk, SlotValue, TurnResult
from services.llm import LlmTimeout
from services.tts import TtsTimeout
from tools.replay import replay_segments

OWNER = "张先生"
SAMPLE_RATE = 16000
PCM_5S = b"\x00\x00" * (5 * SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeClock:
    """可手动推进的单调时钟；EventLogger 与 CallSession 共用同一个实例。"""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, ms: float) -> None:
        self.value += ms


class FakeAsr:
    def __init__(self, clock: FakeClock, texts: list[str], *, step_ms: float = 50.0) -> None:
        self.clock = clock
        self.texts = list(texts)
        self.step_ms = step_ms
        self.calls: list[bytes] = []

    async def transcribe(self, pcm: bytes) -> AsrResult:
        self.calls.append(pcm)
        self.clock.advance(self.step_ms)
        text = self.texts.pop(0)
        return AsrResult(text=text, latency_ms=self.step_ms, audio_ms=len(pcm) / 32.0)


class FakeLlm:
    def __init__(self, clock: FakeClock, responder, *, step_ms: float = 50.0, timeout: bool = False) -> None:
        self.clock = clock
        self.responder = responder
        self.step_ms = step_ms
        self.timeout = timeout
        self.calls: list[dict] = []

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.calls.append(
            {
                "state": state,
                "slots": dict(slots),
                "history": [dict(message) for message in history],
                "user_text": user_text,
                "attempts": dict(attempts or {}),
            }
        )
        if self.timeout:
            self.clock.advance(self.step_ms)
            raise LlmTimeout("fake first token timeout")
        reply, slots_update = self.responder(user_text)
        self.clock.advance(self.step_ms)
        first_ms = self.step_ms
        sentences = [part for part in _split(reply) if part]
        for sentence in sentences:
            yield ReplyChunk(kind="sentence", text=sentence, t_ms=first_ms)
        self.clock.advance(self.step_ms)
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply=reply,
                slots_update=slots_update,
                first_token_ms=first_ms,
                total_ms=2 * self.step_ms,
                parse_ok=True,
                first_sentence_ms=first_ms if sentences else None,
            ),
            t_ms=2 * self.step_ms,
        )


class FakeTts:
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        clock: FakeClock,
        *,
        first_chunk_ms: float = 40.0,
        chunk_ms: float = 20.0,
        chunks: int = 2,
        fail: set[str] | None = None,
    ) -> None:
        self.clock = clock
        self.first_chunk_ms = first_chunk_ms
        self.chunk_ms = chunk_ms
        self.chunks = chunks
        self.fail = set(fail or ())
        self.calls: list[str] = []

    async def synth(self, text, cancel):
        self.calls.append(text)
        if text in self.fail:
            raise TtsTimeout(f"fake tts timeout for {text!r}")
        self.clock.advance(self.first_chunk_ms)
        yield b"\x00\x00" * 160  # 10ms
        for _ in range(max(0, self.chunks - 1)):
            if cancel.is_set():
                return
            self.clock.advance(self.chunk_ms)
            yield b"\x00\x00" * 160


def _split(reply: str) -> list[str]:
    from services.llm import split_reply_sentences

    return split_reply_sentences(reply) if reply else []


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #


def make_config(tmp_path: Path) -> dict:
    return {
        "siliconflow": {"base_url": "https://api.example.test/v1"},
        "owner": {"name": OWNER},
        "dialog": {"max_asks": 2, "max_confirm_rounds": 2},
        "vad": {"stop_secs": 3.0},
        "audio": {"sample_rate": SAMPLE_RATE},
        "tts": {
            "backend": "local",
            "voice": "",
            "phrase_cache": {"dir": str(tmp_path / "cache")},
        },
    }


def make_cache(config: dict, *, fill: bool = True) -> PhraseCache:
    cache = PhraseCache.from_config(config, backend="local", voice="test-voice")
    if fill:
        # 50ms 的“音频”足够让播放事件有时长；内容不重要，测试只查顺序与时长。
        for text in cache_texts(OWNER).values():
            cache.put(text, b"\x00\x00" * 800)
    return cache


def make_session(
    tmp_path: Path,
    asr_texts: list[str],
    responder,
    *,
    clock: FakeClock | None = None,
    tts: FakeTts | None = None,
    cache: PhraseCache | None = None,
    llm_timeout: bool = False,
    confirm_lead: bool = False,
) -> dict:
    clock = clock or FakeClock()
    config = make_config(tmp_path)
    config["tts"]["confirm_lead"] = confirm_lead
    cache = cache or make_cache(config)
    asr = FakeAsr(clock, asr_texts)
    llm = FakeLlm(clock, responder, timeout=llm_timeout)
    tts = tts or FakeTts(clock)
    events = EventLogger(tmp_path / "call", clock=clock)
    session = CallSession(asr, llm, tts, cache, events, config, clock=clock)
    return {
        "session": session,
        "clock": clock,
        "config": config,
        "cache": cache,
        "asr": asr,
        "llm": llm,
        "tts": tts,
        "events": events,
    }


def no_slots(_text: str):
    return "", {}


def play_texts(events: list[dict]) -> list[str]:
    return [event["data"]["text"] for event in events if event["type"] == "play_start"]


# --------------------------------------------------------------------------- #
# 1) 3 轮场景
# --------------------------------------------------------------------------- #


async def test_three_turn_scenario_events_history_and_slots(tmp_path):
    def responder(text: str):
        if "姓李" in text:
            return "", {"name": "李明", "purpose": "想约张先生谈合同"}
        return "", {}

    env = make_session(
        tmp_path,
        [
            "你好，我找一下张先生。",
            "我姓李，想约张先生谈合同。",
            "电话是一三八零零一三八零零零。",
        ],
        responder,
    )
    segments = [(0.0, 1_000.0), (1_500.0, 2_500.0), (3_000.0, 4_000.0)]
    outcomes = await replay_segments(env["session"], segments, PCM_5S)

    assert [o.kind for o in outcomes] == ["reply", "reply", "reply"]
    events = read_events(tmp_path / "call")
    types = [e["type"] for e in events]
    assert types[0] == "call_start"
    assert types[-1] == "call_end"
    assert types.count("asr_request") == 3
    assert types.count("llm_request") == 3
    assert types.count("play_start") == 4  # 问候 + 3 轮回复
    assert types.count("interrupted") == 0
    # 每轮里 ASR → LLM → TTS → 播放 的顺序
    for start in (i for i, t in enumerate(types) if t == "asr_request"):
        tail = types[start:]
        assert tail.index("asr_final") < tail.index("llm_request")
        assert tail.index("llm_request") < tail.index("tts_request")
        assert tail.index("tts_request") < tail.index("play_start")
    # 状态变化与槽位事件
    state_changes = [e["data"] for e in events if e["type"] == "state_change"]
    assert {"from": "GREETING", "to": "ASK_IDENTITY"} in state_changes
    assert {"from": "FILL_MISSING", "to": "CONFIRM"} in state_changes
    slot_events = {e["data"]["slot"]: e["data"] for e in events if e["type"] == "slot_update"}
    assert slot_events["name"]["value"] == "李明"
    assert slot_events["purpose"]["value"] == "想约张先生谈合同"
    assert slot_events["callback_number"]["value"] == "13800138000"

    # slots.json
    payload = json.loads((tmp_path / "call" / "slots.json").read_text(encoding="utf-8"))
    assert payload["state"] == "CONFIRM"
    assert payload["slots"]["name"] == {"value": "李明", "status": "unconfirmed"}
    assert payload["slots"]["purpose"] == {"value": "想约张先生谈合同", "status": "unconfirmed"}
    assert payload["slots"]["callback_number"] == {"value": "13800138000", "status": "unconfirmed"}

    # history 只放 user/assistant，且构造顺序符合 0.5
    llm = env["llm"]
    assert [call["user_text"] for call in llm.calls] == [
        "你好，我找一下张先生。",
        "我姓李，想约张先生谈合同。",
        "电话是一三八零零一三八零零零。",
    ]
    assert llm.calls[0]["history"] == [{"role": "assistant", "content": PHRASES["greet"].format(owner=OWNER)}]
    assert llm.calls[2]["history"] == [
        {"role": "assistant", "content": PHRASES["greet"].format(owner=OWNER)},
        {"role": "user", "content": "你好，我找一下张先生。"},
        {"role": "assistant", "content": PHRASES["ask_name"]},
        {"role": "user", "content": "我姓李，想约张先生谈合同。"},
        {"role": "assistant", "content": PHRASES["ask_callback"]},
    ]
    assert all(m["role"] in ("user", "assistant") for m in llm.calls[2]["history"])


# --------------------------------------------------------------------------- #
# 2) 轮次合并
# --------------------------------------------------------------------------- #


async def test_turn_merge_uses_actual_speech_start_not_pre_roll(tmp_path):
    """T12：`segment_wav()` 的 start_ms 含预卷，合并判断要加回 pre_speech_ms。

    原始间隔 100ms < 本轮实测约 150ms，会用预卷时刻误判为“抢话”；
    加回 500ms 预卷后，下一轮实际开口在 1100ms，不应合并。
    """
    env = make_session(tmp_path, ["喂。", "你好，我找一下张先生。"], no_slots)
    outcomes = await replay_segments(
        env["session"],
        [(0.0, 500.0), (600.0, 1_500.0)],
        PCM_5S,
        pre_speech_ms=500.0,
    )
    assert [outcome.kind for outcome in outcomes] == ["reply", "reply"]
    assert [call["user_text"] for call in env["llm"].calls] == ["喂。", "你好，我找一下张先生。"]
    assert not any(event["type"] == "interrupted" for event in read_events(tmp_path / "call"))


async def test_turn_merged_before_reply_starts(tmp_path):
    env = make_session(tmp_path, ["喂。", "你好，我找一下张先生。"], no_slots)
    # 间隔 20ms < 本轮实测约 140ms（ASR 50 + LLM 100 + 缓存播放 0）
    segments = [(0.0, 500.0), (520.0, 2_000.0)]
    outcomes = await replay_segments(env["session"], segments, PCM_5S)

    assert [o.kind for o in outcomes] == ["merge", "reply"]
    assert env["llm"].calls[0]["user_text"] == "喂。"
    assert env["llm"].calls[1]["user_text"] == "喂。你好，我找一下张先生。"
    assert len(env["asr"].calls) == 2

    events = read_events(tmp_path / "call")
    interrupted = [e for e in events if e["type"] == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["data"]["reason"] == "turn_merged"
    assert interrupted[0]["data"]["text"] == "喂。"

    # 问候 + 合并后的一轮，只有这两次播放
    assert play_texts(events) == [
        PHRASES["greet"].format(owner=OWNER),
        PHRASES["ask_name"],
    ]
    # 第一轮没有推进状态：追问次数只计一次
    assert env["session"].dialog.attempts == {"name": 1}
    assert env["session"].history == [
        {"role": "assistant", "content": PHRASES["greet"].format(owner=OWNER)},
        {"role": "user", "content": "喂。你好，我找一下张先生。"},
        {"role": "assistant", "content": PHRASES["ask_name"]},
    ]


# --------------------------------------------------------------------------- #
# 3) 号码没报完 → 等下一轮拼接
# --------------------------------------------------------------------------- #


async def test_incomplete_phone_waits_for_next_turn(tmp_path):
    def responder(text: str):
        if "姓李" in text:
            return "", {"name": "李明", "purpose": "想找张先生问快递"}
        return "", {}

    env = make_session(
        tmp_path,
        ["我姓李，想问下快递。", "一三八零零一三", "八零零零"],
        responder,
    )
    segments = [(0.0, 1_000.0), (1_500.0, 2_500.0), (2_600.0, 3_600.0)]
    outcomes = await replay_segments(env["session"], segments, PCM_5S)

    assert [o.kind for o in outcomes] == ["reply", "wait", "reply"]
    # 号码没报完的那一轮不调 LLM
    assert [call["user_text"] for call in env["llm"].calls] == [
        "我姓李，想问下快递。",
        "一三八零零一三八零零零",
    ]
    assert env["session"].dialog.slots["callback_number"].value == "13800138000"

    events = read_events(tmp_path / "call")
    confirm_text = compose_confirm_text(env["session"].dialog.slots, owner_name=OWNER)
    assert confirm_text in play_texts(events)
    # 只回复了一次 CONFIRM（另外两次是问候与追问回拨号码）
    assert play_texts(events).count(confirm_text) == 1
    assert play_texts(events).count(PHRASES["ask_callback"]) == 1


async def test_partial_phone_rule_only_applies_while_asking_phone(tmp_path):
    """没在追问号码时，末尾数字串不能触发“等下一轮”。"""

    def responder(text: str):
        if "姓李" in text:
            return "", {"name": "李明", "purpose": "想找张先生问快递"}
        return "", {}

    env = make_session(tmp_path, ["我姓李，工号是1234567。", "就这样。"], responder)
    segments = [(0.0, 1_000.0), (1_500.0, 2_000.0)]
    outcomes = await replay_segments(env["session"], segments, PCM_5S)

    assert outcomes[0].kind == "reply"  # 不是 wait
    assert env["session"].dialog.slots["name"].value == "李明"
    assert env["session"].dialog.slots["purpose"].value == "想找张先生问快递"


# --------------------------------------------------------------------------- #
# 4) 号码没报完且超时 → 按普通轮次处理，计一次追问失败
# --------------------------------------------------------------------------- #


async def test_incomplete_phone_timeout_counts_as_failed_ask(tmp_path):
    def responder(text: str):
        if "姓李" in text:
            return "", {"name": "李明", "purpose": "想找张先生问快递"}
        return "", {}

    env = make_session(
        tmp_path,
        ["我姓李，想问下快递。", "一三八零零一三"],
        responder,
    )
    # 间隔 4s > vad.stop_secs(3s)：不再等
    segments = [(0.0, 1_000.0), (5_000.0, 5_500.0)]
    outcomes = await replay_segments(env["session"], segments, PCM_5S)

    assert [o.kind for o in outcomes] == ["reply", "reply"]
    assert env["session"].dialog.slots["callback_number"].value is None
    assert env["session"].dialog.attempts["callback_number"] == 2
    events = read_events(tmp_path / "call")
    assert any(e["type"] == "error" and e["data"].get("reason") == "incomplete_phone_timeout" for e in events)
    assert play_texts(events).count(PHRASES["ask_callback"]) == 2


# --------------------------------------------------------------------------- #
# 5) CONFIRM 整句合成失败 → render_phone 拼接兜底
# --------------------------------------------------------------------------- #


async def test_confirm_synth_failure_falls_back_to_render_phone(tmp_path):
    confirm_slots = {
        "name": "李明",
        "purpose": "想约张先生谈合同",
        "callback_number": "13800138000",
    }
    confirm_text = compose_confirm_text(
        {name: SlotValue(value) for name, value in confirm_slots.items()},
        owner_name=OWNER,
    )

    def responder(text: str):
        return "", {"name": "李明", "purpose": "想约张先生谈合同"}

    clock = FakeClock()
    env = make_session(
        tmp_path,
        ["我姓李，想约张先生谈合同，电话是一三八零零一三八零零零。"],
        responder,
        clock=clock,
        tts=FakeTts(clock, fail={confirm_text}),
    )
    segments = [(0.0, 1_000.0)]
    await replay_segments(env["session"], segments, PCM_5S)

    assert env["session"].dialog.slots["callback_number"].value == "13800138000"
    events = read_events(tmp_path / "call")
    assert any(e["type"] == "error" and e["data"].get("reason") == "confirm_synth_failed" for e in events)
    plays = play_texts(events)
    fallback = next(text for text in plays if text.startswith(PHRASES["confirm_phone_prefix"]))
    assert "幺三八" in fallback and "八零零零" in fallback
    # render_phone 的后缀已经包含“，对吗？”，不能再追加 confirm_question（避免重复）
    assert fallback.endswith(PHRASES["confirm_phone_suffix"])


# --------------------------------------------------------------------------- #
# 6) LlmTimeout → 兜底话术 + error
# --------------------------------------------------------------------------- #


async def test_llm_timeout_plays_repeat_and_logs_error(tmp_path):
    env = make_session(tmp_path, ["你好，我找一下张先生。"], no_slots, llm_timeout=True)
    segments = [(0.0, 1_000.0)]
    await replay_segments(env["session"], segments, PCM_5S)

    events = read_events(tmp_path / "call")
    assert any(e["type"] == "error" and e["data"].get("stage") == "llm" for e in events)
    assert play_texts(events)[-1] == PHRASES["repeat"]
    assert env["session"].dialog.state.value == "ASK_IDENTITY"


# --------------------------------------------------------------------------- #
# 7) 模板缓存缺失 → 现合成并补写缓存
# --------------------------------------------------------------------------- #


class ParseFailLlm:
    """输出无法解析的 JSON，模拟 s06 那类模型抖动。"""

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        raw = '{"reply": "您是刘先生", "slots": {"purpose": '
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply="",
                slots_update={},
                first_token_ms=100.0,
                total_ms=100.0,
                parse_ok=False,
                raw=raw,
            ),
            t_ms=100.0,
        )


async def test_parse_failure_logs_raw_output_and_plays_repeat(tmp_path):
    """T12b：解析失败事件带模型原文前 300 字，reason 统一为 parse。"""
    env = make_session(tmp_path, ["我姓刘，想问一下聚会。"], no_slots)
    env["session"].llm = ParseFailLlm()
    await replay_segments(env["session"], [(0.0, 1_000.0)], PCM_5S)

    events = read_events(tmp_path / "call")
    error = next(
        event
        for event in events
        if event["type"] == "error"
        and event["data"].get("stage") == "llm"
        and event["data"].get("reason") == "parse"
    )
    assert error["data"]["raw"].startswith('{"reply": "您是刘先生"')
    assert len(error["data"]["raw"]) <= 300
    assert play_texts(events)[-1] == PHRASES["repeat"]


async def test_cached_template_miss_is_synthesized_and_written_back(tmp_path):
    config = make_config(tmp_path)
    empty_cache = make_cache(config, fill=False)
    env = make_session(tmp_path, ["你好，我找一下张先生。"], no_slots, cache=empty_cache)
    # 清掉问候缓存，只留后续可能用到的模板
    await replay_segments(env["session"], [(0.0, 1_000.0)], PCM_5S)

    events = read_events(tmp_path / "call")
    assert any(
        e["type"] == "error" and e["data"].get("reason") == "phrase_cache_miss" for e in events
    )
    greet = PHRASES["greet"].format(owner=OWNER)
    assert env["cache"].get(greet) is not None
    assert play_texts(events)[0] == greet


# --------------------------------------------------------------------------- #
# 8) play_start 取首包时刻，而不是整句合成完之后
# --------------------------------------------------------------------------- #


async def test_play_start_is_logged_at_first_audio_chunk(tmp_path):
    clock = FakeClock()
    tts = FakeTts(clock, first_chunk_ms=100.0, chunk_ms=400.0, chunks=3)
    env = make_session(
        tmp_path,
        ["你好，我找一下张先生。"],
        no_slots,
        clock=clock,
        tts=tts,
        cache=make_cache(make_config(tmp_path), fill=False),
    )
    await replay_segments(env["session"], [(0.0, 1_000.0)], PCM_5S)

    events = read_events(tmp_path / "call")
    first_audio = next(
        e for e in events if e["type"] == "tts_first_audio" and e["data"]["text"] == PHRASES["ask_name"]
    )
    play_start = next(
        e for e in events if e["type"] == "play_start" and e["data"]["text"] == PHRASES["ask_name"]
    )
    play_end = next(
        e for e in events if e["type"] == "play_end" and e["data"]["text"] == PHRASES["ask_name"]
    )
    # 首包与开播同一时刻；后续两块各 400ms，play_end 必须晚于 play_start。
    assert play_start["t"] == first_audio["t"]
    assert play_end["t"] - play_start["t"] >= 800.0


# --------------------------------------------------------------------------- #
# 9) kind="llm" 分支仍然可用（T07c 之后 Dialog 不再返回它）
# --------------------------------------------------------------------------- #


async def test_llm_hint_plays_streamed_sentences(tmp_path):
    def responder(_text: str):
        return "你好。", {}

    env = make_session(tmp_path, ["你好。"], responder)
    session = env["session"]

    def fake_on_user_turn(text, slots_update=None, parse_ok=True):
        return session.dialog.state, ReplyHint(kind="llm", text="")

    session.dialog.on_user_turn = fake_on_user_turn
    segments = [(0.0, 1_000.0)]
    await replay_segments(session, segments, PCM_5S)

    events = read_events(tmp_path / "call")
    # 问候（缓存）之后，LLM 的句子逐句送 TTS 并播放
    assert play_texts(events)[1:] == ["你好。"]


# --------------------------------------------------------------------------- #
# 11) T11b：TTS 开播后中途出错，句子要标“（未说完）”
# --------------------------------------------------------------------------- #


async def test_confirm_lead_plays_prerendered_prefix_then_remainder(tmp_path):
    """T12 对比实验：先播预合成的“好的，跟您确认一下。”，再现合成后半句。"""
    from core.dialog import CONFIRM_PREFIX, compose_confirm_text

    def responder(text: str):
        return "", {"name": "李明", "purpose": "想约张先生谈合同"}

    env = make_session(
        tmp_path,
        ["我姓李，想约张先生谈合同，电话是一三八零零一三八零零零。"],
        responder,
        confirm_lead=True,
    )
    await replay_segments(env["session"], [(0.0, 1_000.0)], PCM_5S)

    confirm = compose_confirm_text(env["session"].dialog.slots, owner_name=OWNER)
    rest = confirm[len(CONFIRM_PREFIX) :]
    events = read_events(tmp_path / "call")
    assert play_texts(events) == [
        PHRASES["greet"].format(owner=OWNER),
        PHRASES["confirm_lead"],
        rest,
    ]
    # 预合成开头命中缓存，没有现合成；只有后半句送 TTS
    assert PHRASES["confirm_lead"] not in env["tts"].calls
    assert rest in env["tts"].calls
    # history 里两句拼成完整确认语
    assert env["session"].history[-1]["content"] == PHRASES["confirm_lead"] + rest


class BrokenMidStreamTts(FakeTts):
    """首块正常（真正开播），随后抛 TtsTimeout：模拟播到一半服务断掉。"""

    async def synth(self, text, cancel):
        self.calls.append(text)
        self.clock.advance(self.first_chunk_ms)
        yield b"\x00\x00" * 160
        raise TtsTimeout("fake tts broke mid-stream")


async def test_tts_error_after_play_start_marks_sentence_unfinished(tmp_path):
    clock = FakeClock()
    env = make_session(
        tmp_path,
        ["你好，我找一下张先生。"],
        no_slots,
        clock=clock,
        tts=BrokenMidStreamTts(clock),
        cache=make_cache(make_config(tmp_path), fill=False),
    )
    await replay_segments(env["session"], [(0.0, 1_000.0)], PCM_5S)

    events = read_events(tmp_path / "call")
    ask = PHRASES["ask_name"]
    sequence = [
        event["type"]
        for event in events
        if (event.get("data") or {}).get("text") == ask
        and event["type"]
        in {"tts_request", "tts_first_audio", "play_start", "error", "interrupted", "play_end"}
        # 缓存没命中的 error 发生在合成之前，不属于“播到一半出错”的序列。
        and (event.get("data") or {}).get("reason") != "phrase_cache_miss"
    ]
    assert sequence == [
        "tts_request",
        "tts_first_audio",
        "play_start",
        "error",
        "interrupted",
        "play_end",
    ]
    interrupted = next(
        event
        for event in events
        if event["type"] == "interrupted" and event["data"].get("text") == ask
    )
    assert interrupted["data"]["reason"] == "tts_error"

    # 摘要侧：这句只播了一部分，必须标“（未说完）”。
    from core.summary import INTERRUPTED_MARK, build_transcript

    assert f"AI：{ask}{INTERRUPTED_MARK}" in build_transcript(events)


async def test_play_pcm_records_only_sent_audio_on_interrupt(tmp_path):
    """T13b：被抢话时 ai_pcm 只写已经送出的部分，不把整句写进录音。"""
    from core.call import PLAYBACK_CHUNK_BYTES

    env = make_session(tmp_path, [], no_slots)
    session = env["session"]
    chunk = b"\x01\x00" * (PLAYBACK_CHUNK_BYTES // 2)
    pcm = chunk * 3
    # 取消开关已在送第一块前置位：_play_pcm 送完第一块就停。
    session._tts_cancel.set()
    await session._play_pcm(pcm, "测试句子")

    assert bytes(session.ai_pcm) == chunk
    assert session._turn_interrupted is True
    events = read_events(tmp_path / "call")
    interrupted = [event for event in events if event["type"] == "interrupted"]
    assert interrupted and interrupted[-1]["data"]["text"] == "测试句子"


def test_mark_last_assistant_interrupted(tmp_path):
    """T13b：尾音抢话时把最后一条 assistant 历史标成（被对方打断）。"""
    env = make_session(tmp_path, [], no_slots)
    session = env["session"]
    session.history = [
        {"role": "user", "content": "喂。"},
        {"role": "assistant", "content": "您好。"},
    ]
    session.mark_last_assistant_interrupted()
    assert session.history[-1]["content"] == "您好。（被对方打断）"

    # 已经标过就不重复追加。
    session.mark_last_assistant_interrupted()
    assert session.history[-1]["content"] == "您好。（被对方打断）"

    # 最后一条不是 assistant（对方已经说了新的一轮）则不动。
    session.history.append({"role": "user", "content": "在吗？"})
    session.mark_last_assistant_interrupted()
    assert session.history[-1]["content"] == "在吗？"


def test_mark_last_assistant_interrupted_truncates_unplayed_sentences(tmp_path):
    """T13c：一轮多句时，标记加在被打断那句后面，它之后的句子（没播出）从历史里去掉。"""
    env = make_session(tmp_path, [], no_slots)
    session = env["session"]
    first = "好的，跟您确认一下。"
    second = "您是王先生，对吗？"
    session.history = [
        {"role": "user", "content": "我姓王。"},
        {"role": "assistant", "content": first + second},
    ]
    session.mark_last_assistant_interrupted(first)
    assert session.history[-1]["content"] == first + "（被对方打断）"

    # 被打断的是最后一句：整条保留，只补标记。
    session.history[-1] = {"role": "assistant", "content": first + second}
    session.mark_last_assistant_interrupted(second)
    assert session.history[-1]["content"] == first + second + "（被对方打断）"

    # 找不到被打断的那句：退回原行为，标记加在整条末尾。
    session.history[-1] = {"role": "assistant", "content": first + second}
    session.mark_last_assistant_interrupted("完全不相干的一句")
    assert session.history[-1]["content"] == first + second + "（被对方打断）"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

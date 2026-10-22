"""T11 验收测试：挂机摘要与校正。

全部用 fake LLM / fake ASR，不联网、不 sleep：

- 七字段齐全、summary.json 落盘；
- 对话文本区分来电者与 AI，被打断的 AI 句子带“（未说完）”并写进提示词；
- 提示词约束“未确认”“被打断不算已告知”；
- 复核不一致 → 进入“未确认信息”，槽位不被覆盖；
- 复核一致（含汉字数字号码）→ 没有“未确认信息”条目。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.summary import (
    INTERRUPTED_MARK,
    SUMMARY_FIELDS,
    RecheckResult,
    SummaryError,
    build_recheck_asr,
    build_summary_messages,
    build_transcript,
    dialogue_lines,
    merge_unconfirmed,
    parse_summary,
    realtime_transcript,
    recheck_slots,
    summarize,
    summarize_call,
    write_summary,
)
from core.types import AsrResult, SlotValue

OWNER = "张先生"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeLlm:
    """只实现摘要要用的 `complete()`，记录收到的 messages。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict], model: str | None = None) -> str:
        self.calls.append(messages)
        return self.reply

    @property
    def last_user_message(self) -> str:
        return self.calls[-1][-1]["content"]

    @property
    def last_system_message(self) -> str:
        return self.calls[-1][0]["content"]


class FakeAsr:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[bytes] = []

    async def transcribe(self, pcm: bytes) -> AsrResult:
        self.calls.append(pcm)
        return AsrResult(text=self.text, latency_ms=12.0, audio_ms=len(pcm) / 32.0)


GOOD_SUMMARY = {
    "来电者": "快递员",
    "来意": "送快递，家里没人",
    "已确认事项": "明天下午三点再送",
    "待办": "明天下午三点重新送件",
    "回拨需求": "已留下回拨号码 13800138000",
    "未确认信息": "无",
    "接管情况": "无",
}


def summary_json(**overrides) -> str:
    payload = dict(GOOD_SUMMARY)
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def slot_dict_with_recheck_events() -> list[dict]:
    """一通简单电话：来电者报名／说来意／报号码，AI 有一句被打断。"""
    return [
        {"call_id": "c1", "t": 0.0, "type": "call_start", "data": {"source": "replay"}},
        {"call_id": "c1", "t": 10.0, "type": "play_start", "data": {"text": "您好，张先生不在，请讲。"}},
        {"call_id": "c1", "t": 900.0, "type": "play_end", "data": {"text": "您好，张先生不在，请讲。", "duration_ms": 890}},
        {"call_id": "c1", "t": 1000.0, "type": "asr_final", "data": {"text": "我是快递，送快递的。"}},
        {"call_id": "c1", "t": 1100.0, "type": "play_start", "data": {"text": "好的，我记一下，您明天"}},
        {"call_id": "c1", "t": 1300.0, "type": "interrupted", "data": {"reason": "barge_in"}},
        {"call_id": "c1", "t": 1400.0, "type": "asr_final", "data": {"text": "不是明天，是后天下午三点。"}},
        {"call_id": "c1", "t": 1500.0, "type": "play_start", "data": {"text": "好的，后天下午三点，对吗？"}},
        {"call_id": "c1", "t": 2400.0, "type": "play_end", "data": {"text": "好的，后天下午三点，对吗？", "duration_ms": 890}},
        {"call_id": "c1", "t": 2500.0, "type": "vad_end", "data": {"end_ms": 2500.0}},
        {"call_id": "c1", "t": 2600.0, "type": "call_end", "data": {"state": "CONFIRM", "turns": 2}},
    ]


def slots_snapshot() -> dict[str, SlotValue]:
    return {
        "name": SlotValue("快递员", "confirmed"),
        "purpose": SlotValue("送快递", "confirmed"),
        "callback_number": SlotValue("13800138000", "confirmed"),
        "time": SlotValue("后天下午三点", "unconfirmed"),
        "location": SlotValue(None, "empty"),
    }


# --------------------------------------------------------------------------- #
# 摘要
# --------------------------------------------------------------------------- #


async def test_summarize_returns_seven_fields_and_writes_summary(tmp_path: Path):
    llm = FakeLlm(summary_json())
    payload = await summarize(
        slot_dict_with_recheck_events(),
        slots_snapshot(),
        llm,
        call_dir=tmp_path,
        owner_name=OWNER,
    )

    assert tuple(payload) == SUMMARY_FIELDS
    assert payload["来电者"] == "快递员"

    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written == payload

    user_message = llm.last_user_message
    assert f"【机主】{OWNER}" in user_message
    assert "来电者：我是快递，送快递的。" in user_message
    assert "AI：您好，张先生不在，请讲。" in user_message
    assert "回拨号码（callback_number）：13800138000（已确认）" in user_message
    assert "人工接管：否" in user_message


async def test_prompt_marks_interrupted_ai_sentence_only():
    events = slot_dict_with_recheck_events()
    transcript = build_transcript(events)

    assert f"AI：好的，我记一下，您明天{INTERRUPTED_MARK}" in transcript
    # 播完的句子不带标记
    assert transcript.endswith("AI：好的，后天下午三点，对吗？")
    assert f"好的，后天下午三点，对吗？{INTERRUPTED_MARK}" not in transcript

    llm = FakeLlm(summary_json())
    await summarize(events, slots_snapshot(), llm)
    user_message = llm.last_user_message
    assert INTERRUPTED_MARK in user_message
    # 提示词必须约束“没说完的句子不算已告知”
    system_message = llm.last_system_message
    assert INTERRUPTED_MARK in system_message
    assert "不算已告知" in system_message
    assert "未确认信息" in system_message


def test_turn_merged_interrupted_is_not_an_ai_line():
    """T10 的轮次合并：interrupted 里带的是来电者文本，且这一轮从未播出。"""
    events = [
        {"type": "asr_final", "data": {"text": "喂。"}},
        {"type": "interrupted", "data": {"reason": "turn_merged", "text": "喂。"}},
        {"type": "asr_final", "data": {"text": "你好，我找一下张先生。"}},
    ]
    lines = dialogue_lines(events)
    assert [line.speaker for line in lines] == ["来电者", "来电者"]
    assert not any(line.interrupted for line in lines)
    assert INTERRUPTED_MARK not in build_transcript(events)


def test_play_start_without_play_end_is_unfinished():
    events = [
        {"type": "play_start", "data": {"text": "喂，您好，这里是"}},
        {"type": "asr_final", "data": {"text": "你好。"}},
    ]
    transcript = build_transcript(events)
    # 没有 play_end 也没有 interrupted：按播放结束处理（不猜测打断），但不与下一句串行
    assert "AI：喂，您好，这里是" in transcript
    assert transcript.count("AI：") == 1


def test_parse_summary_fills_missing_fields_and_joins_lists():
    raw = (
        "```json\n"
        '{"来电者": "快递员", "来意": ["送快递", "家里没人"], "已确认事项": "", '
        '"未确认信息": null, "接管情况": "无"}\n'
        "```"
    )
    payload = parse_summary(raw)
    assert tuple(payload) == SUMMARY_FIELDS
    assert payload["来意"] == "送快递；家里没人"
    assert payload["待办"] == ""
    assert payload["未确认信息"] == ""


def test_parse_summary_rejects_non_json():
    with pytest.raises(SummaryError):
        parse_summary("抱歉，我无法完成这个请求。")


async def test_summarize_accepts_events_path_and_writes_next_to_it(tmp_path: Path):
    call_dir = tmp_path / "calls" / "c2"
    call_dir.mkdir(parents=True)
    (call_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in slot_dict_with_recheck_events()) + "\n",
        encoding="utf-8",
    )
    slots_path = call_dir / "slots.json"
    slots_path.write_text(
        json.dumps(
            {
                "call_id": "c2",
                "state": "CONFIRM",
                "slots": {
                    name: {"value": value.value, "status": value.status}
                    for name, value in slots_snapshot().items()
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    llm = FakeLlm(summary_json())
    payload = await summarize(call_dir / "events.jsonl", json.loads(slots_path.read_text()), llm)
    assert json.loads((call_dir / "summary.json").read_text(encoding="utf-8")) == payload


def test_write_summary_rejects_unknown_fields(tmp_path: Path):
    path = write_summary(tmp_path, {**GOOD_SUMMARY, "多余字段": "丢掉"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(payload) == SUMMARY_FIELDS
    assert "多余字段" not in payload


# --------------------------------------------------------------------------- #
# 校正
# --------------------------------------------------------------------------- #


async def test_recheck_mismatch_goes_to_unconfirmed_and_keeps_slots():
    slots = slots_snapshot()
    before = dict(slots)
    asr_alt = FakeAsr("我是快递员，我的电话是幺三八零零一三八零零一。")
    result = await recheck_slots(b"\x00\x00" * 1600, slots, asr_alt)

    assert isinstance(result, RecheckResult)
    assert not result.consistent
    assert any("13800138001" in note for note in result.mismatches)
    assert result.values["name"] == "快递员"
    assert slots == before  # 槽位不被覆盖

    llm = FakeLlm(summary_json(未确认信息="时间未确认"))
    payload = await summarize(
        slot_dict_with_recheck_events(),
        slots,
        llm,
        recheck_notes=result.mismatches,
    )
    assert "时间未确认" in payload["未确认信息"]
    assert "13800138001" in payload["未确认信息"]
    assert payload["回拨需求"] == GOOD_SUMMARY["回拨需求"]  # 其余字段不受影响


async def test_recheck_name_missing_in_both_transcripts_is_not_reported():
    """T11b：姓名“快递员”是 LLM 归纳的、两遍转写都没有 → 不能报不一致。"""
    asr_alt = FakeAsr("送快递的，电话 13800138000。")
    slots = {
        "name": SlotValue("快递员", "unconfirmed"),
        "callback_number": SlotValue("13800138000", "confirmed"),
    }
    result = await recheck_slots(
        b"\x00\x00" * 1600, slots, asr_alt, realtime_text="我是送快递的，有个包裹。"
    )
    assert result.values["name"] is None
    assert result.consistent, result.mismatches


async def test_recheck_name_in_realtime_but_not_in_alt_is_reported():
    """T11b：实时转写有“张伟”、复核转写是“张为” → 报告。"""
    asr_alt = FakeAsr("我姓张为，电话 13800138000。")
    slots = {
        "name": SlotValue("张伟", "unconfirmed"),
        "callback_number": SlotValue("13800138000", "confirmed"),
    }
    result = await recheck_slots(
        b"\x00\x00" * 1600, slots, asr_alt, realtime_text="我姓张伟，电话一三八零零一三八零零零。"
    )
    assert result.values["name"] is None
    assert any("张伟" in note for note in result.mismatches)


async def test_recheck_corrected_phone_is_consistent_when_recorded_number_is_present():
    """T11b：来电者纠正号码时转写里有两个号码，记录的在其中 → 一致。"""
    asr_alt = FakeAsr("电话是 13800138000，不对，是 13900138000。")
    slots = {"callback_number": SlotValue("13900138000", "confirmed")}
    result = await recheck_slots(b"\x00\x00" * 1600, slots, asr_alt)
    assert result.consistent, result.mismatches
    assert result.values["callback_number"] == "13900138000"


async def test_recheck_corrected_phone_reports_the_last_number():
    """T11b：记录里的号码不在转写中 → 报最后一个号码。"""
    asr_alt = FakeAsr("电话是 13800138000，不对，是 13900138000。")
    slots = {"callback_number": SlotValue("13700137000", "confirmed")}
    result = await recheck_slots(b"\x00\x00" * 1600, slots, asr_alt)
    assert not result.consistent
    assert "13900138000" in result.mismatches[0]
    assert "13700137000" in result.mismatches[0]


async def test_realtime_transcript_joins_asr_finals():
    events = [
        {"type": "asr_final", "data": {"text": "喂。"}},
        {"type": "asr_final", "data": {"text": "我姓张伟。"}},
        {"type": "asr_final", "data": {"text": "   "}},
        {"type": "llm_done", "data": {"reply": "不该出现"}},
    ]
    assert realtime_transcript(events) == "喂。\n我姓张伟。"


def test_tts_error_sentence_is_marked_unfinished():
    """T11b：play_start → error(tts) → interrupted(tts_error) → play_end 的句子要标未说完。"""
    events = [
        {"type": "asr_final", "data": {"text": "你好，我找一下张先生。"}},
        {"type": "play_start", "data": {"text": "好的，请稍等，我帮您转告张先生"}},
        {
            "type": "error",
            "data": {"stage": "tts", "reason": "TtsTimeout", "text": "好的，请稍等，我帮您转告张先生"},
        },
        {"type": "interrupted", "data": {"reason": "tts_error", "text": "好的，请稍等，我帮您转告张先生"}},
        {
            "type": "play_end",
            "data": {"text": "好的，请稍等，我帮您转告张先生", "duration_ms": 320},
        },
    ]
    transcript = build_transcript(events)
    assert f"AI：好的，请稍等，我帮您转告张先生{INTERRUPTED_MARK}" in transcript


async def test_build_recheck_asr_uses_recheck_config(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    config = {
        "siliconflow": {"base_url": "https://api.example.test/v1"},
        "asr": {
            "model": "Qwen/Qwen3-ASR-1.7B",
            "recheck_model": "FunAudioLLM/SenseVoiceSmall",
            "attempt_timeout_s": 3,
            "total_timeout_s": 5,
            "recheck_attempt_timeout_s": 25,
            "recheck_total_timeout_s": 70,
        },
        "audio": {"sample_rate": 16000},
    }
    client = build_recheck_asr(config)
    try:
        assert client.model == "FunAudioLLM/SenseVoiceSmall"
        assert client.attempt_timeout_s == 25
        assert client.total_timeout_s == 70
    finally:
        await client.aclose()


async def test_summarize_call_reports_name_mismatch_using_realtime_transcript(tmp_path: Path):
    """T11b：summarize_call 会把实时 asr_final 拼起来交给复核姓名的判定。"""
    call_dir = tmp_path / "calls" / "c4"
    call_dir.mkdir(parents=True)
    events = [
        {"call_id": "c4", "t": 0.0, "type": "asr_final", "data": {"text": "我姓张伟。"}},
        {"call_id": "c4", "t": 1.0, "type": "asr_final", "data": {"text": "电话 13800138000。"}},
    ]
    (call_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    (call_dir / "slots.json").write_text(
        json.dumps(
            {
                "call_id": "c4",
                "state": "CONFIRM",
                "slots": {
                    "name": {"value": "张伟", "status": "unconfirmed"},
                    "callback_number": {"value": "13800138000", "status": "confirmed"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    import wave

    with wave.open(str(call_dir / "caller.wav"), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)

    llm = FakeLlm(summary_json())
    payload = await summarize_call(call_dir, llm, asr_alt=FakeAsr("我姓张为，电话 13800138000。"))
    assert "张伟" in payload["未确认信息"]


async def test_recheck_consistent_after_chinese_digit_normalization():
    # 复核转写把号码写成汉字数字 + 分组读法，应视为一致
    asr_alt = FakeAsr("我是快递员，回拨号码幺三八，零零幺三，八零零零。")
    result = await recheck_slots(b"\x00\x00" * 1600, slots_snapshot(), asr_alt)
    assert result.consistent, result.mismatches
    assert result.values["callback_number"] == "13800138000"


async def test_recheck_notes_slot_missing_in_record():
    slots = slots_snapshot()
    slots["callback_number"] = SlotValue(None, "unconfirmed")
    slots["name"] = SlotValue(None, "unconfirmed")
    asr_alt = FakeAsr("回拨号码 13800138000。")
    result = await recheck_slots(b"\x00\x00" * 1600, slots, asr_alt)
    assert any("没有记录" in note for note in result.mismatches)


def test_merge_unconfirmed_ignores_blank_notes():
    payload = {field: "" for field in SUMMARY_FIELDS}
    merge_unconfirmed(payload, ["  ", ""])
    assert payload["未确认信息"] == ""


async def test_summarize_call_end_to_end(tmp_path: Path):
    call_dir = tmp_path / "calls" / "c3"
    call_dir.mkdir(parents=True)
    (call_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in slot_dict_with_recheck_events()) + "\n",
        encoding="utf-8",
    )
    (call_dir / "slots.json").write_text(
        json.dumps(
            {
                "call_id": "c3",
                "state": "CONFIRM",
                "slots": {
                    name: {"value": value.value, "status": value.status}
                    for name, value in slots_snapshot().items()
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    import wave

    with wave.open(str(call_dir / "caller.wav"), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)

    llm = FakeLlm(summary_json())
    asr_alt = FakeAsr("我是快递员，电话 13800138001。")
    payload = await summarize_call(call_dir, llm, asr_alt=asr_alt)

    assert asr_alt.calls, "应把 caller.wav 交给复核 ASR"
    assert "13800138001" in payload["未确认信息"]
    assert json.loads((call_dir / "summary.json").read_text(encoding="utf-8")) == payload
    # 槽位文件保持原样
    stored = json.loads((call_dir / "slots.json").read_text(encoding="utf-8"))
    assert stored["slots"]["callback_number"]["value"] == "13800138000"


def test_build_summary_messages_without_events():
    messages = build_summary_messages([], {})
    assert "（无对话记录）" in messages[-1]["content"]
    assert "（没有槽位记录）" in messages[-1]["content"]

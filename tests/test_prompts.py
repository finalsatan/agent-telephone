"""core/prompts.py 的约定测试：身份约束、槽位/S续问描述、组装顺序。"""

from __future__ import annotations

from core.prompts import SYSTEM_PROMPT, build_messages, build_user_message, missing_required
from core.types import SLOT_NAMES, SlotValue, State


def test_system_prompt_keeps_power_limits():
    # 设计文档第 4 节：只接待与记录，不承诺、不确认、不泄露隐私
    assert "不承诺" in SYSTEM_PROMPT
    assert "不确认预约" in SYSTEM_PROMPT
    assert "隐私" in SYSTEM_PROMPT
    assert "我会转告" in SYSTEM_PROMPT or "转告机主" in SYSTEM_PROMPT


def test_system_prompt_requires_json_and_lists_all_slots():
    assert '{"reply"' in SYSTEM_PROMPT and '"slots"' in SYSTEM_PROMPT
    for name in SLOT_NAMES:
        assert name in SYSTEM_PROMPT
    assert "callback_number" in SYSTEM_PROMPT  # 明确告知模型不要输出该槽位


def test_system_prompt_constrains_reply_style():
    assert "最多 2 句" in SYSTEM_PROMPT
    assert "emoji" in SYSTEM_PROMPT


def test_missing_required():
    assert missing_required({}) == ["name", "purpose", "callback_number"]
    slots = {
        "name": SlotValue("张先生", "unconfirmed"),
        "purpose": SlotValue("送快递", "unconfirmed"),
    }
    assert missing_required(slots) == ["callback_number"]
    slots["callback_number"] = SlotValue("13800138000", "confirmed")
    assert missing_required(slots) == []


def test_missing_required_ignores_optional_slots():
    slots = {
        "name": SlotValue("张先生"),
        "purpose": SlotValue("送快递"),
        "callback_number": SlotValue("13800138000"),
    }
    assert missing_required(slots) == []  # time/location 缺失不影响


def test_user_message_contains_state_slots_and_missing():
    slots = {"name": SlotValue("张先生", "unconfirmed"), "purpose": SlotValue(None)}
    msg = build_user_message(State.ASK_PURPOSE, slots, "我找他有点事")
    assert "ASK_PURPOSE" in msg
    recorded = msg.split("已记录的槽位：")[1].split("缺失的必填槽位：")[0]
    assert "name: 张先生（未确认）" in recorded
    assert "purpose" not in recorded  # 空值不算已记录
    assert "缺失的必填槽位：purpose、callback_number" in msg
    assert "「我找他有点事」" in msg


def test_user_message_shows_status_in_chinese():
    slots = {"name": SlotValue("张先生", "corrected")}
    assert "（已纠正）" in build_user_message(State.CONFIRM, slots, "不是，我姓李")


def test_user_message_without_slots_says_none():
    assert "（暂无）" in build_user_message(State.GREETING, {}, "喂")


def test_user_message_without_missing_slots_says_none():
    slots = {
        "name": SlotValue("张先生"),
        "purpose": SlotValue("送快递"),
        "callback_number": SlotValue("13800138000"),
    }
    msg = build_user_message(State.CONFIRM, slots, "对")
    assert "缺失的必填槽位：（无）" in msg


def test_user_message_reports_retry_counts():
    slots = {"name": SlotValue("张先生")}
    msg = build_user_message(State.FILL_MISSING, slots, "嗯……", attempts={"purpose": 2, "callback_number": 0})
    assert "purpose 已追问 2 次" in msg
    assert "callback_number 已追问 0 次" not in msg  # 0 次的不必打扰模型


def test_build_messages_order():
    history = [
        {"role": "assistant", "content": "您好"},
        {"role": "user", "content": "我找张先生"},
    ]
    messages = build_messages(State.ASK_PURPOSE, {}, history, "送快递")
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert "送快递" in messages[-1]["content"]
    assert len(messages) == 4

"""core/prompts.py 的约定测试：身份约束、槽位/追问描述、机主称呼、组装顺序。"""

from __future__ import annotations

from core.prompts import (
    CONFIRM_GUIDE_WITH_PHONE,
    STATE_GUIDE,
    build_messages,
    build_system_prompt,
    build_user_message,
    missing_required,
)
from core.types import SLOT_NAMES, SlotValue, State

SYSTEM_PROMPT = build_system_prompt("张先生")


def test_system_prompt_uses_owner_name():
    # 0.6：机主称呼来自 config.yaml: owner.name，不得在代码里写死
    assert SYSTEM_PROMPT.startswith("你是张先生的电话助理")
    assert "我会转告张先生" in SYSTEM_PROMPT
    assert build_system_prompt("王女士").startswith("你是王女士的电话助理")
    assert "张先生" not in build_system_prompt("王女士")
    # 称呼为空/空白时退回通用称呼，而不是生成“你是的电话助理”
    assert build_system_prompt("  ").startswith("你是机主的电话助理")


def test_system_prompt_requires_reply_first_for_streaming():
    # T02b：边收边播要求 reply 是第一个键
    assert "reply 必须是第一个键" in SYSTEM_PROMPT
    idx_reply = SYSTEM_PROMPT.index('{"reply"')
    idx_slots = SYSTEM_PROMPT.index('"slots"', idx_reply)
    assert idx_reply < idx_slots


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


def test_build_messages_order_and_owner():
    history = [
        {"role": "assistant", "content": "您好"},
        {"role": "user", "content": "我找张先生"},
    ]
    messages = build_messages(State.ASK_PURPOSE, {}, history, "送快递", owner_name="李女士")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("你是李女士的电话助理")
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert "送快递" in messages[-1]["content"]
    assert len(messages) == 4


def test_build_messages_passes_attempts_through():
    messages = build_messages(State.FILL_MISSING, {}, [], "嗯", attempts={"callback_number": 2})
    assert "callback_number 已追问 2 次" in messages[-1]["content"]


def test_confirm_guide_with_phone_leaves_number_to_render_phone():
    # T07b：号码由系统 render_phone 播报，LLM 只复述姓名/来意，且不要用问句结尾
    slots = {
        "name": SlotValue("李先生"),
        "purpose": SlotValue("送快递"),
        "callback_number": SlotValue("13800138000"),
    }
    msg = build_user_message(State.CONFIRM, slots, "对")
    assert "不要念回拨号码" in msg
    assert "系统会接着播报号码并请对方确认" in msg
    assert "复述对方的姓名和来意" in msg
    assert STATE_GUIDE[State.CONFIRM] not in msg


def test_confirm_guide_without_phone_keeps_original():
    # 号码为空时沿用 STATE_GUIDE 里的说明（接口保持不变）
    slots = {"name": SlotValue("李先生"), "purpose": SlotValue("送快递")}
    msg = build_user_message(State.CONFIRM, slots, "对")
    assert STATE_GUIDE[State.CONFIRM] in msg
    assert "不要念回拨号码" not in msg


def test_confirm_guide_with_empty_phone_value_keeps_original():
    slots = {"callback_number": SlotValue(None, "unconfirmed")}
    msg = build_user_message(State.CONFIRM, slots, "对")
    assert STATE_GUIDE[State.CONFIRM] in msg


def test_phone_guide_does_not_leak_to_other_states():
    slots = {"callback_number": SlotValue("13800138000")}
    for state in (State.ASK_IDENTITY, State.ASK_PURPOSE, State.FILL_MISSING, State.CLOSING):
        msg = build_user_message(state, slots, "嗯")
        assert "不要念回拨号码" not in msg
        assert STATE_GUIDE[state] in msg


def test_confirm_negative_reply_asks_which_item_is_wrong():
    # T07b 要求 3：否定且没给新值时让 LLM 追问哪一项不对，不要原样复述
    slots = {
        "name": SlotValue("李先生"),
        "purpose": SlotValue("送快递"),
        "callback_number": SlotValue("13800138000"),
    }
    msg = build_user_message(State.CONFIRM, slots, "不对，好像号码少了一位")
    assert "问是哪一项不对" in msg
    assert "不要念回拨号码" in msg
    assert "不要重复复述" in msg
    assert CONFIRM_GUIDE_WITH_PHONE not in msg


def test_confirm_negative_without_phone_also_asks_correction():
    slots = {"name": SlotValue("李先生")}
    msg = build_user_message(State.CONFIRM, slots, "不对")
    assert "问是哪一项不对" in msg
    assert STATE_GUIDE[State.CONFIRM] not in msg

"""core/prompts.py 的约定测试：身份约束、槽位/追问描述、机主称呼、组装顺序。"""

from __future__ import annotations

import pytest

from core.prompts import (
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


def test_confirm_guide_only_extracts_corrected_slots():
    """T07c：确认语由代码生成，提示词只要求 LLM 提取被纠正的槽位。"""
    slots = {
        "name": SlotValue("李先生"),
        "purpose": SlotValue("送快递"),
        "callback_number": SlotValue("13800138000"),
    }
    msg = build_user_message(State.CONFIRM, slots, "对，我姓李")
    assert "只提取对方这一轮纠正的槽位" in msg
    assert "复述由系统完成" in msg
    # 不再出现“不要念回拨号码 / 系统会接着播报号码”这类 T07b 的说明
    assert "不要念回拨号码" not in msg
    assert "播报号码" not in msg


def test_confirm_guide_is_same_with_or_without_phone():
    with_phone = {"callback_number": SlotValue("13800138000")}
    without_phone = {"name": SlotValue("李先生")}
    for slots in (with_phone, without_phone):
        msg = build_user_message(State.CONFIRM, slots, "对")
        assert STATE_GUIDE[State.CONFIRM] in msg


def test_purpose_writing_rule_is_in_system_prompt():
    """T07c 要求 6：purpose 要能直接接在“您是某某，”后面。"""
    assert "purpose：对方为什么打来，写成能直接接在" in SYSTEM_PROMPT
    assert '"想……"或"关于……"' in SYSTEM_PROMPT
    assert "不超过 15 字" in SYSTEM_PROMPT
    assert "想约张先生谈合同" in SYSTEM_PROMPT
    assert "想约王女士谈合同" in build_system_prompt("王女士")


def test_reply_mode_minimal_fixes_reply_text():
    """T12 对比实验：minimal 模式要求 reply 固定写“好的。”。"""
    from core.prompts import DEFAULT_REPLY_MODE, MINIMAL_REPLY_TEXT

    full = build_system_prompt("张先生")
    minimal = build_system_prompt("张先生", reply_mode="minimal")
    assert DEFAULT_REPLY_MODE == "full"
    assert MINIMAL_REPLY_TEXT == "好的。"
    assert "reply 固定写“好的。”" in minimal
    assert "reply 固定写“好的。”" not in full
    # 仍要保持 reply 在前的 JSON 格式，parse_reply 不用改
    for prompt in (full, minimal):
        assert "reply 必须是第一个键" in prompt
        assert '{"reply"' in prompt
    # messages 组装时透传
    messages = build_messages(State.ASK_PURPOSE, {}, [], "送快递", reply_mode="minimal")
    assert "reply 固定写“好的。”" in messages[0]["content"]
    with pytest.raises(ValueError):
        build_system_prompt("张先生", reply_mode="short")


def test_reply_mode_slots_only_drops_reply():
    """T12b：slots_only 只输出 slots，不再要求 reply 在第一个键。"""
    from core.prompts import REPLY_MODES

    prompt = build_system_prompt("张先生", reply_mode="slots_only")
    assert "slots_only" in REPLY_MODES
    assert "只输出 slots，不要输出 reply 键" in prompt
    assert '{"slots": {"purpose": "对方来意"}}' in prompt
    assert "reply 必须是第一个键" not in prompt
    messages = build_messages(State.ASK_PURPOSE, {}, [], "送快递", reply_mode="slots_only")
    assert "不要输出 reply 键" in messages[0]["content"]


def test_slot_definitions_and_examples_are_in_prompt():
    """T12b：每个槽位要有定义和正反例，避免职业/公司/机主被当成 name。"""
    prompt = build_system_prompt("张先生")
    assert "name：只写来电者本人的姓名" in prompt
    assert "职业、公司不是姓名" in prompt
    assert "对方只报了职业、没有报姓名时，name 留空" in prompt
    assert '对方说"我找张先生" → 不要输出 name' in prompt
    assert "time：对方提到的时间都要提取" in prompt
    assert "对方纠正时间时，只写纠正后的时间" in prompt
    assert "location：地址写在这里" in prompt
    assert "purpose 里不要再带地址" in prompt
    assert "无论当前状态在问什么" in prompt


def test_prompts_does_not_import_dialog():
    """T07c：`prompts` 不再导入 `dialog`，否则 T06b 之后会成环。"""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/prompts.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert not any(module.startswith("core.dialog") for module in modules)


def test_import_dialog_prompts_phrases_together():
    """验收要求：三个模块一起导入不报错（循环导入回归）。"""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", "import core.dialog, core.prompts, core.phrases"],
        check=True,
    )


def test_slots_only_user_message_does_not_ask_for_reply():
    """T12b：slots_only 的状态说明不能再要求“请询问对方”，否则模型会直接回一句中文。"""
    slots_only = build_user_message(State.ASK_IDENTITY, {}, "我找张先生", reply_mode="slots_only")
    assert "本轮只提取槽位，不要生成任何回复文本" in slots_only
    assert STATE_GUIDE[State.ASK_IDENTITY] not in slots_only
    full = build_user_message(State.ASK_IDENTITY, {}, "我找张先生")
    assert STATE_GUIDE[State.ASK_IDENTITY] in full

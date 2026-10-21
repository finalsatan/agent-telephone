"""core/types.py 的约定测试：类型字段、默认值、槽位常量与取值校验。

这些是跨任务约定（0.2），测试的作用是防止后续任务悄悄改掉字段名或默认值。
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from core.types import (
    OPTIONAL_SLOTS,
    REQUIRED_SLOTS,
    SLOT_NAMES,
    SLOT_STATUSES,
    AsrResult,
    SlotValue,
    State,
    TurnResult,
)


def test_asr_result_fields():
    r = AsrResult(text="您好", latency_ms=612.5, audio_ms=2480.0)
    assert [f.name for f in fields(AsrResult)] == ["text", "latency_ms", "audio_ms"]
    assert (r.text, r.latency_ms, r.audio_ms) == ("您好", 612.5, 2480.0)


def test_turn_result_fields_and_parse_ok_default():
    assert [f.name for f in fields(TurnResult)] == [
        "reply",
        "slots_update",
        "first_token_ms",
        "total_ms",
        "parse_ok",
    ]
    t = TurnResult(reply="好的", slots_update={"name": "张先生"}, first_token_ms=540.0, total_ms=900.0)
    assert t.parse_ok is True  # 解析成功是常态，默认 True


def test_slot_value_defaults_to_empty():
    v = SlotValue(value=None)
    assert v.value is None
    assert v.status == "empty"


@pytest.mark.parametrize("status", SLOT_STATUSES)
def test_slot_value_accepts_all_documented_statuses(status):
    assert SlotValue(value="x", status=status).status == status


def test_slot_value_rejects_unknown_status():
    with pytest.raises(ValueError, match="invalid slot status"):
        SlotValue(value="x", status="confirmedd")  # 拼写错误必须立刻暴露


def test_slot_names_are_fixed():
    assert SLOT_NAMES == ("name", "purpose", "callback_number", "time", "location")
    assert REQUIRED_SLOTS == ("name", "purpose", "callback_number")
    assert OPTIONAL_SLOTS == ("time", "location")
    # 必填 + 选填 正好等于全部槽位，不多不少
    assert set(REQUIRED_SLOTS) | set(OPTIONAL_SLOTS) == set(SLOT_NAMES)
    assert not set(REQUIRED_SLOTS) & set(OPTIONAL_SLOTS)


def test_state_values_and_str_enum_behaviour():
    assert [s.value for s in State] == [
        "GREETING",
        "ASK_IDENTITY",
        "ASK_PURPOSE",
        "FILL_MISSING",
        "CONFIRM",
        "CLOSING",
    ]
    # str 子类：可直接与字符串比较、可 JSON 序列化（后续写 events/slots 用到）
    assert State.GREETING == "GREETING"
    assert State("CONFIRM") is State.CONFIRM

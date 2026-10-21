"""T02 验收测试：parse_turn_output 的解析行为（纯离线，不联网）。

覆盖任务书要求的五类情况：正常 JSON、前后带多余文字、带代码围栏、非法 JSON、缺字段。
"""

from __future__ import annotations

import pytest

from services.llm import parse_turn_output


def test_normal_json():
    raw = '{"reply": "请问您是哪位？", "slots": {"name": "张先生"}}'
    reply, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert reply == "请问您是哪位？"
    assert slots == {"name": "张先生"}


def test_json_without_slots_is_still_ok():
    reply, slots, ok = parse_turn_output('{"reply": "好的，我记下了。"}')
    assert (reply, slots, ok) == ("好的，我记下了。", {}, True)


def test_extra_text_around_json():
    raw = '好的，我这样回复：\n{"reply": "请问您有什么事？", "slots": {"purpose": "送快递"}}\n希望有帮助。'
    reply, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert reply == "请问您有什么事？"
    assert slots == {"purpose": "送快递"}


def test_braces_in_extra_text_before_real_json():
    # 前面的花括号不是合法 JSON 对象，必须能跳过并找到真正的那个
    raw = '注意 {这不是 JSON} 然后：{"reply": "请说下号码。", "slots": {}}'
    reply, slots, ok = parse_turn_output(raw)
    assert (reply, slots, ok) == ("请说下号码。", {}, True)


def test_code_fence():
    raw = '```json\n{"reply": "我会转告。", "slots": {"callback_number": "13800138000"}}\n```'
    reply, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert reply == "我会转告。"
    assert slots == {"callback_number": "13800138000"}


@pytest.mark.parametrize(
    "raw",
    [
        "抱歉，我没听清，请您再说一遍。",  # 完全没有 JSON
        '{"reply": "缺少右括号"',  # 非法 JSON
        "```json\n{不是合法 JSON}\n```",
        "",
        "   ",
        "[1, 2, 3]",  # 合法 JSON 但不是对象
    ],
)
def test_invalid_json_yields_failure(raw):
    reply, slots, ok = parse_turn_output(raw)
    assert ok is False
    assert reply == ""
    assert slots == {}


@pytest.mark.parametrize(
    "raw",
    [
        '{"slots": {"name": "张先生"}}',  # 缺 reply
        '{"reply": ""}',  # reply 为空
        '{"reply": "   "}',  # reply 只有空白
        '{"reply": 123}',  # reply 类型不对
        '{"reply": null}',
    ],
)
def test_missing_or_invalid_reply_fails(raw):
    reply, slots, ok = parse_turn_output(raw)
    assert (reply, slots, ok) == ("", {}, False)


def test_unknown_slot_keys_are_dropped():
    raw = '{"reply": "好", "slots": {"name": "李", "phone": "138", "email": "a@b.c"}}'
    reply, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert slots == {"name": "李"}  # 只保留 SLOT_NAMES 里的键


def test_non_string_slot_values_are_dropped():
    raw = '{"reply": "好", "slots": {"name": "李", "purpose": 3, "time": null, "location": ""}}'
    _, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert slots == {"name": "李"}


def test_values_are_stripped():
    _, slots, _ = parse_turn_output('{"reply": "好", "slots": {"name": "  张先生  "}}')
    assert slots == {"name": "张先生"}


def test_nested_objects_are_parsed_as_one():
    raw = '{"reply": "好", "slots": {"purpose": "送快递"}, "meta": {"nested": {"a": 1}}}'
    reply, slots, ok = parse_turn_output(raw)
    assert (reply, slots, ok) == ("好", {"purpose": "送快递"}, True)


def test_reply_is_stripped_but_keeps_inner_text():
    reply, _, _ = parse_turn_output('{"reply": "  请问您是哪位？  "}')
    assert reply == "请问您是哪位？"

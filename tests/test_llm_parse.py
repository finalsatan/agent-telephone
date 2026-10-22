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
    raw = '```json\n{"reply": "我会转告。", "slots": {"purpose": "送快递"}}\n```'
    reply, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert reply == "我会转告。"
    assert slots == {"purpose": "送快递"}


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


def test_callback_number_from_llm_is_discarded():
    """号码只认 ASR 原文 + 规则校验（设计文档第 4 节）；模型常把提示语当值回传。"""
    raw = '{"reply": "请说下号码。", "slots": {"name": "李", "callback_number": "请说一遍您的回拨号码"}}'
    reply, slots, ok = parse_turn_output(raw)
    assert ok is True
    assert slots == {"name": "李"}


# --------------------------------------------------------------------------- #
# T12b：slots_only 模式允许没有 reply
# --------------------------------------------------------------------------- #


def test_slots_only_without_reply_is_ok():
    raw = '{"slots": {"name": "李", "purpose": "送快递"}}'
    reply, slots, ok = parse_turn_output(raw, require_reply=False)
    assert ok is True
    assert reply == ""
    assert slots == {"name": "李", "purpose": "送快递"}


def test_slots_only_empty_slots_is_ok():
    reply, slots, ok = parse_turn_output('{"slots": {}}', require_reply=False)
    assert (reply, slots, ok) == ("", {}, True)


def test_slots_only_still_requires_json():
    reply, slots, ok = parse_turn_output("好的。", require_reply=False)
    assert (reply, slots, ok) == ("", {}, False)


def test_slots_only_tolerates_extra_reply_keys():
    """模型偶尔仍带 reply；slots_only 下不因 reply 格式报错，reply 原样返回。"""
    reply, slots, ok = parse_turn_output(
        '{"reply": "好的。", "slots": {"purpose": "送快递"}}', require_reply=False
    )
    assert (reply, slots, ok) == ("好的。", {"purpose": "送快递"}, True)


# --------------------------------------------------------------------------- #
# T12b：机主称呼不能写成 name / purpose（提示词病态输出的确定性防线）
# --------------------------------------------------------------------------- #


def test_owner_reference_is_dropped_from_name_and_purpose():
    raw = '{"reply": "您是哪位？", "slots": {"name": "张先生", "purpose": "找张先生"}}'
    reply, slots, ok = parse_turn_output(raw, owner_name="张先生")
    assert ok is True
    assert reply == "您是哪位？"
    assert slots == {}


def test_owner_reference_variants_are_dropped():
    for purpose in ("找张先生", "请找一下张先生", "麻烦找张先生", "想找张先生"):
        raw = '{"reply": "好", "slots": {"purpose": "%s"}}' % purpose
        _, slots, _ = parse_turn_output(raw, owner_name="张先生")
        assert slots == {}, purpose


def test_legitimate_purpose_mentioning_owner_is_kept():
    raw = '{"reply": "好", "slots": {"purpose": "想约张先生谈合同"}}'
    _, slots, _ = parse_turn_output(raw, owner_name="张先生")
    assert slots == {"purpose": "想约张先生谈合同"}


def test_owner_filter_is_off_without_owner_name():
    raw = '{"reply": "好", "slots": {"name": "张先生", "purpose": "找张先生"}}'
    _, slots, _ = parse_turn_output(raw)
    assert slots == {"name": "张先生", "purpose": "找张先生"}

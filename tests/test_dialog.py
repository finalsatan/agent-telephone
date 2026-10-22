"""T07 / T07b / T07c 的槽位与状态机测试。

覆盖《开发任务拆分》T07 验收清单：
正常流程走完、追问 2 次后跳过、中途结束意图、纠正流程、parse_ok=False、
非法号码被忽略、汉字数字号码提取、slots_update 里的 callback_number 被忽略、
不在追问号码时日期不写入、在追问号码时只报 8 位座机号能写入。

只测纯逻辑，不联网、不调用 LLM/TTS。
"""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from core.dialog import (
    ASK_WHICH_WRONG_ID,
    DEFAULT_MAX_ASKS,
    DEFAULT_MAX_CONFIRM_ROUNDS,
    END_INTENT_KEYWORDS,
    PHRASES,
    Dialog,
    ReplyHint,
    compose_confirm_text,
)
from core.slots import SlotStore
from core.types import REQUIRED_SLOTS, SLOT_NAMES, SlotValue, State

ROOT = Path(__file__).resolve().parents[1]


def make_dialog(**kwargs) -> Dialog:
    return Dialog(owner_name="张先生", **kwargs)


def _at_confirm(**kwargs) -> Dialog:
    """跑完一轮正常流程，停在 CONFIRM（姓名、来意、手机号都有值）。"""
    dialog = make_dialog(**kwargs)
    dialog.greeting()
    dialog.on_user_turn(*NAME_TURN)
    dialog.on_user_turn(*PURPOSE_TURN)
    state, _ = dialog.on_user_turn(*PHONE_TURN)
    assert state is State.CONFIRM
    return dialog


def _assert_hint(hint: ReplyHint, template_id: str | None = None, *, kind: str | None = None) -> None:
    """断言提示类型，以及模板 id。"""
    if kind is not None:
        assert hint.kind == kind
    if template_id is not None:
        assert hint.kind == "template"
        assert hint.template_id == template_id
        assert hint.text
    assert hint.kind in {"template", "text", "llm"}
    assert hint.use_cache is (hint.kind == "template")
    assert hint.use_text is (hint.kind == "text")
    if hint.kind == "text":
        # T07c：kind="text" 一定是整句现合成，不能同时是模板或 LLM
        assert hint.text
        assert hint.template_id is None


# 常见的一轮输入：姓名、来意、手机号（汉字数字）都齐。
NAME_TURN = ("我姓李，找张先生有点事", {"name": "李先生"})
PURPOSE_TURN = ("我是送快递的，有个包裹要送过来", {"purpose": "送快递"})
PHONE_TURN = ("我的电话是一三八零零一三八零零零", {})


class TestSlotStore:
    def test_initial_state(self):
        store = SlotStore()
        assert [store.get(name).status for name in SLOT_NAMES] == ["empty"] * len(SLOT_NAMES)
        assert store.missing_required() == list(REQUIRED_SLOTS)
        assert all(store.value(name) is None for name in SLOT_NAMES)
        assert store.as_dict().keys() == set(SLOT_NAMES)

    def test_update_marks_unconfirmed_then_corrected(self):
        store = SlotStore()
        assert list(store.update({"time": "今天下午三点"})) == ["time"]
        assert store.get("time") == SlotValue("今天下午三点", "unconfirmed")

        store.update({"time": "明天下午三点"})
        assert store.get("time") == SlotValue("明天下午三点", "corrected")

    def test_same_value_does_not_demote_confirmed(self):
        store = SlotStore()
        store.update({"name": "李先生"})
        store.confirm("name")
        assert store.update({"name": "李先生"}) == {}
        assert store.get("name").status == "confirmed"

    def test_callback_number_in_update_is_ignored(self):
        store = SlotStore()
        assert store.update({"callback_number": "13800138000"}) == {}
        assert store.value("callback_number") is None
        assert store.get("callback_number").status == "empty"

    def test_unknown_name_non_string_and_blank_are_ignored(self):
        store = SlotStore()
        assert store.update({"address": "朝阳路 1 号", "name": 123, "purpose": "   "}) == {}
        assert store.missing_required() == list(REQUIRED_SLOTS)

    def test_confirm_and_confirm_all(self):
        store = SlotStore()
        store.update({"name": "李先生", "purpose": "送快递"})
        assert store.confirm("name") is True
        assert store.confirm("name") is False  # 已确认，不重复计数
        assert store.confirm("time") is False  # 空槽位不能确认
        assert store.confirm_all() == ["purpose"]
        assert store.get("purpose").status == "confirmed"

    def test_abandon_marks_unconfirmed_and_skips(self):
        store = SlotStore()
        assert store.abandon("purpose") is True
        assert store.get("purpose") == SlotValue(None, "unconfirmed")
        assert store.is_abandoned("purpose")
        assert "purpose" not in store.missing_required()
        assert store.as_dict()["purpose"].status == "unconfirmed"

    def test_abandon_keeps_existing_value(self):
        store = SlotStore()
        store.update({"name": "李先生"})
        assert store.abandon("name") is False
        assert store.value("name") == "李先生"

    def test_filling_abandoned_slot_clears_abandon(self):
        store = SlotStore()
        store.abandon("name")
        store.update({"name": "李先生"})
        assert not store.is_abandoned("name")
        assert store.get("name") == SlotValue("李先生", "unconfirmed")

    def test_phone_from_arabic_text(self):
        store = SlotStore()
        assert store.update_phone_from_text("我的电话是13800138000") is True
        assert store.value("callback_number") == "13800138000"
        assert store.get("callback_number").status == "unconfirmed"

    def test_phone_from_chinese_digits(self):
        store = SlotStore()
        assert store.update_phone_from_text("我的电话是一三八零零一三八零零零") is True
        assert store.value("callback_number") == "13800138000"

    def test_phone_correction_marks_corrected(self):
        store = SlotStore()
        store.update_phone_from_text("电话是13800138000")
        store.update_phone_from_text("不对，是13900139000")
        assert store.get("callback_number") == SlotValue("13900139000", "corrected")

    def test_phone_not_found_returns_false(self):
        store = SlotStore()
        assert store.update_phone_from_text("我先问一下再打过来") is False
        assert store.value("callback_number") is None

    def test_phone_landline_needs_allow_landline(self):
        store = SlotStore()
        assert store.update_phone_from_text("八二三四五六七八") is False
        assert store.update_phone_from_text("八二三四五六七八", allow_landline=True) is True
        assert store.value("callback_number") == "82345678"

    def test_phone_with_context_keyword_is_accepted(self):
        store = SlotStore()
        assert store.update_phone_from_text("我家座机是八二三四五六七八") is True
        assert store.value("callback_number") == "82345678"

    def test_as_dict_is_a_copy(self):
        store = SlotStore()
        store.update({"name": "李先生"})
        snapshot = store.as_dict()
        snapshot["name"].value = "王先生"
        assert store.value("name") == "李先生"


class TestDialogFlow:
    def test_greeting_advances_state(self):
        dialog = make_dialog()
        hint = dialog.greeting()
        _assert_hint(hint, "greet")
        assert "张先生" in hint.text
        assert dialog.state is State.ASK_IDENTITY
        assert dialog.asked_slot == "name"

    def test_normal_flow_reaches_closing(self):
        dialog = make_dialog()
        dialog.greeting()

        state, hint = dialog.on_user_turn(*NAME_TURN)
        assert state is State.ASK_PURPOSE
        _assert_hint(hint, "ask_purpose")

        state, hint = dialog.on_user_turn(*PURPOSE_TURN)
        assert state is State.FILL_MISSING
        _assert_hint(hint, "ask_callback")
        assert dialog.asked_slot == "callback_number"

        state, hint = dialog.on_user_turn(*PHONE_TURN)
        assert state is State.CONFIRM
        # T07c：确认语由代码整句拼好现合成，号码已写成汉字读法
        _assert_hint(hint, kind="text")
        assert hint.phone == "13800138000"
        assert "幺三八，零零幺三，八零零零" in hint.text
        assert dialog.slot_store.value("callback_number") == "13800138000"

        state, hint = dialog.on_user_turn("对，没错")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")
        assert all(dialog.slot_store.get(n).status == "confirmed" for n in REQUIRED_SLOTS)

    def test_two_asks_then_skip_name(self):
        dialog = make_dialog()
        dialog.greeting()

        state, hint = dialog.on_user_turn("喂？", {})
        assert state is State.ASK_IDENTITY
        _assert_hint(hint, "ask_name")
        assert dialog.attempts["name"] == 1

        state, hint = dialog.on_user_turn("你说话呀", {})
        assert state is State.ASK_IDENTITY
        _assert_hint(hint, "ask_name")
        assert dialog.attempts["name"] == 2

        # 第三次仍没拿到 → 放弃 name，转去问来意
        state, hint = dialog.on_user_turn("……", {})
        assert state is State.ASK_PURPOSE
        _assert_hint(hint, "ask_purpose")
        assert dialog.slot_store.is_abandoned("name")
        assert dialog.slot_store.get("name") == SlotValue(None, "unconfirmed")
        assert "name" not in dialog.slot_store.missing_required()

    def test_callback_number_skipped_after_two_asks(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        _, hint = dialog.on_user_turn(*PURPOSE_TURN)
        _assert_hint(hint, "ask_callback")

        state, hint = dialog.on_user_turn("我不方便留电话", {})
        assert state is State.FILL_MISSING
        _assert_hint(hint, "ask_callback")
        assert dialog.attempts["callback_number"] == 2

        state, hint = dialog.on_user_turn("说了不方便", {})
        assert dialog.slot_store.is_abandoned("callback_number")
        # 只剩号码没拿到 → 全部有值或已放弃 → CONFIRM（确认语里没有号码那一段）
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert hint.phone is None
        assert "回拨电话" not in hint.text

    def test_end_intent_in_any_state(self):
        for keyword in ("没事了", "再见", "不用了，挂了"):
            dialog = make_dialog()
            dialog.greeting()
            state, hint = dialog.on_user_turn(keyword, {})
            assert state is State.CLOSING
            _assert_hint(hint, "goodbye")
        # 关键词表里的每一项都能触发，且表在模块顶部便于扩展
        assert "挂了" in END_INTENT_KEYWORDS

    def test_end_intent_during_confirm(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn(*PHONE_TURN)
        assert dialog.state is State.CONFIRM

        state, hint = dialog.on_user_turn("先这样吧", {})
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")
        # 没确认过的槽位不应被标成 confirmed
        assert dialog.slot_store.get("name").status == "unconfirmed"

    def test_correction_in_confirm_updates_status_and_reconfirms(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn("下午三点过来，电话是一三八零零一三八零零零", {"time": "今天下午三点"})
        assert dialog.state is State.CONFIRM
        assert dialog.slot_store.get("time").status == "unconfirmed"

        state, hint = dialog.on_user_turn("不是今天，是明天下午三点", {"time": "明天下午三点"})
        assert dialog.slot_store.get("time") == SlotValue("明天下午三点", "corrected")
        # 纠正后重新复述（设计文档第 4 节）：确认语由代码重新拼，带上纠正后的时间
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert "时间是明天下午三点" in hint.text

    def test_confirm_affirmative_after_correction(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn("下午三点过来，电话是一三八零零一三八零零零", {"time": "今天下午三点"})
        dialog.on_user_turn("不是今天，是明天下午三点", {"time": "明天下午三点"})

        state, hint = dialog.on_user_turn("对，这回对了")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")
        assert dialog.slot_store.get("time").status == "confirmed"

    def test_affirmative_with_correction_is_treated_as_correction(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn(*PHONE_TURN)
        assert dialog.state is State.CONFIRM

        # “对”里带了一个新值：不能把刚改过的姓名直接标成已确认
        state, hint = dialog.on_user_turn("对，我姓王", {"name": "王先生"})
        assert dialog.slot_store.get("name") == SlotValue("王先生", "corrected")
        assert dialog.slot_store.get("purpose").status == "unconfirmed"
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert "您是王先生" in hint.text

    def test_corrected_phone_in_confirm_is_not_auto_confirmed(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn(*PHONE_TURN)

        state, hint = dialog.on_user_turn("对的，不过电话是13900139000")
        assert dialog.slot_store.get("callback_number") == SlotValue("13900139000", "corrected")
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert "幺三九，零零幺三，九零零零" in hint.text

    def test_confirm_without_clear_answer_repeats(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn(*PHONE_TURN)

        # 既不含肯定也不含否定的回答 → 重新复述，不推进、不消耗追问次数
        state, hint = dialog.on_user_turn("我没听明白你刚才那句")
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert hint.phone == "13800138000"

    def test_parse_not_ok_keeps_state_and_repeats(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        before = dialog.state

        state, hint = dialog.on_user_turn("我是送快递的", {"purpose": "送快递"}, False)
        assert state is before
        _assert_hint(hint, "repeat")
        # 解析失败的一轮不写入槽位、不推进（追问次数保持上一轮的值）
        assert dialog.slot_store.value("purpose") is None
        assert dialog.attempts["purpose"] == 1

    def test_parse_not_ok_still_extracts_phone(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)

        state, hint = dialog.on_user_turn("一三八零零一三八零零零", {}, False)
        assert dialog.slot_store.value("callback_number") == "13800138000"
        assert state is State.FILL_MISSING
        _assert_hint(hint, "repeat")

    def test_invalid_phone_is_ignored(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        # 位数不够，不是号码
        dialog.on_user_turn("我的电话是一三八零零", {})
        assert dialog.slot_store.value("callback_number") is None

        # LLM 硬塞的 callback_number 一律忽略
        dialog.on_user_turn("嗯", {"callback_number": "123456"})
        assert dialog.slot_store.value("callback_number") is None

    def test_phone_from_chinese_digits_is_extracted(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn("幺三八零零一三八零零零", {})
        assert dialog.slot_store.value("callback_number") == "13800138000"

    def test_date_not_stored_when_not_asking_phone(self):
        dialog = make_dialog()
        dialog.greeting()
        # 此时在问身份，不是在问号码
        assert dialog.asked_slot == "name"
        dialog.on_user_turn("我预约在2026-10-21上午过去", {"purpose": "预约"})
        assert dialog.slot_store.value("callback_number") is None

    def test_landline_accepted_while_asking_phone(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        assert dialog.asked_slot == "callback_number"
        dialog.on_user_turn("八二三四五六七八", {})
        assert dialog.slot_store.value("callback_number") == "82345678"

    def test_phone_only_confirm_uses_code_composed_text(self):
        dialog = make_dialog()
        dialog.greeting()
        # 只有号码，姓名和来意都被放弃（各问 2 次仍然没有）
        dialog.on_user_turn(PHONE_TURN[0], {})
        for _ in range(4):
            state, hint = dialog.on_user_turn("嗯", {})
        assert state is State.CONFIRM
        # 只有号码时确认语里没有姓名/来意，号码仍是汉字读法
        _assert_hint(hint, kind="text")
        assert hint.phone == "13800138000"
        assert "您的回拨电话是" not in hint.text
        assert "回拨电话是幺三八，零零幺三，八零零零" in hint.text
        assert hint.text.startswith("跟您确认一下：")
        assert hint.text.endswith(PHRASES["confirm_question"])

    def test_all_slots_abandoned_closes_with_forward(self):
        dialog = make_dialog()
        dialog.greeting()
        # 每个必填槽位各问 2 次都不给（1+2+2+2 = 7 轮）
        for _ in range(7):
            state, hint = dialog.on_user_turn("我不想说", {})
        assert state is State.CLOSING
        _assert_hint(hint, "forward")
        assert dialog.slot_store.missing_required() == []

    def test_from_config_reads_owner_and_max_asks(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        dialog = Dialog.from_config(config)
        assert dialog.owner_name == config["owner"]["name"]
        assert dialog.max_asks == config["dialog"]["max_asks"] == DEFAULT_MAX_ASKS
        assert dialog.owner_name in dialog.greeting().text

    def test_max_asks_is_configurable(self):
        dialog = make_dialog(max_asks=1)
        dialog.greeting()
        state, hint = dialog.on_user_turn("喂？", {})
        _assert_hint(hint, "ask_name")
        state, hint = dialog.on_user_turn("喂？", {})
        assert dialog.slot_store.is_abandoned("name")
        assert state is State.ASK_PURPOSE

    def test_invalid_max_asks_rejected(self):
        with pytest.raises(ValueError):
            Dialog(max_asks=0)

    def test_attempts_and_slots_exposed_for_llm(self):
        dialog = make_dialog()
        dialog.greeting()
        assert dialog.attempts == {}  # 问候里的身份询问不计入追问次数

        dialog.on_user_turn(*NAME_TURN)
        # 姓名一次就拿到，下一句在问来意（已计 1 次）
        assert dialog.attempts == {"purpose": 1}
        assert set(dialog.slots) == set(SLOT_NAMES)
        assert isinstance(dialog.slots["name"], SlotValue)

        dialog.on_user_turn("嗯", {})
        assert dialog.attempts["purpose"] == 2

    def test_optional_slots_are_recorded_but_not_asked(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn("我明天下午三点来，在朝阳路 1 号", {
            "name": "李先生",
            "purpose": "送快递",
            "time": "明天下午三点",
            "location": "朝阳路 1 号",
        })
        assert dialog.slot_store.value("time") == "明天下午三点"
        assert dialog.slot_store.value("location") == "朝阳路 1 号"
        # 选填槽位不会被追问：下一句仍是问回拨电话
        assert dialog.state is State.FILL_MISSING
        assert dialog.asked_slot == "callback_number"

    def test_turn_before_greeting_is_treated_as_ask_identity(self):
        dialog = make_dialog()
        assert dialog.state is State.GREETING
        # 问候还没播对方就开口了：直接按身份阶段处理
        state, hint = dialog.on_user_turn(*NAME_TURN)
        assert state is State.ASK_PURPOSE
        _assert_hint(hint, "ask_purpose")

    def test_turn_after_closing_stays_closed(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn("没事了", {})
        assert dialog.state is State.CLOSING
        state, hint = dialog.on_user_turn("喂？还在吗？", {"name": "李先生"})
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")

    def test_greeting_text_has_owner_and_contains_no_hardcoded_name(self):
        dialog = Dialog(owner_name="王女士")
        assert "王女士" in dialog.greeting().text
        for text in PHRASES.values():
            assert "张先生" not in text

    def test_from_config_reads_max_confirm_rounds(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        dialog = Dialog.from_config(config)
        assert dialog.max_confirm_rounds == config["dialog"]["max_confirm_rounds"]
        assert dialog.max_confirm_rounds == DEFAULT_MAX_CONFIRM_ROUNDS



class TestConfirmIntent:
    """背景表中的每一行（T07b 要求 1–5）。"""

    def test_sorry_i_did_not_hear_does_not_confirm(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("对不起，我没听清")
        assert state is State.CONFIRM
        # 不算肯定 → 重新播确认语（T07c：代码拼好的整句，号码是汉字读法）
        _assert_hint(hint, kind="text")
        assert hint.phone == "13800138000"
        assert all(dialog.slot_store.get(n).status != "confirmed" for n in REQUIRED_SLOTS)

    def test_missing_a_digit_is_negative(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("好像号码少了一位")
        assert state is State.CONFIRM
        # 否定且没给新值 → 播预合成模板“请问是哪一项不对？”，不接号码复述
        _assert_hint(hint, ASK_WHICH_WRONG_ID)
        assert hint.phone is None
        assert dialog.confirm_rounds == 1

    def test_question_is_not_affirmative(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("你刚才说的是什么？")
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert hint.phone == "13800138000"

    def test_no_problem_is_affirmative(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("对，没有问题")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")
        assert all(dialog.slot_store.get(n).status == "confirmed" for n in REQUIRED_SLOTS)

    def test_sorry_it_is_right_is_affirmative(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("不好意思，是对的")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")

    def test_end_intent_with_affirmative_confirms_then_closes(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("对，就这样吧")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")
        # 先 confirm_all 再挂断
        assert all(dialog.slot_store.get(n).status == "confirmed" for n in REQUIRED_SLOTS)

    def test_end_intent_with_affirmative_and_goodbye(self):
        dialog = _at_confirm()
        state, _ = dialog.on_user_turn("好的，没事了，拜拜")
        assert state is State.CLOSING
        assert all(dialog.slot_store.get(n).status == "confirmed" for n in REQUIRED_SLOTS)

    def test_end_intent_without_affirmative_keeps_slots_unconfirmed(self):
        dialog = _at_confirm()
        state, _ = dialog.on_user_turn("先这样吧")
        assert state is State.CLOSING
        assert dialog.slot_store.get("name").status == "unconfirmed"

    def test_repeated_negative_has_an_exit(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("不对")
        assert state is State.CONFIRM
        assert hint.phone is None  # 追问哪一项不对
        state, hint = dialog.on_user_turn("不对")
        assert state is State.CONFIRM
        assert dialog.confirm_rounds == 2
        # 第 3 次超过 max_confirm_rounds=2 → 收尾，槽位保持未确认
        state, hint = dialog.on_user_turn("不对")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")
        assert all(dialog.slot_store.get(n).status == "unconfirmed" for n in REQUIRED_SLOTS)

    def test_unclear_rounds_also_count_toward_the_limit(self):
        dialog = _at_confirm()
        dialog.on_user_turn("你刚才说什么？")
        dialog.on_user_turn("我没听清")
        state, hint = dialog.on_user_turn("嗯")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")

    def test_invalid_max_confirm_rounds_rejected(self):
        with pytest.raises(ValueError):
            Dialog(max_confirm_rounds=-1)

    def test_max_confirm_rounds_is_configurable(self):
        dialog = _at_confirm(max_confirm_rounds=0)
        state, hint = dialog.on_user_turn("不对")
        assert state is State.CLOSING
        _assert_hint(hint, "goodbye")

    def test_correction_resets_confirm_rounds(self):
        dialog = _at_confirm()
        dialog.on_user_turn("不对")
        assert dialog.confirm_rounds == 1
        state, hint = dialog.on_user_turn("对，不过电话是13900139000")
        assert dialog.confirm_rounds == 0
        assert state is State.CONFIRM
        assert dialog.slot_store.get("callback_number") == SlotValue("13900139000", "corrected")

    def test_landline_correction_in_confirm_is_accepted(self):
        dialog = _at_confirm()
        state, hint = dialog.on_user_turn("不对，是八二三四五六七九")
        # CONFIRM 里允许座机（T07b 要求 4）：号码更新，重新复述
        assert dialog.slot_store.get("callback_number") == SlotValue("82345679", "corrected")
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert hint.phone == "82345679"
        assert "回拨电话是八二三四，五六七九" in hint.text

    def test_confirm_hint_carries_phone_for_tts_fallback(self):
        dialog = _at_confirm()
        _, hint = dialog.on_user_turn("你说什么？")
        _assert_hint(hint, kind="text")
        # phone 填原值（数字串），TTS 现合成失败时 T10 用它退回 render_phone
        assert hint.phone == "13800138000"

    def test_confirm_hint_without_phone(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn(*PURPOSE_TURN)
        dialog.on_user_turn("我不方便留电话", {})
        state, hint = dialog.on_user_turn("说了不方便", {})
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert hint.phone is None


class TestEndIntent:
    """结束意图（T07b 要求 2）。"""

    @pytest.mark.parametrize(
        "text",
        [
            "也没什么事，就是想问下快递放哪",
            "我就这样跟你说吧，合同的事很急",
            "不麻烦的话让他回个电话",
        ],
    )
    def test_early_call_continuation_is_not_end_intent(self, text):
        dialog = make_dialog()
        dialog.greeting()
        state, hint = dialog.on_user_turn(text, {})
        assert state is not State.CLOSING
        assert dialog.state is not State.CLOSING

    def test_end_intent_with_new_slot_value_continues(self):
        dialog = make_dialog()
        dialog.greeting()
        state, _ = dialog.on_user_turn("没事了，不过我姓李", {"name": "李先生"})
        assert state is not State.CLOSING

    def test_short_end_intent_still_closes(self):
        for text in ("没事了", "再见", "不用了，挂了", "先这样吧", "回头再说", "没事了，谢谢"):
            dialog = make_dialog()
            dialog.greeting()
            state, hint = dialog.on_user_turn(text, {})
            assert state is State.CLOSING
            _assert_hint(hint, "goodbye")


class TestParseFailures:
    """连续解析失败超过 2 次改播模板（T07b 要求 6）。"""

    def test_repeat_twice_then_rebroadcast_ask_template(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)  # 现在在问来意
        assert dialog.asked_slot == "purpose"

        _, hint = dialog.on_user_turn("我找张先生", {}, False)
        _assert_hint(hint, "repeat")
        _, hint = dialog.on_user_turn("我找张先生", {}, False)
        _assert_hint(hint, "repeat")
        _, hint = dialog.on_user_turn("我找张先生", {}, False)
        _assert_hint(hint, "ask_purpose")

    def test_rebroadcast_in_confirm_recites_again(self):
        dialog = _at_confirm()
        for _ in range(2):
            _, hint = dialog.on_user_turn("嗯", {}, False)
            _assert_hint(hint, "repeat")
        state, hint = dialog.on_user_turn("嗯", {}, False)
        assert state is State.CONFIRM
        _assert_hint(hint, kind="text")
        assert hint.phone == "13800138000"

    def test_failure_before_any_question_repeats(self):
        # 还没播问候（_asked_slot 为空）时解析失败，没有可重播的追问模板 → repeat
        dialog = make_dialog()
        for _ in range(3):
            _, hint = dialog.on_user_turn("喂", {}, False)
        _assert_hint(hint, "repeat")

    def test_failure_counter_resets_on_parse_ok(self):
        dialog = make_dialog()
        dialog.greeting()
        dialog.on_user_turn(*NAME_TURN)
        dialog.on_user_turn("我找张先生", {}, False)
        dialog.on_user_turn("我找张先生", {}, False)
        _, hint = dialog.on_user_turn("我找张先生", {"purpose": "找张先生"}, True)
        # 解析恢复后正常推进；下一次失败又从 repeat 开始
        assert hint.template_id != "repeat"
        _, hint = dialog.on_user_turn("嗯", {}, False)
        _assert_hint(hint, "repeat")


class TestComposeConfirmText:
    """T07c：确认语由代码拼，号码写成汉字读法（1 读“幺”）。"""

    def _slots(self, **values):
        return {name: SlotValue(values.get(name), "unconfirmed" if values.get(name) else "empty") for name in SLOT_NAMES}

    def test_name_purpose_and_phone(self):
        slots = self._slots(name="李明", purpose="想约张先生谈合同", callback_number="13800138000")
        text = compose_confirm_text(slots, owner_name="张先生")
        assert text == (
            "跟您确认一下：您是李明，想约张先生谈合同，"
            "回拨电话是幺三八，零零幺三，八零零零，对吗？"
        )

    def test_phone_only(self):
        slots = self._slots(callback_number="13800138000")
        text = compose_confirm_text(slots, owner_name="张先生")
        assert text == "跟您确认一下：想找张先生，回拨电话是幺三八，零零幺三，八零零零，对吗？"

    def test_abandoned_name_is_skipped(self):
        slots = self._slots(purpose="想约张先生谈合同", callback_number="13800138000")
        slots["name"] = SlotValue(None, "unconfirmed")  # 追问用尽、已放弃
        text = compose_confirm_text(slots, owner_name="张先生")
        assert "您是" not in text
        assert text.startswith("跟您确认一下：想约张先生谈合同，")
        assert "回拨电话是幺三八，零零幺三，八零零零，对吗？" in text

    def test_time_and_location(self):
        slots = self._slots(
            name="李明",
            purpose="关于送快递的事",
            time="明天下午三点",
            location="朝阳路一号",
            callback_number="01012345678",
        )
        text = compose_confirm_text(slots, owner_name="张先生")
        assert "您是李明，关于送快递的事" in text
        assert "时间是明天下午三点" in text
        assert "地点是朝阳路一号" in text
        assert "回拨电话是零幺零，幺二三四五六七八" in text
        assert text.endswith("对吗？")

    def test_owner_name_from_parameter(self):
        slots = self._slots(name="李明")
        assert "想找王女士" in compose_confirm_text(slots, owner_name="王女士")
        assert "想找张先生" in compose_confirm_text(slots, owner_name="张先生")

    def test_blank_values_are_skipped(self):
        slots = self._slots(name="  ", purpose="想咨询一下", callback_number="82345678")
        text = compose_confirm_text(slots, owner_name="张先生")
        assert "您是" not in text
        assert "想咨询一下" in text
        assert "回拨电话是八二三四，五六七八" in text


def test_reply_hint_kind_helpers():
    # kind="llm" 目前不由 Dialog 产生（T07c 之后都是 template/text），
    # 但 T10 的分流逻辑仍按这三种 kind 写，属性要能正确区分。
    assert ReplyHint(kind="llm").use_llm is True
    assert ReplyHint(kind="template", template_id="repeat").use_cache is True
    text_hint = ReplyHint(kind="text", text="跟您确认一下：…", phone="13800138000")
    assert text_hint.use_text is True
    assert text_hint.use_llm is False and text_hint.use_cache is False

"""T03 验收：号码规范化与方言数字（`core/normalize.py`）。

ASR 实测把手机号读成汉字数字，这里覆盖任务书要求的用例与常见边界。
"""

from __future__ import annotations

import pytest

from core.normalize import (
    cn_digits_to_arabic,
    extract_phone,
    normalize_time_expr,
    phone_groups,
    phone_to_spoken,
)


class TestCnDigitsToArabic:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("一三八零零一三八零零零", "13800138000"),
            ("幺三八零零一三八零零零", "13800138000"),
            ("幺三八零零幺三八零零零", "13800138000"),
            ("一两八零零一三八零零零", "12800138000"),  # “两”在数字串里读作 2
            ("138零零一三八零零零", "13800138000"),  # 汉字与阿拉伯数字混读
            ("二零二六年", "2026年"),
            ("电话一三八零零一三八零零零，谢谢", "电话13800138000，谢谢"),
        ],
    )
    def test_converts_digit_runs(self, raw: str, expected: str) -> None:
        assert cn_digits_to_arabic(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "我一个人去就行",  # 单个汉字数字不算数字串
            "两点再说",  # “两”不在数字串上下文里，不转
            "一百二十三个人",  # 数量级表达不该逐位转换
            "三点一四",  # 小数
            "hello world",
            "",
        ],
    )
    def test_leaves_non_digit_context_untouched(self, raw: str) -> None:
        assert cn_digits_to_arabic(raw) == raw

    def test_is_idempotent_for_arabic_digits(self) -> None:
        assert cn_digits_to_arabic("13800138000") == "13800138000"


class TestExtractPhone:
    def test_cn_digits_mobile(self) -> None:
        assert extract_phone("我的电话是一三八零零一三八零零零") == "13800138000"

    def test_yao_digits_mobile(self) -> None:
        assert extract_phone("幺三八零零一三八零零零") == "13800138000"

    def test_arabic_mobile_passthrough(self) -> None:
        assert extract_phone("13800138000") == "13800138000"

    def test_fractional_length_mobile_is_not_a_phone(self) -> None:
        """不足 11 位返回 None，也不得被截成 8 位座机号。"""
        assert extract_phone("我的电话是一三八零零一三八零零") is None
        assert extract_phone("1380013800") is None

    def test_longer_digit_run_is_not_truncated_to_mobile(self) -> None:
        assert extract_phone("138001380001") is None

    def test_year_is_not_a_phone(self) -> None:
        assert extract_phone("二零二六年") is None
        assert extract_phone("2026 年 5 月") is None

    @pytest.mark.parametrize(
        "raw",
        [
            "我的电话是 138 0013 8000",
            "电话：138-0013-8000",
            "１３８００１３８０００",  # 全角数字
            "138、0013、8000",
        ],
    )
    def test_separators_and_fullwidth_are_normalized(self, raw: str) -> None:
        assert extract_phone(raw) == "13800138000"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("座机是87654321", "87654321"),
            ("分机电话1234567", "1234567"),
            ("联系电话8765-4321", "87654321"),
            ("打我座机8765432", "8765432"),
        ],
    )
    def test_landline_seven_or_eight_digits_with_context(self, raw: str, expected: str) -> None:
        assert extract_phone(raw) == expected

    @pytest.mark.parametrize("raw", ["87654321", "8765-4321", "分机1234567", "房间号8021234"])
    def test_landline_without_context_is_rejected(self, raw: str) -> None:
        """T03b：没有上下文、也没在追问号码时，7–8 位数字不能被当成座机号。"""
        assert extract_phone(raw) is None
        assert extract_phone(raw, allow_landline=True) is not None

    def test_returns_none_without_digits(self) -> None:
        assert extract_phone("喂，你好，我找张先生") is None
        assert extract_phone("") is None

    def test_mobile_wins_over_landline(self) -> None:
        assert extract_phone("13800138000 转 12345678") == "13800138000"

    def test_phone_in_long_sentence(self) -> None:
        text = "喂，你好，我找一下张先生，我是送快递的，我的电话是一三八零零一三八零零零。"
        assert extract_phone(text) == "13800138000"


class TestLandlineContext:
    """T03b：座机号必须有上下文（或 allow_landline），否则日期/订单号/房间号/金额会被误收。"""

    @pytest.mark.parametrize(
        "raw",
        [
            "预约在2026-10-21上午",
            "订单号是12345678",
            "明天下午三点，房间号八零二一二三四",
            "一共一二三四五六七块",
            "房间号一二三四五六",
        ],
    )
    def test_no_landline_without_context(self, raw: str) -> None:
        assert extract_phone(raw) is None

    def test_keyword_context_accepts_landline(self) -> None:
        assert extract_phone("我家座机是八二三四五六七八") == "82345678"

    def test_allow_landline_accepts_bare_digits(self) -> None:
        # 正在追问回拨号码时，对方只报 8 位数字也算有效
        assert extract_phone("八二三四五六七八", allow_landline=True) == "82345678"
        assert extract_phone("82345678", allow_landline=True) == "82345678"

    def test_area_code_landline(self) -> None:
        assert extract_phone("座机零一零一二三四五六七八") == "01012345678"
        assert extract_phone("电话010-12345678") == "01012345678"

    def test_area_code_landline_with_allow_landline(self) -> None:
        assert extract_phone("零一零一二三四五六七八", allow_landline=True) == "01012345678"

    def test_area_code_needs_context_too(self) -> None:
        assert extract_phone("单号01012345678") is None

    def test_keyword_must_be_within_window(self) -> None:
        # 关键词在窗口内
        assert extract_phone("我的电话是82345678") == "82345678"
        # 关键词离号码超过 6 个字：不认（否则整段话里任何数字都能被“电话”带进来）
        assert extract_phone("电话我已经说过了，那就是八二三四五六七八") is None

    def test_mobile_still_works_without_context(self) -> None:
        assert extract_phone("订单号是12345678，手机13800138000") == "13800138000"

    def test_allow_landline_does_not_change_mobile_priority(self) -> None:
        assert extract_phone("13800138000 87654321", allow_landline=True) == "13800138000"


def test_normalize_time_expr_is_a_passthrough_placeholder() -> None:
    assert normalize_time_expr("明天下午三点") == "明天下午三点"
    assert normalize_time_expr("") == ""


class TestCnDigitsMoreEdges:
    def test_digit_run_followed_by_magnitude_is_not_converted(self) -> None:
        # "三四百万" 是模糊数量级表达，逐位转成 "34百万" 是错的
        assert cn_digits_to_arabic("花了三四百万") == "花了三四百万"
        assert cn_digits_to_arabic("一二十个") == "一二十个"

    def test_digit_run_after_decimal_point_is_not_converted(self) -> None:
        assert cn_digits_to_arabic("三点一四") == "三点一四"
        assert cn_digits_to_arabic("3.一四") == "3.一四"


class TestPhoneGroups:
    """T07c：号码分组（从 core/phrases.py 迁来，只留一份）。"""

    def test_mobile_is_3_4_4(self) -> None:
        assert phone_groups("13800138000") == ["138", "0013", "8000"]

    def test_area_code_landline(self) -> None:
        assert phone_groups("01012345678") == ["010", "12345678"]
        assert phone_groups("021-87654321") == ["021", "87654321"]
        assert phone_groups("0755-82345678") == ["0755", "82345678"]

    def test_plain_landline_is_grouped_by_four(self) -> None:
        assert phone_groups("82345678") == ["8234", "5678"]
        assert phone_groups("1234567") == ["1234", "567"]

    def test_separators_are_ignored(self) -> None:
        assert phone_groups("138 0013 8000") == ["138", "0013", "8000"]
        assert phone_groups("(010) 1234-5678") == ["010", "12345678"]

    def test_no_digits_raises(self) -> None:
        with pytest.raises(ValueError):
            phone_groups("无")


class TestPhoneToSpoken:
    """T07c：号码写成 TTS 可直接念的汉字，1 读“幺”，组间逗号。"""

    def test_mobile(self) -> None:
        assert phone_to_spoken("13800138000") == "幺三八，零零幺三，八零零零"

    def test_area_code_landline(self) -> None:
        assert phone_to_spoken("01012345678") == "零幺零，幺二三四五六七八"

    def test_eight_digit_landline(self) -> None:
        assert phone_to_spoken("82345678") == "八二三四，五六七八"

    def test_one_reads_as_yao(self) -> None:
        assert phone_to_spoken("1") == "幺"
        assert phone_to_spoken("114") == "幺幺四"

    def test_all_digits_covered(self) -> None:
        # 2345 6789 覆盖 2–9；0 与 1 由 test_area_code_landline / test_one_reads_as_yao 覆盖
        assert phone_to_spoken("23456789") == "二三四五，六七八九"
        assert phone_to_spoken("01012345678") == "零幺零，幺二三四五六七八"

    def test_no_digits_raises(self) -> None:
        with pytest.raises(ValueError):
            phone_to_spoken("")

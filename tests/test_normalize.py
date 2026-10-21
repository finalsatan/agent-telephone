"""T03 验收：号码规范化与方言数字（`core/normalize.py`）。

ASR 实测把手机号读成汉字数字，这里覆盖任务书要求的用例与常见边界。
"""

from __future__ import annotations

import pytest

from core.normalize import cn_digits_to_arabic, extract_phone, normalize_time_expr


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
            ("分机1234567", "1234567"),
            ("8765-4321", "87654321"),
            ("8765432", "8765432"),
        ],
    )
    def test_landline_seven_or_eight_digits(self, raw: str, expected: str) -> None:
        assert extract_phone(raw) == expected

    def test_returns_none_without_digits(self) -> None:
        assert extract_phone("喂，你好，我找张先生") is None
        assert extract_phone("") is None

    def test_mobile_wins_over_landline(self) -> None:
        assert extract_phone("13800138000 转 12345678") == "13800138000"

    def test_phone_in_long_sentence(self) -> None:
        text = "喂，你好，我找一下张先生，我是送快递的，我的电话是一三八零零一三八零零零。"
        assert extract_phone(text) == "13800138000"


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

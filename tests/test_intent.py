"""`core/intent.py` 的意图判定测试（T07c 拆模块后单独测）。

覆盖第 9 次评审背景表里的判定例子，以及 T07c 新增的“单独的嗯/嗯嗯判为肯定”。
"""

from __future__ import annotations

import pytest

from core.intent import (
    AFFIRMATIVE,
    AFFIRMATIVE_KEYWORDS,
    NEGATIVE,
    UNCLEAR,
    classify_confirm_reply,
    strip_noise,
)


@pytest.mark.parametrize(
    "text",
    [
        "对",
        "对的",
        "是的是的",
        "没错",
        "嗯，对",
        "好的谢谢",
        "没问题",
        "可以可以",
        "对，没有问题",
        "不好意思，是对的",
    ],
)
def test_affirmative(text):
    assert classify_confirm_reply(text) == AFFIRMATIVE


@pytest.mark.parametrize("text", ["嗯", "嗯嗯", "嗯 嗯", "嗯嗯嗯"])
def test_bare_hum_is_affirmative(text):
    # T07c：电话里“嗯”就是“对”
    assert classify_confirm_reply(text) == AFFIRMATIVE


@pytest.mark.parametrize("text", ["不对", "不是", "错了", "号码不对", "好像号码少了一位"])
def test_negative(text):
    assert classify_confirm_reply(text) == NEGATIVE


@pytest.mark.parametrize(
    "text",
    [
        "对不起，我没听清",  # “对不起”里的“对”不算肯定
        "你刚才说的是什么？",  # 疑问句不算肯定
        "对了吗？",
        "是不是这样",
        "不好意思，麻烦再说一遍",  # 客套话被剔除后没有肯定来源
        "谢谢",  # 客套话被剔除后为空
        "，，。",  # 只有标点
        "",
        "   ",
    ],
)
def test_unclear(text):
    assert classify_confirm_reply(text) == UNCLEAR


def test_negation_wins_over_affirmative_substring():
    assert classify_confirm_reply("不对") == NEGATIVE
    assert classify_confirm_reply("不是，你听错了") == NEGATIVE


def test_meiyou_cuo_is_not_negative():
    # “没错”“不错”里的“错”不算否定
    assert classify_confirm_reply("没错") == AFFIRMATIVE
    assert classify_confirm_reply("不错，对的") == AFFIRMATIVE


def test_hum_with_question_is_unclear():
    # “嗯”本身是肯定，但带疑问就不是
    assert classify_confirm_reply("嗯？你刚才说什么？") == UNCLEAR


def test_strip_noise_removes_punctuation_and_fillers():
    assert strip_noise("对的，好的吧") == "对的好的"
    assert strip_noise("？？") == ""


def test_module_only_depends_on_stdlib():
    # 拆分的目的之一：任何模块都能安全导入 intent（不能反过来依赖 core.*）
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/intent.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("core"), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("core"), alias.name


def test_affirmative_keywords_do_not_include_negations():
    assert "不对" not in AFFIRMATIVE_KEYWORDS
    assert "对" in AFFIRMATIVE_KEYWORDS

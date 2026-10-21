"""ASR 文本的规则后处理（T03）。

ASR（`Qwen/Qwen3-ASR-1.7B`）实测会把手机号读成汉字数字（"一三八零零一三八零零零"），
而号码槽位必须以规则校验为准、不经 LLM 听写（设计文档第 4 节）。本模块负责：

- `cn_digits_to_arabic()`：把连续汉字数字串转成阿拉伯数字；
- `extract_phone()`：从文本里取 11 位手机号或 7–8 位座机号；
- `normalize_time_expr()`：时间表达的占位实现（T07/T08 再解析）。

这里只做确定性的字符串处理，不依赖网络、不依赖模型的 tokenizer。
"""

from __future__ import annotations

import re
import unicodedata

# 汉字数字字符 → 阿拉伯数字。幺/两 是口语读法，按“数字串上下文”处理（见下）。
CN_DIGITS: dict[str, str] = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "幺": "1",
    "两": "2",
}

# 数字串 = 汉字数字或阿拉伯数字的连续片段（半角/全角阿拉伯数字由 NFKC 统一）。
_DIGIT_CHARS = "".join(CN_DIGITS) + "0123456789"
_RUN_RE = re.compile(f"[{_DIGIT_CHARS}]+")

# 数字串紧邻这些字符时不当成“逐位读的数字串”：
# - 十百千万亿：数量级表达（"一百二十三"），转成 "1百23" 反而是错的；
# - 点/小数点在数字串之前：小数（"三点一四"）不该逐位转换。
_MAGNITUDE_CHARS = "十百千万亿"
_DECIMAL_BEFORE = "点.．"

# 数字之间的分隔符（口语里的停顿、书写里的横线/空格），提取号码前先合并。
_SEPARATOR_RE = re.compile(r"(?<=\d)[\s\-－—–~～、,，.．·()（）+]+(?=\d)")

# 11 位手机号：中国大陆号段以 1 开头、第二位 3–9。
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 7–8 位座机号（不含区号）。前后不得紧邻其它数字，避免把 10 位以上号码截断成"座机号"。
_LANDLINE_RE = re.compile(r"(?<!\d)\d{7,8}(?!\d)")


def cn_digits_to_arabic(text: str) -> str:
    """把连续汉字数字转成阿拉伯数字，其余字符原样保留。

    "一三八零零一三八零零零" → "13800138000"；"幺三八零零一三八零零零" → "13800138000"。
    单个汉字数字（"我一个人""两点"）不转，避免把普通语气词改坏；
    数量级表达（"一百二十三"）与小数（"三点一四"）也不转，见 `_MAGNITUDE_CHARS`。
    """
    if not text:
        return text

    def _convert(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) < 2:
            return run  # 单个字符不算数字串
        start, end = match.span()
        if start > 0 and (text[start - 1] in _MAGNITUDE_CHARS or text[start - 1] in _DECIMAL_BEFORE):
            return run
        if end < len(text) and text[end] in _MAGNITUDE_CHARS:
            return run
        return "".join(CN_DIGITS.get(ch, ch) for ch in run)

    return _RUN_RE.sub(_convert, text)


def extract_phone(text: str) -> str | None:
    """从文本中提取电话号码：优先 11 位手机号，其次 7–8 位座机号。

    步骤：NFKC（全角数字/空格） → 汉字数字转阿拉伯数字 → 合并数字间的分隔符 → 正则匹配。
    取不到返回 None。长度不足 11 位的数字串不会被当成手机号，也不会被截成座机号。
    """
    if not text:
        return None
    normalized = cn_digits_to_arabic(unicodedata.normalize("NFKC", text))
    compact = _SEPARATOR_RE.sub("", normalized)

    mobile = _MOBILE_RE.search(compact)
    if mobile:
        return mobile.group(0)
    landline = _LANDLINE_RE.search(compact)
    if landline:
        return landline.group(0)
    return None


def normalize_time_expr(text: str) -> str:
    """时间表达的占位实现：原样返回（T03 不解析时间）。

    保留函数名是为了让 T07/T08 的调用方现在就能接上，届时在这里把
    "明天下午三点" 之类解析成结构化时间，不需要改调用点。
    """
    return text

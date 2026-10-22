"""ASR 文本的规则后处理（T03）。

ASR（`Qwen/Qwen3-ASR-1.7B`）实测会把手机号读成汉字数字（"一三八零零一三八零零零"），
而号码槽位必须以规则校验为准、不经 LLM 听写（设计文档第 4 节）。本模块负责：

- `cn_digits_to_arabic()`：把连续汉字数字串转成阿拉伯数字；
- `extract_phone()`：从文本里取 11 位手机号或 7–8 位座机号；
- `extract_phones()`：取文本里的全部号码（T11b 的号码复核用，规则同上）；
- `phone_groups()` / `phone_to_spoken()`：把号码分组并写成可交给 TTS 的汉字读法（T07c）；
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
# 座机号：带区号（0 + 2–3 位区号 + 7–8 位号码，共 10–11 位）优先，其次是不带区号的 7–8 位。
# 前后都不得紧邻其它数字，避免把更长的数字串截断成"座机号"。
_AREA_CODE_LANDLINE_RE = re.compile(r"(?<!\d)0\d{2,3}\d{7,8}(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)\d{7,8}(?!\d)")

# 座机号的上下文关键词（T03b）：7–8 位数字太容易是日期、订单号、房间号、金额，
# 只有在关键词附近，或调用方明确说"我正在追问回拨号码"时才接受。
LANDLINE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "电话",
    "号码",
    "座机",
    "手机",
    "联系",
    "回拨",
    "打",
    "拨",
)
# 关键词必须出现在号码之前这么多字以内（含标点）。
LANDLINE_CONTEXT_WINDOW = 6


def cn_digits_to_arabic(text: str) -> str:
    """把连续汉字数字转成阿拉伯数字，其余字符原样保留。

    "一三八零零一三八零零零" → "13800138000"；"幺三八零零一三八零零零" → "13800138000"。
    单个汉字数字（"我一个人""两点"）不转，避免把普通语气词改坏；
    数量级表达（"一百二十三"）与小数（"三点一四"）也不转，见 `_MAGNITUDE_CHARS`。

    **仅供号码提取（`extract_phone`）使用，不得用来改写要显示或摘要的转写文本**：
    它只做逐位替换，不理解语义，"两三个人"会被转成 "23个人"、"二零二六年" 会变成
    "2026年" 但 "两千" 保持原样，结果并不适合直接展示给对方或写进摘要。
    事件流里的 `asr_final.text` 一律保留 ASR 原文。
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


def extract_phone(text: str, *, allow_landline: bool = False) -> str | None:
    """从文本中提取电话号码：优先 11 位手机号，其次座机号（需满足上下文条件）。

    步骤：NFKC（全角数字/空格） → 汉字数字转阿拉伯数字 → 合并数字间的分隔符 → 正则匹配。
    取不到返回 None。长度不足 11 位的数字串不会被当成手机号，也不会被截成座机号。

    手机号（`1[3-9]` 开头、11 位）始终接受；座机号（7–8 位，或 `0` + 2–3 位区号 + 7–8 位）
    只在 `allow_landline=True`（调用方正在追问回拨号码）或号码前 6 个字以内出现
    `LANDLINE_CONTEXT_KEYWORDS` 里的关键词时才接受（T03b）。否则 "订单号是12345678"、
    "预约在2026-10-21" 这类文本会把无关数字写进回拨号码槽位。
    """
    if not text:
        return None
    normalized = cn_digits_to_arabic(unicodedata.normalize("NFKC", text))
    # 两种候选文本：合并了数字间分隔符的，和原样的。
    # 先试合并后的（"138 0013 8000" 这种分组读法），再退回原样文本，
    # 否则"13800138000 87654321"这种相邻号码会被空格合并成一个 19 位串而丢掉手机号。
    candidates = (_SEPARATOR_RE.sub("", normalized), normalized)

    for candidate in candidates:
        mobile = _MOBILE_RE.search(candidate)
        if mobile:
            return mobile.group(0)

    for candidate in candidates:
        for pattern in (_AREA_CODE_LANDLINE_RE, _LANDLINE_RE):
            match = pattern.search(candidate)
            if match is None:
                continue
            if allow_landline or _has_landline_context(candidate, match.start()):
                return match.group(0)
    return None


def _has_landline_context(candidate: str, start: int) -> bool:
    """号码前面的 `LANDLINE_CONTEXT_WINDOW` 个字里是否出现上下文关键词。"""
    context = candidate[max(0, start - LANDLINE_CONTEXT_WINDOW) : start]
    return any(keyword in context for keyword in LANDLINE_CONTEXT_KEYWORDS)


def extract_phones(text: str, *, allow_landline: bool = False) -> list[str]:
    """提取文本里**所有**号码，按出现顺序去重（T11b 的号码复核用）。

    与 `extract_phone()` 共用同一套规范化与接受规则（11 位手机号总是接受；座机号需要
    `allow_landline` 或上下文关键词），区别只是把全部结果都返回而不是只取第一个。
    来电者纠正号码（“…不对，是…”）时两遍都可能被转写出来，复核需要看到第二个。

    同时出现在合并分隔符前后的同一号码只留一份；被另一个更长号码包含的短号码
    （如“010-12345678”同时命中 `01012345678` 与 `12345678`）会丢弃，避免把号码片段
    当成独立的号码报告。
    """
    if not text:
        return []
    normalized = cn_digits_to_arabic(unicodedata.normalize("NFKC", text))
    # 与 extract_phone 相同的两种候选文本：合并数字间分隔符的、原样的。
    candidates = (_SEPARATOR_RE.sub("", normalized), normalized)

    found: list[str] = []
    for candidate in candidates:
        matches: list[tuple[int, str]] = [
            (match.start(), match.group(0)) for match in _MOBILE_RE.finditer(candidate)
        ]
        for pattern in (_AREA_CODE_LANDLINE_RE, _LANDLINE_RE):
            for match in pattern.finditer(candidate):
                if allow_landline or _has_landline_context(candidate, match.start()):
                    matches.append((match.start(), match.group(0)))
        for _, number in sorted(matches):
            if number not in found:
                found.append(number)

    # 丢掉被其它号码包含的片段，并保持原出现顺序。
    return [number for number in found if not any(number != other and number in other for other in found)]


def normalize_time_expr(text: str) -> str:
    """时间表达的占位实现：原样返回（T03 不解析时间）。

    保留函数名是为了让 T07/T08 的调用方现在就能接上，届时在这里把
    "明天下午三点" 之类解析成结构化时间，不需要改调用点。
    """
    return text


# 号码逐位读法（T07c）：1 读“幺”，电话里更不容易听错，与 core/phrases.py 的数字片段一致。
# 本模块不导入 core.phrases（phrases 反过来要导入本模块的 phone_groups），所以这里单独定义一份。
SPOKEN_DIGITS: dict[str, str] = {
    "0": "零",
    "1": "幺",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def phone_groups(number: str) -> list[str]:
    """按电话分段规则把号码切成组：手机号 3-4-4，座机 区号-号码，其它每 4 位一组。

    非数字字符（空格、横线、括号）先剔除；没有数字时抛 ValueError。
    T07c 起本函数放在这里，`core/phrases.py` 改为从本模块导入（号码分组只有一份）。
    """
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    if not digits:
        raise ValueError(f"phone number has no digits: {number!r}")
    if len(digits) == 11 and digits[0] == "1":
        return [digits[:3], digits[3:7], digits[7:]]
    if digits.startswith("0") and len(digits) >= 10:
        # 010/02x 是 3 位区号（含前导 0），03xx–09xx 是 4 位。
        area_len = 3 if digits[1] in "12" else 4
        if len(digits) - area_len in (7, 8):
            return [digits[:area_len], digits[area_len:]]
    return [digits[i : i + 4] for i in range(0, len(digits), 4)]


def phone_to_spoken(number: str) -> str:
    """把号码写成 TTS 可直接念的汉字：逐位读，组间用逗号。

    `"13800138000"` → `"幺三八，零零幺三，八零零零"`。
    号码文本始终由代码生成，不让 LLM 写（位数会错，阿拉伯数字还可能被读成“一百三十八亿…”）。
    """
    groups = phone_groups(number)
    return "，".join("".join(SPOKEN_DIGITS[digit] for digit in group) for group in groups)


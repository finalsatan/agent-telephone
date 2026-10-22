"""确认回复的意图判定（T07c，第 11 次评审）。

从 `core.dialog` 拆出来的原因有两个：

- `core.prompts` 需要判断“对方这一轮是不是在否定”，原来它 `from core.dialog import ...`；
  T06b 之后 `dialog → phrases → prompts → dialog` 会形成循环导入。
- 判定逻辑只有字符串处理，放在独立模块里更容易测，也能让 `dialog` 与 `prompts` 共用同一套规则。

**本模块只依赖标准库**（不得导入 core 里的其它模块），这样任何模块都可以安全地导入它。

规则（顺序固定，先疑问、再否定、最后肯定）：

1. 规范化后为空 → `UNCLEAR`；
2. 含疑问标记（`？`/`?`/`吗`/`什么`/`哪`/`是不是`/`怎么`/`多少`/`如何`）→ `UNCLEAR`；
3. 删掉客套话（“对不起”“不好意思”“抱歉”“谢谢”…）、含糊语（“好像”“似乎”…）与标点；
4. 命中否定句式 → `NEGATIVE`（`(?<![没不])错` 让“没错”“不错”不被当成否定）；
5. 只有一个语气词“嗯”（“嗯”“嗯嗯”）→ `AFFIRMATIVE`（电话里就是“对”的意思）；
6. 命中肯定词 → `AFFIRMATIVE`；
7. 都不命中 → `UNCLEAR`（“我没听清”“你再说一遍”）。
"""

from __future__ import annotations

import re
import unicodedata

AFFIRMATIVE = "affirmative"
NEGATIVE = "negative"
UNCLEAR = "unclear"

# 肯定词。按“整句意图”判断，不是简单的子串匹配。
AFFIRMATIVE_KEYWORDS: tuple[str, ...] = (
    "没问题",
    "没有问题",
    "没错",
    "没错了",
    "正确",
    "对的",
    "是的",
    "可以",
    "好的",
    "行",
    "对",
    "是",
    "好",
    "ok",
)

# 否定句式（正则）。否定判定先于肯定；`(?<![没不])错` 让“没错／不错”不被当成否定。
NEGATIVE_PATTERNS: tuple[str, ...] = (
    r"不对",
    r"不是",
    r"不准确",
    r"有误",
    r"不可以",
    r"不行",
    r"要改",
    r"改一下",
    r"麻烦改",
    r"少[位一了\d]",
    r"多[位一了\d]",
    r"缺",
    r"漏",
    r"(?<![没不])错",
)

# 疑问标记：出现任一个就不算肯定（“你刚才说的是什么？”“对了吗？”“是不是”）。
QUESTION_MARKERS: tuple[str, ...] = (
    "？",
    "?",
    "吗",
    "什么",
    "哪",
    "是不是",
    "怎么",
    "多少",
    "如何",
)

# 客套话与含糊语：先删掉再判断，避免“对不起”里的“对”被当成肯定、
# “不好意思”里的“不好”被当成否定、“好像”被当成肯定来源。
POLITENESS_PHRASES: tuple[str, ...] = (
    "对不起",
    "不好意思",
    "抱歉",
    "谢谢",
    "没关系",
    "不客气",
)
HEDGE_PHRASES: tuple[str, ...] = ("好像", "似乎", "大概", "印象中")

# 判断前先剔除的标点与语气词。
PUNCTUATION = "，。！？、；：,.!?;:~～…—-·\"'“”‘’（）()[]【】《》〈〉 \t\n\r"
FILLER_CHARS = "啊呀吧呢嘛哦噢哈嗯呃哇啦咯喽"

# 只有语气词“嗯”时的判定（电话里“嗯”就是“对”）。
_AFFIRMATIVE_HUM_RE = re.compile(r"嗯+")


def normalize_text(text: str) -> str:
    """NFKC 归一化（全角数字/标点）并去掉首尾空白。"""
    return unicodedata.normalize("NFKC", text or "").strip()


def strip_noise(text: str) -> str:
    """去掉标点与语气词，只保留有意义的字符。"""
    return "".join(ch for ch in text if ch not in PUNCTUATION and ch not in FILLER_CHARS)


def classify_confirm_reply(text: str) -> str:
    """判断 CONFIRM 状态下的回复意图：`affirmative` / `negative` / `unclear`。"""
    raw = normalize_text(text)
    if not raw:
        return UNCLEAR
    # 含疑问的句子不算肯定（“你刚才说的是什么？”“对了吗？”）
    if any(marker in raw for marker in QUESTION_MARKERS):
        return UNCLEAR

    cleaned = raw
    for phrase in POLITENESS_PHRASES + HEDGE_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    cleaned = "".join(ch for ch in cleaned if ch not in PUNCTUATION)
    if not cleaned.strip():
        return UNCLEAR

    # 否定先于肯定：“不对”里含“对”，但“没错”“不错”不算否定（正则负向后顾）。
    if any(re.search(pattern, cleaned) for pattern in NEGATIVE_PATTERNS):
        return NEGATIVE
    # 只有“嗯”/“嗯嗯”：电话里就是肯定的回应。
    if _AFFIRMATIVE_HUM_RE.fullmatch(cleaned):
        return AFFIRMATIVE
    if any(keyword in cleaned for keyword in AFFIRMATIVE_KEYWORDS):
        return AFFIRMATIVE
    return UNCLEAR

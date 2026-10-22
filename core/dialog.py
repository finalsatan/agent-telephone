"""对话状态机（T07 / T07b）。

设计文档第 4 节：状态由代码维护，LLM 只负责生成当前状态下的 reply 与提取槽位。
`Dialog` 不调用网络、不依赖 LLM，纯逻辑，便于测试与离线回放（T10 直接复用）。

## 状态流转

    GREETING → ASK_IDENTITY → ASK_PURPOSE → FILL_MISSING → CONFIRM → CLOSING

- `greeting()` 给出问候话术并把状态推进到 `ASK_IDENTITY`（问候语里已经问了“您是哪位”）。
- 每收到一轮对方的话，调用方执行 `on_user_turn(text, slots_update, parse_ok)`，
  取回 `(next_state, reply_hint)`。
- 必填槽位都有值（或已放弃）→ `CONFIRM`；仍有缺失 → 继续追问（`ASK_IDENTITY` /
  `ASK_PURPOSE` / `FILL_MISSING` 只是“当前在问哪个槽位”的细化，顺序由必填槽位决定）。
  选填槽位（`time` / `location`）从不主动追问，只在对方主动说出时记录。
- 同一槽位最多追问 `max_asks` 次（默认 2，来自 `config.yaml: dialog.max_asks`），
  仍拿不到就 `abandon()`（标 unconfirmed 并跳过）。问候语里的身份询问不计入次数。
  计数器 `attempts` 记的是“已经问出口的次数”：问过回拨号码而本轮没拿到号码，
  下一轮 `_advance()` 会再计一次，因此两次空手而归后就不再追问。
- `CONFIRM` 收到肯定意图 → 现有槽位全部 `confirmed` 并进入 `CLOSING`；否定关键词
  或本轮改动了槽位值（`SlotStore` 已把被改写的值标 `corrected`）→ 视为纠正，
  回到 `FILL_MISSING` 重新追问缺失项或重新复述（设计文档第 4 节“被纠正的槽位更新后
  重新复述”）。既不能判为肯定也不能判为否定（“无法判定”）时重新复述一次。
- 任何状态下出现结束意图关键词、**且本轮没有产生新槽位值、且关键词后面没有接着说事**，
  才进入 `CLOSING`（否则“也没什么事，就是想问下快递放哪”会被提前挂断）。
- `parse_ok=False`（LLM 输出不是合法 JSON）→ 保持当前状态，用 `repeat` 兜底话术；
  这一轮不推进状态也不计追问失败。连续失败超过 `DEFAULT_MAX_PARSE_FAILURES` 次后
  不再让 LLM 重说，改播当前追问槽位的模板（`CONFIRM` 中则重新复述）。

## 肯定／否定判定（T07b，第 9 次评审）

原先用关键词子串匹配，“对不起，我没听清”“好像号码少了一位”“对，没有问题”都会判错。
现在 `classify_confirm_reply()` 按整句意图判断：

1. 含疑问标记（`？`、`吗`、`什么`、`哪`、`是不是`…）→ 无法判定（不算肯定）；
2. 去掉标点、客套话（“对不起”“不好意思”“抱歉”“谢谢”）与含糊语（“好像”“似乎”…）后，
   匹配否定句式（`不对`、`不是`、`错了`、`少了`、`少一位`…）→ 否定；
3. 匹配肯定词（`对`、`是的`、`没错`、`没问题`、`好的`…）→ 肯定；
4. 都不匹配 → 无法判定。

否定判定先于肯定（“不对”里含“对”），但“没错”里的“错”不算否定（正则用负向后顾）。

## 模板优先（第 4 次评审 / T07b）

云端 TTS 首包太慢，能预合成的话术就不要现合成，因此 `ReplyHint.kind` 分两种：

- `"template"`：`template_id` 是 `PHRASES` 里的 id（与 T06 话术表对齐），
  `text` 是填好机主称呼的完整句子。
- `"llm"`：模板覆盖不了（复述姓名/来意、回应对方的提问），播放 LLM 流式生成的句子。

**号码复述一律用 T06 的 `render_phone`**（逐位读）：`CONFIRM` 里只要 `callback_number` 有值，
`ReplyHint.phone` 就填上号码。`kind="template"` + `phone` 表示直接播 `render_phone(phone)`
（纯号码复述，`template_id="confirm_phone"`）；`kind="llm"` + `phone` 表示先播 LLM 生成的
姓名/来意复述，**再接** `render_phone(phone)`（其中含“您的回拨电话是……，对吗？”）。
唯一的例外是“否定但没给新值”时追问哪一项不对：这时 `phone` 留空，避免把旧号码再念一遍。

`PHRASES` 与 T06 的 `core/phrases.py` 是同一批 id（T07b 要求本任务不新建 `phrases.py`）。
T06 落地时把话术表迁到 `core/phrases.py`，由本模块引用，两边文本必须一致。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from core.slots import SlotStore
from core.types import REQUIRED_SLOTS, SLOT_NAMES, SlotValue, State

# 追问上限的默认值；config.yaml 的 dialog.max_asks 可覆盖（AGENTS.md 规则 4）。
DEFAULT_MAX_ASKS = 2
# CONFIRM 里“否定／无法判定”连续多少轮后收尾；config.yaml 的 dialog.max_confirm_rounds 可覆盖。
DEFAULT_MAX_CONFIRM_ROUNDS = 2
# 连续解析失败超过这个次数就不再播 repeat，改播当前追问的模板（T07b）。
DEFAULT_MAX_PARSE_FAILURES = 2
DEFAULT_OWNER_NAME = "机主"

# 结束意图关键词：任何状态下出现就考虑进入 CLOSING。写在模块顶部便于扩展（T07 要求）。
END_INTENT_KEYWORDS: tuple[str, ...] = (
    "没事了",
    "没什么事",
    "不用了",
    "不麻烦",
    "再见",
    "拜拜",
    "挂了",
    "先这样",
    "就这样",
    "回头再说",
)

# 结束关键词后面出现这些词，说明对方还有事要说，不挂断（T07b）。
END_CONTINUATION_MARKERS: tuple[str, ...] = (
    "就是",
    "但是",
    "不过",
    "可是",
    "还有",
    "另外",
    "想",
    "让他",
    "麻烦他",
    "的话",
)

# 肯定词（用于 classify_confirm_reply）。按“整句意图”判断，不是简单的子串匹配。
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

# 否定句式（正则，T07b）。否定判定先于肯定；`(?<![没不])错` 让“没错／不错”不被当成否定。
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

# 客套话与含糊语：先删掉再判断，避免“对不起”里的“对”、“好像”被当成肯定来源，
# 也避免“不好意思”里的“不好”被当成否定（T07b 的要求）。
POLITENESS_PHRASES: tuple[str, ...] = (
    "对不起",
    "不好意思",
    "抱歉",
    "谢谢",
    "没关系",
    "不客气",
)
HEDGE_PHRASES: tuple[str, ...] = ("好像", "似乎", "大概", "印象中")

_PUNCTUATION = "，。！？、；：,.!?;:~～…—-·\"'“”‘’（）()[]【】《》〈〉 \t\n\r"
_FILLER_CHARS = "啊呀吧呢嘛哦噢哈嗯呃哇啦咯喽"

# 可预合成的话术表（id → 文本模板）。与 T06 的话术 id 对齐，机主称呼由 config.yaml 提供。
PHRASES: dict[str, str] = {
    "greet": "您好，这里是{owner}的电话助理，他暂时不方便接听。请问您是哪位？",
    "ask_name": "请问您是哪位？",
    "ask_purpose": "请问您有什么事？",
    "ask_callback": "请问您的回拨电话是多少？",
    "repeat": "抱歉，我没听清，请您再说一遍。",
    "confirm_phone_prefix": "您的回拨电话是",
    "confirm_phone_suffix": "，对吗？",
    "forward": "好的，我会转告{owner}，请您留一下联系方式。",
    "goodbye": "好的，我会转告{owner}，再见。",
}

# 每个必填槽位对应的话术 id。
ASK_TEMPLATES: dict[str, str] = {
    "name": "ask_name",
    "purpose": "ask_purpose",
    "callback_number": "ask_callback",
}

# 复述号码时用的模板 id（前缀＋逐位数字＋后缀，见 T06 的 render_phone）。
CONFIRM_PHONE_ID = "confirm_phone"

AFFIRMATIVE = "affirmative"
NEGATIVE = "negative"
UNCLEAR = "unclear"


@dataclass(frozen=True)
class ReplyHint:
    """这一轮该说什么。

    - `kind="template"`：播放 `PHRASES[template_id]`（`text` 已填好机主称呼）。
    - `kind="llm"`：播放 LLM 流式生成的句子。

    `phone` 只在 `CONFIRM` 状态出现，非空时表示号码要交给 T06 的 `PhraseCache.render_phone()`：

    - `kind="template"` + `phone`：这一轮就是纯号码复述，直接播 `render_phone(phone)`
      （`text` 是同义的可现合成兜底文本，缓存未命中时用）。
    - `kind="llm"` + `phone`：先播 LLM 生成的姓名/来意复述，**再接着**播
      `render_phone(phone)`（含“您的回拨电话是……，对吗？”）；LLM 的提示词已经要求它
      不要念号码、不要以问句结尾（见 `core.prompts`）。
    - `kind="llm"` + `phone=None`：普通 LLM 回复；否定后追问“哪一项不对”时也走这里，
      目的是不要把旧号码再念一遍。
    """

    kind: str
    template_id: str | None = None
    text: str = ""
    phone: str | None = None

    @property
    def use_llm(self) -> bool:
        return self.kind == "llm"

    @property
    def use_cache(self) -> bool:
        return self.kind == "template"


def classify_confirm_reply(text: str) -> str:
    """判断 CONFIRM 状态下的回复意图：`affirmative` / `negative` / `unclear`。

    规则见模块文档“肯定／否定判定”。无法判定时返回 `unclear`（既不确定也不否定）。
    """
    raw = unicodedata.normalize("NFKC", text or "")
    if not raw.strip():
        return UNCLEAR
    # 含疑问的句子不算肯定（“你刚才说的是什么？”“对了吗？”）
    if any(marker in raw for marker in QUESTION_MARKERS):
        return UNCLEAR

    cleaned = raw
    for phrase in POLITENESS_PHRASES + HEDGE_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    cleaned = "".join(ch for ch in cleaned if ch not in _PUNCTUATION)
    if not cleaned.strip():
        return UNCLEAR

    # 否定先于肯定：“不对”里含“对”，但“没错”“不错”不算否定（正则负向后顾）。
    if any(re.search(pattern, cleaned) for pattern in NEGATIVE_PATTERNS):
        return NEGATIVE
    if any(keyword in cleaned for keyword in AFFIRMATIVE_KEYWORDS):
        return AFFIRMATIVE
    return UNCLEAR


class Dialog:
    """通话状态机 + 槽位管理。所有方法都是同步的，不产生任何 I/O。"""

    def __init__(
        self,
        *,
        owner_name: str = DEFAULT_OWNER_NAME,
        max_asks: int = DEFAULT_MAX_ASKS,
        max_confirm_rounds: int = DEFAULT_MAX_CONFIRM_ROUNDS,
        max_parse_failures: int = DEFAULT_MAX_PARSE_FAILURES,
    ) -> None:
        if max_asks < 1:
            raise ValueError(f"max_asks must be >= 1, got {max_asks!r}")
        if max_confirm_rounds < 0:
            raise ValueError(f"max_confirm_rounds must be >= 0, got {max_confirm_rounds!r}")
        self.owner_name = (owner_name or "").strip() or DEFAULT_OWNER_NAME
        self.max_asks = int(max_asks)
        self.max_confirm_rounds = int(max_confirm_rounds)
        self.max_parse_failures = int(max_parse_failures)
        self.slot_store = SlotStore()
        self.state = State.GREETING
        self._attempts: dict[str, int] = {name: 0 for name in SLOT_NAMES}
        # 上一句提示里问的是哪个槽位（None 表示没在追问，如 CONFIRM/CLOSING）。
        self._asked_slot: str | None = None
        # CONFIRM 里“否定／无法判定”的连续轮数（有进展时归零）。
        self._confirm_rounds = 0
        # 连续解析失败的轮数（parse_ok=True 时归零）。
        self._parse_failures = 0

    # ---- 构造 -------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "Dialog":
        """从 config.yaml 构造；显式 kwargs 优先（测试/离线回放临时覆盖）。"""
        owner = (config.get("owner") or {}).get("name", DEFAULT_OWNER_NAME)
        dialog_cfg = config.get("dialog") or {}
        kwargs.setdefault("owner_name", owner)
        kwargs.setdefault("max_asks", dialog_cfg.get("max_asks", DEFAULT_MAX_ASKS))
        kwargs.setdefault(
            "max_confirm_rounds",
            dialog_cfg.get("max_confirm_rounds", DEFAULT_MAX_CONFIRM_ROUNDS),
        )
        return cls(**kwargs)

    # ---- 只读状态 ---------------------------------------------------------

    @property
    def slots(self) -> dict[str, SlotValue]:
        """槽位快照，直接交给 `core.prompts.build_messages` / `llm.turn_stream`。"""
        return self.slot_store.as_dict()

    @property
    def attempts(self) -> dict[str, int]:
        """各槽位已被追问的次数（>0 的项），用于 LLM 提示词与事件日志。"""
        return {name: n for name, n in self._attempts.items() if n}

    @property
    def asked_slot(self) -> str | None:
        """上一句提示追问的槽位名；没在追问时为 None。"""
        return self._asked_slot

    @property
    def confirm_rounds(self) -> int:
        """CONFIRM 里已经连续“否定／无法判定”的轮数（调试与测试用）。"""
        return self._confirm_rounds

    # ---- 状态机 -----------------------------------------------------------

    def greeting(self) -> ReplyHint:
        """接通后的第一句：问候并询问身份，状态推进到 ASK_IDENTITY。"""
        self.state = State.ASK_IDENTITY
        self._asked_slot = "name"
        return self._template("greet")

    def on_user_turn(
        self,
        text: str,
        slots_update: Mapping[str, object] | None = None,
        parse_ok: bool = True,
    ) -> tuple[State, ReplyHint]:
        """处理对方的一轮话，返回 `(next_state, reply_hint)`。

        - `text`：ASR 最终文本（回拨号码只从这里提取）；
        - `slots_update`：LLM 的 `TurnResult.slots_update`，其中的 `callback_number` 会被忽略；
        - `parse_ok`：LLM 输出是否解析成功；False 时保持状态并播 `repeat`（连续失败转模板）。
        """
        text = text or ""
        asking_phone = self._asked_slot == "callback_number"
        # 号码先提取（不看 parse_ok：号码来自 ASR 文本 + 规则，与 LLM 是否解析成功无关）。
        # CONFIRM 里刚复述过号码，对方很可能直接报正确的号，所以这时也允许座机（T07b）。
        phone_before = self.slot_store.value("callback_number")
        self.slot_store.update_phone_from_text(
            text,
            allow_landline=asking_phone or self.state is State.CONFIRM,
        )
        changed: dict[str, SlotValue] = {}
        if self.slot_store.value("callback_number") != phone_before:
            changed["callback_number"] = self.slot_store.get("callback_number")
        if parse_ok:
            changed.update(self.slot_store.update(slots_update))
            self._parse_failures = 0
        else:
            self._parse_failures += 1

        if self._is_end_intent(text, changed):
            # CONFIRM 里“结束意图＋肯定”要先确认再挂断（“对，就这样吧”）。
            if self.state is State.CONFIRM and classify_confirm_reply(text) == AFFIRMATIVE:
                self.slot_store.confirm_all()
            self.state = State.CLOSING
            self._asked_slot = None
            return self.state, self._template("goodbye")

        if not parse_ok:
            return self.state, self._parse_failure_hint()

        if self.state is State.GREETING:
            # 问候还没播对方就开口了：直接进入身份阶段。
            self.state = State.ASK_IDENTITY
        if self.state is State.CLOSING:
            return self.state, self._template("goodbye")
        if self.state is State.CONFIRM:
            return self._handle_confirm(text, changed)
        return self._advance()

    # ---- 内部：结束意图与解析失败 -----------------------------------------

    def _is_end_intent(self, text: str, changed: Mapping[str, SlotValue]) -> bool:
        """判断是否真的要挂断。

        三个条件同时成立才算：本轮没有新槽位值、出现结束关键词、关键词后面没说别的事
        （“也没什么事，就是想问下快递放哪”“不麻烦的话让他回个电话”都不挂断）。
        """
        if changed:
            return False
        for keyword in END_INTENT_KEYWORDS:
            start = text.find(keyword)
            if start < 0:
                continue
            tail = text[start + len(keyword) :]
            if any(marker in tail for marker in END_CONTINUATION_MARKERS):
                continue
            if _substantive_tail(tail):
                continue
            return True
        return False

    def _parse_failure_hint(self) -> ReplyHint:
        """解析失败时的兜底：先 repeat，连续失败超过上限就重播当前追问的模板。"""
        if self._parse_failures <= self.max_parse_failures:
            return self._template("repeat")
        if self.state is State.CONFIRM:
            return self._confirm_hint()
        if self._asked_slot in ASK_TEMPLATES:
            return self._template(ASK_TEMPLATES[self._asked_slot])
        return self._template("repeat")

    # ---- 内部：CONFIRM -----------------------------------------------------

    def _handle_confirm(self, text: str, changed: Mapping[str, SlotValue]) -> tuple[State, ReplyHint]:
        """CONFIRM 状态：肯定 → 全部确认并收尾；否定/纠正 → 回 FILL_MISSING 重新走流程。

        `changed` 是本轮确实被改动的槽位。带着“对”字但改了值（“对，我姓王”）也算纠正，
        否则会把刚改过的值当成已确认。
        """
        if changed:
            self._confirm_rounds = 0
            self.state = State.FILL_MISSING
            # 纠正后的值已由 update() 标为 corrected；重新追问缺失项或重新复述。
            return self._advance()

        intent = classify_confirm_reply(text)
        if intent == AFFIRMATIVE:
            self.slot_store.confirm_all()
            self._confirm_rounds = 0
            self.state = State.CLOSING
            self._asked_slot = None
            return self.state, self._template("goodbye")

        # 否定与“无法判定”合计超过上限 → 收尾，槽位保持未确认。
        self._confirm_rounds += 1
        if self._confirm_rounds > self.max_confirm_rounds:
            self.state = State.CLOSING
            self._asked_slot = None
            return self.state, self._template("goodbye")

        if intent == NEGATIVE:
            # 追问哪一项不对：交给 LLM（不带号码，避免把旧号码再念一遍）。
            return self.state, ReplyHint(kind="llm")
        # 无法判定（“我没听清”“你说什么？”）：重新复述一次。
        return self.state, self._confirm_hint()

    # ---- 内部：追问与复述 --------------------------------------------------

    def _advance(self) -> tuple[State, ReplyHint]:
        """按必填槽位的顺序找出下一个要问的槽位；都齐了就进入 CONFIRM。"""
        for name in REQUIRED_SLOTS:
            if self.slot_store.has_value(name) or self.slot_store.is_abandoned(name):
                continue
            if self._attempts[name] >= self.max_asks:
                # 追问次数用尽：标 unconfirmed 并跳过（设计文档第 4 节）。
                self.slot_store.abandon(name)
                continue
            self._attempts[name] += 1
            self.state = _STATE_FOR_SLOT[name]
            self._asked_slot = name
            return self.state, self._template(ASK_TEMPLATES[name])

        self._asked_slot = None
        if not any(self.slot_store.has_value(name) for name in SLOT_NAMES):
            # 什么也没记到（全部被放弃）：不再复述，收尾请对方留联系方式。
            self.state = State.CLOSING
            return self.state, self._template("forward")
        self.state = State.CONFIRM
        self._confirm_rounds = 0
        return self.state, self._confirm_hint()

    def _confirm_hint(self) -> ReplyHint:
        """CONFIRM 的复述：号码一律交给 `render_phone`，其余内容交给 LLM。"""
        phone = self.slot_store.value("callback_number")
        others = [
            name
            for name in SLOT_NAMES
            if name != "callback_number" and self.slot_store.has_value(name)
        ]
        if phone and not others:
            prefix = PHRASES["confirm_phone_prefix"].format(owner=self.owner_name)
            suffix = PHRASES["confirm_phone_suffix"].format(owner=self.owner_name)
            return ReplyHint(
                kind="template",
                template_id=CONFIRM_PHONE_ID,
                text=f"{prefix}{phone}{suffix}",
                phone=phone,
            )
        if phone:
            # 先播 LLM 的姓名/来意复述，再由调用方接 render_phone(phone)。
            return ReplyHint(kind="llm", phone=phone)
        return ReplyHint(kind="llm")

    def _template(self, template_id: str) -> ReplyHint:
        return ReplyHint(
            kind="template",
            template_id=template_id,
            text=PHRASES[template_id].format(owner=self.owner_name),
        )


# 追问某个必填槽位时状态机所处的状态。
_STATE_FOR_SLOT: dict[str, State] = {
    "name": State.ASK_IDENTITY,
    "purpose": State.ASK_PURPOSE,
    "callback_number": State.FILL_MISSING,
}


def _substantive_tail(tail: str) -> bool:
    """结束关键词后面是否还接着说事（返回 True 表示“还有内容”，不该挂断）。

    去掉标点、语气词、客套/肯定用语和另一个结束关键词后仍有剩余，就认为对方还在说事：
    “，拜拜” → 空（可以挂断）；“跟你说吧，合同的事很急” → 有内容（不能挂断）。
    """
    text = tail
    for _ in range(3):
        for word in AFFIRMATIVE_KEYWORDS + POLITENESS_PHRASES + ("好吧", "行吧"):
            text = text.replace(word, "")
        for keyword in END_INTENT_KEYWORDS:
            text = text.replace(keyword, "")
        text = "".join(ch for ch in text if ch not in _PUNCTUATION and ch not in _FILLER_CHARS)
    return bool(text.strip())

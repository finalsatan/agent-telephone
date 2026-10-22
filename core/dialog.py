"""对话状态机（T07 / T07b / T07c）。

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
- `CONFIRM` 收到肯定意图 → 现有槽位全部 `confirmed` 并进入 `CLOSING`；否定意图
  或本轮改动了槽位值（`SlotStore` 已把被改写的值标 `corrected`）→ 视为纠正，
  回到 `FILL_MISSING` 重新追问缺失项或重新复述（设计文档第 4 节“被纠正的槽位更新后
  重新复述”）。否定但没给新值时播模板 `ask_which_wrong`；既不能判为肯定也不能判为否定
  （“无法判定”）时重新播确认语。两者合计超过 `max_confirm_rounds` 后收尾。
- 任何状态下出现结束意图关键词、**且本轮没有产生新槽位值、且关键词后面没有接着说事**，
  才进入 `CLOSING`（否则“也没什么事，就是想问下快递放哪”会被提前挂断）。
- `parse_ok=False`（LLM 输出不是合法 JSON）→ 保持当前状态，用 `repeat` 兜底话术；
  这一轮不推进状态也不计追问失败。连续失败超过 `DEFAULT_MAX_PARSE_FAILURES` 次后
  不再让 LLM 重说，改播当前追问槽位的模板（`CONFIRM` 中则重新复述）。

## 肯定／否定判定（T07b → T07c 拆到 `core/intent.py`）

原先用关键词子串匹配，“对不起，我没听清”“好像号码少了一位”“对，没有问题”都会判错。
现在 `core.intent.classify_confirm_reply()` 按整句意图判断（先疑问、再否定、最后肯定），
规则与词表都在那个模块里。拆出去的原因是 `core.prompts` 也要用它，而 `prompts`
不能再导入 `dialog`（`dialog → phrases → prompts → dialog` 会成环）。

## 模板优先与确认语（第 4 / 9 / 11 次评审）

云端与本地 TTS 的首包都在 1s 以上，能预合成的话术就不要现合成，因此 `ReplyHint.kind` 分三种：

- `"template"`：`template_id` 是 `core.phrases.PHRASES` 里的 id（T06b 起话术表只在那里定义一份），
  `text` 是填好机主称呼的完整句子，命中缓存直接播放。
- `"text"`（T07c）：`text` 是要**现合成**的完整句子，不查缓存、也不用 LLM。目前只有 CONFIRM
  的确认语走这里：号码必须写成汉字、和确认语在同一句里整句合成（数字拼接的听感很差，
  用户试听后否决了）。`phone` 同时填上，供 TTS 失败时退回 `render_phone(phone)`。
- `"llm"`：播放 LLM 流式生成的句子。T07c 之后 `Dialog` 不再返回这种提示（复述与追问都由
  代码生成），保留这个取值是为了 T10 的分流逻辑不需要改，也便于以后重新启用。

**为什么确认语由代码生成**：T10 的顺序是“先用当前状态调 LLM，流结束后才把 `slots_update`
交给 `Dialog`”。对方报出最后一个必填项的那一轮，LLM 收到的状态还是 FILL_MISSING，
它生成的是追问；进入 CONFIRM 的那一轮会引用一句 LLM 根本没生成的复述。
让代码按模板拼确认语，既不依赖那一轮 LLM，又能把号码写成正确的汉字读法。

话术表只在 `core/phrases.py` 定义一份（T06b 迁移），本模块 `from core.phrases import PHRASES`；
T07c 新增的 `confirm_question`（“对吗？”）、`ask_which_wrong`（“请问是哪一项不对？”）也由 T06b
预合成。两边文本必须一致，否则缓存永远命不中：`tests/test_phrases.py` 有一致性用例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.intent import (
    AFFIRMATIVE,
    AFFIRMATIVE_KEYWORDS,
    NEGATIVE,
    POLITENESS_PHRASES,
    PUNCTUATION,
    FILLER_CHARS,
    classify_confirm_reply,
)
from core.normalize import phone_to_spoken
from core.phrases import PHRASES
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

# 每个必填槽位对应的话术 id。
ASK_TEMPLATES: dict[str, str] = {
    "name": "ask_name",
    "purpose": "ask_purpose",
    "callback_number": "ask_callback",
}

# CONFIRM 里“否定但没给新值”时追问哪一项不对的话术 id（T06b 预合成）。
ASK_WHICH_WRONG_ID = "ask_which_wrong"


@dataclass(frozen=True)
class ReplyHint:
    """这一轮该说什么（T07 / T07b / T07c）。

    - `kind="template"`：播放 `PHRASES[template_id]`（`text` 已填好机主称呼），命中缓存直接播。
    - `kind="text"`：`text` 是要现合成的完整句子，**不查缓存、不用 LLM**。目前只有 CONFIRM
      的确认语走这里（号码已由 `phone_to_spoken()` 写成汉字，整句合成听感更连贯）。
    - `kind="llm"`：播放 LLM 流式生成的句子。T07c 之后 `Dialog` 不再返回它，保留是为了
      T10 的分流逻辑稳定。

    `phone` 只在 `CONFIRM` 出现，是 `callback_number` 的原值（数字串）。`kind="text"` +
    `phone` 表示“整句合成失败时，可以退回 `render_phone(phone)` 再接 `confirm_question`”，
    即 T06b 的拼接兜底路径。`kind="template"` 的话术目前不需要 `phone`。
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

    @property
    def use_text(self) -> bool:
        return self.kind == "text"


# 确认语的开头（T12 的“确认语开头预合成”实验按这个前缀拆句）。
CONFIRM_PREFIX = "跟您确认一下："
# purpose 以这些词开头时可以直接接在“您是某某，”后面；否则加“来电是关于……”（T12，第 14 次评审）。
PURPOSE_CONNECTORS: tuple[str, ...] = ("想", "关于", "要")


def compose_confirm_text(
    slots: Mapping[str, SlotValue],
    *,
    owner_name: str = DEFAULT_OWNER_NAME,
) -> str:
    """按有值的槽位拼出 CONFIRM 的确认语（T07c / T12）。

    形如“跟您确认一下：您是李明，想约张先生谈合同，回拨电话是幺三八，零零幺三，八零零零，对吗？”。
    没有值（或被放弃）的槽位跳过；号码用 `phone_to_spoken()` 逐位写成汉字（1 读“幺”），
    这样整句交给 TTS 现合成时位数与读法都由代码保证，不依赖 LLM 也不依赖数字片段拼接。

    措辞调整（第 14 次评审，T12）：
    - LLM 有时把 purpose 写成“谈合同”“送快递”，直接接在“您是某某，”后面很生硬，
      这类不以“想／关于／要”开头的来意写成“来电是关于……”。
    - 只有号码、没有姓名和来意时，用“您想找{owner}”补主语（原先是“想找{owner}”）。
    机主称呼来自 `config.yaml: owner.name`。
    """
    owner = (owner_name or "").strip() or DEFAULT_OWNER_NAME
    parts: list[str] = []

    def value(name: str) -> str | None:
        slot = slots.get(name)
        text = getattr(slot, "value", None)
        return text.strip() if isinstance(text, str) and text.strip() else None

    name = value("name")
    if name:
        parts.append(f"您是{name}")
    purpose = value("purpose")
    if purpose:
        # 提示词希望 purpose 写成“想……”或“关于……”；没按格式写时补“来电是关于”。
        if purpose.startswith(PURPOSE_CONNECTORS):
            parts.append(purpose)
        else:
            parts.append(f"来电是关于{purpose}")
    elif name:
        parts.append(f"想找{owner}")
    else:
        # 没有姓名、没有来意（通常只剩号码）：补上主语，避免“跟您确认一下：想找张先生”。
        parts.append(f"您想找{owner}")
    time_expr = value("time")
    if time_expr:
        parts.append(f"时间是{time_expr}")
    location = value("location")
    if location:
        parts.append(f"地点是{location}")
    phone = value("callback_number")
    if phone:
        parts.append(f"回拨电话是{phone_to_spoken(phone)}")

    return CONFIRM_PREFIX + "，".join(parts) + "，" + PHRASES["confirm_question"]


def is_confirm_text(text: str) -> bool:
    """这句是不是 CONFIRM 的确认语（T12 评估按 CONFIRM 轮单独统计时用）。"""
    text = (text or "").strip()
    return text.startswith(CONFIRM_PREFIX) or text.startswith(PHRASES["confirm_lead"])


def split_confirm_lead(text: str) -> tuple[str, str]:
    """把确认语拆成“预合成开头”和“现合成后半句”（T12 对比实验）。

    不开实验时返回 `("", text)`；开头不是 `CONFIRM_PREFIX` 时也不拆。
    拆出来的后半句仍是“您是……，对吗？”，与预先合成的“好的，跟您确认一下。”拼起来语义完整。
    """
    text = text or ""
    if text.startswith(CONFIRM_PREFIX):
        return PHRASES["confirm_lead"], text[len(CONFIRM_PREFIX) :]
    return "", text


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
            # 追问哪一项不对：用预合成模板，不让 LLM 现写（T07c）。
            return self.state, self._template(ASK_WHICH_WRONG_ID)
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
        """CONFIRM 的确认语：由代码整句拼好，交给 TTS 现合成（T07c）。

        `phone` 一并填上，供 TTS 失败时退回 T06b 的 `render_phone(phone)` 拼接兜底。
        """
        phone = self.slot_store.value("callback_number")
        return ReplyHint(
            kind="text",
            text=compose_confirm_text(self.slot_store.as_dict(), owner_name=self.owner_name),
            phone=phone,
        )

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
        text = "".join(ch for ch in text if ch not in PUNCTUATION and ch not in FILLER_CHARS)
    return bool(text.strip())

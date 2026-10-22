"""对话状态机（T07）。

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
- `CONFIRM` 收到肯定关键词 → 现有槽位全部 `confirmed` 并进入 `CLOSING`；否定关键词
  或本轮改动了槽位值（`SlotStore` 已把被改写的值标 `corrected`）→ 视为纠正，
  回到 `FILL_MISSING` 重新追问缺失项或重新复述（设计文档第 4 节“被纠正的槽位更新后
  重新复述”）。否则把复述再说一遍，不消耗追问次数。
- 任何状态下出现结束意图关键词 → 直接 `CLOSING`。
- `parse_ok=False`（LLM 输出不是合法 JSON）→ 保持当前状态，用 `repeat` 兜底话术；
  这一轮不推进状态也不计追问失败。

## 模板优先（第 4 次评审）

云端 TTS 首包太慢，能预合成的话术就不要现合成，因此 `ReplyHint.kind` 分两种：

- `"template"`：`template_id` 是 `PHRASES` 里的 id（与 T06 话术表对齐），
  `text` 是填好机主称呼的完整句子，`phone` 非空时调用方应改用
  `PhraseCache.render_phone(phone)` 拼号码（逐位读，见 T06）。命中缓存直接播放。
- `"llm"`：模板覆盖不了（复述姓名/来意、回应对方的提问），播放 LLM 流式生成的句子。

`PHRASES` 与 T06 的 `core/phrases.py` 是同一批 id。T06 落地时应把话术表迁到
`core/phrases.py` 并由本模块引用（或从本模块导入），两边文本必须一致，否则同一个 id
的缓存会与现合成句的音色/文本不一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.slots import SlotStore
from core.types import REQUIRED_SLOTS, SLOT_NAMES, SlotValue, State

# 追问上限的默认值；config.yaml 的 dialog.max_asks 可覆盖（AGENTS.md 规则 4）。
DEFAULT_MAX_ASKS = 2
DEFAULT_OWNER_NAME = "机主"

# 结束意图关键词：任何状态下出现就进入 CLOSING。写在模块顶部便于扩展（T07 要求）。
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

# CONFIRM 状态的肯定/否定关键词。否定的判定必须先于肯定，因为“不是”里含“是”、“不对”里含“对”。
AFFIRMATIVE_KEYWORDS: tuple[str, ...] = (
    "对",
    "是的",
    "是",
    "没错",
    "正确",
    "没问题",
    "可以",
    "好的",
    "好",
    "行",
    "嗯",
    "ok",
)
NEGATION_KEYWORDS: tuple[str, ...] = (
    "不对",
    "不是",
    "不准确",
    "错了",
    "说错",
    "有误",
    "不好",
    "改一下",
    "要改",
    "没有",
)

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


@dataclass(frozen=True)
class ReplyHint:
    """这一轮该说什么。

    kind="template"：播放 `PHRASES[template_id]`（`text` 已填好机主称呼）；
    `phone` 非空表示应当用 `PhraseCache.render_phone(phone)` 拼接后播放。
    kind="llm"：播放 LLM 流式生成的句子。
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


class Dialog:
    """通话状态机 + 槽位管理。所有方法都是同步的，不产生任何 I/O。"""

    def __init__(
        self,
        *,
        owner_name: str = DEFAULT_OWNER_NAME,
        max_asks: int = DEFAULT_MAX_ASKS,
    ) -> None:
        if max_asks < 1:
            raise ValueError(f"max_asks must be >= 1, got {max_asks!r}")
        self.owner_name = (owner_name or "").strip() or DEFAULT_OWNER_NAME
        self.max_asks = int(max_asks)
        self.slot_store = SlotStore()
        self.state = State.GREETING
        self._attempts: dict[str, int] = {name: 0 for name in SLOT_NAMES}
        # 上一句提示里问的是哪个槽位（null 表示没在追问，如 CONFIRM/CLOSING）。
        self._asked_slot: str | None = None

    # ---- 构造 -------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "Dialog":
        """从 config.yaml 构造；显式 kwargs 优先（测试/离线回放临时覆盖）。"""
        owner = (config.get("owner") or {}).get("name", DEFAULT_OWNER_NAME)
        dialog_cfg = config.get("dialog") or {}
        kwargs.setdefault("owner_name", owner)
        kwargs.setdefault("max_asks", dialog_cfg.get("max_asks", DEFAULT_MAX_ASKS))
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

        - `text`：ASR 最终文本（回拨号码只从这里提取，`allow_landline` 取决于是否在追问号码）；
        - `slots_update`：LLM 的 `TurnResult.slots_update`，其中的 `callback_number` 会被忽略；
        - `parse_ok`：LLM 输出是否解析成功；False 时保持状态并播 `repeat`。
        """
        text = text or ""
        asking_phone = self._asked_slot == "callback_number"
        # 号码先提取（不看 parse_ok：号码来自 ASR 文本 + 规则，与 LLM 是否解析成功无关）。
        phone_before = self.slot_store.value("callback_number")
        self.slot_store.update_phone_from_text(text, allow_landline=asking_phone)
        changed: dict[str, SlotValue] = {}
        if self.slot_store.value("callback_number") != phone_before:
            changed["callback_number"] = self.slot_store.get("callback_number")
        if parse_ok:
            changed.update(self.slot_store.update(slots_update))

        if _contains_any(text, END_INTENT_KEYWORDS):
            self.state = State.CLOSING
            self._asked_slot = None
            return self.state, self._template("goodbye")

        if not parse_ok:
            # 保持当前状态，用兜底话术重复刚才的问题；这一轮不推进、不计追问失败。
            return self.state, self._template("repeat")

        if self.state is State.GREETING:
            # 问候还没播对方就开口了：直接进入身份阶段。
            self.state = State.ASK_IDENTITY
        if self.state is State.CLOSING:
            return self.state, self._template("goodbye")
        if self.state is State.CONFIRM:
            return self._handle_confirm(text, changed)
        return self._advance()

    # ---- 内部 -------------------------------------------------------------

    def _handle_confirm(self, text: str, changed: Mapping[str, SlotValue]) -> tuple[State, ReplyHint]:
        """CONFIRM 状态：肯定 → 全部确认并收尾；否定/纠正 → 回 FILL_MISSING 重新走流程。

        `changed` 是本轮确实被改动的槽位。带着“对”字但改了值（“对，我姓王”）也算纠正，
        否则会把刚改过的值当成已确认。
        """
        if changed or _contains_any(text, NEGATION_KEYWORDS):
            self.state = State.FILL_MISSING
            # 纠正后的值已由 update() 标为 corrected；重新追问缺失项或重新复述。
            return self._advance()
        if _contains_any(text, AFFIRMATIVE_KEYWORDS):
            self.slot_store.confirm_all()
            self.state = State.CLOSING
            self._asked_slot = None
            return self.state, self._template("goodbye")
        # 既没肯定也没否定：把复述再说一遍（不消耗追问次数）。
        return self.state, self._confirm_hint()

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
        return self.state, self._confirm_hint()

    def _confirm_hint(self) -> ReplyHint:
        """CONFIRM 的复述方式：只有号码时用号码模板（可预合成/拼接），否则交给 LLM。"""
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


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)

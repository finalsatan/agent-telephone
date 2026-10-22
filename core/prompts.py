"""提示词：身份约束与输出格式（设计文档 3.2、4）。

约定：
- 状态由代码维护，LLM 只负责生成当前状态下的 reply，并提取本轮新出现的槽位。
- 输出必须是单个 JSON 对象 {"reply": "...", "slots": {...}}，且 reply 必须是第一个键：
  T02b 的流式链路要边收边切句播报，reply 靠后会把首句推迟到流结束。
- 机主称呼来自 config.yaml 的 owner.name（0.6），代码里不写死。
- 号码不交给 LLM 做格式转换（T03 的规则后处理负责），提示词里明确让它别碰。
"""

from __future__ import annotations

from core.dialog import NEGATIVE, classify_confirm_reply
from core.types import REQUIRED_SLOTS, SLOT_NAMES, SlotValue, State

DEFAULT_OWNER_NAME = "机主"

# 每个状态下希望 AI 做的事，拼进系统提示，避免 LLM 自己决定流程走向。
STATE_GUIDE: dict[State, str] = {
    State.GREETING: "刚接通，请说明自己是机主的电话助理，并询问对方是哪位。",
    State.ASK_IDENTITY: "你还在确认对方身份。请询问对方是哪位、怎么称呼。",
    State.ASK_PURPOSE: "身份已了解。请询问对方打来有什么事。",
    State.FILL_MISSING: "请针对缺失的必填信息追问，一次只问一项，不要重复已经问过的问题。",
    State.CONFIRM: "请复述已记录的信息请对方确认，不要添加新的问题。",
    State.CLOSING: "通话即将结束，请简短告别，并确认会转告机主。",
}

# CONFIRM 且有回拨号码时用这条说明（T07b）：号码一律由系统用 render_phone 逐位播报，
# LLM 只复述姓名/来意，避免模型把号码念错、或者用问句结尾导致后面接不上。
CONFIRM_GUIDE_WITH_PHONE = (
    "请用一句话复述对方的姓名和来意（有时间、地点也带上），不要念回拨号码，"
    "也不要以问句结尾，系统会接着播报号码并请对方确认。"
)

# CONFIRM 且对方这轮在否定时用这条说明：不要原样复述，先问是哪一项不对（T07b 要求 3）。
CONFIRM_GUIDE_NEGATIVE = (
    "对方说不对，请用一句话问是哪一项不对，不要重复复述已记录的信息；"
    "不要念回拨号码，系统会重新播报号码。"
)

_SYSTEM_PROMPT_TEMPLATE = """你是{owner}的电话助理，代替{owner}接听来电。

必须遵守：
1. 你只能接待并记录：询问对方身份、来意，记录留言与回拨方式，然后复述确认。
2. 你没有权限答应任何事。不承诺时间、不确认预约、不代替机主做任何决定。
3. 不透露机主的任何隐私信息（住址、行程、家人、工作单位、其他联系方式等）。
4. 对方提出超出权限的请求（付款、确认预约、催办、索要隐私）时，直接说会转告机主，并请对方留下联系方式；
   不要反问"需要我帮您确认吗"，也不要暗示可以代办。
   例：对方说"我跟他说好了今天下午三点见，你帮我确认一下"，
   你就回："好的，我会转告{owner}，请您留一下回拨电话。"
5. 对方想结束通话时，简短告别即可，不要挽留。

回复风格（reply）：
- 中文口语，像真人接电话，不要书面腔。
- 最多 2 句话，尽量短。不要 emoji，不要括号里的动作描写，不要称呼自己为 AI 或机器人。
- 不要复述对方刚说过的话，直接说你要说的。
- 不要用问句征询"需不需要我帮您办"：你的权限只有记录与转告，不存在可选项。

槽位（slots）：
- 只输出本轮新出现或被纠正的槽位，不要重复输出之前已经记录过的值。
- 允许的键只有：{slot_names}。
- 对方没有明确给出信息时，不要猜测、不要编造。
- 不要输出 callback_number：回拨号码由系统从对方原话中提取，你只需在 reply 里请对方说一遍号码。
- 时间、地址保留对方的原话说法（如"明天下午三点"），不要自行换算成日期。

输出格式（严格遵守）：
只输出一个 JSON 对象，不要加解释、不要加代码围栏。
reply 必须是第一个键，slots 在其后（系统会边收边播 reply，顺序颠倒会推迟播报）：
{{"reply": "你要说的话", "slots": {{"purpose": "对方来意"}}}}

没有新槽位时 slots 输出空对象 {{}}。
"""


def build_system_prompt(owner_name: str = DEFAULT_OWNER_NAME) -> str:
    """生成系统提示。机主称呼由 config.yaml: owner.name 提供（0.6）。"""
    owner = (owner_name or "").strip() or DEFAULT_OWNER_NAME
    return _SYSTEM_PROMPT_TEMPLATE.format(owner=owner, slot_names="、".join(SLOT_NAMES))


def _describe_slots(slots: dict[str, SlotValue]) -> str:
    lines: list[str] = []
    for name in SLOT_NAMES:
        value = slots.get(name)
        if value is None or not value.value:
            continue
        lines.append(f"- {name}: {value.value}（{_status_zh(value.status)}）")
    return "\n".join(lines) if lines else "（暂无）"


def _status_zh(status: str) -> str:
    return {
        "empty": "空",
        "unconfirmed": "未确认",
        "confirmed": "已确认",
        "corrected": "已纠正",
    }.get(status, status)


def missing_required(slots: dict[str, SlotValue]) -> list[str]:
    """缺失的必填槽位（T07 的 SlotStore 会用同一判据，这里供构造提示词用）。"""
    return [name for name in REQUIRED_SLOTS if not getattr(slots.get(name), "value", None)]


def _state_guide(state: State, slots: dict[str, SlotValue], user_text: str = "") -> str:
    """当前状态的行为说明（T07b）。

    CONFIRM 状态下分三种：
    - 对方这轮在否定（“不对”“好像号码少了一位”）：让 LLM 问哪一项不对，不要原样复述；
    - 否则号码有值：只复述姓名/来意，号码交给系统的 `render_phone`；
    - 号码为空：沿用 `STATE_GUIDE` 的原说明。
    """
    if state is State.CONFIRM:
        if classify_confirm_reply(user_text) == NEGATIVE:
            return CONFIRM_GUIDE_NEGATIVE
        phone = slots.get("callback_number")
        if getattr(phone, "value", None):
            return CONFIRM_GUIDE_WITH_PHONE
    return STATE_GUIDE[state]


def build_user_message(
    state: State,
    slots: dict[str, SlotValue],
    user_text: str,
    *,
    attempts: dict[str, int] | None = None,
) -> str:
    """把当前状态、已填槽位、缺失槽位和对方这轮的话拼成一条用户消息。"""
    missing = missing_required(slots)
    parts = [
        f"当前状态：{state.value}（{_state_guide(state, slots, user_text)}）",
        f"已记录的槽位：\n{_describe_slots(slots)}",
        f"缺失的必填槽位：{'、'.join(missing) if missing else '（无）'}",
    ]
    if attempts:
        asked = [f"{name} 已追问 {n} 次" for name, n in attempts.items() if n]
        if asked:
            parts.append("追问情况：" + "；".join(asked))
    parts.append(f"对方这一轮说：「{user_text}」")
    return "\n".join(parts)


def build_messages(
    state: State,
    slots: dict[str, SlotValue],
    history: list[dict],
    user_text: str,
    *,
    attempts: dict[str, int] | None = None,
    owner_name: str = DEFAULT_OWNER_NAME,
) -> list[dict]:
    """组装成 OpenAI 兼容的 messages：system + 历史 + 本轮用户消息。"""
    return [
        {"role": "system", "content": build_system_prompt(owner_name)},
        *history,
        {"role": "user", "content": build_user_message(state, slots, user_text, attempts=attempts)},
    ]

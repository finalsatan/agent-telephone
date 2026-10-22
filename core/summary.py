"""挂机后的摘要与校正（T11，设计文档第 8 节）。

- `summarize(events, slots, llm)`：把 `events.jsonl` 还原成对话文本（区分来电者与 AI，
  被打断的 AI 句子标 `（未说完）`），交给摘要大模型（`LlmClient.complete()` 默认用
  `llm.summary_model`）产出七字段 JSON，返回结果，并写入 `summary.json`。
- `recheck_slots(caller_wav, slots, asr_alt)`：用另一个 ASR 后端（`asr.recheck_model`）
  重转写整通录音，只比对姓名与号码；不一致写成“未确认信息”的条目，**不修改槽位**。
- `summarize_call(call_dir, llm, asr_alt=...)`：把上面两步串起来，给 T12 评估与页面直接用。

“说出口”以事件流为准（设计文档第 7 节的时间线是唯一数据源）：

- 来电者：`asr_final.data.text`；
- AI：`play_start.data.text` 才算开播，配对的 `play_end` 表示播完；
  有 `play_start` 却没有 `play_end`、或其后紧跟 `interrupted` 的句子 → 标 `（未说完）`，
  写进提示词时明确告知“没说完的句子不算已告知”；
- `interrupted(reason="turn_merged")`（T10 的轮次合并）那一轮从未播出，不算 AI 说过话。

槽位入参兼容三种形态：`SlotStore.as_dict()` 的 `{name: SlotValue}`、`slots.json` 的
`{name: {"value": ..., "status": ...}}`、以及整份 `slots.json`（`{"call_id": ..., "slots": {...}}`）。
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.events import read_events
from core.normalize import extract_phone
from core.types import SLOT_NAMES, SlotValue

# 设计文档第 8 节的七字段（键名即摘要 JSON 的键名）。
SUMMARY_FIELDS: tuple[str, ...] = (
    "来电者",
    "来意",
    "已确认事项",
    "待办",
    "回拨需求",
    "未确认信息",
    "接管情况",
)

# 被打断的 AI 句子在提示词里的标记（任务书 T11 的措辞）。
INTERRUPTED_MARK = "（未说完）"

# 缺省复核 ASR 后端（config.yaml 的 asr.recheck_model 优先）。
DEFAULT_RECHECK_MODEL = "FunAudioLLM/SenseVoiceSmall"

# 槽位在提示词里的中文名，便于摘要模型理解字段含义。
SLOT_LABELS: dict[str, str] = {
    "name": "来电者姓名",
    "purpose": "来意",
    "callback_number": "回拨号码",
    "time": "时间",
    "location": "地点",
}

_STATUS_LABELS: dict[str, str] = {
    "confirmed": "已确认",
    "unconfirmed": "未确认",
    "corrected": "已更正，待再确认",
    "empty": "未填写",
}

SUMMARY_SYSTEM_PROMPT = f"""\
你是电话代接助理的通话记录整理员。根据用户给出的“槽位记录”和“对话记录”，输出一个 JSON 对象。

JSON 必须恰好包含下面 7 个字段，键名固定，值都是字符串（没有内容时写空字符串）：
{{"来电者": "", "来意": "", "已确认事项": "", "待办": "", "回拨需求": "", "未确认信息": "", "接管情况": ""}}

规则：
1. 只依据给出的记录，不猜测、不补充、不编造。多个事项用中文顿号或分号分隔。
2. “来电者”写对方自报的身份或称呼，“来意”用一句话概括对方为什么打来。这两项即使槽位状态
   是“未确认”也可以填，同时在“未确认信息”里注明尚未确认。
3. “已确认事项”只写对方明确确认过的内容（槽位状态为“已确认”，或对话里对方明确认可）。
4. 对话记录里标了“{INTERRUPTED_MARK}”的 AI 句子不算已告知对方：其中重要的内容写入“未确认信息”，
   并说明 AI 没说完。
5. 槽位状态为“未确认”“已更正，待再确认”或空着的信息，写入“未确认信息”。
6. 对方提出的请求（送货、回拨、转告等）即使尚未确认也要写入“待办”，并注明是否已确认；
   对方是否要求回拨、号码是什么写入“回拨需求”。
7. “接管情况”：记录里显示发生过人工接管就写“已由机主接管”，否则写“无”。
8. 只输出这个 JSON 对象本身：不要代码围栏，不要解释文字。
"""


class SummaryError(RuntimeError):
    """摘要模型输出不可用（没有 JSON 对象）。"""


# --------------------------------------------------------------------------- #
# 对话还原
# --------------------------------------------------------------------------- #


@dataclass
class DialogueLine:
    """对话记录里的一行：说话人 + 内容 + 是否没说完（只对 AI 有意义）。"""

    speaker: str
    text: str
    interrupted: bool = False


def load_events(events: Iterable[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    """接受事件列表、`events.jsonl` 路径或通话目录，统一返回事件列表。"""
    if isinstance(events, (str, Path)):
        path = Path(events)
        if path.is_dir():
            return read_events(path)
        if path.name == "events.jsonl":
            return read_events(path.parent)
        raise FileNotFoundError(f"not an events.jsonl or call directory: {path}")
    return [dict(event) for event in events]


def dialogue_lines(events: Iterable[Mapping[str, Any]] | str | Path) -> list[DialogueLine]:
    """从事件流还原对话。

    AI 的一行在 `play_start` 时建立，`play_end` 时收尾；`interrupted`（轮次合并除外）
    把当前这句标成没说完。来电者的一行在 `asr_final` 时建立，并结束上一句 AI
    （正常情况下打断先记 `interrupted` 再进来电者文本，这里只是兜底防串行）。
    """
    lines: list[DialogueLine] = []
    open_ai: int | None = None

    for event in load_events(events):
        etype = event.get("type")
        data = event.get("data") or {}
        text = data.get("text")
        text = text.strip() if isinstance(text, str) else ""

        if etype == "asr_final":
            open_ai = None
            if text:
                lines.append(DialogueLine("来电者", text))
        elif etype == "play_start":
            if open_ai is not None and lines[open_ai].text != text:
                open_ai = None  # 上一句没等到 play_end 就换了内容：按已结束处理
            if text:
                lines.append(DialogueLine("AI", text))
                open_ai = len(lines) - 1
        elif etype == "play_end":
            if open_ai is not None and lines[open_ai].text == text:
                open_ai = None
        elif etype == "interrupted":
            if data.get("reason") == "turn_merged":
                continue  # 这一轮根本没播，不算 AI 说过话
            if open_ai is not None:
                lines[open_ai].interrupted = True
                open_ai = None
    return lines


def build_transcript(events: Iterable[Mapping[str, Any]] | str | Path) -> str:
    """把对话记录写成“来电者：…/AI：…”的多行文本；没说完的 AI 句子带标记。"""
    lines = dialogue_lines(events)
    if not lines:
        return "（无对话记录）"
    return "\n".join(
        f"{line.speaker}：{line.text}{INTERRUPTED_MARK if line.interrupted else ''}"
        for line in lines
    )


def has_takeover(events: Iterable[Mapping[str, Any]] | str | Path) -> bool:
    return any(event.get("type") == "takeover" for event in load_events(events))


# --------------------------------------------------------------------------- #
# 槽位 + 提示词
# --------------------------------------------------------------------------- #


def slot_items(
    slots: Mapping[str, Any] | None,
) -> dict[str, tuple[str | None, str]]:
    """把三种槽位形态统一成 `{name: (value, status)}`，只保留 5 个白名单槽位。"""
    if not isinstance(slots, Mapping):
        return {}
    nested = slots.get("slots")
    data: Mapping[str, Any] = nested if isinstance(nested, Mapping) else slots

    items: dict[str, tuple[str | None, str]] = {}
    for name in SLOT_NAMES:
        item = data.get(name)
        if item is None:
            continue
        if isinstance(item, SlotValue):
            items[name] = (item.value, item.status)
            continue
        if isinstance(item, Mapping):
            value = item.get("value")
            status = item.get("status") or "empty"
            items[name] = (value if isinstance(value, str) else None, str(status))
    return items


def format_slots(slots: Mapping[str, Any] | None) -> list[str]:
    """槽位记录写成提示词里的若干行；空槽位也写出来，摘要模型才知道“没问到”。"""
    items = slot_items(slots)
    if not items:
        return ["（没有槽位记录）"]
    lines: list[str] = []
    for name, (value, status) in items.items():
        label = _STATUS_LABELS.get(status, status)
        shown = value.strip() if isinstance(value, str) and value.strip() else "（空）"
        lines.append(f"{SLOT_LABELS[name]}（{name}）：{shown}（{label}）")
    return lines


def build_summary_messages(
    events: Iterable[Mapping[str, Any]] | str | Path,
    slots: Mapping[str, Any] | None,
    *,
    owner_name: str | None = None,
) -> list[dict[str, str]]:
    """构造摘要请求的 messages（system + user）。测试直接断言这里的内容。"""
    events_list = load_events(events)
    sections: list[str] = []
    if owner_name:
        sections.append(f"【机主】{owner_name}")
    sections.append("【槽位记录】")
    sections.extend(format_slots(slots))
    sections.append("【通话事件】")
    sections.append("- 人工接管：是" if has_takeover(events_list) else "- 人工接管：否")
    sections.append("【对话记录】")
    sections.append(build_transcript(events_list))
    sections.append("")
    sections.append("请按系统提示只输出 7 字段 JSON。")
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(sections)},
    ]


# --------------------------------------------------------------------------- #
# 摘要
# --------------------------------------------------------------------------- #


def _first_json_object(raw: str) -> dict[str, Any] | None:
    """取文本里第一个能解析成对象的 JSON（容忍代码围栏与前后解释文字）。

    与 `services/llm.py` 的同名私有函数思路一致；这里单独实现，避免 core 依赖
    services 的私有符号。
    """
    decoder = json.JSONDecoder()
    idx = raw.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            idx = raw.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            return obj
        idx = raw.find("{", idx + 1)
    return None


def _field_text(value: Any) -> str:
    """把模型给的字段值压成一个字符串：字符串原样、列表用分号连接、其它转字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Sequence):
        parts = [_field_text(item) for item in value]
        return "；".join(part for part in parts if part)
    return ""


def parse_summary(raw: str) -> dict[str, str]:
    """把模型输出解析成七字段 dict；缺的字段补空字符串，非字符串值尽量救回。

    找不到 JSON 对象时抛 `SummaryError`（调用方可以决定是否重试或退回规则摘要）。
    """
    obj = _first_json_object(raw or "")
    if obj is None:
        raise SummaryError(f"summary model returned no JSON object: {raw[:200]!r}")
    return {field: _field_text(obj.get(field)) for field in SUMMARY_FIELDS}


def merge_unconfirmed(summary: dict[str, str], notes: Sequence[str] | None) -> dict[str, str]:
    """把复核（`recheck_slots`）的不一致条目追加进“未确认信息”。

    空条目被忽略；原有内容保留在前面，用中文分号连接。
    """
    extra = [note.strip() for note in (notes or []) if isinstance(note, str) and note.strip()]
    if not extra:
        return summary
    existing = summary.get("未确认信息", "").strip()
    appended = "；".join(extra)
    summary["未确认信息"] = f"{existing}；{appended}" if existing else appended
    # 返回同一对象，方便链式使用。
    return summary


def write_summary(call_dir: str | Path, summary: Mapping[str, Any]) -> Path:
    """把摘要写成 `<call_dir>/summary.json`（UTF-8，保留中文）。"""
    payload = {field: _field_text(summary.get(field)) for field in SUMMARY_FIELDS}
    path = Path(call_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _call_dir_of(events: Iterable[Mapping[str, Any]] | str | Path) -> Path | None:
    if not isinstance(events, (str, Path)):
        return None
    path = Path(events)
    return path.parent if path.name == "events.jsonl" else path


async def summarize(
    events: Iterable[Mapping[str, Any]] | str | Path,
    slots: Mapping[str, Any] | None,
    llm: Any,
    *,
    call_dir: str | Path | None = None,
    owner_name: str | None = None,
    recheck_notes: Sequence[str] | None = None,
) -> dict[str, str]:
    """调用摘要模型，返回七字段 JSON，并写入 `summary.json`。

    - `events`：事件列表、`events.jsonl` 路径或通话目录；
    - `slots`：槽位快照（三种形态都接受，见 `slot_items`）；
    - `llm`：需要 `async complete(messages)`（`LlmClient` 默认用 `summary_model`）；
    - `call_dir`：摘要写入目录；省略时若 `events` 是路径则用它的通话目录；
    - `recheck_notes`：`recheck_slots()` 的不一致条目，追加到“未确认信息”。
    """
    events_list = load_events(events)
    messages = build_summary_messages(events_list, slots, owner_name=owner_name)
    raw = await llm.complete(messages)
    payload = parse_summary(raw)
    merge_unconfirmed(payload, recheck_notes)
    target = Path(call_dir) if call_dir is not None else _call_dir_of(events)
    if target is not None:
        write_summary(target, payload)
    return payload


# --------------------------------------------------------------------------- #
# 校正：用另一个 ASR 后端重转写整通录音
# --------------------------------------------------------------------------- #


@dataclass
class RecheckResult:
    """复核结果：整通转写、写进“未确认信息”的条目、以及各槽位在复核里的取值。"""

    transcript: str
    mismatches: list[str] = field(default_factory=list)
    values: dict[str, str | None] = field(default_factory=dict)

    @property
    def consistent(self) -> bool:
        return not self.mismatches


def _strip_punct(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in normalized if not ch.isspace() and ch not in "，。！？、,.!?;；:：\"'“”‘’")


def _digits(value: str | None) -> str | None:
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits or None


async def recheck_slots(
    caller_wav: str | Path | bytes,
    slots: Mapping[str, Any] | None,
    asr_alt: Any,
    *,
    allow_landline: bool = True,
) -> RecheckResult:
    """用另一个 ASR 后端重转写 `caller.wav`，比对姓名与回拨号码。

    `caller_wav` 可以是 wav 路径，也可以直接是 16kHz 单声道 16-bit PCM 字节。
    `asr_alt` 需要 `async transcribe(pcm) -> AsrResult`（用 `asr.recheck_model` 构造的
    `AsrClient`，时限用 `asr.recheck_*` 那组更宽松的值，见 config.yaml）。
    比对结果只作为“未确认信息”返回，**不会修改传入的槽位**。
    """
    if isinstance(caller_wav, (bytes, bytearray)):
        pcm = bytes(caller_wav)
    else:
        from services.asr import load_wav_pcm

        pcm = load_wav_pcm(caller_wav)

    result = await asr_alt.transcribe(pcm)
    transcript = (getattr(result, "text", "") or "").strip()

    items = slot_items(slots)
    values: dict[str, str | None] = {}
    mismatches: list[str] = []

    name = items.get("name", (None, "empty"))[0]
    name = name.strip() if isinstance(name, str) else ""
    if name:
        found_name = name if _strip_punct(name) in _strip_punct(transcript) else None
        values["name"] = found_name
        if found_name is None:
            mismatches.append(f"复核转写里没有出现姓名“{name}”，请人工核对")

    recorded = _digits(items.get("callback_number", (None, "empty"))[0])
    found = extract_phone(transcript, allow_landline=allow_landline)
    values["callback_number"] = found
    if recorded and found and recorded != found:
        mismatches.append(f"复核转写里的回拨号码是 {found}，与记录的 {recorded} 不一致")
    elif recorded and not found:
        mismatches.append(f"复核转写里找不到回拨号码，记录的是 {recorded}")
    elif not recorded and found:
        mismatches.append(f"复核转写里出现了回拨号码 {found}，但槽位没有记录")

    return RecheckResult(transcript=transcript, mismatches=mismatches, values=values)


async def summarize_call(
    call_dir: str | Path,
    llm: Any,
    *,
    asr_alt: Any | None = None,
    owner_name: str | None = None,
    allow_landline: bool = True,
) -> dict[str, str]:
    """读一通电话的 `events.jsonl` / `slots.json`（可选 `caller.wav`），写 `summary.json`。

    `asr_alt` 非空时先做复核，把不一致写入“未确认信息”；没有 `caller.wav` 则跳过复核。
    """
    call_dir = Path(call_dir)
    events = read_events(call_dir)

    slots_path = call_dir / "slots.json"
    slots: Mapping[str, Any] = {}
    if slots_path.exists():
        slots = json.loads(slots_path.read_text(encoding="utf-8"))

    notes: list[str] = []
    caller_wav = call_dir / "caller.wav"
    if asr_alt is not None and caller_wav.exists():
        recheck = await recheck_slots(caller_wav, slots, asr_alt, allow_landline=allow_landline)
        notes = recheck.mismatches

    return await summarize(
        events,
        slots,
        llm,
        call_dir=call_dir,
        owner_name=owner_name,
        recheck_notes=notes,
    )

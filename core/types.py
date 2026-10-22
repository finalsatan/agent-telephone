"""跨任务共享的数据类型。

这些类型是 core / services / tools / adapters 之间的约定（见《开发任务拆分》0.2），
改动前需先在 docs/progress.md 记录原因，并同步更新所有调用方与测试（AGENTS.md 规则 2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# 槽位名固定为下列 5 个：前三个必填，后两个选填。
SLOT_NAMES: tuple[str, ...] = ("name", "purpose", "callback_number", "time", "location")
REQUIRED_SLOTS: tuple[str, ...] = ("name", "purpose", "callback_number")
OPTIONAL_SLOTS: tuple[str, ...] = ("time", "location")

# SlotValue.status 的合法取值。
SLOT_STATUSES: tuple[str, ...] = ("empty", "unconfirmed", "confirmed", "corrected")


@dataclass
class AsrResult:
    """一次整段 ASR 的结果。audio_ms 为送识别音频的时长，用于换算实时率。"""

    text: str
    latency_ms: float
    audio_ms: float


@dataclass
class TurnResult:
    """LLM 一轮的输出。parse_ok=False 时 reply 为空字符串，由调用方用兜底话术。

    first_sentence_ms 为流式链路中第一个完整短句的产出时刻（T02b）；
    非流式或没产出任何句子时为 None。
    raw 为模型原始输出（T12b：解析失败时由 core.call 截前 300 字写进 error 事件）。
    """

    reply: str
    slots_update: dict[str, str]
    first_token_ms: float
    total_ms: float
    parse_ok: bool = True
    first_sentence_ms: float | None = None
    raw: str = ""


@dataclass
class ReplyChunk:
    """turn_stream() 的产出（T02b）。

    kind="sentence" 时 text 为一个完整短句，调用方可立即送 TTS；
    kind="done" 时 result 为整轮最终结果，text 为空。
    """

    kind: str
    text: str = ""
    result: TurnResult | None = None
    t_ms: float = 0.0


@dataclass
class SlotValue:
    """单个槽位的值与确认状态。status 取值见 SLOT_STATUSES。"""

    value: str | None
    status: str = "empty"

    def __post_init__(self) -> None:
        if self.status not in SLOT_STATUSES:
            raise ValueError(f"invalid slot status: {self.status!r}, expected one of {SLOT_STATUSES}")


class State(str, Enum):
    """通话状态机状态。流转规则由 core/dialog.py（T07）实现。"""

    GREETING = "GREETING"
    ASK_IDENTITY = "ASK_IDENTITY"
    ASK_PURPOSE = "ASK_PURPOSE"
    FILL_MISSING = "FILL_MISSING"
    CONFIRM = "CONFIRM"
    CLOSING = "CLOSING"

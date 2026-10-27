"""对话历史（`list[{"role","content"}]`）的共用操作。

T10 的 `CallSession`、T15 的 `DialogProcessor`、T16 的 LiveKit `CallAgent` 三处都要做
同样几件事：记一条、只留最近 N 条、把最后一条 assistant 标成“（被对方打断）”、
把从未播出的那句从历史里去掉。T16 起抽到这里，避免三条链路各写一份、行为分叉。

`history` 的格式与 `core/call.py` 一致：`[{"role": "user"|"assistant", "content": str}]`，
只保留最近 `DEFAULT_HISTORY_MESSAGES` 条。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# 设计文档 0.5：history 只保留最近 8 轮（16 条）。
DEFAULT_HISTORY_MESSAGES = 16

# 被抢话/接管的 assistant 消息在历史里的后缀。
INTERRUPTED_SUFFIX = "（被对方打断）"


def trim(history: list[dict[str, str]], limit: int = DEFAULT_HISTORY_MESSAGES) -> None:
    """原地裁到最近 `limit` 条。"""
    if limit > 0 and len(history) > limit:
        del history[: len(history) - limit]


def record_user(
    history: list[dict[str, str]], text: str, *, limit: int = DEFAULT_HISTORY_MESSAGES
) -> None:
    if not (text or "").strip():
        return
    history.append({"role": "user", "content": text})
    trim(history, limit)


def record_assistant(
    history: list[dict[str, str]], text: str, *, limit: int = DEFAULT_HISTORY_MESSAGES
) -> None:
    if not (text or "").strip():
        return
    history.append({"role": "assistant", "content": text})
    trim(history, limit)


def replace_last_assistant(history: list[dict[str, str]], text: str) -> None:
    """把最后一条 assistant 消息换成实际播出的内容。"""
    if not history or history[-1].get("role") != "assistant":
        return
    history[-1]["content"] = text


def drop_unplayed_assistant(history: list[dict[str, str]], text: str) -> str:
    """从历史里去掉从未播出的 `text`（T15d）。

    `text` 是这一轮要说的内容；它可能已经被整条记进最后一条 assistant 消息。
    返回实际保留下来的内容：前一句（`confirm_lead`）已播时只保留它并标“（被对方打断）”；
    一句都没播就删掉这条 assistant 消息。
    """
    if not history or history[-1].get("role") != "assistant":
        return ""
    content = str(history[-1].get("content") or "")
    index = content.rfind(text) if text else -1
    if index < 0:
        return content
    played = content[:index]
    if played:
        history[-1]["content"] = played + INTERRUPTED_SUFFIX
    else:
        history.pop()
    return played


def mark_last_assistant_interrupted(
    history: list[dict[str, str]],
    *,
    interrupted_text: str | None = None,
    sentences: Iterable[str] | None = None,
) -> None:
    """把最后一条 assistant 消息标成“（被对方打断）”（T13b / T15b 的尾音抢话）。

    `interrupted_text` 给出“被打断的那一句”，则只保留到那一句为止；
    `sentences` 是这一轮实际播放的句子序列（用于按句边界截断）。
    """
    if not history or history[-1].get("role") != "assistant":
        return
    content = str(history[-1].get("content") or "")
    if not content or content.endswith(INTERRUPTED_SUFFIX):
        return
    if interrupted_text:
        cut = interrupted_cut(content, interrupted_text, sentences=sentences)
        if cut is not None:
            content = content[:cut]
    history[-1]["content"] = content + INTERRUPTED_SUFFIX


def interrupted_cut(
    content: str, interrupted_text: str, *, sentences: Iterable[str] | None = None
) -> int | None:
    """返回历史内容里“被打断那句”的结束位置（与 T14b 一致）。"""
    if sentences:
        joined = "".join(sentences)
        if joined == content:
            position = 0
            for sentence in sentences:
                position += len(sentence)
                if sentence == interrupted_text:
                    return position
    index = content.rfind(interrupted_text)
    return index + len(interrupted_text) if index >= 0 else None


def as_messages(history: list[dict[str, str]], limit: int = DEFAULT_HISTORY_MESSAGES) -> list[Any]:
    """给 LLM 的历史视图（当前就是同一份 list 的浅拷贝）。"""
    return [dict(item) for item in history[-limit:]]

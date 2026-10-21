"""事件记录器：每通电话的唯一数据源。

格式见《开发任务拆分》0.4，每行一个 JSON：

    {"call_id": "...", "t": 12345.6, "type": "asr_final", "data": {...}}

t 为 time.monotonic()*1000（毫秒），写入后立即 flush，进程被强杀也不丢已写事件。
时延指标一律从这里离线计算，运行时不另埋点（设计文档第 7 节）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

# 事件类型白名单，写入白名单外的事件会抛 ValueError（拼错类型名会立刻暴露）。
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "vad_start",
        "vad_end",
        "turn_end",
        "asr_request",
        "asr_final",
        "llm_request",
        "llm_first_token",
        "llm_done",
        "tts_request",
        "tts_first_audio",
        "play_start",
        "play_end",
        "interrupted",
        "state_change",
        "slot_update",
        "takeover",
        "error",
        "call_start",
        "call_end",
    }
)


def monotonic_ms() -> float:
    """当前的单调时钟毫秒数。"""
    return time.monotonic() * 1000.0


class EventLogger:
    """把事件追加写入 <call_dir>/events.jsonl。call_id 取目录名。

    clock 仅供测试注入假时钟；默认使用 monotonic_ms。
    即使传入的时钟回退，写入的 t 也保证单调不减。
    """

    def __init__(self, call_dir: str | Path, *, clock: Callable[[], float] | None = None) -> None:
        self.call_dir = Path(call_dir)
        self.call_id = self.call_dir.name
        self.call_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.call_dir / "events.jsonl"
        self._clock = clock or monotonic_ms
        self._last_t = 0.0
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, type: str, **data: Any) -> dict[str, Any]:
        """追加一条事件并立即 flush，返回写入的那条记录。"""
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {type!r}")

        t = self._clock()
        if t < self._last_t:
            t = self._last_t
        self._last_t = t

        event = {"call_id": self.call_id, "t": t, "type": type, "data": data}
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()
        return event

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_events(call_dir: str | Path) -> list[dict[str, Any]]:
    """读回 <call_dir>/events.jsonl 的全部事件（按写入顺序）。"""
    path = Path(call_dir) / "events.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

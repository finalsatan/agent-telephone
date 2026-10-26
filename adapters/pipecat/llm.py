"""T15：对话（LLM + 状态机）的 Pipecat 适配层。

Pipecat 自带的 LLM 服务是 OpenAI 风格的“把上下文发给模型、把流式文本吐出来”，而本项目的
LLM 输出是固定 JSON（`{"reply": ..., "slots": ...}`），流程走向由 `core.dialog.Dialog`
用代码决定，`reply` 也大多被模板话术取代（T07c）。因此这里不实现 `LLMService`，而是一个
自定义 `FrameProcessor`：

    TranscriptionFrame（ASR 产出）
        → 攒起来
    UserStoppedSpeakingFrame（Smart Turn 判为说完）
        → 调 LLM 流式接口拿 slots_update → 交给 Dialog → 选一句要播的话
        → 产出 TTSSpeakFrame 给下游的 TTSService
    InterruptionFrame（对方抢话 / 接管）
        → 取消正在跑的 LLM 任务、把最后一条 assistant 历史标“（被打断）”

事件与 T10 的 `core.call.CallSession` 保持同一套类型与字段（`llm_request` /
`llm_first_token` / `llm_done` / `state_change` / `slot_update` / `interrupted`），
这样 `tools/evaluate.py` 的 `extract_turn_metrics()` 能直接读。

**与离线回放的差异**：T10 的“轮次合并”“号码没报完等下一轮”依赖音频时间轴上的
`end_ms` / `next_start_ms`，实时链路里没有这两个量，所以这里不做合并——Smart Turn
已经把“半句话”挡在外面（`UserStoppedSpeakingFrame` 只在语义完整时才发）。
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    DataFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from core.events import EventLogger, monotonic_ms
from core.types import SLOT_NAMES, State
from services.llm import LlmError, LlmTimeout, split_reply_sentences

# 与 core.call.CallSession 一致：history 只保留最近 8 轮（16 条）。
DEFAULT_HISTORY_MESSAGES = 16
# 被抢话的 assistant 消息在历史里的后缀。
INTERRUPTED_SUFFIX = "（被对方打断）"


@dataclass
class TakeoverFrame(DataFrame):
    """用户接管（网页点“接管”）：对话处理器停止再回应，但音频链路继续走。"""

    pass


class DialogProcessor(FrameProcessor):
    """ASR 文本 → LLM + Dialog → TTSSpeakFrame。"""

    def __init__(
        self,
        llm: Any,
        dialog: Any,
        events: EventLogger,
        *,
        history_messages: int = DEFAULT_HISTORY_MESSAGES,
        on_reply: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._dialog = dialog
        self._events = events
        self._history_messages = int(history_messages)
        # 通知外部（PipecatCall）“这一轮要播这句”，用于抢话时判断是否在说。
        self._on_reply = on_reply

        self._history: list[dict[str, str]] = []
        self._user_text: list[str] = []
        self._llm_cancel = asyncio.Event()
        self._turn_task: asyncio.Task | None = None
        self._takeover = False
        self._started = False
        # 轮次已结束（收到 UserStoppedSpeakingFrame）但转写还没到：等 TranscriptionFrame
        # 到了再起 LLM 任务。SegmentedSTTService 在后台任务里转写，转写帧可能比轮次结束
        # 帧晚到几十毫秒（实测过 asr_final 在 turn_end 之后），不这样处理会漏掉一轮。
        self._turn_pending = False
        # 已经推出、可能还在播的回复；TTSService 播完会回调 `on_play_end()`。
        self._playing_text: str | None = None
        self._last_sentences: list[str] = []

    # ------------------------------------------------------------------ #
    # 供 TTSService / PipecatCall 调用
    # ------------------------------------------------------------------ #

    def on_play_end(self, text: str) -> None:
        """TTSService 播完一句时回调（自然播完，不是被打断）。"""
        if text == self._playing_text:
            self._playing_text = None

    def takeover(self) -> None:
        self._takeover = True
        self._cancel_turn()

    @property
    def dialog(self) -> Any:
        return self._dialog

    # ------------------------------------------------------------------ #
    # FrameProcessor
    # ------------------------------------------------------------------ #

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            if not self._takeover:
                self._user_text.append(frame.text)
                if self._turn_pending:
                    self._schedule_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_pending = True
            self._schedule_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._handle_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TakeoverFrame):
            self.takeover()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            self._cancel_turn()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        self._cancel_turn()
        await super().cleanup()

    # ------------------------------------------------------------------ #
    # 问候与一轮
    # ------------------------------------------------------------------ #

    async def _start(self) -> None:
        if self._started:
            return
        self._started = True
        before = self._dialog.state
        hint = self._dialog.greeting()
        self._log_state_change(before, self._dialog.state)
        if hint.text:
            self._record_assistant(hint.text)
            await self.push_frame(TTSSpeakFrame(text=hint.text, append_to_context=False))

    def _schedule_turn(self) -> None:
        if self._takeover:
            return
        if not "".join(self._user_text).strip():
            return
        if self._turn_task is not None and not self._turn_task.done():
            return
        self._turn_pending = False
        self._turn_task = self.create_task(self._run_turn())

    async def _run_turn(self) -> None:
        text = "".join(self._user_text).strip()
        self._user_text.clear()
        if not text:
            return

        before = copy.deepcopy(self._dialog)
        state_before = before.state
        attempts = before.attempts
        self._events.log("llm_request", state=state_before.value, text=text, attempts=attempts)

        sentences: list[str] = []
        result = None
        first_logged = False
        llm_error = False
        self._llm_cancel = asyncio.Event()
        try:
            async for chunk in self._llm.turn_stream(
                state_before, before.slots, list(self._history), text, attempts
            ):
                if self._llm_cancel.is_set():
                    break
                if chunk.kind == "sentence":
                    if not first_logged:
                        self._events.log("llm_first_token", latency_ms=chunk.t_ms)
                        first_logged = True
                    sentences.append(chunk.text)
                elif chunk.kind == "done":
                    result = chunk.result
                    if not first_logged and result is not None:
                        self._events.log("llm_first_token", latency_ms=result.first_token_ms or 0.0)
                    self._events.log(
                        "llm_done",
                        first_token_ms=result.first_token_ms if result else None,
                        total_ms=result.total_ms if result else None,
                        first_sentence_ms=result.first_sentence_ms if result else None,
                        parse_ok=result.parse_ok if result else False,
                        reply=result.reply if result else "",
                        slots=result.slots_update if result else {},
                    )
        except LlmTimeout as exc:
            llm_error = True
            self._events.log("error", stage="llm", reason="timeout", message=str(exc)[:200])
        except LlmError as exc:
            llm_error = True
            self._events.log("error", stage="llm", reason=type(exc).__name__, message=str(exc)[:200])

        if self._llm_cancel.is_set():
            # 还没回复就被抢话：文本顺延到下一轮（与 T10 的轮次合并同义）。
            self._user_text.insert(0, text)
            self._events.log("interrupted", reason="barge_in", stage="llm", text=text)
            return

        parse_ok = bool(result and result.parse_ok and not llm_error)
        slots_update = result.slots_update if (result and not llm_error) else {}
        state_after, hint = self._dialog.on_user_turn(text, slots_update, parse_ok)
        if result is not None and result.parse_ok is False:
            llm_error = True
            self._events.log(
                "error", stage="llm", reason="parse", reply=result.reply[:200], raw=result.raw[:300]
            )
        if result is None and not llm_error:
            llm_error = True
            self._events.log("error", stage="llm", reason="no_done_chunk")

        self._log_state_change(before.state, state_after)
        self._log_slot_updates(before.slots, self._dialog.slots)

        reply_text = self._reply_text(hint, sentences)
        self._record_user(text)
        if reply_text:
            self._record_assistant(reply_text)
            self._playing_text = reply_text
            if self._on_reply is not None:
                self._on_reply(reply_text)
            await self.push_frame(TTSSpeakFrame(text=reply_text, append_to_context=False))

    def _reply_text(self, hint: Any, sentences: list[str]) -> str:
        """按 `ReplyHint.kind` 选要播的文本（与 T10 的 `_play_hint` 分流一致）。"""
        if hint.use_cache or hint.use_text:
            return hint.text
        if getattr(hint, "kind", "") == "llm":
            return "".join(sentences) or split_reply_sentences(getattr(hint, "text", ""))
        return getattr(hint, "text", "") or ""

    # ------------------------------------------------------------------ #
    # 打断
    # ------------------------------------------------------------------ #

    def _handle_interruption(self) -> None:
        self._cancel_turn()
        if self._playing_text is None:
            return
        self._events.log(
            "interrupted", reason="barge_in", text=self._playing_text, tail=True, unplayed=[]
        )
        self._mark_last_assistant_interrupted()
        self._playing_text = None

    def _cancel_turn(self) -> None:
        self._llm_cancel.set()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._turn_task = None

    def _mark_last_assistant_interrupted(self) -> None:
        if not self._history:
            return
        last = self._history[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content") or ""
        if content and not content.endswith(INTERRUPTED_SUFFIX):
            last["content"] = content + INTERRUPTED_SUFFIX

    # ------------------------------------------------------------------ #
    # 事件 / 历史
    # ------------------------------------------------------------------ #

    def _log_state_change(self, before: State, after: State) -> None:
        if before is not after:
            self._events.log("state_change", **{"from": before.value, "to": after.value})

    def _log_slot_updates(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
        for name in SLOT_NAMES:
            old = before.get(name)
            new = after.get(name)
            if old is None or new is None:
                continue
            if old.value == new.value and old.status == new.status:
                continue
            self._events.log("slot_update", slot=name, value=new.value, status=new.status)

    def _record_user(self, text: str) -> None:
        if text:
            self._history.append({"role": "user", "content": text})
            self._trim_history()

    def _record_assistant(self, text: str) -> None:
        if text:
            self._last_sentences = [text]
            self._history.append({"role": "assistant", "content": text})
            self._trim_history()

    def _trim_history(self) -> None:
        if len(self._history) > self._history_messages:
            self._history = self._history[-self._history_messages :]

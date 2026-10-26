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
        → 取消正在跑的 LLM 任务、把还没处理的文本顺延到下一轮

T15b（第 26 次评审）起，一轮的“怎么回复”不再自己写一份，而是复用 `core.call` 里的
`stream_llm_turn()` / `choose_reply_text()` / `is_partial_phone()` / `template_hint()`，
和 `CallSession` 保持同一套判断：号码没报完按超时处理（实时链路没有音频时间轴，与 T13
一致）、ASR 出错或转写为空播兜底话术、CONFIRM 合成失败退回号码片段拼接、`tts.confirm_lead`。

事件与 T10 的 `core.call.CallSession` 保持同一套类型与字段（`llm_request` /
`llm_first_token` / `llm_done` / `state_change` / `slot_update` / `interrupted`），
这样 `tools/evaluate.py` 的 `extract_turn_metrics()` 能直接读。

**打断的 `interrupted` 事件由谁写**：AI 正在播（浏览器队列里还有尾音）时的抢话由
`PipecatCall` 的 `PlaybackTracker` 统一写 `interrupted(tail=True, unplayed=...)`；
LLM 还在跑、还没播时的抢话由这里写 `interrupted(stage="llm")`。两边互斥，不会重复。
"""

from __future__ import annotations

import asyncio
import copy
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
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

from core.call import (
    INTERRUPTED_SUFFIX,
    choose_reply_text,
    is_partial_phone,
    stream_llm_turn,
    template_hint,
)
from core.dialog import split_confirm_lead
from core.events import EventLogger
from core.normalize import phone_to_spoken
from core.phrases import PHRASES
from core.types import SLOT_NAMES, State

# 与 core.call.CallSession 一致：history 只保留最近 8 轮（16 条）。
DEFAULT_HISTORY_MESSAGES = 16


@dataclass
class TakeoverFrame(DataFrame):
    """用户接管（网页点“接管”）：对话处理器停止再回应，但音频链路继续走。"""

    pass


@dataclass
class SpeakHintFrame(TTSSpeakFrame):
    """带号码提示的 TTS 帧：整句确认语合成失败时，TTSService 可退回号码片段拼接。"""

    phone: str | None = None


@dataclass
class TurnDoneFrame(DataFrame):
    """排在本轮全部 Speak 帧之后，经 TTS 音频队列发送到 sink。"""

    interruptible: bool = field(default=False, init=False)
    kind: str = "reply"
    text: str = ""


class DialogProcessor(FrameProcessor):
    """ASR 文本 → LLM + Dialog → TTSSpeakFrame。"""

    def __init__(
        self,
        llm: Any,
        dialog: Any,
        events: EventLogger,
        *,
        cache: Any | None = None,
        config: Mapping[str, Any] | None = None,
        history_messages: int = DEFAULT_HISTORY_MESSAGES,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._dialog = dialog
        self._events = events
        self._cache = cache
        self._history_messages = int(history_messages)
        self._confirm_lead = bool((((config or {}).get("tts") or {}).get("confirm_lead", False)))

        self._history: list[dict[str, str]] = []
        self._last_assistant_sentences: list[str] = []
        self._user_text: list[str] = []
        self._llm_cancel = asyncio.Event()
        self._turn_task: asyncio.Task | None = None
        self._takeover = False
        self._started = False
        # 轮次已结束（收到 UserStoppedSpeakingFrame）但转写还没到：等 TranscriptionFrame
        # 到了再起 LLM 任务。SegmentedSTTService 在后台任务里转写，转写帧可能比轮次结束
        # 帧晚到几十毫秒（实测过 asr_final 在 turn_end 之后），不这样处理会漏掉一轮。
        self._turn_pending = False
        # ASR 出错或转写为空：没有 TranscriptionFrame，靠 STT 的 AsrFailedFrame 触发兜底。
        self._asr_failed = False
        # 上一轮被抢话后顺延的文本：下一轮开始前要等新转写一起处理（否则会把顺延文本
        # 单独回复，把新说的话挤到再下一轮）。
        self._segments: list[int] = []
        self._segment_results: dict[int, Any] = {}
        self._pending_turns: deque[tuple[int, ...]] = deque()
        self._segmented = False
        self._carried_text = ""
        self._turn_cancels: set[asyncio.Event] = set()
        # 正在等 TTS 合成结果的 CONFIRM 句（合成失败时要改历史、退回号码拼接）。
        self._confirm_pending: dict[str, Any] | None = None
        # 已经成功说出的轮数（PipecatCall 写 call_end / takeover 用）。
        self.turn_count = 0

    # ------------------------------------------------------------------ #
    # 供 TTSService / PipecatCall 调用
    # ------------------------------------------------------------------ #

    def takeover(self) -> None:
        self._takeover = True
        self._cancel_turn()

    @property
    def dialog(self) -> Any:
        return self._dialog

    def mark_playing_interrupted(self, interrupted_text: str | None = None) -> None:
        """尾音抢话：把历史里最后一条 assistant 消息标成“（被对方打断）”（与 T13b 一致）。"""
        if not self._history:
            return
        last = self._history[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content") or ""
        if not content or content.endswith(INTERRUPTED_SUFFIX):
            return
        if interrupted_text:
            index = self._interrupted_cut(content, interrupted_text)
            if index is not None:
                content = content[:index]
        last["content"] = content + INTERRUPTED_SUFFIX

    def on_tts_fallback(self, text: str) -> None:
        """TTSService 用号码片段兜底播了 `text`：把历史里的确认语换成实际播出的内容。"""
        pending = self._confirm_pending
        self._confirm_pending = None
        if not pending:
            return
        self._replace_last_assistant(str(pending.get("lead") or "") + text)

    # ------------------------------------------------------------------ #
    # FrameProcessor
    # ------------------------------------------------------------------ #

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        from adapters.pipecat.stt import AsrSegmentStartedFrame, AsrSegmentFinishedFrame

        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, AsrSegmentStartedFrame):
            self._segmented = True
            self._segments.append(frame.segment_id)
        elif isinstance(frame, AsrSegmentFinishedFrame):
            self._segment_results[frame.segment_id] = frame
            self._schedule_turn()
        elif isinstance(frame, TranscriptionFrame):
            if not self._takeover and not self._segmented:
                self._user_text.append(frame.text)
                self._schedule_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._segmented:
                self._pending_turns.append(tuple(self._segments))
                self._segments.clear()
            self._turn_pending = True
            self._schedule_turn()
            await self.push_frame(frame, direction)
        elif _is_asr_failed(frame):
            if not self._takeover and not self._segmented:
                self._asr_failed = True
                self._schedule_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._cancel_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TakeoverFrame):
            self.takeover()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            self.takeover()
            if self._turn_task is not None and not self._turn_task.done():
                await self.cancel_task(self._turn_task)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        self._cancel_turn()
        if self._turn_task is not None and not self._turn_task.done():
            await self.cancel_task(self._turn_task)
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

    @property
    def busy(self) -> bool:
        return bool(self._pending_turns or self._turn_pending or
                    (self._turn_task is not None and not self._turn_task.done()))

    def _schedule_turn(self) -> None:
        if not self._turn_pending:
            return
        if self._segmented:
            while self._pending_turns:
                ids = self._pending_turns[0]
                if any(i not in self._segment_results for i in ids):
                    return
                self._pending_turns.popleft()
                results = [self._segment_results.pop(i) for i in ids]
                text = "".join(r.text for r in results).strip()
                self._queue_turn(text, any(r.failed for r in results))
            self._turn_pending = False
        elif self._user_text or self._asr_failed:
            text = "".join(self._user_text).strip()
            self._user_text.clear()
            failed, self._asr_failed = self._asr_failed, False
            self._turn_pending = False
            self._queue_turn(text, failed)

    def _queue_turn(self, text: str, failed: bool) -> None:
        previous = self._turn_task
        cancel = asyncio.Event()
        self._llm_cancel = cancel
        self._turn_cancels.add(cancel)
        self._turn_task = self.create_task(self._run_turn_after(previous, text, failed, cancel))

    async def _run_turn_after(
        self, previous: asyncio.Task | None, text: str, failed: bool, cancel: asyncio.Event
    ) -> None:
        try:
            if previous is not None:
                try:
                    await previous
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if cancel.is_set() or (current is not None and current.cancelling()):
                        raise
                    self._events.log(
                        "error", stage="turn", reason="previous_turn_cancelled"
                    )
                except Exception as exc:  # noqa: BLE001 - 一轮失败不能阻塞后续轮
                    self._events.log(
                        "error",
                        stage="turn",
                        reason=type(exc).__name__,
                        message=str(exc)[:200],
                    )
            try:
                await self._run_turn(text, failed, cancel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 记录本轮错误，允许下一轮继续
                self._events.log(
                    "error",
                    stage="turn",
                    reason=type(exc).__name__,
                    message=str(exc)[:200],
                )
        finally:
            self._turn_cancels.discard(cancel)

    async def _run_turn(
        self, text: str | None = None, asr_failed: bool | None = None,
        cancel: asyncio.Event | None = None,
    ) -> None:
        legacy_direct_call = text is None
        if text is None:  # 保留旧的直接调用单测；管线始终传入冻结的本轮输入。
            text = "".join(self._user_text).strip()
            self._user_text.clear()
        if asr_failed is None:
            asr_failed, self._asr_failed = self._asr_failed, False
        if cancel is None:
            cancel = self._llm_cancel = asyncio.Event()
        text = self._carried_text + text
        self._carried_text = ""
        if self._takeover:
            await self.push_frame(TurnDoneFrame(kind="skipped", text=text))
            return

        if not text:
            if asr_failed:
                # ASR 出错 / 转写为空：播兜底话术（与 CallSession._reply_with_fallback 一致）。
                self._events.log("error", stage="asr", reason="empty_transcript")
                await self._reply_with_fallback("")
            else:
                await self.push_frame(TurnDoneFrame(kind="skipped"))
            return

        before = copy.deepcopy(self._dialog)
        state_before = before.state

        # 追问号码时，末尾数字串可能只是号码的一半。实时链路没有音频时间轴，
        # 按“超时”处理（与 T13 的 RealtimeCall 一致）：不计入座机、计一次追问失败。
        partial_phone = self._dialog.asked_slot == "callback_number" and is_partial_phone(text)
        dialog_text = "" if partial_phone else None
        if partial_phone:
            self._events.log("error", stage="asr", reason="incomplete_phone_timeout", text=text)

        sentences, result, llm_error = await stream_llm_turn(
            self._llm,
            self._events,
            state_before,
            before.slots,
            self._history,
            text,
            before.attempts,
            cancel,
        )

        if cancel.is_set():
            # 还没回复就被抢话：文本顺延到下一轮（与 T10 的轮次合并同义）。
            # LLM 尚未提交任何状态变更，不需要替换或回滚共享 Dialog 对象。
            self._carried_text = text
            if legacy_direct_call:
                # 纯逻辑兼容：旧测试直接读取 _user_text；管线路径统一走 _carried_text。
                self._user_text.insert(0, text)
                self._carried_text = ""
            self._events.log("interrupted", reason="barge_in", stage="llm", text=text)
            await self.push_frame(TurnDoneFrame(kind="interrupted", text=text))
            return

        parse_ok = bool(result and result.parse_ok and not llm_error)
        slots_update = result.slots_update if (result and not llm_error) else {}
        state_after, hint = self._dialog.on_user_turn(
            text if dialog_text is None else dialog_text, slots_update, parse_ok
        )
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

        reply_text = choose_reply_text(hint, sentences)
        self._record_user(text)
        self.turn_count += 1
        if reply_text:
            played = await self._emit_reply(hint, reply_text)
            self._record_assistant(played or reply_text)
        await self.push_frame(TurnDoneFrame(kind="reply" if reply_text else "skipped", text=text))

    async def _reply_with_fallback(self, text: str) -> None:
        """ASR 失败 / 空转写的兜底：播 `repeat`（与 CallSession._reply_with_fallback 一致）。"""
        hint = template_hint(self._dialog, "repeat")
        self.turn_count += 1
        if text.strip():
            self._record_user(text)
        await self._emit_reply(hint, hint.text)
        self._record_assistant(hint.text)
        await self.push_frame(TurnDoneFrame(text=text))

    async def _emit_reply(self, hint: Any, reply_text: str) -> str:
        """把要播的文本推给下游 TTS，返回实际播出的文本（`confirm_lead` 拆句后）。"""
        played = reply_text
        lead = ""
        if hint.use_text and self._confirm_lead:
            lead, rest = split_confirm_lead(reply_text)
            if lead:
                self._confirm_pending = None
                await self.push_frame(TTSSpeakFrame(text=lead, append_to_context=False))
                reply_text = rest
                played = lead + rest
        phone = getattr(hint, "phone", None) if hint.use_text else None
        self._confirm_pending = (
            {"lead": lead, "text": reply_text, "phone": phone} if phone else None
        )
        await self.push_frame(
            SpeakHintFrame(text=reply_text, phone=phone, append_to_context=False)
        )
        return played

    # ------------------------------------------------------------------ #
    # 打断
    # ------------------------------------------------------------------ #

    def _cancel_turn(self) -> None:
        """置位取消开关；不 `task.cancel()`，让 `_run_turn()` 跑到顺延文本的分支。

        直接 `cancel()` 会在下一个 await 抛 `CancelledError`，`_run_turn()` 里
        “文本顺延 + 记 `interrupted(stage="llm")`”的代码执行不到，来电者的前半句会丢。
        """
        self._llm_cancel.set()
        for cancel in self._turn_cancels:
            cancel.set()

    def _interrupted_cut(self, content: str, interrupted_text: str) -> int | None:
        """返回历史内容里“被打断那句”的结束位置（与 T14b 一致）。"""
        sentences = self._last_assistant_sentences
        if sentences and "".join(sentences) == content:
            position = 0
            for sentence in sentences:
                position += len(sentence)
                if sentence == interrupted_text:
                    return position
        index = content.rfind(interrupted_text)
        return index + len(interrupted_text) if index >= 0 else None

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
            self._last_assistant_sentences = [text]
            self._history.append({"role": "assistant", "content": text})
            self._trim_history()

    def _replace_last_assistant(self, text: str) -> None:
        if not self._history or self._history[-1].get("role") != "assistant":
            return
        self._last_assistant_sentences = [text]
        self._history[-1]["content"] = text

    def _trim_history(self) -> None:
        if len(self._history) > self._history_messages:
            self._history = self._history[-self._history_messages :]


def _is_asr_failed(frame: Frame) -> bool:
    """延迟导入避免与 stt.py 形成环形依赖。"""
    from adapters.pipecat.stt import AsrFailedFrame

    return isinstance(frame, AsrFailedFrame)


def confirm_phone_text(phone: str) -> str:
    """号码片段兜底要播的完整文本（与 `CallSession._play_text` 一致）。"""
    return (
        f"{PHRASES['confirm_phone_prefix']}{phone_to_spoken(phone)}"
        f"{PHRASES['confirm_phone_suffix']}"
    )

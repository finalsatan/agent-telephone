"""T16：对话（LLM + 状态机）的 LiveKit 适配层。

LiveKit 的 `Agent` 提供三个可覆写的节点，正好对上本项目的一轮：

    on_user_turn_completed(turn_ctx, new_message)   本轮用户文本已确定（ASR 全部到齐）
        → 调 services/llm 拿 slots → core.dialog.Dialog 决定这一轮怎么回
        → 把要播的文本段存起来
    llm_node(chat_ctx, tools, settings)             把上面存的段按 FlushSentinel 切段吐出
    tts_node(text, settings)                        用 services/tts + PhraseCache 合成

为什么不用 LiveKit 的 LLM 服务：本项目 LLM 输出固定 JSON、流程由 `Dialog` 用代码决定
（T07c 之后 reply 基本不播），而 LiveKit 的 `llm.LLM` 假设“上下文进、流式文本出”。
覆写 `llm_node` 后框架照样负责分段、合成、播放、打断与播放进度统计（notes 1.2 / 1.6）。

**行为对齐**（要求 4，与 T15b–T15d 的 Pipecat 路径逐条对应）：

- 一轮多段发声：由轮次检测器截取整轮音频并转写一次，拿到完整文本后再起 LLM。
- 打断阈值只管打断：VAD 的开口阈值与 `interruption.min_duration` 是框架里两套独立参数，
  短回答不会被吞（notes 1.4）。
- AI 还在播时抢话：`PacedAudioSink` 上报 `playback_position`，框架算出“这段播到哪”，
  这里据此记 `interrupted(tail=True)` 并给历史打“（被对方打断）”。
- LLM 阶段抢话：本轮处理期间对方又开口（`_turn_barge_in`）→ 文本顺延到下一轮，
  记 `interrupted(stage="llm")`，不播这一轮。
- TTS 首块音频之前被抢话：这段 `playback_position == 0`，历史里不写没播出的句子
  （`confirm_lead` 前一句已播时只保留它）。
- 号码兜底、`confirm_lead`、ASR 失败兜底、`dialog` 单一对象、事件格式、`turn_done`
  都与 `CallSession` / Pipecat 一致（复用 `core.call` 与 `core.history` 的共用实现）。
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

from livekit import rtc
from livekit.agents import (
    Agent,
    ChatContext,
    ChatMessage,
    FlushSentinel,
    ModelSettings,
    StopResponse,
)

from core.call import (
    choose_reply_text,
    is_partial_phone,
    stream_llm_turn,
    template_hint,
)
from core.dialog import split_confirm_lead
from core.events import EventLogger
from core.history import (
    DEFAULT_HISTORY_MESSAGES,
    INTERRUPTED_SUFFIX,
    drop_unplayed_assistant,
    record_assistant,
    record_user,
    replace_last_assistant,
)
from core.types import SLOT_NAMES, State

INSTRUCTIONS = "本项目的回复由代码生成，不使用框架的 LLM。"


class CallAgent(Agent):
    """一轮通话的对话处理器（与 `CallSession` / Pipecat 的 `DialogProcessor` 同语义）。"""

    def __init__(
        self,
        *,
        llm: Any,
        dialog: Any,
        events: EventLogger,
        speaker: Any,
        config: Mapping[str, Any] | None = None,
        history_messages: int = DEFAULT_HISTORY_MESSAGES,
        on_turn_done: Any = None,
        wait_for_greeting: Any = None,
    ) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._llm_client = llm
        self._dialog = dialog
        self._events = events
        self._speaker = speaker
        # 号码片段兜底播出后要让对话处理器改历史（T15b 的 confirm_phone 路径）。
        speaker.on_fallback = self.on_tts_fallback
        self._history_messages = int(history_messages)
        self._confirm_lead = bool((((config or {}).get("tts") or {}).get("confirm_lead", False)))
        # `LiveKitCall` 收到 turn_done 后转发给页面（与 T13b 的消息格式一致）。
        self._on_turn_done = on_turn_done
        self._wait_for_greeting = wait_for_greeting

        self._history: list[dict[str, str]] = []
        self._last_assistant_sentences: list[str] = []
        # 本轮要播的段（confirm_lead 拆句后是两段）；`llm_node` 按 FlushSentinel 切段吐出。
        self._pending_segments: list[str] = []
        self._pending_phone: str | None = None
        self._turn_user_text = ""
        # 本轮决定要播的段（confirm_lead 拆句后是两段），以及已经开播的段数。
        self._turn_segments: list[str] = []
        self._played_count = 0
        # 本轮已经开播的段，以及最后一段是否被打断（由音频 sink 回调）。
        self._played_segments: list[str] = []
        self._tail_interrupted = False
        # 本轮处理期间对方又开口（LLM 阶段抢话）。
        self._turn_barge_in = False
        self._turn_in_flight = False
        self._awaiting_turn_end = False
        self._interrupt_reason: str | None = None
        self._takeover = False
        self._greeted = False
        # 被抢话后顺延到下一轮的文本。
        self._carried_text = ""
        self.turn_count = 0

    # ------------------------------------------------------------------ #
    # 对外（LiveKitCall）
    # ------------------------------------------------------------------ #

    @property
    def dialog(self) -> Any:
        return self._dialog

    @property
    def history(self) -> list[dict[str, str]]:
        return self._history

    @property
    def takeover(self) -> bool:
        return self._takeover

    def request_takeover(self, reason: str = "takeover") -> None:
        self._interrupt_reason = reason
        self._takeover = True

    def notify_user_speaking(self) -> None:
        """本轮还在处理（LLM 阶段）时对方开口 → 标记抢话，文本顺延到下一轮。

        由 `LiveKitCall` 的 `TurnDetector` 判定达到打断阈值时调用。
        """
        if self._turn_in_flight:
            self._turn_barge_in = True

    def on_play_start(self, text: str) -> None:
        self._events.log("play_start", text=text)
        if not self._awaiting_turn_end:
            # 问候语等：只记事件，不算进某一轮。
            return
        self._played_segments.append(text)
        self._played_count += 1

    def on_play_end(self, text: str, played_s: float, interrupted: bool) -> None:
        if played_s <= 0:
            # 这段一个字都没播（TTS 首块之前被抢话）：不记 play_end。
            if interrupted:
                self._tail_interrupted = True
            return
        self._events.log("play_end", text=text, duration_ms=round(played_s * 1000.0, 2))
        if interrupted:
            self._tail_interrupted = True
            self._events.log(
                "interrupted",
                reason=self._interrupt_reason or "barge_in",
                tail=True,
                text=text,
            )

    def on_tts_fallback(self, spoken: str) -> None:
        """号码片段兜底播了 `spoken`：把历史里的确认语换成实际播出的内容。"""
        if not self._history or self._history[-1].get("role") != "assistant":
            return
        for index in range(len(self._turn_segments) - 1, -1, -1):
            if self._turn_segments[index] != spoken:
                self._turn_segments[index] = spoken
                break
        self._replace_last_assistant(self._turn_full_text)

    @property
    def _turn_full_text(self) -> str:
        return "".join(self._turn_segments)

    # ------------------------------------------------------------------ #
    # LiveKit 节点
    # ------------------------------------------------------------------ #

    async def on_enter(self) -> None:
        """问候语：与 `CallSession.start()` 一样，只记状态与历史，由缓存直接播出。"""
        if self._greeted:
            return
        if self._wait_for_greeting is not None:
            await self._wait_for_greeting()
        self._greeted = True
        before = self._dialog.state
        hint = self._dialog.greeting()
        self._log_state_change(before, self._dialog.state)
        if not hint.text:
            return
        self._record_assistant(hint.text)
        self._segments_for_turn([hint.text], phone=None)
        # 问候语不发 turn_done（页面等问候语的 play_start/play_end，见 T14b）。
        self._awaiting_turn_end = False
        self.session.say(hint.text)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        text = (new_message.text_content or "").strip()
        self._turn_user_text = text
        if self._takeover:
            self._emit_turn_done("skipped", text)
            raise StopResponse()
        if not text:
            # 本轮一个字都没转出来（ASR 报错 / 空转写）：播兜底话术
            # （与 `CallSession._reply_with_fallback` 一致）。
            await self.speak_fallback()
            raise StopResponse()

        self._turn_barge_in = False
        self._turn_in_flight = True
        try:
            await self._run_turn(text)
        finally:
            self._turn_in_flight = False

        if self._turn_barge_in:
            # LLM 阶段抢话：文本顺延到下一轮，这一轮不播（与 Pipecat 一致）。
            self._carried_text = text
            self._events.log("interrupted", reason="barge_in", stage="llm", text=text)
            self._emit_turn_done("interrupted", text)
            raise StopResponse()

        if not self._pending_segments:
            self._emit_turn_done("skipped", text)
            raise StopResponse()
        # 等 `speech_created` 给到 SpeechHandle，播完之后再收尾（记 turn_done、修历史）。
        self._awaiting_turn_end = True

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list[Any],
        model_settings: ModelSettings,
    ) -> AsyncIterator[str | FlushSentinel]:
        """把 `on_user_turn_completed()` 决定的文本按 FlushSentinel 切段吐出。"""
        segments, self._pending_segments = self._pending_segments, []
        for index, segment in enumerate(segments):
            if index:
                # 段边界：框架会把每一段单独合成、顺序播出（confirm_lead 靠它拆句）。
                yield FlushSentinel()
            yield segment

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        buffer = ""
        async for chunk in text:
            buffer += chunk
        if not buffer.strip():
            return
        cancel = asyncio.Event()
        try:
            async for frame in self._speaker.synth(buffer, cancel, phone=self._pending_phone):
                yield frame
        finally:
            cancel.set()

    # ------------------------------------------------------------------ #
    # 一轮
    # ------------------------------------------------------------------ #

    async def _run_turn(self, text: str) -> None:
        text = self._carried_text + text
        self._carried_text = ""

        before = copy.deepcopy(self._dialog)
        state_before = before.state

        # 追问号码时末尾数字串可能只是一半。实时链路没有音频时间轴，按“超时”处理
        # （与 T13 / Pipecat 一致）：不计入座机、计一次追问失败。
        partial_phone = self._dialog.asked_slot == "callback_number" and is_partial_phone(text)
        dialog_text = "" if partial_phone else None
        if partial_phone:
            self._events.log("error", stage="asr", reason="incomplete_phone_timeout", text=text)

        cancel = asyncio.Event()
        sentences, result, llm_error = await stream_llm_turn(
            self._llm_client,
            self._events,
            state_before,
            before.slots,
            self._history,
            text,
            before.attempts,
            cancel,
        )
        if self._turn_barge_in:
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
        record_user(self._history, text, limit=self._history_messages)
        self.turn_count += 1
        if not reply_text:
            self._record_assistant("")
            return

        segments, phone = self._segments_for_reply(hint, reply_text)
        self._record_assistant("".join(segments))
        self._segments_for_turn(segments, phone=phone)

    def _segments_for_reply(self, hint: Any, reply_text: str) -> tuple[list[str], str | None]:
        """按 `confirm_lead` 决定这一轮要播的段（与 `DialogProcessor._emit_reply` 一致）。"""
        phone = getattr(hint, "phone", None) if hint.use_text else None
        if hint.use_text and self._confirm_lead:
            lead, rest = split_confirm_lead(reply_text)
            if lead:
                return ([lead, rest] if rest else [lead]), phone
        return [reply_text], phone

    def _segments_for_turn(self, segments: list[str], *, phone: str | None) -> None:
        self._turn_segments = list(segments)
        self._pending_segments = list(segments)
        self._pending_phone = phone
        self._played_segments = []
        self._played_count = 0
        self._tail_interrupted = False

    # ------------------------------------------------------------------ #
    # 本轮收尾（由 LiveKitCall 在 speech 播完后调用）
    # ------------------------------------------------------------------ #

    def finish_speech(self, *, interrupted: bool) -> None:
        """一次 speech 播完（或被抢话）后的收尾：修历史 + 发 `turn_done`。

        一个字都没播出时（TTS 首块之前被抢话）走 `_never_played()`：历史里不写这句。
        """
        if not self._awaiting_turn_end:
            return
        self._awaiting_turn_end = False
        if self._played_count == 0:
            self._never_played()
            return
        if interrupted or self._tail_interrupted:
            # 已经播出的部分保留，标“（被对方打断）”。
            played = "".join(self._turn_segments[: self._played_count])
            self._last_assistant_sentences = [played]
            replace_last_assistant(self._history, played + INTERRUPTED_SUFFIX)
            self._emit_turn_done("interrupted", self._turn_user_text)
            return
        self._last_assistant_sentences = [self._turn_full_text]
        replace_last_assistant(self._history, self._turn_full_text)
        self._emit_turn_done("reply", self._turn_user_text)

    def _never_played(self) -> None:
        """整轮一个字都没播出（TTS 首块之前被抢话 / 接管，T15d）。"""
        self._events.log(
            "interrupted",
            reason=self._interrupt_reason or "barge_in",
            text=self._turn_full_text,
        )
        played = drop_unplayed_assistant(self._history, self._turn_full_text)
        self._last_assistant_sentences = [played] if played else []
        self._emit_turn_done("interrupted", self._turn_user_text)

    @property
    def awaiting_turn_end(self) -> bool:
        return self._awaiting_turn_end

    @property
    def busy(self) -> bool:
        """本轮还在处理：LLM 在跑，或者要播的话还没播完（含 TTS 还在合成）。

        抢话判定用它覆盖“首块音频之前”的窗口（T15d 的那一类）。
        """
        return self._turn_in_flight or self._awaiting_turn_end

    def _emit_turn_done(self, kind: str, text: str) -> None:
        if self._on_turn_done is not None:
            self._on_turn_done(kind, text)

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
        record_user(self._history, text, limit=self._history_messages)

    def _record_assistant(self, text: str) -> None:
        if text:
            self._last_assistant_sentences = [text]
        record_assistant(self._history, text, limit=self._history_messages)

    def _replace_last_assistant(self, text: str) -> None:
        if not self._history or self._history[-1].get("role") != "assistant":
            return
        self._last_assistant_sentences = [text]
        replace_last_assistant(self._history, text)

    # ---- ASR 失败兜底（由 LiveKitCall 的整轮转写结果触发） ----

    async def speak_fallback(self, *, log_empty: bool = True) -> None:
        """ASR 失败 / 转写为空：播 `repeat`（与 `CallSession._reply_with_fallback` 一致）。"""
        if self._takeover:
            self._emit_turn_done("skipped", "")
            return
        if log_empty:
            self._events.log("error", stage="asr", reason="empty_transcript")
        hint = template_hint(self._dialog, "repeat")
        self.turn_count += 1
        self._record_assistant(hint.text)
        self._segments_for_turn([hint.text], phone=None)
        self._turn_user_text = ""
        self._awaiting_turn_end = True
        self.session.say(hint.text)


__all__ = ["CallAgent", "INTERRUPTED_SUFFIX"]

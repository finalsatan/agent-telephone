"""离线回放的一轮通话主流程（T10）。

`CallSession` 把 T03/T07/T02b/T06b 的部件串成设计文档第 4 节规定的一轮：

    来电者音频 → ASR → LLM（只取槽位）→ Dialog → 选一句要播的话 → TTS/缓存 → 记录

三处容易踩坑的地方（第 7 / 9 / 11 次评审）：

1. **LLM 用的是本轮之前的状态**。T07c 之后复述与追问都由 `Dialog` 用代码生成，
   只有 `ReplyHint.kind="llm"` 才该播放 LLM 的 reply；其余情况只取它的 `slots_update`。
2. **轮次合并**。T09 会在句间停顿处切轮（“喂。”单独一轮），AI 不能抢着回应半句话。
   离线回放用音频时间轴判断：下一轮的 `start_ms` 早于 `本轮 end_ms + 本轮实测
   （ASR＋LLM 首句＋TTS 首包）耗时` 时，取消本次 LLM/TTS 并把两轮文本拼起来重做一轮。
   判断发生在第一次 `play_start` 之前；此时 `Dialog` 可能已经推进过状态，因此用深拷贝
   快照回滚，避免槽位被更新两次、追问次数被多计。
3. **号码没报完**。追问回拨号码时，如果 ASR 文本末尾只剩 1–10 位数字、且还取不到完整号码，
   先不回复，等下一轮（最多 `vad.stop_secs`）拼接后再处理；超时才按普通轮次处理并计一次
   追问失败。

所有外部服务（ASR／LLM／TTS）与缓存都可注入，`tests/test_replay.py` 用 fake 跑完整流程，
`tools/replay.py` 用真实服务跑一条 wav。播放不真正等待：`play_end` 事件里记录音频时长，
离线回放的时钟推进由调用方决定（T10 只要求事件顺序与槽位正确）。
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.events import EventLogger, monotonic_ms
from core.normalize import cn_digits_to_arabic, extract_phone, phone_to_spoken
from core.phrases import PHRASES, PhraseCache, PhraseMissing
from core.types import SLOT_NAMES, SlotValue, State, TurnResult
from services.asr import AsrError, AsrTimeout
from services.llm import LlmError, LlmTimeout, split_reply_sentences
from services.tts import TtsError, TtsTimeout

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STOP_SECS = 3.0
# 设计文档 0.5：history 只保留最近 8 轮（16 条）。
DEFAULT_HISTORY_MESSAGES = 16

# 被抢话/接管的 assistant 消息在历史里的后缀（T11b 起用于未播完的句子，T13b 尾音抢话也用它）。
INTERRUPTED_SUFFIX = "（被对方打断）"

# 末尾的纯数字串（1–10 位）。用 (?<!\d) 保证匹配的是完整数字串，而不是 11 位号码的后 10 位。
_TRAILING_DIGITS = re.compile(r"(?<!\d)(\d{1,10})$")


@dataclass
class TurnOutcome:
    """`CallSession.handle_turn()` 的结果，供回放循环决定下一步。

    kind：
    - `"reply"`：本轮已经回复（或尝试回复）；
    - `"merge"`：本轮回复还没开始播，下一轮就开口了，`text` 是拼接后的文本，
      调用方应把它作为 `carried_text` 交给下一轮；
    - `"wait"`：号码没报完，`text` 是已收到的文本，调用方应等下一轮拼接；
    - `"interrupted"`：本轮被来电者抢话或用户接管打断（T13 实时链路），
      `played` 是打断前已经播出/送出的句子；`played` 为空表示还没开播就被打断，
      调用方应把 `text` 当 `carried_text` 交给下一轮（与 `merge` 同义）。
    - `"empty"`：没有识别到内容。
    """

    text: str = ""
    kind: str = "reply"
    state: State | None = None
    played: list[str] = field(default_factory=list)


@dataclass
class _PlayContext:
    """第一次播放前的合并判断所需的音频时间轴与计时起点。"""

    turn_start_ms: float
    end_ms: float | None = None
    next_start_ms: float | None = None


@dataclass
class _PlayResult:
    played: list[str] = field(default_factory=list)
    merge: bool = False
    error: str | None = None
    pcm: bytes = b""
    interrupted: bool = False


# 播放分块（T13）：缓存整句一次发完太粗，抢话时来不及停；32ms 一块与 VAD 帧对齐。
PLAYBACK_CHUNK_BYTES = 1024


class CallSession:
    """一通电话（T10 的离线回放，也供 T13 实时链路复用同一套单轮流程）。"""

    def __init__(
        self,
        asr: Any,
        llm: Any,
        tts: Any,
        cache: PhraseCache,
        events: EventLogger,
        config: Mapping[str, Any],
        *,
        dialog: Any | None = None,
        sample_rate: int | None = None,
        stop_secs: float | None = None,
        clock: Callable[[], float] | None = None,
        history_messages: int = DEFAULT_HISTORY_MESSAGES,
        confirm_lead: bool | None = None,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.cache = cache
        self.events = events
        self.config = config
        self.clock = clock or monotonic_ms
        audio_cfg = config.get("audio") or {}
        self.sample_rate = int(sample_rate or audio_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
        vad_cfg = config.get("vad") or {}
        self.stop_secs = float(stop_secs if stop_secs is not None else vad_cfg.get("stop_secs", DEFAULT_STOP_SECS))
        self.history_messages = int(history_messages)
        if confirm_lead is None:
            confirm_lead = bool((config.get("tts") or {}).get("confirm_lead", False))
        self.confirm_lead = bool(confirm_lead)

        if dialog is None:
            from core.dialog import Dialog

            dialog = Dialog.from_config(config)
        self.dialog = dialog

        self.history: list[dict[str, str]] = []
        self.caller_pcm = bytearray()
        self.ai_pcm = bytearray()
        # T12：按模拟时钟对齐的双声道 call.wav 用 (距通话开始的毫秒数, PCM) 片段。
        self.caller_segments: list[tuple[float, bytes]] = []
        self.ai_segments: list[tuple[float, bytes]] = []
        self._t0: float | None = None
        self.turn_count = 0
        self._started = False
        self._finished = False
        # 打断用：合并轮次或实时链路的抢话/接管时置位，让正在跑的 LLM/TTS 尽快退出。
        self._llm_cancel = asyncio.Event()
        self._tts_cancel = asyncio.Event()
        # 本轮是否被打断、以及打断原因（T13）。
        self._turn_interrupted = False
        self._interrupt_reason: str | None = None

    # ------------------------------------------------------------------ #
    # 通话开始 / 结束
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """记录 call_start 并播放问候语（模板缓存，不调 LLM）。"""
        if self._started:
            return
        self._started = True
        self._t0 = self.clock()
        self.events.log("call_start", sample_rate=self.sample_rate, source="replay")
        before = self.dialog.state
        hint = self.dialog.greeting()
        played, _ = await self._play_template(hint.text, _PlayContext(turn_start_ms=self.clock()))
        self._log_state_change(before, self.dialog.state)
        if played:
            self._record_assistant(played)

    def finish(self) -> None:
        """写 slots.json 与分轨 wav，记录 call_end。"""
        if self._finished:
            return
        self._finished = True
        self.events.log(
            "call_end",
            state=self.dialog.state.value,
            turns=self.turn_count,
            ai_audio_ms=self._pcm_ms(bytes(self.ai_pcm)),
            caller_audio_ms=self._pcm_ms(bytes(self.caller_pcm)),
        )
        call_dir = Path(self.events.call_dir)
        self._write_wav(call_dir / "caller.wav", bytes(self.caller_pcm))
        self._write_wav(call_dir / "ai.wav", bytes(self.ai_pcm))
        self._write_call_wav(call_dir / "call.wav")
        slots = {
            name: {"value": value.value, "status": value.status}
            for name, value in self.dialog.slots.items()
        }
        payload = {"call_id": self.events.call_id, "state": self.dialog.state.value, "slots": slots}
        (call_dir / "slots.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # 一轮
    # ------------------------------------------------------------------ #

    async def handle_turn(
        self,
        pcm: bytes | None = None,
        *,
        text: str | None = None,
        start_ms: float | None = None,
        end_ms: float | None = None,
        next_start_ms: float | None = None,
        carried_text: str = "",
        timeline_ms: float | None = None,
    ) -> TurnOutcome:
        """处理来电者的一轮。

        `pcm` 为这一轮的音频（可为 None，表示只有 ASR 文本，如合并后的重跑）；
        `carried_text` 是上一轮因合并 / 等号码而暂存的文本；
        `end_ms` 与 `next_start_ms` 是音频时间轴上的本轮回合结束与下一轮开口时刻，
        仅离线回放用来判断是否需要在播放前合并轮次。
        `timeline_ms` 是这段来电录音在通话时间轴上的起点（按轮场景的模拟时钟）；
        省略时取当前时钟，供 call.wav 对齐用。
        """
        turn_start = self.clock()
        self.turn_count += 1
        self._turn_interrupted = False
        self._interrupt_reason = None
        self._llm_cancel = asyncio.Event()
        self._tts_cancel = asyncio.Event()

        if pcm is not None:
            self.caller_pcm += pcm
            self._capture(
                self.caller_segments,
                pcm,
                at=None if timeline_ms is None or self._t0 is None else timeline_ms - self._t0,
            )
            self.events.log(
                "asr_request",
                audio_ms=self._pcm_ms(pcm),
                start_ms=start_ms,
                end_ms=end_ms,
            )
            try:
                asr_result = await self.asr.transcribe(pcm)
            except (AsrTimeout, AsrError) as exc:
                self.events.log("error", stage="asr", reason=type(exc).__name__, message=str(exc)[:200])
                return await self._reply_with_fallback(
                    carried_text, _PlayContext(turn_start, end_ms, next_start_ms)
                )
            text = asr_result.text
            self.events.log(
                "asr_final",
                text=text,
                latency_ms=asr_result.latency_ms,
                audio_ms=asr_result.audio_ms,
            )
        if text is None:
            raise ValueError("handle_turn() 需要 pcm 或 text 之一")

        full_text = (carried_text or "") + (text or "")
        if not full_text.strip():
            self.events.log("error", stage="asr", reason="empty_transcript")
            return await self._reply_with_fallback("", _PlayContext(turn_start, end_ms, next_start_ms))

        ctx = _PlayContext(turn_start, end_ms, next_start_ms)

        # 追问号码时，末尾数字串可能只是号码的一半：先等下一轮拼接（最多 stop_secs）。
        partial_phone = (
            self.dialog.asked_slot == "callback_number" and self._is_partial_phone(full_text)
        )
        if partial_phone and self._has_next_turn(ctx):
            return TurnOutcome(text=full_text, kind="wait", state=self.dialog.state)

        self._llm_cancel = asyncio.Event()
        before = copy.deepcopy(self.dialog)
        # 号码没报完且已经超时：按普通轮次处理，但要计一次追问失败。
        # Dialog 用空文本更新（不把半个号码当座机写入），槽位更新照常应用；LLM 仍看原文。
        dialog_text = "" if partial_phone else None
        if partial_phone:
            self.events.log("error", stage="asr", reason="incomplete_phone_timeout", text=full_text)
        state_after, hint, sentences, _result, _llm_error = await self._run_llm(
            full_text, before, dialog_text=dialog_text
        )

        if self._llm_cancel.is_set():
            # 还没开播就被打断（T13）：本轮作废，回滚状态机，文本交给下一轮重做。
            # 这与 T10 的轮次合并同义：调用方把这个 `text` 当 `carried_text` 交给下一轮，
            # 既不丢对方说的话，也不会因为同一句话被追问两次。
            self.dialog = before
            self.events.log(
                "interrupted",
                reason=self._interrupt_reason or "cancelled",
                stage="llm",
                text=full_text,
            )
            return TurnOutcome(text=full_text, kind="interrupted", state=self.dialog.state)

        played, merge = await self._play_hint(
            hint,
            sentences,
            ctx,
            check_merge=True,
        )

        if self._turn_interrupted:
            self._record_user(full_text)
            if played:
                self._record_assistant(played, interrupted=True)
            self._log_state_change(before.state, state_after)
            self._log_slot_updates(before.slots, self.dialog.slots)
            return TurnOutcome(
                text=full_text, kind="interrupted", state=state_after, played=played
            )

        if merge:
            # 置位取消开关（T13 实时链路里 LLM/TTS 可能还在跑）；离线时它们通常已经结束。
            self._llm_cancel.set()
            self._tts_cancel.set()
            # Dialog 已经推进过状态：回滚到本轮之前，拼接后的文本会重新走一遍。
            self.dialog = before
            self.events.log("interrupted", reason="turn_merged", text=full_text)
            return TurnOutcome(text=full_text, kind="merge", state=self.dialog.state, played=[])

        self._record_user(full_text)
        if played:
            self._record_assistant(played)
        self._log_state_change(before.state, state_after)
        self._log_slot_updates(before.slots, self.dialog.slots)
        return TurnOutcome(text=full_text, kind="reply", state=state_after, played=played)

    # ------------------------------------------------------------------ #
    # ASR 失败 / 空文本的兜底
    # ------------------------------------------------------------------ #

    async def _reply_with_fallback(self, text: str, ctx: _PlayContext) -> TurnOutcome:
        hint = _template_hint(self.dialog, "repeat")
        played, merge = await self._play_hint(hint, [], ctx, check_merge=True)
        if merge:
            self.events.log("interrupted", reason="turn_merged", text=text)
            return TurnOutcome(text=text, kind="merge", state=self.dialog.state)
        if self._turn_interrupted:
            if text.strip():
                self._record_user(text)
            return TurnOutcome(text=text, kind="interrupted", state=self.dialog.state, played=played)
        if text.strip():
            self._record_user(text)
        if played:
            self._record_assistant(played)
        return TurnOutcome(text=text, kind="reply", state=self.dialog.state, played=played)

    # ------------------------------------------------------------------ #
    # LLM + Dialog
    # ------------------------------------------------------------------ #

    async def _run_llm(
        self,
        text: str,
        before: Any,
        *,
        dialog_text: str | None = None,
    ) -> tuple[State, Any, list[str], TurnResult | None, bool]:
        """调 LLM 流式接口，再把结果交给 Dialog。

        返回 `(state_after, hint, sentences, result, llm_error)`；`llm_error=True`
        表示超时或解析失败，此时 `hint` 已是该状态的兜底话术。
        """
        state_before = before.state
        slots_before = before.slots
        attempts = before.attempts
        self.events.log(
            "llm_request",
            state=state_before.value,
            text=text,
            attempts=attempts,
        )

        sentences: list[str] = []
        result: TurnResult | None = None
        first_logged = False
        llm_error = False
        try:
            async for chunk in self.llm.turn_stream(
                state_before, slots_before, list(self.history), text, attempts
            ):
                if self._llm_cancel.is_set():
                    break
                if chunk.kind == "sentence":
                    if not first_logged:
                        self.events.log("llm_first_token", latency_ms=chunk.t_ms)
                        first_logged = True
                    sentences.append(chunk.text)
                elif chunk.kind == "done":
                    result = chunk.result
                    if not first_logged and result is not None:
                        self.events.log("llm_first_token", latency_ms=result.first_token_ms or 0.0)
                    self.events.log(
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
            self.events.log("error", stage="llm", reason="timeout", message=str(exc)[:200])
        except LlmError as exc:
            llm_error = True
            self.events.log("error", stage="llm", reason=type(exc).__name__, message=str(exc)[:200])

        parse_ok = bool(result and result.parse_ok and not llm_error)
        slots_update = result.slots_update if (result and not llm_error) else {}
        state_after, hint = self.dialog.on_user_turn(
            text if dialog_text is None else dialog_text, slots_update, parse_ok
        )
        if result is not None and result.parse_ok is False:
            llm_error = True
            # T12b：附上模型原始输出的前 300 字，便于排查 s06 这类解析失败。
            self.events.log(
                "error",
                stage="llm",
                reason="parse",
                reply=result.reply[:200],
                raw=result.raw[:300],
            )
        if result is None and not llm_error and not self._llm_cancel.is_set():
            llm_error = True
            self.events.log("error", stage="llm", reason="no_done_chunk")
        return state_after, hint, sentences, result, llm_error

    # ------------------------------------------------------------------ #
    # 选择播放内容
    # ------------------------------------------------------------------ #

    async def _play_hint(
        self,
        hint: Any,
        sentences: list[str],
        ctx: _PlayContext,
        *,
        check_merge: bool,
    ) -> tuple[list[str], bool]:
        """按 `ReplyHint.kind` 播放，返回 `(played_texts, merged)`。"""
        if self._turn_interrupted:
            return [], False
        if hint.use_cache:
            return await self._play_template(hint.text, ctx, check_merge=check_merge)
        if hint.use_text:
            return await self._play_text(hint, ctx, check_merge=check_merge)
        # kind="llm"：T07c 之后 Dialog 不再返回它，但保留分支（测试也直接构造）。
        texts = sentences or (split_reply_sentences(getattr(hint, "text", "")) if hint.text else [])
        played: list[str] = []
        for sentence in texts:
            result = await self._render_text(
                sentence, ctx, check_merge=check_merge and not played
            )
            if result.merge:
                return played, True
            played.extend(result.played)
        return played, False

    async def _play_template(
        self, text: str, ctx: _PlayContext, *, check_merge: bool = True
    ) -> tuple[list[str], bool]:
        pcm = self.cache.get(text)
        if pcm is not None:
            return await self._play_cached(text, pcm, ctx, check_merge=check_merge)
        # 缓存没命中：现合成并补写缓存，同时记 error（预合成没跑或换了音色）。
        self.events.log("error", stage="tts", reason="phrase_cache_miss", text=text)
        result = await self._render_text(text, ctx, check_merge=check_merge)
        if result.pcm and not result.merge:
            try:
                self.cache.put(text, result.pcm)
            except OSError as exc:  # 写缓存失败不该影响这一通电话
                self.events.log("error", stage="tts", reason="cache_write_failed", message=str(exc)[:200])
        return result.played, result.merge

    async def _play_text(
        self, hint: Any, ctx: _PlayContext, *, check_merge: bool
    ) -> tuple[list[str], bool]:
        """CONFIRM 的整句确认语：现合成；失败时退回号码片段拼接（T07c / T06b）。

        T12 的对比实验（`tts.confirm_lead`）：先播预合成的“好的，跟您确认一下。”，
        同时现合成后半句（“您是……，对吗？”）。后半句失败时仍按原路径退回号码拼接。
        """
        from core.dialog import split_confirm_lead

        played_prefix: list[str] = []
        text = hint.text
        if self.confirm_lead:
            lead, rest = split_confirm_lead(text)
            if lead:
                prefix, merged = await self._play_template(lead, ctx, check_merge=check_merge)
                if merged:
                    return prefix, True
                if not prefix:
                    # 预合成开头也没播出来：按原路径整句现合成。
                    rest = text
                else:
                    played_prefix = prefix
                    text = rest
                    check_merge = False  # 已经开播，不再做轮次合并

        result = await self._render_text(text, ctx, check_merge=check_merge)
        if result.merge:
            return played_prefix, True
        if not result.error:
            return played_prefix + result.played, False

        phone = getattr(hint, "phone", None)
        if not phone:
            # 没有号码可兜底：播 repeat，至少不让这一轮彻底静音。
            fallback = _template_hint(self.dialog, "repeat")
            suffix, merged = await self._play_template(fallback.text, ctx, check_merge=check_merge)
            return played_prefix + suffix, merged

        self.events.log("error", stage="tts", reason="confirm_synth_failed", text=hint.text, phone=phone)
        try:
            pcm = self.cache.render_phone(phone)
        except (PhraseMissing, ValueError) as exc:
            self.events.log("error", stage="tts", reason="phone_fallback_failed", message=str(exc)[:200])
            fallback = _template_hint(self.dialog, "repeat")
            suffix, merged = await self._play_template(fallback.text, ctx, check_merge=check_merge)
            return played_prefix + suffix, merged

        # render_phone 的后缀已经是 PHRASES["confirm_phone_suffix"]（"，对吗？"），
        # 等价于再播 confirm_question，所以不再追加，避免“，对吗？对吗？”。
        spoken = (
            f"{PHRASES['confirm_phone_prefix']}{phone_to_spoken(phone)}"
            f"{PHRASES['confirm_phone_suffix']}"
        )
        suffix, merged = await self._play_cached(
            spoken, pcm, ctx, check_merge=check_merge, logged_text=hint.text
        )
        return played_prefix + suffix, merged

    # ------------------------------------------------------------------ #
    # 缓存 / TTS / 播放
    # ------------------------------------------------------------------ #

    async def _play_cached(
        self,
        text: str,
        pcm: bytes,
        ctx: _PlayContext,
        *,
        check_merge: bool,
        logged_text: str | None = None,
    ) -> tuple[list[str], bool]:
        label = logged_text or text
        self.events.log("tts_request", text=label, cached=True)
        self.events.log("tts_first_audio", text=label, latency_ms=0.0)
        if check_merge and self._should_merge(ctx):
            return [], True
        if self._turn_interrupted:
            return [], False
        await self._play_pcm(pcm, text)
        return [text], False

    async def _render_text(
        self, text: str, ctx: _PlayContext, *, check_merge: bool
    ) -> _PlayResult:
        """现合成一句并播放。首块到达后做一次轮次合并判断。"""
        if not text:
            return _PlayResult()
        if self._turn_interrupted:
            return _PlayResult(interrupted=True)
        self.events.log("tts_request", text=text, cached=False)
        started = self.clock()
        cancel = asyncio.Event()
        self._tts_cancel = cancel
        chunks: list[bytes] = []
        playing = False
        play_started_at: float | None = None
        try:
            async for chunk in self.tts.synth(text, cancel):
                if not chunk:
                    continue
                if not chunks:
                    self.events.log(
                        "tts_first_audio",
                        text=text,
                        latency_ms=max(0.0, self.clock() - started),
                    )
                    if check_merge and self._should_merge(ctx):
                        cancel.set()
                        return _PlayResult(merge=True)
                    # 首块到手就开播：play_start 的时刻必须是首包时刻，
                    # 否则 T12 的 vad_end→play_start 会把整句合成时间算成响应时延。
                    play_started_at = self._elapsed()
                    self.events.log("play_start", text=text)
                    playing = True
                chunks.append(chunk)
                self.ai_pcm += chunk
                await self._emit_chunk(chunk, text)
                if cancel.is_set():
                    break
        except (TtsTimeout, TtsError) as exc:
            self.events.log("error", stage="tts", reason=type(exc).__name__, text=text, message=str(exc)[:200])
            pcm = b"".join(chunks)
            if playing:
                # 已经开播：这句只播了一部分（T11b）。先标 interrupted，再按已播出的部分收尾，
                # 摘要（core/summary.py）会据此写"（未说完）"，不当成完整说出。
                self.events.log("interrupted", reason="tts_error", text=text)
                self.events.log("play_end", text=text, duration_ms=self._pcm_ms(pcm))
                self._capture(self.ai_segments, pcm, at=play_started_at)
                return _PlayResult(played=[text], pcm=pcm, interrupted=True)
            return _PlayResult(error=type(exc).__name__)
        if not chunks:
            if cancel.is_set():
                # 首块之前就被取消（T13 抢话 / 接管）：这一句从未播出。
                self._turn_interrupted = True
                self.events.log(
                    "interrupted", reason=self._interrupt_reason or "cancelled", text=text
                )
                return _PlayResult(interrupted=True)
            self.events.log("error", stage="tts", reason="empty_audio", text=text)
            return _PlayResult(error="empty_audio")
        pcm = b"".join(chunks)
        if cancel.is_set():
            # 播放中被取消（T13 抢话 / 接管）：按已产出的部分收尾。
            self._turn_interrupted = True
            self.events.log("interrupted", reason=self._interrupt_reason or "cancelled", text=text)
            self.events.log("play_end", text=text, duration_ms=self._pcm_ms(pcm))
            self._capture(self.ai_segments, pcm, at=play_started_at)
            return _PlayResult(played=[text], pcm=pcm, interrupted=True)
        self.events.log("play_end", text=text, duration_ms=self._pcm_ms(pcm))
        self._capture(self.ai_segments, pcm, at=play_started_at)
        return _PlayResult(played=[text], pcm=pcm)

    async def _play_pcm(self, pcm: bytes, text: str) -> None:
        """播放一段已经合成好的音频（预合成话术命中缓存时走这里）。

        T13 起按 32ms 分块交给 `_emit_chunk()`，每块后检查取消开关：抢话时
        不再继续往外送，浏览器收到 `interrupted` 后清空待播队列。
        """
        play_started_at = self._elapsed()
        self.events.log("play_start", text=text)
        sent = bytearray()
        for offset in range(0, len(pcm), PLAYBACK_CHUNK_BYTES):
            chunk = pcm[offset : offset + PLAYBACK_CHUNK_BYTES]
            await self._emit_chunk(chunk, text)
            sent += chunk
            if self._tts_cancel.is_set():
                break
        # T13b：只把已经送出的部分写进录音。被抢话时整句写进 ai.wav 会误导复盘
        # （听起来 AI 说完了，其实只播了一部分）。
        self.ai_pcm += bytes(sent)
        self._capture(self.ai_segments, bytes(sent), at=play_started_at)
        if len(sent) < len(pcm) and self._tts_cancel.is_set():
            self._turn_interrupted = True
            self.events.log(
                "interrupted", reason=self._interrupt_reason or "cancelled", text=text
            )
            self.events.log("play_end", text=text, duration_ms=self._pcm_ms(bytes(sent)))
            return
        self.events.log("play_end", text=text, duration_ms=self._pcm_ms(pcm))

    async def _emit_chunk(self, chunk: bytes, text: str) -> None:
        """把一块待播放的音频交给调用方。

        离线回放不真正播放，默认什么都不做；T13 的实时链路覆写为推给浏览器。
        """

    def interrupt(self, reason: str = "barge_in") -> None:
        """取消正在进行的 LLM 与 TTS（T13：来电者抢话、用户点“接管”）。

        置位取消开关后，`turn_stream()` 与 `synth()` 会在下一个 chunk 处退出，
        `_render_text()` / `_play_pcm()` 会记 `interrupted` 并把这一句当作未说完。
        没有在播的轮次上调用也无害：下一轮的 `handle_turn()` 会重建这两个开关。
        """
        self._interrupt_reason = reason
        self._llm_cancel.set()
        self._tts_cancel.set()

    def _should_merge(self, ctx: _PlayContext) -> bool:
        """音频时间轴上“下一轮在回复开始前开口”判定（第 7 次评审）。"""
        if ctx.end_ms is None or ctx.next_start_ms is None:
            return False
        elapsed = max(0.0, self.clock() - ctx.turn_start_ms)
        return ctx.next_start_ms < ctx.end_ms + elapsed

    def _has_next_turn(self, ctx: _PlayContext) -> bool:
        """追问号码时，下一轮是否在 stop_secs 内开口（是则等它拼接）。"""
        if ctx.end_ms is None or ctx.next_start_ms is None:
            return False
        gap = ctx.next_start_ms - ctx.end_ms
        return 0 <= gap <= self.stop_secs * 1000.0

    @staticmethod
    def _is_partial_phone(text: str) -> bool:
        """追问号码时，文本末尾是否是“没报完”的数字串。

        规则（第 7 次评审 + 验收用例“一三八零零一三”＋“八零零零”）：
        - 末尾必须是完整的 1–10 位数字串（`cn_digits_to_arabic` 后）；
        - `extract_phone(..., allow_landline=True)` 取不到号码，或末尾是 7–10 位、
          以 1 开头的字符串（可能是没报完的手机号；`extract_phone` 会把它当成 7 位座机）。
        """
        normalized = cn_digits_to_arabic(text)
        match = _TRAILING_DIGITS.search(normalized)
        if match is None:
            return False
        tail = match.group(1)
        if extract_phone(text, allow_landline=True) is None:
            return True
        return 7 <= len(tail) <= 10 and tail.startswith("1")

    # ------------------------------------------------------------------ #
    # 事件 / 历史 / 音频
    # ------------------------------------------------------------------ #

    def _log_state_change(self, before: State, after: State) -> None:
        if before is not after:
            self.events.log("state_change", **{"from": before.value, "to": after.value})

    def _log_slot_updates(
        self,
        before: Mapping[str, SlotValue],
        after: Mapping[str, SlotValue],
    ) -> None:
        for name in SLOT_NAMES:
            old = before.get(name)
            new = after.get(name)
            if old is None or new is None:
                continue
            if old.value == new.value and old.status == new.status:
                continue
            self.events.log(
                "slot_update",
                slot=name,
                value=new.value,
                status=new.status,
            )

    def _record_user(self, text: str) -> None:
        if text:
            self.history.append({"role": "user", "content": text})
            self._trim_history()

    def _record_assistant(self, played: list[str], *, interrupted: bool = False) -> None:
        content = "".join(played)
        if not content:
            return
        if interrupted:
            content += INTERRUPTED_SUFFIX
        self.history.append({"role": "assistant", "content": content})
        self._trim_history()

    def mark_last_assistant_interrupted(self) -> None:
        """把历史里最后一条 assistant 消息标成“（被对方打断）”（T13b 尾音抢话）。

        尾音抢话发生在这一轮已经结束、浏览器里还在播的时候：`_record_assistant()`
        已经把这句按完整说出记进历史，这里只补标记，不新增消息。若最后一条不是
        assistant（对方已经说了新的一轮），或已经标过，则不动。
        """
        if not self.history:
            return
        last = self.history[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content") or ""
        if content and not content.endswith(INTERRUPTED_SUFFIX):
            last["content"] = content + INTERRUPTED_SUFFIX

    def _trim_history(self) -> None:
        if len(self.history) > self.history_messages:
            self.history = self.history[-self.history_messages :]

    def _pcm_ms(self, pcm: bytes) -> float:
        return len(pcm) / (2 * self.sample_rate) * 1000.0

    def _elapsed(self) -> float:
        """距通话开始的毫秒数（T12 的 call.wav 与模拟时钟用）。"""
        if self._t0 is None:
            return 0.0
        return max(0.0, self.clock() - self._t0)

    def _capture(
        self, target: list[tuple[float, bytes]], pcm: bytes, *, at: float | None = None
    ) -> None:
        if pcm:
            target.append((self._elapsed() if at is None else at, pcm))

    def _write_call_wav(self, path: Path) -> None:
        """写左声道来电者、右声道 AI 的双声道录音（T12）。

        位置按 `CallSession` 的时钟对齐：分轨回放时工具的模拟时钟会在每轮之间推进
        （AI 播放时长 + gap_ms），因此这里记录的就是对话的真实先后。整条 wav 模式
        （`--wav`）也用同一份时钟，但来电者音频是连续处理的，听起来会比真实通话快。
        """
        sample_rate = self.sample_rate
        total_ms = 0.0
        for segments in (self.caller_segments, self.ai_segments):
            for offset, pcm in segments:
                total_ms = max(total_ms, offset + self._pcm_ms(pcm))
        frames = max(1, int(round(total_ms * sample_rate / 1000.0)))
        left = np.zeros(frames, dtype=np.int16)
        right = np.zeros(frames, dtype=np.int16)
        for target, segments in ((left, self.caller_segments), (right, self.ai_segments)):
            for offset, pcm in segments:
                samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
                start = int(round(offset * sample_rate / 1000.0))
                end = min(frames, start + samples.size)
                if end > start:
                    target[start:end] = samples[: end - start]
        interleaved = np.stack([left, right], axis=1).reshape(-1)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(interleaved.tobytes())

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            writer.writeframes(pcm)


def _template_hint(dialog: Any, template_id: str) -> Any:
    """构造一条模板提示（用于 ASR 失败等没有 Dialog 返回值的兜底）。"""
    from core.dialog import ReplyHint

    text = PHRASES[template_id].format(owner=dialog.owner_name)
    return ReplyHint(kind="template", template_id=template_id, text=text)

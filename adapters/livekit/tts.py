"""T16：TTS 的 LiveKit 适配层（薄封装）。

LiveKit 的 `Agent.tts_node(text, model_settings)` 接收**一段**文本（框架按
`FlushSentinel` 切段，见 notes 1.6），要产出 `AsyncIterable[rtc.AudioFrame]`。
这里把本项目的 `services/tts` + `core/phrases.PhraseCache` 包成这个形状：

1. 先查 T06 的预合成话术缓存，命中就直接产出 PCM，不调网络（首包 0）；
2. 没命中再调 `synth()` 流式合成；
3. 每块帧上打 `userdata["lk_text"] = text`，让音频 sink 知道这段在说哪句话，
   从而把 `play_start` / `play_end` / `interrupted` 记到正确的文本上；
4. CONFIRM 的确认语整句合成失败且一块音频都没播出时，退回
   `PhraseCache.render_phone()` 的号码片段拼接（与 `CallSession._play_text` 一致）。

T15d 的“首块音频之前被抢话”：这里不记事件，交给框架——`PacedAudioSink` 会按
`playback_finished(interrupted=True, playback_position=0)` 上报“这段一个字都没播”，
框架的 `forward_generation()` 据此把这段判为 `skipped`，`CallAgent` 再从历史里去掉它。

打断：`tts_node` 的生成器被框架取消时（`asyncio.CancelledError`），`finally` 里置位
传给 `synth()` 的 cancel 事件，让底层 HTTP 流立刻关闭。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from livekit import rtc

from adapters.livekit.audio import TEXT_KEY, frames_from_pcm
from core.events import EventLogger, monotonic_ms
from core.normalize import phone_to_spoken
from core.phrases import PHRASES, PhraseMissing


def confirm_phone_text(phone: str) -> str:
    """号码片段兜底要播的完整文本（与 `CallSession._play_text` 一致）。"""
    return (
        f"{PHRASES['confirm_phone_prefix']}{phone_to_spoken(phone)}"
        f"{PHRASES['confirm_phone_suffix']}"
    )


class ReplySpeaker:
    """一句文本 → 一串 16kHz 单声道 PCM 帧（缓存 / 现合成 / 号码兜底）。"""

    def __init__(
        self,
        tts: Any,
        cache: Any,
        events: EventLogger,
        *,
        sample_rate: int = 16000,
        on_fallback: Any = None,
    ) -> None:
        self._tts = tts
        self._cache = cache
        self._events = events
        self._sample_rate = int(sample_rate)
        # CONFIRM 合成失败退回号码拼接时回调（`CallAgent` 据此改历史）。
        self.on_fallback = on_fallback

    async def synth(
        self,
        text: str,
        cancel: asyncio.Event,
        *,
        phone: str | None = None,
    ) -> AsyncIterator[rtc.AudioFrame]:
        if not text.strip():
            return

        cached = self._cache.get(text) if self._cache is not None else None
        self._events.log("tts_request", text=text, cached=cached is not None)
        if cached:
            self._events.log("tts_first_audio", text=text, latency_ms=0.0)
            for frame in self._stamp(cached, text):
                yield frame
            return

        started = monotonic_ms()
        sent = 0
        playing = False
        try:
            async for chunk in self._tts.synth(text, cancel):
                if not chunk:
                    continue
                if not playing:
                    self._events.log(
                        "tts_first_audio",
                        text=text,
                        latency_ms=max(0.0, monotonic_ms() - started),
                    )
                    playing = True
                sent += len(chunk)
                for frame in self._stamp(chunk, text):
                    yield frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 合成失败不该让整条链路挂掉
            self._events.log(
                "error", stage="tts", reason=type(exc).__name__, text=text, message=str(exc)[:200]
            )
            if playing or not phone:
                return
            if cancel.is_set():
                # 这一轮已经被抢话 / 接管：号码片段同样没播出，不能走兜底（T15d 要求 3）。
                return
            fallback = self._phone_fallback(text, phone)
            if fallback is None:
                return
            spoken, pcm = fallback
            self._events.log("tts_request", text=text, cached=True)
            self._events.log("tts_first_audio", text=text, latency_ms=0.0)
            if self.on_fallback is not None:
                self.on_fallback(spoken)
            for frame in self._stamp(pcm, spoken):
                yield frame
        finally:
            cancel.set()

    def _phone_fallback(self, text: str, phone: str) -> tuple[str, bytes] | None:
        try:
            pcm = self._cache.render_phone(phone)
        except (PhraseMissing, ValueError) as exc:
            self._events.log(
                "error", stage="tts", reason="phone_fallback_failed", message=str(exc)[:200]
            )
            return None
        self._events.log(
            "error", stage="tts", reason="confirm_synth_failed", text=text, phone=phone
        )
        return confirm_phone_text(phone), pcm

    def _stamp(self, pcm: bytes, text: str):
        for frame in frames_from_pcm(pcm, sample_rate=self._sample_rate):
            frame.userdata[TEXT_KEY] = text
            yield frame

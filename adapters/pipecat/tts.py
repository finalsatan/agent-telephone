"""T15：TTS 的 Pipecat 适配层（薄封装）。

Pipecat 的 `TTSService` 负责文本聚合、打断（`InterruptionFrame` 时取消当前合成）以及
`TTSStartedFrame` / `TTSAudioRawFrame` / `TTSStoppedFrame` 的产出；这里只实现
`run_tts()`：

1. 先查 T06 的预合成话术缓存（`core.phrases.PhraseCache`），命中就直接产出 PCM，
   不调网络——问候语、追问、兜底话术都走这条路，首包为 0（与 T10/T13 一致）；
2. 没命中再调 `services.tts` / `services.tts_local` 的 `synth()` 流式合成；
3. 把 T10 格式的 `tts_request` / `tts_first_audio` / `play_start` / `play_end` 事件写进
   同一份 `EventLogger`。`play_start` 在送出第一块音频时记，`play_end` 在整句送完时记，
   与 `core.call.CallSession` 的语义一致（服务端不等浏览器真正播完）。

打断：`TTSService` 在收到 `InterruptionFrame` 时会取消正在跑的 `run_tts()` 任务，
这里在 `finally` 里置位传给 `synth()` 的 cancel 事件，让底层 HTTP 流立刻关闭。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import TTSSettings

from core.events import EventLogger, monotonic_ms


class SiliconFlowTTSService(TTSService):
    """`TTSService` + 预合成话术缓存 + `services.tts`（云端/本地）流式合成。"""

    def __init__(
        self,
        tts: Any,
        cache: Any,
        events: EventLogger,
        *,
        model: str | None = None,
        voice: str | None = None,
        sample_rate: int = 16000,
        on_play_end: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._tts = tts
        self._cache = cache
        self._events = events
        # 自然播完时通知对话处理器“这句话说完了”（被打断时不通知，由它自己标记）。
        self._on_play_end = on_play_end
        self._settings = TTSSettings(model=model, voice=voice)
        # `self.sample_rate` 要等 Pipecat setup 后才有值；留一份构造时的值供单元测试。
        self._init_sample_rate = int(sample_rate)

    @property
    def _out_sample_rate(self) -> int:
        return int(self.sample_rate or self._init_sample_rate)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """把一句文本合成成 16kHz PCM，边合成边产出 `TTSAudioRawFrame`。"""
        if not text.strip():
            return

        cached = self._cache.get(text) if self._cache is not None else None
        self._events.log("tts_request", text=text, cached=cached is not None)

        if cached:
            self._events.log("tts_first_audio", text=text, latency_ms=0.0)
            self._events.log("play_start", text=text)
            for offset in range(0, len(cached), 4096):
                yield TTSAudioRawFrame(
                    audio=cached[offset : offset + 4096],
                    sample_rate=self._out_sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
            self._events.log("play_end", text=text, duration_ms=self._pcm_ms(len(cached)))
            if self._on_play_end is not None:
                self._on_play_end(text)
            return

        cancel = asyncio.Event()
        started = monotonic_ms()
        sent = 0
        playing = False
        interrupted = False
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
                    self._events.log("play_start", text=text)
                    playing = True
                sent += len(chunk)
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._out_sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except asyncio.CancelledError:
            # 被打断：任务被取消，finally 里置位 cancel 关闭底层流。
            # 这里不写 `interrupted` 事件：由 DialogProcessor 统一写（它能判断是
            # 正在播还是已经播完、并给历史打“（被打断）”，避免一条打断两条事件）。
            interrupted = True
            raise
        except Exception as exc:  # noqa: BLE001 - 合成失败不该让整条链路挂掉
            self._events.log(
                "error", stage="tts", reason=type(exc).__name__, text=text, message=str(exc)[:200]
            )
            # 合成失败 = 这句没播，通知对话处理器不要把它当成“在播/被打断”。
            if self._on_play_end is not None:
                self._on_play_end(text)
            return
        finally:
            cancel.set()
            if playing:
                self._events.log("play_end", text=text, duration_ms=self._pcm_ms(sent))
                if not interrupted and self._on_play_end is not None:
                    self._on_play_end(text)

    def _pcm_ms(self, byte_count: int) -> float:
        return byte_count / (2 * self._out_sample_rate) * 1000.0

"""T15：把服务层接进 Pipecat 管线的实时链路。

管线（与 T13 的级联链路对应）：

    InputAudioRawFrame
        → VADProcessor(SileroVADAnalyzer)            # 对应 T09 的 Silero VAD
        → SiliconFlowSTTService                       # ASR（整段上传）
        → UserTurnProcessor(Smart Turn 停止策略)      # 对应 T09 的 LocalSmartTurnAnalyzerV3
        → TurnEventLogger                             # vad_start / vad_end / turn_end
        → BargeInMonitor                              # 抢话：记 interrupted / 标历史
        → DialogProcessor                             # LLM + 状态机 + 选话术
        → SiliconFlowTTSService                       # 预合成缓存 / 流式合成
        → WsAudioSink                                 # 把 PCM 推给浏览器

`UserTurnProcessor` 放在 STT 之后：Smart Turn 停止策略要等一个 final 的转写才发
`UserStoppedSpeakingFrame`（与 Pipecat 官方级联模板 `input → stt → user_aggregator` 一致）；
STT 因此保留 `audio_passthrough=True`，把音频继续交给它跑 Smart Turn。

与 T13 的 `RealtimeCall` 的区别：

- T13 自己写了一个逐帧的 `TurnDetector`（内部也用 pipecat 的 Silero VAD / Smart Turn）；
  这里直接用 Pipecat 的 `VADProcessor` + `UserTurnProcessor`，轮次与打断由框架驱动。
- 服务层与事件流不变：三个适配层都调用 `services/` 与 `core/`，事件写进同一份
  `EventLogger`，字段与 T10 的 `CallSession` 对齐（见 `docs/pipecat-notes.md`）。

**参数**：VAD 阈值、静音确认时长、Smart Turn 阈值、`stop_secs` 全部从 `config.yaml`
的 `vad` 段读（T13c 调过的值），不在适配层写死。Pipecat 1.13 的 `LocalSmartTurnAnalyzerV3`
没有暴露概率阈值，所以这里用 `ConfiguredSmartTurn` 子类补上（见 notes 的“踩坑”）。

**T15c**：开口使用独立的 VAD 参数；Silero 同一次推理累计有声采样，只在 AI
播放或处理轮次时按页面阈值广播打断。STT 段编号确保一轮所有识别结果到齐再调 LLM；
TurnDoneFrame 经 TTS 音频队列抵达 sink 后转成 turn_done，供页面顺序回放。

"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from core.dialog import Dialog
from core.events import EventLogger
from core.playback import PlaybackTracker
from core.turn import TurnSpeechCounter
from adapters.pipecat.llm import DialogProcessor, TakeoverFrame, TurnDoneFrame
from adapters.pipecat.stt import SiliconFlowSTTService
from adapters.pipecat.tts import SiliconFlowTTSService
from tools.channel_sim import ChannelSimulator

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_INTERRUPT_MIN_SPEECH_MS = 300.0


def build_vad_params(vad_cfg: Mapping[str, Any], min_speech_ms: float) -> VADParams:
    """开口用独立 VAD 默认值；抢话阈值在 InterruptVADProcessor 中检查。"""
    return VADParams(
        confidence=float(vad_cfg.get("threshold", 0.5)),
        start_secs=float(vad_cfg.get("start_secs", VADParams().start_secs)),
        stop_secs=float(vad_cfg.get("speech_end_silence_ms", 600.0)) / 1000.0,
    )


class InterruptSilero(SileroVADAnalyzer):
    """复用同一次 Silero 推理，按当前轮内的语音帧累计打断时长。"""

    def __init__(self, **kwargs):
        self.speech_counter = TurnSpeechCounter()
        self._voice_confidence = 0.0
        super().__init__(**kwargs)

    @property
    def speech_ms(self) -> float:
        return self.speech_counter.speech_ms

    def reset_speech_counter(self) -> None:
        self.speech_counter.reset()

    def voice_confidence(self, buffer: bytes) -> float:
        self._voice_confidence = float(np.asarray(super().voice_confidence(buffer)).item())
        return self._voice_confidence

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = super()._get_smoothed_volume(audio)
        # Match TurnDetector: only VAD confidence decides whether a frame counts.
        # Low-confidence frames inside the same turn do not clear accumulated speech;
        # the counter is reset when UserTurnProcessor confirms turn_end.
        if self._voice_confidence >= self.params.confidence:
            self.speech_counter.add_voice_frame(
                len(audio) / (2 * self.sample_rate) * 1000.0
            )
        return volume


class InterruptVADProcessor(VADProcessor):
    """VAD 照常开口，只有忙碌且有声时长达到页面阈值才广播打断。"""

    def __init__(self, analyzer, min_speech_ms, is_busy, **kwargs):
        super().__init__(vad_analyzer=analyzer, **kwargs)
        self.analyzer = analyzer
        self.min_speech_ms = min_speech_ms
        self.is_busy = is_busy
        self._interrupted = False

    def reset_turn_speech(self) -> None:
        self.analyzer.reset_speech_counter()
        self._interrupted = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if not self.analyzer.speech_ms:
                self._interrupted = False
            elif (not self._interrupted and self.analyzer.speech_ms >= self.min_speech_ms
                  and self.is_busy()):
                self._interrupted = True
                await self.broadcast_interruption()


class ConfiguredSmartTurn(LocalSmartTurnAnalyzerV3):
    """把 `config.yaml: vad.smart_turn_threshold` 应用到 Pipecat 的 Smart Turn 上。

    Pipecat 1.13 的 `BaseSmartTurn._process_speech_segment()` 只看模型的布尔预测
    （内部按概率 0.5 判），没有暴露概率阈值。T13c 把阈值调到 0.85 才把误切率压到
    10% 以内，所以这里覆写 `_predict_endpoint()` 再按阈值收一次（与 `core.turn.SmartTurn`
    的做法一致）。

    T15b：记录最后一次结束的判定来源，供 `turn_end.reason` 区分 `smart_turn` / `stop_secs`。
    """

    def __init__(self, *, threshold: float = 0.85, **kwargs: Any) -> None:
        self._configured_threshold = float(threshold)
        self.last_reason: str | None = None
        self._was_speech = False
        super().__init__(**kwargs)

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        result = dict(super()._predict_endpoint(audio_array))
        probability = float(result.get("probability") or 0.0)
        result["prediction"] = int(
            bool(result.get("prediction")) and probability >= self._configured_threshold
        )
        return result

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        # 一段新的发声开始：清掉上一轮的来源，避免误标。
        if is_speech and not self._was_speech:
            self.last_reason = None
        self._was_speech = is_speech
        state = super().append_audio(buffer, is_speech)
        if state == EndOfTurnState.COMPLETE:
            # `append_audio` 只在静音超过 stop_secs 时返回 COMPLETE（stop_secs 兜底）。
            self.last_reason = "stop_secs"
        return state

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, Any]:
        state, metrics = await super().analyze_end_of_turn()
        if state == EndOfTurnState.COMPLETE and self.last_reason != "stop_secs":
            self.last_reason = "smart_turn"
        return state, metrics


class WsAudioSink(FrameProcessor):
    """管线末端：把 TTS 音频推给浏览器，并累计 AI 分轨录音。"""

    def __init__(
        self,
        send_audio: Callable[[bytes], None],
        *,
        on_audio: Callable[[float], None] | None = None,
        send_json: Callable[[Mapping[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._send_audio = send_audio
        self._on_audio = on_audio
        self._send_json = send_json
        self.ai_pcm = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            self.ai_pcm += frame.audio
            self._send_audio(frame.audio)
            if self._on_audio is not None:
                self._on_audio(len(frame.audio))
        elif isinstance(frame, TurnDoneFrame) and self._send_json is not None:
            self._send_json({"type": "turn_done", "kind": frame.kind, "text": frame.text,
                             "interrupted": frame.kind == "interrupted"})
        await self.push_frame(frame, direction)


class TurnEventLogger(FrameProcessor):
    """把 Pipecat 的 VAD / 轮次帧翻成 T10 格式的事件。

    T13 的 `TurnDetector` 会写 `vad_start` / `vad_end` / `turn_end`（含 Smart Turn 耗时）；
    Pipecat 的 VAD 与 Smart Turn 在框架内部，这里把它们暴露出来的帧补记成同样的事件，
    `tools/evaluate.py` / `tools/ws_client.py` 才能算时延。`smart_turn` 的推理耗时事件
    在 Pipecat 里拿不到（模型在框架内部线程里跑），见 `docs/pipecat-notes.md`。

    T15b：`turn_end.reason` 从 `ConfiguredSmartTurn.last_reason` 取（`smart_turn` /
    `stop_secs`），取不到时退回 `smart_turn`。
    """

    def __init__(
        self,
        events: EventLogger,
        *,
        smart_turn: ConfiguredSmartTurn | None = None,
        reset_turn_speech: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._events = events
        self._smart_turn = smart_turn
        self._reset_turn_speech = reset_turn_speech

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._events.log("vad_start", t_ms=None, speech_ms=None)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._events.log("vad_end", t_ms=None, speech_ms=None)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            reason = getattr(self._smart_turn, "last_reason", None) or "smart_turn"
            self._events.log("turn_end", t_ms=None, reason=reason, speech_ms=None)
            if self._reset_turn_speech is not None:
                self._reset_turn_speech()
        await self.push_frame(frame, direction)


class BargeInMonitor(FrameProcessor):
    """抢话监控：`InterruptionFrame` 到达时回调（T15b）。

    Pipecat 在来电者开口（VAD 达到 `start_secs`）时广播 `InterruptionFrame`。这里在
    `DialogProcessor` 之前截住，让 `PipecatCall` 用播放队列模型判断浏览器里是否还在播，
    是则补记 `interrupted(tail=True, unplayed=...)`。LLM 阶段（还没播）的抢话由
    `DialogProcessor` 自己写 `interrupted(stage="llm")`，两者互斥。
    """

    def __init__(self, on_interrupt: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._on_interrupt = on_interrupt

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._on_interrupt()
        await self.push_frame(frame, direction)


class PipecatCall:
    """一通实时电话（Pipecat 版），对外接口与 T13 的 `RealtimeCall` 一致。"""

    def __init__(
        self,
        *,
        call_dir: str | Path,
        config: Mapping[str, Any],
        services: Any,
        send_audio: Callable[[bytes], None],
        send_json: Callable[[Mapping[str, Any]], None],
        channel_sim: bool,
        interrupt_min_speech_ms: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        event_logger_factory: Callable[[Path, Callable[[Mapping[str, Any]], None]], EventLogger]
        | None = None,
    ) -> None:
        self.call_dir = Path(call_dir)
        self.call_id = self.call_dir.name
        self.config = config
        self.services = services
        self.channel_sim = bool(channel_sim)
        self.simulator = ChannelSimulator() if self.channel_sim else None
        self.min_speech_ms = float(interrupt_min_speech_ms)
        self.sample_rate = int(sample_rate)
        self._send_audio = send_audio
        self._send_json = send_json

        if event_logger_factory is None:
            from web.server import ForwardingEventLogger

            event_logger_factory = ForwardingEventLogger
        self.events = event_logger_factory(self.call_dir, self._on_event)

        self.dialog = Dialog.from_config(config)
        self.caller_pcm = bytearray()
        self.finished = False
        self._takeover = False
        self._started_event = asyncio.Event()
        self._worker_done = asyncio.Event()
        # 浏览器端待播队列的排队模型（T13b，与级联链路共用实现）。
        self.playback = PlaybackTracker()

        vad_cfg = config.get("vad") or {}
        self.vad = InterruptVADProcessor(
            InterruptSilero(params=build_vad_params(vad_cfg, self.min_speech_ms)),
            self.min_speech_ms,
            lambda: not self.finished and not self._takeover and
                    (self.playback.current_playing() is not None or self.dialog_processor.busy
                     or bool(self.tts.get_audio_contexts())),
        )
        smart_turn = ConfiguredSmartTurn(
            threshold=float(vad_cfg.get("smart_turn_threshold", 0.85)),
            sample_rate=self.sample_rate,
            params=self._smart_turn_params(vad_cfg),
        )
        self.smart_turn = smart_turn
        self.turn = UserTurnProcessor(
            user_turn_strategies=UserTurnStrategies(
                # VAD 只负责开口；打断由 InterruptVADProcessor 的独立阈值决定。
                start=[VADUserTurnStartStrategy(enable_interruptions=False)],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=smart_turn,
                        # ASR 失败/空转写时不会永远等转写，由 AsrFailedFrame 兜底。
                        wait_for_transcript=False,
                    )
                ],
            )
        )
        tts_cfg = config.get("tts") or {}
        self.stt = SiliconFlowSTTService(
            services.asr,
            self.events,
            model=(config.get("asr") or {}).get("model"),
            sample_rate=self.sample_rate,
            # 音频要留给后面的 UserTurnProcessor 跑 Smart Turn，所以不在这里吞掉。
            audio_passthrough=True,
        )
        self.turn_logger = TurnEventLogger(
            self.events,
            smart_turn=smart_turn,
            reset_turn_speech=self.vad.reset_turn_speech,
        )
        self.dialog_processor = DialogProcessor(
            services.llm,
            self.dialog,
            self.events,
            cache=services.cache,
            config=config,
        )
        self.barge_monitor = BargeInMonitor(self._on_barge_in)
        self.tts = SiliconFlowTTSService(
            services.tts,
            services.cache,
            self.events,
            model=tts_cfg.get("model"),
            voice=tts_cfg.get("voice"),
            sample_rate=self.sample_rate,
            on_fallback=self.dialog_processor.on_tts_fallback,
        )
        self.sink = WsAudioSink(self._send_audio, on_audio=self._on_audio, send_json=self._send_json)

        self.worker = PipelineWorker(
            Pipeline(
                [
                    self.vad,
                    self.stt,
                    self.turn,
                    self.turn_logger,
                    self.barge_monitor,
                    self.dialog_processor,
                    self.tts,
                    self.sink,
                ]
            ),
            name=f"pipecat-{self.call_id}",
        )
        self.runner = WorkerRunner(handle_sigint=False, handle_sigterm=False, force_gc=False)
        self._runner_task: asyncio.Task | None = None

        @self.worker.event_handler("on_pipeline_started")
        async def _on_started(_worker, _frame):
            self._started_event.set()

        @self.worker.event_handler("on_pipeline_finished")
        async def _on_finished(_worker, _frame):
            self._worker_done.set()

    @staticmethod
    def _smart_turn_params(vad_cfg: Mapping[str, Any]):
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

        return SmartTurnParams(
            stop_secs=float(vad_cfg.get("stop_secs", 2.0)),
            pre_speech_ms=float(vad_cfg.get("pre_speech_ms", 500.0)),
            max_duration_secs=float(vad_cfg.get("max_segment_s", 8.0)),
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """记 `call_start`、启动 Pipecat 管线，等 StartFrame 走完（问候语已排进 TTS）。"""
        self.events.log(
            "call_start", sample_rate=self.sample_rate, source="pipecat", channel_sim=self.channel_sim
        )
        await self.runner.add_workers(self.worker)
        self._runner_task = asyncio.create_task(self.runner.run(auto_end=False))
        try:
            await asyncio.wait_for(self._started_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            self.events.log("error", stage="pipeline", reason="start_timeout")
            await self.runner.end("start_timeout")
            if self._runner_task is not None:
                self._runner_task.cancel()
            raise

    async def feed(self, pcm: bytes) -> None:
        """喂一段来电者音频（16kHz、单声道、16-bit 小端）。"""
        if self.finished:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return
        if self.simulator is not None:
            pcm = self.simulator.feed(pcm)
            if not pcm:
                return
        # T15b：接管后来电者的音频照常录进 caller.wav（只是不再进管线、AI 不再回应）。
        self.caller_pcm += pcm
        if self._takeover:
            return
        await self.worker.queue_frame(
            InputAudioRawFrame(
                audio=pcm, sample_rate=self.sample_rate, num_channels=1
            )
        )

    def takeover(self) -> None:
        """用户接管：记 `takeover`，让对话处理器不再回应，并打断当前轮。"""
        if self.finished or self._takeover:
            return
        self._takeover = True
        self.dialog_processor.takeover()
        self.playback.reset()
        self.events.log("takeover", turns=self.dialog_processor.turn_count)
        self._schedule(self.worker.queue_frame(TakeoverFrame()))
        self._schedule(self.worker.queue_frame(InterruptionFrame()))

    async def finish(self) -> None:
        """挂断：打断当前轮、等管线收尾、写 slots.json / 分轨 wav / `call_end`。"""
        if self.finished:
            return
        self.finished = True
        self._takeover = True
        self.dialog_processor.takeover()
        self.playback.reset()
        try:
            await self.worker.queue_frame(InterruptionFrame())
            await self.worker.queue_frame(EndFrame(reason="hangup"))
            await asyncio.wait_for(self._worker_done.wait(), timeout=10.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        await self.runner.end("hangup")
        if self._runner_task is not None:
            try:
                await asyncio.wait_for(self._runner_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self._runner_task.cancel()
        if self.simulator is not None:
            self.simulator.flush()
        self._write_outputs()
        self.events.close()
        await self.services.aclose()

    def _schedule(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    # ------------------------------------------------------------------ #
    # 抢话
    # ------------------------------------------------------------------ #

    def _on_audio(self, byte_count: int) -> None:
        """WsAudioSink 每送出一块音频，把它加到浏览器播放队列的结束时刻上。"""
        self.playback.on_audio(byte_count / (2 * self.sample_rate) * 1000.0)

    def _on_barge_in(self) -> None:
        """`InterruptionFrame` 到达：若浏览器里还在播，补记尾音抢话事件。"""
        if self.finished or self._takeover:
            return
        segment = self.playback.current_playing()
        if segment is None:
            # LLM 阶段（还没播）：由 DialogProcessor 写 interrupted(stage="llm")。
            return
        text, played_ms, unplayed = self.playback.interrupt()
        self.events.log(
            "interrupted",
            reason="barge_in",
            text=text,
            played_ms=None if played_ms is None else round(played_ms, 2),
            tail=True,
            unplayed=unplayed,
        )
        self.dialog_processor.mark_playing_interrupted(text)

    # ------------------------------------------------------------------ #
    # 事件转发与输出
    # ------------------------------------------------------------------ #

    def _on_event(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        self._send_json({"type": kind, "t": round(float(event.get("t") or 0.0), 3), "data": data})

        if kind == "asr_final" and data.get("text"):
            self._send_json({"type": "transcript", "role": "caller", "text": data["text"]})
        elif kind == "play_start":
            text = str(data.get("text") or "")
            self.playback.on_play_start(text)
            self._send_json(
                {"type": "transcript", "role": "ai", "text": text, "interrupted": False}
            )
        elif kind == "interrupted" and data.get("tail") and data.get("text"):
            # 只有尾音抢话（AI 确实在播）才生成 AI 逐句；LLM 阶段的抢话没有 AI 输出，
            # 事件里的 text 是来电者的话，不能当成 AI 说的。
            payload: dict[str, Any] = {
                "type": "transcript",
                "role": "ai",
                "text": str(data.get("text") or ""),
                "interrupted": True,
            }
            unplayed = [str(item) for item in (data.get("unplayed") or []) if item]
            if unplayed:
                payload["unplayed"] = unplayed
            self._send_json(payload)
        if kind in ("call_start", "state_change", "slot_update", "takeover"):
            self._send_snapshot()

    def _send_snapshot(self) -> None:
        self._send_json({"type": "state", "state": self.dialog.state.value})
        self._send_json(
            {
                "type": "slots",
                "slots": {
                    name: {"value": value.value, "status": value.status}
                    for name, value in self.dialog.slots.items()
                },
            }
        )

    def _write_outputs(self) -> None:
        self.events.log(
            "call_end",
            state=self.dialog.state.value,
            turns=self.dialog_processor.turn_count,
            ai_audio_ms=self._pcm_ms(bytes(self.sink.ai_pcm)),
            caller_audio_ms=self._pcm_ms(bytes(self.caller_pcm)),
        )
        self._write_wav(self.call_dir / "caller.wav", bytes(self.caller_pcm))
        self._write_wav(self.call_dir / "ai.wav", bytes(self.sink.ai_pcm))
        slots = {
            name: {"value": value.value, "status": value.status}
            for name, value in self.dialog.slots.items()
        }
        payload = {"call_id": self.call_id, "state": self.dialog.state.value, "slots": slots}
        (self.call_dir / "slots.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _pcm_ms(self, pcm: bytes) -> float:
        return len(pcm) / (2 * self.sample_rate) * 1000.0

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            writer.writeframes(pcm)

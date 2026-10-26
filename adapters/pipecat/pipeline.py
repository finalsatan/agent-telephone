"""T15：把服务层接进 Pipecat 管线的实时链路。

管线（与 T13 的级联链路对应）：

    InputAudioRawFrame
        → VADProcessor(SileroVADAnalyzer)            # 对应 T09 的 Silero VAD
        → SiliconFlowSTTService                       # ASR（整段上传）
        → UserTurnProcessor(Smart Turn 停止策略)      # 对应 T09 的 LocalSmartTurnAnalyzerV3
        → TurnEventLogger                             # vad_start / vad_end / turn_end
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
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
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
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from core.dialog import Dialog
from core.events import EventLogger
from adapters.pipecat.llm import DialogProcessor, TakeoverFrame
from adapters.pipecat.stt import SiliconFlowSTTService
from adapters.pipecat.tts import SiliconFlowTTSService
from tools.channel_sim import ChannelSimulator

DEFAULT_SAMPLE_RATE = 16000


class ConfiguredSmartTurn(LocalSmartTurnAnalyzerV3):
    """把 `config.yaml: vad.smart_turn_threshold` 应用到 Pipecat 的 Smart Turn 上。

    Pipecat 1.13 的 `BaseSmartTurn._process_speech_segment()` 只看模型的布尔预测
    （内部按概率 0.5 判），没有暴露概率阈值。T13c 把阈值调到 0.85 才把误切率压到
    10% 以内，所以这里覆写 `_predict_endpoint()` 再按阈值收一次（与 `core.turn.SmartTurn`
    的做法一致）。
    """

    def __init__(self, *, threshold: float = 0.85, **kwargs: Any) -> None:
        self._configured_threshold = float(threshold)
        super().__init__(**kwargs)

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        result = dict(super()._predict_endpoint(audio_array))
        probability = float(result.get("probability") or 0.0)
        result["prediction"] = int(
            bool(result.get("prediction")) and probability >= self._configured_threshold
        )
        return result


class WsAudioSink(FrameProcessor):
    """管线末端：把 TTS 音频推给浏览器，并累计 AI 分轨录音。"""

    def __init__(self, send_audio: Callable[[bytes], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._send_audio = send_audio
        self.ai_pcm = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            self.ai_pcm += frame.audio
            self._send_audio(frame.audio)
        await self.push_frame(frame, direction)


class TurnEventLogger(FrameProcessor):
    """把 Pipecat 的 VAD / 轮次帧翻成 T10 格式的事件。

    T13 的 `TurnDetector` 会写 `vad_start` / `vad_end` / `turn_end`（含 Smart Turn 耗时）；
    Pipecat 的 VAD 与 Smart Turn 在框架内部，这里把它们暴露出来的帧补记成同样的事件，
    `tools/evaluate.py` / `tools/ws_client.py` 才能算时延。`smart_turn` 的推理耗时事件
    在 Pipecat 里拿不到（模型在框架内部线程里跑），见 `docs/pipecat-notes.md`。
    """

    def __init__(self, events: EventLogger, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._events = events

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._events.log("vad_start", t_ms=None, speech_ms=None)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._events.log("vad_end", t_ms=None, speech_ms=None)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._events.log("turn_end", t_ms=None, reason="smart_turn", speech_ms=None)
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

        vad_cfg = config.get("vad") or {}
        self.vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=float(vad_cfg.get("threshold", 0.5)),
                    stop_secs=float(vad_cfg.get("speech_end_silence_ms", 600.0)) / 1000.0,
                )
            )
        )
        smart_turn = ConfiguredSmartTurn(
            threshold=float(vad_cfg.get("smart_turn_threshold", 0.85)),
            sample_rate=self.sample_rate,
            params=self._smart_turn_params(vad_cfg),
        )
        self.turn = UserTurnProcessor(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)]
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
        self.turn_logger = TurnEventLogger(self.events)
        self.dialog_processor = DialogProcessor(services.llm, self.dialog, self.events)
        self.tts = SiliconFlowTTSService(
            services.tts,
            services.cache,
            self.events,
            model=tts_cfg.get("model"),
            voice=tts_cfg.get("voice"),
            sample_rate=self.sample_rate,
            on_play_end=self.dialog_processor.on_play_end,
        )
        self.sink = WsAudioSink(self._send_audio)

        self.worker = PipelineWorker(
            Pipeline(
                [
                    self.vad,
                    self.stt,
                    self.turn,
                    self.turn_logger,
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
        if self.finished or self._takeover:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return
        if self.simulator is not None:
            pcm = self.simulator.feed(pcm)
            if not pcm:
                return
        self.caller_pcm += pcm
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
        self.events.log("takeover", turns=None)
        self._schedule(self.worker.queue_frame(TakeoverFrame()))
        self._schedule(self.worker.queue_frame(InterruptionFrame()))

    async def finish(self) -> None:
        """挂断：打断当前轮、等管线收尾、写 slots.json / 分轨 wav / `call_end`。"""
        if self.finished:
            return
        self.finished = True
        self._takeover = True
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
    # 事件转发与输出
    # ------------------------------------------------------------------ #

    def _on_event(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        self._send_json({"type": kind, "t": round(float(event.get("t") or 0.0), 3), "data": data})

        if kind == "asr_final" and data.get("text"):
            self._send_json({"type": "transcript", "role": "caller", "text": data["text"]})
        elif kind == "play_start":
            self._send_json(
                {
                    "type": "transcript",
                    "role": "ai",
                    "text": str(data.get("text") or ""),
                    "interrupted": False,
                }
            )
        elif kind == "interrupted" and data.get("text"):
            self._send_json(
                {
                    "type": "transcript",
                    "role": "ai",
                    "text": str(data.get("text") or ""),
                    "interrupted": True,
                }
            )
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
            turns=None,
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

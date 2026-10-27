"""T16：一通 LiveKit 电话（`LiveKitCall`）。

对外接口与 T13 的 `RealtimeCall`、T15 的 `PipecatCall` 完全一致
（`start` / `feed` / `takeover` / `finish` / `finished` / `events` / `services` / `dialog`），
所以 `web/server.py` 的路由、事件转发、`slots.json`、分轨 wav 都能直接复用。

接线（与 notes 1.1 的结论一致）：

    QueueAudioInput ──► AgentSession ──► PacedAudioSink
                          │
                          ├── silero VAD（本地 ONNX，config: vad.*；只喂 STT 的分段与用户状态）
                          ├── SiliconFlowSTT（非流式 → 框架自动 StreamAdapter 按 VAD 切段）
                          └── CallAgent（Dialog + services.llm + services.tts + PhraseCache）

    LiveKitCall.feed()  ──► core.turn.TurnDetector（Silero + Smart Turn + stop_secs，与级联同一套）
                              └── TURN_END → session.commit_user_turn() → on_user_turn_completed

**不连 livekit-server**：`session.start(agent)` 不传 `room`，音频走自定义 input/output。
管线级测试与线上跑的是同一份代码（notes 1.1）。

**为什么用 `turn_detection="manual"` + 自己提交轮次**（notes 1.3 与 5 节）：
框架的轮次判断靠 VAD 的 `END_OF_SPEECH` 触发，而本项目的 ASR 是非流式的
（`StreamAdapter` 在这一刻才发起识别），所以转写永远比 VAD 事件晚到。单段发声时框架会
用 `trigger="stt"` 补一次 EOU，但**一轮多段发声**（Smart Turn 判第一段没说完）时框架会在
第二段的 VAD EOS 上拿第一段的旧转写提交本轮，后半句被挤到下一轮——这正是 T15c 第 1 条
的问题。改成手动提交后：我们自己的 `TurnDetector` 判出 `turn_end`，再
`commit_user_turn()` 让框架把 STT 冲干净、等这一轮所有段的转写都回来，才提交本轮。
打断也用同一套 `TurnSpeechCounter`（与级联、T15 的语义一致），见 `_maybe_barge_in()`。

框架仍然负责：分段合成（`FlushSentinel`）、播放调度、每段的播放进度
（`PacedAudioSink` 的 `playback_position`）、打断时取消合成、`say()` / `generate_reply()`
的生命周期。
"""

from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from livekit.agents import AgentSession, TurnHandlingOptions

from adapters.livekit.audio import PacedAudioSink, QueueAudioInput
from adapters.livekit.dialog import CallAgent
from adapters.livekit.nodes import NodeLLM, NodeTTS
from adapters.livekit.stt import SiliconFlowSTT
from adapters.livekit.tts import ReplySpeaker
from core.dialog import Dialog
from core.events import EventLogger
from core.history import INTERRUPTED_SUFFIX
from core.turn import SPEECH_END, SPEECH_START, TURN_END, TurnDetector
from tools.channel_sim import ChannelSimulator

DEFAULT_SAMPLE_RATE = 16000
# 提交轮次时等转写的上限（含 STT 冲缓冲），与 T03b 的 ASR 总时限一致。
DEFAULT_TRANSCRIPT_TIMEOUT_S = 5.0


def build_vad(config: Mapping[str, Any]):
    """按 `config.yaml: vad` 建 Silero VAD（本地 ONNX，不联网）。

    只用于两件事：给 `StreamAdapter` 切 STT 的段、以及框架的 `user_state_changed`。
    轮次判断由 `core.turn.TurnDetector` 自己做（`feed()` 里）。
    """
    from livekit.plugins import silero

    section = config.get("vad") or {}
    return silero.VAD.load(
        activation_threshold=float(section.get("threshold", 0.5)),
        min_silence_duration=float(section.get("speech_end_silence_ms", 600.0)) / 1000.0,
        prefix_padding_duration=float(section.get("pre_speech_ms", 500.0)) / 1000.0,
        sample_rate=DEFAULT_SAMPLE_RATE,
        force_cpu=True,
    )


def build_turn_handling(config: Mapping[str, Any]) -> TurnHandlingOptions:
    """框架只做播放与打断的“执行”，轮次由我们手动提交（`turn_detection="manual"`）。

    - `turn_detection="manual"`：框架不再自己跑 EOU（我们的 ASR 非流式，见模块 docstring）。
    - `endpointing.min_delay`：提交轮次后框架再等一小段（与级联的 `speech_end_silence_ms` 同量级），
      实际等待时间会因为 `last_speaking_time` 的回溯而接近 0。
    - `interruption`：`enabled=True` 保证 `session.interrupt()` 能打断当前这句（框架在手动模式下
      不再自己按 VAD 打断，打断由 `_maybe_barge_in()` 按 `TurnSpeechCounter` 触发）；
      `mode="vad"` 显式写死，避免 dev 模式下拉云端自适应打断；不自动续播（与 T13/T15 一致）。
    - `preemptive_generation` 关闭：回复要先由 `Dialog` 更新状态再决定，提前生成与之冲突。
    """
    vad = config.get("vad") or {}
    return TurnHandlingOptions(
        turn_detection="manual",
        endpointing={
            "min_delay": float(vad.get("speech_end_silence_ms", 600.0)) / 1000.0,
            "max_delay": float(vad.get("stop_secs", 2.0)),
        },
        interruption={
            "enabled": True,
            "mode": "vad",
            "resume_false_interruption": False,
            "false_interruption_timeout": None,
        },
        preemptive_generation={"enabled": False},
    )


class LiveKitCall:
    """一通实时电话（LiveKit 版）。"""

    def __init__(
        self,
        *,
        call_dir: str | Path,
        config: Mapping[str, Any],
        services: Any,
        send_audio: Callable[[bytes], None] | None = None,
        send_json: Callable[[Mapping[str, Any]], None],
        channel_sim: bool,
        interrupt_min_speech_ms: float | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        detector: Any = None,
        vad: Any = None,
        event_logger_factory: Callable[[Path, Callable[[Mapping[str, Any]], None]], EventLogger]
        | None = None,
    ) -> None:
        self.call_dir = Path(call_dir)
        self.call_id = self.call_dir.name
        self.config = config
        self.services = services
        self.sample_rate = int(sample_rate)
        self.channel_sim = bool(channel_sim)
        self.simulator = ChannelSimulator() if self.channel_sim else None
        self.min_speech_ms = float(
            interrupt_min_speech_ms
            if interrupt_min_speech_ms is not None
            else (config.get("interrupt") or {}).get("min_speech_ms", 300.0)
        )
        self.transcript_timeout = float(
            (config.get("asr") or {}).get("total_timeout_s", DEFAULT_TRANSCRIPT_TIMEOUT_S)
        )
        # 浏览器页面走 LiveKit 的房间音轨；`lk_client.py` / 测试直接读这里。
        self._send_audio = send_audio
        self._send_json = send_json

        if event_logger_factory is None:
            from web.server import ForwardingEventLogger

            event_logger_factory = ForwardingEventLogger
        self.events = event_logger_factory(self.call_dir, self._on_event)

        self.dialog = Dialog.from_config(config)
        # 与级联（T13 的 `RealtimeCall`）同一套轮次判断。
        self.detector = detector if detector is not None else TurnDetector.from_config(config)
        self.caller_pcm = bytearray()
        self.ai_pcm = bytearray()
        self.finished = False
        self._takeover = False
        self._barged_in = False
        # AI 正在处理这一轮（提交轮次 / 等转写 / 等 LLM）或音频还在播。
        self._busy = False
        self._playing = False
        self._turn_tasks: set[asyncio.Task] = set()
        # 提交轮次要串行：`commit_user_turn()` 会取消上一次未完成的提交。
        self._commit_lock = asyncio.Lock()

        self.input = QueueAudioInput(sample_rate=self.sample_rate)
        self.sink = PacedAudioSink(
            sample_rate=self.sample_rate,
            label="livekit",
            on_play_start=self._on_play_start,
            on_play_end=self._on_play_end,
            on_frame=self._on_ai_frame,
        )
        self.speaker = ReplySpeaker(
            services.tts,
            services.cache,
            self.events,
            sample_rate=self.sample_rate,
        )
        self.agent = CallAgent(
            llm=services.llm,
            dialog=self.dialog,
            events=self.events,
            speaker=self.speaker,
            config=config,
            on_turn_done=self._on_turn_done,
        )
        self.vad = vad if vad is not None else build_vad(config)
        self.stt = SiliconFlowSTT(
            services.asr,
            self.events,
            model=(config.get("asr") or {}).get("model"),
            sample_rate=self.sample_rate,
        )
        self.session = AgentSession(
            vad=self.vad,
            stt=self.stt,
            # 文本与音频实际由 CallAgent 的 llm_node() / tts_node() 产出；
            # 这两个占位实现只为满足框架的前置检查（见 nodes.py 的说明）。
            llm=NodeLLM(model=(config.get("llm") or {}).get("realtime_model")),
            tts=NodeTTS(
                model=(config.get("tts") or {}).get("model"), sample_rate=self.sample_rate
            ),
            turn_handling=build_turn_handling(config),
        )
        self.session.input.audio = self.input
        self.session.output.audio = self.sink
        self._watch_tasks: set[asyncio.Task] = set()
        self._started = False

        self.session.on("speech_created", self._on_speech_created)
        self.session.on("user_state_changed", self._on_user_state_changed)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """记 `call_start` 并启动会话（问候语在 `CallAgent.on_enter` 里播出）。"""
        if self._started:
            return
        self._started = True
        self.sink.start()
        self.events.log(
            "call_start",
            sample_rate=self.sample_rate,
            source="livekit",
            channel_sim=self.channel_sim,
        )
        await self.session.start(self.agent)

    async def feed(self, pcm: bytes) -> None:
        """喂一段来电者音频（16kHz、单声道、16-bit 小端）。

        与级联一样：先过信道模拟，再交给轮次检测器，最后推进 STT／框架。
        """
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
        # 接管后来电者的音频照常录进 caller.wav（只是不再进管线、AI 不再回应）。
        self.caller_pcm += pcm
        if self._takeover:
            return

        for event in self.detector.feed(pcm):
            self._on_turn_event(event)
        self._maybe_barge_in()
        self.input.push_pcm(pcm)

    def takeover(self) -> None:
        """用户接管：记 `takeover`、打断当前轮，之后 AI 不再出声。"""
        if self.finished or self._takeover:
            return
        self._takeover = True
        self.agent.request_takeover("takeover")
        self.events.log("takeover", turns=self.agent.turn_count)
        self._schedule(self.session.interrupt())

    async def finish(self) -> None:
        """挂断：打断当前轮、等排队轮次收尾、写 slots.json / 分轨 wav / `call_end`。"""
        if self.finished:
            return
        self.finished = True
        self._takeover = True
        self.agent.request_takeover("hangup")
        try:
            await asyncio.wait_for(self.session.interrupt(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        pending = [task for task in self._turn_tasks if not task.done()]
        if pending:
            _done, still_running = await asyncio.wait(pending, timeout=5.0)
            for task in still_running:
                task.cancel()
            if still_running:
                await asyncio.gather(*still_running, return_exceptions=True)
        self.input.close()
        try:
            await asyncio.wait_for(self.session.aclose(), timeout=10.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        for task in list(self._watch_tasks):
            task.cancel()
        if self._watch_tasks:
            await asyncio.gather(*self._watch_tasks, return_exceptions=True)
        await self.sink.aclose()
        if self.simulator is not None:
            self.simulator.flush()
        self._write_outputs()
        self.events.close()
        await self.services.aclose()

    # ------------------------------------------------------------------ #
    # 轮次与打断（与 T13 的 RealtimeCall 同一套语义）
    # ------------------------------------------------------------------ #

    def _on_turn_event(self, event: Any) -> None:
        if event.kind == SPEECH_START:
            self.events.log("vad_start", t_ms=event.t_ms, speech_ms=event.speech_ms)
        elif event.kind == SPEECH_END:
            self.events.log("vad_end", t_ms=event.t_ms, speech_ms=event.speech_ms)
            if self.detector.last_smart_turn_ms is not None:
                self.events.log(
                    "smart_turn",
                    latency_ms=round(self.detector.last_smart_turn_ms, 2),
                    offload=self.detector.smart_turn_offload,
                )
        elif event.kind == TURN_END:
            self.events.log(
                "turn_end", t_ms=event.t_ms, reason=event.reason, speech_ms=event.speech_ms
            )
            self._schedule_turn()

    def _schedule_turn(self) -> None:
        task = asyncio.create_task(self._commit_turn())
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _commit_turn(self) -> None:
        """`turn_end` 后提交本轮：让框架把 STT 冲干净、等所有段的转写回来再交给 LLM。"""
        if self.finished or self._takeover:
            self._send_json({"type": "turn_done", "kind": "skipped"})
            return
        self._busy = True
        try:
            async with self._commit_lock:
                await self.session.commit_user_turn(
                    transcript_timeout=self.transcript_timeout,
                )
        except Exception as exc:  # noqa: BLE001 - 单轮异常不该让整通电话挂掉
            self.events.log(
                "error", stage="turn", reason=type(exc).__name__, message=str(exc)[:200]
            )
        finally:
            self._busy = False

    def _maybe_barge_in(self) -> None:
        """来电者累计有声时长达到阈值，且 AI 正在说话／还在处理本轮 → 打断。

        用 `TurnDetector.turn_speech_ms`（只计 VAD 判为语音的帧）而不是“距开口的时长”，
        否则一声短噪加上后面的停顿也会凑够 300ms（与 T13 一致）。
        """
        if self._barged_in or self.finished or self._takeover:
            return
        if not (self._busy or self._playing or self.agent.busy):
            return
        if self.detector.turn_speech_ms < self.min_speech_ms:
            return
        self._barged_in = True
        # 本轮还在处理（LLM 阶段）时：文本顺延到下一轮，不播这一轮。
        self.agent.notify_user_speaking()
        self._schedule(self.session.interrupt())

    # ------------------------------------------------------------------ #
    # 会话事件
    # ------------------------------------------------------------------ #

    def _on_speech_created(self, ev: Any) -> None:
        if not self.agent.awaiting_turn_end:
            return
        task = asyncio.create_task(self._watch_speech(ev.speech_handle))
        self._watch_tasks.add(task)
        task.add_done_callback(self._watch_tasks.discard)

    async def _watch_speech(self, handle: Any) -> None:
        """等这一轮的话全部播完（或被抢话），再收尾。"""
        try:
            await handle.wait_for_playout()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 播完的收尾不该因为异常丢掉
            pass
        self._barged_in = False
        self.agent.finish_speech(interrupted=bool(handle.interrupted))

    def _on_user_state_changed(self, ev: Any) -> None:
        if str(getattr(ev, "new_state", "")) == "speaking":
            self.agent.notify_user_speaking()

    # ------------------------------------------------------------------ #
    # 播放回调
    # ------------------------------------------------------------------ #

    def _on_play_start(self, text: str) -> None:
        self._playing = True
        self.agent.on_play_start(text)

    def _on_play_end(self, text: str, played_s: float, interrupted: bool) -> None:
        self._playing = False
        self.agent.on_play_end(text, played_s, interrupted)

    def _on_ai_frame(self, frame: Any) -> None:
        self.ai_pcm += bytes(frame.data)
        if self._send_audio is not None:
            self._send_audio(bytes(frame.data))

    def _schedule(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    # ------------------------------------------------------------------ #
    # 事件转发与输出
    # ------------------------------------------------------------------ #

    def _on_turn_done(self, kind: str, text: str) -> None:
        self._send_json(
            {
                "type": "turn_done",
                "kind": kind,
                "text": text,
                "interrupted": kind == "interrupted",
            }
        )

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
        elif kind == "interrupted" and data.get("tail") and data.get("text"):
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
            turns=self.agent.turn_count,
            ai_audio_ms=self._pcm_ms(bytes(self.ai_pcm)),
            caller_audio_ms=self._pcm_ms(bytes(self.caller_pcm)),
        )
        self._write_wav(self.call_dir / "caller.wav", bytes(self.caller_pcm))
        self._write_wav(self.call_dir / "ai.wav", bytes(self.ai_pcm))
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


__all__ = ["LiveKitCall", "build_turn_handling", "build_vad", "INTERRUPTED_SUFFIX"]

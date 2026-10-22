"""T09：VAD 与轮次判断。

链路：16kHz PCM → Silero VAD（32ms 帧，512 采样）→ Smart Turn v3.2（ONNX）→ 事件。

模型来源与复用方式（详见 docs/progress.md）：

- Silero VAD 用 pipecat 自带的 `silero_vad.onnx`（与 silero-vad 官方 ONNX 模型一致），
  推理复用 `pipecat.audio.vad.silero.SileroOnnxModel`。没有安装 `silero-vad` 包，因为它
  硬依赖 torch，而本项目只需要 ONNX；pipecat 已经是依赖（T15 也要用）。
- Smart Turn 用 pipecat 1.13 的 `LocalSmartTurnAnalyzerV3`（内含 `smart-turn-v3.2-cpu.onnx`
  与 Whisper log-mel 的 numpy 实现），本项目只调用它的 `_predict_endpoint()`。
  requirements 里把 pipecat-ai 固定到 1.13.0；升级前要确认该私有方法仍在。

事件（任务书里写的 `Event`，这里叫 `TurnEvent` 以免和 core/events.py 的日志事件混淆）：

- `speech_start`：一段发声开始（每次停顿后重新开口都会再发一次，对应 Pipecat 的 vad_start）。
- `speech_end`：连续静音达到 `speech_end_silence_ms`（默认 200ms），一段发声结束。
- `turn_end`：一整轮结束。`reason="smart_turn"` 表示 Smart Turn 判为说完；
  `reason="stop_secs"` 表示 Smart Turn 判为没说完、但静音已超过 `stop_secs` 的兜底。

`t_ms` 是**音频时间轴**上的毫秒（从第一次 feed 算起，按 32ms 帧累加），不是墙钟：
离线回放要按它切片，在线链路由调用方用 `EventLogger` 再记 `time.monotonic()`。

帧边界：`feed()` 接受任意长度的 PCM 字节，内部按 512 采样（1024 字节）切帧，余数留到
下一次 feed；不用凑整也能用。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Protocol

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 512
FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit 单声道

SPEECH_START = "speech_start"
SPEECH_END = "speech_end"
TURN_END = "turn_end"
EVENT_KINDS: frozenset[str] = frozenset({SPEECH_START, SPEECH_END, TURN_END})

DEFAULT_THRESHOLD = 0.5
# T13c：Smart Turn 判为“说完”的概率阈值（模型自带的布尔结果固定按 0.5 判）。
DEFAULT_SMART_TURN_THRESHOLD = 0.5
DEFAULT_SPEECH_END_SILENCE_MS = 200.0
DEFAULT_STOP_SECS = 3.0
DEFAULT_PRE_SPEECH_MS = 500.0
DEFAULT_MAX_SEGMENT_S = 8.0
# T13b：Smart Turn 单次推理超过该值就改到线程里跑（见 TurnDetector.feed_async）。
DEFAULT_SMART_TURN_OFFLOAD_MS = 20.0


@dataclass(frozen=True)
class TurnEvent:
    """轮次检测事件。`speech_ms` 为本轮累计的有效语音时长（只计 VAD 判为语音的帧）。"""

    kind: str
    t_ms: float
    speech_ms: float = 0.0
    probability: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"invalid event kind: {self.kind!r}, expected one of {sorted(EVENT_KINDS)}")


class VadModel(Protocol):
    """逐帧 VAD 模型。实现方必须保证 `frame` 恰好是一帧（32ms / 512 采样）。"""

    def speech_probability(self, frame: bytes) -> float: ...

    def reset(self) -> None: ...


class SmartTurnModel(Protocol):
    """轮次结束判定模型。`pcm` 为本轮的 16kHz 16-bit PCM。"""

    def predict(self, pcm: bytes) -> tuple[bool, float]: ...


class SileroVad:
    """Silero VAD 的 ONNX 版本（模型与推理包装复用 pipecat，见模块 docstring）。"""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"SileroVad 只支持 {SAMPLE_RATE}Hz，收到 {sample_rate}")

        # 延迟导入：只注入 fake 的测试不需要 onnxruntime。
        from importlib.resources import files

        from pipecat.audio.vad.silero import SileroOnnxModel

        model_path = str(files("pipecat.audio.vad.data").joinpath("silero_vad.onnx"))
        self._model = SileroOnnxModel(model_path)
        self.sample_rate = sample_rate

    def speech_probability(self, frame: bytes) -> float:
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        return float(self._model(x, self.sample_rate)[0][0])

    def reset(self) -> None:
        self._model.reset_states()


class SmartTurn:
    """Smart Turn v3.2 的 ONNX 推理（复用 pipecat 的模型文件与特征提取）。"""

    def __init__(self, *, model_path: str | None = None, cpu_count: int = 1) -> None:
        # 延迟导入：只注入 fake 的测试不需要 onnxruntime。
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        self._analyzer = LocalSmartTurnAnalyzerV3(
            sample_rate=SAMPLE_RATE,
            smart_turn_model_path=model_path,
            cpu_count=cpu_count,
        )

    def predict(self, pcm: bytes) -> tuple[bool, float]:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # pipecat 只公开了 async/帧驱动的接口；这里直接调用它内部的推理函数，
        # 复用已验证的 Whisper log-mel 特征与 ONNX 会话（pipecat 版本已固定）。
        result = self._analyzer._predict_endpoint(audio)
        return bool(result["prediction"]), float(result["probability"])


class TurnDetector:
    """逐帧喂音频，产出 speech_start / speech_end / turn_end 事件。

    状态机：

    1. 未进入轮次时保留最近 `pre_speech_ms` 的静音帧作为预卷，等第一个语音帧。
    2. 第一个语音帧 → `speech_start`，把预卷和后续帧收集成本轮音频段。
    3. 静音累计到 `speech_end_silence_ms` → `speech_end`，立刻跑一次 Smart Turn：
       - 判为说完（概率 ≥ `smart_turn_threshold`）→ `turn_end(reason="smart_turn")`，本轮结束；
       - 判为没说完 → 继续等，静音累计到 `stop_secs` → `turn_end(reason="stop_secs")`；
         期间对方重新开口则不结束本轮，只再发一次 `speech_start`。
    4. 语音中间的短停顿（< `speech_end_silence_ms`）不会触发 `speech_end`，更不会结束轮次。

    只做 CPU 推理，不发网络请求；模型可以注入，便于测试。
    """

    def __init__(
        self,
        *,
        vad: VadModel | None = None,
        smart_turn: SmartTurnModel | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        smart_turn_threshold: float = DEFAULT_SMART_TURN_THRESHOLD,
        speech_end_silence_ms: float = DEFAULT_SPEECH_END_SILENCE_MS,
        stop_secs: float = DEFAULT_STOP_SECS,
        pre_speech_ms: float = DEFAULT_PRE_SPEECH_MS,
        max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
        frame_ms: int = FRAME_MS,
        sample_rate: int = SAMPLE_RATE,
        smart_turn_offload_ms: float = DEFAULT_SMART_TURN_OFFLOAD_MS,
    ) -> None:
        if frame_ms != FRAME_MS or sample_rate != SAMPLE_RATE:
            raise ValueError(f"只支持 {FRAME_MS}ms 帧 / {SAMPLE_RATE}Hz（Silero VAD 的 16kHz 窗口）")

        self.threshold = threshold
        self.smart_turn_threshold = float(smart_turn_threshold)
        self.speech_end_silence_ms = speech_end_silence_ms
        self.stop_secs = stop_secs
        self.pre_speech_ms = pre_speech_ms
        self.max_segment_s = max_segment_s
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        # T13b：Smart Turn 推理在事件循环里同步跑，实测可能卡住整条实时链路。
        # 每次推理都计时；一旦超过阈值，后续 feed_async() 就改到线程里执行。
        self.smart_turn_offload_ms = float(smart_turn_offload_ms)
        self.last_smart_turn_ms: float | None = None
        self.last_smart_turn_probability: float | None = None
        self.smart_turn_offload = False

        # 真实模型加载较慢（约 0.1s + 8.6MB），注入 fake 时完全不碰 onnxruntime。
        self._vad: VadModel = vad if vad is not None else SileroVad(sample_rate)
        self._smart_turn: SmartTurnModel = smart_turn if smart_turn is not None else SmartTurn()

        self._max_pre_frames = max(1, int(pre_speech_ms // frame_ms))
        self._max_segment_frames = max(1, int(max_segment_s * 1000 // frame_ms))
        self._pending = bytearray()

        self._frame_index = 0
        self._pre_roll: Deque[bytes] = deque(maxlen=self._max_pre_frames)
        self._segment: Deque[bytes] = deque(maxlen=self._max_segment_frames)

        self._in_speech = False
        self._turn_active = False
        self._waiting_turn = False
        self._silence_ms = 0.0
        self._turn_speech_ms = 0.0

    # ---- 对外接口 ---------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        vad: VadModel | None = None,
        smart_turn: SmartTurnModel | None = None,
    ) -> "TurnDetector":
        """从 config.yaml 的 `vad` 段构造；模型参数可注入（测试用 fake）。"""
        section = config.get("vad") or {}
        return cls(
            vad=vad,
            smart_turn=smart_turn,
            threshold=float(section.get("threshold", DEFAULT_THRESHOLD)),
            smart_turn_threshold=float(
                section.get("smart_turn_threshold", DEFAULT_SMART_TURN_THRESHOLD)
            ),
            speech_end_silence_ms=float(
                section.get("speech_end_silence_ms", DEFAULT_SPEECH_END_SILENCE_MS)
            ),
            stop_secs=float(section.get("stop_secs", DEFAULT_STOP_SECS)),
            pre_speech_ms=float(section.get("pre_speech_ms", DEFAULT_PRE_SPEECH_MS)),
            max_segment_s=float(section.get("max_segment_s", DEFAULT_MAX_SEGMENT_S)),
        )

    def feed(self, frame: bytes) -> list[TurnEvent]:
        """喂入任意长度的 PCM 字节（16kHz、16-bit 小端、单声道），返回本次产生的事件。"""
        self._pending.extend(frame)
        events: list[TurnEvent] = []
        while len(self._pending) >= FRAME_BYTES:
            chunk = bytes(self._pending[:FRAME_BYTES])
            del self._pending[:FRAME_BYTES]
            events.extend(self._process_frame(chunk))
        return events

    async def feed_async(self, frame: bytes) -> list[TurnEvent]:
        """`feed()` 的异步版本，供实时链路用（T13b）。

        Smart Turn 实测超过 `smart_turn_offload_ms` 时，把整帧处理丢到线程里执行，
        避免阻塞事件循环；否则与原同步实现完全一致。`feed()` 本身仍是同步的，
        离线回放与测试继续直接调用它。
        """
        if self.smart_turn_offload:
            return await asyncio.to_thread(self.feed, frame)
        return self.feed(frame)

    def reset(self) -> None:
        """开始新一通电话时调用：清空模型状态与所有缓冲。"""
        self._pending.clear()
        self._pre_roll.clear()
        self._segment.clear()
        self._frame_index = 0
        self._in_speech = False
        self._turn_active = False
        self._waiting_turn = False
        self._silence_ms = 0.0
        self._turn_speech_ms = 0.0
        self._vad.reset()

    @property
    def in_speech(self) -> bool:
        """当前是否处在一次发声内（两次 speech_start / speech_end 之间）。"""
        return self._in_speech

    @property
    def turn_active(self) -> bool:
        """当前是否有一轮尚未 turn_end（含 Smart Turn 判为没说完、等兜底的阶段）。"""
        return self._turn_active

    @property
    def turn_speech_ms(self) -> float:
        """本轮累计的有效语音时长（只计 VAD 判为语音的帧，不含静音）。

        T13 的抢话阈值用它：不能用“距 speech_start 的音频时长”，否则一声短噪
        （100ms）加上后面的停顿也会凑够 300ms。
        """
        return self._turn_speech_ms

    # ---- 内部状态机 -------------------------------------------------------

    def _process_frame(self, frame: bytes) -> list[TurnEvent]:
        t_ms = float(self._frame_index * self.frame_ms)
        self._frame_index += 1

        probability = self._vad.speech_probability(frame)
        is_speech = probability >= self.threshold

        if not self._turn_active:
            # 还没开始一轮：缓存静音作为预卷，等第一个语音帧。
            if is_speech:
                return self._begin_turn(frame, t_ms)
            self._pre_roll.append(frame)
            return []

        # 本轮音频段：包含预卷、语音、静音，按需截断到 max_segment_s。
        self._segment.append(frame)

        if is_speech:
            return self._on_speech(t_ms)

        return self._on_silence(t_ms)

    def _begin_turn(self, frame: bytes, t_ms: float) -> list[TurnEvent]:
        self._turn_active = True
        self._in_speech = True
        self._waiting_turn = False
        self._silence_ms = 0.0
        self._turn_speech_ms = float(self.frame_ms)

        self._segment.clear()
        self._segment.extend(self._pre_roll)
        self._segment.append(frame)
        self._pre_roll.clear()

        return [TurnEvent(SPEECH_START, t_ms, self._turn_speech_ms)]

    def _on_speech(self, t_ms: float) -> list[TurnEvent]:
        self._turn_speech_ms += self.frame_ms
        self._silence_ms = 0.0

        if self._in_speech:
            return []

        # 停顿后重新开口：Smart Turn 判过没说完，或者停顿不足 speech_end 但 VAD 中间掉过。
        self._in_speech = True
        self._waiting_turn = False
        return [TurnEvent(SPEECH_START, t_ms, self._turn_speech_ms)]

    def _on_silence(self, t_ms: float) -> list[TurnEvent]:
        if self._in_speech:
            self._silence_ms += self.frame_ms
            if self._silence_ms < self.speech_end_silence_ms:
                return []

            self._in_speech = False
            events = [TurnEvent(SPEECH_END, t_ms, self._turn_speech_ms)]

            started = time.perf_counter()
            complete, probability = self._smart_turn.predict(self._segment_pcm())
            # T13c：模型自带的布尔结果按 0.5 判；阈值调高后要求概率也达到 smart_turn_threshold。
            complete = bool(complete) and probability >= self.smart_turn_threshold
            self.last_smart_turn_probability = probability
            self.last_smart_turn_ms = (time.perf_counter() - started) * 1000.0
            if self.last_smart_turn_ms > self.smart_turn_offload_ms:
                # 一旦观察到超时，后续整帧处理都改到线程里（T13b）。
                self.smart_turn_offload = True
            if complete:
                events.append(
                    TurnEvent(
                        TURN_END,
                        t_ms,
                        self._turn_speech_ms,
                        probability=probability,
                        reason="smart_turn",
                    )
                )
                self._finish_turn()
            else:
                self._waiting_turn = True
            return events

        if self._waiting_turn:
            self._silence_ms += self.frame_ms
            if self._silence_ms >= self.stop_secs * 1000:
                events = [
                    TurnEvent(
                        TURN_END,
                        t_ms,
                        self._turn_speech_ms,
                        reason="stop_secs",
                    )
                ]
                self._finish_turn()
                return events
        return []

    def _segment_pcm(self) -> bytes:
        return b"".join(self._segment)

    def _finish_turn(self) -> None:
        self._turn_active = False
        self._in_speech = False
        self._waiting_turn = False
        self._silence_ms = 0.0
        self._turn_speech_ms = 0.0
        self._segment.clear()
        self._pre_roll.clear()


def segment_wav(pcm: bytes, *, detector: TurnDetector | None = None) -> list[tuple[float, float]]:
    """把整段 16kHz PCM 切成轮次，返回 [(start_ms, end_ms), ...]。

    实现就是把 PCM 按帧喂给 `TurnDetector`，再在末尾补一段静音把未结束的轮次收尾
    （补的静音不会算进返回的 end_ms，end_ms 最多到音频结尾）。每轮的起点往前带
    `pre_speech_ms` 的预卷（不早于上一轮结束、不早于 0），避免切掉第一个字的起音。

    `detector` 可注入（测试用 fake）；默认用真实模型。
    """
    detector = detector if detector is not None else TurnDetector()
    detector.reset()

    duration_ms = len(pcm) / 2 / SAMPLE_RATE * 1000.0
    events: list[TurnEvent] = []
    for offset in range(0, len(pcm), FRAME_BYTES):
        events.extend(detector.feed(pcm[offset : offset + FRAME_BYTES]))

    # 音频在说话中结束：补静音，让 Smart Turn / stop_secs 兜底把最后一轮收掉。
    # 先把不足一帧的尾巴补齐，避免补的静音和原音频错位。
    remainder = (-len(pcm)) % FRAME_BYTES
    if remainder:
        events.extend(detector.feed(bytes(remainder)))
    pad_ms = detector.stop_secs * 1000.0 + 2.0 * detector.frame_ms
    pad_frames = int(pad_ms // detector.frame_ms)
    silence_frame = bytes(FRAME_BYTES)
    events.extend(detector.feed(silence_frame * pad_frames))

    segments: list[tuple[float, float]] = []
    start_ms: float | None = None
    for event in events:
        if event.kind == SPEECH_START and start_ms is None:
            start_ms = max(0.0, event.t_ms - detector.pre_speech_ms)
            if segments:
                start_ms = max(start_ms, segments[-1][1])
        elif event.kind == TURN_END and start_ms is not None:
            end_ms = min(event.t_ms, duration_ms)
            if end_ms > start_ms:
                segments.append((start_ms, end_ms))
            start_ms = None
    return segments

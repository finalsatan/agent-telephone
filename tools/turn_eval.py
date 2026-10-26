"""T13c：离线评估轮次切分（不调用任何网络服务）。

把全部场景的每个 `turn_XX.wav` 逐帧送进 `TurnDetector`，信道模拟开、关各一遍：

- **误切率**：一条 wav 是一句完整的话（一轮），出现超过一次 `turn_end` 就算误切；
- **结束延迟**：语音真正结束（`speech_end_ms()` 用 20ms RMS 估出的最后有声点）到
  **最后一次** `turn_end` 的时间，另含该帧本身的 32ms（帧收齐才能处理）；
- 每次 `turn_end` 的原因（`smart_turn` 或 `stop_secs` 兜底）与 Smart Turn 给出的概率。

用法：

    uv run python tools/turn_eval.py                 # 当前 config.yaml 参数，信道模拟开/关
    uv run python tools/turn_eval.py --sweep         # 参数扫描：静音确认 × Smart Turn 阈值
    uv run python tools/turn_eval.py --sweep --out report/turn_eval.json --verbose

    # 实时时延长尾：把真实通话的 events.jsonl 拆成 轮次判断 / ASR / LLM / TTS 首包
    uv run python tools/turn_eval.py --breakdown calls/web_xxx [calls/web_yyy ...] --slow-ms 5000

每条 wav 后面补 `stop_secs + 0.6s` 的数字静音（与 `ws_client.py` 补静音的做法一致），
让兜底也能走完。Silero 的逐帧概率对一条 wav 只算一次，Smart Turn 的结果按输入 PCM
缓存，所以扫描 18 种组合不会把推理重复 18 遍。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.turn import (  # noqa: E402
    FRAME_BYTES,
    FRAME_MS,
    TURN_END,
    SileroVad,
    SmartTurn,
    TurnDetector,
)
from services.asr import load_wav_pcm  # noqa: E402
from tools.channel_sim import simulate  # noqa: E402
from tools.replay import ROOT, discover_scenarios, scenario_playable  # noqa: E402
from tools.ws_client import percentile, speech_end_ms  # noqa: E402

DEFAULT_SILENCE_GRID = (200.0, 400.0, 600.0)
DEFAULT_THRESHOLD_GRID = (0.5, 0.7, 0.85)
# 选默认值的两条硬指标（任务书 T13c 要求 2）。
MAX_FALSE_CUT_RATE = 0.10
MAX_P50_INCREASE_MS = 300.0
# stop_secs 之外多补的静音，保证兜底的 turn_end 一定落在补的静音里。
TAIL_PAD_MS = 600.0


# --------------------------------------------------------------------------- #
# 可复用的模型包装
# --------------------------------------------------------------------------- #


class ReplayVad:
    """回放预先算好的逐帧 Silero 概率（概率只取决于音频，与轮次参数无关）。"""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._index = 0

    def speech_probability(self, frame: bytes) -> float:
        value = self._probabilities[self._index]
        self._index += 1
        return value

    def reset(self) -> None:
        self._index = 0


class CachingSmartTurn:
    """按输入 PCM 缓存 Smart Turn 的结果，并记录真正推理的耗时（毫秒）。"""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model if model is not None else SmartTurn()
        self._cache: dict[str, tuple[bool, float]] = {}
        self.inference_ms: list[float] = []
        self.calls: list[tuple[bool, float]] = []  # 最近一遍的逐次结果，供 run_wav 取用

    def predict(self, pcm: bytes) -> tuple[bool, float]:
        key = hashlib.sha1(pcm).hexdigest()
        result = self._cache.get(key)
        if result is None:
            started = time.perf_counter()
            result = self._model.predict(pcm)
            self.inference_ms.append((time.perf_counter() - started) * 1000.0)
            self._cache[key] = result
        self.calls.append(result)
        return result


def silero_probabilities(pcm: bytes, vad: Any | None = None) -> list[float]:
    """整段 PCM（长度已按帧对齐）的逐帧语音概率。"""
    vad = vad if vad is not None else SileroVad()
    vad.reset()
    return [
        vad.speech_probability(pcm[offset : offset + FRAME_BYTES])
        for offset in range(0, len(pcm), FRAME_BYTES)
    ]


# --------------------------------------------------------------------------- #
# 单条 wav 的评估
# --------------------------------------------------------------------------- #


@dataclass
class TurnEndRecord:
    t_ms: float
    reason: str
    probability: float | None


@dataclass
class WavResult:
    """一条 wav 在一组参数下的结果。"""

    name: str
    true_end_ms: float
    turn_ends: list[TurnEndRecord] = field(default_factory=list)
    # 每次 Smart Turn 推理（含判为没说完的）：(音频时间轴 ms, 概率, 是否判为说完)。
    smart_turn_calls: list[tuple[float, float, bool]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.turn_ends)

    @property
    def false_cut(self) -> bool:
        return self.count > 1

    @property
    def latency_ms(self) -> float | None:
        """语音真正结束 → 最后一次 turn_end（整句真正被切出来）的延迟。"""
        if not self.turn_ends:
            return None
        return self.turn_ends[-1].t_ms + FRAME_MS - self.true_end_ms

    @property
    def fallback_count(self) -> int:
        return sum(1 for record in self.turn_ends if record.reason == "stop_secs")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "true_end_ms": round(self.true_end_ms, 1),
            "turn_end_count": self.count,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "turn_ends": [
                {
                    "t_ms": record.t_ms,
                    "reason": record.reason,
                    "probability": None
                    if record.probability is None
                    else round(record.probability, 4),
                }
                for record in self.turn_ends
            ],
            "smart_turn_calls": [
                {"t_ms": t, "probability": round(p, 4), "complete": c}
                for t, p, c in self.smart_turn_calls
            ],
        }


@dataclass
class PreparedWav:
    """已算好 Silero 概率的一条 wav（某个信道模式下）。"""

    name: str
    true_end_ms: float
    pcm: bytes  # 已含尾部静音，且按帧对齐
    vad_probabilities: list[float]


def prepare_wav(
    name: str,
    pcm: bytes,
    *,
    channel_sim: bool,
    stop_secs: float,
    vad: Any | None = None,
) -> PreparedWav:
    """补尾部静音、(可选)过信道、算 Silero 概率。真正的语音结束点按原始 PCM 估。"""
    true_end = speech_end_ms(pcm)
    audio = simulate(pcm) if channel_sim else pcm
    pad_bytes = int((stop_secs * 1000.0 + TAIL_PAD_MS) // FRAME_MS) * FRAME_BYTES
    audio = audio + bytes((-len(audio)) % FRAME_BYTES) + bytes(pad_bytes)
    return PreparedWav(name, true_end, audio, silero_probabilities(audio, vad))


def run_wav(
    prepared: PreparedWav,
    smart_turn: CachingSmartTurn,
    *,
    speech_end_silence_ms: float,
    smart_turn_threshold: float,
    stop_secs: float,
    vad_threshold: float,
    pre_speech_ms: float,
    max_segment_s: float,
) -> WavResult:
    detector = TurnDetector(
        vad=ReplayVad(prepared.vad_probabilities),
        smart_turn=smart_turn,
        threshold=vad_threshold,
        smart_turn_threshold=smart_turn_threshold,
        speech_end_silence_ms=speech_end_silence_ms,
        stop_secs=stop_secs,
        pre_speech_ms=pre_speech_ms,
        max_segment_s=max_segment_s,
    )
    result = WavResult(prepared.name, prepared.true_end_ms)
    pcm = prepared.pcm
    for offset in range(0, len(pcm), FRAME_BYTES):
        smart_turn.calls.clear()
        events = detector.feed(pcm[offset : offset + FRAME_BYTES])
        t_ms = float(offset // FRAME_BYTES * FRAME_MS)
        for _, probability in smart_turn.calls:
            result.smart_turn_calls.append(
                (t_ms, probability, probability >= smart_turn_threshold)
            )
        for event in events:
            if event.kind == TURN_END:
                result.turn_ends.append(TurnEndRecord(event.t_ms, event.reason, event.probability))
    return result


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


@dataclass
class Summary:
    channel_sim: bool
    speech_end_silence_ms: float
    smart_turn_threshold: float
    wavs: int
    false_cuts: int
    no_turn_end: int
    fallbacks: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None

    @property
    def false_cut_rate(self) -> float:
        return self.false_cuts / self.wavs if self.wavs else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "channel_sim": self.channel_sim,
            "speech_end_silence_ms": self.speech_end_silence_ms,
            "smart_turn_threshold": self.smart_turn_threshold,
            "wavs": self.wavs,
            "false_cuts": self.false_cuts,
            "false_cut_rate": round(self.false_cut_rate, 4),
            "no_turn_end": self.no_turn_end,
            "stop_secs_fallbacks": self.fallbacks,
            "latency_p50_ms": None if self.latency_p50_ms is None else round(self.latency_p50_ms, 1),
            "latency_p95_ms": None if self.latency_p95_ms is None else round(self.latency_p95_ms, 1),
        }


def summarize(
    results: list[WavResult],
    *,
    channel_sim: bool,
    speech_end_silence_ms: float,
    smart_turn_threshold: float,
) -> Summary:
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    return Summary(
        channel_sim=channel_sim,
        speech_end_silence_ms=speech_end_silence_ms,
        smart_turn_threshold=smart_turn_threshold,
        wavs=len(results),
        false_cuts=sum(1 for r in results if r.false_cut),
        no_turn_end=sum(1 for r in results if r.count == 0),
        fallbacks=sum(r.fallback_count for r in results),
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
    )


def select_default(
    summaries: list[Summary],
    *,
    baseline_silence_ms: float,
    baseline_threshold: float,
    max_false_cut_rate: float = MAX_FALSE_CUT_RATE,
    max_p50_increase_ms: float = MAX_P50_INCREASE_MS,
) -> list[tuple[float, float]]:
    """返回同时满足两条指标的 (静音确认 ms, Smart Turn 阈值)，按 P50 延迟从低到高排序。

    指标要在信道模拟开、关两种模式下**都**满足；P50 的增量相对 (baseline_silence_ms,
    baseline_threshold) 在同一信道模式下的结果算。
    """

    def key(s: Summary) -> tuple[bool, float, float]:
        return (s.channel_sim, s.speech_end_silence_ms, s.smart_turn_threshold)

    by_key = {key(s): s for s in summaries}
    combos = sorted({(s.speech_end_silence_ms, s.smart_turn_threshold) for s in summaries})
    passing: list[tuple[float, tuple[float, float]]] = []
    for silence, threshold in combos:
        worst_p50 = 0.0
        ok = True
        for channel in (False, True):
            current = by_key.get((channel, silence, threshold))
            base = by_key.get((channel, baseline_silence_ms, baseline_threshold))
            if current is None or base is None or current.latency_p50_ms is None:
                ok = False
                break
            if current.false_cut_rate > max_false_cut_rate:
                ok = False
                break
            if base.latency_p50_ms is None:
                ok = False
                break
            increase = current.latency_p50_ms - base.latency_p50_ms
            if increase > max_p50_increase_ms:
                ok = False
                break
            worst_p50 = max(worst_p50, current.latency_p50_ms)
        if ok:
            passing.append((worst_p50, (silence, threshold)))
    return [combo for _, combo in sorted(passing)]


# --------------------------------------------------------------------------- #
# 数据与命令行
# --------------------------------------------------------------------------- #


def collect_wavs(scenarios_dir: Path) -> list[tuple[str, Path]]:
    """全部场景的每个 turn wav（按解析后的路径去重，抢话变体复用的不重复算）。"""
    seen: set[Path] = set()
    wavs: list[tuple[str, Path]] = []
    for scenario in discover_scenarios(scenarios_dir):
        if not scenario_playable(scenario) or not scenario.turns:
            continue
        for index, turn in enumerate(scenario.turns, start=1):
            path = turn.wav.resolve()
            if path in seen:
                continue
            seen.add(path)
            wavs.append((f"{scenario.id}/{path.name}", path))
    return wavs


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def evaluate(
    wavs: list[tuple[str, Path]],
    config: dict[str, Any],
    *,
    silences: tuple[float, ...],
    thresholds: tuple[float, ...],
    channels: tuple[bool, ...] = (False, True),
    progress: bool = False,
) -> tuple[list[Summary], dict[tuple[bool, float, float], list[WavResult]], CachingSmartTurn]:
    vad_cfg = config.get("vad") or {}
    stop_secs = float(vad_cfg.get("stop_secs", 3.0))
    vad_threshold = float(vad_cfg.get("threshold", 0.5))
    pre_speech_ms = float(vad_cfg.get("pre_speech_ms", 500))
    max_segment_s = float(vad_cfg.get("max_segment_s", 8.0))

    vad = SileroVad()
    smart_turn = CachingSmartTurn()
    pcms = [(name, load_wav_pcm(path)) for name, path in wavs]

    summaries: list[Summary] = []
    detail: dict[tuple[bool, float, float], list[WavResult]] = {}
    for channel in channels:
        prepared = [
            prepare_wav(name, pcm, channel_sim=channel, stop_secs=stop_secs, vad=vad)
            for name, pcm in pcms
        ]
        for silence in silences:
            for threshold in thresholds:
                results = [
                    run_wav(
                        item,
                        smart_turn,
                        speech_end_silence_ms=silence,
                        smart_turn_threshold=threshold,
                        stop_secs=stop_secs,
                        vad_threshold=vad_threshold,
                        pre_speech_ms=pre_speech_ms,
                        max_segment_s=max_segment_s,
                    )
                    for item in prepared
                ]
                detail[(channel, silence, threshold)] = results
                summaries.append(
                    summarize(
                        results,
                        channel_sim=channel,
                        speech_end_silence_ms=silence,
                        smart_turn_threshold=threshold,
                    )
                )
                if progress:
                    print(
                        f"  channel={'on ' if channel else 'off'} silence={silence:.0f} "
                        f"threshold={threshold}: 误切 {summaries[-1].false_cuts}/{summaries[-1].wavs}",
                        file=sys.stderr,
                    )
    return summaries, detail, smart_turn


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def format_table(summaries: list[Summary]) -> str:
    lines = [
        "| 信道模拟 | 静音确认(ms) | Smart Turn 阈值 | 误切(条) | 误切率 | 兜底次数 | 结束延迟 P50(ms) | P95(ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {'开' if s.channel_sim else '关'} | {s.speech_end_silence_ms:.0f} | "
            f"{s.smart_turn_threshold} | {s.false_cuts}/{s.wavs} | {s.false_cut_rate:.1%} | "
            f"{s.fallbacks} | {_fmt(s.latency_p50_ms)} | {_fmt(s.latency_p95_ms)} |"
        )
    return "\n".join(lines)


def format_detail(results: list[WavResult]) -> str:
    lines = ["| wav | 语音结束(ms) | turn_end 次数 | 各次 turn_end (t_ms 原因 概率) | 各次 Smart Turn 概率 |", "|---|---|---|---|---|"]
    for r in results:
        ends = "; ".join(
            f"{e.t_ms:.0f} {e.reason} {'-' if e.probability is None else f'{e.probability:.2f}'}"
            for e in r.turn_ends
        )
        probs = " ".join(f"{p:.2f}{'*' if c else ''}" for _, p, c in r.smart_turn_calls)
        lines.append(f"| {r.name} | {r.true_end_ms:.0f} | {r.count} | {ends} | {probs} |")
    return "\n".join(lines)


def format_inference(inference_ms: list[float]) -> str:
    if not inference_ms:
        return "Smart Turn 推理次数 0"
    return (
        f"Smart Turn 单次推理（缓存未命中的真实推理 {len(inference_ms)} 次）："
        f"P50 {percentile(inference_ms, 0.5):.1f}ms / P95 {percentile(inference_ms, 0.95):.1f}ms / "
        f"max {max(inference_ms):.1f}ms；超过 20ms 的 {sum(1 for v in inference_ms if v > 20.0)} 次"
    )


async def measure_detector_inference(
    wavs: list[tuple[str, Path]], config: dict[str, Any]
) -> dict[str, Any]:
    """用真实 `TurnDetector.feed_async()` 逐帧跑一遍，记录 Smart Turn 单次耗时与是否转线程。

    和扫描里的 `CachingSmartTurn` 不同，这里每次都是真实推理，也走实时链路用的
    异步入口：detector 自己计时（`last_smart_turn_ms`），超过 `smart_turn_offload_ms`
    （20ms）后 `smart_turn_offload` 变 True、后续帧改用 `asyncio.to_thread`。
    """
    detector = TurnDetector.from_config(config)
    timings: list[float] = []
    first_offload_at: int | None = None
    for _, path in wavs:
        detector.reset()
        pcm = load_wav_pcm(path)
        pcm += bytes((-len(pcm)) % FRAME_BYTES) + bytes(int(detector.stop_secs * 1000 // FRAME_MS + 4) * FRAME_BYTES)
        previous = detector.last_smart_turn_ms
        for offset in range(0, len(pcm), FRAME_BYTES):
            await detector.feed_async(pcm[offset : offset + FRAME_BYTES])
            current = detector.last_smart_turn_ms
            if current is not None and current != previous:
                timings.append(current)
                previous = current
                if detector.smart_turn_offload and first_offload_at is None:
                    first_offload_at = len(timings)
    return {
        "count": len(timings),
        "p50_ms": percentile(timings, 0.5),
        "p95_ms": percentile(timings, 0.95),
        "max_ms": max(timings) if timings else None,
        "over_20ms": sum(1 for v in timings if v > detector.smart_turn_offload_ms),
        "offload_triggered": detector.smart_turn_offload,
        "offload_after_inference_no": first_offload_at,
        "first_ms": timings[0] if timings else None,
    }


# --------------------------------------------------------------------------- #
# 实时时延长尾：events.jsonl → 每个回复的分段耗时（T13c 要求 3）
# --------------------------------------------------------------------------- #


@dataclass
class ResponseRow:
    """一个回复（从来电者说完到 AI 第一句开始播放）的分段耗时，毫秒。"""

    call: str
    turn_judge_ms: float | None  # 估计：说完 → turn_end（静音确认 + Smart Turn + 兜底等待）
    turn_reason: str
    asr_ms: float | None
    llm_ms: float | None
    tts_first_ms: float | None
    redo: int  # 因抢话/轮次合并被丢弃重做的次数
    total_ms: float | None  # 最后一次说完 → AI 第一句 play_start
    chain_ms: float | None  # 第一次 turn_end → play_start（含重做白费的时间）
    text: str = ""


def latency_breakdown(
    events: list[dict[str, Any]], *, call: str = "", speech_end_silence_ms: float = 200.0
) -> list[ResponseRow]:
    """把一通电话的事件流拆成回复行。

    - 说完时刻估为 `vad_end.t - speech_end_silence_ms`（vad_end 在静音累计够时才发）；
    - 一个回复到「来电者说完之后第一个 play_start」为止；
    - 在 play_start 之前出现的 `interrupted`（LLM 阶段抢话、turn_merged）算一次重做，
      该次的 ASR/LLM/TTS 数据丢弃，只保留最后一次尝试；
    - 问候语（通话开头、没有 turn_end 在前）不产生行。
    """
    rows: list[ResponseRow] = []
    vad_end_t: float | None = None
    turn_end_t: float | None = None
    turn_reason = ""
    chain_start: float | None = None
    redo = 0
    marks: dict[str, float] = {}

    def reset_attempt() -> None:
        marks.clear()

    for event in events:
        kind = event.get("type")
        t = float(event.get("t") or 0.0)
        data = event.get("data") or {}
        if kind == "vad_end":
            vad_end_t = t
        elif kind == "turn_end":
            turn_end_t = t
            turn_reason = str(data.get("reason") or "")
            if chain_start is None:
                chain_start = t
            # T15b：Pipecat 路径的 asr_request 可能在 turn_end 之前发出（STT 在轮次处理器
            # 上游），asr_final 在之后。把已记录的 ASR 标记带到这一轮，否则 ASR 一栏为空。
            carried = {k: v for k, v in marks.items() if k in ("asr_request", "asr_final")}
            reset_attempt()
            marks.update(carried)
        elif kind in ("asr_request", "asr_final", "llm_request", "llm_done"):
            marks.setdefault(kind, t)
        elif kind == "tts_request":
            if "llm_request" in marks:
                marks.setdefault("tts_request", t)
        elif kind == "tts_first_audio":
            if "tts_request" in marks:
                marks.setdefault("tts_first_audio", t)
        elif kind == "interrupted":
            if turn_end_t is not None:  # 还没开播就被丢弃：这一次的处理白做了
                redo += 1
                reset_attempt()
        elif kind == "play_start":
            if turn_end_t is None:
                continue  # 问候语等不是对来电者的回复
            end_t = (vad_end_t - speech_end_silence_ms) if vad_end_t is not None else None
            rows.append(
                ResponseRow(
                    call=call,
                    turn_judge_ms=None if end_t is None else turn_end_t - end_t,
                    turn_reason=turn_reason,
                    asr_ms=_span(marks, "asr_request", "asr_final"),
                    llm_ms=_span(marks, "llm_request", "llm_done"),
                    tts_first_ms=_span(marks, "tts_request", "tts_first_audio"),
                    redo=redo,
                    total_ms=None if end_t is None else t - end_t,
                    chain_ms=None if chain_start is None else t - chain_start,
                    text=str(data.get("text") or ""),
                )
            )
            turn_end_t = vad_end_t = chain_start = None
            redo = 0
            reset_attempt()
    return rows


def _span(marks: dict[str, float], start: str, end: str) -> float | None:
    if start in marks and end in marks and marks[end] >= marks[start]:
        return marks[end] - marks[start]
    return None


def format_breakdown(rows: list[ResponseRow], *, slow_ms: float) -> str:
    def col(values: list[float | None]) -> str:
        data = [v for v in values if v is not None]
        if not data:
            return "-"
        return f"{percentile(data, 0.5):.0f} / {percentile(data, 0.95):.0f}"

    lines = [
        f"回复 {len(rows)} 个；总等待 P50/P95 = {col([r.total_ms for r in rows])} ms",
        "分段 P50/P95(ms)：轮次判断 "
        f"{col([r.turn_judge_ms for r in rows])}；ASR {col([r.asr_ms for r in rows])}；"
        f"LLM {col([r.llm_ms for r in rows])}；TTS 首包 {col([r.tts_first_ms for r in rows])}",
        f"重做 {sum(r.redo for r in rows)} 次；turn_end 原因：smart_turn "
        f"{sum(1 for r in rows if r.turn_reason == 'smart_turn')} / stop_secs "
        f"{sum(1 for r in rows if r.turn_reason == 'stop_secs')}",
        "",
        f"总等待 > {slow_ms:.0f}ms 的回复：",
        "| 通话 | 总等待 | 轮次判断(原因) | ASR | LLM | TTS 首包 | 重做 | 链条总长 | 文本 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.total_ms is None or r.total_ms <= slow_ms:
            continue
        lines.append(
            f"| {r.call} | {_fmt(r.total_ms)} | {_fmt(r.turn_judge_ms)}({r.turn_reason}) | "
            f"{_fmt(r.asr_ms)} | {_fmt(r.llm_ms)} | {_fmt(r.tts_first_ms)} | {r.redo} | "
            f"{_fmt(r.chain_ms)} | {r.text[:20]} |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T13c：离线评估轮次切分（不联网）")
    parser.add_argument("--scenarios", default=str(ROOT / "scenarios"))
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--sweep", action="store_true", help="扫描静音确认 × Smart Turn 阈值")
    parser.add_argument("--silence-ms", type=float, nargs="+", help="覆盖静音确认取值")
    parser.add_argument("--smart-turn-threshold", type=float, nargs="+", help="覆盖阈值取值")
    parser.add_argument("--stop-secs", type=float, help="覆盖 vad.stop_secs（兜底静音，秒）")
    parser.add_argument(
        "--measure", action="store_true", help="用真实 feed_async 测 Smart Turn 单次耗时与是否转线程"
    )
    parser.add_argument(
        "--breakdown", nargs="+", metavar="CALL_DIR", help="拆解这些通话目录的实时时延（不跑轮次评估）"
    )
    parser.add_argument("--slow-ms", type=float, default=5000.0, help="--breakdown 列出超过该值的回复")
    parser.add_argument("--verbose", action="store_true", help="逐条 wav 打印明细")
    parser.add_argument("--out", help="把汇总与明细写成 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    if args.stop_secs is not None:
        config.setdefault("vad", {})["stop_secs"] = args.stop_secs
    vad_cfg = config.get("vad") or {}
    current = (
        float(vad_cfg.get("speech_end_silence_ms", 200)),
        float(vad_cfg.get("smart_turn_threshold", 0.5)),
    )
    if args.sweep:
        silences = tuple(args.silence_ms or DEFAULT_SILENCE_GRID)
        thresholds = tuple(args.smart_turn_threshold or DEFAULT_THRESHOLD_GRID)
    else:
        silences = tuple(args.silence_ms or (current[0],))
        thresholds = tuple(args.smart_turn_threshold or (current[1],))

    if args.breakdown:
        from core.events import read_events

        rows: list[ResponseRow] = []
        for call_dir in args.breakdown:
            rows += latency_breakdown(
                read_events(call_dir),
                call=Path(call_dir).name,
                speech_end_silence_ms=current[0],
            )
        print(format_breakdown(rows, slow_ms=args.slow_ms))
        return 0

    wavs = collect_wavs(Path(args.scenarios))
    if not wavs:
        print("没有找到场景 wav", file=sys.stderr)
        return 2
    print(f"wav 数 {len(wavs)}；静音确认 {silences}；阈值 {thresholds}", file=sys.stderr)

    summaries, detail, smart_turn = evaluate(
        wavs, config, silences=silences, thresholds=thresholds, progress=args.sweep
    )
    print(format_table(summaries))
    print()
    print(format_inference(smart_turn.inference_ms))

    if args.measure:
        import asyncio

        measured = asyncio.run(measure_detector_inference(wavs, config))
        print()
        print("TurnDetector.feed_async 实测（真实推理，每次 speech_end 一次）：")
        print(json.dumps(measured, ensure_ascii=False, indent=2))

    base = (200.0, 0.5)
    if args.sweep and (False, *base) in {
        (s.channel_sim, s.speech_end_silence_ms, s.smart_turn_threshold) for s in summaries
    }:
        winners = select_default(
            summaries, baseline_silence_ms=base[0], baseline_threshold=base[1]
        )
        print()
        if winners:
            print("满足 误切率≤10% 且 P50 增量≤300ms（信道开/关都满足）的组合（P50 从低到高）：")
            for silence, threshold in winners:
                print(f"  speech_end_silence_ms={silence:.0f}, smart_turn_threshold={threshold}")
        else:
            print("没有组合同时满足 误切率≤10% 与 P50 增量≤300ms，需要用户在上表里取舍。")

    if args.verbose:
        for (channel, silence, threshold), results in detail.items():
            print(f"\n## 信道模拟{'开' if channel else '关'} 静音确认 {silence:.0f}ms 阈值 {threshold}")
            print(format_detail(results))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summaries": [s.to_json() for s in summaries],
            "inference_ms": smart_turn.inference_ms,
            "detail": {
                f"channel={int(c)},silence={s:.0f},threshold={t}": [r.to_json() for r in results]
                for (c, s, t), results in detail.items()
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n明细写入 {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""T13b：按场景节奏的实时客户端（真实 WebSocket，实时速度）。

用法：

    # 启动服务端（另开一个终端）
    uv run python -m web.server

    # 跑一条场景：按 32ms 一帧、实时速度把每轮 wav 发给 WS /ws/call
    uv run python tools/ws_client.py --scenario scenarios/s03_appointment.json

    # 跑全部场景，汇总时延
    uv run python tools/ws_client.py --all --out report/ws_latency.json

    # 抢话变体：turn_04 的 gap_ms 为负，在确认语播到约 1 秒时开口
    uv run python tools/ws_client.py --scenario scenarios/s03b_barge_in.json

设计：

- **发送节奏**：所有帧（含等待期间的静音帧）按统一的 32ms 时钟发送，模拟真实来电。
  每轮 wav 的「语音结束点」由 `speech_end_ms()` 在 PCM 上估出（最后一段高于阈值的
  采样），客户端据此得到「来电者说完」的墙钟时刻。
- **下一轮开口时刻**：客户端自己维护一个播放游标 `playback_cursor_ms`。收到 AI 音频
  时 `start = max(now, cursor)`、`cursor = start + 音频时长`，即「排队模型」估出它在
  客户端的播完时刻。下一轮在 `cursor + gap_ms` 开口；`gap_ms` 为负表示在 AI 播完前
  就开口（抢话）。
- **实时时延**：每轮记录「来电者说完」到「收到 AI 第一块音频」的墙钟差，这是包含
  轮次判断（VAD 静音确认 + Smart Turn）在内的真实等待时间。
- **抢话实测**：`interrupted` 事件到达时记录「从来电者开口到收到 interrupted」的时间。

`--out` 写出的 JSON 含每轮明细、时延统计与 `call_id`，可直接对照服务端
`calls/<call_id>/events.jsonl`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.asr import load_wav_pcm  # noqa: E402
from tools.replay import (  # noqa: E402
    ROOT,
    Scenario,
    discover_scenarios,
    load_scenario,
)

FRAME_MS = 32
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_URL = "ws://127.0.0.1:8000/ws/call"
DEFAULT_LINGER_MS = 5000.0
DEFAULT_SCENARIOS_DIR = ROOT / "scenarios"
# 判断「静音」的 int16 RMS 阈值（与 config.yaml: tts.phrase_cache.trim_threshold 同量级）。
SILENCE_THRESHOLD = 200.0
# 等待问候语音频的上限；超过就直接开始第一轮（服务端没出声时不能卡死）。
GREETING_WAIT_S = 15.0
# 挂断后等待 call_ended 的上限。
HANGUP_WAIT_S = 10.0
# 服务端安静多久算「这一轮处理完了」（VAD 会把一轮 wav 切成多段，段间有 ASR/LLM/TTS
# 间隔，最长可能 5s）。
IDLE_SETTLE_MS = 5000.0
# 等服务端安静的最长时间。
IDLE_TIMEOUT_S = 40.0
# 抢话变体：等匹配的 AI 句子出现的最长时间。
BARGE_IN_START_TIMEOUT_S = 40.0


def now_ms() -> float:
    """单调时钟，毫秒（与 core.events.monotonic_ms 同一基准）。"""
    return time.monotonic() * 1000.0


def pcm_ms(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """16-bit 单声道 PCM 的时长（毫秒）。"""
    return len(pcm) / 2.0 / sample_rate * 1000.0


def speech_end_ms(
    pcm: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    threshold: float = SILENCE_THRESHOLD,
    frame_ms: float = 20.0,
) -> float:
    """估一段 16k 单声道 16-bit PCM 的「语音结束点」（毫秒）。

    以固定长度的滑动帧算 RMS，返回最后一个高于阈值的帧的结束时刻；整段都低于阈值
    时返回 0。场景 wav 是 TTS 合成、末尾带静音，这个方法足够稳定；不需要加载 VAD
    模型，客户端可以纯离线地算出「来电者说完」的时刻。
    """
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return 0.0
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    last = 0
    for start in range(0, samples.size, frame):
        chunk = samples[start : start + frame]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > threshold:
            last = start + chunk.size
    return last / sample_rate * 1000.0


def percentile(values: list[float], pct: float) -> float | None:
    """线性插值分位数；空列表返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


@dataclass
class TurnPlan:
    """一轮的发送计划。"""

    index: int
    text: str
    wav: Path
    gap_ms: float
    pcm: bytes
    speech_end_ms: float
    # 抢话变体（T13b）：等 AI 说出匹配 `barge_in_match` 的句子后，再等 `barge_in_ms`
    # 才开口。比负 `gap_ms` 可靠——确认语的时长会变，负 gap 有时会落在句尾之后。
    barge_in_ms: float | None = None
    barge_in_match: str | None = None

    @property
    def duration_ms(self) -> float:
        return pcm_ms(self.pcm)


@dataclass
class TurnMeasurement:
    """一轮的实测时延。"""

    index: int
    text: str
    speech_start_at: float | None = None
    speech_end_at: float | None = None
    first_audio_at: float | None = None
    latency_ms: float | None = None
    interrupted_at: float | None = None
    interrupted_latency_ms: float | None = None
    interrupted_text: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "speech_end_ms": None
            if self.speech_end_at is None or self.speech_start_at is None
            else round(self.speech_end_at - self.speech_start_at, 2),
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 2),
            "interrupted": self.interrupted_at is not None,
            "interrupted_latency_ms": None
            if self.interrupted_latency_ms is None
            else round(self.interrupted_latency_ms, 2),
            "interrupted_text": self.interrupted_text,
        }


class ScenarioWsClient:
    """把一条场景按真实节奏通过 `WS /ws/call` 跑完。"""

    def __init__(
        self,
        scenario: Scenario,
        *,
        url: str = DEFAULT_URL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        gap_override: float | None = None,
        linger_ms: float = DEFAULT_LINGER_MS,
        idle_settle_ms: float = IDLE_SETTLE_MS,
        log=print,
    ) -> None:
        self.scenario = scenario
        self.url = url
        self.sample_rate = int(sample_rate)
        self.frame_ms = float(frame_ms)
        self.frame_bytes = int(sample_rate * frame_ms / 1000.0) * 2
        self.gap_override = gap_override
        self.linger_ms = float(linger_ms)
        self.idle_settle_ms = float(idle_settle_ms)
        self._log = log

        self.plans = self._build_plans()
        self.measurements = [
            TurnMeasurement(index=plan.index, text=plan.text) for plan in self.plans
        ]
        self.events: list[dict[str, Any]] = []
        self.call_id: str | None = None

        # 客户端播放游标：收到 AI 音频后按排队模型推进（毫秒，单调时钟）。
        self.playback_cursor_ms = 0.0
        self.audio_started = asyncio.Event()
        self.call_ended = asyncio.Event()
        self.turn_done_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending_turn: int | None = None
        self.active_turn: int | None = None
        # 服务端最近一次活动（收到音频或 turn_done）的时刻，用来判「一轮处理完了」。
        self.last_activity_at = 0.0
        # 抢话变体：匹配 `barge_in_match` 的 AI 句子开始播放的时刻。
        self.matched_play_start_at: float | None = None

        self._ws: Any = None
        self._next_frame_at = 0.0

    # ---- 计划 -------------------------------------------------------------

    def _build_plans(self) -> list[TurnPlan]:
        # 抢话变体需要从原始 json 读可选字段（`tools.replay.load_scenario` 不解析它们）。
        raw_turns: list[dict[str, Any]] = []
        try:
            raw = json.loads(self.scenario.path.read_text(encoding="utf-8"))
            raw_turns = [
                dict(item) for item in (raw.get("turns") or []) if isinstance(item, dict)
            ]
        except (OSError, json.JSONDecodeError):
            raw_turns = []

        plans: list[TurnPlan] = []
        for index, turn in enumerate(self.scenario.turns):
            pcm = load_wav_pcm(turn.wav, self.sample_rate)
            gap = self.gap_override if self.gap_override is not None else turn.gap_ms
            extra = raw_turns[index] if index < len(raw_turns) else {}
            barge_in_ms = extra.get("barge_in_ms")
            plans.append(
                TurnPlan(
                    index=index,
                    text=turn.text,
                    wav=turn.wav,
                    gap_ms=float(gap),
                    pcm=pcm,
                    speech_end_ms=speech_end_ms(pcm, self.sample_rate),
                    barge_in_ms=None if barge_in_ms is None else float(barge_in_ms),
                    barge_in_match=(
                        str(extra["barge_in_match"]) if extra.get("barge_in_match") else None
                    ),
                )
            )
        return plans

    # ---- 主流程 -----------------------------------------------------------

    async def run(self) -> list[TurnMeasurement]:
        import websockets

        async with websockets.connect(self.url, max_size=None) as ws:
            self._ws = ws
            await ws.send(json.dumps({"type": "start"}))
            receiver = asyncio.create_task(self._recv_loop())
            try:
                await self._send_loop()
                await asyncio.sleep(self.linger_ms / 1000.0)
                await ws.send(json.dumps({"type": "hangup"}))
                try:
                    await asyncio.wait_for(self.call_ended.wait(), timeout=HANGUP_WAIT_S)
                except asyncio.TimeoutError:
                    self._log("[warn] 挂断后未收到 call_ended")
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        return self.measurements

    async def _send_loop(self) -> None:
        self._next_frame_at = now_ms()
        try:
            await asyncio.wait_for(self.audio_started.wait(), timeout=GREETING_WAIT_S)
        except asyncio.TimeoutError:
            self._log("[warn] 未收到问候语音频，直接开始第一轮")
        for index, plan in enumerate(self.plans):
            if plan.barge_in_ms is not None:
                await self._wait_for_barge_in(plan)
            else:
                await self._wait_for_gap(plan)
            await self._send_turn(plan)
            # 下一轮要抢话时不能等服务端空闲（否则确认语早播完了），
            # 交给 _wait_for_barge_in 按确认语的播放起点计时。
            next_plan = self.plans[index + 1] if index + 1 < len(self.plans) else None
            if next_plan is not None and next_plan.barge_in_ms is not None:
                continue
            # 等服务端把这一轮 wav 处理完（VAD 可能把它切成多段，段间有间隔，所以
            # 用「安静了 IDLE_SETTLE_MS」判结束），再算下一轮的开口时刻。
            await self._wait_until_idle(plan)
        self.active_turn = None

    async def _wait_for_barge_in(self, plan: TurnPlan) -> None:
        """等 AI 说出匹配 `barge_in_match` 的句子，再等 `barge_in_ms` 后开口（抢话）。"""
        silence = bytes(self.frame_bytes)
        deadline = now_ms() + BARGE_IN_START_TIMEOUT_S * 1000.0
        # 等匹配的句子开始播（`play_start` 事件里带文本）。
        while now_ms() < deadline and self.matched_play_start_at is None:
            await self._send_frame(silence)
        if self.matched_play_start_at is None:
            self._log(f"[warn] 第 {plan.index + 1} 轮未等到匹配 {plan.barge_in_match!r} 的 AI 句子")
            return
        target = self.matched_play_start_at + float(plan.barge_in_ms or 0.0)
        while now_ms() < target:
            await self._send_frame(silence)

    async def _wait_until_idle(self, plan: TurnPlan) -> None:
        """等服务端把当前轮处理完（连续 `IDLE_SETTLE_MS` 没有新音频/ turn_done）。"""
        silence = bytes(self.frame_bytes)
        deadline = now_ms() + IDLE_TIMEOUT_S * 1000.0
        saw_done = False
        while now_ms() < deadline:
            drained = False
            while not self.turn_done_queue.empty():
                self.turn_done_queue.get_nowait()
                self.last_activity_at = now_ms()
                saw_done = True
                drained = True
            if drained:
                continue
            if (
                saw_done
                and self.last_activity_at > 0
                and now_ms() - self.last_activity_at >= self.idle_settle_ms
            ):
                return
            await self._send_frame(silence)
        self._log(f"[warn] 第 {plan.index + 1} 轮等不到服务端空闲")

    async def _wait_for_gap(self, plan: TurnPlan) -> None:
        """等到 `playback_cursor + gap_ms` 再开口；等待期间持续送静音帧保持线路。"""
        while True:
            target = self.playback_cursor_ms + plan.gap_ms
            if now_ms() >= target:
                return
            await self._send_frame(bytes(self.frame_bytes))

    async def _send_turn(self, plan: TurnPlan) -> None:
        self.active_turn = plan.index
        measurement = self.measurements[plan.index]
        measurement.speech_start_at = now_ms()
        measurement.speech_end_at = measurement.speech_start_at + plan.speech_end_ms
        self.pending_turn = plan.index
        for offset in range(0, len(plan.pcm), self.frame_bytes):
            frame = plan.pcm[offset : offset + self.frame_bytes]
            if len(frame) < self.frame_bytes:
                frame = frame + bytes(self.frame_bytes - len(frame))
            await self._send_frame(frame)

    async def _send_frame(self, frame: bytes) -> None:
        """发一帧并保证整体节奏是实时（32ms/帧）。

        等待 `gap_ms` 期间可能几秒没发帧；下一帧的截止时刻用 `max(下一帧, 现在)`
        重算，避免“追帧”式地把静音成批灌进服务端（否则服务端音频时间轴会跑飞）。
        """
        now = now_ms()
        if self._next_frame_at > now:
            await asyncio.sleep((self._next_frame_at - now) / 1000.0)
        await self._ws.send(frame)
        self._next_frame_at = max(self._next_frame_at, now_ms()) + self.frame_ms

    # ---- 接收 -------------------------------------------------------------

    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, (bytes, bytearray)):
                    self._on_audio(bytes(message))
                else:
                    self._on_event(json.loads(message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 连接断开时给出原因
            self._log(f"[warn] 接收循环退出：{type(exc).__name__}: {exc}")

    def _on_audio(self, frame: bytes) -> None:
        now = now_ms()
        self.audio_started.set()
        self.last_activity_at = now
        start = max(now, self.playback_cursor_ms)
        self.playback_cursor_ms = start + pcm_ms(frame, self.sample_rate)

        if self.pending_turn is None:
            return
        measurement = self.measurements[self.pending_turn]
        if (
            measurement.speech_end_at is not None
            and measurement.first_audio_at is None
            and now >= measurement.speech_end_at
        ):
            measurement.first_audio_at = now
            measurement.latency_ms = now - measurement.speech_end_at
            self.pending_turn = None

    def _on_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        etype = event.get("type")
        data = event.get("data") or {}
        if etype == "call_started":
            self.call_id = str(event.get("call_id") or "") or None
        elif etype == "interrupted":
            # 浏览器收到 interrupted 会清空待播队列：游标回到「现在」。
            if data.get("reason") != "turn_merged":
                now = now_ms()
                self.playback_cursor_ms = now
                if self.active_turn is not None:
                    measurement = self.measurements[self.active_turn]
                    measurement.interrupted_at = now
                    measurement.interrupted_text = str(data.get("text") or "") or None
                    if measurement.speech_start_at is not None:
                        measurement.interrupted_latency_ms = now - measurement.speech_start_at
        elif etype == "turn_done":
            self.last_activity_at = now_ms()
            self.turn_done_queue.put_nowait(event)
        elif etype == "play_start":
            text = str(data.get("text") or "")
            if any(
                plan.barge_in_match and plan.barge_in_match in text for plan in self.plans
            ):
                self.matched_play_start_at = now_ms()
        elif etype == "call_ended":
            self.call_ended.set()


# --------------------------------------------------------------------------- #
# 汇总与 CLI
# --------------------------------------------------------------------------- #


def summarize(measurements: list[TurnMeasurement]) -> dict[str, Any]:
    latencies = [m.latency_ms for m in measurements if m.latency_ms is not None]
    interrupted = [m for m in measurements if m.interrupted_at is not None]
    return {
        "turns": len(measurements),
        "latency_samples": len(latencies),
        "latency_p50_ms": None if not latencies else round(percentile(latencies, 0.5), 2),
        "latency_p95_ms": None if not latencies else round(percentile(latencies, 0.95), 2),
        "interrupted_turns": len(interrupted),
        "details": [m.to_json() for m in measurements],
    }


async def run_scenario(
    scenario: Scenario,
    *,
    url: str,
    sample_rate: int,
    frame_ms: int,
    gap_override: float | None,
    linger_ms: float,
    idle_settle_ms: float = IDLE_SETTLE_MS,
    log=print,
) -> tuple[list[TurnMeasurement], str | None, list[dict[str, Any]]]:
    client = ScenarioWsClient(
        scenario,
        url=url,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        gap_override=gap_override,
        linger_ms=linger_ms,
        idle_settle_ms=idle_settle_ms,
        log=log,
    )
    measurements = await client.run()
    return measurements, client.call_id, client.events


def select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    paths: list[Path] = []
    if args.all:
        paths = [
            scenario.path
            for scenario in discover_scenarios(args.scenarios_dir)
            if not scenario.pending
        ]
    for raw in args.scenario or []:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif any(char in raw for char in "*?["):
            paths.extend(sorted(Path().glob(raw)))
        else:
            paths.append(path)
    seen: set[Path] = set()
    scenarios: list[Scenario] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        scenarios.append(load_scenario(path))
    return scenarios


async def run(args: argparse.Namespace) -> int:
    scenarios = select_scenarios(args)
    if not scenarios:
        print("没有可跑的场景", file=sys.stderr)
        return 2

    sample_rate = int(args.sample_rate)
    results: list[dict[str, Any]] = []
    all_measurements: list[TurnMeasurement] = []
    for scenario in scenarios:
        print(f"[{scenario.id}] {scenario.desc}")
        measurements, call_id, events = await run_scenario(
            scenario,
            url=args.url,
            sample_rate=sample_rate,
            frame_ms=args.frame_ms,
            gap_override=args.gap_ms,
            linger_ms=args.linger_ms,
            idle_settle_ms=args.idle_settle_ms,
        )
        summary = summarize(measurements)
        summary["scenario"] = scenario.id
        summary["call_id"] = call_id
        summary["event_count"] = len(events)
        results.append(summary)
        all_measurements.extend(measurements)
        for item in summary["details"]:
            line = (
                f"  turn {item['index'] + 1}: latency={item['latency_ms']}ms"
                f" speech={item['speech_end_ms']}ms"
            )
            if item["interrupted"]:
                line += (
                    f"  interrupted after {item['interrupted_latency_ms']}ms"
                    f"  text={item['interrupted_text']!r}"
                )
            print(line)
        print(
            f"  -> call_id={call_id} p50={summary['latency_p50_ms']}ms"
            f" p95={summary['latency_p95_ms']}ms"
        )

    aggregate = summarize(all_measurements)
    print("\n汇总（全部场景）")
    print(f"  轮数 {aggregate['turns']}，时延样本 {aggregate['latency_samples']}")
    print(f"  P50 {aggregate['latency_p50_ms']}ms  P95 {aggregate['latency_p95_ms']}ms")
    print(f"  被抢话轮次 {aggregate['interrupted_turns']}")

    if args.out:
        payload = {
            "url": args.url,
            "scenarios": results,
            "aggregate": aggregate,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写入 {out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T13b 按场景节奏的实时 WebSocket 客户端")
    parser.add_argument(
        "--scenario",
        action="append",
        help="场景 json（可重复；也可给目录或 glob），如 scenarios/s03_appointment.json",
    )
    parser.add_argument("--all", action="store_true", help="跑 scenarios/ 下所有非 pending 场景")
    parser.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS_DIR))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, default=FRAME_MS)
    parser.add_argument("--gap-ms", type=float, default=None, help="覆盖所有轮的 gap_ms")
    parser.add_argument("--linger-ms", type=float, default=DEFAULT_LINGER_MS)
    parser.add_argument(
        "--idle-settle-ms",
        type=float,
        default=IDLE_SETTLE_MS,
        help="服务端安静多久算这一轮处理完（VAD 会把一轮 wav 切成多段）",
    )
    parser.add_argument("--out", default=None, help="时延明细 JSON 的输出路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover - 手动运行用
    raise SystemExit(main())

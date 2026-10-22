"""场景批跑与评估（T12）。

用法：

    # 跑全部按轮场景，报告写到 report/
    uv run python tools/evaluate.py --scenarios scenarios/ --out report/

    # 对比实验：精简 reply / 确认语开头预合成
    uv run python tools/evaluate.py --scenarios scenarios/ --out report/minimal --reply-mode minimal
    uv run python tools/evaluate.py --scenarios scenarios/ --out report/confirm_lead --confirm-lead

评估口径（与任务书 T12 一致）：

- **必填槽位准确率**：`scenarios/*.json` 的 `expected_slots` 与 `slots.json` 比对。
  号码按数字比较；姓名、来意等文本先去掉空白与标点，再允许双向包含（LLM 的措辞
  可能与期望不同，例如“想约张先生谈合同”与“谈合同”）。
- **端到端时延**：每轮 `vad_end → play_start`，并拆成轮次判断（`vad_end → turn_end`）、
  ASR（`asr_final.data.latency_ms`）、LLM（`llm_done.data.total_ms`，因为 Dialog 要等
  slots）、TTS 首包（`tts_first_audio.data.latency_ms`，缓存命中为 0）。CONFIRM 轮单独统计。
- **缓存命中率**：`tts_request.data.cached` 为 True 的比例。
- **解析失败率**：`llm_done.data.parse_ok=False` 与 LLM 超时（没有 `llm_done`）合计 / LLM 请求数。
- **确认语开头预合成的空档**：CONFIRM 轮若有两次播放（预合成开头 + 现合成后半句），
  计算 `第二段 play_start - (第一段 play_start + 第一段时长)`，即两段之间真正静音的时长。

按轮场景的音频与 json 格式见 `tools/make_scenarios.py`；整条 wav 的真人录音场景
（json 里写 `wav`、`turns` 为空）也能跑，但时延拆分依赖事件流，同样适用。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import shutil
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dialog import is_confirm_text  # noqa: E402
from core.events import read_events  # noqa: E402
from core.types import REQUIRED_SLOTS, SLOT_NAMES  # noqa: E402
from tools.replay import (  # noqa: E402
    ReplayServices,
    Scenario,
    discover_scenarios,
    load_config,
    open_services,
    replay_scenario_to_dir,
    scenario_playable,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT / "scenarios"
DEFAULT_OUT = ROOT / "report"


# --------------------------------------------------------------------------- #
# 槽位比对
# --------------------------------------------------------------------------- #


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return "".join(ch for ch in text if not ch.isspace() and ch not in "，。！？、,.!?;；:：\"'“”‘’（）()[]【】")


def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def slot_value_matches(name: str, actual: Any, expected: Any) -> bool:
    """单个槽位是否匹配（`expected` 可以是字符串或可接受字符串的列表）。"""
    if expected is None or expected == "":
        return False
    candidates = expected if isinstance(expected, (list, tuple, set)) else [expected]
    if name == "callback_number":
        actual_digits = _digits(actual)
        if not actual_digits:
            return False
        return any(actual_digits == _digits(item) for item in candidates)
    actual_text = _normalize_text(actual)
    if not actual_text:
        return False
    for item in candidates:
        expected_text = _normalize_text(item)
        if expected_text and (
            actual_text == expected_text
            or expected_text in actual_text
            or actual_text in expected_text
        ):
            return True
    return False


def slots_from_call_dir(call_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(call_dir) / "slots.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    nested = payload.get("slots")
    return dict(nested) if isinstance(nested, Mapping) else {}


def score_slots(
    expected: Mapping[str, Any],
    actual: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """逐槽位比对，返回 `(每槽位是否命中, 实际值快照)`。"""
    results: dict[str, bool] = {}
    seen: dict[str, Any] = {}
    for name in SLOT_NAMES:
        if name not in expected:
            continue
        expected_value = expected.get(name)
        if expected_value in (None, "", []):
            continue
        item = actual.get(name) or {}
        value = item.get("value") if isinstance(item, Mapping) else None
        seen[name] = {"value": value, "status": item.get("status") if isinstance(item, Mapping) else None}
        results[name] = slot_value_matches(name, value, expected_value)
    return results, seen


# --------------------------------------------------------------------------- #
# 事件 → 每轮时延
# --------------------------------------------------------------------------- #


@dataclass
class TurnMetrics:
    index: int
    vad_end_ms: float
    turn_end_ms: float | None = None
    asr_ms: float | None = None
    llm_ms: float | None = None
    tts_first_ms: float | None = None
    play_start_ms: float | None = None
    e2e_ms: float | None = None
    confirm: bool = False
    parse_ok: bool | None = None
    played: list[float] = field(default_factory=list)  # 本轮每次 play_start 的时刻
    play_durations: list[float] = field(default_factory=list)  # 与 played 对应的播放时长
    lead_gap_ms: float | None = None


def extract_turn_metrics(events: Sequence[Mapping[str, Any]]) -> list[TurnMetrics]:
    """从事件流里拆出每轮的时延。

    轮次边界用 `vad_end`（每轮一条；T10/T12 回放都会在 ASR 前写）。没有 `vad_end`
    的问候语等播放不计入轮次时延。
    """
    turns: list[TurnMetrics] = []
    current: TurnMetrics | None = None
    asr_start: float | None = None
    llm_start: float | None = None
    pending_tts_cached: bool | None = None

    for event in events:
        etype = event.get("type")
        data = event.get("data") or {}
        t = float(event.get("t") or 0.0)

        if etype == "vad_end":
            current = TurnMetrics(index=len(turns), vad_end_ms=t)
            turns.append(current)
            asr_start = llm_start = None
            pending_tts_cached = None
            continue
        if current is None:
            continue

        if etype == "turn_end":
            current.turn_end_ms = t
        elif etype == "asr_request":
            asr_start = t
        elif etype == "asr_final":
            latency = data.get("latency_ms")
            current.asr_ms = float(latency) if isinstance(latency, (int, float)) else (
                t - asr_start if asr_start is not None else None
            )
        elif etype == "llm_request":
            llm_start = t
        elif etype == "llm_done":
            total = data.get("total_ms")
            current.llm_ms = float(total) if isinstance(total, (int, float)) else (
                t - llm_start if llm_start is not None else None
            )
            parse_ok = data.get("parse_ok")
            current.parse_ok = bool(parse_ok) if parse_ok is not None else None
        elif etype == "tts_request":
            if current.tts_first_ms is None:
                pending_tts_cached = bool(data.get("cached"))
        elif etype == "tts_first_audio":
            if current.tts_first_ms is None:
                latency = data.get("latency_ms")
                current.tts_first_ms = (
                    float(latency) if isinstance(latency, (int, float)) else 0.0
                )
        elif etype == "play_start":
            if current.play_start_ms is None:
                current.play_start_ms = t
                current.e2e_ms = max(0.0, t - current.vad_end_ms)
                current.confirm = is_confirm_text(str(data.get("text") or ""))
            current.played.append(t)
        elif etype == "play_end":
            current.play_durations.append(float(data.get("duration_ms") or 0.0))
        elif etype == "error" and data.get("stage") == "llm":
            if current.parse_ok is None:
                current.parse_ok = False

    # CONFIRM 轮两段之间的空档（预合成开头 → 现合成后半句）。
    for turn in turns:
        if turn.confirm and len(turn.played) >= 2 and turn.play_durations:
            first_end = turn.played[0] + turn.play_durations[0]
            turn.lead_gap_ms = max(0.0, turn.played[1] - first_end)
    return turns


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    # 线性插值（与 numpy 默认的 linear 一致），P50/P95 更稳定。
    position = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary_stats(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    return {
        "n": len(clean),
        "p50": round(percentile(clean, 50), 1) if clean else None,
        "p95": round(percentile(clean, 95), 1) if clean else None,
        "mean": round(sum(clean) / len(clean), 1) if clean else None,
    }


# --------------------------------------------------------------------------- #
# 单条场景
# --------------------------------------------------------------------------- #


def cache_hit_stats(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = [
        bool((event.get("data") or {}).get("cached"))
        for event in events
        if event.get("type") == "tts_request"
    ]
    hits = sum(1 for cached in requested if cached)
    return {
        "requests": len(requested),
        "hits": hits,
        "hit_rate": round(hits / len(requested), 4) if requested else None,
    }


def parse_stats(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requests = sum(1 for event in events if event.get("type") == "llm_request")
    done = [event for event in events if event.get("type") == "llm_done"]
    parse_failures = sum(1 for event in done if not (event.get("data") or {}).get("parse_ok"))
    timeouts = sum(
        1
        for event in events
        if event.get("type") == "error"
        and (event.get("data") or {}).get("stage") == "llm"
        and (event.get("data") or {}).get("reason") == "timeout"
    )
    failures = parse_failures + timeouts
    return {
        "llm_requests": requests,
        "parse_failures": parse_failures,
        "llm_timeouts": timeouts,
        "failure_rate": round(failures / requests, 4) if requests else None,
    }


def evaluate_call(
    scenario: Scenario,
    call_dir: Path,
    *,
    summary_ms: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """读一通回放的产物，算这条场景的指标。"""
    events = read_events(call_dir) if (call_dir / "events.jsonl").exists() else []
    turns = extract_turn_metrics(events)
    actual_slots = slots_from_call_dir(call_dir)
    slot_hits, slot_values = score_slots(scenario.expected_slots, actual_slots)
    confirms = [turn for turn in turns if turn.confirm]
    others = [turn for turn in turns if not turn.confirm]
    return {
        "id": scenario.id,
        "desc": scenario.desc,
        "error": error,
        "turns": len(turns),
        "slot_hits": slot_hits,
        "slot_values": slot_values,
        "expected_slots": scenario.expected_slots,
        "metrics": [asdict(turn) for turn in turns],
        "latency_ms": {
            "e2e": summary_stats([turn.e2e_ms for turn in turns if turn.e2e_ms is not None]),
            "non_confirm": summary_stats([turn.e2e_ms for turn in others if turn.e2e_ms is not None]),
            "confirm": summary_stats([turn.e2e_ms for turn in confirms if turn.e2e_ms is not None]),
            "turn_decision": summary_stats(
                [
                    (turn.turn_end_ms - turn.vad_end_ms)
                    for turn in turns
                    if turn.turn_end_ms is not None
                ]
            ),
            "asr": summary_stats([turn.asr_ms for turn in turns if turn.asr_ms is not None]),
            "llm": summary_stats([turn.llm_ms for turn in turns if turn.llm_ms is not None]),
            "tts_first": summary_stats(
                [turn.tts_first_ms for turn in turns if turn.tts_first_ms is not None]
            ),
        },
        "cache": cache_hit_stats(events),
        "parse": parse_stats(events),
        "confirm_lead_gap_ms": summary_stats(
            [turn.lead_gap_ms for turn in confirms if turn.lead_gap_ms is not None]
        ),
        "summary_ms": round(summary_ms, 1) if summary_ms is not None else None,
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slot_totals: dict[str, dict[str, int]] = {
        name: {"correct": 0, "total": 0} for name in SLOT_NAMES
    }
    required_correct = required_total = 0
    for result in results:
        for name, hit in (result.get("slot_hits") or {}).items():
            bucket = slot_totals.setdefault(name, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(bool(hit))
            if name in REQUIRED_SLOTS:
                required_total += 1
                required_correct += int(bool(hit))
    for name, bucket in slot_totals.items():
        bucket["rate"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else None

    def merged(field_name: str) -> list[float]:
        values: list[float] = []
        keys = {
            "asr": "asr_ms",
            "llm": "llm_ms",
            "tts_first": "tts_first_ms",
        }
        for result in results:
            for turn in result.get("metrics") or []:
                if field_name == "turn_decision":
                    if turn.get("turn_end_ms") is not None:
                        values.append(float(turn["turn_end_ms"]) - float(turn["vad_end_ms"]))
                else:
                    key = keys[field_name]
                    if turn.get(key) is not None:
                        values.append(float(turn[key]))
        return values

    all_e2e = [
        float(turn["e2e_ms"])
        for result in results
        for turn in result.get("metrics") or []
        if turn.get("e2e_ms") is not None
    ]

    confirm_e2e: list[float] = []
    non_confirm_e2e: list[float] = []
    for result in results:
        for turn in result.get("metrics") or []:
            if turn.get("e2e_ms") is None:
                continue
            (confirm_e2e if turn.get("confirm") else non_confirm_e2e).append(float(turn["e2e_ms"]))

    cache_requests = sum(int((result.get("cache") or {}).get("requests") or 0) for result in results)
    cache_hits = sum(int((result.get("cache") or {}).get("hits") or 0) for result in results)
    llm_requests = sum(int((result.get("parse") or {}).get("llm_requests") or 0) for result in results)
    parse_failures = sum(int((result.get("parse") or {}).get("parse_failures") or 0) for result in results)
    llm_timeouts = sum(int((result.get("parse") or {}).get("llm_timeouts") or 0) for result in results)
    lead_gaps = [
        float(turn["lead_gap_ms"])
        for result in results
        for turn in result.get("metrics") or []
        if turn.get("lead_gap_ms") is not None
    ]
    summary_times = [
        float(result["summary_ms"]) for result in results if result.get("summary_ms") is not None
    ]

    return {
        "slot_accuracy": slot_totals,
        "required_slot_accuracy": {
            "correct": required_correct,
            "total": required_total,
            "rate": round(required_correct / required_total, 4) if required_total else None,
        },
        "latency_ms": {
            "e2e": summary_stats(all_e2e),
            "non_confirm": summary_stats(non_confirm_e2e),
            "confirm": summary_stats(confirm_e2e),
            "turn_decision": summary_stats(merged("turn_decision")),
            "asr": summary_stats(merged("asr")),
            "llm": summary_stats(merged("llm")),
            "tts_first": summary_stats(merged("tts_first")),
        },
        "cache": {
            "requests": cache_requests,
            "hits": cache_hits,
            "hit_rate": round(cache_hits / cache_requests, 4) if cache_requests else None,
        },
        "parse": {
            "llm_requests": llm_requests,
            "parse_failures": parse_failures,
            "llm_timeouts": llm_timeouts,
            "failure_rate": (
                round((parse_failures + llm_timeouts) / llm_requests, 4) if llm_requests else None
            ),
        },
        "confirm_lead_gap_ms": summary_stats(lead_gaps),
        "summary_ms": summary_stats(summary_times),
        "scenarios": len(results),
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    agg = report["aggregate"]
    config = report["config"]
    lines: list[str] = []
    lines.append("# T12 场景评估报告\n")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 场景数：{agg['scenarios']}（跳过未填/缺音频的场景）")
    lines.append(
        f"- 配置：TTS 后端 `{config['backend']}`，`llm.reply_mode={config['reply_mode']}`，"
        f"`tts.confirm_lead={str(config['confirm_lead']).lower()}`"
    )
    lines.append(f"- 总耗时：{report['elapsed_s']:.1f}s\n")

    lines.append("## 必填槽位准确率\n")
    lines.append("| 槽位 | 命中 / 期望 | 准确率 |")
    lines.append("|---|---|---|")
    for name in REQUIRED_SLOTS:
        bucket = agg["slot_accuracy"].get(name, {})
        lines.append(
            f"| {name} | {bucket.get('correct', 0)} / {bucket.get('total', 0)} | "
            f"{_fmt(bucket.get('rate'), 3)} |"
        )
    optional_used = [name for name in SLOT_NAMES if agg["slot_accuracy"].get(name, {}).get("total")]
    for name in optional_used:
        if name in REQUIRED_SLOTS:
            continue
        bucket = agg["slot_accuracy"][name]
        lines.append(f"| {name}（选填） | {bucket['correct']} / {bucket['total']} | {_fmt(bucket.get('rate'), 3)} |")
    required = agg["required_slot_accuracy"]
    lines.append(
        f"\n必填槽位合计：{required['correct']} / {required['total']}（{_fmt(required['rate'], 3)}）\n"
    )

    lines.append("## 时延（ms）\n")
    lines.append("| 指标 | n | P50 | P95 | 均值 |")
    lines.append("|---|---|---|---|---|")
    labels = {
        "e2e": "端到端 vad_end→play_start",
        "non_confirm": "非 CONFIRM 轮端到端",
        "confirm": "CONFIRM 轮端到端",
        "turn_decision": "轮次判断 vad_end→turn_end",
        "asr": "ASR",
        "llm": "LLM 总耗时",
        "tts_first": "TTS 首包（缓存=0）",
    }
    for key, label in labels.items():
        stats = agg["latency_ms"].get(key) or {}
        lines.append(
            f"| {label} | {stats.get('n', 0)} | {_fmt(stats.get('p50'))} | "
            f"{_fmt(stats.get('p95'))} | {_fmt(stats.get('mean'))} |"
        )
    gap = agg["confirm_lead_gap_ms"]
    lines.append(
        f"\n确认语开头预合成的两段空档：n={gap.get('n', 0)}，P50={_fmt(gap.get('p50'))}ms，"
        f"P95={_fmt(gap.get('p95'))}ms。\n"
    )

    lines.append("## 缓存与解析\n")
    cache = agg["cache"]
    parse = agg["parse"]
    lines.append(
        f"- 缓存命中：{cache['hits']} / {cache['requests']}（{_fmt(cache.get('hit_rate'), 3)}）"
    )
    lines.append(
        f"- LLM 请求 {parse['llm_requests']} 次：解析失败 {parse['parse_failures']}，"
        f"超时 {parse['llm_timeouts']}，合计失败率 {_fmt(parse.get('failure_rate'), 3)}"
    )
    summary_stats_data = agg.get("summary_ms") or {}
    if summary_stats_data.get("n"):
        lines.append(
            f"- 摘要（含复核 ASR）：n={summary_stats_data['n']}，"
            f"P50={_fmt(summary_stats_data.get('p50'))}ms，P95={_fmt(summary_stats_data.get('p95'))}ms"
        )
    lines.append("")

    lines.append("## 逐条场景\n")
    lines.append("| 场景 | 轮数 | 槽位 | 端到端 P50/P95 (ms) | 缓存命中 | 解析失败 | 错误 |")
    lines.append("|---|---|---|---|---|---|---|")
    for result in report["scenarios"]:
        hits = result["slot_hits"]
        hit_text = " ".join(f"{name}={'✓' if hit else '✗'}" for name, hit in hits.items()) or "-"
        latency = (result.get("latency_ms") or {}).get("e2e") or {}
        cache = result.get("cache") or {}
        parse = result.get("parse") or {}
        lines.append(
            f"| {result['id']} | {result['turns']} | {hit_text} | "
            f"{_fmt(latency.get('p50'))} / {_fmt(latency.get('p95'))} | "
            f"{cache.get('hits', 0)}/{cache.get('requests', 0)} | "
            f"{parse.get('parse_failures', 0)} | {result.get('error') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


async def evaluate_scenarios(
    scenarios: Sequence[Scenario],
    out_dir: str | Path,
    config: Mapping[str, Any],
    services: ReplayServices,
    *,
    only: set[str] | None = None,
    summarize: bool = False,
) -> dict[str, Any]:
    """逐条跑场景并汇总；`services` 由调用方构造（测试可注入 fake）。"""
    out_dir = Path(out_dir)
    calls_dir = out_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        scenario
        for scenario in scenarios
        if (only is None or scenario.id in only) and scenario_playable(scenario)
    ]
    skipped = [
        scenario.id
        for scenario in scenarios
        if scenario not in selected
    ]

    recheck_asr = None
    owner_name = (config.get("owner") or {}).get("name")
    if summarize:
        from core.summary import build_recheck_asr

        recheck_asr = build_recheck_asr(dict(config))

    results: list[dict[str, Any]] = []
    try:
        for scenario in selected:
            call_dir = calls_dir / scenario.id
            if call_dir.exists():
                shutil.rmtree(call_dir)
            started = time.perf_counter()
            error: str | None = None
            try:
                await replay_scenario_to_dir(scenario, services, config, call_dir)
            except Exception as exc:  # noqa: BLE001 - 一条场景失败不该中断整批
                error = f"{type(exc).__name__}: {exc}"
                print(f"[error] {scenario.id}: {error}", file=sys.stderr)

            summary_ms: float | None = None
            if summarize and error is None:
                from core.summary import summarize_call

                summary_started = time.perf_counter()
                try:
                    await summarize_call(
                        call_dir,
                        services.llm,
                        asr_alt=recheck_asr,
                        owner_name=owner_name,
                    )
                except Exception as exc:  # noqa: BLE001 - 摘要失败只记录，不影响时延统计
                    print(f"[warn] {scenario.id} 摘要失败：{exc}", file=sys.stderr)
                summary_ms = (time.perf_counter() - summary_started) * 1000.0

            result = evaluate_call(
                scenario, call_dir, summary_ms=summary_ms, error=error
            )
            result["elapsed_s"] = round(time.perf_counter() - started, 2)
            results.append(result)
            print(
                f"[done] {scenario.id}: {result['turns']} 轮, "
                f"槽位 {sum(result['slot_hits'].values())}/{len(result['slot_hits'])}, "
                f"{result['elapsed_s']}s"
            )
    finally:
        if recheck_asr is not None:
            aclose = getattr(recheck_asr, "aclose", None)
            if aclose is not None:
                await aclose()

    aggregate_data = aggregate(results)
    report = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "elapsed_s": round(sum(result.get("elapsed_s", 0.0) for result in results), 1),
        "config": {
            "backend": services.backend,
            "reply_mode": (config.get("llm") or {}).get("reply_mode"),
            "confirm_lead": bool((config.get("tts") or {}).get("confirm_lead")),
        },
        "skipped": skipped,
        "scenarios": results,
        "aggregate": aggregate_data,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _apply_overrides(
    config: dict[str, Any],
    *,
    reply_mode: str | None,
    confirm_lead: bool | None,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if reply_mode:
        config.setdefault("llm", {})["reply_mode"] = reply_mode
    if confirm_lead is not None:
        config.setdefault("tts", {})["confirm_lead"] = bool(confirm_lead)
    return config


async def run(args: argparse.Namespace) -> int:
    config = _apply_overrides(
        load_config(Path(args.config)),
        reply_mode=args.reply_mode,
        confirm_lead=args.confirm_lead,
    )
    scenarios = discover_scenarios(Path(args.scenarios))
    if not scenarios:
        print(f"没有找到场景 json：{args.scenarios}", file=sys.stderr)
        return 2
    only = {item.strip() for item in (args.only or "").split(",") if item.strip()} or None
    playable = [
        scenario
        for scenario in scenarios
        if scenario_playable(scenario) and (only is None or scenario.id in only)
    ]
    if not playable:
        print("没有可播放的场景（音频缺失或还没填）", file=sys.stderr)
        return 2
    print(
        f"scenarios={len(playable)} backend={(config.get('tts') or {}).get('backend')} "
        f"reply_mode={(config.get('llm') or {}).get('reply_mode')} "
        f"confirm_lead={bool((config.get('tts') or {}).get('confirm_lead'))}"
    )
    services = await open_services(config, args.backend)
    print(f"cache={services.cache.directory} voice={services.cache.voice_key}")
    try:
        report = await evaluate_scenarios(
            scenarios,
            Path(args.out),
            config,
            services,
            only=only,
            summarize=args.summarize,
        )
    finally:
        await services.aclose()
    agg = report["aggregate"]
    print(
        f"必填槽位 {agg['required_slot_accuracy']['correct']}/"
        f"{agg['required_slot_accuracy']['total']}，"
        f"端到端 P50 {agg['latency_ms']['e2e']['p50']}ms / "
        f"P95 {agg['latency_ms']['e2e']['p95']}ms，"
        f"缓存命中率 {agg['cache']['hit_rate']}，"
        f"解析失败率 {agg['parse']['failure_rate']}"
    )
    print(f"报告：{Path(args.out) / 'summary.md'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T12 场景批跑与评估")
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS), help="场景目录或单个 json")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="报告输出目录")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--backend", choices=("cloud", "local"), default=None)
    parser.add_argument("--reply-mode", choices=("full", "minimal"), default=None,
                        help="覆盖 llm.reply_mode（对比实验）")
    parser.add_argument("--confirm-lead", action=argparse.BooleanOptionalAction, default=None,
                        help="覆盖 tts.confirm_lead（对比实验）")
    parser.add_argument("--only", default=None, help="只跑这些场景 id（逗号分隔）")
    parser.add_argument("--summarize", action="store_true",
                        help="额外跑摘要与复核 ASR（慢），把耗时写进报告")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

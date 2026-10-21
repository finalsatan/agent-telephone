"""LLM 首 token、首句与总耗时压测（T02b 验收用）。

用法：
    source ~/.bashrc                 # 需要 SILICONFLOW_API_KEY
    uv run python tools/bench_llm.py --n 20
    uv run python tools/bench_llm.py --n 20 --model Qwen/Qwen3.5-9B

每轮用一条贴近真实来电的 user_text，state 轮换，逐轮打印，最后给 P50/P95。
首句耗时取自流式短句产出时刻；LlmTimeout 单独计数（首 token 2.5s / 整轮 8s，
可在 config.yaml 调整）。网络环境会影响结果，跑之前记一下当前网络（设计文档 15.3）。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.types import SlotValue, State  # noqa: E402
from services.llm import LlmClient, LlmError, LlmTimeout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 贴近真实场景的一轮输入：(状态, 已有槽位, 对方这句话)
CASES: list[tuple[State, dict[str, SlotValue], str]] = [
    (State.ASK_IDENTITY, {}, "喂，你好，我找一下张先生。"),
    (State.ASK_PURPOSE, {"name": SlotValue("李先生", "unconfirmed")}, "我是快递，有个包裹要送过来。"),
    (
        State.FILL_MISSING,
        {"name": SlotValue("李先生", "unconfirmed"), "purpose": SlotValue("送快递", "unconfirmed")},
        "电话是一三八零零一三八零零零。",
    ),
    (
        State.CONFIRM,
        {
            "name": SlotValue("李先生", "unconfirmed"),
            "purpose": SlotValue("送快递", "unconfirmed"),
            "callback_number": SlotValue("13800138000", "unconfirmed"),
        },
        "对，没错。",
    ),
    (State.CLOSING, {"name": SlotValue("李先生", "confirmed")}, "行，那我先挂了，谢谢。"),
    (State.FILL_MISSING, {"name": SlotValue("王女士", "unconfirmed")}, "嗯……我想想，大概是明天下午吧。"),
]


def pct(values: list[float], p: float) -> float:
    """线性插值分位数（p 取 0–100），与 numpy 默认行为一致，免得多一个依赖。"""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


async def run(args: argparse.Namespace) -> int:
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    kwargs = {"retry_base_delay": 0.5}
    if args.model:
        kwargs["realtime_model"] = args.model
    client = LlmClient.from_config(config, **kwargs)
    if args.model:
        client.realtime_model = args.model

    first_tokens: list[float] = []
    first_sentences: list[float] = []
    totals: list[float] = []
    parse_ok = 0
    failures = 0
    timeouts = 0

    print(f"model={client.realtime_model}  n={args.n}")
    print(f"{'#':>3}  {'first_tok':>9}  {'first_sent':>10}  {'total':>7}  {'ok':>2}  reply")
    try:
        for i in range(args.n):
            state, slots, text = CASES[i % len(CASES)]
            try:
                result = await client.turn(state, slots, [], text)
            except LlmTimeout as exc:
                timeouts += 1
                failures += 1
                print(f"{i:>3}  {'-':>9}  {'-':>10}  {'-':>7}  {'-':>2}  TIMEOUT {exc}")
                continue
            except LlmError as exc:
                failures += 1
                print(f"{i:>3}  {'-':>9}  {'-':>10}  {'-':>7}  {'-':>2}  ERROR {exc}")
                continue
            first_tokens.append(result.first_token_ms)
            if result.first_sentence_ms is not None:
                first_sentences.append(result.first_sentence_ms)
            totals.append(result.total_ms)
            parse_ok += int(result.parse_ok)
            first_sent = (
                f"{result.first_sentence_ms:>10.0f}" if result.first_sentence_ms is not None else f"{'-':>10}"
            )
            print(
                f"{i:>3}  {result.first_token_ms:>9.0f}  {first_sent}  {result.total_ms:>7.0f}  "
                f"{'✓' if result.parse_ok else '✗':>2}  {result.reply[:40]!r}"
            )
    finally:
        await client.aclose()

    done = len(first_tokens)
    print()
    print(f"超时 {timeouts} 次（首 token {client.first_token_timeout_s}s / 整轮 {client.total_timeout_s}s）")
    if not done:
        print("没有任何成功样本。")
        return 1

    print(f"成功 {done}/{args.n}，失败 {failures}，JSON 解析成功 {parse_ok}/{done}")
    for name, values in (
        ("首 token", first_tokens),
        ("首句", first_sentences),
        ("总耗时", totals),
    ):
        if not values:
            print(f"{name}: 无样本")
            continue
        print(
            f"{name}: P50 {pct(values, 50):.0f}ms  P95 {pct(values, 95):.0f}ms  "
            f"min {min(values):.0f}ms  max {max(values):.0f}ms  mean {statistics.mean(values):.0f}ms"
        )
    return 0 if parse_ok == done else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 首 token、首句与总耗时压测")
    parser.add_argument("--n", type=int, default=10, help="轮数（默认 10）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", default=None, help="覆盖 llm.realtime_model")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

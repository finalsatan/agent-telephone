"""TTS 首包与总耗时压测（T04 验收用）。

用法：
    source ~/.bashrc                 # 需要 SILICONFLOW_API_KEY
    uv run python tools/bench_tts.py --n 10
    uv run python tools/bench_tts.py --n 10 --voice FunAudioLLM/CosyVoice2-0.5B:alex

对 5 句不同长度的话各测 n 次。首包耗时由调用方包装器在收到第一块时记录，
与 T10 实时链路的测法一致（不在客户端内部埋点）。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.tts import TtsClient, TtsError, TtsTimeout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 5 句由短到长，覆盖招呼、追问、说明、要号码与整段收尾。
CASES: list[str] = [
    "您好。",
    "请问您是哪位？",
    "我是张先生的助理，他现在不方便接听。",
    "麻烦您留一下姓名、来意和回拨电话，我记下来转告他。",
    "好的，我记下了您的来意，请问您怎么称呼？稍后我会把您说的情况转告张先生，请他尽快给您回电话，谢谢。",
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


async def measure(client: TtsClient, text: str) -> tuple[float, float, float]:
    """返回 (首包 ms, 总耗时 ms, 音频 ms)。首包时刻由调用方在这里记录。"""
    cancel = asyncio.Event()
    started = time.perf_counter()
    first_ms: float | None = None
    total_bytes = 0
    async for chunk in client.synth(text, cancel):
        if first_ms is None:
            first_ms = (time.perf_counter() - started) * 1000.0
        total_bytes += len(chunk)
    total_ms = (time.perf_counter() - started) * 1000.0
    audio_ms = total_bytes / 2 / client.sample_rate * 1000.0
    return (first_ms if first_ms is not None else float("nan"), total_ms, audio_ms)


async def run(args: argparse.Namespace) -> int:
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    kwargs = {"retry_base_delay": 0.5}
    if args.voice:
        kwargs["voice"] = args.voice
    client = TtsClient.from_config(config, **kwargs)

    all_first: list[float] = []
    all_total: list[float] = []
    timeouts = 0
    failures = 0

    print(f"model={client.model}  voice={client.voice}  n={args.n}/句")
    print(f"{'case':>4}  {'chars':>5}  {'first P50':>9}  {'first P95':>9}  {'total P50':>9}  {'total P95':>9}  text")
    try:
        for idx, text in enumerate(CASES, 1):
            firsts: list[float] = []
            totals: list[float] = []
            audio_ms: list[float] = []
            for _ in range(args.n):
                try:
                    first_ms, total_ms, audio = await measure(client, text)
                except TtsTimeout as exc:
                    timeouts += 1
                    failures += 1
                    print(f"{idx:>4}  {'-':>5}  TIMEOUT {exc}")
                    continue
                except TtsError as exc:
                    failures += 1
                    print(f"{idx:>4}  {'-':>5}  ERROR {exc}")
                    continue
                firsts.append(first_ms)
                totals.append(total_ms)
                audio_ms.append(audio)
            if not firsts:
                continue
            all_first.extend(firsts)
            all_total.extend(totals)
            print(
                f"{idx:>4}  {len(text):>5}  {pct(firsts, 50):>9.0f}  {pct(firsts, 95):>9.0f}  "
                f"{pct(totals, 50):>9.0f}  {pct(totals, 95):>9.0f}  {text[:24]!r}"
            )
            print(
                f"      音频 P50 {pct(audio_ms, 50):.0f}ms，"
                f"实时率（音频/总耗时）P50 {statistics.median(audio_ms) / pct(totals, 50):.2f}×"
            )
    finally:
        await client.aclose()

    print()
    print(f"requests={len(all_first) + failures}  ok={len(all_first)}  timeouts={timeouts}  failures={failures}")
    if all_first:
        print(
            f"首包  P50 {pct(all_first, 50):.0f}ms  P95 {pct(all_first, 95):.0f}ms  "
            f"min {min(all_first):.0f}ms  max {max(all_first):.0f}ms"
        )
        print(
            f"总耗时 P50 {pct(all_total, 50):.0f}ms  P95 {pct(all_total, 95):.0f}ms  "
            f"min {min(all_total):.0f}ms  max {max(all_total):.0f}ms"
        )
    return 0 if all_first else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="硅基流动 TTS 首包/总耗时压测")
    parser.add_argument("--n", type=int, default=10, help="每句重复次数（默认 10）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--voice", default=None, help="覆盖 config.yaml 里的音色")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

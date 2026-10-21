"""ASR 耗时压测（T03 验收用）。

用法：
    source ~/.bashrc                 # 需要 SILICONFLOW_API_KEY
    uv run python tools/bench_asr.py --file scenarios/asr_01.wav --n 10
    uv run python tools/bench_asr.py --file scenarios/asr_01.wav --n 10 --model FunAudioLLM/SenseVoiceSmall

用同一条音频重复整段上传，打印耗时与实时率（latency/audio）的 P50/P95，以及首条转写
文本与号码提取结果（确认规则后处理可用）。网络环境会影响结果（设计文档 15.3）。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.normalize import extract_phone  # noqa: E402
from services.asr import AsrClient, AsrError, load_wav_pcm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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
    pcm = load_wav_pcm(ROOT / args.file)
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    client = AsrClient.from_config(config, **kwargs)

    latencies: list[float] = []
    rtf: list[float] = []
    failures = 0
    audio_ms = 0.0
    first_text = ""

    print(f"model={client.model}  file={args.file}  n={args.n}  audio={len(pcm) / 32:.0f}ms")
    print(f"{'#':>3}  {'latency':>8}  {'audio':>8}  {'rtf':>6}  text")
    try:
        for i in range(args.n):
            try:
                result = await client.transcribe(pcm)
            except AsrError as exc:
                failures += 1
                print(f"{i:>3}  {'-':>8}  {'-':>8}  {'-':>6}  ERROR {exc}")
                continue
            audio_ms = result.audio_ms
            latencies.append(result.latency_ms)
            rtf.append(result.latency_ms / result.audio_ms if result.audio_ms else float("nan"))
            first_text = first_text or result.text
            print(
                f"{i:>3}  {result.latency_ms:>7.0f}ms  {result.audio_ms:>7.0f}ms  "
                f"{rtf[-1]:>6.2f}  {result.text[:48]}"
            )
    finally:
        await client.aclose()

    print()
    if not latencies:
        print("没有任何成功样本。")
        return 1
    print(f"成功 {len(latencies)}/{args.n}，失败 {failures}")
    print(
        f"耗时: P50 {pct(latencies, 50):.0f}ms  P95 {pct(latencies, 95):.0f}ms  "
        f"min {min(latencies):.0f}ms  max {max(latencies):.0f}ms  "
        f"mean {statistics.mean(latencies):.0f}ms"
    )
    print(
        f"实时率(latency/audio): P50 {pct(rtf, 50):.2f}  P95 {pct(rtf, 95):.2f}  "
        f"min {min(rtf):.2f}  max {max(rtf):.2f}  mean {statistics.mean(rtf):.2f}"
    )
    if audio_ms:
        print(f"音频时长: {audio_ms:.0f}ms")
    print(f"文本样例: {first_text!r}")
    print(f"号码提取: {extract_phone(first_text)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR 耗时压测")
    parser.add_argument("--file", required=True, help="wav 文件路径（任何采样率，会自动转 16k 单声道）")
    parser.add_argument("--n", type=int, default=10, help="次数（默认 10）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", default=None, help="覆盖 asr.model")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

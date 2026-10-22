"""本地 CosyVoice3 服务压测（T05 验收用）。

前置：先按 `local_tts/README.md` 启动服务（模型加载 + 预热约 15–25s）。

用法：
    uv run python local_tts/bench.py --n 5
    uv run python local_tts/bench.py --n 5 --url http://127.0.0.1:50000

对 5 句不同长度的话各测 n 次（默认 5），首包由调用方包装器在收到第一块时记录；
统计首包/总耗时 P50、P95，实时率 = 合成耗时 / 音频时长，并采样 GPU 显存峰值。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.tts import TtsError, TtsTimeout  # noqa: E402
from services.tts_local import LocalTtsClient  # noqa: E402

CASES: list[str] = [
    "您好。",
    "请问您是哪位？",
    "我是张先生的助理，他现在不方便接听。",
    "麻烦您留一下姓名、来意和回拨电话，我记下来转告他。",
    "好的，我记下了您的来意，请问您怎么称呼？稍后我会把您说的情况转告张先生，请他尽快给您回电话，谢谢。",
]


def pct(values: list[float], p: float) -> float:
    """线性插值分位数（p 取 0–100）。"""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


class GpuSampler(threading.Thread):
    """后台采样整卡显存占用（包含其它进程，报告时给出基线与峰值）。"""

    def __init__(self, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                ).stdout
                self.samples.append(int(out.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001 - 没 GPU/没驱动时静默跳过
                return
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    @property
    def peak(self) -> int | None:
        return max(self.samples) if self.samples else None


async def wait_ready(url: str, timeout_s: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(base_url=url, timeout=5.0) as client:
        while True:
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise SystemExit(f"local tts not ready after {timeout_s:.0f}s: {url}/health")
            await asyncio.sleep(1.0)


async def measure(client: LocalTtsClient, text: str) -> tuple[float, float, float]:
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
    url = args.url or (config.get("tts") or {}).get("local_url", "http://127.0.0.1:50000")
    health = await wait_ready(url)
    print(f"server={url}  health={health}")

    client = LocalTtsClient.from_config(config, base_url=url)
    sampler = GpuSampler()
    sampler.start()
    baseline_gpu = None
    all_first: list[float] = []
    all_total: list[float] = []
    timeouts = 0
    failures = 0
    try:
        # 预热一次，排除客户端连接与首轮 kernel 的影响。
        await measure(client, "您好。")
        baseline_gpu = sampler.peak

        print(f"n={args.n}/句")
        print(
            f"{'case':>4}  {'chars':>5}  {'first P50':>9}  {'first P95':>9}  "
            f"{'total P50':>9}  {'total P95':>9}  {'audio P50':>9}  {'RTF P50':>7}  text"
        )
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
            rtf = statistics.median(totals) / statistics.median(audio_ms)
            print(
                f"{idx:>4}  {len(text):>5}  {pct(firsts, 50):>9.0f}  {pct(firsts, 95):>9.0f}  "
                f"{pct(totals, 50):>9.0f}  {pct(totals, 95):>9.0f}  "
                f"{pct(audio_ms, 50):>9.0f}  {rtf:>7.2f}  {text[:24]!r}"
            )
    finally:
        sampler.stop()
        await client.aclose()

    print()
    print(
        f"requests={len(all_first) + failures}  ok={len(all_first)}  "
        f"timeouts={timeouts}  failures={failures}"
    )
    if all_first:
        print(
            f"首包   P50 {pct(all_first, 50):.0f}ms  P95 {pct(all_first, 95):.0f}ms  "
            f"min {min(all_first):.0f}ms  max {max(all_first):.0f}ms"
        )
        print(
            f"总耗时 P50 {pct(all_total, 50):.0f}ms  P95 {pct(all_total, 95):.0f}ms  "
            f"min {min(all_total):.0f}ms  max {max(all_total):.0f}ms"
        )
    if sampler.peak is not None:
        print(
            f"GPU 显存: baseline {baseline_gpu} MiB  peak {sampler.peak} MiB  "
            f"增量 {sampler.peak - (baseline_gpu or 0)} MiB"
        )
    return 0 if all_first else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 CosyVoice3 TTS 服务压测")
    parser.add_argument("--n", type=int, default=5, help="每句重复次数（默认 5）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--url", default=None, help="覆盖 config.yaml 里的 tts.local_url")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

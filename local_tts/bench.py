"""本地 CosyVoice3 服务压测（T05/T05b 验收用）。

前置：先按 `local_tts/README.md` 启动服务（模型加载 + 预热约 20–60s）。

用法：
    uv run python local_tts/bench.py --n 5
    uv run python local_tts/bench.py --n 5 --wav-dir /tmp/t05b_wavs

对 5 句不同长度的话各测 n 次（默认 5），首包由调用方包装器在收到第一块时记录；
统计首包/总耗时 P50、P95，块数，首块音频时长，实时率 = 合成耗时 / 音频时长，
并采样 GPU 显存与利用率。

T05b 新增“打断后重说”用例：合成 49 字句，收到第一块后 cancel，紧接着合成
“请问您是哪位？”，记录后者的首包，共 n 次，用于评估 T10 的插话打断。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
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
INTERRUPT_TEXT = "请问您是哪位？"


@dataclass
class Measurement:
    first_ms: float
    total_ms: float
    audio_ms: float
    chunks: int
    first_audio_ms: float
    pcm: bytes


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
    """后台采样整卡显存与利用率（包含其它进程，报告时给出基线与峰值）。"""

    def __init__(self, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.memory: list[int] = []
        self.util: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                ).stdout
                mem, util = (part.strip() for part in out.strip().splitlines()[0].split(","))
                self.memory.append(int(mem))
                self.util.append(int(util))
            except Exception:  # noqa: BLE001 - 没 GPU/没驱动时静默跳过
                return
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    @property
    def peak_memory(self) -> int | None:
        return max(self.memory) if self.memory else None

    @property
    def max_util(self) -> int | None:
        return max(self.util) if self.util else None

    @property
    def avg_util(self) -> float | None:
        return statistics.fmean(self.util) if self.util else None


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


async def measure(client: LocalTtsClient, text: str) -> Measurement:
    """完整合成一句，记录首包/总耗时/块数/首块音频/音频数据。"""
    cancel = asyncio.Event()
    started = time.perf_counter()
    first_ms: float | None = None
    first_bytes = 0
    chunks = 0
    pcm = bytearray()
    async for chunk in client.synth(text, cancel):
        if first_ms is None:
            first_ms = (time.perf_counter() - started) * 1000.0
            first_bytes = len(chunk)
        chunks += 1
        pcm.extend(chunk)
    total_ms = (time.perf_counter() - started) * 1000.0
    audio_ms = len(pcm) / 2 / client.sample_rate * 1000.0
    return Measurement(
        first_ms=first_ms if first_ms is not None else float("nan"),
        total_ms=total_ms,
        audio_ms=audio_ms,
        chunks=chunks,
        first_audio_ms=first_bytes / 2 / client.sample_rate * 1000.0,
        pcm=bytes(pcm),
    )


async def interrupted_restart_first_chunk(
    client: LocalTtsClient, long_text: str, short_text: str
) -> float:
    """长句收到第一块后 cancel，紧接着合成短句，返回短句首包耗时（ms）。"""
    cancel = asyncio.Event()
    agen = client.synth(long_text, cancel)
    await anext(agen)  # 等到第一块，模拟“AI 正在说话”
    cancel.set()
    await agen.aclose()  # 客户端停止读取并关闭连接

    started = time.perf_counter()
    agen2 = client.synth(short_text, asyncio.Event())
    await anext(agen2)
    elapsed = (time.perf_counter() - started) * 1000.0
    await agen2.aclose()
    return elapsed


def save_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm)


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
    case_first_p50: dict[int, float] = {}
    try:
        # 预热一次，排除客户端连接与首轮 kernel 的影响。
        await measure(client, "您好。")
        baseline_gpu = sampler.peak_memory

        print(f"n={args.n}/句  voice_id={client.voice_id}")
        print(
            f"{'case':>4}  {'chars':>5}  {'chunks':>6}  {'first P50':>9}  {'first P95':>9}  "
            f"{'1st audio':>9}  {'total P50':>9}  {'total P95':>9}  {'audio P50':>9}  "
            f"{'RTF P50':>7}  text"
        )
        for idx, text in enumerate(CASES, 1):
            firsts: list[float] = []
            totals: list[float] = []
            audio_ms: list[float] = []
            chunks: list[int] = []
            first_audio: list[float] = []
            saved = False
            for _ in range(args.n):
                try:
                    m = await measure(client, text)
                except TtsTimeout as exc:
                    timeouts += 1
                    failures += 1
                    print(f"{idx:>4}  {'-':>5}  TIMEOUT {exc}")
                    continue
                except TtsError as exc:
                    failures += 1
                    print(f"{idx:>4}  {'-':>5}  ERROR {exc}")
                    continue
                firsts.append(m.first_ms)
                totals.append(m.total_ms)
                audio_ms.append(m.audio_ms)
                chunks.append(m.chunks)
                first_audio.append(m.first_audio_ms)
                if args.wav_dir and not saved:
                    save_wav(
                        Path(args.wav_dir) / f"{idx}_{len(text)}chars.wav",
                        m.pcm,
                        client.sample_rate,
                    )
                    saved = True
            if not firsts:
                continue
            all_first.extend(firsts)
            all_total.extend(totals)
            case_first_p50[idx] = pct(firsts, 50)
            rtf = statistics.median(totals) / statistics.median(audio_ms)
            print(
                f"{idx:>4}  {len(text):>5}  {int(statistics.median(chunks)):>6}  "
                f"{pct(firsts, 50):>9.0f}  {pct(firsts, 95):>9.0f}  "
                f"{pct(first_audio, 50):>9.0f}  {pct(totals, 50):>9.0f}  "
                f"{pct(totals, 95):>9.0f}  {pct(audio_ms, 50):>9.0f}  "
                f"{rtf:>7.2f}  {text[:24]!r}"
            )

        # T05c：打断后重说（长句出第一块 → cancel → 立即说短句）
        # 服务端会先停上游 LLM，再把剩余收尾跑完；放宽超时以拿到真实数值。
        print("\n打断后重说（49 字 → cancel → 7 字）：")
        restart_first: list[float] = []
        normal_7 = case_first_p50.get(2)
        saved_first_timeout = client.first_chunk_timeout_s
        saved_total_timeout = client.total_timeout_s
        client.first_chunk_timeout_s = max(saved_first_timeout, 30.0)
        client.total_timeout_s = max(saved_total_timeout, 60.0)
        try:
            for i in range(args.n):
                try:
                    first_ms = await interrupted_restart_first_chunk(
                        client, CASES[-1], INTERRUPT_TEXT
                    )
                except (TtsError, TtsTimeout) as exc:
                    failures += 1
                    print(f"  run={i} ERROR {exc}")
                    continue
                restart_first.append(first_ms)
                extra = max(0.0, first_ms - normal_7) if normal_7 else float("nan")
                print(f"  run={i} 短句首包 {first_ms:.0f}ms  额外等待≈{extra:.0f}ms")
        finally:
            client.first_chunk_timeout_s = saved_first_timeout
            client.total_timeout_s = saved_total_timeout
        if restart_first:
            normal_text = f"{normal_7:.0f}ms" if normal_7 is not None else "n/a"
            print(
                f"  打断后首包 P50 {pct(restart_first, 50):.0f}ms  "
                f"P95 {pct(restart_first, 95):.0f}ms；正常 7 字首包 P50 {normal_text}"
            )
            if normal_7 is not None:
                target = normal_7 + 700.0
                p50 = pct(restart_first, 50)
                print(
                    f"  T05c 目标（≤ 正常 + 700ms = {target:.0f}ms）："
                    f"{'PASS' if p50 <= target else 'FAIL'}"
                )

        # T05c：连续取消压测，验证不会 NaN/卡死
        if args.stress > 0:
            print(f"\n取消压测 --stress {args.stress}（49 字 → cancel → 7 字）：")
            stress_first: list[float] = []
            stress_fail = 0
            client.first_chunk_timeout_s = max(saved_first_timeout, 30.0)
            client.total_timeout_s = max(saved_total_timeout, 60.0)
            try:
                for i in range(args.stress):
                    try:
                        first_ms = await interrupted_restart_first_chunk(
                            client, CASES[-1], INTERRUPT_TEXT
                        )
                    except (TtsError, TtsTimeout) as exc:
                        stress_fail += 1
                        failures += 1
                        print(f"  run={i} FAIL {exc}")
                        continue
                    stress_first.append(first_ms)
                    extra = max(0.0, first_ms - normal_7) if normal_7 else float("nan")
                    print(
                        f"  run={i} 短句首包 {first_ms:.0f}ms  额外等待≈{extra:.0f}ms"
                    )
            finally:
                client.first_chunk_timeout_s = saved_first_timeout
                client.total_timeout_s = saved_total_timeout
            print(
                f"  stress 成功 {len(stress_first)}/{args.stress}，失败 {stress_fail}；"
                f"首包 P50 {pct(stress_first, 50):.0f}ms  P95 {pct(stress_first, 95):.0f}ms"
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
    if sampler.peak_memory is not None:
        avg_util = sampler.avg_util
        print(
            f"GPU 显存: baseline {baseline_gpu} MiB  peak {sampler.peak_memory} MiB  "
            f"增量 {sampler.peak_memory - (baseline_gpu or 0)} MiB；"
            f"利用率 max {sampler.max_util}%  avg {avg_util:.0f}%"
        )
    return 0 if all_first else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 CosyVoice3 TTS 服务压测")
    parser.add_argument("--n", type=int, default=5, help="每句重复次数（默认 5）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--url", default=None, help="覆盖 config.yaml 里的 tts.local_url")
    parser.add_argument(
        "--wav-dir", default=None, help="把每句第一次合成的 wav 存到此目录（试听用）"
    )
    parser.add_argument(
        "--stress",
        type=int,
        default=20,
        help="连续“长句取消→立即短句”的次数（默认 20，0 关闭）",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

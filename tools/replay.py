"""离线回放：把一条来电录音跑完整条链路（T10）。

用法：

    # 用当前 config.yaml 的本地 TTS 跑一条场景
    uv run python tools/replay.py --wav scenarios/asr_01.wav --out calls/asr_01

    # 指定后端 / 只看前两轮
    uv run python tools/replay.py --wav x.wav --out calls/x --backend cloud --max-turns 2

流程：`segment_wav()` 用 T09 的 VAD＋Smart Turn 把整段 wav 切成轮次 → 每轮交给
`core.call.CallSession` → 最后写 `events.jsonl`、`slots.json`、`caller.wav`、`ai.wav`。
AI 回复不真正等待播放：`play_end` 里只记录音频时长（T10 要求）。

轮次合并 / 号码没报完的判断需要“下一轮什么时候开口”，因此这里带一眼 lookahead
把 `next_start_ms` 交给 `CallSession`，由它决定是否取消本轮并把文本拼接后重做。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.call import CallSession, TurnOutcome  # noqa: E402
from core.events import EventLogger  # noqa: E402
from core.phrases import PhraseCache, backend_name  # noqa: E402
from core.turn import TurnDetector, segment_wav  # noqa: E402
from services.asr import load_wav_pcm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_RATE = 16000


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_tts(config: Mapping[str, Any], backend: str) -> Any:
    """按 `tts.backend` 构造 TTS 客户端（与 tools/prerender_phrases.py 同一套选择）。"""
    if backend == "cloud":
        from services.tts import TtsClient

        return TtsClient.from_config(dict(config))
    if backend == "local":
        from services.tts_local import LocalTtsClient

        return LocalTtsClient.from_config(dict(config))
    raise SystemExit(f"unknown tts.backend: {backend!r} (expected 'cloud' or 'local')")


async def ensure_voice_id(config: Mapping[str, Any], tts: Any, backend: str) -> str | None:
    """本地后端先取 `/health` 的 voice_id，保证缓存键与预合成时一致（T05b/T06）。"""
    if backend != "local":
        return None
    fetch = getattr(tts, "fetch_voice_id", None)
    if fetch is None:
        return None
    try:
        return await fetch()
    except Exception as exc:  # noqa: BLE001 - 取不到只影响缓存键，不阻塞回放
        print(f"[warn] 读取本地 TTS voice_id 失败，缓存键可能用占位值：{exc}", file=sys.stderr)
        return None


async def replay_segments(
    session: CallSession,
    segments: Sequence[tuple[float, float]],
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[TurnOutcome]:
    """依次把切好的轮次交给 `CallSession`，处理合并 / 等号码的暂存文本。"""
    await session.start()
    outcomes: list[TurnOutcome] = []
    pending = ""

    def slice_pcm(start_ms: float, end_ms: float) -> bytes:
        start = max(0, int(round(start_ms * sample_rate / 1000.0))) * 2
        end = max(0, int(round(end_ms * sample_rate / 1000.0))) * 2
        return pcm[start:end]

    for index, (start_ms, end_ms) in enumerate(segments):
        next_start = segments[index + 1][0] if index + 1 < len(segments) else None
        # 与实时链路对齐：先记 VAD / 轮次事件，再开始 ASR。
        session.events.log("vad_start", start_ms=start_ms, end_ms=end_ms)
        session.events.log("vad_end", start_ms=start_ms, end_ms=end_ms)
        session.events.log("turn_end", start_ms=start_ms, end_ms=end_ms)
        outcome = await session.handle_turn(
            slice_pcm(start_ms, end_ms),
            start_ms=start_ms,
            end_ms=end_ms,
            next_start_ms=next_start,
            carried_text=pending,
        )
        outcomes.append(outcome)
        pending = outcome.text if outcome.kind in ("merge", "wait") else ""

    if pending:
        # 理论上不会走到（合并 / 等待都需要下一轮），兜底处理最后一段。
        outcomes.append(
            await session.handle_turn(
                text=pending,
                end_ms=segments[-1][1] if segments else None,
                next_start_ms=None,
            )
        )

    session.finish()
    return outcomes


async def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    backend = (args.backend or backend_name(config)).strip().lower()
    wav_path = Path(args.wav)
    out_dir = Path(args.out)
    if (out_dir / "events.jsonl").exists() and not args.force:
        print(f"{out_dir}/events.jsonl 已存在；换一个 --out，或加 --force 追加。", file=sys.stderr)
        return 2

    sample_rate = int((config.get("audio") or {}).get("sample_rate", DEFAULT_SAMPLE_RATE))
    started = time.perf_counter()
    pcm = load_wav_pcm(wav_path, sample_rate)
    detector = TurnDetector.from_config(config)
    segments = segment_wav(pcm, detector=detector)
    if args.max_turns:
        segments = segments[: args.max_turns]
    print(f"wav={wav_path} {len(pcm) / 2 / sample_rate:.2f}s  turns={len(segments)}")

    from services.asr import AsrClient
    from services.llm import LlmClient

    asr = AsrClient.from_config(dict(config))
    llm = LlmClient.from_config(dict(config))
    tts = build_tts(config, backend)
    try:
        await ensure_voice_id(config, tts, backend)
        cache = PhraseCache.from_config(config, tts=tts, backend=backend)
        print(f"backend={backend} cache={cache.directory} voice={cache.voice_key}")
        if not cache.has(_greet_text(config)):
            print(
                "[warn] 问候语不在缓存里（会现合成并补写）；建议先跑 "
                "`uv run python tools/prerender_phrases.py`",
                file=sys.stderr,
            )
        events = EventLogger(out_dir)
        session = CallSession(asr, llm, tts, cache, events, config)
        try:
            outcomes = await replay_segments(session, segments, pcm, sample_rate=sample_rate)
        finally:
            events.close()
    finally:
        for client in (asr, llm, tts):
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()

    replies = [o for o in outcomes if o.kind == "reply"]
    merged = [o for o in outcomes if o.kind == "merge"]
    waited = [o for o in outcomes if o.kind == "wait"]
    print(
        f"done in {time.perf_counter() - started:.1f}s: replies={len(replies)} "
        f"merged={len(merged)} waited={len(waited)} state={session.dialog.state.value}"
    )
    for name, value in session.dialog.slots.items():
        print(f"  {name:<16} {value.value!r} ({value.status})")
    print(f"events: {out_dir / 'events.jsonl'}")
    print(f"slots:  {out_dir / 'slots.json'}")
    return 0


def _greet_text(config: Mapping[str, Any]) -> str:
    from core.phrases import PHRASES

    owner = (config.get("owner") or {}).get("name") or ""
    return PHRASES["greet"].format(owner=owner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T10 离线回放")
    parser.add_argument("--wav", required=True, help="来电者录音（任意采样率/声道，会转成 16k 单声道）")
    parser.add_argument("--out", required=True, help="输出目录（写 events.jsonl / slots.json / 分轨 wav）")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--backend", choices=("cloud", "local"), default=None, help="默认取 tts.backend")
    parser.add_argument("--max-turns", type=int, default=0, help="只跑前 N 轮（0 表示全部）")
    parser.add_argument("--force", action="store_true", help="输出目录已有事件时继续追加")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

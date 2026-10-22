"""预合成话术缓存（T06 验收／日常使用）。

用法：

    # 缺什么补什么（已存在的跳过）
    uv run python tools/prerender_phrases.py

    # 全部重生成（换了音色/参考音频后一般不需要，哈希会自动失效）
    uv run python tools/prerender_phrases.py --force

    # 指定后端（默认取 config.yaml: tts.backend）
    uv run python tools/prerender_phrases.py --backend cloud

    # 合成一段号码复述用于试听
    uv run python tools/prerender_phrases.py --phone 13800138000 --phone-wav /tmp/phone.wav

缓存位置与命名见 `core/phrases.py`：`cache/tts/<backend>/<sha1(文本+音色)>.pcm`。
必须是实时链路实际使用的同一后端、同一音色生成，否则缓存句与现合成句音色不一致。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.phrases import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    LOCAL_VOICE_PLACEHOLDER,
    PhraseCache,
    backend_name,
    cache_texts,
    resolve_voice_key,
)
from services.tts import TtsError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_URL = "http://127.0.0.1:50000"


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_tts(config: Mapping, backend: str) -> Any:
    """按后端名构造 TTS 客户端（T05 起 cloud|local）。"""
    if backend == "cloud":
        from services.tts import TtsClient

        return TtsClient.from_config(dict(config))
    if backend == "local":
        from services.tts_local import LocalTtsClient

        return LocalTtsClient.from_config(dict(config))
    raise SystemExit(f"unknown tts.backend: {backend!r} (expected 'cloud' or 'local')")


async def probe_local_voice_id(config: Mapping, *, timeout: float = 5.0) -> str | None:
    """本地服务的 `/health` 里有参考音色的 voice_id（T05b 会交给客户端提供）。

    T05b 合入前客户端还没有 `voice_id` 属性，这里直接问一次服务，
    免得缓存键落到占位值、等 T05b 合入后整批缓存作废。
    """
    base_url = (config.get("tts") or {}).get("local_url") or DEFAULT_LOCAL_URL
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(f"{str(base_url).rstrip('/')}/health")
            response.raise_for_status()
            voice_id = response.json().get("voice_id")
    except (httpx.HTTPError, ValueError) as exc:
        print(f"[warn] 读取本地 TTS /health 失败，缓存键用占位音色：{exc}", file=sys.stderr)
        return None
    return str(voice_id) if voice_id else None


async def resolve_voice(config: Mapping, tts: Any, backend: str, explicit: str | None) -> str:
    """确定缓存键里的音色标识，并把来源打印出来，避免缓存在无声中错位。"""
    if explicit:
        print(f"voice key: {explicit}（--voice-id 显式指定）")
        return explicit
    client_voice_id = getattr(tts, "voice_id", None)
    if client_voice_id:
        print(f"voice key: {client_voice_id}（客户端 voice_id）")
        return str(client_voice_id)
    if backend == "local":
        probed = await probe_local_voice_id(config)
        if probed:
            print(f"voice key: {probed}（探测 /health 的 voice_id；T05b 合入后由客户端提供）")
            return probed
    key = resolve_voice_key(config, backend=backend, tts=tts)
    if key == LOCAL_VOICE_PLACEHOLDER:
        print(
            "[warn] 本地后端拿不到 voice_id，缓存键用占位值；"
            "换参考音频后需要 --force 或删掉缓存目录。",
            file=sys.stderr,
        )
    print(f"voice key: {key or '(empty)'}")
    return key


async def prerender(
    cache: PhraseCache,
    tts: Any,
    texts: Mapping[str, str],
    *,
    force: bool = False,
    log: Any = print,
) -> dict[str, int]:
    """逐条合成缺失的文本。返回 {"generated", "skipped", "failed"}。"""
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    cancel = asyncio.Event()
    for phrase_id, text in texts.items():
        path = cache.path(text)
        if not force and path.is_file():
            stats["skipped"] += 1
            log(f"skip  {phrase_id:<24} {text}")
            continue
        chunks: list[bytes] = []
        started = time.perf_counter()
        try:
            async for chunk in tts.synth(text, cancel):
                chunks.append(chunk)
        except (TtsError, ValueError, OSError) as exc:
            stats["failed"] += 1
            print(f"[error] {phrase_id} ({text!r}): {exc}", file=sys.stderr)
            continue
        pcm = b"".join(chunks)
        if not pcm:
            stats["failed"] += 1
            print(f"[error] {phrase_id} ({text!r}): TTS 没有产出音频", file=sys.stderr)
            continue
        cache.put(text, pcm)
        stats["generated"] += 1
        audio_ms = len(pcm) / 2 / cache.sample_rate * 1000.0
        log(
            f"synth {phrase_id:<24} {text} "
            f"({(time.perf_counter() - started) * 1000:.0f}ms, 音频 {audio_ms:.0f}ms)"
        )
    return stats


def write_phone_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """把 render_phone 的输出写成 wav，供人工试听。"""
    import soundfile as sf

    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    backend = (args.backend or backend_name(config)).strip().lower()
    tts = build_tts(config, backend)
    try:
        voice = await resolve_voice(config, tts, backend, args.voice_id)
        cache = PhraseCache.from_config(config, tts=tts, backend=backend, voice=voice, root=args.cache_dir)
        texts = cache_texts(args.owner if args.owner is not None else (config.get("owner") or {}).get("name"))
        print(f"backend={cache.backend} voice={cache.voice_key} dir={cache.directory}")
        if args.phone_wav and not args.phone:
            print("--phone-wav 需要同时给 --phone", file=sys.stderr)
            return 2
        started = time.perf_counter()
        stats = await prerender(cache, tts, texts, force=args.force)
        elapsed = time.perf_counter() - started
        print(
            f"\n{stats['generated']} generated, {stats['skipped']} skipped, "
            f"{stats['failed']} failed in {elapsed:.1f}s"
        )
        if args.phone:
            pcm = cache.render_phone(args.phone)
            out = Path(args.phone_wav) if args.phone_wav else Path(f"/tmp/phone_{args.phone}.wav")
            write_phone_wav(out, pcm, cache.sample_rate)
            print(f"phone demo: {args.phone} -> {out} ({len(pcm) / 2 / cache.sample_rate:.2f}s)")
        return 1 if stats["failed"] else 0
    finally:
        aclose = getattr(tts, "aclose", None)
        if aclose is not None:
            await aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预合成 T06 话术缓存")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--backend", choices=("cloud", "local"), default=None, help="默认取 tts.backend")
    parser.add_argument("--voice-id", default=None, help="缓存键里的音色标识（默认按后端自动确定）")
    parser.add_argument("--owner", default=None, help="机主称呼（默认取 config.yaml: owner.name）")
    parser.add_argument("--cache-dir", default=None, help=f"缓存根目录（默认 {DEFAULT_CACHE_ROOT}）")
    parser.add_argument("--force", action="store_true", help="已存在的也重新生成")
    parser.add_argument("--phone", default=None, help="合成这一段号码复述（试听用）")
    parser.add_argument("--phone-wav", default=None, help="号码复述的输出 wav 路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

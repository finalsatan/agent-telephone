"""预合成话术缓存（T06 验收／日常使用）。

用法：

    # 缺什么补什么（已存在的跳过）
    uv run python tools/prerender_phrases.py

    # 全部重生成（换了音色/参考音频后一般不需要，哈希会自动失效）
    uv run python tools/prerender_phrases.py --force

    # 指定后端（默认取 config.yaml: tts.backend）
    uv run python tools/prerender_phrases.py --backend cloud

    # 合成一段号码复述用于试听，并用 ASR 复验读数
    uv run python tools/prerender_phrases.py --phone 13800138000 --phone-wav /tmp/phone.wav --verify

缓存位置与命名见 `core/phrases.py`：`cache/tts/<backend>/<sha1(版本+文本+音色)>.pcm`。
必须是实时链路实际使用的同一后端、同一音色生成，否则缓存句与现合成句音色不一致。

数字片段（T06b）不是逐字合成的：一次合成整句 `DIGIT_SENTENCE`，按停顿切成 10 段分别缓存。
切分失败直接报错退出，不退回单字合成。
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
    DIGIT_ORDER,
    DIGIT_READINGS,
    DIGIT_SENTENCE,
    LOCAL_VOICE_PLACEHOLDER,
    PhraseCache,
    PhraseSplitError,
    backend_name,
    digit_texts,
    phrase_texts,
    resolve_voice_key,
    split_digit_clips,
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


async def prerender_digits(
    cache: PhraseCache,
    tts: Any,
    *,
    force: bool = False,
    takes: int = 1,
    log: Any = print,
) -> dict[str, int]:
    """整句合成数字并切分缓存（T06b）：一条句子的语速与韵律一致，位时长不再抖动。

    单次合成的语速波动很大（实测同一句话 10 个数字合计 1.9–2.8s），所以默认合成
    `takes` 次，取切分成功且总时长最短的一版。这是离线一次性开销。
    """
    texts = digit_texts()
    if not force and all(cache.has(text) for text in texts.values()):
        log(f"skip  digit segments (10)      {DIGIT_SENTENCE}")
        return {"generated": 0, "skipped": len(texts), "failed": 0}

    cancel = asyncio.Event()
    best: tuple[int, list[bytes], float] | None = None
    best_take = 0
    errors: list[str] = []
    for take in range(max(1, takes)):
        started = time.perf_counter()
        try:
            chunks = [chunk async for chunk in tts.synth(DIGIT_SENTENCE, cancel)]
        except (TtsError, ValueError, OSError) as exc:
            errors.append(f"take {take}: 合成失败: {exc}")
            print(f"[error] 数字整句合成失败（take {take}）: {exc}", file=sys.stderr)
            continue
        pcm = b"".join(chunks)
        if not pcm:
            errors.append(f"take {take}: 没有产出音频")
            print(f"[error] 数字整句合成没有产出音频（take {take}）", file=sys.stderr)
            continue
        try:
            clips = split_digit_clips(pcm, sample_rate=cache.sample_rate)
        except PhraseSplitError as exc:
            # T06b：不静默退回单字合成，否则会悄悄用回 170–620ms 的长片段。
            errors.append(f"take {take}: 切分失败: {exc}")
            print(f"[error] 数字整句切分失败（take {take}）：{exc}", file=sys.stderr)
            continue
        total_ms = len(pcm) / 2 / cache.sample_rate * 1000.0
        digits_ms = sum(len(clip) for clip in clips) / 2 / cache.sample_rate * 1000.0
        log(
            f"synth digit sentence {take}       {DIGIT_SENTENCE} "
            f"({(time.perf_counter() - started) * 1000:.0f}ms, 音频 {total_ms:.0f}ms, "
            f"10 段合计 {digits_ms:.0f}ms)"
        )
        if best is None or len(pcm) < best[0]:
            best, best_take = (len(pcm), clips, total_ms), take

    if best is None:
        print(f"[error] 数字片段全部 {max(1, takes)} 次尝试都失败：{'; '.join(errors)}", file=sys.stderr)
        return {"generated": 0, "skipped": 0, "failed": len(texts)}

    _, clips, total_ms = best
    log(f"picked digit take {best_take}（音频 {total_ms:.0f}ms，最短的一版）")
    for digit, clip in zip(DIGIT_ORDER, clips):
        cache.put(texts[digit], clip)
        log(
            f"digit {digit} {DIGIT_READINGS[digit]}                "
            f"{len(clip) / 2 / cache.sample_rate * 1000:.0f}ms"
        )
    return {"generated": len(clips), "skipped": 0, "failed": 0}


def write_phone_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """把 render_phone 的输出写成 wav，供人工试听。"""
    import soundfile as sf

    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


async def verify_with_asr(config: Mapping, items: list[tuple[str, bytes]], *, sample_rate: int) -> None:
    """用 ASR 逐段复验读数（T06b 的验收要求），结果直接打印到 stdout。"""
    from services.asr import AsrClient

    asr_cfg = config.get("asr") or {}
    async with AsrClient.from_config(
        dict(config),
        attempt_timeout_s=asr_cfg.get("recheck_attempt_timeout_s", 20),
        total_timeout_s=asr_cfg.get("recheck_total_timeout_s", 60),
    ) as asr:
        for label, pcm in items:
            try:
                result = await asr.transcribe(pcm)
            except Exception as exc:  # noqa: BLE001 - 复验失败不该让预合成整体失败
                print(f"verify {label:<24} ASR 失败: {exc}", file=sys.stderr)
                continue
            audio_ms = len(pcm) / 2 / sample_rate * 1000.0
            print(f"verify {label:<24} -> {result.text!r} ({audio_ms:.0f}ms)")


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    backend = (args.backend or backend_name(config)).strip().lower()
    tts = build_tts(config, backend)
    try:
        voice = await resolve_voice(config, tts, backend, args.voice_id)
        cache = PhraseCache.from_config(config, tts=tts, backend=backend, voice=voice, root=args.cache_dir)
        owner = args.owner if args.owner is not None else (config.get("owner") or {}).get("name")
        print(f"backend={cache.backend} voice={cache.voice_key} dir={cache.directory}")
        if args.phone_wav and not args.phone:
            print("--phone-wav 需要同时给 --phone", file=sys.stderr)
            return 2
        started = time.perf_counter()
        stats = await prerender(cache, tts, phrase_texts(owner), force=args.force)
        digits = await prerender_digits(cache, tts, force=args.force, takes=args.digit_takes)
        for key in stats:
            stats[key] += digits[key]
        elapsed = time.perf_counter() - started
        print(
            f"\n{stats['generated']} generated, {stats['skipped']} skipped, "
            f"{stats['failed']} failed in {elapsed:.1f}s"
        )

        phone_pcm: bytes | None = None
        if args.phone:
            phone_pcm = cache.render_phone(args.phone)
            out = Path(args.phone_wav) if args.phone_wav else Path(f"/tmp/phone_{args.phone}.wav")
            write_phone_wav(out, phone_pcm, cache.sample_rate)
            print(f"phone demo: {args.phone} -> {out} ({len(phone_pcm) / 2 / cache.sample_rate:.2f}s)")

        if args.verify:
            items = [(f"digit {digit}", cache.get(digit_texts()[digit]) or b"") for digit in DIGIT_ORDER]
            if phone_pcm is not None:
                items.append((f"phone {args.phone}", phone_pcm))
            await verify_with_asr(config, items, sample_rate=cache.sample_rate)
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
    parser.add_argument("--verify", action="store_true", help="用 ASR 逐段复验数字读数（需要网络与密钥）")
    parser.add_argument(
        "--digit-takes",
        type=int,
        default=3,
        help="数字整句合成几次、取语速最快的一版（默认 3；单次语速波动可达 50%%）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

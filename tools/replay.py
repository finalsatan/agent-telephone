"""离线回放：把一条来电录音跑完整条链路（T10 / T12）。

两种输入：

    # 整条 wav（T10；只用于 T09 切轮和真人录音）
    uv run python tools/replay.py --wav scenarios/asr_01.wav --out calls/asr_01

    # 按轮场景（T12）：每轮一个音频片段，用模拟时钟排下一轮开口时刻
    uv run python tools/replay.py --scenario scenarios/s01_delivery.json --out calls/s01

整条 wav 模式用 `segment_wav()` 把整段切成轮次；按轮模式对每个片段也跑一次
`segment_wav()`（用来得到 `vad_end`），但无论切出几段都按一轮处理。

按轮模式的时钟：上一轮 AI 的 `play_start` + 播放时长 + `gap_ms` 是下一轮开口时刻。
`gap_ms` 默认为 800ms，负数表示对方在 AI 说完之前就开口（离线回放无法真的中途切掉
音频，只在事件流里记 `interrupted(reason="barge_in")`；真正的打断由 T13 实时链路验证）。
AI 回复不真正等待播放：`play_end` 里只记录音频时长。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.call import CallSession, TurnOutcome  # noqa: E402
from core.events import EventLogger, read_events  # noqa: E402
from core.phrases import PhraseCache, backend_name  # noqa: E402
from core.turn import TurnDetector, segment_wav  # noqa: E402
from services.asr import load_wav_pcm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_GAP_MS = 800.0


@dataclass
class ScenarioTurn:
    """场景里的一轮：来电者的 wav、文本（基准转写）与接话间隔。"""

    text: str
    wav: Path
    gap_ms: float = DEFAULT_GAP_MS


@dataclass
class Scenario:
    """一条场景（T12）：按轮片段（`turns`）或整条录音（`wav`）二选一。"""

    path: Path
    id: str
    desc: str = ""
    expected_slots: dict[str, Any] = field(default_factory=dict)
    turns: list[ScenarioTurn] = field(default_factory=list)
    wav: Path | None = None
    pending: bool = False

    @property
    def audio_dir(self) -> Path:
        return self.path.parent / self.id


@dataclass
class ReplayServices:
    """一条回放链路的外部服务（评估时在场景之间复用，省掉重复鉴权与预热）。"""

    asr: Any
    llm: Any
    tts: Any
    cache: PhraseCache
    backend: str

    async def aclose(self) -> None:
        for client in (self.asr, self.llm, self.tts):
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()


class CallClock:
    """跟着真实时间走、但可以手动向前跳的时钟（按轮场景用）。

    真实处理耗时（ASR／LLM／TTS）必须计入事件时间，否则 T12 的时延拆分没有意义；
    AI 播放时长与 `gap_ms` 又不是真的等出来的，所以用 `advance()` 把时间往前跳。
    """

    def __init__(self, start: float = 0.0) -> None:
        self._base = float(start)
        self._t0 = time.monotonic()

    def __call__(self) -> float:
        return self._base + (time.monotonic() - self._t0) * 1000.0

    def advance(self, ms: float) -> None:
        if ms > 0:
            self._base += float(ms)

    def advance_to(self, target_ms: float) -> None:
        self.advance(float(target_ms) - self())


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


async def open_services(
    config: Mapping[str, Any], backend: str | None = None
) -> ReplayServices:
    """构造一条回放链路需要的 ASR／LLM／TTS／话术缓存。"""
    from services.asr import AsrClient
    from services.llm import LlmClient

    backend = (backend or backend_name(config)).strip().lower()
    asr = AsrClient.from_config(dict(config))
    llm = LlmClient.from_config(dict(config))
    tts = build_tts(config, backend)
    try:
        await ensure_voice_id(config, tts, backend)
        cache = PhraseCache.from_config(config, tts=tts, backend=backend)
    except Exception:
        for client in (asr, llm, tts):
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()
        raise
    return ReplayServices(asr=asr, llm=llm, tts=tts, cache=cache, backend=backend)


# --------------------------------------------------------------------------- #
# 场景文件
# --------------------------------------------------------------------------- #


def load_scenario(path: str | Path, *, default_gap_ms: float = DEFAULT_GAP_MS) -> Scenario:
    """读一条场景 json。`wav` 相对 json 所在目录解析，`turns[].wav` 相对 `<dir>/<id>/`。

    `turns` 为空而给了 `wav` 时是整条录音模式（真人录音用）；两者都为空表示还没填。
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario_id = str(data.get("id") or path.stem)
    audio_dir = path.parent / scenario_id
    turns: list[ScenarioTurn] = []
    for index, raw in enumerate(data.get("turns") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        wav_name = str(raw.get("wav") or f"turn_{index:02d}.wav")
        wav_path = audio_dir / wav_name
        if not wav_path.exists() and (path.parent / wav_name).exists():
            wav_path = path.parent / wav_name
        turns.append(
            ScenarioTurn(
                text=str(raw.get("text") or ""),
                wav=wav_path,
                gap_ms=float(raw.get("gap_ms", default_gap_ms)),
            )
        )
    whole_wav = path.parent / str(data["wav"]) if data.get("wav") else None
    return Scenario(
        path=path,
        id=scenario_id,
        desc=str(data.get("desc") or ""),
        expected_slots=dict(data.get("expected_slots") or {}),
        turns=turns,
        wav=whole_wav,
        pending=bool(data.get("pending", False)),
    )


def discover_scenarios(directory: str | Path) -> list[Scenario]:
    """列出目录下的场景 json（按文件名排序，跳过还没填的真人录音）。"""
    directory = Path(directory)
    paths = [directory] if directory.is_file() else sorted(directory.glob("*.json"))
    scenarios: list[Scenario] = []
    for path in paths:
        try:
            scenarios.append(load_scenario(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return scenarios


def scenario_playable(scenario: Scenario) -> bool:
    """场景音频是否齐全（评估时跳过错过的占位场景）。"""
    if scenario.pending:
        return False
    if scenario.turns:
        return all(turn.wav.is_file() for turn in scenario.turns)
    return bool(scenario.wav and scenario.wav.is_file())


# --------------------------------------------------------------------------- #
# 回放
# --------------------------------------------------------------------------- #


def playback_end_ms(events: Sequence[Mapping[str, Any]]) -> float | None:
    """一段事件里 AI 播放结束的时刻：第一个 `play_start` + 所有 `play_end` 的时长。"""
    starts = [event for event in events if event.get("type") == "play_start"]
    if not starts:
        return None
    total = sum(
        float((event.get("data") or {}).get("duration_ms") or 0.0)
        for event in events
        if event.get("type") == "play_end"
    )
    return float(starts[0]["t"]) + total


def _last_play_text(events: Sequence[Mapping[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "play_start":
            return str((event.get("data") or {}).get("text") or "")
    return ""


async def replay_turn_scenario(
    session: CallSession,
    scenario: Scenario,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    default_gap_ms: float = DEFAULT_GAP_MS,
) -> list[TurnOutcome]:
    """按轮回放：每轮一个片段，模拟时钟决定下一轮的开口时刻（T12）。

    `session` 必须使用 `tools.replay.CallClock`（或可 `advance()` 的时钟）。
    """
    clock = session.clock
    if not hasattr(clock, "advance_to"):
        raise TypeError("replay_turn_scenario() 需要可 advance 的时钟（CallClock）")
    detector = TurnDetector.from_config(session.config)
    await session.start()
    outcomes: list[TurnOutcome] = []
    events = read_events(session.events.call_dir)
    previous_ai_end = playback_end_ms(events)
    pending = ""

    for index, turn in enumerate(scenario.turns):
        gap_ms = turn.gap_ms if turn.gap_ms is not None else default_gap_ms
        if previous_ai_end is not None:
            clock.advance_to(previous_ai_end + gap_ms)
        start = float(clock())
        pcm = load_wav_pcm(turn.wav, sample_rate)
        segments = segment_wav(pcm, detector=detector)
        full_duration = len(pcm) / 2 / sample_rate * 1000.0
        speech_ms = segments[-1][1] if segments else full_duration
        # 来电者说话的时长也占用模拟时钟：vad_end 必须落在他说完的时刻，
        # 否则 vad_end→play_start 会把“对方说了多久”漏掉。
        clock.advance_to(start + speech_ms)
        speech_start = start
        vad_end = start + speech_ms
        session.events.log("vad_start", start_ms=speech_start, end_ms=vad_end)
        session.events.log("vad_end", start_ms=speech_start, end_ms=vad_end)
        session.events.log("turn_end", start_ms=speech_start, end_ms=vad_end)
        outcome = await session.handle_turn(
            pcm,
            start_ms=speech_start,
            end_ms=vad_end,
            next_start_ms=None,
            carried_text=pending,
            timeline_ms=speech_start,
        )
        outcomes.append(outcome)
        pending = outcome.text if outcome.kind in ("merge", "wait") else ""

        events = read_events(session.events.call_dir)
        ai_end = playback_end_ms([event for event in events if float(event["t"]) >= start])
        if ai_end is not None:
            previous_ai_end = ai_end
            next_gap = (
                scenario.turns[index + 1].gap_ms
                if index + 1 < len(scenario.turns)
                else None
            )
            if next_gap is not None and next_gap < 0:
                # 对方在 AI 说完之前开口。离线回放不能真的中途切音频，只记事件供评估统计。
                session.events.log(
                    "interrupted",
                    reason="barge_in",
                    text=_last_play_text(events),
                    gap_ms=next_gap,
                )

    session.finish()
    return outcomes


async def replay_scenario_to_dir(
    scenario: Scenario,
    services: ReplayServices,
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    clock: CallClock | None = None,
) -> list[TurnOutcome]:
    """把一条场景跑完并写到 `out_dir`（events.jsonl / slots.json / 分轨 wav / call.wav）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = int((config.get("audio") or {}).get("sample_rate", DEFAULT_SAMPLE_RATE))
    default_gap = float((config.get("scenarios") or {}).get("default_gap_ms", DEFAULT_GAP_MS))
    clock = clock or CallClock()
    events = EventLogger(out_dir, clock=clock)
    session = CallSession(
        services.asr,
        services.llm,
        services.tts,
        services.cache,
        events,
        config,
        clock=clock,
        sample_rate=sample_rate,
    )
    try:
        if scenario.turns:
            return await replay_turn_scenario(
                session, scenario, sample_rate=sample_rate, default_gap_ms=default_gap
            )
        if scenario.wav is None or not scenario.wav.is_file():
            raise FileNotFoundError(f"scenario has no playable audio: {scenario.path}")
        pcm = load_wav_pcm(scenario.wav, sample_rate)
        detector = TurnDetector.from_config(config)
        segments = segment_wav(pcm, detector=detector)
        return await replay_segments(
            session, segments, pcm, sample_rate=sample_rate, pre_speech_ms=detector.pre_speech_ms
        )
    finally:
        events.close()


async def replay_segments(
    session: CallSession,
    segments: Sequence[tuple[float, float]],
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    pre_speech_ms: float = 0.0,
) -> list[TurnOutcome]:
    """依次把切好的轮次交给 `CallSession`，处理合并 / 等号码的暂存文本。

    `segment_wav()` 返回的 `start_ms` 含 `pre_speech_ms` 的预卷；合并判断要用实际
    开口时刻，因此把 `pre_speech_ms` 加到下一轮的 `start_ms` 上（T12，第 15 次评审）。
    手工构造 `segments`（如 T10 测试）时传 0 即可保持原语义。
    """
    await session.start()
    outcomes: list[TurnOutcome] = []
    pending = ""

    def slice_pcm(start_ms: float, end_ms: float) -> bytes:
        start = max(0, int(round(start_ms * sample_rate / 1000.0))) * 2
        end = max(0, int(round(end_ms * sample_rate / 1000.0))) * 2
        return pcm[start:end]

    for index, (start_ms, end_ms) in enumerate(segments):
        next_start = (
            segments[index + 1][0] + max(0.0, pre_speech_ms)
            if index + 1 < len(segments)
            else None
        )
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
    out_dir = Path(args.out)
    if (out_dir / "events.jsonl").exists() and not args.force:
        print(f"{out_dir}/events.jsonl 已存在；换一个 --out，或加 --force 追加。", file=sys.stderr)
        return 2

    sample_rate = int((config.get("audio") or {}).get("sample_rate", DEFAULT_SAMPLE_RATE))
    started = time.perf_counter()
    services = await open_services(config, backend)
    print(f"backend={services.backend} cache={services.cache.directory} voice={services.cache.voice_key}")
    if not services.cache.has(_greet_text(config)):
        print(
            "[warn] 问候语不在缓存里（会现合成并补写）；建议先跑 "
            "`uv run python tools/prerender_phrases.py`",
            file=sys.stderr,
        )
    try:
        if args.scenario:
            scenario = load_scenario(Path(args.scenario))
            if not scenario_playable(scenario):
                print(f"场景音频不齐或还没填：{scenario.path}", file=sys.stderr)
                return 2
            print(f"scenario={scenario.id} turns={len(scenario.turns)} desc={scenario.desc}")
            outcomes = await replay_scenario_to_dir(scenario, services, config, out_dir)
        else:
            wav_path = Path(args.wav)
            pcm = load_wav_pcm(wav_path, sample_rate)
            detector = TurnDetector.from_config(config)
            segments = segment_wav(pcm, detector=detector)
            if args.max_turns:
                segments = segments[: args.max_turns]
            print(f"wav={wav_path} {len(pcm) / 2 / sample_rate:.2f}s  turns={len(segments)}")
            events = EventLogger(out_dir)
            session = CallSession(
                services.asr,
                services.llm,
                services.tts,
                services.cache,
                events,
                config,
                sample_rate=sample_rate,
            )
            try:
                outcomes = await replay_segments(
                    session,
                    segments,
                    pcm,
                    sample_rate=sample_rate,
                    pre_speech_ms=detector.pre_speech_ms,
                )
            finally:
                events.close()
    finally:
        await services.aclose()

    replies = [o for o in outcomes if o.kind == "reply"]
    merged = [o for o in outcomes if o.kind == "merge"]
    waited = [o for o in outcomes if o.kind == "wait"]
    print(
        f"done in {time.perf_counter() - started:.1f}s: replies={len(replies)} "
        f"merged={len(merged)} waited={len(waited)}"
    )
    slots_path = out_dir / "slots.json"
    if slots_path.exists():
        payload = json.loads(slots_path.read_text(encoding="utf-8"))
        print(f"state={payload.get('state')}")
        for name, value in (payload.get("slots") or {}).items():
            print(f"  {name:<16} {value.get('value')!r} ({value.get('status')})")
    print(f"events: {out_dir / 'events.jsonl'}")
    print(f"slots:  {slots_path}")
    print(f"call:   {out_dir / 'call.wav'}")
    return 0


def _greet_text(config: Mapping[str, Any]) -> str:
    from core.phrases import PHRASES

    owner = (config.get("owner") or {}).get("name") or ""
    return PHRASES["greet"].format(owner=owner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T10/T12 离线回放")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="整条来电录音（任意采样率/声道，会转成 16k 单声道）")
    source.add_argument("--scenario", help="按轮场景 json（T12）：每轮一个 wav + gap_ms")
    parser.add_argument("--out", required=True, help="输出目录（写 events.jsonl / slots.json / 分轨 wav / call.wav）")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--backend", choices=("cloud", "local"), default=None, help="默认取 tts.backend")
    parser.add_argument("--max-turns", type=int, default=0, help="只跑前 N 轮（0 表示全部，仅 --wav）")
    parser.add_argument("--force", action="store_true", help="输出目录已有事件时继续追加")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

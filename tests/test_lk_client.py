"""T16：LiveKit 场景回放客户端（`tools/lk_client.py`）的离线测试。

只测传输无关的部分——静音帧过滤、以及 `_wait_for_gap()` 的“AI 还在说就继续等”。
不需要 livekit-server：`livekit.rtc` 只在 `open()` / `_send_frame()` 里才 import。
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from tools import ws_client
from tools.lk_client import SILENCE_RMS, LiveKitScenarioClient, _rms
from tools.replay import Scenario, ScenarioTurn


def write_wav(path: Path, pcm: bytes, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


def make_client(tmp_path: Path, *, gap_ms: float = 800.0) -> LiveKitScenarioClient:
    wav = tmp_path / "turn_01.wav"
    write_wav(wav, np.full(3200, 3000, dtype="<i2").tobytes())
    scenario = Scenario(
        path=tmp_path / "s.json",
        id="s",
        desc="",
        turns=[ScenarioTurn(text="喂。", wav=wav, gap_ms=gap_ms)],
    )
    return LiveKitScenarioClient(
        scenario, url="ws://test", linger_ms=0.0, log=lambda *_a, **_k: None
    )


def fake_clock(monkeypatch: pytest.MonkeyPatch, start: float = 1000.0) -> dict[str, float]:
    """把 `ws_client.now_ms()` 换成一个可以手工推进的假时钟。"""
    clock = {"now": start}
    monkeypatch.setattr(ws_client, "now_ms", lambda: clock["now"])
    return clock


def test_rms_of_silence_and_speech() -> None:
    assert _rms(bytes(320)) == 0.0
    assert _rms(np.full(160, 3000, dtype="<i2").tobytes()) == pytest.approx(3000.0, rel=1e-6)


def test_on_audio_drops_silence_frames(tmp_path: Path) -> None:
    """WebRTC 音轨是连续的：静音帧不能当成“AI 在说话”。"""
    client = make_client(tmp_path)
    client._on_audio(bytes(320))
    assert not client.audio_started.is_set()
    assert client._last_audio_at_ms == 0.0

    client._on_audio(np.full(160, int(SILENCE_RMS) + 500, dtype="<i2").tobytes())
    assert client.audio_started.is_set()
    assert client._last_audio_at_ms > 0.0


async def test_wait_for_gap_keeps_waiting_while_audio_is_still_arriving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回归：`target` 必须在循环里重算，否则问候语开头的一小段就能把 target 定死。

    WebRTC 的音频一小段一小段到达，`_last_audio_at_ms` 每次都要重新参与计算；
    只算一次的话，问候语的第一段 + `gap_ms` 到点就开口，问候语被当成抢话打断
    （首轮时延因此不可信）。
    """
    client = make_client(tmp_path, gap_ms=800.0)
    plan = client.plans[0]
    clock = fake_clock(monkeypatch)
    # 问候语第一小段刚到（实测只有 50ms 左右）。
    client._last_audio_at_ms = 1000.0

    async def fake_send(_frame: bytes) -> None:
        clock["now"] += 32.0
        # 300ms 空档之后，问候语继续（第二段到达）。
        if clock["now"] >= 1500.0 and client._last_audio_at_ms < 1500.0:
            client._last_audio_at_ms = 1500.0

    client._send_frame = fake_send  # type: ignore[method-assign]
    await client._wait_for_gap(plan)

    # 必须等到第二段之后再过 gap_ms；第一段 + gap_ms（1800ms）就开口是错的。
    assert clock["now"] >= 1500.0 + plan.gap_ms


async def test_wait_for_gap_returns_after_audio_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path, gap_ms=800.0)
    plan = client.plans[0]
    clock = fake_clock(monkeypatch)
    client._last_audio_at_ms = 1000.0

    async def fake_send(_frame: bytes) -> None:
        clock["now"] += 32.0

    client._send_frame = fake_send  # type: ignore[method-assign]
    await client._wait_for_gap(plan)

    # 没有新音频：等到「最后一段 + gap_ms」就返回，不会一直等下去。
    assert 1000.0 + plan.gap_ms <= clock["now"] < 1000.0 + plan.gap_ms + 100.0


async def test_wait_for_gap_uses_play_end_as_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """云端 TTS 比实时慢时，句子中间有超过 gap_ms 的空档，用 `play_end` 兜住。"""
    client = make_client(tmp_path, gap_ms=800.0)
    plan = client.plans[0]
    clock = fake_clock(monkeypatch)
    client._last_audio_at_ms = 1000.0

    async def fake_send(_frame: bytes) -> None:
        clock["now"] += 32.0
        if clock["now"] >= 1200.0 and client._last_play_end_at_ms < 2000.0:
            # 音频在 1100ms 处断了，但服务端直到 2000ms 才说这句播完。
            client._last_play_end_at_ms = 2000.0

    client._send_frame = fake_send  # type: ignore[method-assign]
    await client._wait_for_gap(plan)

    assert clock["now"] >= 2000.0 + plan.gap_ms


async def test_wait_for_gap_is_immediate_between_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第 2 轮之后：上一轮的 `play_end` 早就过了 gap_ms，应立刻开口（行为与 T13 一致）。"""
    client = make_client(tmp_path, gap_ms=800.0)
    plan = client.plans[0]
    clock = fake_clock(monkeypatch, start=10_000.0)
    client._last_audio_at_ms = 5_000.0
    client._last_play_end_at_ms = 5_500.0
    calls: list[float] = []

    async def fake_send(_frame: bytes) -> None:
        calls.append(clock["now"])

    client._send_frame = fake_send  # type: ignore[method-assign]
    await client._wait_for_gap(plan)

    assert calls == []

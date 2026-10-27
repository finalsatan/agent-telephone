"""T16 live 测试：用真实服务跑一遍 LiveKit 链路。

    uv run pytest -m live tests/live/test_livekit_live.py -v

需要 `SILICONFLOW_API_KEY`（ASR／LLM／TTS）。第二个用例还要自托管
`livekit-server --dev` 与 `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`，缺失时跳过。

**为什么这里只放这两个用例**：管线级用例（帧顺序、抢话、`turn_done` 等）不需要
livekit-server（`session.input.audio` / `session.output.audio` 换成自己的实现即可，
见 notes 1.1），所以它们都在默认测试 `tests/test_livekit.py` 里。这里只放必须真实
服务的部分：真实 ASR/LLM/TTS 的一轮对话，以及真实 WebRTC 房间的往返。
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

from adapters.livekit.session import LiveKitCall
from adapters.livekit.tokens import build_token, new_room_name
from core.events import read_events
from services.asr import load_wav_pcm
from tools.evaluate import extract_turn_metrics
from tools.replay import open_services

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
FRAME_BYTES = 640  # 20ms at 16 kHz, 16-bit mono
TURN_WAV = ROOT / "scenarios" / "s01_delivery" / "turn_01.wav"


@pytest.fixture(scope="module")
def config() -> dict:
    value = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    value["tts"]["backend"] = "cloud"
    return value


@pytest.fixture(scope="module")
def require_env() -> None:
    if not os.environ.get("SILICONFLOW_API_KEY"):
        pytest.skip("SILICONFLOW_API_KEY 未设置")


def _server_reachable(config: dict) -> bool:
    url = str((config.get("livekit") or {}).get("url") or "")
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 7880), timeout=2.0):
            return True
    except OSError:
        return False


async def _wait_for(call_dir: Path, event_type: str, timeout_s: float = 60.0) -> bool:
    """等 `events.jsonl` 里出现某类事件。"""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if any(event["type"] == event_type for event in read_events(call_dir)):
            return True
        await asyncio.sleep(0.2)
    return False


async def _feed(call: LiveKitCall, pcm: bytes, *, frame_ms: int = 20) -> None:
    for offset in range(0, len(pcm), FRAME_BYTES):
        await call.feed(pcm[offset : offset + FRAME_BYTES])
        await asyncio.sleep(frame_ms / 1000.0)


async def test_livekit_one_turn(config, require_env, tmp_path: Path) -> None:
    """真实 ASR／LLM／TTS 跑一轮（不连房间，与管线级用例同一份适配层代码）。"""
    call_dir = tmp_path / "livekit_live"
    services = await open_services(config)
    call = LiveKitCall(
        call_dir=call_dir,
        config=config,
        services=services,
        send_json=lambda payload: None,
        channel_sim=False,
        interrupt_min_speech_ms=300.0,
        sample_rate=SAMPLE_RATE,
    )
    try:
        await call.start()
        assert await _wait_for(call_dir, "play_end"), "问候语没有播出来"
        await _feed(call, load_wav_pcm(TURN_WAV))
        await _feed(call, bytes(FRAME_BYTES) * 150)  # 3s 静音收尾
        assert await _wait_for(call_dir, "llm_done"), "一轮没有跑完（没有 llm_done）"
        await asyncio.sleep(1.0)
    finally:
        await call.finish()

    events = read_events(call_dir)
    types = [event["type"] for event in events]
    for expected in ("vad_start", "vad_end", "turn_end", "asr_final", "llm_done", "play_start", "call_end"):
        assert expected in types, f"缺少 {expected} 事件：{types}"
    assert (call_dir / "slots.json").is_file()
    turns = extract_turn_metrics(events)
    assert turns and turns[0].parse_ok is True


async def test_livekit_room_round_trip(config, require_env, tmp_path: Path) -> None:
    """真实 WebRTC 房间往返：来电者音轨进、AI 音轨出。"""
    if not os.environ.get("LIVEKIT_API_KEY") or not os.environ.get("LIVEKIT_API_SECRET"):
        pytest.skip("LIVEKIT_API_KEY / LIVEKIT_API_SECRET 未设置")
    if not _server_reachable(config):
        pytest.skip(f"livekit-server 未启动（{config.get('livekit', {}).get('url')}）")

    from livekit import rtc

    call_dir = tmp_path / "livekit_room"
    services = await open_services(config)
    room_name = new_room_name("t16-live-test", call_dir.name)
    ttl = 600
    call = LiveKitCall(
        call_dir=call_dir,
        config=config,
        services=services,
        send_json=lambda payload: None,
        channel_sim=False,
        interrupt_min_speech_ms=300.0,
        sample_rate=SAMPLE_RATE,
        room_url=str((config.get("livekit") or {}).get("url")),
        room_name=room_name,
        room_token=build_token(room=room_name, identity="assistant", ttl_s=ttl),
        browser_token=build_token(room=room_name, identity="caller", ttl_s=ttl),
    )
    room = rtc.Room()
    ai_pcm = bytearray()
    subscribed = asyncio.Event()
    tasks: set[asyncio.Task] = set()

    @room.on("track_subscribed")
    def _on_track_subscribed(track, _publication, _participant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        async def _consume() -> None:
            stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
            subscribed.set()
            try:
                async for event in stream:
                    ai_pcm.extend(bytes(event.frame.data))
            finally:
                await stream.aclose()

        task = asyncio.create_task(_consume())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    try:
        await call.start()
        await room.connect(call.room_url, call.browser_token)
        source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=200)
        await room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("caller", source)
        )
        await asyncio.wait_for(subscribed.wait(), timeout=20.0)
        # 等问候语的第一块 AI 音频到达客户端。
        deadline = asyncio.get_running_loop().time() + 30.0
        while not ai_pcm and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
        assert ai_pcm, "没有收到 AI 音轨的音频"
        assert await _wait_for(call_dir, "play_end"), "问候语没有播完"

        # 按 20ms 实时节奏把一轮场景 wav 发进房间，末尾补 3s 静音让 VAD 收尾。
        pcm = load_wav_pcm(TURN_WAV) + bytes(FRAME_BYTES) * 150
        for offset in range(0, len(pcm), FRAME_BYTES):
            chunk = pcm[offset : offset + FRAME_BYTES]
            if len(chunk) < FRAME_BYTES:
                chunk = chunk + bytes(FRAME_BYTES - len(chunk))
            await source.capture_frame(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
            )
            await asyncio.sleep(0.02)
        assert await _wait_for(call_dir, "asr_final", timeout_s=30.0), "房间音频没有进 STT"
        assert await _wait_for(call_dir, "llm_done", timeout_s=60.0), "一轮没有跑完"
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await room.disconnect()
        except Exception:  # noqa: BLE001 - 收尾失败不影响断言
            pass
        await call.finish()

    events = read_events(call_dir)
    types = [event["type"] for event in events]
    assert "vad_start" in types and "asr_final" in types and "call_end" in types
    assert (call_dir / "caller.wav").is_file()
    caller_pcm = (call_dir / "caller.wav").read_bytes()
    assert len(caller_pcm) > 44 + FRAME_BYTES * 10, "来电者音频没有录进 caller.wav"

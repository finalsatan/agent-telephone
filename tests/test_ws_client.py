"""T13b：按场景节奏的实时客户端（`tools/ws_client.py`）测试。

纯函数（语音结束点、分位数）用合成 PCM；`ScenarioWsClient` 用假 WebSocket 对象直接
驱动内部方法，验证播放游标、时延归属与抢话归属。不联网、不依赖真实服务。
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from tools.replay import Scenario, ScenarioTurn, load_scenario
from tools.ws_client import (
    ScenarioWsClient,
    TurnMeasurement,
    now_ms,
    percentile,
    pcm_ms,
    speech_end_ms,
    summarize,
)


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #


def make_pcm(speech_ms: int, silence_ms: int, sample_rate: int = 16000) -> bytes:
    """speech_ms 段振幅 3000 的正弦，后面跟 silence_ms 段静音。"""
    speech = np.full(int(sample_rate * speech_ms / 1000), 3000, dtype="<i2")
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype="<i2")
    return np.concatenate([speech, silence]).tobytes()


def test_speech_end_ms_finds_last_speech() -> None:
    pcm = make_pcm(speech_ms=100, silence_ms=200)
    assert speech_end_ms(pcm) == pytest.approx(100.0, abs=25.0)


def test_speech_end_ms_all_silence_is_zero() -> None:
    assert speech_end_ms(bytes(16000)) == 0.0


def test_speech_end_ms_empty() -> None:
    assert speech_end_ms(b"") == 0.0


def test_pcm_ms() -> None:
    # 16000Hz 16-bit 单声道：16000 采样 = 1000ms
    assert pcm_ms(bytes(32000)) == pytest.approx(1000.0)


def test_percentile() -> None:
    assert percentile([], 0.5) is None
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


# --------------------------------------------------------------------------- #
# ScenarioWsClient 内部状态
# --------------------------------------------------------------------------- #


def write_wav(path: Path, pcm: bytes, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


def make_client(tmp_path: Path, *, gap_ms: float = 0.0) -> ScenarioWsClient:
    wav = tmp_path / "turn_01.wav"
    write_wav(wav, make_pcm(speech_ms=200, silence_ms=100))
    scenario = Scenario(
        path=tmp_path / "s.json",
        id="s",
        desc="",
        turns=[ScenarioTurn(text="喂。", wav=wav, gap_ms=gap_ms)],
    )
    return ScenarioWsClient(scenario, url="ws://test", linger_ms=0.0, log=lambda *_a, **_k: None)


def test_playback_cursor_accumulates_audio(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.audio_started.clear()
    frame = bytes(32000)  # 1000ms
    client._on_audio(frame)
    assert client.audio_started.is_set()
    first_end = client.playback_cursor_ms
    assert first_end > 0
    client._on_audio(frame)
    # 第二块紧跟在第一块后面（排队），游标再往后推 1000ms
    assert client.playback_cursor_ms - first_end == pytest.approx(1000.0, abs=1.0)


def test_latency_attributed_after_speech_end(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    measurement = client.measurements[0]
    base = now_ms()
    measurement.speech_start_at = base
    measurement.speech_end_at = base + 100_000.0  # 未来：还没说完
    client.pending_turn = 0

    # 音频在语音结束前到达：不记录
    client._on_audio(bytes(3200))  # 100ms
    assert measurement.first_audio_at is None

    measurement.speech_end_at = base - 1000.0  # 已过语音结束点
    client._on_audio(bytes(3200))
    assert measurement.first_audio_at is not None
    assert measurement.latency_ms is not None
    assert client.pending_turn is None

    # 第一块之后不再重复记录
    previous = measurement.first_audio_at
    client._on_audio(bytes(3200))
    assert measurement.first_audio_at == previous


def test_interrupted_resets_cursor_and_records_latency(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.measurements[0].speech_start_at = now_ms() - 50.0
    client.active_turn = 0
    client.playback_cursor_ms = now_ms() + 100_000.0
    client._on_event(
        {
            "type": "interrupted",
            "data": {"reason": "barge_in", "text": "好的，我帮您转告张先生。", "tail": True},
        }
    )
    measurement = client.measurements[0]
    assert measurement.interrupted_at is not None
    assert measurement.interrupted_latency_ms is not None
    assert measurement.interrupted_text == "好的，我帮您转告张先生。"
    # 浏览器清空队列后，游标回到「现在」，不再把旧音频算作还在播
    assert client.playback_cursor_ms <= now_ms() + 1.0


def test_turn_merged_does_not_reset_cursor(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.playback_cursor_ms = now_ms() + 100_000.0
    client._on_event({"type": "interrupted", "data": {"reason": "turn_merged", "text": "x"}})
    assert client.playback_cursor_ms == pytest.approx(now_ms() + 100_000.0, abs=5.0)


def test_build_plans_reads_barge_in_fields(tmp_path: Path) -> None:
    """T13b：抢话变体从原始 json 读 `barge_in_ms` / `barge_in_match`。"""
    wav = tmp_path / "turn_01.wav"
    write_wav(wav, make_pcm(speech_ms=200, silence_ms=100))
    scenario_path = tmp_path / "s.json"
    scenario_path.write_text(
        json.dumps(
            {
                "id": "s",
                "turns": [
                    {
                        "text": "不对。",
                        "wav": "turn_01.wav",
                        "gap_ms": 800,
                        "barge_in_ms": 1000,
                        "barge_in_match": "确认一下",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = ScenarioWsClient(
        load_scenario(scenario_path), url="ws://test", linger_ms=0.0, log=lambda *_a, **_k: None
    )
    assert client.plans[0].barge_in_ms == 1000.0
    assert client.plans[0].barge_in_match == "确认一下"


def test_play_start_match_sets_barge_in_marker(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.plans[0].barge_in_match = "确认一下"
    client._on_event({"type": "play_start", "data": {"text": "您好。"}})
    assert client.matched_play_start_at is None
    client._on_event({"type": "play_start", "data": {"text": "跟您确认一下：…"}})
    assert client.matched_play_start_at is not None


def test_turn_done_is_queued(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client._on_event({"type": "turn_done", "kind": "reply", "text": "好的"})
    assert client.turn_done_queue.qsize() == 1


def test_summarize() -> None:
    measurements = [
        TurnMeasurement(0, "a", latency_ms=100.0),
        TurnMeasurement(1, "b", latency_ms=200.0),
        TurnMeasurement(2, "c"),
    ]
    summary = summarize(measurements)
    assert summary["turns"] == 3
    assert summary["latency_samples"] == 2
    assert summary["latency_p50_ms"] == pytest.approx(150.0)
    assert summary["latency_p95_ms"] == pytest.approx(195.0)
    assert summary["interrupted_turns"] == 0


# --------------------------------------------------------------------------- #
# 假服务端：整条客户端循环
# --------------------------------------------------------------------------- #


async def test_client_against_fake_server(tmp_path: Path) -> None:
    """假服务端收帧、回音频；验证客户端能算出一轮的实时时延并正常挂断。"""
    import asyncio

    import websockets

    wav = tmp_path / "turn_01.wav"
    write_wav(wav, make_pcm(speech_ms=200, silence_ms=100))
    scenario = Scenario(
        path=tmp_path / "s.json",
        id="s",
        desc="",
        turns=[ScenarioTurn(text="喂。", wav=wav, gap_ms=0.0)],
    )

    async def handler(ws) -> None:
        await ws.recv()  # start
        await ws.send(json.dumps({"type": "call_started", "call_id": "fake"}))
        await ws.send(bytes(1600))  # 50ms 问候语音频
        frames = 0
        async for message in ws:
            if isinstance(message, (bytes, bytearray)):
                frames += 1
                if frames == 12:  # 约 384ms 后来电者已经说完，回一段 AI 音频
                    await ws.send(bytes(3200))
                    await ws.send(json.dumps({"type": "turn_done", "kind": "reply", "text": "好的"}))
            else:
                if json.loads(message).get("type") == "hangup":
                    await ws.send(json.dumps({"type": "call_ended", "call_id": "fake"}))
                    return

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = ScenarioWsClient(
            scenario,
            url=f"ws://127.0.0.1:{port}",
            linger_ms=200.0,
            log=lambda *_a, **_k: None,
        )
        measurements = await client.run()

    assert client.call_id == "fake"
    assert client.call_ended.is_set()
    measurement = measurements[0]
    assert measurement.first_audio_at is not None
    assert measurement.latency_ms is not None
    assert measurement.latency_ms >= 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

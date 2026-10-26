"""T15 live 测试：用真实服务跑一遍 Pipecat 链路。

先按 `local_tts/README.md` 启动本地 TTS，再运行

    uv run pytest -m live tests/live/test_pipecat_live.py -v

本地 TTS 未启动时跳过（`tts.backend=local` 是默认配置）。ASR／LLM 需要
`SILICONFLOW_API_KEY`，没有时也跳过。测试只喂一轮场景 wav，验证：

- 问候语从预合成缓存播出（`play_start`，缓存命中）；
- 一轮 `vad_start` / `vad_end` / `turn_end` / `asr_final` / `llm_done` / `play_start`；
- `call_end` 与 `slots.json` 落盘，事件流能被 `tools/evaluate.py` 解析。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest
import yaml

from adapters.pipecat.pipeline import PipecatCall
from core.events import read_events
from services.asr import load_wav_pcm
from tools.evaluate import extract_turn_metrics
from tools.replay import open_services

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
FRAME_BYTES = 1024  # 32ms


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def require_env(config) -> None:
    if not os.environ.get("SILICONFLOW_API_KEY"):
        pytest.skip("SILICONFLOW_API_KEY 未设置")
    backend = (config.get("tts") or {}).get("backend", "local")
    if backend == "local":
        url = (config.get("tts") or {}).get("local_url", "http://127.0.0.1:50000")
        try:
            response = httpx.get(url.rstrip("/") + "/health", timeout=3.0)
        except httpx.HTTPError:
            pytest.skip(f"本地 TTS 未启动（{url}）；见 local_tts/README.md")
        if response.status_code != 200:
            pytest.skip(f"本地 TTS 未就绪：{response.status_code}")


async def _feed(call: PipecatCall, pcm: bytes) -> None:
    for offset in range(0, len(pcm), FRAME_BYTES):
        await call.feed(pcm[offset : offset + FRAME_BYTES])
        await asyncio.sleep(0.032)


async def test_pipecat_one_turn(config, require_env, tmp_path: Path) -> None:
    call_dir = tmp_path / "pipecat_live"
    services = await open_services(config)
    sent_audio: list[bytes] = []
    call = PipecatCall(
        call_dir=call_dir,
        config=config,
        services=services,
        send_audio=sent_audio.append,
        send_json=lambda payload: None,
        channel_sim=False,
        interrupt_min_speech_ms=300.0,
        sample_rate=SAMPLE_RATE,
    )
    try:
        await call.start()
        # 等问候语播出。
        for _ in range(100):
            if any(e["type"] == "play_end" for e in read_events(call_dir)):
                break
            await asyncio.sleep(0.2)
        pcm = load_wav_pcm(ROOT / "scenarios" / "s01_delivery" / "turn_01.wav")
        await _feed(call, pcm)
        await _feed(call, bytes(FRAME_BYTES) * 100)  # 3.2s 静音收尾
        for _ in range(150):
            if any(e["type"] == "llm_done" for e in read_events(call_dir)):
                break
            await asyncio.sleep(0.2)
        await asyncio.sleep(1.0)
    finally:
        await call.finish()

    events = read_events(call_dir)
    types = [event["type"] for event in events]
    assert "vad_start" in types and "vad_end" in types and "turn_end" in types
    assert "asr_final" in types
    assert "llm_done" in types
    assert "play_start" in types
    assert "call_end" in types
    assert (call_dir / "slots.json").is_file()
    assert sent_audio, "没有把 AI 音频推给客户端"

    turns = extract_turn_metrics(events)
    assert turns and turns[0].parse_ok is True

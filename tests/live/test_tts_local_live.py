"""T05 live 测试：连已启动的本地 CosyVoice3 服务。

这是可选验收：先按 `local_tts/README.md` 启动服务，再运行

    uv run pytest -m live tests/live/test_tts_local_live.py -v

服务未启动时跳过（模型加载约 20–60s，不适合在 pytest 里自动拉起）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import yaml

from services.tts_local import LocalTtsClient

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base_url(config) -> str:
    url = (config.get("tts") or {}).get("local_url", "http://127.0.0.1:50000")
    try:
        response = httpx.get(url.rstrip("/") + "/health", timeout=3.0)
    except httpx.HTTPError:
        pytest.skip(f"local TTS 服务未启动（{url}）；见 local_tts/README.md")
    if response.status_code != 200:
        pytest.skip(f"local TTS 服务未就绪：{response.status_code} {response.text[:120]}")
    return url


@pytest.fixture
async def client(config, base_url):
    c = LocalTtsClient.from_config(
        config,
        base_url=base_url,
        first_chunk_timeout_s=20,
        total_timeout_s=60,
    )
    try:
        yield c
    finally:
        await c.aclose()


async def test_synth_returns_16k_pcm(client):
    """合成“您好”得到非空 PCM，长度大于 0.3 秒（16kHz × 2 字节 × 0.3s）。"""
    chunks = [chunk async for chunk in client.synth("您好", asyncio.Event())]
    audio = b"".join(chunks)

    assert chunks, "没有收到任何音频块"
    assert len(audio) % 2 == 0
    assert len(audio) > 0.3 * client.sample_rate * 2, f"音频只有 {len(audio)} 字节"


async def test_cancel_after_first_chunk_stops_within_200ms(client):
    """收到第一块后置位 cancel，客户端应在 200ms 内结束。"""
    loop = asyncio.get_running_loop()
    cancel = asyncio.Event()

    agen = client.synth(
        "您好，我是张先生的助理。他现在不方便接听电话，请问您是哪位？"
        "您可以留下姓名、来意和回拨电话，我会尽快转告他，谢谢。",
        cancel,
    )
    assert await anext(agen)

    started = loop.time()
    cancel.set()
    with pytest.raises(StopAsyncIteration):
        await anext(agen)
    elapsed = loop.time() - started

    assert elapsed < 0.2, f"取消后仍等了 {elapsed * 1000:.0f}ms"

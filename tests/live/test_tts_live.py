"""T04 验收（live）：真实调用硅基流动 TTS，确认 PCM 流式产出与取消。

需要环境变量 SILICONFLOW_API_KEY，默认不运行：

    source ~/.bashrc
    uv run pytest -m live tests/live/test_tts_live.py -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from services.tts import TtsClient

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]

# 云端 CosyVoice2 首包 p50 约 2.5s、长句实测到 6.6s（设计文档 16.2）；
# 功能验收用宽松的截止时间抵消公网抖动，生产阈值由 config.yaml 与 bench 数据决定。
LENIENT_FIRST_CHUNK_S = 20.0
LENIENT_TOTAL_S = 60.0

LONG_TEXT = (
    "您好，我是张先生的助理。他现在不方便接听电话，请问您是哪位？"
    "您可以留下姓名、来意和回拨电话，我会尽快转告他，谢谢。"
)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
async def client(config):
    c = TtsClient.from_config(
        config,
        first_chunk_timeout_s=LENIENT_FIRST_CHUNK_S,
        total_timeout_s=LENIENT_TOTAL_S,
    )
    try:
        yield c
    finally:
        await c.aclose()


async def test_synth_returns_non_empty_pcm(client):
    """合成“您好”得到非空 PCM，长度大于 0.3 秒（16kHz × 2 字节 × 0.3s = 9600 字节）。"""
    chunks = [chunk async for chunk in client.synth("您好", asyncio.Event())]
    audio = b"".join(chunks)

    assert chunks, "没有收到任何音频块"
    assert len(audio) % 2 == 0, "PCM 应为 16-bit 采样，字节数必须是偶数"
    assert len(audio) > 0.3 * client.sample_rate * 2, f"音频只有 {len(audio)} 字节"


async def test_cancel_after_first_chunk_stops_within_200ms(client):
    """收到第一块后置位 cancel，生成器应在 200ms 内结束（不抛异常）。"""
    loop = asyncio.get_running_loop()
    cancel = asyncio.Event()

    agen = client.synth(LONG_TEXT, cancel)
    first = await anext(agen)
    assert first, "第一块为空，无法验证取消"

    started = loop.time()
    cancel.set()
    with pytest.raises(StopAsyncIteration):
        await anext(agen)
    elapsed = loop.time() - started

    assert elapsed < 0.2, f"取消后仍等了 {elapsed * 1000:.0f}ms"

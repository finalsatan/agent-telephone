"""T03 验收（live）：用 `scenarios/` 里的真实录音跑硅基流动 ASR。

需要环境变量 SILICONFLOW_API_KEY，默认不运行：

    source ~/.bashrc
    uv run pytest -m live tests/live/test_asr_live.py -v

`scenarios/asr_01.wav` 是一通模拟来电（16kHz 单声道、16-bit，8.8s，"我的电话是
一三八零零一三八零零零"）。它同时用于 tools/bench_asr.py 的压测，保证 bench 与实际
链路上的输入形态一致；`scenarios/asr_02.wav` 是 2.5s 的单轮短句（"喂，你好，我找一下
张先生。"），用作压测里"一轮实际长度"的对照。两个文件都用 TTS 合成，T12 会换成正式
场景集。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.normalize import extract_phone
from services.asr import AsrClient, load_wav_pcm

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "scenarios" / "asr_01.wav"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pcm() -> bytes:
    if not SCENARIO.exists():
        pytest.skip(f"缺少样例音频 {SCENARIO}（T12 会生成正式场景集）")
    return load_wav_pcm(SCENARIO)


@pytest.fixture
async def client(config):
    # 功能验收用自己的 20s 超时；真实链路参数由 config 控制，时间语义由离线测试覆盖。
    c = AsrClient.from_config(config)
    try:
        yield c
    finally:
        await c.aclose()


async def test_transcribe_returns_non_empty_text(client, pcm):
    result = await client.transcribe(pcm)

    assert result.text.strip(), "转写结果为空"
    assert result.latency_ms > 0
    # 整段上传：audio_ms 应约等于样例时长（8.8s），差值来自封装与帧对齐
    assert 8000 <= result.audio_ms <= 9000, f"audio_ms 异常: {result.audio_ms}"


async def test_phone_is_recoverable_with_rules(client, pcm):
    """T03 的核心假设：ASR 给汉字数字，规则后处理能还原出 11 位号码（不经过 LLM）。"""
    result = await client.transcribe(pcm)

    assert extract_phone(result.text) == "13800138000", f"原文: {result.text!r}"


async def test_empty_audio_does_not_hit_network(client):
    result = await client.transcribe(b"")
    assert result.text == "" and result.audio_ms == 0.0 and result.latency_ms == 0.0

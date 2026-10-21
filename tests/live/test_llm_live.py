"""T02 验收（live）：真实调用硅基流动，确认可用性与 JSON 解析。

需要环境变量 SILICONFLOW_API_KEY，默认不运行：

    source ~/.bashrc
    uv run pytest -m live tests/live/test_llm_live.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.types import SlotValue, State
from services.llm import LlmClient

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
async def client(config):
    c = LlmClient.from_config(config)
    try:
        yield c
    finally:
        await c.aclose()


async def test_turn_returns_parsable_json(client):
    result = await client.turn(State.ASK_IDENTITY, {}, [], "喂，你好，我找一下张先生。")

    assert result.parse_ok is True, "模型没按 JSON 格式输出，提示词需要加强"
    assert result.reply.strip(), "reply 为空"
    assert result.reply == result.reply.strip()
    assert len(result.reply) <= 120, f"reply 超过 2 句的预期长度: {result.reply!r}"
    assert result.first_token_ms > 0
    assert result.total_ms >= result.first_token_ms
    # 首 token 控制在任务书的可接受范围内（设计文档 16.3：p50 约 0.54s）
    assert result.first_token_ms < 5000, f"首 token 过慢: {result.first_token_ms:.0f}ms"


async def test_turn_extracts_slots(client):
    slots = {"name": SlotValue("李先生", "unconfirmed")}
    result = await client.turn(State.ASK_PURPOSE, slots, [], "我是快递，有个包裹要送过来。")

    assert result.parse_ok is True
    assert result.slots_update.get("purpose"), f"没提取到 purpose: {result.slots_update}"


async def test_turn_does_not_leak_or_promise(client):
    """超出权限的请求：不能承诺、不能确认预约（设计文档第 4 节）。"""
    result = await client.turn(
        State.ASK_PURPOSE,
        {"name": SlotValue("李先生", "unconfirmed")},
        [],
        "我跟他约好了今天下午三点见面，你帮我确认一下。",
    )

    assert result.parse_ok is True
    forbidden = ("确认好", "已经确认", "没问题", "可以，我帮您", "我保证", "一定")
    assert not any(word in result.reply for word in forbidden), f"回复越权: {result.reply!r}"
    assert "转告" in result.reply or "联系" in result.reply or "不方便" in result.reply


async def test_complete_returns_text(client):
    text = await client.complete([{"role": "user", "content": "用一句话说明你是什么。"}])
    assert isinstance(text, str) and text.strip()

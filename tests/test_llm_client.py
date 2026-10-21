"""T02 离线测试：LlmClient 的流式读取、重试、超时与请求参数（用 httpx.MockTransport，不联网）。"""

from __future__ import annotations

import json

import httpx
import pytest

from core.types import SlotValue, State
from services.llm import LlmClient, LlmError, parse_turn_output


def sse(*chunks: str) -> bytes:
    """按 OpenAI 兼容格式拼 SSE 响应体（含 role 帧、空帧与 [DONE]）。"""
    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}', ""]
    for chunk in chunks:
        payload = {"choices": [{"delta": {"content": chunk}}]}
        lines.append("data: " + json.dumps(payload, ensure_ascii=False))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines).encode()


def make_client(handler, **kwargs) -> LlmClient:
    kwargs.setdefault("retry_base_delay", 0.0)  # 测试里不要真的等退避
    return LlmClient(
        base_url="https://api.example.test/v1",
        realtime_model="Qwen/Qwen3-8B",
        summary_model="Qwen/Qwen3-32B",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 流式 turn()
# --------------------------------------------------------------------------- #


async def test_turn_streams_and_parses():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["enable_thinking"] is False  # Qwen3 必须关闭思考
        assert body["model"] == "Qwen/Qwen3-8B"
        return httpx.Response(200, content=sse('{"reply": "请问您是', '哪位？", "slots": {"name": "张先生"}}'))

    client = make_client(handler)
    async with client:
        result = await client.turn(State.ASK_IDENTITY, {}, [], "我找张先生")

    assert result.parse_ok is True
    assert result.reply == "请问您是哪位？"
    assert result.slots_update == {"name": "张先生"}
    assert 0.0 <= result.first_token_ms <= result.total_ms
    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert seen[0].url.path == "/v1/chat/completions"


async def test_turn_sends_prompt_context_in_user_message():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse('{"reply": "好", "slots": {}}'))

    client = make_client(handler)
    slots = {"name": SlotValue("张先生", "unconfirmed")}
    async with client:
        await client.turn(State.ASK_PURPOSE, slots, [{"role": "assistant", "content": "您好"}], "送快递")

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "assistant", "content": "您好"}
    user_msg = messages[-1]["content"]
    assert "ASK_PURPOSE" in user_msg
    assert "张先生" in user_msg
    assert "送快递" in user_msg
    assert "callback_number" in user_msg  # 缺失的必填槽位要告知模型


async def test_unparsable_output_yields_parse_ok_false():
    client = make_client(lambda request: httpx.Response(200, content=sse("抱歉，我没听清。")))
    async with client:
        result = await client.turn(State.ASK_IDENTITY, {}, [], "喂")

    assert result.parse_ok is False
    assert result.reply == ""
    assert result.slots_update == {}
    # 即使解析失败，时延也要有值，便于统计解析失败率
    assert result.total_ms >= 0.0
    assert result.first_token_ms >= 0.0


async def test_sse_heartbeat_and_empty_deltas_ignored():
    empty_delta = b'data: {"choices":[{"delta":{"content":""}}]}\n\n\n'  # 空 delta + 心跳空行
    body = empty_delta + sse('{"reply": "好", ', '"slots": {}}')

    client = make_client(lambda request: httpx.Response(200, content=body))
    async with client:
        result = await client.turn(State.CLOSING, {}, [], "没事了")
    assert result.reply == "好"
    assert result.parse_ok is True


# --------------------------------------------------------------------------- #
# 重试
# --------------------------------------------------------------------------- #


async def test_retries_on_429_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, content=sse('{"reply": "好的", "slots": {}}'))

    client = make_client(handler)
    async with client:
        result = await client.turn(State.CONFIRM, {}, [], "对")
    assert result.reply == "好的"
    assert attempts["n"] == 2


async def test_retries_on_5xx_up_to_two_times_then_fails():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, text="upstream error")

    client = make_client(handler)
    async with client:
        with pytest.raises(LlmError, match="llm stream failed"):
            await client.turn(State.GREETING, {}, [], "喂")
    assert attempts["n"] == 3  # 首次 + 2 次重试


async def test_succeeds_on_third_attempt():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, content=sse('{"reply": "在的", "slots": {}}'))

    client = make_client(handler)
    async with client:
        result = await client.turn(State.GREETING, {}, [], "喂")
    assert result.reply == "在的"
    assert attempts["n"] == 3


async def test_client_error_not_retried():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, text='{"message": "bad model"}')

    client = make_client(handler)
    async with client:
        with pytest.raises(LlmError, match="HTTP 400"):
            await client.turn(State.GREETING, {}, [], "喂")
    assert attempts["n"] == 1  # 4xx 不重试


async def test_timeout_is_retried_then_fails():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectTimeout("timeout", request=request)

    client = make_client(handler, timeout=0.01)
    async with client:
        with pytest.raises(LlmError):
            await client.turn(State.GREETING, {}, [], "喂")
    assert attempts["n"] == 3


async def test_no_retry_after_partial_output():
    """已经开始输出后再失败不能重试，否则会重复播报。"""
    attempts = {"n": 0}

    class _Boom(httpx.AsyncByteStream):
        async def __aiter__(self):
            payload = {"choices": [{"delta": {"content": '{"reply": "第一'}}]}
            yield ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
            raise httpx.ReadError("stream broke")

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(200, stream=_Boom())

    client = make_client(handler)
    async with client:
        with pytest.raises(LlmError):
            await client.turn(State.ASK_PURPOSE, {}, [], "送快递")
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- #
# 密钥与 complete()
# --------------------------------------------------------------------------- #


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(LlmError, match="SILICONFLOW_API_KEY"):
        LlmClient(base_url="https://api.example.test/v1", realtime_model="m")


async def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-from-env")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-from-env"
        return httpx.Response(200, content=sse('{"reply": "好", "slots": {}}'))

    client = LlmClient(
        base_url="https://api.example.test/v1",
        realtime_model="m",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
    )
    async with client:
        assert (await client.turn(State.CLOSING, {}, [], "再见")).reply == "好"


async def test_complete_uses_summary_model_and_returns_text():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "这是摘要"}}]},
        )

    client = make_client(handler)
    async with client:
        text = await client.complete([{"role": "user", "content": "总结一下"}])

    assert text == "这是摘要"
    assert captured["model"] == "Qwen/Qwen3-32B"
    assert captured["stream"] is False
    assert captured["enable_thinking"] is False


async def test_complete_retries_then_raises_on_bad_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = make_client(handler)
    async with client:
        with pytest.raises(LlmError, match="unexpected completion payload"):
            await client.complete([{"role": "user", "content": "x"}])


async def test_from_config_reads_models_and_base_url(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-env")
    config = {
        "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
        "llm": {"realtime_model": "Qwen/Qwen3-8B", "summary_model": "Qwen/Qwen3-32B"},
    }
    client = LlmClient.from_config(config)
    try:
        assert client.realtime_model == "Qwen/Qwen3-8B"
        assert client.summary_model == "Qwen/Qwen3-32B"
        assert str(client._client.base_url).rstrip("/") == "https://api.siliconflow.cn/v1"
    finally:
        await client.aclose()


def test_parse_roundtrip_via_client_function_is_shared():
    """turn() 内部用的就是同一个 parse_turn_output，行为一致。"""
    assert parse_turn_output('{"reply": "好", "slots": {}}') == ("好", {}, True)


async def test_malformed_sse_frames_ignored():
    body = (
        "data: {\n\n"  # 半截 JSON
        'data: {"choices": []}\n\n'  # 没有 choices
        'data: {"choices": [{"delta": {}}]}\n\n'  # delta 里没有 content
        'data: {"choices": [{"delta": {"content": "{\\"reply\\": \\"好\\", \\"slots\\": {}}"}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    client = make_client(lambda request: httpx.Response(200, content=body))
    async with client:
        result = await client.turn(State.CLOSING, {}, [], "再见")
    assert result.reply == "好"


async def test_complete_retries_on_429_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"choices": [{"message": {"content": "摘要"}}]})

    client = make_client(handler)
    async with client:
        assert await client.complete([{"role": "user", "content": "x"}]) == "摘要"
    assert attempts["n"] == 2


async def test_complete_4xx_not_retried():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401, text="unauthorized")

    client = make_client(handler)
    async with client:
        with pytest.raises(LlmError, match="HTTP 401"):
            await client.complete([{"role": "user", "content": "x"}])
    assert attempts["n"] == 1

"""T02/T02b 离线测试：LlmClient 的流式读取、短句切分、重试、截止时间与请求参数
（用 httpx.MockTransport，不联网）。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.types import SlotValue, State
from services.llm import (
    LlmClient,
    LlmError,
    LlmTimeout,
    parse_turn_output,
    split_reply_sentences,
)


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


def sse_event(content: str) -> bytes:
    payload = {"choices": [{"delta": {"content": content}}]}
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


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
    assert result.first_sentence_ms is not None
    assert 0.0 <= result.first_sentence_ms <= result.total_ms
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


# --------------------------------------------------------------------------- #
# T02b：流式短句
# --------------------------------------------------------------------------- #


async def test_first_sentence_emitted_before_stream_finishes():
    """reply 在前时，首句必须不等流结束就产出（TTS 边收边播的前提）。"""
    release = asyncio.Event()

    class _Gated(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse_event('{"reply": "第一句。')
            await release.wait()
            yield sse_event('第二句！", "slots": {}}') + b"data: [DONE]\n\n"

    client = make_client(lambda request: httpx.Response(200, stream=_Gated()))
    chunks = []
    async with client:
        agen = client.turn_stream(State.ASK_PURPOSE, {}, [], "送快递")
        first = await anext(agen)
        assert first.kind == "sentence"
        assert first.text == "第一句。"
        assert not release.is_set()  # 流还没结束，首句已经到手
        release.set()
        async for chunk in agen:
            chunks.append(chunk)

    assert [c.text for c in chunks if c.kind == "sentence"] == ["第二句！"]
    assert chunks[-1].kind == "done"
    assert chunks[-1].result.reply == "第一句。第二句！"


async def test_reply_escapes_decoded_even_when_split_across_chunks():
    """引号、反斜杠、换行、\\uXXXX 都要正确解码，且允许转义序列跨 SSE 块。"""
    reply_text = '他说"好"\n第二句\\路径你好。'
    body_text = '{"reply": ' + json.dumps(reply_text, ensure_ascii=True) + ', "slots": {}}'
    pieces = [body_text[i : i + 3] for i in range(0, len(body_text), 3)]

    client = make_client(lambda request: httpx.Response(200, content=sse(*pieces)))
    async with client:
        result = await client.turn(State.ASK_PURPOSE, {}, [], "喂")
        sentences = [
            c.text async for c in client.turn_stream(State.ASK_PURPOSE, {}, [], "喂") if c.kind == "sentence"
        ]

    assert result.parse_ok is True
    assert result.reply == reply_text.strip()
    assert sentences == split_reply_sentences(reply_text)
    assert sentences == ['他说"好"', "第二句\\路径你好。"]


async def test_reply_without_punctuation_flushes_as_one_sentence():
    client = make_client(
        lambda request: httpx.Response(200, content=sse('{"reply": "稍等一下", "slots": {}}'))
    )
    async with client:
        chunks = [c async for c in client.turn_stream(State.CONFIRM, {}, [], "对")]

    assert [(c.kind, c.text) for c in chunks[:-1]] == [("sentence", "稍等一下")]
    assert chunks[-1].kind == "done"
    assert chunks[-1].result.first_sentence_ms is not None


async def test_slots_before_reply_falls_back_to_split_after_stream():
    body = sse('{"slots": {"name": "张先生"}, "reply": "请问您是哪位？好的，请稍等。"}')
    client = make_client(lambda request: httpx.Response(200, content=body))
    async with client:
        chunks = [c async for c in client.turn_stream(State.ASK_IDENTITY, {}, [], "我找张先生")]

    assert [(c.kind, c.text) for c in chunks[:-1]] == [
        ("sentence", "请问您是哪位？"),
        ("sentence", "好的，请稍等。"),
    ]
    done = chunks[-1]
    assert done.kind == "done"
    assert done.result.parse_ok is True
    assert done.result.slots_update == {"name": "张先生"}
    assert done.result.first_sentence_ms is not None


def test_split_reply_sentences_rules():
    assert split_reply_sentences("你好。再见！还等吗？等；好") == [
        "你好。",
        "再见！",
        "还等吗？",
        "等；",
        "好",
    ]
    assert split_reply_sentences("第一行\n第二行。") == ["第一行", "第二行。"]
    assert split_reply_sentences("   ") == []
    # 连续标点不会产出空句
    assert split_reply_sentences("好。。") == ["好。"]


# --------------------------------------------------------------------------- #
# T02b：截止时间
# --------------------------------------------------------------------------- #


async def test_first_token_timeout_raises_and_does_not_retry():
    attempts = {"n": 0}

    class _SlowStart(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(5)
            yield sse_event('{"reply": "好", "slots": {}}')

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(200, stream=_SlowStart())

    client = make_client(handler, first_token_timeout_s=0.05, total_timeout_s=5)
    async with client:
        with pytest.raises(LlmTimeout, match="first-token"):
            async for _ in client.turn_stream(State.GREETING, {}, [], "喂"):
                pass
    assert attempts["n"] == 1  # 截止时间触发不重试


async def test_total_timeout_after_sentence_keeps_sentence_but_no_done():
    class _SentenceThenHang(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse_event('{"reply": "第一句。')
            await asyncio.sleep(5)
            yield sse_event('第二句", "slots": {}}') + b"data: [DONE]\n\n"

    client = make_client(
        lambda request: httpx.Response(200, stream=_SentenceThenHang()),
        first_token_timeout_s=5,
        total_timeout_s=0.1,
    )
    chunks = []
    async with client:
        with pytest.raises(LlmTimeout, match="total timeout"):
            async for chunk in client.turn_stream(State.ASK_PURPOSE, {}, [], "送快递"):
                chunks.append(chunk)

    assert [(c.kind, c.text) for c in chunks] == [("sentence", "第一句。")]  # 已产出的句子有效


async def test_timeout_is_llm_error_subclass():
    assert issubclass(LlmTimeout, LlmError)


# --------------------------------------------------------------------------- #
# T02b：max_tokens、attempts、机主称呼
# --------------------------------------------------------------------------- #


async def test_max_tokens_attempts_and_owner_reach_the_request():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse('{"reply": "好", "slots": {}}'))

    client = make_client(handler, max_tokens=42, owner_name="李女士")
    async with client:
        await client.turn(
            State.FILL_MISSING,
            {},
            [],
            "嗯……",
            attempts={"purpose": 2, "callback_number": 0},
        )

    assert captured["max_tokens"] == 42
    messages = captured["messages"]
    assert messages[0]["content"].startswith("你是李女士的电话助理")
    assert "purpose 已追问 2 次" in messages[-1]["content"]
    assert "callback_number 已追问 0 次" not in messages[-1]["content"]


async def test_from_config_reads_t02b_settings(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-env")
    config = {
        "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
        "owner": {"name": "王女士"},
        "llm": {
            "realtime_model": "Qwen/Qwen3-8B",
            "summary_model": "Qwen/Qwen3-32B",
            "first_token_timeout_s": 3.0,
            "total_timeout_s": 9.0,
            "max_tokens": 120,
        },
    }
    client = LlmClient.from_config(config)
    try:
        assert client.owner_name == "王女士"
        assert client.first_token_timeout_s == 3.0
        assert client.total_timeout_s == 9.0
        assert client.max_tokens == 120
    finally:
        await client.aclose()


async def test_from_config_kwargs_override_config(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-env")
    config = {
        "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
        "owner": {"name": "王女士"},
        "llm": {
            "realtime_model": "m",
            "first_token_timeout_s": 3.0,
            "total_timeout_s": 9.0,
            "max_tokens": 120,
        },
    }
    client = LlmClient.from_config(config, max_tokens=64)
    try:
        assert client.max_tokens == 64
        assert client.owner_name == "王女士"
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- #
# T02b：逐字符分块（真实模型按 token 返回，冒号/引号常被切开）
# --------------------------------------------------------------------------- #


def test_extractor_survives_char_by_char_chunks():
    reply = "第一句。第二句！"
    raw = '{"reply": ' + json.dumps(reply, ensure_ascii=False) + ', "slots": {"name": "张先生"}}'
    from services.llm import _ReplyExtractor

    extractor = _ReplyExtractor()
    sentences: list[str] = []
    for ch in raw:
        sentences.extend(extractor.feed(ch))

    assert extractor.abandoned is False
    assert extractor.complete is True
    assert extractor.decoded == reply
    assert sentences == split_reply_sentences(reply) == ["第一句。", "第二句！"]


async def test_turn_stream_with_char_by_char_sse_deltas():
    body_text = '{"reply": "第一句。第二句！", "slots": {}}'
    client = make_client(
        lambda request: httpx.Response(200, content=sse(*[ch for ch in body_text]))
    )
    async with client:
        chunks = [c async for c in client.turn_stream(State.ASK_PURPOSE, {}, [], "送快递")]

    assert [(c.kind, c.text) for c in chunks[:-1]] == [
        ("sentence", "第一句。"),
        ("sentence", "第二句！"),
    ]
    done = chunks[-1]
    assert done.kind == "done"
    assert done.result.parse_ok is True
    assert done.result.reply == "第一句。第二句！"
    assert done.result.first_sentence_ms is not None


def test_extractor_abandons_on_malformed_reply_head():
    from services.llm import _ReplyExtractor

    for raw in ('{"reply" "没有冒号"}', '{"reply": 123}', '{"reply": null}'):
        extractor = _ReplyExtractor()
        assert extractor.feed(raw) == []
        assert extractor.abandoned is True
        assert extractor.complete is False


def test_extractor_accepts_whitespace_around_key_and_colon():
    from services.llm import _ReplyExtractor

    extractor = _ReplyExtractor()
    sentences = extractor.feed('{  "reply"  :   "第一句。第二句"  , "slots": {}}')
    assert extractor.abandoned is False
    assert sentences == ["第一句。", "第二句"]


async def test_reply_word_inside_slots_is_not_mistaken_for_key():
    body = sse('{"slots": {"purpose": "\\"reply\\": \\"假的。\\""}, "reply": "真的。"}')
    client = make_client(lambda request: httpx.Response(200, content=body))
    async with client:
        chunks = [c async for c in client.turn_stream(State.ASK_PURPOSE, {}, [], "送快递")]

    assert [(c.kind, c.text) for c in chunks[:-1]] == [("sentence", "真的。")]
    assert chunks[-1].result.reply == "真的。"

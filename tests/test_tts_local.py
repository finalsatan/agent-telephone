"""T05 离线测试：LocalTtsClient 的本地服务协议、无鉴权、取消与截止时间。

服务端（CosyVoice3）本身不做单元测试：它需要 GPU 与模型，由 `local_tts/bench.py`
和 live 验收覆盖。这里用 httpx.MockTransport 测客户端，并验证它确实继承了
T04/T04b 的超时与取消语义。
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest

from services.tts import TtsError, TtsTimeout
from services.tts_local import LocalTtsClient

PCM_A = b"\x01\x00" * 100
PCM_B = b"\x02\x00" * 100


class ScriptedStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes],
        *,
        delays: list[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.delays = delays or [0.0] * len(chunks)
        self.error = error

    async def __aiter__(self):
        for chunk, delay in zip(self.chunks, self.delays):
            if delay:
                await asyncio.sleep(delay)
            yield chunk
        if self.error is not None:
            raise self.error


VOICE_ID = "abc123def456"  # 12 位，和服务端 sha1[:12] 一致


def make_client(handler, *, health: dict | None = None, **kwargs) -> LocalTtsClient:
    """包装 handler：GET /health 默认返回带 voice_id 的响应，POST 交给 handler。"""
    health_payload = {"status": "ok", "voice_id": VOICE_ID} if health is None else health

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json=health_payload)
        return handler(request)

    kwargs.setdefault("retry_base_delay", 0.0)
    return LocalTtsClient(
        base_url="http://127.0.0.1:50000",
        transport=httpx.MockTransport(wrapped),
        **kwargs,
    )


async def collect(client: LocalTtsClient, text: str, cancel: asyncio.Event | None = None) -> list[bytes]:
    return [chunk async for chunk in client.synth(text, cancel or asyncio.Event())]


# --------------------------------------------------------------------------- #
# 协议
# --------------------------------------------------------------------------- #


async def test_synth_posts_text_to_local_endpoint_without_auth():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=ScriptedStream([PCM_A, PCM_B]))

    client = make_client(handler)
    async with client:
        chunks = await collect(client, "您好，请问您找哪位？")

    assert chunks == [PCM_A, PCM_B]
    assert len(seen) == 1
    request = seen[0]
    assert request.url.path == "/synth"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {"text": "您好，请问您找哪位？"}


async def test_works_without_api_key_in_environment(monkeypatch):
    """本地服务不需要 SILICONFLOW_API_KEY；客户端构造不得依赖它。"""
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    client = make_client(lambda request: httpx.Response(200, stream=ScriptedStream([PCM_A])))
    async with client:
        assert await collect(client, "您好") == [PCM_A]


async def test_from_config_reads_local_url_and_sample_rate(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    config = {
        "tts": {
            "backend": "local",
            "local_url": "http://127.0.0.1:50999",
            "model": "FunAudioLLM/CosyVoice2-0.5B",
            "voice": "FunAudioLLM/CosyVoice2-0.5B:alex",
        },
        "audio": {"sample_rate": 16000},
    }
    client = LocalTtsClient.from_config(config)
    try:
        assert str(client._client.base_url).rstrip("/") == "http://127.0.0.1:50999"
        assert client.sample_rate == 16000
        assert client.endpoint == "/synth"
    finally:
        await client.aclose()


async def test_voice_id_is_fetched_once_before_first_synth():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "voice_id": VOICE_ID})
        return httpx.Response(200, stream=ScriptedStream([PCM_A]))

    client = LocalTtsClient(
        base_url="http://127.0.0.1:50000",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
    )
    assert client.voice_id is None
    async with client:
        assert await collect(client, "您好") == [PCM_A]
        assert client.voice_id == VOICE_ID
        assert await collect(client, "再说一句") == [PCM_A]

    # 只在第一次 synth 前取一次 /health；第二次直接用缓存
    assert calls == ["GET /health", "POST /synth", "POST /synth"]


async def test_fetch_voice_id_can_be_called_explicitly():
    client = make_client(lambda request: httpx.Response(200, stream=ScriptedStream([PCM_A])))
    async with client:
        assert await client.fetch_voice_id() == VOICE_ID
        assert client.voice_id == VOICE_ID


async def test_missing_voice_id_in_health_raises():
    client = make_client(
        lambda request: httpx.Response(200, stream=ScriptedStream([PCM_A])),
        health={"status": "ok"},
    )
    async with client:
        with pytest.raises(TtsError, match="missing voice_id"):
            await collect(client, "您好")


async def test_empty_text_makes_no_request():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=ScriptedStream([PCM_A]))

    client = make_client(handler)
    async with client:
        assert await collect(client, "  \n ") == []
    assert calls == 0


# --------------------------------------------------------------------------- #
# 取消（与 T04/T04b 同一语义）
# --------------------------------------------------------------------------- #


async def test_cancel_between_chunks_stops_within_200ms():
    loop = asyncio.get_running_loop()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=ScriptedStream([PCM_A, PCM_B, PCM_A], delays=[0.0, 5.0, 0.0])
        )

    client = make_client(handler)
    cancel = asyncio.Event()
    async with client:
        agen = client.synth("这是一段很长的文本，用来验证取消能立刻生效。", cancel)
        assert await anext(agen) == PCM_A

        started = loop.time()
        cancel.set()
        with pytest.raises(StopAsyncIteration):
            await anext(agen)
        elapsed = loop.time() - started

    assert elapsed < 0.2


async def test_cancel_during_retry_backoff_ends_quietly():
    """T04b 的回归在本地客户端同样成立：_Cancelled 不得泄漏。"""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="busy")

    client = make_client(handler, retry_base_delay=1.0, max_retries=2)
    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    cancel_at: list[float] = []

    async def cancel_soon() -> None:
        await asyncio.sleep(0.2)
        cancel_at.append(loop.time())
        cancel.set()

    canceller = asyncio.create_task(cancel_soon())
    async with client:
        agen = client.synth("您好", cancel)
        with pytest.raises(StopAsyncIteration):
            await anext(agen)
        elapsed = loop.time() - cancel_at[0]
    await asyncio.gather(canceller)

    assert calls == 1
    assert elapsed < 0.2


# --------------------------------------------------------------------------- #
# 截止时间
# --------------------------------------------------------------------------- #


async def test_first_chunk_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ScriptedStream([PCM_A], delays=[5.0]))

    client = make_client(handler, first_chunk_timeout_s=0.1, total_timeout_s=5)
    async with client:
        with pytest.raises(TtsTimeout, match="first-chunk"):
            await collect(client, "您好")


async def test_total_timeout_after_first_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ScriptedStream([PCM_A, PCM_B], delays=[0.0, 5.0]))

    client = make_client(handler, first_chunk_timeout_s=5, total_timeout_s=0.2)
    async with client:
        agen = client.synth("您好，我记下了。", asyncio.Event())
        assert await anext(agen) == PCM_A
        with pytest.raises(TtsTimeout, match="total timeout"):
            await anext(agen)


async def test_consumer_await_is_not_cancelled_by_total_timeout():
    unraisable: list[object] = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=ScriptedStream([PCM_A, PCM_B], delays=[0.0, 5.0]))

        client = make_client(handler, first_chunk_timeout_s=5, total_timeout_s=0.2)
        async with client:
            agen = client.synth("您好，我记下了。", asyncio.Event())
            assert await anext(agen) == PCM_A
            await asyncio.sleep(0.35)  # 比剩余总时限长
            with pytest.raises(TtsTimeout, match="total timeout"):
                await anext(agen)
    finally:
        sys.unraisablehook = old_hook

    assert unraisable == []


# --------------------------------------------------------------------------- #
# 重试边界
# --------------------------------------------------------------------------- #


async def test_retries_on_429_then_succeeds():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="busy")
        return httpx.Response(200, stream=ScriptedStream([PCM_A]))

    client = make_client(handler, max_retries=2)
    async with client:
        assert await collect(client, "您好") == [PCM_A]
    assert calls == 2


async def test_no_retry_after_partial_output():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200, stream=ScriptedStream([PCM_A], error=httpx.ReadError("connection lost"))
            )
        return httpx.Response(200, stream=ScriptedStream([PCM_B]))

    client = make_client(handler, max_retries=2)
    received: list[bytes] = []
    async with client:
        with pytest.raises(TtsError, match="tts stream failed"):
            async for chunk in client.synth("您好", asyncio.Event()):
                received.append(chunk)
    assert received == [PCM_A]
    assert calls == 1

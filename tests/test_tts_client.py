"""T04 离线测试：TtsClient 的请求参数、流式产出、取消、两层截止时间与重试
（用 httpx.MockTransport，不联网）。"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest

from services.tts import TtsClient, TtsError, TtsTimeout

PCM_A = b"\x01\x00" * 100
PCM_B = b"\x02\x00" * 100


class ScriptedStream(httpx.AsyncByteStream):
    """按脚本产出音频块；可在块之间 sleep，或在结束时抛传输错误。"""

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


def make_client(handler, **kwargs) -> TtsClient:
    kwargs.setdefault("retry_base_delay", 0.0)  # 测试里不要真的等退避
    return TtsClient(
        base_url="https://api.example.test/v1",
        model="FunAudioLLM/CosyVoice2-0.5B",
        voice="FunAudioLLM/CosyVoice2-0.5B:alex",
        api_key="sk-test",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def collect(client: TtsClient, text: str, cancel: asyncio.Event | None = None) -> list[bytes]:
    return [chunk async for chunk in client.synth(text, cancel or asyncio.Event())]


# --------------------------------------------------------------------------- #
# 请求与产出
# --------------------------------------------------------------------------- #


async def test_synth_sends_expected_payload_and_yields_chunks():
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
    assert request.url.path == "/v1/audio/speech"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {
        "model": "FunAudioLLM/CosyVoice2-0.5B",
        "input": "您好，请问您找哪位？",
        "voice": "FunAudioLLM/CosyVoice2-0.5B:alex",
        "response_format": "pcm",
        "sample_rate": 16000,
        "stream": True,
    }
    assert client.sample_rate == 16000


async def test_from_config_reads_tts_and_audio_settings(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-env")
    config = {
        "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
        "tts": {
            "model": "FunAudioLLM/CosyVoice2-0.5B",
            "voice": "FunAudioLLM/CosyVoice2-0.5B:alex",
            "first_chunk_timeout_s": 7,
            "total_timeout_s": 25,
        },
        "audio": {"sample_rate": 8000},
    }
    client = TtsClient.from_config(config)
    try:
        assert client.model == "FunAudioLLM/CosyVoice2-0.5B"
        assert client.voice == "FunAudioLLM/CosyVoice2-0.5B:alex"
        assert client.sample_rate == 8000
        assert client.first_chunk_timeout_s == 7
        assert client.total_timeout_s == 25
        assert str(client._client.base_url).rstrip("/") == "https://api.siliconflow.cn/v1"
    finally:
        await client.aclose()


async def test_from_config_kwargs_override_config(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-env")
    config = {
        "siliconflow": {"base_url": "https://api.example.test/v1"},
        "tts": {
            "model": "m",
            "voice": "v",
            "first_chunk_timeout_s": 7,
            "total_timeout_s": 25,
        },
        "audio": {"sample_rate": 16000},
    }
    client = TtsClient.from_config(config, first_chunk_timeout_s=1.5, sample_rate=44100)
    try:
        assert client.first_chunk_timeout_s == 1.5
        assert client.sample_rate == 44100
        assert client.total_timeout_s == 25
    finally:
        await client.aclose()


async def test_empty_text_makes_no_request():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=ScriptedStream([PCM_A]))

    client = make_client(handler)
    async with client:
        assert await collect(client, "") == []
        assert await collect(client, "   \n\t ") == []
    assert calls == 0


async def test_empty_audio_response_raises():
    client = make_client(lambda request: httpx.Response(200, stream=ScriptedStream([])))
    async with client:
        with pytest.raises(TtsError, match="no audio"):
            await collect(client, "您好")


# --------------------------------------------------------------------------- #
# 取消
# --------------------------------------------------------------------------- #


async def test_cancel_before_start_makes_no_request():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=ScriptedStream([PCM_A]))

    client = make_client(handler)
    cancel = asyncio.Event()
    cancel.set()
    async with client:
        assert await collect(client, "您好", cancel) == []
    assert calls == 0


async def test_cancel_between_chunks_stops_within_200ms():
    """收到第一块后置位 cancel：不抛异常，立即结束，不再等后面的音频。"""
    loop = asyncio.get_running_loop()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=ScriptedStream([PCM_A, PCM_B, PCM_A], delays=[0.0, 5.0, 0.0])
        )

    client = make_client(handler)
    cancel = asyncio.Event()
    async with client:
        agen = client.synth("这是一段很长的文本，用来验证取消能立刻生效。", cancel)
        first = await anext(agen)
        assert first == PCM_A

        started = loop.time()
        cancel.set()
        with pytest.raises(StopAsyncIteration):
            await anext(agen)
        elapsed = loop.time() - started

    assert elapsed < 0.2, f"取消后仍等了 {elapsed * 1000:.0f}ms"


async def test_cancel_during_retry_backoff_ends_quietly():
    """T04b 回归：退避等待期间置位 cancel，内部 _Cancelled 不得泄漏给调用方。

    旧实现里 _pause 抛 _Cancelled，这个异常产生在 synth() 的 except 分支内部，
    同级的 except _Cancelled 接不住，调用方会收到 _Cancelled。
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="rate limited")

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

    assert calls == 1, "退避期间不应已经发起第二次请求"
    assert elapsed < 0.2, f"取消后仍等了 {elapsed * 1000:.0f}ms"


# --------------------------------------------------------------------------- #
# 截止时间（范围不跨越 yield）
# --------------------------------------------------------------------------- #


async def test_first_chunk_timeout():
    loop = asyncio.get_running_loop()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ScriptedStream([PCM_A], delays=[5.0]))

    client = make_client(handler, first_chunk_timeout_s=0.1, total_timeout_s=5)
    started = loop.time()
    async with client:
        with pytest.raises(TtsTimeout, match="first-chunk"):
            await collect(client, "您好")
    assert loop.time() - started < 1.0


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
    """消费者在两块之间 await（写文件、播报）不能被 TTS 的总时限取消。"""
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

            # 比剩余总时限长；旧实现会把 CancelledError 泄漏给消费者。
            await asyncio.sleep(0.35)

            with pytest.raises(TtsTimeout, match="total timeout"):
                await anext(agen)
    finally:
        sys.unraisablehook = old_hook

    assert unraisable == []  # 没有无人接收的异常


def test_timeout_is_tts_error_subclass():
    assert issubclass(TtsTimeout, TtsError)
    assert issubclass(TtsError, RuntimeError)


# --------------------------------------------------------------------------- #
# 重试
# --------------------------------------------------------------------------- #


async def test_retries_on_429_then_succeeds():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, stream=ScriptedStream([PCM_A]))

    client = make_client(handler, max_retries=2)
    async with client:
        assert await collect(client, "您好") == [PCM_A]
    assert calls == 2


async def test_retries_on_transport_error_before_audio():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom", request=request)
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
                200,
                stream=ScriptedStream([PCM_A], error=httpx.ReadError("connection lost")),
            )
        return httpx.Response(200, stream=ScriptedStream([PCM_B]))

    client = make_client(handler, max_retries=2)
    received: list[bytes] = []
    async with client:
        with pytest.raises(TtsError, match="tts stream failed"):
            async for chunk in client.synth("您好", asyncio.Event()):
                received.append(chunk)
    assert received == [PCM_A]  # 已经送出的块保持有效
    assert calls == 1  # 不重发已经开始的请求


async def test_client_error_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad voice")

    client = make_client(handler, max_retries=2)
    async with client:
        with pytest.raises(TtsError, match="HTTP 400"):
            await collect(client, "您好")
    assert calls == 1


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(TtsError, match="missing API key"):
        TtsClient(
            base_url="https://api.example.test/v1",
            model="m",
            voice="v",
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        )

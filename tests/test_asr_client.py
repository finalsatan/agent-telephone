"""T03：`services/asr.py` 的离线测试（httpx.MockTransport，不联网）。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from core.normalize import extract_phone
from services.asr import (
    API_KEY_ENV,
    AsrClient,
    AsrError,
    load_wav_pcm,
    pcm_duration_ms,
    pcm_to_wav,
)

# 0.4s 的 16kHz 16-bit 单声道 PCM（内容不重要，只需能从上传体里认出原始字节）
PCM = b"\x01\x02\x03\x04" * 3200


def make_client(handler, **kwargs) -> AsrClient:
    kwargs.setdefault("api_key", "test-key")
    return AsrClient(
        base_url="https://asr.test/v1",
        model="Qwen/Qwen3-ASR-1.7B",
        transport=httpx.MockTransport(handler),
        retry_base_delay=0.0,
        **kwargs,
    )


def json_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=json.dumps(payload, ensure_ascii=False))


class TestTranscribeRequest:
    async def test_posts_wav_multipart_and_parses_text(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response({"text": "  喂，你好。  "})

        client = make_client(handler)
        result = await client.transcribe(PCM)
        await client.aclose()

        assert result.text == "喂，你好。"
        assert result.audio_ms == pytest.approx(pcm_duration_ms(PCM))
        assert result.latency_ms >= 0.0

        request = seen[0]
        assert request.method == "POST"
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = request.read()
        assert b'name="model"' in body and b"Qwen/Qwen3-ASR-1.7B" in body
        assert b'name="file"' in body and b"audio.wav" in body
        # 上传的是 WAV 容器而不是裸 PCM，且 PCM 内容原样在里面
        assert body.startswith(b"--") and b"RIFF" in body and b"WAVE" in body
        assert PCM in body

    async def test_pcm_to_wav_header(self) -> None:
        wav = pcm_to_wav(PCM)
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
        # 16-bit 单声道 16kHz 的字节率 = 32000，data 块长度 = len(PCM)
        assert PCM in wav
        assert b"\x00\x7d\x00\x00" in wav  # 32000 字节/秒

    async def test_empty_audio_returns_empty_result_without_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - 不该被调用
            raise AssertionError("空音频不应发请求")

        client = make_client(handler)
        result = await client.transcribe(b"")
        await client.aclose()

        assert result.text == ""
        assert result.audio_ms == 0.0
        assert result.latency_ms == 0.0


class TestRetry:
    async def test_retries_on_5xx_then_succeeds(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return json_response({"error": "busy"}, status=503)
            return json_response({"text": "第二次成功"})

        client = make_client(handler)
        result = await client.transcribe(PCM)
        await client.aclose()

        assert result.text == "第二次成功"
        assert len(calls) == 2

    async def test_retries_exhausted_raises(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return json_response({"error": "busy"}, status=500)

        client = make_client(handler, max_retries=2)
        with pytest.raises(AsrError, match="3 attempt"):
            await client.transcribe(PCM)
        await client.aclose()

        assert len(calls) == 3

    async def test_429_is_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return json_response({"error": "rate limited"}, status=429)
            return json_response({"text": "ok"})

        client = make_client(handler)
        assert (await client.transcribe(PCM)).text == "ok"
        await client.aclose()
        assert len(calls) == 2

    async def test_4xx_is_not_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return json_response({"error": "bad model"}, status=400)

        client = make_client(handler)
        with pytest.raises(AsrError):
            await client.transcribe(PCM)
        await client.aclose()
        assert len(calls) == 1

    async def test_transport_error_is_retried_then_wrapped(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            raise httpx.ConnectError("no route", request=request)

        client = make_client(handler, max_retries=1)
        with pytest.raises(AsrError, match="2 attempt"):
            await client.transcribe(PCM)
        await client.aclose()
        assert len(calls) == 2


class TestResponseParsing:
    async def test_missing_text_field_raises(self) -> None:
        client = make_client(lambda request: json_response({"usage": {"seconds": 1}}))
        with pytest.raises(AsrError, match="no text"):
            await client.transcribe(PCM)
        await client.aclose()

    async def test_non_json_body_raises(self) -> None:
        client = make_client(lambda request: httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(AsrError, match="not JSON"):
            await client.transcribe(PCM)
        await client.aclose()

    async def test_uses_whitespace_trimmed_text(self) -> None:
        client = make_client(lambda request: json_response({"text": "\n 一三八零零一三八零零零 \n"}))
        result = await client.transcribe(PCM)
        await client.aclose()
        assert extract_phone(result.text) == "13800138000"


class TestFromConfig:
    def test_reads_base_url_model_and_sample_rate(self) -> None:
        config = {
            "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
            "asr": {"model": "Qwen/Qwen3-ASR-1.7B"},
            "audio": {"sample_rate": 16000},
        }
        client = AsrClient.from_config(config, api_key="k")
        assert client.model == "Qwen/Qwen3-ASR-1.7B"
        assert client.sample_rate == 16000
        assert str(client._client.base_url) == "https://api.siliconflow.cn/v1/"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        with pytest.raises(AsrError, match=API_KEY_ENV):
            AsrClient(base_url="https://api.siliconflow.cn/v1", model="m")

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV, "from-env")
        client = AsrClient(base_url="https://api.siliconflow.cn/v1", model="m")
        assert client._client.headers["authorization"] == "Bearer from-env"


class TestLoadWavPcm:
    def test_16k_mono_roundtrip(self, tmp_path: Path) -> None:
        samples = (np.sin(np.linspace(0, 20, 16000)) * 20000).astype("<i2")
        path = tmp_path / "a.wav"
        sf.write(str(path), samples, 16000, subtype="PCM_16")

        pcm = load_wav_pcm(path)

        assert len(pcm) == len(samples) * 2
        decoded = np.frombuffer(pcm, dtype="<i2")
        assert np.max(np.abs(decoded.astype(int) - samples.astype(int))) <= 1

    def test_stereo_is_downmixed(self, tmp_path: Path) -> None:
        left = (np.ones(1600) * 1000).astype("<i2")
        right = (np.ones(1600) * 3000).astype("<i2")
        path = tmp_path / "s.wav"
        sf.write(str(path), np.stack([left, right], axis=1), 16000, subtype="PCM_16")

        pcm = load_wav_pcm(path)

        assert len(pcm) == 1600 * 2
        assert np.allclose(np.frombuffer(pcm, dtype="<i2"), 2000, atol=2)

    def test_resamples_to_16k(self, tmp_path: Path) -> None:
        path = tmp_path / "8k.wav"
        sf.write(str(path), np.zeros(8000, dtype="<i2"), 8000, subtype="PCM_16")

        pcm = load_wav_pcm(path)

        assert len(pcm) == 16000 * 2  # 1s @ 8k → 1s @ 16k


class TestAsyncContextManager:
    async def test_client_can_be_used_as_context_manager(self) -> None:
        async with make_client(lambda request: json_response({"text": "ok"})) as client:
            assert (await client.transcribe(PCM)).text == "ok"

    async def test_non_object_json_body_raises(self) -> None:
        client = make_client(lambda request: json_response(["not", "an", "object"]))
        with pytest.raises(AsrError, match="not an object"):
            await client.transcribe(PCM)
        await client.aclose()

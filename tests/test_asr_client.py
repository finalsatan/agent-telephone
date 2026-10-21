"""T03 / T03b：`services/asr.py` 的离线测试（httpx.MockTransport，不联网）。"""

from __future__ import annotations

import asyncio
import json
import time
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
    AsrTimeout,
    load_wav_pcm,
    pcm_duration_ms,
    pcm_to_wav,
)

# 0.4s 的 16kHz 16-bit 单声道 PCM（内容不重要，只需能从上传体里认出原始字节）
PCM = b"\x01\x02\x03\x04" * 3200


def make_client(handler, **kwargs) -> AsrClient:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("retry_base_delay", 0.0)
    return AsrClient(
        base_url="https://asr.test/v1",
        model="Qwen/Qwen3-ASR-1.7B",
        transport=httpx.MockTransport(handler),
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


class TestTimeouts:
    """T03b：实时链路的 deadline 不得随重试无限放大（旧实现最坏 20s×3＋退避 ≈ 61s）。"""

    async def test_attempt_timeout_is_retried_within_total(self) -> None:
        calls: list[int] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                await asyncio.sleep(0.5)  # 超过单次尝试的截止时间
            return json_response({"text": "第二次成功"})

        client = make_client(
            handler, attempt_timeout_s=0.05, total_timeout_s=1.0, max_retries=1
        )
        result = await client.transcribe(PCM)
        await client.aclose()

        assert result.text == "第二次成功"
        assert len(calls) == 2

    async def test_total_timeout_raises_asr_timeout_and_stops_retrying(self) -> None:
        starts: list[float] = []
        t0 = time.perf_counter()

        async def handler(request: httpx.Request) -> httpx.Response:
            starts.append(time.perf_counter() - t0)
            await asyncio.sleep(1.0)  # 永不返回
            return json_response({"text": "never"})  # pragma: no cover

        client = make_client(
            handler,
            attempt_timeout_s=0.05,
            total_timeout_s=0.12,
            max_retries=5,
        )
        started = time.perf_counter()
        with pytest.raises(AsrTimeout, match="total timeout"):
            await client.transcribe(PCM)
        elapsed = time.perf_counter() - started
        await client.aclose()

        assert elapsed < 0.5, f"总截止没生效: {elapsed:.3f}s"
        # 总截止之后不得再发起尝试
        assert starts and max(starts) <= 0.12 + 0.05, f"截止后仍重试: {starts}"

    async def test_backoff_past_total_deadline_does_not_retry(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return json_response({"error": "busy"}, status=503)

        client = make_client(
            handler,
            attempt_timeout_s=1.0,
            total_timeout_s=0.12,
            retry_base_delay=0.5,  # 退避本身就会越过总截止
            max_retries=2,
        )
        with pytest.raises(AsrTimeout, match="total timeout"):
            await client.transcribe(PCM)
        await client.aclose()

        assert len(calls) == 1, "退避会越过总截止时不应再试一次"

    async def test_latency_ms_includes_retries_and_backoff(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return json_response({"error": "busy"}, status=500)
            return json_response({"text": "ok"})

        client = make_client(handler, retry_base_delay=0.05, max_retries=1)
        result = await client.transcribe(PCM)
        await client.aclose()

        assert len(calls) == 2
        assert result.latency_ms >= 45.0, f"退避没计入 latency_ms: {result.latency_ms:.1f}"


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

    def test_reads_t03b_timeouts(self) -> None:
        config = {
            "siliconflow": {"base_url": "https://api.siliconflow.cn/v1"},
            "asr": {
                "model": "Qwen/Qwen3-ASR-1.7B",
                "attempt_timeout_s": 3,
                "total_timeout_s": 5,
                "recheck_attempt_timeout_s": 20,
                "recheck_total_timeout_s": 60,
            },
        }
        client = AsrClient.from_config(config, api_key="k")
        assert client.attempt_timeout_s == 3
        assert client.total_timeout_s == 5
        # T11 离线校正是构造时显式传入，from_config 只透传 kwargs
        recheck = AsrClient.from_config(
            config,
            api_key="k",
            attempt_timeout_s=config["asr"]["recheck_attempt_timeout_s"],
            total_timeout_s=config["asr"]["recheck_total_timeout_s"],
        )
        assert (recheck.attempt_timeout_s, recheck.total_timeout_s) == (20, 60)

    def test_defaults_when_config_lacks_timeouts(self) -> None:
        client = AsrClient.from_config(
            {"siliconflow": {"base_url": "https://x/v1"}, "asr": {"model": "m"}}, api_key="k"
        )
        assert (client.attempt_timeout_s, client.total_timeout_s) == (3.0, 5.0)

    def test_repo_config_declares_t03b_timeouts(self) -> None:
        import yaml

        root = Path(__file__).resolve().parents[1]
        asr = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["asr"]
        assert asr["attempt_timeout_s"] == 3
        assert asr["total_timeout_s"] == 5
        assert asr["recheck_attempt_timeout_s"] == 20
        assert asr["recheck_total_timeout_s"] == 60

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


class TestTimeoutSemantics:
    async def test_retry_exhausted_by_attempt_timeouts_is_plain_error(self) -> None:
        """单次尝试超时且重试用尽，但没到总截止：抛 AsrError 而非 AsrTimeout。"""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(1.0)  # 每次都超时
            return json_response({"text": "never"})  # pragma: no cover

        client = make_client(
            handler, attempt_timeout_s=0.03, total_timeout_s=5.0, max_retries=1
        )
        with pytest.raises(AsrError) as excinfo:
            await client.transcribe(PCM)
        await client.aclose()

        assert not isinstance(excinfo.value, AsrTimeout)

    async def test_httpx_timeout_defaults_to_the_larger_deadline(self) -> None:
        client = make_client(lambda request: json_response({"text": "ok"}))
        recheck = AsrClient(
            base_url="https://asr.test/v1",
            model="m",
            api_key="k",
            attempt_timeout_s=20,
            total_timeout_s=60,
        )
        try:
            assert client.timeout == 5.0  # max(attempt 3s, total 5s)
            assert recheck.timeout == 60.0  # 离线校正不被默认读超时先打断
        finally:
            await client.aclose()
            await recheck.aclose()

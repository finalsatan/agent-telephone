"""T05b 离线测试：本地 TTS 服务的 hop 复位、锁释放、voice_id 与参数校验。

不加载真模型、不需要 GPU：用 fake engine 模拟上游在流式推理中修改
`model.token_hop_len`（这正是 T05 首包变慢的根因）。
"""

from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from local_tts.server import (
    OUT_SAMPLE_RATE,
    SYNTH_LOCK,
    compute_voice_id,
    create_app,
    iter_pcm,
    parse_args,
)


class FakeTensor:
    """只实现 iter_pcm 用到的 torch.Tensor 子集。"""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    def squeeze(self, dim: int) -> "FakeTensor":
        return FakeTensor(np.squeeze(self._data, axis=dim))

    def float(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._data


class FakeModel:
    def __init__(self, hop: int = 25) -> None:
        self.token_hop_len = hop
        self.token_max_hop_len = 100
        self.stream_scale_factor = 2


class FakeEngine:
    """模拟 CosyVoice2Model.tts()：每出一块就把 token_hop_len 翻倍且不复位。"""

    sample_rate = OUT_SAMPLE_RATE

    def __init__(self) -> None:
        self.model = FakeModel()
        self.initial_token_hop_len = self.model.token_hop_len
        self.hops: list[int] = []

    def inference_zero_shot(self, text, *args, **kwargs):
        assert kwargs.get("stream") is True
        self.hops.append(self.model.token_hop_len)
        for _ in range(2):
            self.model.token_hop_len = min(
                self.model.token_max_hop_len,
                self.model.token_hop_len * self.model.stream_scale_factor,
            )
            yield {"tts_speech": FakeTensor(np.zeros((1, 1600), dtype=np.float32))}


@pytest.fixture(autouse=True)
def fake_soxr(monkeypatch):
    """测试环境没装 soxr；引擎采样率刚好是 16k，重采样器不会被用到。"""
    module = types.SimpleNamespace(
        ResampleStream=lambda *args, **kwargs: types.SimpleNamespace(
            resample_chunk=lambda x, last=False: x
        )
    )
    monkeypatch.setitem(sys.modules, "soxr", module)


def test_hop_is_reset_before_every_request():
    engine = FakeEngine()

    list(iter_pcm(engine, "第一句"))
    assert engine.model.token_hop_len == 100  # 上游已经把实例属性翻倍到上限
    list(iter_pcm(engine, "第二句"))

    # 第二次推理开始时应恢复为加载时保存的初始值，而不是沿用 100
    assert engine.hops == [25, 25]


def test_lock_released_when_generator_closed_early():
    engine = FakeEngine()
    generator = iter_pcm(engine, "第一句")
    next(generator)  # 持锁并产出一块
    assert SYNTH_LOCK.locked()

    generator.close()  # 客户端取消/断开
    assert not SYNTH_LOCK.locked()

    # 锁还能立刻被下一次请求拿到
    assert SYNTH_LOCK.acquire(blocking=False)
    SYNTH_LOCK.release()


def test_pcm_chunks_are_non_empty():
    engine = FakeEngine()
    chunks = list(iter_pcm(engine, "您好"))
    assert len(chunks) == 2
    assert all(chunk for chunk in chunks)
    assert all(len(chunk) % 2 == 0 for chunk in chunks)


def test_empty_text_returns_400():
    app = create_app(parse_args([]))
    client = TestClient(app)  # 不进入 lifespan，避免加载真模型

    response = client.post("/synth", json={"text": "   "})
    assert response.status_code == 400


def test_health_exposes_voice_id():
    app = create_app(parse_args([]))
    app.state.engine = object()
    app.state.voice_id = "abc123def456"
    client = TestClient(app)

    body = client.get("/health").json()
    assert body["voice_id"] == "abc123def456"
    assert body["sample_rate"] == OUT_SAMPLE_RATE


def test_voice_id_is_sha1_of_prompt_audio_plus_text(tmp_path):
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"fake-wav-bytes")
    text = "You are a helpful assistant.<|endofprompt|>你好。"

    expected = hashlib.sha1(b"fake-wav-bytes" + text.encode("utf-8")).hexdigest()[:12]
    assert compute_voice_id(prompt, text) == expected
    assert len(compute_voice_id(prompt, text)) == 12

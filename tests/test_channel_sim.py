"""T08 电话信道模拟的离线测试（纯 numpy，不联网、不依赖真实音频文件）。

覆盖任务书的三条验收：输出长度、8kHz 以上频率能量明显衰减、静音仍为静音；
另加 μ-law 编解码自身的一致性用例，便于以后改动滤波器时定位问题。
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.channel_sim import (
    MULAW_BIAS,
    MULAW_CLIP,
    mulaw_decode,
    mulaw_encode,
    simulate,
)

FS = 16000


def tone(freq_hz: float, *, seconds: float = 0.5, amplitude: float = 0.5) -> np.ndarray:
    """16kHz 单声道 int16 正弦波（amplitude 为满量程比例）。"""
    t = np.arange(int(FS * seconds), dtype=np.float64) / FS
    return np.round(amplitude * 32767 * np.sin(2 * np.pi * freq_hz * t)).astype("<i2")


def rms(samples: np.ndarray) -> float:
    x = samples.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def sim(samples: np.ndarray) -> np.ndarray:
    return np.frombuffer(simulate(samples.astype("<i2").tobytes()), dtype="<i2")


# --------------------------------------------------------------------------- #
# 输出长度与格式
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_samples", [1, 2, 3, 159, 160, 161, 1599, 1600, 8000])
def test_output_length_matches_input(num_samples: int) -> None:
    source = np.zeros(num_samples, dtype="<i2")
    out = simulate(source.tobytes())

    # 字节数必须是偶数的 16-bit PCM，且与输入等长（任务书允许 <= 20ms 误差，
    # 这里实测为 0 个样本）。
    assert len(out) % 2 == 0
    assert len(out) == len(source.tobytes())


def test_empty_input_returns_empty() -> None:
    assert simulate(b"") == b""


def test_odd_byte_length_rejected() -> None:
    with pytest.raises(ValueError, match="偶数"):
        simulate(b"\x00\x00\x00")


def test_output_is_little_endian_int16_and_bounded() -> None:
    # 0.8 满量程是正常说话电平；满量程（1.0）时插值滤波器的过冲会贴到
    # int16 边界，那是真实线路也会有的削顶，不作为本用例的断言对象。
    source = tone(1000, seconds=0.2, amplitude=0.8)
    out = np.frombuffer(simulate(source.tobytes()), dtype="<i2")

    assert out.dtype == np.dtype("<i2")
    # int16 范围内，且不落在 rails 上（落上去就是削顶失真）
    assert out.min() > -32768 and out.max() < 32767
    # 没有环绕/爆音：1000Hz、0.2s 应有约 400 次过零
    zero_crossings = int(np.sum(np.diff(np.signbit(out)) != 0))
    assert 385 <= zero_crossings <= 415


# --------------------------------------------------------------------------- #
# 频响：语音带保留、8kHz 以上衰减
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq_hz", [300, 500, 1000, 2000, 3000])
def test_speech_band_is_preserved(freq_hz: int) -> None:
    source = tone(freq_hz)
    out = sim(source)

    ratio = rms(out) / rms(source)
    assert 0.7 < ratio < 1.3, f"{freq_hz}Hz 能量比异常：{ratio:.3f}"


@pytest.mark.parametrize("freq_hz", [5000, 10000, 12000])
def test_frequencies_above_4khz_are_attenuated(freq_hz: int) -> None:
    source = tone(freq_hz)
    out = sim(source)

    ratio = rms(out) / rms(source)
    # 任务书要求"明显衰减"：这里要求至少 20dB。
    assert ratio < 0.1, f"{freq_hz}Hz 能量比 {ratio:.4f}，衰减不足 20dB"


def test_4khz_sits_in_transition_band_and_is_suppressed() -> None:
    # 4kHz 正好是 8kHz 采样的奈奎斯特频率，必须被低通压掉，否则会落到直流。
    source = tone(4000)
    ratio = rms(sim(source)) / rms(source)

    assert ratio < 0.1


# --------------------------------------------------------------------------- #
# 静音
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_samples", [16, 160, 1600])
def test_silence_stays_silent(num_samples: int) -> None:
    out = sim(np.zeros(num_samples, dtype="<i2"))

    # μ-law 的零码字解码为精确 0，滤波器对全零输入也保持全零。
    assert np.all(out == 0)


def test_no_noise_added_to_digital_silence() -> None:
    # 与上面同义，但明确"没有抖动/底噪"这一设计选择，避免以后误加 dither。
    out = simulate(b"\x00\x00" * 1600)
    assert set(out) == {0}


# --------------------------------------------------------------------------- #
# G.711 μ-law 编解码
# --------------------------------------------------------------------------- #
def test_mulaw_zero_is_exact_for_both_signs() -> None:
    codes = mulaw_encode(np.array([0, 0, 0], dtype=np.int16))
    assert list(codes) == [0xFF, 0xFF, 0xFF]
    assert list(mulaw_decode(codes)) == [0, 0, 0]


def test_mulaw_sign_bit_convention() -> None:
    # ITU G.711：正值码字 MSB=1（如 +0 = 0xFF），负值 MSB=0（如 -0 = 0x7F）。
    assert int(mulaw_encode(np.array([32767], dtype=np.int16))[0]) == 0x80
    assert int(mulaw_encode(np.array([-32768], dtype=np.int16))[0]) == 0x00
    assert int(mulaw_encode(np.array([1], dtype=np.int16))[0]) == 0xFF
    assert int(mulaw_encode(np.array([-1], dtype=np.int16))[0]) == 0x7F


def test_mulaw_decode_is_monotonic() -> None:
    ramp = np.arange(-32768, 32768, dtype=np.int32)
    decoded = mulaw_decode(mulaw_encode(ramp)).astype(np.int64)

    assert np.all(np.diff(decoded) >= 0)


def test_mulaw_roundtrip_error_within_codec_tolerance() -> None:
    ramp = np.arange(-30000, 30001, dtype=np.int32)
    decoded = mulaw_decode(mulaw_encode(ramp)).astype(np.int64)
    error = np.abs(decoded - ramp)

    # μ-law 的相对量化误差约 3%；近零区有 132 的死区，单看绝对值有界即可。
    loud = np.abs(ramp) >= 1000
    assert np.max(error[loud] / np.abs(ramp[loud])) < 0.05
    assert np.max(error) <= MULAW_BIAS * 5


def test_mulaw_clips_instead_of_wrapping() -> None:
    over = np.array([40000, -40000, 32767, -32768], dtype=np.int64)
    codes = mulaw_encode(over)

    assert list(codes) == [0x80, 0x00, 0x80, 0x00]
    assert MULAW_CLIP < 40000  # 常量本身说明削顶发生在编码前


# --------------------------------------------------------------------------- #
# 其它性质
# --------------------------------------------------------------------------- #
def test_simulate_is_deterministic() -> None:
    source = tone(700, seconds=0.1).tobytes()
    assert simulate(source) == simulate(source)


def test_simulate_does_not_mutate_input() -> None:
    source = tone(700, seconds=0.1)
    before = source.copy()
    simulate(source.tobytes())
    assert np.array_equal(source, before)

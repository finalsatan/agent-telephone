"""电话信道模拟（T08）：16kHz → 8kHz → G.711 μ-law 编解码 → 16kHz。

用途：让离线回放与测试页面听到的是"经过电话线路"的声音，避免我们按宽带音频
（8kHz 以上有能量）调参数，而真实来电是窄带的。设计文档第 2 节的链路上，
本模块位于"麦克风/回放"之后、VAD 之前，默认开启、可开关。

处理步骤（每一项都要能在没有真实线路时复现）：

1. 16kHz → 8kHz：先用窗函数 sinc 低通（截止 ``LOWPASS_CUTOFF_HZ``）抗混叠，
   再隔点抽取；
2. G.711 μ-law 编码再解码（8kHz、8-bit 非线性量化，电话线路的典型量化噪声）；
3. 8kHz → 16kHz：零插值 + 同一条低通滤波器，保持 16kHz、单声道、16-bit PCM。

约定：输入输出都是 16kHz 单声道 16-bit 小端 PCM；输出与输入等长（点数相同，
误差 0 点，任务书允许 20ms）。

不使用 ``audioop``（Python 3.13 已移除）：μ-law 与重采样都用 numpy 实现。
"""

from __future__ import annotations

import numpy as np

# 任务书约定：整条链路 16kHz、单声道、16-bit PCM。
SAMPLE_RATE_HZ = 16000
TELEPHONE_SAMPLE_RATE_HZ = 8000

# 低通截止频率：G.711 的采样率是 8kHz（奈奎斯特 4kHz），电话话音频段约
# 300–3400Hz。取 3400Hz 作抗混叠截止，既保留语音主要能量，又能让 8kHz
# 以上的分量在降采样前被压掉。
LOWPASS_CUTOFF_HZ = 3400.0
# 63 抽头 + Blackman 窗：过渡带约 1.4kHz，10kHz 处衰减 > 80dB；群延迟
# 31 个样本（16kHz 下 < 2ms），两端用边缘延拓补齐，不会引入直流跳变。
LOWPASS_TAPS = 63

# G.711 μ-law 常量（ITU-T G.711 / 常见参考实现）。
MULAW_BIAS = 0x84  # 132
MULAW_CLIP = 32635
# 段表：编码幅度（已加 BIAS、16-bit 范围）落在哪一段，索引即 3-bit 指数。
_MULAW_SEGMENT_ENDS = np.array(
    [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF],
    dtype=np.int32,
)


def _design_lowpass(
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    num_taps: int = LOWPASS_TAPS,
) -> np.ndarray:
    """窗函数法设计线性相位低通 FIR（归一化到直流增益 1）。"""
    if num_taps % 2 == 0:
        raise ValueError("num_taps 必须是奇数，否则群延迟不是整数个样本")
    if not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError("cutoff_hz 必须在 (0, 奈奎斯特频率) 之间")

    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    # 理想低通冲激响应：h[n] = 2fc/fs * sinc(2fc/fs * n)
    normalized = cutoff_hz / (sample_rate_hz / 2.0)
    taps = normalized * np.sinc(normalized * n)
    taps *= np.blackman(num_taps)
    return taps / taps.sum()


_LOWPASS_16K = _design_lowpass()


def _apply_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """等长滤波；两端用边缘值延拓，避免首尾的瞬态被当成语音。"""
    half = len(taps) // 2
    padded = np.pad(x, (half, half), mode="edge")
    return np.convolve(padded, taps, mode="valid")


def _downsample_by_2(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    return _apply_fir(x, taps)[::2]


def _upsample_by_2(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    up = np.zeros(2 * len(x), dtype=np.float64)
    up[::2] = x
    # 零插值把能量摊到两倍带宽上，幅度减半，这里乘 2 补偿。
    return 2.0 * _apply_fir(up, taps)


def mulaw_encode(samples: np.ndarray) -> np.ndarray:
    """16-bit 线性 PCM → 8-bit μ-law 码字（无符号 uint8，MSB=1 表示正值）。"""
    s = np.clip(np.round(np.asarray(samples, dtype=np.float64)), -32768, 32767)
    s = s.astype(np.int32)

    magnitude = np.minimum(np.abs(s), MULAW_CLIP) + MULAW_BIAS
    # 段表索引 = floor(log2(magnitude)) - 7，限定在 0..7。
    exponent = np.searchsorted(_MULAW_SEGMENT_ENDS, magnitude, side="left")
    exponent = np.minimum(exponent, 7).astype(np.int32)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    sign = np.where(s < 0, 0x80, 0x00)
    # μ-law 是反码形式：先按位取反，再由解码端还原。
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """8-bit μ-law 码字 → 16-bit 线性 PCM（int16）。"""
    u = ~np.asarray(codes, dtype=np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = ((mantissa << 3) + MULAW_BIAS) << exponent
    # 取反后的 MSB 为 1 表示负值，0 表示正值。
    sample = np.where((u & 0x80) != 0, MULAW_BIAS - magnitude, magnitude - MULAW_BIAS)
    return sample.astype(np.int16)


def simulate(pcm16k: bytes) -> bytes:
    """把 16kHz PCM 过一遍电话信道，返回等长的 16kHz PCM。"""
    if len(pcm16k) % 2:
        raise ValueError("PCM 字节数必须是偶数（16-bit 小端）")
    samples = np.frombuffer(pcm16k, dtype="<i2")
    if samples.size == 0:
        return b""

    x = samples.astype(np.float64)

    # 16kHz → 8kHz（先低通再抽取，抗混叠）
    telephone = _downsample_by_2(x, _LOWPASS_16K)

    # 电话线路的 8-bit 非线性量化
    codes = mulaw_encode(telephone)
    decoded = mulaw_decode(codes).astype(np.float64)

    # 8kHz → 16kHz
    restored = _upsample_by_2(decoded, _LOWPASS_16K)

    # 8kHz 往返最多多出 1 个样本，裁到与输入等长。
    out = np.clip(np.round(restored[: samples.size]), -32768, 32767)
    return out.astype("<i2").tobytes()


__all__ = [
    "LOWPASS_CUTOFF_HZ",
    "LOWPASS_TAPS",
    "MULAW_BIAS",
    "MULAW_CLIP",
    "SAMPLE_RATE_HZ",
    "TELEPHONE_SAMPLE_RATE_HZ",
    "mulaw_decode",
    "mulaw_encode",
    "simulate",
]

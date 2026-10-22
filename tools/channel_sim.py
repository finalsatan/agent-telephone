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

两种用法：

- `simulate(pcm)`：整段处理，内部两端做边缘延拓（离线回放用，T08/T10）；
- `ChannelSimulator.feed(frame)` / `flush()`：流式逐帧处理（T13 实时链路用）。
  逐帧调用 `simulate()` 会在每个帧边界重新做边缘延拓，帧与帧之间不连续，实测
  信噪比只有 29.5dB（第 6 次评审），因此实时链路必须用带状态的 `ChannelSimulator`。

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


class ChannelSimulator:
    """流式电话信道模拟（T13）：逐帧 `feed()`，帧间保留滤波器状态与抽取相位。

    与 `simulate()` 的差别只在边界处理：整段处理对首尾做边缘延拓，流式处理把
    尚未到达的样本当 0。因此除首尾各一个滤波器长度（`LOWPASS_TAPS` 个样本）外，
    两者的输出逐点相同（`tests/test_channel_sim.py` 用 ≥60dB 信噪比断言）。

    **群延迟**：输出相对输入固定延迟 ``2 * (LOWPASS_TAPS // 2) = 62`` 个样本
    （16kHz 下 3.875ms）。即 `feed()`/`flush()` 拼起来的第 i 个输出样本，对应
    整段处理的第 i-62 个输入样本。总和与输入等长：延迟的那 62 个样本在开头以
    静音（0）出现，`flush()` 交出末尾剩下的样本。

    实现上不做“每帧从零开始卷积再拼”，而是按绝对样本索引推进：

    - 降采样：8k 第 m 点 = ``Σ_k h[k] · x[2m + half - k]``，等输入攒到
      ``2m + half`` 才算第 m 点，抽到的相位由绝对索引决定（奇数长度帧也对）；
    - μ-law 编解码逐点进行，与整段处理完全一致；
    - 升采样：16k 第 i 点 = ``2 · Σ_{k ≡ i+half (mod 2)} h[k] · d[(i+half-k)/2]``，
      需要 8k 样本到 ``(i+half)/2``，整体因此产生 2·half 的延迟。

    只支持 16kHz、单声道、16-bit 小端；一个实例只能用一通电话，结束前调 `flush()`。
    """

    def __init__(self) -> None:
        self._taps = _LOWPASS_16K
        self._half = len(self._taps) // 2
        # 输出相对输入的固定延迟（样本数，16kHz）。
        self.delay_samples = 2 * self._half

        # 输入缓冲：self._x[0] 的绝对索引是 self._x0。
        self._x = np.zeros(0, dtype=np.float64)
        self._x0 = 0
        self._n_in = 0

        # 已解码的 8kHz 样本：self._d[0] 的绝对索引是 self._d0。
        self._d: list[float] = []
        self._d0 = 0
        self._n_d = 0

        self._out: list[float] = []
        self._n_out = 0
        self._finished = False

    # ---- 对外接口 ---------------------------------------------------------

    def feed(self, frame: bytes) -> bytes:
        """喂入任意长度的 PCM 帧，返回本次能确定的输出（长度可小于输入帧）。

        帧长不必是任何数的整数倍（任务书要求 512 / 160 / 161 点都要正确）。
        开头会先输出静音，凑满固定群延迟后输出才与整段处理对齐。
        """
        if self._finished:
            raise RuntimeError("ChannelSimulator 已 flush()，不能继续 feed()")
        if len(frame) % 2:
            raise ValueError("PCM 字节数必须是偶数（16-bit 小端）")
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float64)
        if samples.size == 0:
            return b""
        self._x = np.concatenate([self._x, samples])
        self._n_in += int(samples.size)
        self._produce(allow_padding=False)
        return self._take_output()

    def flush(self) -> bytes:
        """交出剩余的延迟样本（不足的部分按 0 补齐），使总输出与总输入等长。"""
        self._finished = True
        self._produce(allow_padding=True)
        return self._take_output()

    # ---- 内部：绝对索引推进 -----------------------------------------------

    def _produce(self, *, allow_padding: bool) -> None:
        """尽量多产出 16k 输出样本；`allow_padding=False` 时不够数据就停下。"""
        while self._n_out < self._n_in:
            j = self._n_out - self.delay_samples
            if j < 0:
                # 群延迟：开头先补静音。
                self._out.append(0.0)
                self._n_out += 1
                continue

            parity = (j + self._half) & 1
            j_hi = (j + self._half - parity) // 2
            while self._n_d <= j_hi:
                if not self._produce_telephone_sample(allow_padding=allow_padding):
                    return
            self._out.append(self._output_sample(j, parity))
            self._n_out += 1
            self._trim_telephone()

    def _produce_telephone_sample(self, *, allow_padding: bool) -> bool:
        """产出 8kHz 第 `self._n_d` 点（低通 + 抽取 + μ-law 编解码）。"""
        m = self._n_d
        lo = 2 * m - self._half
        hi = 2 * m + self._half
        if not allow_padding and hi >= self._n_in:
            return False

        window = self._x_window(lo, hi)
        # y8[m] = Σ_k h[k] · x[2m + half - k]，即 h 反序点乘 x[2m-half .. 2m+half]。
        value = float(np.dot(window, self._taps[::-1]))
        code = mulaw_encode(np.array([value], dtype=np.float64))
        self._d.append(float(mulaw_decode(code)[0]))
        self._n_d += 1

        # 下一段 8k 样本只需要 x[2(m+1)-half ..] 之后的输入，前面的可以丢掉。
        keep_from = 2 * self._n_d - self._half
        if keep_from > self._x0:
            drop = min(keep_from - self._x0, len(self._x))
            self._x = self._x[drop:]
            self._x0 += drop
        return True

    def _output_sample(self, j: int, parity: int) -> float:
        """16k 第 j 点（未加延迟）的升采样结果。"""
        coeffs = self._taps[parity::2]
        j_hi = (j + self._half - parity) // 2
        lo = j_hi - len(coeffs) + 1
        take_lo = max(lo, self._d0)
        if j_hi < take_lo:
            return 0.0
        segment = np.asarray(self._d[take_lo - self._d0 : j_hi + 1 - self._d0], dtype=np.float64)
        # d[j_hi - t] 配 h[parity + 2t]，即 segment 反序点乘 coeffs 的前 len(segment) 项。
        weights = coeffs[: len(segment)][::-1]
        return 2.0 * float(np.dot(segment, weights))

    def _x_window(self, lo: int, hi: int) -> np.ndarray:
        """取绝对索引区间 [lo, hi] 的输入样本，越界（还没收到）按 0。"""
        size = hi - lo + 1
        window = np.zeros(size, dtype=np.float64)
        start = max(lo, self._x0)
        end = min(hi + 1, self._x0 + len(self._x))
        if end > start:
            window[start - lo : end - lo] = self._x[start - self._x0 : end - self._x0]
        return window

    def _trim_telephone(self) -> None:
        """丢弃不再需要的 8k 历史（输出只需最近一个滤波器长度的样本）。"""
        keep_from = self._n_d - (self._half + 2)
        if keep_from > self._d0:
            drop = min(keep_from - self._d0, len(self._d))
            del self._d[:drop]
            self._d0 += drop

    def _take_output(self) -> bytes:
        if not self._out:
            return b""
        samples = np.clip(np.round(np.asarray(self._out, dtype=np.float64)), -32768, 32767)
        self._out = []
        return samples.astype("<i2").tobytes()


__all__ = [
    "ChannelSimulator",
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

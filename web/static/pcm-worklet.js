// T14：麦克风采集 AudioWorklet。
//
// 输入：AudioContext 的浮点采样（页面尽量申请 16kHz；拿不到时按实际采样率重采样）。
// 输出：16kHz、单声道、16-bit 小端 PCM，按 32ms（512 采样）一块 postMessage 给主线程，
// 主线程再经 WebSocket 发送（服务端接受任意帧长）。
//
// 说明：重采样用线性插值；当 AudioContext 已是 16kHz 时 ratio===1，等价于直接量化。
// 多声道输入先平均成单声道。静音/未连接输入时继续返回 true，保持 processor 存活。

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = Number(opts.targetRate) || 16000;
    // 每个输出采样对应的输入采样数。sampleRate 是 AudioWorkletGlobalScope 的全局变量。
    this.ratio = sampleRate / this.targetRate;
    this.frameSamples = Math.max(
      1,
      Math.round((this.targetRate * (Number(opts.frameMs) || 32)) / 1000)
    );
    this.pending = new Float32Array(0); // 尚未消费的输入采样
    this.pos = 0; // pending 内的读取位置（可为小数，也可暂时超出长度）
    this.frame = new Int16Array(this.frameSamples);
    this.frameCount = 0;
  }

  _append(mono) {
    // 丢掉已经消费的前缀；pos 超出旧样本的部分顺延到新样本，保持重采样相位。
    const drop = Math.floor(this.pos);
    if (drop >= this.pending.length) {
      this.pos -= this.pending.length;
      this.pending = new Float32Array(0);
    } else if (drop > 0) {
      this.pending = this.pending.slice(drop);
      this.pos -= drop;
    }
    const merged = new Float32Array(this.pending.length + mono.length);
    merged.set(this.pending, 0);
    merged.set(mono, this.pending.length);
    this.pending = merged;
  }

  _emit(sample) {
    const clamped = sample < -1 ? -1 : sample > 1 ? 1 : sample;
    this.frame[this.frameCount] = Math.round(clamped * 32767);
    this.frameCount += 1;
    if (this.frameCount === this.frameSamples) {
      // 拷一份再转移，避免下一帧复用同一块内存。
      const out = this.frame.slice(0);
      this.port.postMessage(out.buffer, [out.buffer]);
      this.frameCount = 0;
    }
  }

  _resample() {
    while (this.pos < this.pending.length) {
      const i0 = Math.floor(this.pos);
      const frac = this.pos - i0;
      const a = this.pending[i0];
      const b = i0 + 1 < this.pending.length ? this.pending[i0 + 1] : a;
      this._emit(a * (1 - frac) + b * frac);
      this.pos += this.ratio;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0] || input[0].length === 0) {
      return true;
    }
    const channels = input.length;
    const n = input[0].length;
    const mono = new Float32Array(n);
    for (let c = 0; c < channels; c += 1) {
      const ch = input[c];
      for (let i = 0; i < n; i += 1) mono[i] += ch[i] / channels;
    }
    this._append(mono);
    this._resample();
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);

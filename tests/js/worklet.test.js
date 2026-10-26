'use strict';

// T14：AudioWorklet 重采样单测。在 Node 里用一个假的 AudioWorkletGlobalScope 加载
// web/static/pcm-worklet.js，验证 16k 直通、48k→16k 降采样、多声道下混。

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadWorklet(sampleRate) {
  const code = fs.readFileSync(
    path.join(__dirname, '..', '..', 'web', 'static', 'pcm-worklet.js'),
    'utf8'
  );
  const messages = [];
  let Processor = null;
  const sandbox = {
    sampleRate,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { postMessage: (buffer) => messages.push(new Int16Array(buffer)) };
      }
    },
    registerProcessor: (name, cls) => {
      Processor = cls;
    },
    Float32Array,
    Int16Array,
    Math,
    Number,
  };
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox);
  return { Processor, messages };
}

test('16kHz 直通：512 采样一块，值正确量化', () => {
  const { Processor, messages } = loadWorklet(16000);
  const proc = new Processor({ processorOptions: { targetRate: 16000, frameMs: 32 } });
  proc.process([[new Float32Array(512).fill(0.5)]]);
  assert.equal(messages.length, 1);
  assert.equal(messages[0].length, 512);
  assert.equal(messages[0][0], Math.round(0.5 * 32767));
  assert.equal(messages[0][511], Math.round(0.5 * 32767));
});

test('超过 ±1 的采样被截断', () => {
  const { Processor, messages } = loadWorklet(16000);
  const proc = new Processor({ processorOptions: { targetRate: 16000, frameMs: 32 } });
  const input = new Float32Array(512);
  input[0] = 3.0;
  input[1] = -3.0;
  proc.process([[input]]);
  assert.equal(messages[0][0], 32767);
  assert.equal(messages[0][1], -32767);
});

test('48kHz 降到 16kHz：输出约 1/3，每块 512 采样', () => {
  const { Processor, messages } = loadWorklet(48000);
  const proc = new Processor({ processorOptions: { targetRate: 16000, frameMs: 32 } });
  for (let sent = 0; sent < 4800; sent += 128) {
    proc.process([[new Float32Array(128).fill(0.25)]]);
  }
  const total = messages.reduce((sum, frame) => sum + frame.length, 0);
  assert.ok(total >= 1536 && total <= 1600, `total=${total}`);
  for (const frame of messages) assert.equal(frame.length, 512);
});

test('多声道平均成单声道', () => {
  const { Processor, messages } = loadWorklet(16000);
  const proc = new Processor({ processorOptions: { targetRate: 16000, frameMs: 32 } });
  const left = new Float32Array(512).fill(1);
  const right = new Float32Array(512).fill(0);
  proc.process([[left, right]]);
  assert.equal(messages.length, 1);
  assert.equal(messages[0][0], Math.round(0.5 * 32767));
});

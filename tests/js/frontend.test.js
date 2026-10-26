'use strict';

// T14：前端纯逻辑单测（Node 自带 test runner，无第三方依赖）。
// 运行：node --test tests/js   （tests/test_frontend.py 会调它）

const test = require('node:test');
const assert = require('node:assert/strict');

const app = require('../../web/static/app.js');

test('percentile 线性插值', () => {
  assert.equal(app.percentile([], 0.5), null);
  assert.equal(app.percentile([10], 0.95), 10);
  assert.equal(app.percentile([0, 10], 0.5), 5);
  assert.equal(app.percentile([0, 100], 0.95), 95);
});

test('buildTurnRows 折出分段耗时并跳过问候语', () => {
  const events = [
    { type: 'play_start', t: 100, data: { text: '您好' } }, // 问候语，无 turn_end
    { type: 'vad_end', t: 1000, data: {} },
    { type: 'turn_end', t: 1200, data: { reason: 'smart_turn' } },
    { type: 'asr_request', t: 1210, data: {} },
    { type: 'asr_final', t: 1310, data: {} },
    { type: 'llm_request', t: 1320, data: {} },
    { type: 'llm_first_token', t: 1620, data: {} },
    { type: 'tts_request', t: 1630, data: {} },
    { type: 'tts_first_audio', t: 1930, data: {} },
    { type: 'play_start', t: 2000, data: { text: '好的' } },
  ];
  const rows = app.buildTurnRows(events);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].turnJudgeMs, 200);
  assert.equal(rows[0].asrMs, 100);
  assert.equal(rows[0].llmFirstMs, 300);
  assert.equal(rows[0].ttsFirstMs, 300);
  assert.equal(rows[0].totalMs, 1000);
  assert.equal(rows[0].turnReason, 'smart_turn');
  assert.equal(rows[0].text, '好的');
  assert.equal(rows[0].redo, 0);

  const stats = app.computeLatencyStats(rows);
  assert.equal(stats.count, 1);
  assert.equal(stats.totalP50, 1000);
  assert.equal(stats.totalP95, 1000);
});

test('buildTurnRows 把 play_start 前的 interrupted 计为重做并只保留最后一次尝试', () => {
  const events = [
    { type: 'vad_end', t: 0, data: {} },
    { type: 'turn_end', t: 100, data: { reason: 'smart_turn' } },
    { type: 'asr_request', t: 110, data: {} },
    { type: 'asr_final', t: 210, data: {} },
    { type: 'interrupted', t: 300, data: { reason: 'barge_in', stage: 'llm' } },
    { type: 'asr_request', t: 400, data: {} },
    { type: 'asr_final', t: 500, data: {} },
    { type: 'play_start', t: 600, data: { text: '重做' } },
  ];
  const rows = app.buildTurnRows(events);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].redo, 1);
  assert.equal(rows[0].asrMs, 100);
  assert.equal(rows[0].totalMs, 600);
});

test('applyTranscript 把同一句的 interrupted 合并，不新增一行', () => {
  const dialog = { caller: [], ai: [] };
  app.applyTranscript(dialog, { role: 'ai', text: '好的', interrupted: false });
  app.applyTranscript(dialog, { role: 'ai', text: '好的', interrupted: true });
  assert.equal(dialog.ai.length, 1);
  assert.equal(dialog.ai[0].interrupted, true);
  app.applyTranscript(dialog, { role: 'caller', text: '喂' });
  assert.equal(dialog.caller.length, 1);
  assert.equal(dialog.caller[0].interrupted, false);
});

test('applyTranscript 找不到原句时补一条被打断的行', () => {
  const dialog = { caller: [], ai: [] };
  app.applyTranscript(dialog, { role: 'ai', text: '只播了一半', interrupted: true });
  assert.equal(dialog.ai.length, 1);
  assert.equal(dialog.ai[0].interrupted, true);
});

test('slotAccuracy 单向包含比对', () => {
  const expected = { name: ['王', '王先生'], callback_number: '13700137000', time: null };
  const actual = {
    name: { value: '王强', status: 'unconfirmed' },
    callback_number: { value: '13700137000' },
  };
  const result = app.slotAccuracy(expected, actual);
  assert.equal(result.total, 2); // time 为 null，不计入
  assert.equal(result.passed, 2);
  assert.equal(result.rate, 1);

  const bad = app.slotAccuracy({ name: '李' }, actual);
  assert.equal(bad.passed, 0);
  assert.equal(bad.rate, 0);
});

test('toInt16Mono 下混多声道并重采样到 16k', () => {
  const fake = {
    sampleRate: 48000,
    numberOfChannels: 2,
    length: 6,
    getChannelData: (channel) =>
      channel === 0
        ? Float32Array.from([1, 0, 1, 0, 1, 0])
        : Float32Array.from([0, 1, 0, 1, 0, 1]),
  };
  const pcm = app.toInt16Mono(fake, 16000);
  assert.equal(pcm.length, 2); // 48000/16000 = 3
  assert.equal(pcm[0], Math.round(0.5 * 32767));
  assert.equal(pcm[1], Math.round(0.5 * 32767));
});

test('esc 转义 HTML 特殊字符', () => {
  assert.equal(app.esc('<b>"x" & y</b>'), '&lt;b&gt;&quot;x&quot; &amp; y&lt;/b&gt;');
});

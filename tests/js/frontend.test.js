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

// ---- T14b ----------------------------------------------------------------

test('endSilenceMs = stop_secs + 0.6s', () => {
  assert.equal(app.endSilenceMs(2), 2600);
  assert.equal(app.endSilenceMs(3), 3600);
  assert.equal(app.endSilenceMs(0), 600);
  assert.equal(app.endSilenceMs(undefined), 600);
});

test('buildTurnRows 一轮没播出就结束时 redo 不带进下一轮', () => {
  const events = [
    { type: 'vad_end', t: 0, data: {} },
    { type: 'turn_end', t: 100, data: { reason: 'smart_turn' } },
    { type: 'interrupted', t: 200, data: { reason: 'barge_in' } },
    { type: 'vad_end', t: 300, data: {} },
    { type: 'turn_end', t: 400, data: { reason: 'smart_turn' } },
    { type: 'asr_request', t: 410, data: {} },
    { type: 'asr_final', t: 500, data: {} },
    { type: 'play_start', t: 600, data: { text: '新一轮' } },
  ];
  const rows = app.buildTurnRows(events);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].redo, 0);
  assert.equal(rows[0].totalMs, 300);
});

test('applyTranscript 把同一轮没播出的句子标成 unplayed', () => {
  const dialog = { caller: [], ai: [] };
  app.applyTranscript(dialog, { role: 'ai', text: '第一句', interrupted: false });
  app.applyTranscript(dialog, { role: 'ai', text: '第二句', interrupted: false });
  app.applyTranscript(dialog, {
    role: 'ai',
    text: '第一句',
    interrupted: true,
    unplayed: ['第二句'],
  });
  assert.equal(dialog.ai.length, 2);
  assert.equal(dialog.ai[0].interrupted, true);
  assert.equal(dialog.ai[1].unplayed, true);
});

test('batchSummary 合计槽位准确率与总响应 P50/P95', () => {
  const results = [
    { status: '完成', passed: 2, total: 3, totals: [100, 200] },
    { status: '完成', passed: 1, total: 1, totals: [300] },
    { status: '失败', passed: 0, total: 0, totals: [] },
  ];
  const summary = app.batchSummary(results);
  assert.equal(summary.count, 2);
  assert.equal(summary.passed, 3);
  assert.equal(summary.total, 4);
  assert.equal(summary.rate, 0.75);
  assert.equal(summary.p50, 200);
  assert.equal(summary.p95, 290);
});

test('prepareInputSource 在发 start 前校验输入源并申请麦克风', async () => {
  const noop = async () => {};
  await assert.rejects(
    () => app.prepareInputSource('scenario', { scenarioId: () => '', file: () => null, startMic: noop }),
    /没有选择场景/
  );
  await assert.rejects(
    () => app.prepareInputSource('upload', { scenarioId: () => '', file: () => null, startMic: noop }),
    /没有选择 wav 文件/
  );
  let mic = false;
  const prepared = await app.prepareInputSource('mic', {
    scenarioId: () => '',
    file: () => null,
    startMic: async () => {
      mic = true;
    },
  });
  assert.equal(mic, true);
  assert.deepEqual(prepared, {});
  const scenario = await app.prepareInputSource('scenario', {
    scenarioId: () => 's03_appointment',
    file: () => null,
    startMic: noop,
  });
  assert.equal(scenario.scenarioId, 's03_appointment');
});

test('streamScenarioTurns 先等问候语，之后等 turn_done 和播完，按本轮 gap 开口，最后补结尾静音', async () => {
  const log = [];
  const turns = [
    { index: 0, gap_ms: 500 },
    { index: 1, gap_ms: 800 },
  ];
  let greetingResolve = null;
  let turnDoneResolve = null;
  const io = {
    cancelled: () => false,
    sendTurn: async (turn) => {
      log.push('turn' + turn.index);
    },
    startSilence: () => {
      log.push('silence:start');
      return () => log.push('silence:stop');
    },
    sendSilence: async (ms) => {
      log.push('silence:' + ms);
    },
    waitGreeting: () =>
      new Promise((resolve) => {
        log.push('waitGreeting');
        greetingResolve = resolve;
      }),
    waitTurnDone: () =>
      new Promise((resolve) => {
        log.push('waitDone');
        turnDoneResolve = resolve;
      }),
    waitPlaybackDrained: async () => {
      log.push('drained');
    },
    gapMs: (turn) => turn.gap_ms,
    endSilenceMs: () => 2600,
    sleep: async (ms) => {
      log.push('sleep:' + ms);
    },
  };
  const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
  const promise = app.streamScenarioTurns(turns, io);
  await tick();
  // 问候语没到之前不开口（不打断问候语）。
  assert.deepEqual(log, ['silence:start', 'waitGreeting']);
  greetingResolve();
  await tick();
  assert.deepEqual(log, [
    'silence:start',
    'waitGreeting',
    'drained',
    'sleep:500',
    'silence:stop',
    'turn0',
    'waitDone',
    'silence:start',
  ]);
  turnDoneResolve();
  await promise;
  assert.deepEqual(log.slice(8), ['drained', 'sleep:800', 'silence:stop', 'turn1', 'silence:2600']);
});

test('streamScenarioTurns 挂断后不再开口', async () => {
  const log = [];
  let cancelled = false;
  const io = {
    cancelled: () => cancelled,
    sendTurn: async (turn) => log.push('turn' + turn.index),
    startSilence: () => () => {},
    sendSilence: async (ms) => log.push('silence:' + ms),
    waitGreeting: async () => {
      cancelled = true; // 等问候语期间挂断
    },
    waitTurnDone: async () => {},
    waitPlaybackDrained: async () => {},
    gapMs: () => 0,
    endSilenceMs: () => 2600,
    sleep: async () => {},
  };
  await app.streamScenarioTurns([{ index: 0, gap_ms: 800 }], io);
  assert.deepEqual(log, []);
});

test('AudioPlayer.whenDrained 的超时按队列剩余时长算', async () => {
  const ctx = {
    currentTime: 0,
    destination: {},
    createBuffer: (channels, length, rate) => ({
      duration: length / rate,
      getChannelData: () => new Float32Array(length),
    }),
    createBufferSource: () => ({ connect() {}, start() {}, stop() {}, disconnect() {} }),
  };
  const player = new app.AudioPlayer(ctx);
  player.push(new Int16Array(16000 * 10).buffer); // 10 秒的确认语
  assert.equal(Math.round(player.remainingMs()), 10000);
  let seen = null;
  const realSetTimeout = global.setTimeout;
  global.setTimeout = (fn, ms) => {
    seen = ms;
    return realSetTimeout(fn, 0);
  };
  try {
    await player.whenDrained((remaining) => remaining + 3000);
  } finally {
    global.setTimeout = realSetTimeout;
  }
  assert.equal(Math.round(seen), 13000);
});

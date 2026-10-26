'use strict';

/*
 * T14：agent-telephone 测试页面前端（原生 JS，无框架）。
 *
 * - 麦克风：AudioWorklet（pcm-worklet.js）采集并转成 16kHz/单声道/16-bit PCM，经 WS 发送；
 * - 播放：AudioContext 缓冲队列，收到 interrupted / takeover 立即清空；
 * - 显示：全部来自后端转发的同一份事件流（见 web/server.py 的协议说明）；
 * - 纯函数（percentile / buildTurnRows / ...）放在文件上半部，Node 里可直接单测。
 */

/* ===================================================================== *
 * 纯函数（不依赖 DOM，供 tests 使用）
 * ===================================================================== */

const SLOT_NAMES = ['name', 'purpose', 'callback_number', 'time', 'location'];

function percentile(values, p) {
  const data = (values || []).filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (data.length === 0) return null;
  const ordered = data.slice().sort((a, b) => a - b);
  if (ordered.length === 1) return ordered[0];
  const rank = (ordered.length - 1) * p;
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) return ordered[low];
  return ordered[low] + (ordered[high] - ordered[low]) * (rank - low);
}

function _span(marks, start, end) {
  if (!(start in marks) || !(end in marks)) return null;
  if (marks[end] < marks[start]) return null;
  return marks[end] - marks[start];
}

/**
 * 把事件流折成「每次回复」的分段耗时。
 * 与 tools/turn_eval.py 的 latency_breakdown 同一套口径：
 *   - 轮次等待 = vad_end → turn_end（静音确认 + Smart Turn + 兜底）；
 *   - ASR = asr_request → asr_final；
 *   - LLM 首 token = llm_request → llm_first_token；
 *   - TTS 首包 = tts_request → tts_first_audio；
 *   - 总响应 = vad_end → 该轮第一个 play_start；
 *   - play_start 之前的 interrupted（抢话/轮次合并）算一次重做，丢弃该次数据。
 * 开头的问候语（没有 turn_end）不产生行。
 */
function buildTurnRows(events) {
  const rows = [];
  let vadEnd = null;
  let turnEnd = null;
  let turnReason = '';
  let marks = {};
  let redo = 0;
  const reset = () => {
    marks = {};
  };
  for (const event of events || []) {
    const kind = event.type;
    const t = Number(event.t) || 0;
    const data = event.data || {};
    if (kind === 'vad_end') {
      vadEnd = t;
    } else if (kind === 'turn_end') {
      turnEnd = t;
      turnReason = String(data.reason || '');
      reset();
    } else if (
      kind === 'asr_request' ||
      kind === 'asr_final' ||
      kind === 'llm_request' ||
      kind === 'llm_first_token' ||
      kind === 'llm_done'
    ) {
      if (!(kind in marks)) marks[kind] = t;
    } else if (kind === 'tts_request') {
      if ('llm_request' in marks && !('tts_request' in marks)) marks.tts_request = t;
    } else if (kind === 'tts_first_audio') {
      if ('tts_request' in marks && !('tts_first_audio' in marks)) marks.tts_first_audio = t;
    } else if (kind === 'interrupted') {
      if (turnEnd !== null) {
        redo += 1;
        reset();
      }
    } else if (kind === 'play_start') {
      if (turnEnd === null) continue; // 问候语
      rows.push({
        turnJudgeMs: vadEnd === null ? null : turnEnd - vadEnd,
        turnReason,
        asrMs: _span(marks, 'asr_request', 'asr_final'),
        llmFirstMs: _span(marks, 'llm_request', 'llm_first_token'),
        llmMs: _span(marks, 'llm_request', 'llm_done'),
        ttsFirstMs: _span(marks, 'tts_request', 'tts_first_audio'),
        totalMs: vadEnd === null ? null : t - vadEnd,
        redo,
        text: String(data.text || ''),
      });
      turnEnd = null;
      vadEnd = null;
      redo = 0;
      reset();
    }
  }
  return rows;
}

function computeLatencyStats(rows) {
  const list = rows || [];
  const pick = (key) => list.map((row) => row[key]);
  return {
    count: list.length,
    totalP50: percentile(pick('totalMs'), 0.5),
    totalP95: percentile(pick('totalMs'), 0.95),
    turnJudgeP50: percentile(pick('turnJudgeMs'), 0.5),
    asrP50: percentile(pick('asrMs'), 0.5),
    llmFirstP50: percentile(pick('llmFirstMs'), 0.5),
    ttsFirstP50: percentile(pick('ttsFirstMs'), 0.5),
  };
}

/**
 * 把服务端的 transcript 消息合并进对话。AI 的「被打断」消息与前面同一句合并，
 * 不新增一行（否则会看到同一句出现两次）。
 */
function applyTranscript(dialog, message) {
  const role = message.role === 'caller' ? 'caller' : 'ai';
  const text = String(message.text || '');
  const lines = dialog[role];
  if (message.interrupted) {
    for (let i = lines.length - 1; i >= 0; i -= 1) {
      if (lines[i].text === text) {
        lines[i].interrupted = true;
        return dialog;
      }
    }
    lines.push({ text, interrupted: true, pending: false });
    return dialog;
  }
  lines.push({ text, interrupted: false, pending: false });
  return dialog;
}

function slotMatches(expected, actual) {
  if (expected === null || expected === undefined) return true;
  const list = Array.isArray(expected) ? expected : [expected];
  if (actual === null || actual === undefined || actual === '') return false;
  const value = String(actual);
  return list.some((item) => value.includes(String(item)));
}

/** 单向包含比对（与 tools/evaluate.py 一致）：期望写法必须出现在实际值里。 */
function slotAccuracy(expectedSlots, actualSlots) {
  const details = [];
  for (const name of Object.keys(expectedSlots || {})) {
    const expected = expectedSlots[name];
    const empty =
      expected === null ||
      expected === undefined ||
      expected === '' ||
      (Array.isArray(expected) && expected.length === 0);
    if (empty) continue;
    const raw = (actualSlots || {})[name];
    const actual = raw && typeof raw === 'object' ? raw.value : raw;
    details.push({ name, expected, actual, ok: slotMatches(expected, actual) });
  }
  const passed = details.filter((item) => item.ok).length;
  return {
    total: details.length,
    passed,
    rate: details.length ? passed / details.length : null,
    details,
  };
}

/** AudioBuffer（或同形状对象）→ 16kHz 单声道 Int16。 */
function toInt16Mono(audioBuffer, targetRate) {
  const srcRate = Number(audioBuffer.sampleRate) || targetRate;
  const channels = Number(audioBuffer.numberOfChannels) || 1;
  const length = Number(audioBuffer.length) || 0;
  const mono = new Float32Array(length);
  for (let c = 0; c < channels; c += 1) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < length; i += 1) mono[i] += data[i] / channels;
  }
  let samples = mono;
  if (srcRate !== targetRate && targetRate > 0) {
    const ratio = srcRate / targetRate;
    const outLen = Math.max(0, Math.floor(length / ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i += 1) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      const a = mono[i0];
      const b = i0 + 1 < length ? mono[i0 + 1] : a;
      out[i] = a * (1 - frac) + b * frac;
    }
    samples = out;
  }
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = samples[i] < -1 ? -1 : samples[i] > 1 ? 1 : samples[i];
    pcm[i] = Math.round(s * 32767);
  }
  return pcm;
}

/* ===================================================================== *
 * 浏览器状态与 DOM 辅助
 * ===================================================================== */

const EVENT_TYPES = new Set([
  'call_start',
  'vad_start',
  'vad_end',
  'turn_end',
  'smart_turn',
  'asr_request',
  'asr_final',
  'llm_request',
  'llm_first_token',
  'llm_done',
  'tts_request',
  'tts_first_audio',
  'play_start',
  'play_end',
  'interrupted',
  'state_change',
  'slot_update',
  'takeover',
  'error',
  'call_end',
]);

const AI_STATUS = {
  vad_start: '听到声音',
  vad_end: '轮次确认中',
  turn_end: '处理中',
  asr_request: '识别中',
  asr_final: '识别完成',
  llm_request: '思考中',
  llm_first_token: '生成中',
  llm_done: '生成完成',
  tts_request: '合成中',
  tts_first_audio: '合成中',
  play_start: '说话中',
  play_end: '听',
  interrupted: '被打断',
  takeover: '已接管',
  error: '出错',
};

const app = {
  ws: null,
  callId: null,
  active: false,
  source: 'mic',
  audioCtx: null,
  player: null,
  mic: null,
  streamCancel: false,
  dialog: { caller: [], ai: [] },
  events: [],
  slots: {},
  state: null,
  scenarioList: [],
  calls: [],
  config: null,
  batchRunning: false,
  _connectPromise: null,
};

function $(id) {
  return document.getElementById(id);
}

function esc(value) {
  return String(value === null || value === undefined ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function fmt(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
  return String(Math.round(Number(value)));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setConnStatus(text, bad) {
  const node = $('conn-status');
  if (!node) return;
  node.textContent = text;
  node.classList.toggle('bad', !!bad);
}

function setAiStatus(text) {
  const node = $('ai-status');
  if (node) node.textContent = text;
}

function showError(message) {
  const node = $('error-log');
  const text = message && message.message ? message.message : String(message);
  if (node) node.textContent = text;
  setAiStatus('出错：' + text.slice(0, 60));
}

/* ===================================================================== *
 * 音频播放（AI 音频缓冲队列）
 * ===================================================================== */

class AudioPlayer {
  constructor(ctx) {
    this.ctx = ctx;
    this.sources = new Set();
    this.nextTime = 0;
  }

  push(buffer) {
    const samples = new Int16Array(buffer);
    if (samples.length === 0) return;
    const audioBuffer = this.ctx.createBuffer(1, samples.length, 16000);
    const data = audioBuffer.getChannelData(0);
    for (let i = 0; i < samples.length; i += 1) data[i] = samples[i] / 32768;
    const source = this.ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.ctx.destination);
    const start = Math.max(this.ctx.currentTime, this.nextTime);
    source.start(start);
    this.nextTime = start + audioBuffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  /** 抢话 / 接管：立即停掉待播与正在播的音频。 */
  clear() {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch (err) {
        /* 已经播完 */
      }
      try {
        source.disconnect();
      } catch (err) {
        /* ignore */
      }
    }
    this.sources.clear();
    this.nextTime = 0;
  }
}

function ensureAudioContext() {
  if (app.audioCtx && app.audioCtx.state !== 'closed') return app.audioCtx;
  let ctx;
  try {
    ctx = new AudioContext({ sampleRate: 16000 });
  } catch (err) {
    ctx = new AudioContext();
  }
  app.audioCtx = ctx;
  app.player = new AudioPlayer(ctx);
  return ctx;
}

/* ===================================================================== *
 * WebSocket
 * ===================================================================== */

function connect() {
  if (app.ws && app.ws.readyState === WebSocket.OPEN) return Promise.resolve();
  if (app._connectPromise) return app._connectPromise;
  app._connectPromise = new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(proto + '//' + location.host + '/ws/call');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      app._connectPromise = null;
      setConnStatus('已连接', false);
      resolve();
    };
    ws.onerror = () => {
      app._connectPromise = null;
      setConnStatus('连接错误', true);
      reject(new Error('WebSocket 连接失败'));
    };
    ws.onclose = () => {
      app._connectPromise = null;
      setConnStatus('已断开', true);
      if (app.active) {
        app.active = false;
        stopAudioInput();
        updateButtons();
      }
    };
    ws.onmessage = (event) => handleMessage(event.data);
    app.ws = ws;
  });
  return app._connectPromise;
}

function handleMessage(data) {
  if (typeof data !== 'string') {
    if (app.player) app.player.push(data);
    return;
  }
  let message;
  try {
    message = JSON.parse(data);
  } catch (err) {
    return;
  }
  const type = message.type;
  if (type === 'call_started') {
    app.callId = message.call_id;
    const node = $('call-id');
    if (node) node.textContent = message.call_id || '-';
    return;
  }
  if (type === 'transcript') {
    applyTranscript(app.dialog, message);
    renderDialog();
    return;
  }
  if (type === 'slots') {
    app.slots = message.slots || {};
    renderSlots();
    return;
  }
  if (type === 'state') {
    app.state = message.state;
    renderState();
    return;
  }
  if (type === 'call_ended') {
    onCallEnded();
    return;
  }
  if (type === 'turn_done') {
    return;
  }
  if (type === 'config') {
    if (message.config) app.config = message.config;
    return;
  }
  if (type === 'error' && typeof message.t !== 'number') {
    showError(message.stage ? message.stage + ': ' + (message.message || '') : message.message || 'error');
    return;
  }
  if (EVENT_TYPES.has(type) && typeof message.t === 'number') {
    onEvent(message);
  }
}

function onEvent(message) {
  app.events.push(message);
  if (AI_STATUS[message.type]) setAiStatus(AI_STATUS[message.type]);
  if (message.type === 'interrupted' || message.type === 'takeover') {
    if (app.player) app.player.clear();
  }
  if (message.type === 'play_start' || message.type === 'turn_end') renderLatency();
  if (message.type === 'call_end') {
    renderLatency();
    loadCalls();
  }
}

/* ===================================================================== *
 * 通话控制
 * ===================================================================== */

async function startCall() {
  if (app.active) return;
  try {
    const ctx = ensureAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();
    await connect();
    resetCallState();
    app.streamCancel = false;
    app.active = true;
    updateButtons();
    app.ws.send(JSON.stringify({ type: 'start' }));

    const source = currentSource();
    if (source === 'mic') {
      await startMic();
    } else if (source === 'scenario') {
      const scenarioId = $('scenario-select').value;
      if (!scenarioId) throw new Error('没有选择场景');
      streamScenario(scenarioId).catch(showError);
    } else if (source === 'upload') {
      const file = $('file-input').files[0];
      if (!file) throw new Error('没有选择 wav 文件');
      streamUpload(file).catch(showError);
    }
  } catch (err) {
    showError(err);
    app.active = false;
    updateButtons();
  }
}

function hangup() {
  if (!app.active) return;
  app.active = false;
  app.streamCancel = true;
  stopAudioInput();
  if (app.player) app.player.clear();
  if (app.ws && app.ws.readyState === WebSocket.OPEN) {
    app.ws.send(JSON.stringify({ type: 'hangup' }));
  }
  updateButtons();
}

function takeover() {
  if (!app.active) return;
  if (app.player) app.player.clear();
  if (app.ws && app.ws.readyState === WebSocket.OPEN) {
    app.ws.send(JSON.stringify({ type: 'takeover' }));
  }
}

function onCallEnded() {
  app.active = false;
  app.streamCancel = true;
  stopAudioInput();
  if (app.player) app.player.clear();
  updateButtons();
  renderLatency();
  loadCalls();
}

function updateButtons() {
  const start = $('btn-start');
  const hang = $('btn-hangup');
  const take = $('btn-takeover');
  if (start) start.disabled = app.active;
  if (hang) hang.disabled = !app.active;
  if (take) take.disabled = !app.active;
}

function resetCallState() {
  app.dialog = { caller: [], ai: [] };
  app.events = [];
  app.slots = {};
  app.state = null;
  app.callId = null;
  const node = $('call-id');
  if (node) node.textContent = '-';
  const err = $('error-log');
  if (err) err.textContent = '';
  renderDialog();
  renderSlots();
  renderState();
  renderLatency();
}

/* ===================================================================== *
 * 输入源：麦克风 / 场景 / 上传
 * ===================================================================== */

function currentSource() {
  const checked = document.querySelector('input[name="source"]:checked');
  return checked ? checked.value : 'mic';
}

async function startMic() {
  const echo = $('echo-cancel') ? $('echo-cancel').checked : true;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: echo,
      noiseSuppression: echo,
      autoGainControl: echo,
      channelCount: 1,
    },
  });
  const ctx = ensureAudioContext();
  await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
  const source = ctx.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(ctx, 'pcm-capture', {
    processorOptions: { targetRate: 16000, frameMs: 32 },
  });
  worklet.port.onmessage = (event) => {
    if (app.active && app.source === 'mic' && app.ws && app.ws.readyState === WebSocket.OPEN) {
      app.ws.send(event.data);
    }
  };
  // 必须接到 destination 才会被驱动；增益设 0，避免把麦克风原声放出来。
  const gain = ctx.createGain();
  gain.gain.value = 0;
  source.connect(worklet);
  worklet.connect(gain);
  gain.connect(ctx.destination);
  app.mic = { stream, source, worklet, gain };
}

function stopAudioInput() {
  if (app.mic) {
    const { stream, source, worklet, gain } = app.mic;
    try {
      worklet.port.onmessage = null;
      worklet.disconnect();
      source.disconnect();
      gain.disconnect();
    } catch (err) {
      /* ignore */
    }
    for (const track of stream.getTracks()) track.stop();
    app.mic = null;
  }
}

function sendPcmBuffer(buffer) {
  if (!app.active || !app.ws || app.ws.readyState !== WebSocket.OPEN) return false;
  app.ws.send(buffer);
  return true;
}

/** 按实时节奏（每 intervalMs 发 frameSamples 个采样）把 PCM 送给服务端。 */
function streamSamples(samples, frameSamples, intervalMs) {
  return new Promise((resolve) => {
    let offset = 0;
    const step = () => {
      if (app.streamCancel || !app.active) {
        resolve();
        return;
      }
      if (offset >= samples.length) {
        resolve();
        return;
      }
      const end = Math.min(samples.length, offset + frameSamples);
      const chunk = samples.subarray(offset, end);
      sendPcmBuffer(chunk.slice().buffer);
      offset = end;
      if (offset >= samples.length) {
        resolve();
        return;
      }
      setTimeout(step, intervalMs);
    };
    step();
  });
}

async function streamScenario(scenarioId) {
  const scenario = app.scenarioList.find((item) => item.id === scenarioId);
  if (!scenario) throw new Error('未知场景：' + scenarioId);
  const ctx = ensureAudioContext();
  for (const turn of scenario.turns || []) {
    if (app.streamCancel || !app.active) return;
    const url =
      '/api/scenarios/' + encodeURIComponent(scenarioId) + '/turns/' + turn.index + '/audio';
    const response = await fetch(url);
    if (!response.ok) throw new Error('加载场景音频失败：' + url);
    const bytes = await response.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(bytes);
    const samples = toInt16Mono(audioBuffer, 16000);
    await streamSamples(samples, 512, 32);
    await sleep(Math.max(0, Number(turn.gap_ms) || 0));
  }
}

async function streamUpload(file) {
  const ctx = ensureAudioContext();
  const bytes = await file.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(bytes);
  const samples = toInt16Mono(audioBuffer, 16000);
  await streamSamples(samples, 512, 32);
}

/* ===================================================================== *
 * 渲染
 * ===================================================================== */

function renderDialog() {
  const callerNode = $('dialog-caller');
  const aiNode = $('dialog-ai');
  if (!callerNode || !aiNode) return;
  callerNode.innerHTML = app.dialog.caller
    .map((line) => '<div class="line">' + esc(line.text) + '</div>')
    .join('');
  aiNode.innerHTML = app.dialog.ai
    .map((line) => {
      const cls = line.interrupted ? 'line interrupted' : 'line';
      const tag = line.interrupted ? ' <span class="tag">[被打断]</span>' : '';
      return '<div class="' + cls + '">' + esc(line.text) + tag + '</div>';
    })
    .join('');
  callerNode.scrollTop = callerNode.scrollHeight;
  aiNode.scrollTop = aiNode.scrollHeight;
}

function renderSlots() {
  const tbody = $('slots-body');
  if (!tbody) return;
  if (!app.slots || Object.keys(app.slots).length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">尚未通话</td></tr>';
    return;
  }
  tbody.innerHTML = SLOT_NAMES.map((name) => {
    const item = app.slots[name] || {};
    const value = item.value === null || item.value === undefined ? '' : String(item.value);
    const status = String(item.status || '');
    const cls =
      status === 'confirmed'
        ? 'tag-confirmed'
        : status === 'corrected'
          ? 'tag-corrected'
          : 'tag-unconfirmed';
    return (
      '<tr><td>' +
      esc(name) +
      '</td><td>' +
      esc(value) +
      '</td><td class="' +
      cls +
      '">' +
      esc(status) +
      '</td></tr>'
    );
  }).join('');
}

function renderState() {
  const node = $('dialog-state');
  if (node) node.textContent = app.state || '-';
}

function renderLatency() {
  const tbody = $('latency-body');
  const rows = buildTurnRows(app.events);
  if (tbody) {
    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="muted">尚未通话</td></tr>';
    } else {
      tbody.innerHTML = rows
        .map(
          (row, index) =>
            '<tr><td>' +
            (index + 1) +
            '</td><td>' +
            fmt(row.turnJudgeMs) +
            '</td><td>' +
            esc(row.turnReason) +
            '</td><td>' +
            fmt(row.asrMs) +
            '</td><td>' +
            fmt(row.llmFirstMs) +
            '</td><td>' +
            fmt(row.ttsFirstMs) +
            '</td><td>' +
            fmt(row.totalMs) +
            '</td><td>' +
            row.redo +
            '</td><td>' +
            esc(row.text) +
            '</td></tr>'
        )
        .join('');
    }
  }
  const stats = computeLatencyStats(rows);
  const summary = $('latency-summary');
  if (summary) {
    summary.textContent = stats.count
      ? '本通 ' +
        stats.count +
        ' 轮：总响应 P50 ' +
        fmt(stats.totalP50) +
        'ms / P95 ' +
        fmt(stats.totalP95) +
        'ms；轮次等待 P50 ' +
        fmt(stats.turnJudgeP50) +
        'ms；ASR P50 ' +
        fmt(stats.asrP50) +
        'ms；LLM 首 token P50 ' +
        fmt(stats.llmFirstP50) +
        'ms；TTS 首包 P50 ' +
        fmt(stats.ttsFirstP50) +
        'ms'
      : '本通：-';
  }
}

/* ===================================================================== *
 * 参数
 * ===================================================================== */

async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    const config = await response.json();
    app.config = config;
    if ($('param-channel-sim')) $('param-channel-sim').checked = !!config.channel_sim;
    if ($('param-asr')) $('param-asr').value = (config.models && config.models.asr) || '';
    if ($('param-llm')) $('param-llm').value = (config.models && config.models.llm) || '';
    if ($('param-tts-backend')) {
      $('param-tts-backend').value = (config.models && config.models.tts_backend) || '';
    }
    if ($('param-interrupt')) {
      $('param-interrupt').value =
        config.interrupt_min_speech_ms === null || config.interrupt_min_speech_ms === undefined
          ? ''
          : config.interrupt_min_speech_ms;
    }
  } catch (err) {
    showError(err);
  }
}

async function applyConfig() {
  try {
    await connect();
    const interruptValue = $('param-interrupt').value;
    app.ws.send(
      JSON.stringify({
        type: 'config',
        channel_sim: $('param-channel-sim').checked,
        models: {
          asr: $('param-asr').value.trim() || null,
          llm: $('param-llm').value.trim() || null,
          tts_backend: $('param-tts-backend').value || null,
        },
        interrupt_min_speech_ms: interruptValue === '' ? null : Number(interruptValue),
      })
    );
    $('config-note').textContent = '参数已发送，下一通生效。';
  } catch (err) {
    showError(err);
  }
}

/* ===================================================================== *
 * 通话历史
 * ===================================================================== */

async function loadCalls() {
  try {
    const response = await fetch('/api/calls');
    const data = await response.json();
    app.calls = data.calls || [];
    renderCalls();
  } catch (err) {
    showError(err);
  }
}

function renderCalls() {
  const tbody = $('calls-list');
  const note = $('calls-note');
  if (note) note.textContent = app.calls.length + ' 通';
  if (!tbody) return;
  if (app.calls.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="muted">暂无</td></tr>';
    return;
  }
  tbody.innerHTML = app.calls
    .map((call) => {
      const audio = [call.has_audio && call.has_audio.caller ? '来电者' : '', call.has_audio && call.has_audio.ai ? 'AI' : '']
        .filter(Boolean)
        .join('+');
      return (
        '<tr><td><button data-call="' +
        esc(call.call_id) +
        '">' +
        esc(call.call_id) +
        '</button></td><td>' +
        (call.event_count || 0) +
        '</td><td>' +
        esc(call.state || '-') +
        '</td><td>' +
        (call.has_summary ? '有' : '-') +
        '</td><td>' +
        esc(audio || '-') +
        '</td><td>' +
        (call.takeover ? '是' : '-') +
        '</td></tr>'
      );
    })
    .join('');
  for (const button of tbody.querySelectorAll('button[data-call]')) {
    button.addEventListener('click', () => showCall(button.getAttribute('data-call')));
  }
}

async function showCall(callId) {
  const container = $('call-detail');
  if (!container) return;
  try {
    const response = await fetch('/api/calls/' + encodeURIComponent(callId));
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const detail = await response.json();
    container.innerHTML = renderCallDetail(detail);
  } catch (err) {
    container.innerHTML = '<p class="muted">加载失败：' + esc(err.message || err) + '</p>';
  }
}

function renderCallDetail(detail) {
  const callId = detail.call_id;
  const slots = (detail.slots_detail && detail.slots_detail.slots) || {};
  const slotRows = SLOT_NAMES.map((name) => {
    const item = slots[name] || {};
    return (
      '<tr><td>' +
      esc(name) +
      '</td><td>' +
      esc(item.value === undefined || item.value === null ? '' : item.value) +
      '</td><td>' +
      esc(item.status || '') +
      '</td></tr>'
    );
  }).join('');
  const events = detail.events || [];
  const eventRows = events
    .map(
      (event) =>
        '<tr><td>' +
        fmt(event.t) +
        '</td><td>' +
        esc(event.type) +
        '</td><td class="mono">' +
        esc(JSON.stringify(event.data || {})) +
        '</td></tr>'
    )
    .join('');
  const summary = detail.summary
    ? '<pre class="mono">' + esc(JSON.stringify(detail.summary, null, 2)) + '</pre>'
    : '<p class="muted">无摘要（可能未挂断或未生成）。</p>';
  const hasAudio = detail.has_audio || {};
  const audioNodes = [];
  if (hasAudio.caller) {
    audioNodes.push(
      '<audio controls src="/api/calls/' + encodeURIComponent(callId) + '/audio/caller"></audio>'
    );
  }
  if (hasAudio.ai) {
    audioNodes.push(
      '<audio controls src="/api/calls/' + encodeURIComponent(callId) + '/audio/ai"></audio>'
    );
  }
  return (
    '<h3 class="mono">' +
    esc(callId) +
    '</h3>' +
    '<div class="row">' +
    (audioNodes.length ? audioNodes.join('') : '<span class="muted">没有分轨录音</span>') +
    '</div>' +
    '<h3>槽位</h3><table><thead><tr><th>槽位</th><th>值</th><th>状态</th></tr></thead><tbody>' +
    slotRows +
    '</tbody></table>' +
    '<h3>摘要</h3>' +
    summary +
    '<h3>事件时间线（' +
    events.length +
    '）</h3><div class="scroll-box"><table><thead><tr><th>t(ms)</th><th>type</th><th>data</th></tr></thead><tbody>' +
    eventRows +
    '</tbody></table></div>'
  );
}

/* ===================================================================== *
 * 场景批跑
 * ===================================================================== */

async function loadScenarios() {
  try {
    const response = await fetch('/api/scenarios');
    const data = await response.json();
    app.scenarioList = data.scenarios || [];
    renderScenarioSelect();
    renderScenarioCheckboxes();
  } catch (err) {
    showError(err);
  }
}

function renderScenarioSelect() {
  const select = $('scenario-select');
  if (!select) return;
  const playable = app.scenarioList.filter((item) => item.playable);
  select.innerHTML = playable
    .map((item) => '<option value="' + esc(item.id) + '">' + esc(item.id) + '（' + esc(item.desc || '') + '）</option>')
    .join('');
}

function renderScenarioCheckboxes() {
  const box = $('scenarios-checkboxes');
  if (!box) return;
  box.innerHTML = app.scenarioList
    .map(
      (item) =>
        '<label><input type="checkbox" value="' +
        esc(item.id) +
        '"' +
        (item.playable ? '' : ' disabled') +
        ' /> ' +
        esc(item.id) +
        (item.playable ? '' : '（音频缺失）') +
        ' ' +
        esc(item.desc || '') +
        '</label>'
    )
    .join('');
}

async function waitForCall(callId, timeoutMs) {
  const limit = timeoutMs || 180000;
  const start = Date.now();
  while (Date.now() - start < limit) {
    const response = await fetch('/api/calls/' + encodeURIComponent(callId));
    if (response.ok) {
      const detail = await response.json();
      const events = detail.events || [];
      if (
        events.some((event) => event.type === 'call_end') ||
        events.some((event) => event.type === 'error' && event.data && event.data.stage === 'replay')
      ) {
        return detail;
      }
    }
    await sleep(500);
  }
  throw new Error('回放超时');
}

function summarizeBatch(scenarioId, detail) {
  const scenario = app.scenarioList.find((item) => item.id === scenarioId) || {};
  const actual = (detail.slots_detail && detail.slots_detail.slots) || {};
  const accuracy = slotAccuracy(scenario.expected_slots || {}, actual);
  const rows = buildTurnRows(detail.events || []);
  const stats = computeLatencyStats(rows);
  return {
    id: scenarioId,
    callId: detail.call_id,
    status: '完成',
    rate: accuracy.rate,
    p50: stats.totalP50,
    p95: stats.totalP95,
    turns: rows.length,
    note: accuracy.total
      ? accuracy.passed + '/' + accuracy.total
      : '无期望槽位',
  };
}

function renderBatch(results) {
  const tbody = $('batch-table');
  if (!tbody) return;
  if (!results || results.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">尚未跑批</td></tr>';
    return;
  }
  tbody.innerHTML = results
    .map(
      (item) =>
        '<tr><td>' +
        esc(item.id) +
        '</td><td class="mono">' +
        esc(item.callId || '-') +
        '</td><td>' +
        esc(item.status) +
        '</td><td>' +
        (item.rate === null || item.rate === undefined ? '-' : Math.round(item.rate * 100) + '%') +
        '</td><td>' +
        fmt(item.p50) +
        ' / ' +
        fmt(item.p95) +
        '</td><td>' +
        (item.turns || 0) +
        '</td><td>' +
        esc(item.note || '') +
        '</td></tr>'
    )
    .join('');
}

async function runBatch() {
  if (app.batchRunning) return;
  const checked = Array.from(
    document.querySelectorAll('#scenarios-checkboxes input:checked')
  ).map((input) => input.value);
  if (checked.length === 0) {
    $('batch-progress').textContent = '没有选择场景';
    return;
  }
  app.batchRunning = true;
  $('btn-run-batch').disabled = true;
  const results = [];
  for (const scenarioId of checked) {
    $('batch-progress').textContent = '跑 ' + scenarioId + '…';
    try {
      const response = await fetch('/api/replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenarioId }),
      });
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const payload = await response.json();
      const detail = await waitForCall(payload.call_id);
      results.push(summarizeBatch(scenarioId, detail));
    } catch (err) {
      results.push({
        id: scenarioId,
        callId: '-',
        status: '失败',
        rate: null,
        p50: null,
        p95: null,
        turns: 0,
        note: String((err && err.message) || err),
      });
    }
    renderBatch(results);
  }
  app.batchRunning = false;
  $('btn-run-batch').disabled = false;
  $('batch-progress').textContent = '完成（' + results.length + ' 条）';
}

/* ===================================================================== *
 * 初始化
 * ===================================================================== */

function updateSourceVisibility() {
  app.source = currentSource();
  const scenarioRow = $('scenario-source');
  const uploadRow = $('upload-source');
  if (scenarioRow) scenarioRow.style.display = app.source === 'scenario' ? '' : 'none';
  if (uploadRow) uploadRow.style.display = app.source === 'upload' ? '' : 'none';
}

function init() {
  updateSourceVisibility();
  for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener('change', updateSourceVisibility);
  }
  $('btn-start').addEventListener('click', startCall);
  $('btn-hangup').addEventListener('click', hangup);
  $('btn-takeover').addEventListener('click', takeover);
  $('btn-apply-config').addEventListener('click', applyConfig);
  $('btn-refresh-calls').addEventListener('click', loadCalls);
  $('btn-run-batch').addEventListener('click', runBatch);
  $('btn-select-all-scenarios').addEventListener('click', () => {
    for (const input of document.querySelectorAll('#scenarios-checkboxes input:not(:disabled)')) {
      input.checked = true;
    }
  });
  loadConfig();
  loadScenarios();
  loadCalls();
  updateButtons();
  connect().catch(() => {
    /* 打开页面时后端可能还没起来，开始通话时会重连 */
  });
}

if (typeof document !== 'undefined') {
  init();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    SLOT_NAMES,
    percentile,
    buildTurnRows,
    computeLatencyStats,
    applyTranscript,
    slotMatches,
    slotAccuracy,
    toInt16Mono,
    esc,
    fmt,
  };
}

'use strict';

// T16b：浏览器侧 LiveKit 音频模块的单测。
//
// 第 32 次评审第 4 条：`app.audioCtx` 在多次通话之间复用（`ensureAudioContext()` 只在关闭后
// 才新建），而 `disconnect()` 把 `playTrack` 置空。第二次 `publishCaller()` 会再次
// `addModule()`，worklet 里的 `registerProcessor('pcm-play')` 抛 NotSupportedError，
// 同一页面第二通场景回放 / 上传电话必失败。这里用假的 AudioContext 复现这一点。
//
// 运行：node --test tests/js   （tests/test_frontend.py 会调它）

const test = require('node:test');
const assert = require('node:assert/strict');

const { LiveKitAudio } = require('../../web/static/livekit-audio.js');

const ROOM_INFO = { url: 'ws://127.0.0.1:7880', token: 'fake-token' };

/** 假 AudioContext：第二次 `addModule()` 抛 NotSupportedError（与真实浏览器一致）。 */
function makeContext() {
  const state = { addModuleCalls: 0, revoked: 0 };
  const track = { stop() {} };
  const ctx = {
    state: 'running',
    audioWorklet: {
      async addModule() {
        state.addModuleCalls += 1;
        if (state.addModuleCalls > 1) {
          const error = new Error("Failed to execute 'addModule' on 'AudioWorklet'");
          error.name = 'NotSupportedError';
          throw error;
        }
      },
    },
    createMediaStreamDestination: () => ({ stream: { getAudioTracks: () => [track] } }),
  };
  return { ctx, state };
}

class FakeRoom {
  constructor() {
    this.handlers = {};
    this.published = [];
    this.unpublished = [];
    this.localParticipant = {
      publishTrack: async (track) => {
        this.published.push(track);
      },
      unpublishTrack: async (track) => {
        this.unpublished.push(track);
      },
    };
  }

  on(event, handler) {
    this.handlers[event] = handler;
  }

  async connect() {}

  async startAudio() {}

  async disconnect() {
    this.disconnected = true;
  }
}

function installGlobals(room) {
  global.LivekitClient = {
    Room: function Room() {
      return room;
    },
    RoomEvent: { TrackSubscribed: 'trackSubscribed', TrackUnsubscribed: 'trackUnsubscribed' },
    createLocalAudioTrack: async () => ({ stop() {} }),
  };
  global.AudioWorkletNode = class {
    constructor(ctx, name) {
      this.ctx = ctx;
      this.name = name;
      this.port = { postMessage() {} };
    }

    connect() {}
  };
}

test('同一 AudioContext 上第二通电话不再重复 addModule', async () => {
  const room = new FakeRoom();
  installGlobals(room);
  const { ctx, state } = makeContext();

  await LiveKitAudio.connect(ROOM_INFO);
  const firstPush = await LiveKitAudio.publishCaller(ctx);
  firstPush(new Int16Array(4));
  await LiveKitAudio.disconnect();

  // 第二通：`disconnect()` 已经把 playTrack 置空，这里会重新建 worklet 节点。
  await LiveKitAudio.connect(ROOM_INFO);
  const secondPush = await LiveKitAudio.publishCaller(ctx);
  secondPush(new Int16Array(4));

  assert.equal(state.addModuleCalls, 1, '同一个 ctx 只该 addModule 一次');
  assert.equal(room.published.length, 2, '两通电话各发布一条来电者音轨');
  await LiveKitAudio.disconnect();
});

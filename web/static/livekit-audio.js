'use strict';

/*
 * T16：浏览器侧的 LiveKit 音频（采集与播放）。
 *
 * 只有“音频怎么进出”按框架分两份；对话、槽位、时延、历史等面板仍然订阅同一个事件
 * WebSocket（`web/static/app.js`），所以面板代码两边通用。
 *
 * - AI 音频：`RoomEvent.TrackSubscribed` → `track.attach()` 挂一个 `<audio>` 播放；
 * - 麦克风：`createLocalAudioTrack()`（回声消除／降噪／自动增益与 T14 页面一致）；
 * - 场景回放／上传 wav：把 16kHz PCM 推进一个 AudioWorklet，实时输出到
 *   `MediaStreamAudioDestinationNode` 再发布成音轨，这样 T14 的场景回放循环
 *   （`streamScenarioTurns`）不用改。
 *
 * 依赖 `web/static/vendor/livekit-client.umd.js`（固定版本，见 docs/livekit-notes.md）。
 */

const LiveKitAudio = (() => {
  // 播放队列：把推入的 16kHz int16 帧按实时节奏写进输出。
  const PLAY_WORKLET = `
class PcmPlay extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.cur = null;
    this.pos = 0;
    this.port.onmessage = (event) => { this.queue.push(new Int16Array(event.data)); };
  }
  process(_inputs, outputs) {
    const out = outputs[0][0];
    let i = 0;
    while (i < out.length) {
      if (!this.cur) {
        if (this.queue.length === 0) break;
        this.cur = this.queue.shift();
        this.pos = 0;
      }
      const n = Math.min(out.length - i, this.cur.length - this.pos);
      for (let k = 0; k < n; k += 1) out[i + k] = this.cur[this.pos + k] / 32768;
      this.pos += n;
      i += n;
      if (this.pos >= this.cur.length) this.cur = null;
    }
    return true;
  }
}
registerProcessor('pcm-play', PcmPlay);
`;

  let room = null;
  let info = null;
  let playNode = null;
  let playTrack = null;
  let micTrack = null;

  function lib() {
    if (typeof LivekitClient === 'undefined') {
      throw new Error('livekit-client 未加载（web/static/vendor/livekit-client.umd.js）');
    }
    return LivekitClient;
  }

  /** 连房间并播放 AI 音轨；`audioInfo` 是 `call_started` 消息里的 `livekit` 字段。 */
  async function connect(audioInfo, { ctx } = {}) {
    const { Room, RoomEvent } = lib();
    if (!audioInfo || !audioInfo.url || !audioInfo.token) {
      throw new Error('缺少 LiveKit 房间信息');
    }
    await disconnect();
    info = audioInfo;
    room = new Room({ adaptiveStream: false, dynacast: false });

    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== 'audio') return;
      const element = track.attach();
      element.autoplay = true;
      element.setAttribute('data-livekit-audio', 'ai');
      document.body.appendChild(element);
    });
    room.on(RoomEvent.TrackUnsubscribed, (track) => {
      for (const element of track.detach()) element.remove();
    });

    await room.connect(audioInfo.url, audioInfo.token);
    // 浏览器自动播放策略：开始通话是用户手势，这里显式起播。
    try {
      await room.startAudio();
    } catch (err) {
      /* 已经能播时 startAudio() 会抛，忽略 */
    }
    return room;
  }

  /** 发布麦克风；约束与 T14 页面的 `startMic()` 一致。 */
  async function publishMic({ echoCancellation = true } = {}) {
    const { createLocalAudioTrack } = lib();
    if (!room) throw new Error('还没有连接 LiveKit 房间');
    if (micTrack) return micTrack;
    micTrack = await createLocalAudioTrack({
      echoCancellation,
      noiseSuppression: echoCancellation,
      autoGainControl: echoCancellation,
      channelCount: 1,
    });
    await room.localParticipant.publishTrack(micTrack);
    return micTrack;
  }

  /** 发布一条“来电者”音轨，用于场景回放／上传 wav；返回推 PCM 的函数。 */
  async function publishCaller(ctx) {
    if (!room) throw new Error('还没有连接 LiveKit 房间');
    if (playTrack) {
      return (buffer) => {
        if (playNode) playNode.port.postMessage(buffer);
      };
    }
    const url = URL.createObjectURL(
      new Blob([PLAY_WORKLET], { type: 'application/javascript' }),
    );
    await ctx.audioWorklet.addModule(url);
    playNode = new AudioWorkletNode(ctx, 'pcm-play', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    const destination = ctx.createMediaStreamDestination();
    playNode.connect(destination);
    // 直接发布 WebAudio 出来的音轨（`publishTrack()` 接受 MediaStreamTrack）。
    playTrack = destination.stream.getAudioTracks()[0];
    await room.localParticipant.publishTrack(playTrack);
    return (buffer) => {
      if (playNode) playNode.port.postMessage(buffer);
    };
  }

  async function disconnect() {
    try {
      if (micTrack) {
        await room?.localParticipant.unpublishTrack(micTrack);
        micTrack.stop();
      }
    } catch (err) {
      /* 断开时忽略 */
    }
    try {
      if (playTrack) {
        await room?.localParticipant.unpublishTrack(playTrack);
        playTrack.stop();
      }
    } catch (err) {
      /* 断开时忽略 */
    }
    micTrack = null;
    playTrack = null;
    playNode = null;
    const current = room;
    room = null;
    if (current) {
      try {
        await current.disconnect();
      } catch (err) {
        /* 断开时忽略 */
      }
    }
  }

  return {
    connect,
    publishMic,
    publishCaller,
    disconnect,
    isConnected: () => Boolean(room),
    room: () => room,
    info: () => info,
  };
})();

if (typeof window !== 'undefined') window.LiveKitAudio = LiveKitAudio;
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LiveKitAudio };
}

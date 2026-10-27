"""T13：测试页面的后端（FastAPI + WebSocket）。

页面的全部显示都来自同一份事件流（设计文档第 10 节）：`EventLogger` 写
`calls/<call_id>/events.jsonl` 的同时，把同一条事件转发给 WebSocket；页面不需要
另一套埋点。历史页直接读 `events.jsonl` / `slots.json` / `summary.json`。

## 接口

- `GET /`：页面（`web/static/index.html`，T14 实现；这里只要求文件存在）。
- `GET /static/*`：页面的 `app.js` / `style.css` / `pcm-worklet.js`（T14）。
- `GET /api/scenarios`：列出 `scenarios/*.json`（T12 的场景集），含每轮的 `index` 与 `expected_slots`。
- `GET /api/scenarios/{id}/turns/{index}/audio`：取某轮 wav（T14 实时场景回放用）。
- `GET /api/calls`：列出 `calls/` 下的历史通话（状态、槽位、是否有摘要/录音）。
- `GET /api/calls/{id}`：一条通话的事件、槽位、摘要。
- `GET /api/calls/{id}/audio/{caller|ai}`：分轨 wav。
- `POST /api/replay`：用一条场景跑离线回放（后台任务，事件轮询 `GET /api/calls/{id}`）。
- `GET /api/config`：当前页面的参数（信道模拟、模型名、打断阈值）。
- `WS /ws/call`：实时通话。

## WebSocket 协议

浏览器 → 服务端：

- 文本 JSON 控制消息：
  - `{"type": "start"}`：开始一通电话（`source` 字段预留，T14 用）；
  - `{"type": "config", "channel_sim": bool, "models": {"asr": ..., "llm": ..., "tts_backend": "cloud|local"},
    "interrupt_min_speech_ms": number}`：改参数，**只对下一通生效**；
  - `{"type": "takeover"}`：用户接管，AI 立即静默；
  - `{"type": "hangup"}`：挂断，写 `slots.json` / 分轨 wav / `call_end`。
- 二进制帧：来电者音频，16kHz、单声道、16-bit 小端 PCM（任意帧长）。

服务端 → 浏览器：

- 二进制帧：AI 音频（同样的 16kHz PCM），按 TTS / 缓存产出的顺序推送；浏览器自行缓冲播放。
- 文本 JSON：`EventLogger` 的每条事件（`{"type": <事件类型>, "t": <monotonic ms>, "data": {...}}`），
  外加汇总消息 `transcript`（`role="caller"|"ai"`，被打断的句子带 `interrupted=true`）、
  `slots`、`state`；控制消息 `call_started` / `turn_done` / `call_ended` / `error`。

  `transcript`（T14b）：尾音抢话打断一句、而同一轮它后面还有已经送进浏览器队列、
  但从未开播的句子时，这条 AI 消息会多带 `"unplayed": ["句子", ...]`，页面据此把那些
  句子灰显（标 `[未播出]`），与 T13c 的摘要／历史一致。

  `turn_done`（T13b）在 `handle_turn()` 返回、即这一轮 AI 音频已全部送进浏览器队列后发出
  （`kind` 为 `reply` / `interrupted` / `merge` / `skipped` 等），供 T14 面板和 `ws_client.py`
  判断「这一轮回复结束」，再按真实节奏安排下一轮开口。

## 实时链路与离线回放的关系

每轮的 ASR → LLM → Dialog → 选话术 → TTS/缓存 全部复用 `core.call.CallSession`
（`RealtimeSession` 只覆写 `_emit_chunk()` 把音频推给浏览器），所以事件顺序、槽位更新、
模板兜底与 T10/T12 完全一致。

T13 新增、离线没有的部分：

1. **流式信道模拟**：逐帧用 `ChannelSimulator`（不能逐帧调 `simulate()`，见第 6 次评审）。
2. **轮次判断**：`TurnDetector` 逐帧喂音频；`vad_start` / `vad_end` / `turn_end` 事件带
   音频时间轴上的 `t_ms`；`turn_end` 时按 `t_ms` 从调用方缓冲里截出本轮音频（T09 评审）。
3. **抢话**：AI 在说（或本轮还在 ASR/LLM）时，来电者连续语音达到 `interrupt.min_speech_ms`
   就调用 `CallSession.interrupt()`：取消 LLM/TTS、记 `interrupted(reason="barge_in")`、
   把这一句按“未说完”收尾（`play_start → interrupted → play_end`），浏览器收到后清空待播队列。
4. **接管**：记 `takeover` 事件后中断当前轮；此后到挂断前不再切轮、不再说话。

已知限制（T14 已用浏览器自带的回声消除缓解）：外放时若不请求 `echoCancellation`，
浏览器会把 AI 自己的声音当来电者；`getUserMedia` 已默认打开回声消除／降噪／自动增益。
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.call import CallSession
from core.events import EventLogger, read_events
from core.playback import PlaybackSegment, PlaybackTracker
from core.turn import SPEECH_END, SPEECH_START, TURN_END, TurnDetector
from tools.channel_sim import ChannelSimulator
from tools.replay import (
    ROOT,
    ReplayServices,
    discover_scenarios,
    open_services,
    replay_scenario_to_dir,
    scenario_playable,
)

# 页面参数默认值（config.yaml 的 web / interrupt 段可覆盖）。
DEFAULT_CHANNEL_SIM = True
DEFAULT_INTERRUPT_MIN_SPEECH_MS = 300.0
DEFAULT_CALLS_DIR = "calls"
DEFAULT_SCENARIOS_DIR = "scenarios"
# 来电者音频缓冲：保留最近 12s 够 turn_end 时切片（最长的轮次 8s + 预卷 0.5s）。
CALLER_BUFFER_MS = 12_000.0
# 页面只允许读写 calls/ 下这一层目录，防止路径穿越。
CALL_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


# --------------------------------------------------------------------------- #
# 参数（只对下一通生效）
# --------------------------------------------------------------------------- #


@dataclass
class WebConfig:
    """页面参数。`config` 消息改这里的值，下一通电话构造服务/链路时读取。"""

    channel_sim: bool = DEFAULT_CHANNEL_SIM
    asr_model: str | None = None
    llm_model: str | None = None
    tts_backend: str | None = None
    interrupt_min_speech_ms: float = DEFAULT_INTERRUPT_MIN_SPEECH_MS
    # 只读：VAD 的轮次兜底时长（秒）。页面用它算场景回放结尾要补多少静音。
    stop_secs: float = 3.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "WebConfig":
        web_cfg = config.get("web") or {}
        interrupt_cfg = config.get("interrupt") or {}
        vad_cfg = config.get("vad") or {}
        return cls(
            channel_sim=bool(web_cfg.get("channel_sim", DEFAULT_CHANNEL_SIM)),
            asr_model=(config.get("asr") or {}).get("model"),
            llm_model=(config.get("llm") or {}).get("realtime_model"),
            tts_backend=(config.get("tts") or {}).get("backend"),
            interrupt_min_speech_ms=float(
                interrupt_cfg.get("min_speech_ms", DEFAULT_INTERRUPT_MIN_SPEECH_MS)
            ),
            stop_secs=float(vad_cfg.get("stop_secs", 3.0)),
        )

    def apply(self, payload: Mapping[str, Any]) -> None:
        if "channel_sim" in payload and payload["channel_sim"] is not None:
            self.channel_sim = bool(payload["channel_sim"])
        models = payload.get("models") or {}
        if models.get("asr"):
            self.asr_model = str(models["asr"])
        if models.get("llm"):
            self.llm_model = str(models["llm"])
        if models.get("tts_backend"):
            backend = str(models["tts_backend"]).strip().lower()
            if backend not in ("cloud", "local"):
                raise ValueError(f"unknown tts backend: {backend!r}")
            self.tts_backend = backend
        if payload.get("interrupt_min_speech_ms") is not None:
            self.interrupt_min_speech_ms = float(payload["interrupt_min_speech_ms"])

    def to_json(self) -> dict[str, Any]:
        return {
            "channel_sim": self.channel_sim,
            "models": {
                "asr": self.asr_model,
                "llm": self.llm_model,
                "tts_backend": self.tts_backend,
            },
            "interrupt_min_speech_ms": self.interrupt_min_speech_ms,
            "stop_secs": self.stop_secs,
        }


def apply_web_config(config: Mapping[str, Any], web: WebConfig) -> dict[str, Any]:
    """把页面参数合进一份 config 副本（不动原始 config.yaml 的内容）。"""
    merged = copy.deepcopy(dict(config))
    if web.asr_model:
        merged.setdefault("asr", {})["model"] = web.asr_model
    if web.llm_model:
        merged.setdefault("llm", {})["realtime_model"] = web.llm_model
    if web.tts_backend:
        merged.setdefault("tts", {})["backend"] = web.tts_backend
    return merged


# --------------------------------------------------------------------------- #
# 事件转发与 WebSocket 发送
# --------------------------------------------------------------------------- #


class ForwardingEventLogger(EventLogger):
    """写文件 + 实时转发（T13）。sink 收到的是刚写入的那条事件 dict。"""

    def __init__(self, call_dir: str | Path, sink) -> None:
        super().__init__(call_dir)
        self._sink = sink

    def log(self, type: str, **data: Any) -> dict[str, Any]:
        event = super().log(type, **data)
        try:
            self._sink(event)
        except Exception:  # noqa: BLE001 - 推送失败（页面已关）不能影响一通电话
            pass
        return event


class WsSender:
    """串行化的发送队列：事件与音频共用一个队列，保证到达顺序与产生顺序一致。"""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.closed = False

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def send_json(self, payload: Mapping[str, Any]) -> None:
        if not self.closed:
            self._queue.put_nowait(("json", dict(payload)))

    def send_bytes(self, pcm: bytes) -> None:
        if not self.closed and pcm:
            self._queue.put_nowait(("bytes", pcm))

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "bytes":
                    await self._ws.send_bytes(payload)
                else:
                    await self._ws.send_json(payload)
            except Exception:  # noqa: BLE001 - 页面断开后发送会抛错，直接停
                self.closed = True
                return

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 实时链路
# --------------------------------------------------------------------------- #


class RealtimeSession(CallSession):
    """`CallSession` + 把音频推给浏览器（T13）。

    只覆写 `_emit_chunk()`；ASR / LLM / Dialog / 话术 / 打断逻辑全部复用离线回放的实现，
    这样实时与离线的事件流、槽位更新完全一致（T12 的评估结果才能对照）。
    """

    def __init__(self, *args: Any, send_audio, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._send_audio = send_audio

    async def _emit_chunk(self, chunk: bytes, text: str) -> None:
        await self._send_audio(chunk, text)


class RealtimeCall:
    """一通实时电话：来电者音频 → 信道模拟 → VAD/轮次 → ASR → LLM → TTS → 浏览器。"""

    def __init__(
        self,
        *,
        call_dir: str | Path,
        config: Mapping[str, Any],
        services: ReplayServices,
        send_audio,
        send_json,
        detector: TurnDetector,
        channel_sim: bool,
        interrupt_min_speech_ms: float,
        sample_rate: int,
    ) -> None:
        self.call_dir = Path(call_dir)
        self.call_id = self.call_dir.name
        self.config = config
        self.services = services
        self.detector = detector
        self.sample_rate = int(sample_rate)
        self.channel_sim = bool(channel_sim)
        self.simulator = ChannelSimulator() if self.channel_sim else None
        self.min_speech_ms = float(interrupt_min_speech_ms)
        self._send_json = send_json
        self._send_bytes = send_audio

        self.events = ForwardingEventLogger(self.call_dir, self._on_event)
        self.session = RealtimeSession(
            services.asr,
            services.llm,
            services.tts,
            services.cache,
            self.events,
            config,
            sample_rate=self.sample_rate,
            send_audio=self._send_audio,
        )

        # 来电者音频缓冲（音频时间轴），turn_end 时按 t_ms 切片。
        self._buffer = bytearray()
        self._buffer_start_ms = 0.0
        self._audio_ms = 0.0
        self._turn_start_ms: float | None = None
        # AI 在浏览器里“还要播多久”：用于尾音期间的抢话判定（T13b，T15b 起与 Pipecat 共用）。
        self.playback = PlaybackTracker()
        self._ai_text = ""

        self._turn_tasks: set[asyncio.Task] = set()
        self._turn_lock = asyncio.Lock()
        # 上一轮还没开播就被抢话时，它的文本顺延到下一轮（与 T10 的轮次合并同义）。
        self._carried_text = ""
        self._busy = False
        self._barged_in = False

        self.takeover_mode = False
        self.finished = False

    # ---- 对外接口 ---------------------------------------------------------

    async def start(self) -> None:
        """记 call_start 并播放问候语（模板缓存，不调 LLM）。"""
        await self.session.start()

    async def feed(self, pcm: bytes) -> None:
        """处理一段来电者音频（16kHz、单声道、16-bit 小端）。

        T13b 起改为 async：Smart Turn 推理实测超过 20ms 时，`feed_async()` 会把整帧
        处理丢到线程里，避免阻塞事件循环。
        """
        if self.finished or self.takeover_mode:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return
        if self.simulator is not None:
            pcm = self.simulator.feed(pcm)
            if not pcm:
                return

        self._audio_ms += self._pcm_ms(pcm)
        self._buffer += pcm
        self._trim_buffer()

        for event in await self.detector.feed_async(pcm):
            self._on_turn_event(event)
        self._maybe_barge_in()

    def takeover(self) -> None:
        """用户接管：记 takeover 事件并立即中断当前轮，之后 AI 不再出声。"""
        if self.finished or self.takeover_mode:
            return
        self.takeover_mode = True
        self.playback.reset()
        self.events.log("takeover", turns=self.session.turn_count)
        self.session.interrupt("takeover")

    async def finish(self) -> None:
        """挂断：中断当前轮、等所有排队轮次收尾，写 call_end / slots.json / 分轨 wav。

        挂断时先 `interrupt("hangup")`：LLM/TTS 在下一个 chunk 就退出，用户不用等
        完整句合成完。排队中的轮次在拿到轮次锁后看到 `finished` 会直接跳过。
        """
        if self.finished:
            return
        self.finished = True
        self.playback.reset()
        self.session.interrupt("hangup")

        pending = [task for task in self._turn_tasks if not task.done()]
        if pending:
            _done, still_running = await asyncio.wait(pending, timeout=5.0)
            for task in still_running:
                task.cancel()
            if still_running:
                await asyncio.gather(*still_running, return_exceptions=True)

        if self.simulator is not None:
            # 流式模拟的尾部延迟样本（62 个）不必再送进 VAD，只是收尾释放状态。
            self.simulator.flush()
        self.session.finish()
        self.events.close()
        await self.services.aclose()

    # ---- 事件与汇总消息 ---------------------------------------------------

    def _on_event(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        self._send_json({"type": kind, "t": round(float(event.get("t") or 0.0), 3), "data": data})

        if kind == "asr_final" and data.get("text"):
            self._send_json({"type": "transcript", "role": "caller", "text": data["text"]})
        elif kind == "play_start":
            self._ai_text = str(data.get("text") or "")
            # T13b：新一句从前一句队列结束（或当前时刻）开始；具体时长由后续音频块累加。
            self.playback.on_play_start(self._ai_text)
            self._send_json(
                {"type": "transcript", "role": "ai", "text": self._ai_text, "interrupted": False}
            )
        elif kind == "interrupted":
            if data.get("tail"):
                # T13b 尾音抢话：event.text 就是当前正在播的那一句（可能排在前一句之后）。
                interrupted_text = str(data.get("text") or "")
                if interrupted_text:
                    payload: dict[str, Any] = {
                        "type": "transcript",
                        "role": "ai",
                        "text": interrupted_text,
                        "interrupted": True,
                    }
                    # T14b：同一轮里排在被抢话那句之后、已经送出但从未开播的句子。
                    unplayed = [str(item) for item in (data.get("unplayed") or []) if item]
                    if unplayed:
                        payload["unplayed"] = unplayed
                    self._send_json(payload)
            elif self._ai_text and data.get("text") == self._ai_text:
                # LLM 阶段的抢话（stage="llm"）没有播出，不生成 AI 逐句，
                # 否则会把来电者的文本标到 AI 头上。
                self._send_json(
                    {"type": "transcript", "role": "ai", "text": self._ai_text, "interrupted": True}
                )
        if kind in ("call_start", "state_change", "slot_update", "takeover"):
            self._send_snapshot()

    def _send_snapshot(self) -> None:
        dialog = getattr(self.session, "dialog", None)
        if dialog is None:
            return
        self._send_json({"type": "state", "state": dialog.state.value})
        self._send_json(
            {
                "type": "slots",
                "slots": {
                    name: {"value": value.value, "status": value.status}
                    for name, value in dialog.slots.items()
                },
            }
        )

    # ---- 轮次与打断 -------------------------------------------------------

    def _on_turn_event(self, event) -> None:
        if event.kind == SPEECH_START:
            if self._turn_start_ms is None:
                self._turn_start_ms = float(event.t_ms)
            self._barged_in = False
            self.events.log("vad_start", t_ms=event.t_ms, speech_ms=event.speech_ms)
        elif event.kind == SPEECH_END:
            self.events.log("vad_end", t_ms=event.t_ms, speech_ms=event.speech_ms)
            # T13b：记录 Smart Turn 单次推理耗时；超阈值后 TurnDetector 会改到线程里跑。
            if self.detector.last_smart_turn_ms is not None:
                self.events.log(
                    "smart_turn",
                    latency_ms=round(self.detector.last_smart_turn_ms, 2),
                    offload=self.detector.smart_turn_offload,
                )
        elif event.kind == TURN_END:
            self.events.log(
                "turn_end", t_ms=event.t_ms, reason=event.reason, speech_ms=event.speech_ms
            )
            self._schedule_turn(event)

    def _schedule_turn(self, event) -> None:
        speech_start = self._turn_start_ms if self._turn_start_ms is not None else event.t_ms
        start_ms = max(0.0, speech_start - self.detector.pre_speech_ms)
        end_ms = min(float(event.t_ms), self._audio_ms)
        self._turn_start_ms = None
        pcm = self._slice(start_ms, end_ms)
        if not pcm:
            return
        task = asyncio.create_task(self._run_turn(pcm, start_ms, end_ms))
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _run_turn(self, pcm: bytes, start_ms: float, end_ms: float) -> None:
        # 串行化：上一轮被抢话后正在收尾时，新一轮等它结束再改 Dialog 状态。
        async with self._turn_lock:
            if self.finished or self.takeover_mode:
                self._send_json({"type": "turn_done", "kind": "skipped"})
                return
            self._busy = True
            outcome = None
            try:
                outcome = await self.session.handle_turn(
                    pcm,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    next_start_ms=None,
                    carried_text=self._carried_text,
                )
                # 还没开播就被抢话：把这句话顺延给下一轮，不丢内容也不重复追问。
                self._carried_text = (
                    outcome.text if outcome.kind == "interrupted" and not outcome.played else ""
                )
            except Exception as exc:  # noqa: BLE001 - 单轮异常不该让整通电话挂掉
                self.events.log(
                    "error", stage="turn", reason=type(exc).__name__, message=str(exc)[:200]
                )
            finally:
                self._busy = False
                # T13b：这一轮的 AI 音频已经全部送进浏览器队列（`handle_turn` 返回即
                # 音频发送完毕）。告诉客户端「本轮回复结束」，它才能按真实节奏算下一轮
                # 的开口时刻（否则会在 AI 音频到达前就抢话）。
                self._send_json(
                    {
                        "type": "turn_done",
                        "kind": outcome.kind if outcome is not None else "error",
                        "text": outcome.text if outcome is not None else "",
                        "interrupted": bool(outcome is not None and outcome.kind == "interrupted"),
                    }
                )

    def _maybe_barge_in(self) -> None:
        """来电者累计语音达到阈值，且 AI 正在说话/正在生成 → 打断。

        用 `TurnDetector.turn_speech_ms`（只计 VAD 判为语音的帧）而不是距开口的时长，
        否则短噪 + 随后的静音也会被算成“说了 300ms”。`_busy` 覆盖 ASR/LLM/TTS 阶段，
        `_playback_until_ms` 覆盖轮次已结束、浏览器里还有尾音的情况。
        """
        if self._barged_in:
            return
        speaking = self._busy or self.playback.playing()
        if not speaking:
            return
        if self.detector.turn_speech_ms < self.min_speech_ms:
            return

        self._barged_in = True
        if not self._busy:
            # T13b：轮次已经结束，只剩浏览器里排队没播完的尾音。找出当前正在播的那一句，
            # 补记 interrupted（带 text 与已播时长），并把对话历史最后一条 assistant 标成
            # “（被对方打断）”。服务端早已记过 play_end，摘要需要靠这条事件才认得出来。
            text, played_ms, unplayed = self.playback.interrupt()
            text = text or (self._ai_text or "")
            self.events.log(
                "interrupted",
                reason="barge_in",
                text=text,
                played_ms=None if played_ms is None else round(played_ms, 2),
                tail=True,
                unplayed=unplayed,
            )
            self.session.mark_last_assistant_interrupted(text)
        self.playback.reset()
        self.session.interrupt("barge_in")

    def _current_playing(self) -> PlaybackSegment | None:
        """当前（或下一句待播）的播放区间；都已经播完则返回 None。"""
        return self.playback.current_playing()

    # ---- 音频缓冲 ---------------------------------------------------------

    async def _send_audio(self, pcm: bytes, text: str = "") -> None:
        # 音频一经送出就进入浏览器队列；按实时速率估出队列结束时刻，供尾音抢话判定。
        self.playback.on_audio(self._pcm_ms(pcm), text or self._ai_text)
        self._send_bytes(pcm)

    def _pcm_ms(self, pcm: bytes) -> float:
        return len(pcm) / (2 * self.sample_rate) * 1000.0

    def _trim_buffer(self) -> None:
        keep_bytes = int(CALLER_BUFFER_MS / 1000.0 * self.sample_rate) * 2
        if len(self._buffer) > keep_bytes:
            drop = len(self._buffer) - keep_bytes
            drop -= drop % 2
            del self._buffer[:drop]
            self._buffer_start_ms += drop / (2 * self.sample_rate) * 1000.0

    def _slice(self, start_ms: float, end_ms: float) -> bytes:
        def offset(ms: float) -> int:
            samples = int(round((ms - self._buffer_start_ms) * self.sample_rate / 1000.0))
            return max(0, min(len(self._buffer), samples * 2))

        start = offset(start_ms)
        end = offset(end_ms)
        if end <= start:
            return b""
        return bytes(self._buffer[start:end])


# --------------------------------------------------------------------------- #
# 应用状态与回放
# --------------------------------------------------------------------------- #


class AppState:
    """`create_app()` 的依赖：配置、目录、服务构造器（测试可注入 fake）。"""

    def __init__(
        self,
        *,
        root: str | Path = ROOT,
        config: Mapping[str, Any] | None = None,
        calls_dir: str | Path | None = None,
        scenarios_dir: str | Path | None = None,
        services_factory=None,
        detector_factory=None,
        replay_runner=None,
        adapter: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = dict(config if config is not None else load_config(self.root / "config.yaml"))
        web_cfg = self.config.get("web") or {}
        # 实时链路的实现：cascade（T13，手写 TurnDetector）或 pipecat（T15）。
        self.adapter = str(adapter or web_cfg.get("adapter") or "cascade").strip().lower()
        self.calls_dir = Path(
            calls_dir or self.root / str(web_cfg.get("calls_dir") or DEFAULT_CALLS_DIR)
        )
        self.scenarios_dir = Path(scenarios_dir or self.root / DEFAULT_SCENARIOS_DIR)
        self.static_dir = self.root / "web" / "static"
        self.services_factory = services_factory or open_services
        self.detector_factory = detector_factory or (lambda cfg: TurnDetector.from_config(cfg))
        self.replay_runner = replay_runner or run_scenario_replay
        self.web_config = WebConfig.from_config(self.config)
        self.replay_tasks: dict[str, asyncio.Task] = {}
        self.calls_dir.mkdir(parents=True, exist_ok=True)

    def sample_rate(self) -> int:
        return int((self.config.get("audio") or {}).get("sample_rate", 16000))

    def call_config(self) -> dict[str, Any]:
        return apply_web_config(self.config, self.web_config)

    async def start_call(self, sender: "WsSender"):
        config = self.call_config()
        call_id = new_call_id("web")
        call_dir = self.calls_dir / call_id
        services = await self.services_factory(config)
        try:
            if self.adapter == "livekit":
                from adapters.livekit.session import LiveKitCall
                from adapters.livekit.tokens import build_token, new_room_name

                lk = self.config.get("livekit") or {}
                room = new_room_name(str(lk.get("room_prefix") or "agent-telephone"), call_id)
                ttl = int(lk.get("token_ttl_s") or 3600)
                return LiveKitCall(
                    call_dir=call_dir,
                    config=config,
                    services=services,
                    send_audio=None,
                    send_json=sender.send_json,
                    channel_sim=self.web_config.channel_sim,
                    interrupt_min_speech_ms=self.web_config.interrupt_min_speech_ms,
                    sample_rate=self.sample_rate(),
                    detector=self.detector_factory(config),
                    room_url=str(lk.get("url") or "ws://127.0.0.1:7880"),
                    room_name=room,
                    room_token=build_token(room=room, identity="assistant", ttl_s=ttl),
                    browser_token=build_token(room=room, identity="caller", ttl_s=ttl),
                )
            if self.adapter == "pipecat":
                from adapters.pipecat.pipeline import PipecatCall

                return PipecatCall(
                    call_dir=call_dir,
                    config=config,
                    services=services,
                    send_audio=sender.send_bytes,
                    send_json=sender.send_json,
                    channel_sim=self.web_config.channel_sim,
                    interrupt_min_speech_ms=self.web_config.interrupt_min_speech_ms,
                    sample_rate=self.sample_rate(),
                )
            detector = self.detector_factory(config)
            return RealtimeCall(
                call_dir=call_dir,
                config=config,
                services=services,
                send_audio=sender.send_bytes,
                send_json=sender.send_json,
                detector=detector,
                channel_sim=self.web_config.channel_sim,
                interrupt_min_speech_ms=self.web_config.interrupt_min_speech_ms,
                sample_rate=self.sample_rate(),
            )
        except Exception:
            await services.aclose()
            raise

    def call_dir(self, call_id: str) -> Path:
        if not CALL_ID_RE.match(call_id) or call_id in (".", ".."):
            raise HTTPException(status_code=404, detail="unknown call")
        return self.calls_dir / call_id


def new_call_id(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


async def run_scenario_replay(state: AppState, scenario, out_dir: Path) -> None:
    """默认的 `/api/replay` 执行体：用真实服务把场景跑完并写到 out_dir。"""
    config = state.call_config()
    services = await state.services_factory(config)
    try:
        await replay_scenario_to_dir(scenario, services, config, out_dir)
    finally:
        await services.aclose()


async def _run_replay_safely(state: AppState, scenario, out_dir: Path) -> None:
    try:
        await state.replay_runner(state, scenario, out_dir)
    except Exception as exc:  # noqa: BLE001 - 回放失败要能在历史页看到原因
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = EventLogger(out_dir)
        try:
            logger.log(
                "error", stage="replay", reason=type(exc).__name__, message=str(exc)[:200]
            )
        finally:
            logger.close()


# --------------------------------------------------------------------------- #
# HTTP / WebSocket 路由
# --------------------------------------------------------------------------- #


def create_app(
    *,
    root: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    calls_dir: str | Path | None = None,
    scenarios_dir: str | Path | None = None,
    services_factory=None,
    detector_factory=None,
    replay_runner=None,
    adapter: str | None = None,
) -> FastAPI:
    """构造应用。测试通过参数注入 fake 服务、临时目录，不联网。"""
    state = AppState(
        root=root or ROOT,
        config=config,
        calls_dir=calls_dir,
        scenarios_dir=scenarios_dir,
        services_factory=services_factory,
        detector_factory=detector_factory,
        replay_runner=replay_runner,
        adapter=adapter,
    )
    app = FastAPI(title="agent-telephone 测试页面后端（T13）")
    app.state.web = state

    # -- 静态资源（T14：app.js / style.css / pcm-worklet.js） ---------------- #

    state.static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(state.static_dir)), name="static")

    # -- 页面 --------------------------------------------------------------- #

    @app.get("/")
    async def index() -> FileResponse:
        page = state.static_dir / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="web/static/index.html 不存在（T14 实现）")
        return FileResponse(page, media_type="text/html")

    # -- 参数与场景 --------------------------------------------------------- #

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return state.web_config.to_json()

    @app.get("/api/scenarios")
    async def list_scenarios() -> dict[str, Any]:
        items = []
        for scenario in discover_scenarios(state.scenarios_dir):
            items.append(
                {
                    "id": scenario.id,
                    "desc": scenario.desc,
                    "pending": scenario.pending,
                    "playable": scenario_playable(scenario),
                    "source": "turns" if scenario.turns else ("wav" if scenario.wav else None),
                    "expected_slots": scenario.expected_slots,
                    "turns": [
                        {
                            "index": index,
                            "text": turn.text,
                            "wav": turn.wav.name,
                            "gap_ms": turn.gap_ms,
                        }
                        for index, turn in enumerate(scenario.turns)
                    ],
                }
            )
        return {"scenarios": items}

    @app.get("/api/scenarios/{scenario_id}/turns/{index}/audio")
    async def scenario_turn_audio(scenario_id: str, index: int) -> FileResponse:
        """T14：实时“场景回放”输入源按轮取 wav（`scenario_turn_audio` 按索引，
        这样 `../s03_appointment/turn_01.wav` 这类跨目录引用也能取到，且不能穿越目录）。
        """
        scenario = next(
            (item for item in discover_scenarios(state.scenarios_dir) if item.id == scenario_id),
            None,
        )
        if scenario is None or index < 0 or index >= len(scenario.turns):
            raise HTTPException(status_code=404, detail="unknown scenario turn")
        wav = scenario.turns[index].wav
        if not wav.is_file():
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(wav, media_type="audio/wav", filename=wav.name)

    # -- 历史通话 ----------------------------------------------------------- #

    @app.get("/api/calls")
    async def list_calls() -> dict[str, Any]:
        calls = []
        if state.calls_dir.is_dir():
            for call_dir in state.calls_dir.iterdir():
                if not call_dir.is_dir() or not (call_dir / "events.jsonl").is_file():
                    continue
                calls.append(_call_overview(call_dir))
        calls.sort(key=lambda item: item["mtime"], reverse=True)
        return {"calls": calls}

    @app.get("/api/calls/{call_id}")
    async def get_call(call_id: str) -> dict[str, Any]:
        call_dir = state.call_dir(call_id)
        if not (call_dir / "events.jsonl").is_file():
            raise HTTPException(status_code=404, detail="unknown call")
        payload = _call_overview(call_dir)
        payload["events"] = read_events(call_dir)
        payload["slots_detail"] = _read_json(call_dir / "slots.json")
        payload["summary"] = _read_json(call_dir / "summary.json")
        return payload

    @app.get("/api/calls/{call_id}/audio/{track}")
    async def get_audio(call_id: str, track: str) -> FileResponse:
        if track not in ("caller", "ai"):
            raise HTTPException(status_code=404, detail="track must be caller or ai")
        path = state.call_dir(call_id) / f"{track}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(path, media_type="audio/wav", filename=f"{call_id}_{track}.wav")

    # -- 场景回放 ----------------------------------------------------------- #

    @app.post("/api/replay")
    async def replay(body: dict = Body(...)) -> dict[str, Any]:
        scenario_id = str(body.get("scenario") or "")
        scenario = next(
            (item for item in discover_scenarios(state.scenarios_dir) if item.id == scenario_id),
            None,
        )
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"unknown scenario: {scenario_id}")
        if not scenario_playable(scenario):
            raise HTTPException(status_code=400, detail=f"scenario audio missing: {scenario_id}")
        call_id = new_call_id(f"replay_{scenario.id}")
        out_dir = state.calls_dir / call_id
        task = asyncio.create_task(_run_replay_safely(state, scenario, out_dir))
        state.replay_tasks[call_id] = task
        task.add_done_callback(lambda _t, key=call_id: state.replay_tasks.pop(key, None))
        return {"call_id": call_id, "status": "running"}

    # -- 实时通话 ----------------------------------------------------------- #

    @app.websocket("/ws/call")
    async def ws_call(websocket: WebSocket) -> None:
        await websocket.accept()
        sender = WsSender(websocket)
        sender.start()
        call: RealtimeCall | None = None
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if call is not None and not call.finished:
                        # T16：livekit 适配器的来电者音频走房间音轨，忽略 WS 里的二进制帧。
                        if getattr(call, "accepts_ws_audio", True):
                            await call.feed(message["bytes"])
                    # 帧可能已经在缓冲区里（`receive()` 不需要挂起就返回），显式让出一次
                    # 事件循环，否则转写/合成任务会被持续不断地报帧饿死。
                    await asyncio.sleep(0)
                    continue

                try:
                    payload = json.loads(message.get("text") or "{}")
                except json.JSONDecodeError:
                    sender.send_json({"type": "error", "stage": "protocol", "message": "bad json"})
                    continue
                kind = str(payload.get("type") or "")

                if kind == "start":
                    if call is not None and not call.finished:
                        sender.send_json(
                            {"type": "error", "stage": "start", "message": "call already started"}
                        )
                        continue
                    try:
                        call = await state.start_call(sender)
                        await call.start()
                    except Exception as exc:  # noqa: BLE001 - 服务没配好时告诉页面
                        sender.send_json(
                            {
                                "type": "error",
                                "stage": "start",
                                "message": f"{type(exc).__name__}: {exc}"[:200],
                            }
                        )
                        # T14b：start 失败也要把已经建好的服务／事件文件关掉，
                        # 不留下没 finish 的通话（页面收到 error 后会复位）。
                        if call is not None:
                            try:
                                call.finished = True
                                call.events.close()
                                await call.services.aclose()
                            except Exception:  # noqa: BLE001
                                pass
                        call = None
                        continue
                    sender.send_json(
                        {
                            "type": "call_started",
                            "call_id": call.call_id,
                            "channel_sim": call.channel_sim,
                            # livekit 适配器多给一段房间信息，页面用它连 LiveKit 房间。
                            "livekit": getattr(call, "livekit_info", lambda: None)(),
                        }
                    )
                elif kind == "config":
                    try:
                        state.web_config.apply(payload)
                    except ValueError as exc:
                        sender.send_json(
                            {"type": "error", "stage": "config", "message": str(exc)[:200]}
                        )
                        continue
                    # 只对下一通生效；已经开始的这通继续用旧参数。
                    sender.send_json({"type": "config", "config": state.web_config.to_json()})
                elif kind == "takeover":
                    if call is None or call.finished:
                        sender.send_json(
                            {"type": "error", "stage": "takeover", "message": "no active call"}
                        )
                        continue
                    call.takeover()
                elif kind == "hangup":
                    if call is None or call.finished:
                        sender.send_json(
                            {"type": "error", "stage": "hangup", "message": "no active call"}
                        )
                        continue
                    await call.finish()
                    sender.send_json({"type": "call_ended", "call_id": call.call_id})
                else:
                    sender.send_json(
                        {"type": "error", "stage": "protocol", "message": f"unknown type: {kind!r}"}
                    )
        except WebSocketDisconnect:
            pass
        finally:
            if call is not None and not call.finished:
                await call.finish()
            await sender.close()

    return app


# --------------------------------------------------------------------------- #
# 历史通话的读取辅助
# --------------------------------------------------------------------------- #


def _call_overview(call_dir: Path) -> dict[str, Any]:
    slots_payload = _read_json(call_dir / "slots.json") or {}
    slots = slots_payload.get("slots") or {}
    events = read_events(call_dir) if (call_dir / "events.jsonl").is_file() else []
    return {
        "call_id": call_dir.name,
        "mtime": call_dir.joinpath("events.jsonl").stat().st_mtime,
        "event_count": len(events),
        "state": slots_payload.get("state"),
        "slots": slots,
        "has_summary": (call_dir / "summary.json").is_file(),
        "has_audio": {
            "caller": (call_dir / "caller.wav").is_file(),
            "ai": (call_dir / "ai.wav").is_file(),
        },
        "takeover": any(event.get("type") == "takeover" for event in events),
    }


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


app = create_app()


def main() -> None:  # pragma: no cover - 手动启动用
    import uvicorn

    config = load_config(ROOT / "config.yaml")
    host = str((config.get("web") or {}).get("host") or "127.0.0.1")
    port = int((config.get("web") or {}).get("port") or 8000)
    uvicorn.run("web.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()

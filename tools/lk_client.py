"""T16：LiveKit 路径的场景回放客户端。

`ws_client.py` 把音频与事件都塞进一条 WebSocket，驱动不了 LiveKit 路径。这个客户端
按任务书要求分两半：

- **控制与事件**：仍然是同一个 `WS /ws/call`（`start` / `hangup` 控制消息，
  事件流用于时延与抢话统计）；
- **音频**：LiveKit 房间（WebRTC）——`call_started` 里带回房间 URL 与浏览器令牌，
  本客户端进房间、发布一条“来电者”音轨（按 20ms 实时节奏推场景 wav），
  并订阅 AI 音轨当作“听到 AI 开口”的时刻。

**测量与报告与 `ws_client.py` 共用同一份代码**：`ScenarioRunner` 负责轮次计划、节奏、
时延/抢话统计与事件解析，这里只实现 `open()/close()/hangup()/_send_frame()`。
`--out` 出来的 JSON 结构与 `ws_client.py` 完全一致（多一个 `transport` 字段），
现有统计脚本可以直接读。

**WebRTC 比 WebSocket 多出的固定时延**：Opus 编解码 + 接收侧抖动缓冲 + 播放队列
（本项目 `AudioSource` 的 `queue_size_ms=200`）。这些只影响“客户端听到 AI 开口”的时刻
（`TurnMeasurement.latency_ms` 的右端），不影响服务端事件里的分段时延
（`asr_*` / `llm_*` / `tts_*` / `play_start` 都是服务端记的）。同机对比时可以用
`ws_client.py` 的 `latency_ms` 减出这段差值，报告里单独标注。

用法：

    uv run python -m tools.lk_client --all --out /tmp/lk.json
    uv run python -m tools.lk_client --scenario scenarios/barge_in/s03b_barge_in.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from tools import ws_client
from tools.ws_client import (
    Scenario,
    ScenarioRunner,
    TurnMeasurement,
    run_scenarios,
)

# 等 `call_started`（服务端要先把 LiveKit 通话建起来并签发令牌）。
CALL_START_TIMEOUT_S = 20.0
# 等 AI 音轨订阅上。
SUBSCRIBE_TIMEOUT_S = 20.0
# WebRTC 的音轨是**连续**的（不说话时也在送静音帧）。低于该 RMS 的帧当成静音丢掉，
# 否则“收到音频 = AI 在说话”的假设不成立：`first_audio_at` / `last_activity_at` /
# `playback_cursor_ms` 全部会被静音帧推着走（踩过：首轮时延量成 10ms、
# 每轮都等不到服务端空闲）。
SILENCE_RMS = 30.0


class LiveKitScenarioClient(ScenarioRunner):
    """事件走 WebSocket、音频走 LiveKit 房间的场景客户端。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ws: Any = None
        self._receiver: asyncio.Task | None = None
        self._room: Any = None
        self._source: Any = None
        self._audio_tasks: set[asyncio.Task] = set()
        self._subscribed = asyncio.Event()
        # 最近一次收到 AI 音频的墙钟时刻（毫秒），以及最近一次 `play_end` 的时刻。
        self._last_audio_at_ms = 0.0
        self._last_play_end_at_ms = 0.0

    # ---- 传输钩子 ---------------------------------------------------------

    async def open(self) -> None:
        import websockets

        self._ws = await websockets.connect(self.url, max_size=None).__aenter__()
        self._receiver = asyncio.create_task(self._recv_loop())
        await self._ws.send(json.dumps({"type": "start"}))
        started = asyncio.create_task(self.call_started.wait())
        failed = asyncio.create_task(self.start_failed.wait())
        try:
            done, _pending = await asyncio.wait(
                [started, failed],
                timeout=CALL_START_TIMEOUT_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            started.cancel()
            failed.cancel()
        if self.start_failed.is_set():
            raise SystemExit(f"服务端 start 失败：{self.start_error}")
        if not done:
            raise SystemExit("等服务端 call_started 超时")

        info = (self.call_started_message or {}).get("livekit") or {}
        if not info.get("url") or not info.get("token"):
            raise SystemExit(
                "服务端没有返回 LiveKit 房间信息；确认 config.yaml: web.adapter 是 livekit"
            )
        await self._join_room(info)

    async def _join_room(self, info: dict[str, Any]) -> None:
        from livekit import rtc

        self._room = rtc.Room()

        @self._room.on("track_subscribed")
        def _on_track_subscribed(track: Any, _publication: Any, participant: Any) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            task = asyncio.create_task(self._consume_track(track))
            self._audio_tasks.add(task)
            task.add_done_callback(self._audio_tasks.discard)
            self._subscribed.set()
            self._log(f"[info] 已订阅 {participant.identity} 的 AI 音轨")

        await self._room.connect(info["url"], info["token"])
        self._source = rtc.AudioSource(self.sample_rate, 1, queue_size_ms=200)
        track = rtc.LocalAudioTrack.create_audio_track("caller", self._source)
        await self._room.local_participant.publish_track(track)
        try:
            await asyncio.wait_for(self._subscribed.wait(), timeout=SUBSCRIBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._log("[warn] 没等到 AI 音轨订阅上")

    def _on_audio(self, frame: bytes) -> None:
        if _rms(frame) < SILENCE_RMS:
            return
        self._last_audio_at_ms = ws_client.now_ms()
        super()._on_audio(frame)

    def _on_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "play_end":
            self._last_play_end_at_ms = ws_client.now_ms()
        super()._on_event(event)

    async def _wait_for_gap(self, plan: Any) -> None:
        """等到 AI 确实说完了、再静 `gap_ms` 才开口。

        与 WebSocket 路径的差别：那边服务端把音频一次推完，`playback_cursor_ms` 会跑在
        墙钟前面，等“队列播完 + gap”才有意义；WebRTC 这边音频是实时到达的，
        `playback_cursor_ms ≈ now`，所以改成“最后一次 AI 音频 / `play_end` 之后过了 gap_ms”。

        两个都要看：云端 TTS 比实时慢时，一句话的音频中间会有超过 `gap_ms` 的空档
        （合成跟不上播放），只看“没有音频”会在句子中间就抢话（实测把问候语打断在
        160ms 处）。`play_end` 是服务端在整句播完时写的，用它兜住这种情况。
        """
        target = max(self._last_audio_at_ms, self._last_play_end_at_ms) + plan.gap_ms
        while ws_client.now_ms() < target:
            await self._send_frame(bytes(self.frame_bytes))

    async def _consume_track(self, track: Any) -> None:
        from livekit import rtc

        stream = rtc.AudioStream(track, sample_rate=self.sample_rate, num_channels=1)
        try:
            async for event in stream:
                self._on_audio(bytes(event.frame.data))
        except asyncio.CancelledError:
            raise
        finally:
            await stream.aclose()

    async def _send_frame(self, frame: bytes) -> None:
        from livekit import rtc

        await self._pace()
        await self._source.capture_frame(
            rtc.AudioFrame(
                data=frame,
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=len(frame) // 2,
            )
        )

    async def hangup(self) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps({"type": "hangup"}))

    async def close(self) -> None:
        for task in list(self._audio_tasks):
            task.cancel()
        if self._audio_tasks:
            await asyncio.gather(*self._audio_tasks, return_exceptions=True)
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:  # noqa: BLE001 - 收尾失败不影响已量到的数据
                pass
            self._room = None
        if self._receiver is not None:
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
            self._receiver = None
        if self._ws is not None:
            await self._ws.__aexit__(None, None, None)
            self._ws = None

    # ---- 接收（事件） -----------------------------------------------------

    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, (bytes, bytearray)):
                    # 音频走房间；WS 上不该有二进制帧。
                    continue
                self._on_event(json.loads(message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 连接断开时给出原因
            self._log(f"[warn] 接收循环退出：{type(exc).__name__}: {exc}")


def _rms(frame: bytes) -> float:
    """一帧 16-bit 单声道 PCM 的均方根（用来把静音帧滤掉）。"""
    import numpy as np

    samples = np.frombuffer(frame, dtype="<i2")
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


async def run_scenario_livekit(
    scenario: Scenario,
    *,
    url: str,
    sample_rate: int,
    frame_ms: int,
    gap_override: float | None,
    linger_ms: float,
    idle_settle_ms: float,
    log=print,
) -> tuple[list[TurnMeasurement], str | None, list[dict[str, Any]]]:
    client = LiveKitScenarioClient(
        scenario,
        url=url,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        gap_override=gap_override,
        linger_ms=linger_ms,
        idle_settle_ms=idle_settle_ms,
        log=log,
    )
    measurements = await client.run()
    return measurements, client.call_id, client.events


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = ws_client.parse_args(argv)
    args.transport = "livekit"
    return args


async def run(args: argparse.Namespace) -> int:
    return await run_scenarios(
        args,
        lambda scenario: run_scenario_livekit(
            scenario,
            url=args.url,
            sample_rate=int(args.sample_rate),
            frame_ms=args.frame_ms,
            gap_override=args.gap_ms,
            linger_ms=args.linger_ms,
            idle_settle_ms=args.idle_settle_ms,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(run(parse_args(argv)))
    except KeyboardInterrupt:  # pragma: no cover - 手动运行用
        return 130


if __name__ == "__main__":  # pragma: no cover - 手动运行用
    raise SystemExit(main())

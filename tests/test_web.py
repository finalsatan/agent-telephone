"""T13 验收测试：测试页面后端（FastAPI + WebSocket）与实时链路。

全部用 fake ASR／LLM／TTS／VAD／Smart Turn 和临时目录，不联网、不加载 ONNX。
`TurnDetector` 用注入的 fake 模型，这样切轮、预卷与 `t_ms` 是真实实现。

覆盖：

- `GET /` 返回页面；`GET /api/scenarios` 列出场景；`GET /api/calls` / `{id}` 返回历史；
  音频接口返回 wav；
- `WS /ws/call`：`start` 收到 `call_start`；发假音频帧后收到 `vad_start` / `vad_end` /
  `turn_end` / `asr_final` / `play_start` 事件与 `slots` / `state` 汇总消息，以及 AI 音频帧；
- `takeover` 收到 `takeover` 事件，之后不再转写、不再出声；
- 抢话（实时链路）：来电者在 AI 说话时连续语音达到阈值 → `interrupted(reason="barge_in")`；
- `config` 消息只对下一通生效（`GET /api/config`）；
- `POST /api/replay` 用场景回放，事件与槽位可通过历史接口轮询到。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from core.events import EventLogger, read_events
from core.phrases import PhraseCache, cache_texts
from core.turn import TurnDetector
from core.types import AsrResult, ReplyChunk, TurnResult
from tools.replay import ReplayServices
from web.server import apply_web_config, create_app

SAMPLE_RATE = 16000
OWNER = "张先生"
# 32ms 一帧（512 采样）：非零样本算“说话”，全零算静音（fake VAD）。
SPEECH_FRAME = b"\x01\x00" * 512
SILENCE_FRAME = bytes(1024)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FrameVad:
    """按“帧里有没有非零样本”判语音，避免加载 Silero ONNX。"""

    def __init__(self) -> None:
        self.reset_calls = 0

    def speech_probability(self, frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    def reset(self) -> None:
        self.reset_calls += 1


class AlwaysComplete:
    """Smart Turn 永远判“说完了”，让 turn_end 在 speech_end 时立刻触发。"""

    def predict(self, pcm: bytes) -> tuple[bool, float]:
        return True, 0.98


class FakeAsr:
    def __init__(self, text: str = "你好，我姓王。") -> None:
        self.text = text
        self.calls: list[int] = []

    async def transcribe(self, pcm: bytes) -> AsrResult:
        self.calls.append(len(pcm))
        return AsrResult(text=self.text, latency_ms=5.0, audio_ms=len(pcm) / 32.0)

    async def aclose(self) -> None:
        return None


class FakeLlm:
    def __init__(self, slots_update: dict[str, str] | None = None, *, delay_s: float = 0.0) -> None:
        self.slots_update = dict(slots_update or {})
        self.delay_s = delay_s
        self.calls: list[str] = []
        self.histories: list[list[dict[str, str]]] = []

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.calls.append(user_text)
        self.histories.append([dict(message) for message in history])
        if self.delay_s:
            # 慢 LLM：给“AI 还在生成时被抢话”留出时间窗口。
            await asyncio.sleep(self.delay_s)
        yield ReplyChunk(kind="sentence", text="好的。", t_ms=5.0)
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply="好的。",
                slots_update=dict(self.slots_update),
                first_token_ms=5.0,
                total_ms=10.0,
                parse_ok=True,
                first_sentence_ms=5.0,
            ),
            t_ms=10.0,
        )

    async def aclose(self) -> None:
        return None


class FakeTts:
    """`delay_s` 用来给抢话留出时间；每块 10ms 音频。"""

    sample_rate = SAMPLE_RATE

    def __init__(self, *, delay_s: float = 0.0, chunks: int = 2) -> None:
        self.delay_s = delay_s
        self.chunks = chunks
        self.calls: list[str] = []

    async def synth(self, text, cancel):
        self.calls.append(text)
        for _ in range(self.chunks):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if cancel.is_set():
                return
            yield b"\x00\x00" * 160

    async def aclose(self) -> None:
        return None


def make_cache(config: dict, *, fill: bool, samples: int = 800) -> PhraseCache:
    cache = PhraseCache.from_config(config, backend="local", voice="test-voice")
    if fill:
        for text in cache_texts(OWNER).values():
            cache.put(text, b"\x00\x00" * samples)
    return cache


def make_config(tmp_path: Path) -> dict:
    return {
        "siliconflow": {"base_url": "https://api.example.test/v1"},
        "owner": {"name": OWNER},
        "dialog": {"max_asks": 2, "max_confirm_rounds": 2},
        "vad": {"stop_secs": 3.0, "pre_speech_ms": 500, "speech_end_silence_ms": 200},
        "audio": {"sample_rate": SAMPLE_RATE},
        "asr": {"model": "Test/Default-ASR"},
        "llm": {"realtime_model": "Test/Default-LLM"},
        "interrupt": {"min_speech_ms": 300},
        "web": {"channel_sim": False, "calls_dir": "calls"},
        "tts": {
            "backend": "local",
            "voice": "",
            "phrase_cache": {"dir": str(tmp_path / "cache")},
        },
    }


# --------------------------------------------------------------------------- #
# 构造客户端与读取辅助
# --------------------------------------------------------------------------- #


def make_client(
    tmp_path: Path,
    *,
    config: dict | None = None,
    fill_cache: bool = True,
    cache_samples: int = 800,
    tts: FakeTts | None = None,
    asr: FakeAsr | None = None,
    llm: FakeLlm | None = None,
    replay_runner=None,
) -> tuple[TestClient, dict]:
    root = tmp_path / "root"
    (root / "web" / "static").mkdir(parents=True)
    (root / "web" / "static" / "index.html").write_text("<html>T13</html>", encoding="utf-8")
    calls_dir = tmp_path / "calls"
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    config = config or make_config(tmp_path)
    created: list[ReplayServices] = []

    async def services_factory(cfg):
        services = ReplayServices(
            asr=asr or FakeAsr(),
            llm=llm or FakeLlm({"name": "王强"}),
            tts=tts or FakeTts(),
            cache=make_cache(cfg, fill=fill_cache, samples=cache_samples),
            backend="local",
        )
        created.append(services)
        return services

    def detector_factory(cfg):
        return TurnDetector(vad=FrameVad(), smart_turn=AlwaysComplete())

    app = create_app(
        root=root,
        config=config,
        calls_dir=calls_dir,
        scenarios_dir=scenarios_dir,
        services_factory=services_factory,
        detector_factory=detector_factory,
        replay_runner=replay_runner,
    )
    client = TestClient(app)
    return client, {"created": created, "calls_dir": calls_dir, "scenarios_dir": scenarios_dir}


def ws_json(message: dict) -> dict | None:
    text = message.get("text")
    return json.loads(text) if text else None


def recv_until(ws, predicate, *, limit: int = 500) -> list[dict]:
    """一直收消息（文本 + 二进制）直到 predicate(message) 为真。"""
    messages: list[dict] = []
    for _ in range(limit):
        message = ws.receive()
        messages.append(message)
        if predicate(message):
            return messages
    raise AssertionError(f"did not see expected message in {limit} messages")


def event_types(message: dict) -> list[str]:
    payload = ws_json(message)
    return [payload["type"]] if payload else []


def has_event(name: str):
    def predicate(message: dict) -> bool:
        payload = ws_json(message)
        return bool(payload and payload.get("type") == name)

    return predicate


def feed_speech_and_silence(ws, *, speech_frames: int = 6, silence_frames: int = 8) -> None:
    for _ in range(speech_frames):
        ws.send_bytes(SPEECH_FRAME)
    for _ in range(silence_frames):
        ws.send_bytes(SILENCE_FRAME)


def start_call(ws) -> list[dict]:
    ws.send_json({"type": "start"})
    return recv_until(ws, has_event("call_started"))


# --------------------------------------------------------------------------- #
# HTTP 接口
# --------------------------------------------------------------------------- #


def test_index_returns_page(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "T13" in response.text


def test_config_endpoint(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    payload = client.get("/api/config").json()
    assert payload["channel_sim"] is False
    assert payload["models"] == {
        "asr": "Test/Default-ASR",
        "llm": "Test/Default-LLM",
        "tts_backend": "local",
    }
    assert payload["interrupt_min_speech_ms"] == 300.0


def test_scenarios_endpoint(tmp_path: Path) -> None:
    client, env = make_client(tmp_path)
    scenario_dir = env["scenarios_dir"]
    (scenario_dir / "s01_demo").mkdir()
    (scenario_dir / "s01_demo" / "turn_01.wav").write_bytes(b"RIFF")
    (scenario_dir / "s01_demo" / "turn_02.wav").write_bytes(b"RIFF")
    (scenario_dir / "s01_demo.json").write_text(
        json.dumps(
            {
                "id": "s01_demo",
                "desc": "演示场景",
                "turns": [
                    {"text": "你好。", "gap_ms": 700},
                    {"text": "我姓王。", "gap_ms": 900},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = client.get("/api/scenarios").json()
    assert [item["id"] for item in payload["scenarios"]] == ["s01_demo"]
    scenario = payload["scenarios"][0]
    assert scenario["playable"] is True
    assert scenario["source"] == "turns"
    assert [turn["text"] for turn in scenario["turns"]] == ["你好。", "我姓王。"]
    assert scenario["turns"][0]["gap_ms"] == 700


def test_calls_endpoints_after_a_call(tmp_path: Path) -> None:
    client, env = make_client(tmp_path)
    with client.websocket_connect("/ws/call") as ws:
        assert any(ws_json(m) and ws_json(m)["type"] == "call_start" for m in start_call(ws))
        feed_speech_and_silence(ws)
        recv_until(ws, has_event("asr_final"))
        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))

    listing = client.get("/api/calls").json()["calls"]
    assert len(listing) == 1
    call_id = listing[0]["call_id"]
    assert call_id.startswith("web_")
    assert listing[0]["has_audio"]["caller"] is True
    assert listing[0]["has_audio"]["ai"] is True

    detail = client.get(f"/api/calls/{call_id}").json()
    types = [event["type"] for event in detail["events"]]
    assert types[0] == "call_start"
    assert types[-1] == "call_end"
    assert "asr_final" in types
    assert detail["slots_detail"]["slots"]["name"]["value"] == "王强"

    audio = client.get(f"/api/calls/{call_id}/audio/caller")
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"
    assert client.get(f"/api/calls/{call_id}/audio/nope").status_code == 404
    assert client.get("/api/calls/../etc/passwd").status_code == 404


def test_replay_endpoint_runs_scenario(tmp_path: Path) -> None:
    calls: list[Path] = []

    async def fake_replay_runner(state, scenario, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = EventLogger(out_dir)
        try:
            logger.log("call_start", source="replay", scenario=scenario.id)
            logger.log("call_end", state="CLOSING", turns=2)
        finally:
            logger.close()
        (out_dir / "slots.json").write_text(
            json.dumps({"call_id": out_dir.name, "state": "CLOSING", "slots": {}}),
            encoding="utf-8",
        )
        calls.append(out_dir)

    client, env = make_client(tmp_path, replay_runner=fake_replay_runner)
    scenario_dir = env["scenarios_dir"]
    (scenario_dir / "s02_demo").mkdir()
    (scenario_dir / "s02_demo" / "turn_01.wav").write_bytes(b"RIFF")
    (scenario_dir / "s02_demo.json").write_text(
        json.dumps({"id": "s02_demo", "turns": [{"text": "喂。"}]}), encoding="utf-8"
    )

    response = client.post("/api/replay", json={"scenario": "s02_demo"})
    assert response.status_code == 200
    call_id = response.json()["call_id"]
    assert call_id.startswith("replay_s02_demo")

    detail = None
    for _ in range(100):
        detail = client.get(f"/api/calls/{call_id}")
        if detail.status_code == 200 and len(detail.json()["events"]) >= 2:
            break
        time.sleep(0.02)
    assert detail is not None and detail.status_code == 200
    assert [event["type"] for event in detail.json()["events"]] == ["call_start", "call_end"]
    assert calls and calls[0].name == call_id

    assert client.post("/api/replay", json={"scenario": "missing"}).status_code == 404


# --------------------------------------------------------------------------- #
# WebSocket：开始、事件转发、挂断
# --------------------------------------------------------------------------- #


def test_ws_start_forwards_events_and_audio(tmp_path: Path) -> None:
    client, env = make_client(tmp_path)
    with client.websocket_connect("/ws/call") as ws:
        messages = start_call(ws)
        payloads = [ws_json(m) for m in messages if ws_json(m)]
        assert any(payload["type"] == "call_start" for payload in payloads)
        # 问候语是缓存话术：应收到 AI 音频帧与槽位/状态汇总
        assert any(m.get("bytes") for m in messages)
        assert any(payload["type"] == "slots" for payload in payloads)
        assert any(payload["type"] == "state" for payload in payloads)

        feed_speech_and_silence(ws)
        messages = recv_until(
            ws,
            lambda m: bool((p := ws_json(m)) and p.get("type") == "slots" and p["slots"]["name"]["value"] == "王强"),
        )
        payloads = [p for p in (ws_json(m) for m in messages) if p]
        types = [payload["type"] for payload in payloads]
        for name in ("vad_start", "vad_end", "turn_end", "asr_request", "asr_final"):
            assert name in types, f"{name} 未转发"
        # 0.3 秒语音不算抢话（本轮 AI 还没开始说话）
        assert "interrupted" not in types
        # 通话助手回的是模板话术（缓存命中）：播完问候与回复都有音频帧
        assert any(m.get("bytes") for m in messages)

        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))

    assert len(env["created"]) == 1
    # 事件已落盘并含转写与回复
    call_dirs = [path for path in env["calls_dir"].iterdir() if path.is_dir()]
    assert len(call_dirs) == 1
    types = [event["type"] for event in read_events(call_dirs[0])]
    assert types[0] == "call_start"
    assert types[-1] == "call_end"
    assert "asr_final" in types and "play_start" in types


def test_ws_speech_triggers_turn_and_slots(tmp_path: Path) -> None:
    client, _ = make_client(
        tmp_path,
        asr=FakeAsr("我姓王，电话是一三八零零一三八零零零。"),
        llm=FakeLlm({"name": "王强", "callback_number": "13800138000"}),
    )
    with client.websocket_connect("/ws/call") as ws:
        start_call(ws)
        feed_speech_and_silence(ws, speech_frames=6, silence_frames=8)
        recv_until(
            ws,
            lambda m: bool(
                (p := ws_json(m))
                and p.get("type") == "slots"
                and p["slots"]["callback_number"]["value"] == "13800138000"
            ),
        )
        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))


def test_ws_takeover_silences_ai(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/call") as ws:
        start_call(ws)
        feed_speech_and_silence(ws)
        recv_until(ws, has_event("asr_final"))

        ws.send_json({"type": "takeover"})
        messages = recv_until(ws, has_event("takeover"))
        assert any(
            (p := ws_json(m)) and p.get("type") == "state" and p.get("state") == "ASK_PURPOSE"
            for m in messages
        )

        # 接管后再说话：不再切轮、不再转写
        feed_speech_and_silence(ws, speech_frames=6, silence_frames=10)
        ws.send_json({"type": "hangup"})
        messages = recv_until(ws, has_event("call_ended"))
        after = [p for p in (ws_json(m) for m in messages) if p]
        assert not any(payload["type"] == "asr_final" for payload in after)
        assert not any(payload["type"] == "turn_end" for payload in after)


def test_ws_config_applies_to_next_call(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/call") as ws:
        ws.send_json(
            {
                "type": "config",
                "channel_sim": True,
                "models": {"asr": "Test/ASR", "llm": "Test/LLM", "tts_backend": "cloud"},
                "interrupt_min_speech_ms": 500,
            }
        )
        messages = recv_until(ws, has_event("config"))
        assert any(
            (p := ws_json(m)) and p.get("type") == "config" and p["config"]["channel_sim"] is True
            for m in messages
        )
        ws.send_json({"type": "start"})
        recv_until(ws, has_event("call_started"))
        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))

    payload = client.get("/api/config").json()
    assert payload["channel_sim"] is True
    assert payload["models"] == {"asr": "Test/ASR", "llm": "Test/LLM", "tts_backend": "cloud"}
    assert payload["interrupt_min_speech_ms"] == 500.0

    # config 只影响下一通：参数合进 config 副本，不改原始 config.yaml 内容
    config = make_config(tmp_path)
    merged = apply_web_config(
        config,
        type(
            "Web",
            (),
            {"asr_model": "Test/ASR", "llm_model": "Test/LLM", "tts_backend": "local"},
        )(),
    )
    assert merged["asr"]["model"] == "Test/ASR"
    assert merged["llm"]["realtime_model"] == "Test/LLM"
    assert config["asr"]["model"] != "Test/ASR"


# --------------------------------------------------------------------------- #
# WebSocket：抢话（实时链路）
# --------------------------------------------------------------------------- #


def test_ws_barge_in_interrupts_ai_speech(tmp_path: Path) -> None:
    # 缓存故意不填，让回复走真实 TTS 路径（fake TTS 每块 sleep 80ms，留出打断时间）。
    client, _ = make_client(tmp_path, fill_cache=False, tts=FakeTts(delay_s=0.08, chunks=8))
    with client.websocket_connect("/ws/call") as ws:
        start_call(ws)
        # 先说一句，触发一轮回复；回复的 TTS 会持续约 0.6s
        feed_speech_and_silence(ws, speech_frames=4, silence_frames=8)
        recv_until(ws, has_event("play_start"))
        # AI 还在说：来电者连续说话 12 帧（384ms）≥ interrupt.min_speech_ms(300)
        for _ in range(14):
            ws.send_bytes(SPEECH_FRAME)
        recv_until(
            ws,
            lambda m: bool(
                (p := ws_json(m))
                and p.get("type") == "interrupted"
                and p.get("data", {}).get("reason") == "barge_in"
            ),
        )
        # 事件之后紧跟被打断句子的 transcript 消息（页面据此标 [被打断]）
        message = recv_until(ws, lambda m: bool((p := ws_json(m)) and p.get("type") == "transcript"))
        transcript = ws_json(message[-1])
        assert transcript["role"] == "ai"
        assert transcript["interrupted"] is True

        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))

    call_dirs = [path for path in (tmp_path / "calls").iterdir() if path.is_dir()]
    events = read_events(call_dirs[0])
    types = [event["type"] for event in events]
    play_start = types.index("play_start")
    interrupt = types.index("interrupted")
    play_end = types.index("play_end", interrupt)
    # 事件顺序：play_start → interrupted(barge_in) → play_end（T11b 的“未说完”约定）
    assert play_start < interrupt < play_end
    assert events[interrupt]["data"]["reason"] == "barge_in"


def test_ws_short_noise_does_not_interrupt(tmp_path: Path) -> None:
    """短促的噪声（不足 min_speech_ms）不打断 AI。"""
    client, _ = make_client(tmp_path, fill_cache=False, tts=FakeTts(delay_s=0.05, chunks=10))
    with client.websocket_connect("/ws/call") as ws:
        start_call(ws)
        feed_speech_and_silence(ws, speech_frames=4, silence_frames=8)
        recv_until(ws, has_event("play_start"))
        # 160ms < 300ms：不应触发抢话
        for _ in range(5):
            ws.send_bytes(SPEECH_FRAME)
        for _ in range(6):
            ws.send_bytes(SILENCE_FRAME)
        ws.send_json({"type": "hangup"})
        messages = recv_until(ws, has_event("call_ended"))
        payloads = [p for p in (ws_json(m) for m in messages) if p]
        assert not any(
            payload["type"] == "interrupted" and payload.get("data", {}).get("reason") == "barge_in"
            for payload in payloads
        )


def test_ws_tail_barge_in_marks_sentence_unfinished(tmp_path: Path) -> None:
    """T13b：问候语已发完、浏览器还在播时抢话 → 摘要标未说完、历史带（被对方打断）。

    服务端不等播放（缓存话术一次推完），所以抢话时早已记过 `play_end`；补记的
    `interrupted` 排在它后面，摘要必须靠这条事件才认得出来。
    """
    from core.summary import dialogue_lines

    llm = FakeLlm({"name": "王强"})
    # 缓存话术 3 秒：服务端瞬间送完并记 play_end，浏览器还要播很久（尾音抢话）。
    client, env = make_client(tmp_path, llm=llm, cache_samples=3 * SAMPLE_RATE)
    with client.websocket_connect("/ws/call") as ws:
        # start_call 已经消费到 call_started，问候语的 play_start / play_end 都已收到。
        start_call(ws)
        time.sleep(0.05)
        # 尾音抢话：连续 12 帧（384ms）≥ min_speech_ms(300)
        for _ in range(12):
            ws.send_bytes(SPEECH_FRAME)
        recv_until(
            ws,
            lambda m: bool(
                (p := ws_json(m))
                and p.get("type") == "interrupted"
                and p.get("data", {}).get("reason") == "barge_in"
                and p.get("data", {}).get("tail")
            ),
        )
        # 页面收到 AI 逐句的“被打断”标记
        message = recv_until(
            ws,
            lambda m: bool(
                (p := ws_json(m))
                and p.get("type") == "transcript"
                and p.get("role") == "ai"
                and p.get("interrupted")
            ),
        )
        assert ws_json(message[-1])["interrupted"] is True
        # 继续把这一轮说完，触发 LLM（用来检查历史）
        for _ in range(8):
            ws.send_bytes(SILENCE_FRAME)
        recv_until(ws, lambda m: len(llm.calls) >= 1)
        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))

    call_dirs = [path for path in env["calls_dir"].iterdir() if path.is_dir()]
    events = read_events(call_dirs[0])
    tail = [
        event
        for event in events
        if event["type"] == "interrupted" and event["data"].get("tail")
    ]
    assert tail, "没有尾音抢话事件"
    assert tail[0]["data"]["played_ms"] is not None
    assert tail[0]["data"]["text"]

    # 摘要的对话还原把这句标成未说完
    lines = dialogue_lines(events)
    assert any(line.speaker == "AI" and line.interrupted for line in lines)
    # 历史里最后一条 assistant 带（被对方打断）
    assert any(
        message["role"] == "assistant" and message["content"].endswith("（被对方打断）")
        for message in llm.histories[-1]
    )


def test_ws_barge_in_before_playback_carries_text_to_next_turn(tmp_path: Path) -> None:
    """AI 还没开播就被抢话：这一轮的文本顺延到下一轮，不丢内容也不重复追问。

    对应离线回放的“轮次合并”：不把被放弃那一轮的文本丢掉，也不把它标成 AI 说的话。
    """
    asr = FakeAsr("我想找张先生。")
    llm = FakeLlm({}, delay_s=0.4)
    client, _ = make_client(
        tmp_path, fill_cache=False, tts=FakeTts(delay_s=0.02, chunks=2), asr=asr, llm=llm
    )
    with client.websocket_connect("/ws/call") as ws:
        start_call(ws)
        # 第一轮：说完后 LLM 开始生成（0.4s 窗口）
        feed_speech_and_silence(ws, speech_frames=4, silence_frames=8)
        recv_until(ws, has_event("llm_request"))
        # AI 还在生成时抢话：连续 12 帧（384ms）≥ 300ms 阈值
        for _ in range(12):
            ws.send_bytes(SPEECH_FRAME)
        recv_until(
            ws,
            lambda m: bool(
                (p := ws_json(m))
                and p.get("type") == "interrupted"
                and p.get("data", {}).get("stage") == "llm"
            ),
        )
        # 第二轮说完后，LLM 应收到“上一轮 + 本轮”的拼接文本
        for _ in range(10):
            ws.send_bytes(SILENCE_FRAME)
        messages = recv_until(
            ws,
            lambda m: bool(
                (p := ws_json(m)) and p.get("type") == "asr_final" and len(llm.calls) >= 2
            ),
        )
        payloads = [p for p in (ws_json(m) for m in messages) if p]
        # LLM 阶段的抢话没有播出音频，不能生成“AI 被打断”的逐句
        assert not any(
            payload["type"] == "transcript"
            and payload.get("role") == "ai"
            and payload.get("interrupted")
            for payload in payloads
        )
        ws.send_json({"type": "hangup"})
        recv_until(ws, has_event("call_ended"))

    assert llm.calls[0] == "我想找张先生。"
    assert llm.calls[1] == "我想找张先生。我想找张先生。"

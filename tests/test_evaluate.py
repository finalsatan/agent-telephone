"""T12 验收：场景按轮回放、评估指标与场景生成。

全部用 fake 服务和合成出来的「静音 wav」，不联网、不依赖本地 TTS。
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from core.phrases import PhraseCache, cache_texts
from core.types import AsrResult, ReplyChunk, TurnResult
from tools.evaluate import (
    aggregate,
    evaluate_call,
    evaluate_scenarios,
    extract_turn_metrics,
    score_slots,
    slot_value_matches,
)
from tools.make_scenarios import (
    REAL_PLACEHOLDERS,
    scenario_payload,
    turn_text,
    write_real_placeholders,
    write_scenario,
)
from tools.replay import (
    ReplayServices,
    load_scenario,
    replay_scenario_to_dir,
    scenario_playable,
)

OWNER = "张先生"
SAMPLE_RATE = 16000
PCM_100MS = b"\x00\x00" * (SAMPLE_RATE // 10)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeAsr:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[bytes] = []

    async def transcribe(self, pcm: bytes) -> AsrResult:
        self.calls.append(pcm)
        text = self.texts.pop(0) if self.texts else ""
        return AsrResult(text=text, latency_ms=120.0, audio_ms=len(pcm) / 32.0)


class FakeLlm:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[str] = []

    async def turn_stream(self, state, slots, history, user_text, attempts=None):
        self.calls.append(user_text)
        reply, slots_update = self.responder(user_text)
        sentences = [part for part in reply.split("。") if part]
        for sentence in sentences:
            yield ReplyChunk(kind="sentence", text=sentence + "。", t_ms=10.0)
        yield ReplyChunk(
            kind="done",
            result=TurnResult(
                reply=reply,
                slots_update=slots_update,
                first_token_ms=10.0,
                total_ms=100.0,
                parse_ok=True,
                first_sentence_ms=10.0 if sentences else None,
            ),
            t_ms=100.0,
        )


class FakeTts:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synth(self, text, cancel):
        self.calls.append(text)
        yield b"\x00\x00" * 1600  # 0.1s
        yield b"\x00\x00" * 1600


# --------------------------------------------------------------------------- #
# 场景文件
# --------------------------------------------------------------------------- #


def write_wav(path: Path, pcm: bytes = PCM_100MS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm)


def write_scenario_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "siliconflow": {"base_url": "https://api.example.test/v1"},
        "owner": {"name": OWNER},
        "dialog": {"max_asks": 2, "max_confirm_rounds": 2},
        "vad": {"stop_secs": 3.0},
        "audio": {"sample_rate": SAMPLE_RATE},
        "scenarios": {"default_gap_ms": 800},
        "llm": {"reply_mode": "full"},
        "tts": {
            "backend": "local",
            "voice": "",
            "confirm_lead": False,
            "phrase_cache": {"dir": str(tmp_path / "cache")},
        },
    }
    for key, value in overrides.items():
        config[key] = value
    return config


def make_cache(config: dict) -> PhraseCache:
    cache = PhraseCache.from_config(config, backend="local", voice="test-voice")
    for text in cache_texts(OWNER).values():
        cache.put(text, b"\x00\x00" * 1600)
    return cache


def build_two_scenarios(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "scenarios"
    first = root / "s01_delivery.json"
    write_scenario_json(
        first,
        {
            "id": "s01_delivery",
            "desc": "快递送货",
            "expected_slots": {
                "name": ["李", "李先生"],
                "purpose": ["送快递"],
                "callback_number": "13800138000",
            },
            "turns": [
                {"text": "你好，我找张先生。", "wav": "turn_01.wav", "gap_ms": 800},
                {"text": "我姓李，是送快递的。", "wav": "turn_02.wav", "gap_ms": 800},
                {"text": "电话是一三八零零一三八零零零。", "wav": "turn_03.wav", "gap_ms": 800},
            ],
        },
    )
    for index in range(1, 4):
        write_wav(root / "s01_delivery" / f"turn_{index:02d}.wav")

    second = root / "s02_message.json"
    write_scenario_json(
        second,
        {
            "id": "s02_message",
            "desc": "一般留言",
            "expected_slots": {
                "name": ["陈"],
                "purpose": ["让他回电话"],
                "callback_number": "13500135000",
            },
            "turns": [
                {"text": "我是老陈，让他回个电话。", "wav": "turn_01.wav", "gap_ms": 500},
                {"text": "号码是幺三五零零一三五零零零。", "wav": "turn_02.wav", "gap_ms": 500},
            ],
        },
    )
    for index in range(1, 3):
        write_wav(root / "s02_message" / f"turn_{index:02d}.wav")
    return first, second


def responder(text: str):
    if "老陈" in text:
        return "", {"name": "陈", "purpose": "让他回电话"}
    if "姓李" in text:
        return "", {"name": "李", "purpose": "送快递"}
    return "", {}


def make_services(config: dict) -> ReplayServices:
    return ReplayServices(
        asr=FakeAsr(["你好，我找张先生。", "我姓李，是送快递的。", "电话是一三八零零一三八零零零。",
                     "我是老陈，让他回个电话。", "号码是幺三五零零一三五零零零。"]),
        llm=FakeLlm(responder),
        tts=FakeTts(),
        cache=make_cache(config),
        backend="local",
    )


# --------------------------------------------------------------------------- #
# 1) 两条场景跑完，报告与指标齐全
# --------------------------------------------------------------------------- #


async def test_evaluate_two_scenarios_with_fakes(tmp_path):
    first, second = build_two_scenarios(tmp_path)
    config = make_config(tmp_path)
    services = make_services(config)
    scenarios = [load_scenario(first), load_scenario(second)]
    assert all(scenario_playable(scenario) for scenario in scenarios)

    report = await evaluate_scenarios(scenarios, tmp_path / "report", config, services)

    out = tmp_path / "report"
    assert (out / "summary.md").is_file()
    assert (out / "summary.json").is_file()
    markdown = (out / "summary.md").read_text(encoding="utf-8")
    assert "必填槽位准确率" in markdown
    assert "CONFIRM 轮端到端" in markdown
    assert "场景 × 槽位（实际值 vs 期望值）" in markdown
    assert "| s01_delivery | name |" in markdown

    aggregate_data = report["aggregate"]
    assert aggregate_data["scenarios"] == 2
    # 两条场景的姓名、来意、号码都应命中
    required = aggregate_data["required_slot_accuracy"]
    assert required["total"] == 6
    assert required["correct"] == 6
    assert aggregate_data["parse"]["failure_rate"] == 0.0
    assert aggregate_data["cache"]["requests"] > 0
    assert aggregate_data["cache"]["hits"] > 0
    assert aggregate_data["latency_ms"]["e2e"]["n"] > 0

    by_id = {result["id"]: result for result in report["scenarios"]}
    assert by_id["s01_delivery"]["slot_hits"] == {
        "name": True,
        "purpose": True,
        "callback_number": True,
    }
    assert by_id["s01_delivery"]["slot_values"]["callback_number"]["value"] == "13800138000"
    assert by_id["s02_message"]["slot_hits"]["name"] is True

    # 双声道 call.wav：左声道来电者、右声道 AI
    for scenario_id in ("s01_delivery", "s02_message"):
        with wave.open(str(out / "calls" / scenario_id / "call.wav"), "rb") as reader:
            assert reader.getnchannels() == 2
            assert reader.getframerate() == SAMPLE_RATE
            assert reader.getnframes() > 0


async def test_evaluate_repeat_reports_mean_and_range(tmp_path):
    first, _second = build_two_scenarios(tmp_path)
    config = make_config(tmp_path)
    services = make_services(config)
    # 同一条场景跑两次，fake ASR 的文本要够两遍用
    services.asr = FakeAsr(
        [
            "你好，我找张先生。",
            "我姓李，是送快递的。",
            "电话是一三八零零一三八零零零。",
        ]
        * 2
    )
    report = await evaluate_scenarios(
        [load_scenario(first)], tmp_path / "report", config, services, repeat=2
    )
    assert report["repeat"] == 2
    assert len(report["scenarios"]) == 2
    assert {result["run"] for result in report["scenarios"]} == {1, 2}
    accuracy_runs = report["aggregate"]["accuracy_runs"]
    assert len(accuracy_runs["runs"]) == 2
    assert accuracy_runs["required"]["mean"] == 1.0
    markdown = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "重复 2 次" in markdown
    assert "各次运行准确率" in markdown
    assert (tmp_path / "report" / "calls" / "run_01" / "s01_delivery").is_dir()
    assert (tmp_path / "report" / "calls" / "run_02" / "s01_delivery").is_dir()


async def test_evaluate_reports_slot_miss(tmp_path):
    first, _second = build_two_scenarios(tmp_path)
    config = make_config(tmp_path)
    services = make_services(config)
    services.llm = FakeLlm(lambda text: ("", {"name": "王", "purpose": "别的"}))
    report = await evaluate_scenarios(
        [load_scenario(first)], tmp_path / "report", config, services
    )
    result = report["scenarios"][0]
    assert result["slot_hits"]["name"] is False
    assert result["slot_hits"]["purpose"] is False
    assert report["aggregate"]["required_slot_accuracy"]["correct"] == 1  # 只有号码对
    assert report["aggregate"]["required_slot_accuracy"]["total"] == 3


# --------------------------------------------------------------------------- #
# 2) 时延拆分与 CONFIRM 轮
# --------------------------------------------------------------------------- #


def _event(t: float, type_: str, **data) -> dict:
    return {"t": t, "type": type_, "data": data}


def test_extract_turn_metrics_splits_latency_and_confirm():
    from core.phrases import PHRASES

    events = [
        _event(10, "call_start"),
        # 非 CONFIRM 轮
        _event(100, "vad_end"),
        _event(105, "turn_end"),
        _event(106, "asr_request"),
        _event(406, "asr_final", text="我姓李", latency_ms=290.0),
        _event(410, "llm_request", state="ASK_IDENTITY"),
        _event(1610, "llm_done", total_ms=1180.0, parse_ok=True),
        _event(1612, "tts_request", cached=True),
        _event(1612, "tts_first_audio", latency_ms=0.0),
        _event(1612, "play_start", text=PHRASES["ask_callback"]),
        _event(2612, "play_end", duration_ms=1000.0),
        # CONFIRM 轮：预合成开头 + 现合成后半句
        _event(3000, "vad_end"),
        _event(3002, "turn_end"),
        _event(3003, "asr_request"),
        _event(3300, "asr_final", text="对", latency_ms=280.0),
        _event(3301, "llm_request", state="CONFIRM"),
        _event(4000, "llm_done", total_ms=680.0, parse_ok=True),
        _event(4001, "tts_request", cached=True),
        _event(4001, "tts_first_audio", latency_ms=0.0),
        _event(4001, "play_start", text=PHRASES["confirm_lead"]),
        _event(5001, "play_end", duration_ms=1000.0),
        _event(5002, "tts_request", cached=False),
        _event(6700, "tts_first_audio", latency_ms=1698.0),
        _event(6700, "play_start", text="跟您确认一下：您是李明，回拨电话是……，对吗？"),
        _event(7700, "play_end", duration_ms=1000.0),
    ]
    turns = extract_turn_metrics(events)
    assert len(turns) == 2
    first, second = turns
    assert first.e2e_ms == pytest.approx(1512.0)
    assert first.turn_end_ms - first.vad_end_ms == 5.0
    assert first.asr_ms == 290.0
    assert first.llm_ms == 1180.0
    assert first.tts_first_ms == 0.0
    assert first.confirm is False

    assert second.confirm is True
    assert second.e2e_ms == pytest.approx(1001.0)  # 从 vad_end 到预合成开头
    assert second.tts_first_ms == 0.0
    # 第二段开头 6700 - (4001 + 1000) = 1699ms
    assert second.lead_gap_ms == pytest.approx(1699.0)


def test_aggregate_handles_empty_and_confirm_buckets():
    results = [
        {
            "slot_hits": {"name": True, "callback_number": False},
            "metrics": [
                {"e2e_ms": 1000.0, "vad_end_ms": 0.0, "turn_end_ms": 5.0,
                 "asr_ms": 200.0, "llm_ms": 300.0, "tts_first_ms": 0.0,
                 "confirm": False, "lead_gap_ms": None},
                {"e2e_ms": 3000.0, "vad_end_ms": 100.0, "turn_end_ms": 105.0,
                 "asr_ms": 250.0, "llm_ms": 600.0, "tts_first_ms": 1500.0,
                 "confirm": True, "lead_gap_ms": 500.0},
            ],
            "cache": {"requests": 3, "hits": 2},
            "parse": {"llm_requests": 2, "parse_failures": 0, "llm_timeouts": 0},
            "summary_ms": None,
        }
    ]
    data = aggregate(results)
    assert data["required_slot_accuracy"]["correct"] == 1
    assert data["required_slot_accuracy"]["total"] == 2
    assert data["latency_ms"]["confirm"]["p50"] == 3000.0
    assert data["latency_ms"]["non_confirm"]["p50"] == 1000.0
    assert data["confirm_lead_gap_ms"]["p50"] == 500.0
    assert data["cache"]["hit_rate"] == round(2 / 3, 4)


def test_render_markdown_splits_original_and_held_out_groups():
    """T12c：summary.md 要把原场景与留出场景的准确率分开列出。"""
    from tools.evaluate import render_markdown

    def result(scenario_id: str, hit: bool) -> dict:
        return {
            "id": scenario_id,
            "run": 1,
            "turns": 1,
            "expected_slots": {"purpose": ["修空调"]},
            "slot_values": {"purpose": {"value": "修空调"}},
            "slot_hits": {"name": False, "purpose": hit, "callback_number": False},
            "metrics": [],
            "cache": {"requests": 1, "hits": 0},
            "parse": {"llm_requests": 1, "parse_failures": 0, "llm_timeouts": 0},
            "summary_ms": None,
        }

    scenarios = [result("s01_delivery", True), result("h01_aircon", False)]
    report = {
        "generated_at": "2026-10-22T00:00:00Z",
        "elapsed_s": 1.0,
        "repeat": 1,
        "config": {"backend": "local", "reply_mode": "minimal", "confirm_lead": False},
        "aggregate": aggregate(scenarios),
        "scenarios": scenarios,
    }
    markdown = render_markdown(report)
    assert "分组准确率（T12c：原场景 vs 留出场景）" in markdown
    assert "原场景 s01–s10" in markdown
    assert "留出场景 h01–h05" in markdown


# --------------------------------------------------------------------------- #
# 3) 槽位比对
# --------------------------------------------------------------------------- #


def test_slot_value_matches_phone_and_text():
    assert slot_value_matches("callback_number", "138-0013-8000", "13800138000")
    assert slot_value_matches("callback_number", "01088886666", ["010-8888-6666"])
    assert not slot_value_matches("callback_number", "13800138001", "13800138000")
    assert slot_value_matches("name", "李先生", "李")
    # T12b：单向包含——期望写法必须在实际值里，反过来不算
    assert slot_value_matches("purpose", "想约张先生谈合同", "谈合同")
    assert not slot_value_matches("purpose", "谈合同", "想约张先生谈合同")
    assert not slot_value_matches("purpose", "推销", "送快递")
    assert not slot_value_matches("name", "", "李")


def test_time_comparison_converts_chinese_numerals():
    """T12b：时间里的汉字数字先转成阿拉伯数字再比。"""
    assert slot_value_matches("time", "明天上午10点", "明天上午十点")
    assert slot_value_matches("time", "后天下午3点", "后天下午三点")
    assert slot_value_matches("time", "十一月二十三号", "11月23号")
    # 单向包含：实际只有“明天”时，期望“明天上午十点”不算对
    assert not slot_value_matches("time", "明天", "明天上午十点")


def test_score_slots_ignores_empty_expectations():
    hits, seen = score_slots(
        {"name": "李", "purpose": "", "callback_number": ["13800138000"]},
        {"name": {"value": "李"}, "callback_number": {"value": "13800138000"}},
    )
    assert hits == {"name": True, "callback_number": True}
    assert "purpose" not in seen


# --------------------------------------------------------------------------- #
# 4) 场景生成（fake TTS）
# --------------------------------------------------------------------------- #


class FakeCallerTts:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synth(self, text, cancel):
        self.texts.append(text)
        yield b"\x00\x00" * 800


async def test_make_scenarios_writes_turns_and_real_placeholders(tmp_path):
    spec = {
        "id": "s_test",
        "desc": "测试场景",
        "expected_slots": {"name": "李"},
        "turns": [
            {"say": "你好。", "gap_ms": 700},
            {"say": [("我想一下……", 300), ("我姓李。", 0)], "gap_ms": 900},
        ],
    }
    tts = FakeCallerTts()
    scenarios_dir = tmp_path / "scenarios"
    assert await write_scenario(spec, scenarios_dir, tts, force=True) is True

    payload = json.loads((scenarios_dir / "s_test.json").read_text(encoding="utf-8"))
    assert payload["id"] == "s_test"
    assert [turn["wav"] for turn in payload["turns"]] == ["turn_01.wav", "turn_02.wav"]
    assert [turn["gap_ms"] for turn in payload["turns"]] == [700.0, 900.0]
    assert payload["turns"][1]["text"] == "我想一下……我姓李。"
    assert (scenarios_dir / "s_test" / "turn_01.wav").is_file()
    # 思考停顿：两个片段之间插入 300ms 静音
    with wave.open(str(scenarios_dir / "s_test" / "turn_02.wav"), "rb") as reader:
        assert reader.getnframes() == (800 + 300 * 16 + 800)

    loaded = load_scenario(scenarios_dir / "s_test.json")
    assert scenario_playable(loaded)
    assert loaded.turns[1].gap_ms == 900.0
    assert loaded.expected_slots == {"name": "李"}

    # 已存在且未 --force 时不再调用 TTS
    before = len(tts.texts)
    assert await write_scenario(spec, scenarios_dir, tts) is False
    assert len(tts.texts) == before

    write_real_placeholders(scenarios_dir)
    for placeholder in REAL_PLACEHOLDERS:
        path = scenarios_dir / f"{placeholder['id']}.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["pending"] is True
        assert data["turns"] == []
    # 占位场景没有 wav，评估会跳过
    assert not scenario_playable(load_scenario(scenarios_dir / "real_01.json"))


def test_turn_text_and_payload_for_plain_string():
    spec = {"id": "x", "turns": [{"say": "你好。", "gap_ms": 800}]}
    assert turn_text(spec["turns"][0]) == "你好。"
    payload = scenario_payload(spec, ["turn_01.wav"])
    assert payload["turns"] == [{"text": "你好。", "wav": "turn_01.wav", "gap_ms": 800.0}]


# --------------------------------------------------------------------------- #
# 5) 直接跑一条场景的输出
# --------------------------------------------------------------------------- #


async def test_replay_scenario_writes_events_slots_and_call_wav(tmp_path):
    first, _ = build_two_scenarios(tmp_path)
    config = make_config(tmp_path)
    services = make_services(config)
    scenario = load_scenario(first)
    call_dir = tmp_path / "call"
    await replay_scenario_to_dir(scenario, services, config, call_dir)

    events = [
        json.loads(line)
        for line in (call_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    types = [event["type"] for event in events]
    assert types[0] == "call_start"
    assert types[-1] == "call_end"
    assert types.count("vad_end") == 3
    assert types.count("play_start") >= 4  # 问候 + 3 轮回复

    slots = json.loads((call_dir / "slots.json").read_text(encoding="utf-8"))["slots"]
    assert slots["name"]["value"] == "李"
    assert slots["purpose"]["value"] == "送快递"
    assert slots["callback_number"]["value"] == "13800138000"

    with wave.open(str(call_dir / "call.wav"), "rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getnframes() > 0
    # 评估一条已完成的通话
    result = evaluate_call(scenario, call_dir)
    assert result["slot_hits"]["callback_number"] is True
    assert result["turns"] == 3

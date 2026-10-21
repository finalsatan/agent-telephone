"""T01 验收测试：core/events.py 的写入格式、时钟单调性、类型白名单与目录创建。"""

from __future__ import annotations

import json

import pytest

from core.events import EVENT_TYPES, EventLogger, monotonic_ms, read_events


def raw_lines(call_dir) -> list[str]:
    return (call_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()


def test_every_line_is_valid_json(tmp_path):
    with EventLogger(tmp_path / "call-abc") as log:
        log.log("call_start", source="wav")
        log.log("asr_final", text="您好，我找张先生", latency_ms=612.5, audio_ms=2480.0)
        log.log("slot_update", slot="name", value="张先生", status="unconfirmed")

    lines = raw_lines(tmp_path / "call-abc")
    assert len(lines) == 3
    for line in lines:
        event = json.loads(line)  # 每行都必须是合法 JSON
        assert set(event) == {"call_id", "t", "type", "data"}
        assert event["call_id"] == "call-abc"
        assert isinstance(event["t"], float)
        assert isinstance(event["data"], dict)


def test_event_fields_carry_data(tmp_path):
    with EventLogger(tmp_path / "c1") as log:
        written = log.log("asr_final", text="您好", latency_ms=600.0)

    event = read_events(tmp_path / "c1")[0]
    assert event == written
    assert event["type"] == "asr_final"
    assert event["data"] == {"text": "您好", "latency_ms": 600.0}
    # ensure_ascii=False：中文直接可读，不写成 \uXXXX
    assert "您好" in raw_lines(tmp_path / "c1")[0]


def test_t_is_non_decreasing_with_rollback_clock(tmp_path):
    ticks = iter([100.0, 90.0, 90.0, 250.0])
    log = EventLogger(tmp_path / "c1", clock=lambda: next(ticks))
    for _ in range(4):
        log.log("vad_start")
    log.close()

    ts = [e["t"] for e in read_events(tmp_path / "c1")]
    assert ts == [100.0, 100.0, 100.0, 250.0]


def test_t_uses_monotonic_ms_by_default(tmp_path):
    before = monotonic_ms()
    with EventLogger(tmp_path / "c1") as log:
        log.log("call_start")
    after = monotonic_ms()

    t = read_events(tmp_path / "c1")[0]["t"]
    assert before <= t <= after


def test_unknown_type_raises_and_writes_nothing(tmp_path):
    with EventLogger(tmp_path / "c1") as log:
        with pytest.raises(ValueError, match="unknown event type"):
            log.log("asr_finshed", text="打错了")
        assert raw_lines(tmp_path / "c1") == []


@pytest.mark.parametrize("event_type", sorted(EVENT_TYPES))
def test_all_whitelisted_types_accepted(tmp_path, event_type):
    with EventLogger(tmp_path / "c1") as log:
        log.log(event_type)
    assert read_events(tmp_path / "c1")[0]["type"] == event_type


def test_call_dir_created_recursively(tmp_path):
    call_dir = tmp_path / "calls" / "20261021-153000-deadbeef"
    assert not call_dir.exists()
    with EventLogger(call_dir) as log:
        log.log("call_start")
    assert call_dir.is_dir()
    assert log.call_id == "20261021-153000-deadbeef"


def test_writes_are_flushed_immediately(tmp_path):
    call_dir = tmp_path / "c1"
    log = EventLogger(call_dir)
    log.log("call_start")
    # 不 close，另开句柄读：内容必须已经落盘
    with (call_dir / "events.jsonl").open(encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == 1
    log.close()


def test_appends_across_instances(tmp_path):
    with EventLogger(tmp_path / "c1") as log:
        log.log("call_start")
    with EventLogger(tmp_path / "c1") as log:
        log.log("call_end")

    assert [e["type"] for e in read_events(tmp_path / "c1")] == ["call_start", "call_end"]

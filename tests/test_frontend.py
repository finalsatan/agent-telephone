"""T14 验收：测试页面前端。

覆盖：

- `GET /` 引用 `/static/app.js`、`/static/style.css` 与各面板元素；
- `/static/*` 能取到 app.js / style.css / pcm-worklet.js；
- `/api/scenarios` 带 `expected_slots` 与每轮 `index`（批跑算准确率、实时回放取音频要用）；
- `/api/scenarios/{id}/turns/{index}/audio` 能取到 wav（含跨目录引用的 barge_in），越界返回 404；
- 前端纯逻辑（时延拆分、transcript 合并、槽位准确率、重采样）的 Node 单测。

浏览器里的麦克风/播放/回声消除属于手动验收，见 docs/progress.md。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools.replay import ROOT
from web.server import create_app


def make_page_client(tmp_path: Path) -> TestClient:
    """真实 `web/static` + `scenarios/`，但通话写到临时目录（不启动任何服务）。"""
    app = create_app(
        root=ROOT,
        calls_dir=tmp_path / "calls",
        scenarios_dir=ROOT / "scenarios",
    )
    return TestClient(app)


def test_index_references_frontend_assets_and_panels(tmp_path: Path) -> None:
    client = make_page_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "/static/app.js" in response.text
    assert "/static/style.css" in response.text
    for element in (
        "btn-start",
        "btn-hangup",
        "btn-takeover",
        "ai-status",
        "dialog-state",
        "dialog-caller",
        "dialog-ai",
        "slots-body",
        "latency-body",
        "latency-summary",
        "param-channel-sim",
        "param-interrupt",
        "calls-list",
        "batch-table",
        "echo-cancel",
    ):
        assert f'id="{element}"' in response.text, f"缺少元素 {element}"


def test_static_assets_served(tmp_path: Path) -> None:
    client = make_page_client(tmp_path)
    for name, marker in (
        ("app.js", "buildTurnRows"),
        ("style.css", "dialog-list"),
        ("pcm-worklet.js", "registerProcessor"),
    ):
        response = client.get(f"/static/{name}")
        assert response.status_code == 200, name
        assert marker in response.text, name


def test_scenarios_include_expected_slots_and_turn_index(tmp_path: Path) -> None:
    client = make_page_client(tmp_path)
    payload = client.get("/api/scenarios").json()
    scenario = next(item for item in payload["scenarios"] if item["id"] == "s03_appointment")
    assert scenario["expected_slots"]["callback_number"] == "13700137000"
    assert [turn["index"] for turn in scenario["turns"]] == [0, 1, 2]
    assert scenario["turns"][0]["gap_ms"] == 800.0
    assert scenario["playable"] is True


def test_scenario_turn_audio_endpoint(tmp_path: Path) -> None:
    client = make_page_client(tmp_path)
    response = client.get("/api/scenarios/s03_appointment/turns/0/audio")
    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"

    assert client.get("/api/scenarios/s03_appointment/turns/99/audio").status_code == 404
    assert client.get("/api/scenarios/nope/turns/0/audio").status_code == 404


def test_frontend_logic_node_tests() -> None:
    """用 Node 自带 test runner 跑 app.js 的纯函数与 worklet 的重采样。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不在 PATH 上，跳过前端纯逻辑单测")
    js_dir = ROOT / "tests" / "js"
    result = subprocess.run(
        [node, "--test", str(js_dir / "*.test.js")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""T15：启动 Pipecat 版测试页面后端。

用法（与 T13 的 `web.server` 并列，端口不同，页面和 WebSocket 协议相同）：

    uv run python -m adapters.pipecat.run

    # 另开终端，用场景回放跑一条（与 T13 的 ws_client 完全相同）
    uv run python tools/ws_client.py --scenario scenarios/s03_appointment.json \
        --url ws://127.0.0.1:8001/ws/call

页面打开 http://127.0.0.1:8001/ 。`config.yaml: web.adapter` 也支持把 Pipecat 链路
挂到默认端口 8000（把 adapter 改成 pipecat 后用 `uv run python -m web.server` 启动）。
"""

from __future__ import annotations

from pathlib import Path


def build_app(*, port: int | None = None):
    """构造启用 Pipecat 适配器的 FastAPI 应用（不联网，只改适配器选择）。"""
    from web.server import ROOT, create_app, load_config

    config = load_config(ROOT / "config.yaml")
    web_cfg = config.setdefault("web", {})
    web_cfg["adapter"] = "pipecat"
    if port is not None:
        web_cfg["port"] = port
    return create_app(config=config)


def main() -> None:  # pragma: no cover - 手动启动用
    import uvicorn

    from web.server import ROOT, load_config

    config = load_config(ROOT / "config.yaml")
    host = str((config.get("web") or {}).get("host") or "127.0.0.1")
    # 默认换到 8001，避免和 T13 的 8000 冲突。
    port = int((config.get("web") or {}).get("pipecat_port") or 8001)
    app = build_app(port=port)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()

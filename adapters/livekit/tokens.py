"""T16：自托管 LiveKit 的房间令牌。

密钥只从环境变量读（AGENTS 规则 3）：`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`。
`livekit-server --dev` 的默认凭据见 `docs/livekit-notes.md`，不进代码与配置。

浏览器与后端各拿一张令牌、进同一个房间：浏览器发布麦克风音轨，后端（`LiveKitCall`）
发布 AI 音轨并订阅来电者音轨。
"""

from __future__ import annotations

import datetime
import os

DEFAULT_TTL_S = 3600


def api_credentials() -> tuple[str, str]:
    key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "缺少 LIVEKIT_API_KEY / LIVEKIT_API_SECRET（自托管 livekit-server 的密钥）"
        )
    return key, secret


def build_token(
    *,
    room: str,
    identity: str,
    ttl_s: int = DEFAULT_TTL_S,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> str:
    """签发一张只能加入 `room` 的令牌。"""
    from livekit import api

    if api_key is None or api_secret is None:
        api_key, api_secret = api_credentials()
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_ttl(datetime.timedelta(seconds=int(ttl_s)))
        .to_jwt()
    )


def new_room_name(prefix: str, call_id: str) -> str:
    return f"{prefix}-{call_id}"


__all__ = ["build_token", "api_credentials", "new_room_name", "DEFAULT_TTL_S"]

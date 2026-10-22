"""本地 CosyVoice3 TTS 客户端（T05）。

接口与 `services.tts.TtsClient` 完全一致（0.3 的 `synth(text, cancel)` / `sample_rate`），
只是把请求换成本地服务的 `POST /synth`（body `{"text": "..."}`，返回裸 16kHz PCM）。
流式读取、首包/总时限、取消与重试边界全部复用 T04/T04b 的实现，避免两套语义：

- 超时范围只包住单次网络等待，`yield` 在超时范围之外；
- cancel 置位后立即结束，退避期间被打断也不泄漏内部异常；
- 未产出音频前才重试 429/5xx/传输错误。

本地服务默认监听 `http://127.0.0.1:50000`，地址取自 `config.yaml: tts.local_url`。
"""

from __future__ import annotations

from typing import Any

from services.tts import DEFAULT_SAMPLE_RATE, TtsClient

DEFAULT_LOCAL_URL = "http://127.0.0.1:50000"


class LocalTtsClient(TtsClient):
    """本地 CosyVoice3 服务的客户端。本地服务不需要鉴权，因此不发 Authorization 头。"""

    endpoint = "/synth"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LOCAL_URL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs: Any,
    ) -> None:
        # api_key="" 是 T04 预留的“显式无鉴权”路径；本地服务不校验密钥。
        super().__init__(
            base_url=base_url,
            model="",
            voice="",
            sample_rate=sample_rate,
            api_key="",
            **kwargs,
        )

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "LocalTtsClient":
        tts_cfg = config.get("tts") or {}
        audio_cfg = config.get("audio") or {}
        kwargs.setdefault("sample_rate", audio_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
        kwargs.setdefault("base_url", tts_cfg.get("local_url", DEFAULT_LOCAL_URL))
        return cls(**kwargs)

    def _build_payload(self, text: str) -> dict[str, Any]:
        return {"text": text}

"""T16：满足框架前置检查的占位 LLM / TTS。

本项目的文本与音频分别由 `CallAgent.llm_node()` / `CallAgent.tts_node()` 产出，
不走 LiveKit 的 `llm.LLM` / `tts.TTS` 抽象（固定 JSON 的 LLM 与带话术缓存的 TTS
都对不上它们的假设，见 notes 1.2）。但 `AgentSession` 有两处硬性检查：

- `AgentActivity.on_end_of_turn()` 里 `elif self.llm is None: return`——没有 LLM 就
  不生成回复，`on_user_turn_completed()` 白跑；
- `AgentActivity.say()` 里没有 `tts` 就抛 `RuntimeError`，问候语与兜底话术都播不了。

所以挂两个占位实现：只提供 `model` / `provider` 标签（框架的指标会用到）与采样率，
真正的 `chat()` / `synthesize()` / `stream()` 永远不会被调到。
"""

from __future__ import annotations

from typing import Any

from livekit.agents import llm as lk_llm
from livekit.agents import tts as lk_tts


class NodeLLM(lk_llm.LLM):
    """占位 LLM：文本由 `CallAgent.llm_node()` 产出。"""

    def __init__(self, *, model: str | None = None) -> None:
        super().__init__()
        self._model = model or "unknown"

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "siliconflow"

    def chat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("文本由 CallAgent.llm_node() 产出")


class NodeTTS(lk_tts.TTS):
    """占位 TTS：音频由 `CallAgent.tts_node()` 产出。"""

    def __init__(self, *, model: str | None = None, sample_rate: int = 16000) -> None:
        super().__init__(
            capabilities=lk_tts.TTSCapabilities(streaming=False),
            sample_rate=int(sample_rate),
            num_channels=1,
        )
        self._model = model or "unknown"

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "local"

    def synthesize(self, text: str, **kwargs: Any) -> Any:
        raise NotImplementedError("音频由 CallAgent.tts_node() 产出")

    def stream(self, **kwargs: Any) -> Any:
        raise NotImplementedError("音频由 CallAgent.tts_node() 产出")


__all__ = ["NodeLLM", "NodeTTS"]

"""LLM 服务：硅基流动的 OpenAI 兼容接口（T02）。

- `turn()`：一次对话轮，流式读取，记录首 token 与总耗时，解析成 TurnResult。
- `complete()`：非流式，用于摘要等不在实时链路上的调用。
- `parse_turn_output()`：纯函数，不碰网络，便于离线测试与 T10 复用。
- 对 Qwen3 系列必须显式 `enable_thinking: false`，否则首 token 要等到思考结束（M1 实测 12.7s）。
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.events import monotonic_ms
from core.prompts import build_messages
from core.types import SLOT_NAMES, SlotValue, State, TurnResult

API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_S = 0.5


class LlmError(RuntimeError):
    """LLM 调用失败：重试用尽、响应不可用或缺少密钥。"""


# --------------------------------------------------------------------------- #
# 解析（纯逻辑）
# --------------------------------------------------------------------------- #


def _first_json_object(raw: str) -> dict[str, Any] | None:
    """找到文本中第一个能解析成对象的 JSON。

    从头逐个 '{' 尝试 raw_decode，因此代码围栏、前后多余解释文字都不影响；
    嵌套对象由 raw_decode 一次性吃掉，不会切错。
    从 '{' 起解码成功必然是 dict，故无需再判类型。
    """
    decoder = json.JSONDecoder()
    idx = raw.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            idx = raw.find("{", idx + 1)
            continue
        return obj
    return None


def parse_turn_output(raw: str) -> tuple[str, dict[str, str], bool]:
    """把模型原始输出解析成 (reply, slots_update, parse_ok)。

    判定规则：
    - 找不到 JSON 对象、或 `reply` 缺失/不是非空字符串 → parse_ok=False，reply 为空，
      由调用方使用该状态的兜底话术。
    - `slots` 缺失或不是对象 → 视为本轮没有槽位更新，但保留 parse_ok=True。
    - `slots` 里非白名单的键、非字符串的值、空字符串，一律丢弃（不猜测、不编造）。
    """
    obj = _first_json_object(raw)
    if obj is None:
        return "", {}, False

    reply = obj.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        return "", {}, False

    slots_update: dict[str, str] = {}
    slots = obj.get("slots")
    if isinstance(slots, dict):
        for name, value in slots.items():
            if name not in SLOT_NAMES or not isinstance(value, str):
                continue
            value = value.strip()
            if value:
                slots_update[name] = value

    return reply.strip(), slots_update, True


# --------------------------------------------------------------------------- #
# 流式读取辅助
# --------------------------------------------------------------------------- #


class _HttpStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _HttpStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return isinstance(exc, httpx.TransportError)


async def _iter_sse_content(response: httpx.Response) -> AsyncIterator[str]:
    """从 SSE 流里取出 delta.content。忽略心跳、role-only 帧与 [DONE]。"""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            yield content


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #


class LlmClient:
    """实现 0.3 的 LLM 接口。api_key 缺省从环境变量 SILICONFLOW_API_KEY 读取。"""

    def __init__(
        self,
        *,
        base_url: str,
        realtime_model: str,
        summary_model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise LlmError(
                f"missing API key: set the {API_KEY_ENV} environment variable "
                "(keys must never be hardcoded or committed)"
            )
        self.realtime_model = realtime_model
        self.summary_model = summary_model or realtime_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "LlmClient":
        """用 config.yaml 的内容构造（tools/ 与 tests/live/ 用）。"""
        return cls(
            base_url=config["siliconflow"]["base_url"],
            realtime_model=config["llm"]["realtime_model"],
            summary_model=config["llm"].get("summary_model"),
            **kwargs,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LlmClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- 实时链路 ---------------------------------------------------------- #

    async def turn(
        self,
        state: State,
        slots: dict[str, SlotValue],
        history: list[dict],
        user_text: str,
    ) -> TurnResult:
        messages = build_messages(state, slots, history, user_text)
        start = monotonic_ms()
        first_token_ms: float | None = None
        chunks: list[str] = []

        async for delta in self._stream_chat(messages, self.realtime_model):
            if first_token_ms is None and delta.strip():
                first_token_ms = monotonic_ms() - start
            chunks.append(delta)

        total_ms = monotonic_ms() - start
        reply, slots_update, parse_ok = parse_turn_output("".join(chunks))
        return TurnResult(
            reply=reply,
            slots_update=slots_update,
            first_token_ms=total_ms if first_token_ms is None else first_token_ms,
            total_ms=total_ms,
            parse_ok=parse_ok,
        )

    async def _stream_chat(self, messages: list[dict], model: str) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "enable_thinking": False,  # Qwen3 系列必须关闭，否则首 token 不可接受
        }
        attempt = 0
        while True:
            emitted = False
            try:
                async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")[:500]
                        raise _HttpStatusError(response.status_code, body)
                    async for delta in _iter_sse_content(response):
                        emitted = True
                        yield delta
                return
            except (_HttpStatusError, httpx.TransportError) as exc:
                # 已经产出内容的流不再重试：重发会重复输出，且调用方已可能开始播报。
                if emitted or not _is_retryable(exc) or attempt >= self.max_retries:
                    raise LlmError(f"llm stream failed: {exc}") from exc
                await asyncio.sleep(self.retry_base_delay * (2**attempt))
                attempt += 1

    # -- 非实时链路 -------------------------------------------------------- #

    async def complete(self, messages: list[dict], model: str | None = None) -> str:
        """非流式补全，返回文本内容（摘要、校正等用途）。"""
        payload = {
            "model": model or self.summary_model,
            "messages": messages,
            "stream": False,
            "enable_thinking": False,
        }
        data = await self._post_json(payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"unexpected completion payload: {str(data)[:300]}") from exc

    async def _post_json(self, payload: dict) -> dict:
        attempt = 0
        while True:
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code >= 400:
                    raise _HttpStatusError(response.status_code, response.text[:500])
                return response.json()
            except (_HttpStatusError, httpx.TransportError) as exc:
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    raise LlmError(f"llm request failed: {exc}") from exc
                await asyncio.sleep(self.retry_base_delay * (2**attempt))
                attempt += 1

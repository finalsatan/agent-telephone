"""LLM 服务：硅基流动的 OpenAI 兼容接口（T02 / T02b）。

- `turn_stream()`：实时链路用，流式读 SSE，边收边把 reply 按短句产出，流结束产出 done。
- `turn()`：消费 `turn_stream()`，返回最终 TurnResult。
- `complete()`：非流式，用于摘要等不在实时链路上的调用。
- `parse_turn_output()`：纯函数，不碰网络，便于离线测试与 T10 复用。
- 两层截止时间用 `asyncio.timeout` 实现：首 token 超时与整轮总时长超时（T02b）。
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
from core.prompts import DEFAULT_OWNER_NAME, build_messages
from core.types import SLOT_NAMES, ReplyChunk, SlotValue, State, TurnResult

API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_S = 0.5
DEFAULT_FIRST_TOKEN_TIMEOUT_S = 2.5
DEFAULT_TOTAL_TIMEOUT_S = 8.0
DEFAULT_MAX_TOKENS = 150

# reply 的短句切分：句号类标点与换行。分号也算句末，因为 TTS 每句都要能独立播。
SENTENCE_TERMINATORS = "。！？；"


class LlmError(RuntimeError):
    """LLM 调用失败：重试用尽、响应不可用或缺少密钥。"""


class LlmTimeout(LlmError):
    """首 token 或整轮截止时间触发（T02b）。不重试，由调用方播兜底话术。"""



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
    - `callback_number` 一律丢弃：号码以 ASR 原文 + 规则校验（T03/T07）为准，
      实测模型会把提示语当值回传。
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
            # callback_number 不从 LLM 取：设计文档第 4 节要求号码以 ASR 原文 + 规则校验为准。
            # 实测 Qwen3-8B 会把提示语（如“请说一遍您的回拨号码”）当值回传，不可信。
            if name not in SLOT_NAMES or name == "callback_number" or not isinstance(value, str):
                continue
            value = value.strip()
            if value:
                slots_update[name] = value

    return reply.strip(), slots_update, True


# --------------------------------------------------------------------------- #
# reply 短句切分（流式增量与结束后兜底共用同一规则）
# --------------------------------------------------------------------------- #

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def split_reply_sentences(text: str) -> list[str]:
    """把一段 reply 按短句切开。标点保留在句末（换行不保留），空白句丢弃。

    流式增量识别失败（例如 slots 在 reply 之前）时，用它把完整 reply 切句兜底。
    """
    sentences: list[str] = []
    pending: list[str] = []
    for ch in text:
        if ch == "\n":
            s = "".join(pending).strip()
            pending = []
            if s:
                sentences.append(s)
        elif ch in SENTENCE_TERMINATORS:
            s = "".join(pending).strip()
            pending = []
            if s:
                sentences.append(s + ch)
        else:
            pending.append(ch)
    s = "".join(pending).strip()
    if s:
        sentences.append(s)
    return sentences


class _ReplyExtractor:
    """从流式 JSON 文本中增量取出 reply 字符串，并按短句产出。

    只接受 reply 作为 JSON 对象的第一个键（提示词已要求）。这样能避免把 slots 值里
    恰好出现的 "reply" 误当成键；若模型输出 slots 在前则直接放弃（abandoned=True），
    由调用方在流结束后用完整解析结果兜底，不报错、不重复播报。

    JSON 转义支持 \\" \\\\ / \\b \\f \\n \\r \\t 与 \\uXXXX；转义序列可能跨 SSE 块，
    因此按字符消费并在长度不足时等待下一块。
    """

    _KEY = '"reply"'

    def __init__(self) -> None:
        self._buf = ""
        self._pos = 0
        self._state = "seek_object"  # seek_object | seek_key | seek_colon | seek_value | in_string | done
        self._pending: list[str] = []
        self.decoded = ""
        self.complete = False
        self.abandoned = False

    def feed(self, chunk: str) -> list[str]:
        """喂入新收到的文本，返回本次可产出的完整短句。"""
        self._buf += chunk
        out: list[str] = []
        while self._state != "done" and self._pos < len(self._buf):
            if self._state == "seek_object":
                idx = self._buf.find("{", self._pos)
                if idx == -1:
                    self._pos = len(self._buf)
                    break
                self._pos = idx + 1
                self._state = "seek_key"
                continue

            if self._state == "seek_key":
                while self._pos < len(self._buf) and self._buf[self._pos].isspace():
                    self._pos += 1
                if len(self._buf) - self._pos < len(self._KEY):
                    break  # 可能是半个键，等下一块
                if not self._buf.startswith(self._KEY, self._pos):
                    self.abandoned = True
                    self._state = "done"
                    break
                self._pos += len(self._KEY)
                self._state = "seek_colon"
                continue

            if self._state == "seek_colon":
                while self._pos < len(self._buf) and self._buf[self._pos].isspace():
                    self._pos += 1
                if self._pos >= len(self._buf):
                    break
                if self._buf[self._pos] != ":":
                    self.abandoned = True
                    self._state = "done"
                    break
                self._pos += 1
                self._state = "seek_value"
                continue

            if self._state == "seek_value":
                # 冒号与值的引号可能被 SSE 块切开，必须把两个步骤拆成两个状态；
                # 合并时一旦在“跳空格”处 break，下一块会重新期待冒号而误判放弃。
                while self._pos < len(self._buf) and self._buf[self._pos].isspace():
                    self._pos += 1
                if self._pos >= len(self._buf):
                    break
                if self._buf[self._pos] != '"':
                    self.abandoned = True
                    self._state = "done"
                    break
                self._pos += 1
                self._state = "in_string"
                continue

            # in_string
            ch = self._buf[self._pos]
            if ch == "\\":
                if self._pos + 1 >= len(self._buf):
                    break
                esc = self._buf[self._pos + 1]
                if esc == "u":
                    if self._pos + 6 > len(self._buf):
                        break
                    code = self._buf[self._pos + 2 : self._pos + 6]
                    try:
                        decoded = chr(int(code, 16))
                    except ValueError:
                        decoded = "\\u" + code
                    self._pos += 6
                else:
                    decoded = _ESCAPES.get(esc, esc)
                    self._pos += 2
                out.extend(self._push(decoded))
                continue
            if ch == '"':
                self._pos += 1
                self._state = "done"
                self.complete = True
                out.extend(self._flush())
                break
            self._pos += 1
            out.extend(self._push(ch))
        return out

    def _push(self, ch: str) -> list[str]:
        self.decoded += ch
        if ch in SENTENCE_TERMINATORS or ch == "\n":
            text = "".join(self._pending).strip()
            self._pending = []
            if text and ch != "\n":
                text += ch
            return [text] if text else []
        self._pending.append(ch)
        return []

    def _flush(self) -> list[str]:
        text = "".join(self._pending).strip()
        self._pending = []
        return [text] if text else []


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
        first_token_timeout_s: float = DEFAULT_FIRST_TOKEN_TIMEOUT_S,
        total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        owner_name: str = DEFAULT_OWNER_NAME,
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
        self.first_token_timeout_s = first_token_timeout_s
        self.total_timeout_s = total_timeout_s
        self.max_tokens = max_tokens
        self.owner_name = owner_name
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "LlmClient":
        """用 config.yaml 的内容构造（tools/ 与 tests/live/ 用）。

        截止时间、max_tokens、机主称呼都从 config 读取（0.6、T02b）；
        显式传入的 kwargs 优先，便于压测/测试临时覆盖。
        """
        llm_cfg = config.get("llm") or {}
        owner_cfg = config.get("owner") or {}
        kwargs.setdefault(
            "first_token_timeout_s",
            llm_cfg.get("first_token_timeout_s", DEFAULT_FIRST_TOKEN_TIMEOUT_S),
        )
        kwargs.setdefault("total_timeout_s", llm_cfg.get("total_timeout_s", DEFAULT_TOTAL_TIMEOUT_S))
        kwargs.setdefault("max_tokens", llm_cfg.get("max_tokens", DEFAULT_MAX_TOKENS))
        kwargs.setdefault("owner_name", owner_cfg.get("name", DEFAULT_OWNER_NAME))
        return cls(
            base_url=config["siliconflow"]["base_url"],
            realtime_model=llm_cfg["realtime_model"],
            summary_model=llm_cfg.get("summary_model"),
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
        attempts: dict[str, int] | None = None,
    ) -> TurnResult:
        """消费 turn_stream()，返回 done 中的结果。超时会抛 LlmTimeout。"""
        result: TurnResult | None = None
        async for chunk in self.turn_stream(state, slots, history, user_text, attempts):
            if chunk.kind == "done":
                result = chunk.result
        if result is None:  # 正常结束必有 done；只有超时才会没有，而超时会抛异常
            raise LlmError("llm turn produced no done chunk")
        return result

    async def turn_stream(
        self,
        state: State,
        slots: dict[str, SlotValue],
        history: list[dict],
        user_text: str,
        attempts: dict[str, int] | None = None,
    ) -> AsyncIterator[ReplyChunk]:
        """实时链路：边收边产出 reply 的完整短句，最后产出 done。

        截止时间（T02b）：
        - 收到第一段非空内容超过 first_token_timeout_s → LlmTimeout；
        - 整轮超过 total_timeout_s → LlmTimeout。已产出的句子保持有效，done 不再产出。
        截止时间触发时不重试，由调用方播兜底话术。
        """
        messages = build_messages(
            state,
            slots,
            history,
            user_text,
            attempts=attempts,
            owner_name=self.owner_name,
        )
        start = monotonic_ms()
        first_token_ms: float | None = None
        first_sentence_ms: float | None = None
        raw_parts: list[str] = []
        extractor = _ReplyExtractor()
        sentences_emitted = 0

        stream = self._stream_chat(messages)
        try:
            async with asyncio.timeout(self.total_timeout_s) as total_scope:
                while True:
                    try:
                        if first_token_ms is None:
                            async with asyncio.timeout(self.first_token_timeout_s):
                                delta = await anext(stream)
                        else:
                            delta = await anext(stream)
                    except StopAsyncIteration:
                        break
                    except TimeoutError as exc:
                        if total_scope.expired():
                            raise  # 交给外层统一报总时长超时
                        raise LlmTimeout(
                            "no content within "
                            f"{self.first_token_timeout_s}s (first-token timeout)"
                        ) from exc

                    if first_token_ms is None and delta.strip():
                        first_token_ms = monotonic_ms() - start
                    raw_parts.append(delta)
                    for sentence in extractor.feed(delta):
                        if first_sentence_ms is None:
                            first_sentence_ms = monotonic_ms() - start
                        sentences_emitted += 1
                        yield ReplyChunk(
                            kind="sentence",
                            text=sentence,
                            t_ms=monotonic_ms() - start,
                        )
        except TimeoutError as exc:
            raise LlmTimeout(
                f"turn exceeded {self.total_timeout_s}s (total timeout); "
                f"emitted {sentences_emitted} sentence(s), no done chunk"
            ) from exc
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不应盖住原始异常
                pass

        total_ms = monotonic_ms() - start
        reply, slots_update, parse_ok = parse_turn_output("".join(raw_parts))
        if parse_ok and sentences_emitted == 0:
            # 模型没把 reply 放在第一个键（或增量识别失败）：不报错，整段切句兜底。
            # 首句在流结束时才可用，因此耗时记 total_ms。
            for sentence in split_reply_sentences(reply):
                if first_sentence_ms is None:
                    first_sentence_ms = total_ms
                sentences_emitted += 1
                yield ReplyChunk(
                    kind="sentence",
                    text=sentence,
                    t_ms=monotonic_ms() - start,
                )

        result = TurnResult(
            reply=reply,
            slots_update=slots_update,
            first_token_ms=total_ms if first_token_ms is None else first_token_ms,
            total_ms=total_ms,
            parse_ok=parse_ok,
            first_sentence_ms=first_sentence_ms,
        )
        yield ReplyChunk(kind="done", result=result, t_ms=monotonic_ms() - start)

    async def _stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.realtime_model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
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

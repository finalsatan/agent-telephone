"""TTS 服务：硅基流动 CosyVoice2 的流式 PCM 合成（T04）。

接口按《开发任务拆分》0.3：`synth(text, cancel)` 异步迭代出 16kHz、单声道、
16-bit 小端 PCM 块；调用方把块直接送播放器或缓存。

关键约束（与 T02c 同一套做法）：
- 首包超时与总时限用绝对截止时刻 + `asyncio.timeout_at()`，**只包住对网络的单次等待**，
  `yield` 全部在超时范围之外。消费者在两块之间 await（播报、写文件）不会被本模块取消，
  也不会被误计入 TTS 的等待时间；`total_timeout_s` 是墙钟语义，超过之后下一次取块立即超时。
- `cancel` 事件被置位时立即关闭连接并停止产出，不抛异常（打断时调用方只想静默收尾）。
- 已经开始输出的请求不重试：重发会让调用方把同一句话播两遍。
- 空文本（含纯空白）直接返回，不发请求。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Coroutine
from typing import Any

import httpx

API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_HTTP_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_S = 0.5
# T04 实测（50 次）：首包 p50 约 4.1s、p95 约 5.8s，比设计文档 16.2 的 2.5s 更慢。
# 首包上限按实测 p95 再留余量取 10s；T10 的响应时间靠 T06 预合成与兜底话术控制，
# 不指望这个超时值压低延迟（设得过紧只会把本来能播的句子丢掉）。
DEFAULT_FIRST_CHUNK_TIMEOUT_S = 10.0
# 总时限是整段音频生成完的墙钟上限；长句的音频本身可能就有十几秒。
DEFAULT_TOTAL_TIMEOUT_S = 30.0
DEFAULT_SAMPLE_RATE = 16000


class TtsError(RuntimeError):
    """TTS 调用失败：重试用尽、响应不可用或缺少密钥。"""


class TtsTimeout(TtsError):
    """首包或总时限触发。不重试，由调用方播兜底话术或静默跳过。"""


class _Cancelled(Exception):
    """内部信号：cancel 事件被置位，正常停止。"""


class _HttpStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _HttpStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return isinstance(exc, httpx.TransportError)


class TtsClient:
    """实现 0.3 的 TTS 接口。api_key 缺省从环境变量 SILICONFLOW_API_KEY 读取。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        voice: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        api_key: str | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY_S,
        first_chunk_timeout_s: float = DEFAULT_FIRST_CHUNK_TIMEOUT_S,
        total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        if not api_key:
            raise TtsError(
                f"missing API key: set the {API_KEY_ENV} environment variable "
                "(keys must never be hardcoded or committed)"
            )
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.first_chunk_timeout_s = first_chunk_timeout_s
        self.total_timeout_s = total_timeout_s
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "TtsClient":
        """用 config.yaml 的内容构造（tools/ 与 tests/live/ 用）。

        模型、音色、采样率、两个截止时间都从 config 读取；显式传入的 kwargs 优先，
        便于压测/测试临时覆盖（live 用例用更宽松的截止时间抵消公网抖动）。
        """
        tts_cfg = config.get("tts") or {}
        audio_cfg = config.get("audio") or {}
        kwargs.setdefault(
            "first_chunk_timeout_s",
            tts_cfg.get("first_chunk_timeout_s", DEFAULT_FIRST_CHUNK_TIMEOUT_S),
        )
        kwargs.setdefault("total_timeout_s", tts_cfg.get("total_timeout_s", DEFAULT_TOTAL_TIMEOUT_S))
        kwargs.setdefault("sample_rate", audio_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
        return cls(
            base_url=config["siliconflow"]["base_url"],
            model=tts_cfg["model"],
            voice=tts_cfg["voice"],
            **kwargs,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "TtsClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- 合成 -------------------------------------------------------------- #

    async def synth(self, text: str, cancel: asyncio.Event) -> AsyncIterator[bytes]:
        """流式合成，按到达顺序产出 PCM 块。

        - 空文本/纯空白：直接返回。
        - cancel 置位：安静结束，不抛异常。
        - 首包超时 → TtsTimeout("first-chunk")；总时限 → TtsTimeout("total")。
        - 429/5xx/传输错误在**尚未产出任何音频**时按指数退避重试，最多 max_retries 次。
        """
        if not text or not text.strip():
            return
        if cancel.is_set():
            return

        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "sample_rate": self.sample_rate,
            "stream": True,
        }
        loop = asyncio.get_running_loop()
        total_deadline = loop.time() + self.total_timeout_s
        first_deadline = min(loop.time() + self.first_chunk_timeout_s, total_deadline)

        attempt = 0
        while True:
            emitted = False
            try:
                async for chunk in self._stream_pcm(payload, cancel, first_deadline, total_deadline):
                    emitted = True
                    # yield 在所有超时范围之外：消费者自己的 await 不会被本模块取消。
                    yield chunk
                if not emitted:
                    raise TtsError("tts returned no audio")
                return
            except _Cancelled:
                # 被打断：关闭连接（_stream_pcm 的 finally）后安静结束。
                return
            except TtsTimeout:
                raise
            except (_HttpStatusError, httpx.TransportError) as exc:
                # 已经产出音频的请求不重试：重发会让调用方重复播报。
                if emitted or not _is_retryable(exc) or attempt >= self.max_retries:
                    raise TtsError(f"tts stream failed: {exc}") from exc
                await self._pause(self.retry_base_delay * (2**attempt), cancel)
                attempt += 1

    # -- 内部：单次请求 ---------------------------------------------------- #

    async def _stream_pcm(
        self,
        payload: dict[str, Any],
        cancel: asyncio.Event,
        first_deadline: float,
        total_deadline: float,
    ) -> AsyncIterator[bytes]:
        request = self._client.build_request("POST", "/audio/speech", json=payload)
        # 建连 + 等响应头也算首包等待的一部分（T04：首包从请求开始计时）。
        response = await self._network_wait(
            self._client.send(request, stream=True),
            first_deadline,
            total_deadline,
            cancel,
        )
        try:
            if response.status_code >= 400:
                body = await self._network_wait(
                    response.aread(),
                    total_deadline,
                    total_deadline,
                    cancel,
                )
                raise _HttpStatusError(response.status_code, body.decode("utf-8", "replace")[:500])

            iterator = response.aiter_bytes()
            first_chunk = True
            while True:
                deadline = (
                    min(first_deadline, total_deadline) if first_chunk else total_deadline
                )
                try:
                    chunk = await self._network_wait(
                        anext(iterator), deadline, total_deadline, cancel
                    )
                except StopAsyncIteration:
                    return
                if not chunk:
                    continue
                first_chunk = False
                yield chunk
        finally:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不应盖住原始异常
                pass

    async def _network_wait(
        self,
        coro: Coroutine[Any, Any, Any],
        deadline: float,
        total_deadline: float,
        cancel: asyncio.Event,
    ) -> Any:
        """等一次网络操作完成，受截止时刻约束，并且可被 cancel 立即打断。

        超时范围只覆盖这一次等待；返回后调用方可能 `yield`，那不在任何超时上下文里。
        """
        loop = asyncio.get_running_loop()
        if cancel.is_set():
            coro.close()
            raise _Cancelled
        if loop.time() >= deadline:
            coro.close()
            raise self._timeout_error(loop.time() >= total_deadline)

        task = asyncio.ensure_future(coro)
        cancel_task = asyncio.ensure_future(cancel.wait())
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    await asyncio.wait({task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            except TimeoutError:
                raise self._timeout_error(loop.time() >= total_deadline) from None
            if cancel.is_set():
                raise _Cancelled
            return task.result()
        finally:
            cancel_task.cancel()
            if not task.done():
                task.cancel()
            # 收掉被取消任务的结果，避免 "Task exception was never retrieved"。
            await asyncio.gather(task, cancel_task, return_exceptions=True)

    def _timeout_error(self, total_exceeded: bool) -> TtsTimeout:
        if total_exceeded:
            return TtsTimeout(f"tts stream exceeded {self.total_timeout_s}s (total timeout)")
        return TtsTimeout(f"no audio within {self.first_chunk_timeout_s}s (first-chunk timeout)")

    async def _pause(self, delay: float, cancel: asyncio.Event) -> None:
        """重试前的退避等待；cancel 置位时立即结束（抛内部 _Cancelled）。"""
        if cancel.is_set():
            raise _Cancelled
        sleep_task = asyncio.ensure_future(asyncio.sleep(delay))
        cancel_task = asyncio.ensure_future(cancel.wait())
        try:
            await asyncio.wait({sleep_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel.is_set():
                raise _Cancelled
        finally:
            cancel_task.cancel()
            if not sleep_task.done():
                sleep_task.cancel()
            await asyncio.gather(sleep_task, cancel_task, return_exceptions=True)

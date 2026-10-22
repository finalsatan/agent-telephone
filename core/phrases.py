"""预合成话术与缓存（T06）。

设计文档第 4 次评审的结论：云端 TTS 首包太慢，T05 本地 CosyVoice3 的首包也在 1s 以上，
实时链路不能等 TTS 现合成。这里把高频、不含变量的句子提前合成成 PCM，实时链路直接播。

三部分：

1. 话术表 `PHRASES`：键为 id（T07 的 `reply_hint` 会引用这些 id），值可含 `{owner}`，
   机主称呼由 `config.yaml: owner.name` 填入（0.6），代码里不写死。
2. `PhraseCache`：按文本哈希读写 `cache/tts/<backend>/<sha1(文本+voice)>.pcm`。
   缓存必须由**实时链路实际使用的同一后端、同一音色**生成，否则缓存句与现合成句音色不一致，
   因此目录按后端分、哈希里带音色标识（云端是 `tts.voice`，本地是 T05b 的 `voice_id`）。
3. 号码复述片段：逐位数字 + 固定前后缀拼接（`render_phone`），手机号按 3-4-4 分组、
   座机按 区号-号码 分组，组间 150ms、位间 40ms 静音，每段首尾裁掉静音。
   其他含变量的复述（姓名、来意）仍走现合成，不做拼接。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from core.prompts import DEFAULT_OWNER_NAME
from core.types import State

OWNER_PLACEHOLDER = "{owner}"

# 话术表：键为 id，值为文本模板。每个状态至少有一条（见 STATE_PHRASES）。
PHRASES: dict[str, str] = {
    "greet": "您好，这里是{owner}的电话助理，他暂时不方便接听。请问您是哪位？",
    "ask_name": "请问您怎么称呼？",
    "ask_purpose": "请问您找他有什么事？",
    "ask_callback": "请留一下您的回拨电话，我记下来转告他。",
    "ask_time": "请问具体是什么时间？",
    "repeat": "抱歉，我没听清，请您再说一遍。",
    "please_wait_confirm": "好的，我核对一下，请您稍等。",
    "forward": "我会转告，请您留下联系方式。",
    "goodbye": "好的，我会转告{owner}，再见。",
}

# 复述号码用的固定前后缀（第 4 次评审新增：号码不能现合成）。
PHONE_PREFIX = "confirm_phone_prefix"
PHONE_SUFFIX = "confirm_phone_suffix"
PHONE_AFFIX_PHRASES: dict[str, str] = {
    PHONE_PREFIX: "您的回拨电话是",
    PHONE_SUFFIX: "，对吗？",
}

# 数字读法：1 读"幺"（电话里更不容易听错），其余按日常读法。
DIGIT_READINGS: dict[str, str] = {
    "0": "零",
    "1": "幺",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}

# 每个状态至少有一条可预合成的模板话术（T07 模板优先）。
STATE_PHRASES: dict[State, tuple[str, ...]] = {
    State.GREETING: ("greet",),
    State.ASK_IDENTITY: ("ask_name",),
    State.ASK_PURPOSE: ("ask_purpose",),
    State.FILL_MISSING: ("ask_callback", "ask_time"),
    State.CONFIRM: ("please_wait_confirm",),
    State.CLOSING: ("goodbye",),
}

DEFAULT_CACHE_ROOT = Path("cache/tts")
DEFAULT_SAMPLE_RATE = 16000
# 数字片段首尾静音裁剪的默认阈值（int16 RMS，约 -44dBFS）。TTS 输出即使"静音"也有底噪，
# 所以不能只裁全零样本；阈值可在 config.yaml: tts.phrase_cache.trim_threshold 调。
DEFAULT_TRIM_THRESHOLD = 200
DEFAULT_TRIM_FRAME_MS = 10
DEFAULT_DIGIT_GAP_MS = 40
DEFAULT_GROUP_GAP_MS = 150

# 本地后端拿不到 voice_id 时的占位（T05b 合入后由客户端从 /health 提供）。
# 占位值是固定的，所以不会把不同参考音色的缓存混在一起，只是换参考音频时不会自动失效。
LOCAL_VOICE_PLACEHOLDER = "local-reference-unknown"


class PhraseMissing(RuntimeError):
    """缓存里缺某个片段：还没跑 tools/prerender_phrases.py，或换了音色/后端。"""


def owner_name(value: str | None = None) -> str:
    """机主称呼：空值退回通用称呼，避免生成"这里是 的电话助理"。"""
    return (value or "").strip() or DEFAULT_OWNER_NAME


def fill_owner(text: str, owner: str | None = None) -> str:
    return text.replace(OWNER_PLACEHOLDER, owner_name(owner))


def phrase_text(phrase_id: str, owner: str | None = None) -> str:
    """取一条话术的最终文本（机主称呼已填好）。"""
    return fill_owner(PHRASES[phrase_id], owner)


def digit_text(digit: str) -> str:
    """单个数字的读法文本（复述号码用）。"""
    try:
        return DIGIT_READINGS[str(digit)]
    except KeyError as exc:  # pragma: no cover - 调用方已过滤成数字
        raise ValueError(f"not a digit: {digit!r}") from exc


def cache_texts(owner: str | None = None) -> dict[str, str]:
    """需要预合成的全部文本：话术 + 号码前后缀 + 10 个数字片段。"""
    texts = {phrase_id: fill_owner(text, owner) for phrase_id, text in PHRASES.items()}
    texts.update(PHONE_AFFIX_PHRASES)
    texts.update({f"digit_{digit}": reading for digit, reading in DIGIT_READINGS.items()})
    return texts


def backend_name(config: Mapping) -> str:
    """当前 TTS 后端名（T05 起可在 config.yaml: tts.backend 选择 cloud|local）。"""
    return str((config.get("tts") or {}).get("backend") or "cloud").strip().lower()


def resolve_voice_key(
    config: Mapping,
    *,
    backend: str | None = None,
    tts: object | None = None,
    voice: str | None = None,
) -> str:
    """缓存键里的音色标识。

    优先级：显式 `voice`（工具参数）> 客户端的 `voice_id`（T05b 的本地客户端从 /health 取）
    > `config.yaml: tts.local_voice_id`（本地后端的手动固定值）> 占位常量。
    云端用 `tts.voice`。
    """
    if voice:
        return str(voice)
    backend = (backend or backend_name(config)).strip().lower()
    tts_cfg = config.get("tts") or {}
    if backend != "local":
        return str(tts_cfg.get("voice") or "")
    voice_id = getattr(tts, "voice_id", None)
    if voice_id:
        return str(voice_id)
    configured = tts_cfg.get("local_voice_id")
    if configured:
        return str(configured)
    return LOCAL_VOICE_PLACEHOLDER


def silence(ms: float, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """生成 ms 毫秒的 16-bit 单声道静音。"""
    samples = max(0, int(round(sample_rate * float(ms) / 1000.0)))
    return b"\x00\x00" * samples


def _to_samples(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:  # 半个样本不该出现，容错截掉
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2")


def trim_silence(
    pcm: bytes,
    *,
    threshold: int = DEFAULT_TRIM_THRESHOLD,
    frame_ms: int = DEFAULT_TRIM_FRAME_MS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """裁掉首尾 RMS 低于阈值的帧。整段都低于阈值时返回空字节串。

    只在 `render_phone()` 拼片段时用：TTS 每段的首尾静音长短不一，不裁掉的话
    数字之间听起来会一顿一顿的。
    """
    samples = _to_samples(pcm)
    if samples.size == 0:
        return b""
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    n_frames = -(-samples.size // frame)
    # 转 float64 再平方，避免 int16 的 abs(-32768) 溢出。
    frames = np.zeros(n_frames * frame, dtype=np.float64)
    frames[: samples.size] = samples.astype(np.float64)
    rms = np.sqrt(np.mean(frames.reshape(n_frames, frame) ** 2, axis=1))
    loud = np.nonzero(rms >= threshold)[0]
    if loud.size == 0:
        return b""
    start = int(loud[0]) * frame
    end = min((int(loud[-1]) + 1) * frame, samples.size)
    return samples[start:end].tobytes()


def phone_groups(number: str) -> list[str]:
    """按电话分段规则把号码切成组：手机号 3-4-4，座机 区号-号码，其它每 4 位一组。"""
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    if not digits:
        raise ValueError(f"phone number has no digits: {number!r}")
    if len(digits) == 11 and digits[0] == "1":
        return [digits[:3], digits[3:7], digits[7:]]
    if digits.startswith("0") and len(digits) >= 10:
        # 010/02x 是 3 位区号（含前导 0），03xx–09xx 是 4 位。
        area_len = 3 if digits[1] in "12" else 4
        if len(digits) - area_len in (7, 8):
            return [digits[:area_len], digits[area_len:]]
    return [digits[i : i + 4] for i in range(0, len(digits), 4)]


class PhraseCache:
    """按文本哈希读写预合成的 PCM。同一目录也用于 T10 实时链路查表。"""

    def __init__(
        self,
        backend: str,
        voice_key: str,
        *,
        root: str | Path | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        trim_threshold: int = DEFAULT_TRIM_THRESHOLD,
        trim_frame_ms: int = DEFAULT_TRIM_FRAME_MS,
        digit_gap_ms: float = DEFAULT_DIGIT_GAP_MS,
        group_gap_ms: float = DEFAULT_GROUP_GAP_MS,
    ) -> None:
        self.backend = str(backend).strip().lower()
        self.voice_key = str(voice_key)
        self.root = Path(root) if root is not None else DEFAULT_CACHE_ROOT
        self.sample_rate = sample_rate
        self.trim_threshold = trim_threshold
        self.trim_frame_ms = trim_frame_ms
        self.digit_gap_ms = digit_gap_ms
        self.group_gap_ms = group_gap_ms

    @classmethod
    def from_config(
        cls,
        config: Mapping,
        *,
        tts: object | None = None,
        backend: str | None = None,
        voice: str | None = None,
        root: str | Path | None = None,
    ) -> "PhraseCache":
        """用 config.yaml 构造（backend 决定子目录，voice 决定哈希）。"""
        backend = (backend or backend_name(config)).strip().lower()
        cache_cfg = (config.get("tts") or {}).get("phrase_cache") or {}
        audio_cfg = config.get("audio") or {}
        return cls(
            backend,
            resolve_voice_key(config, backend=backend, tts=tts, voice=voice),
            root=root if root is not None else cache_cfg.get("dir", DEFAULT_CACHE_ROOT),
            sample_rate=int(audio_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            trim_threshold=int(cache_cfg.get("trim_threshold", DEFAULT_TRIM_THRESHOLD)),
            trim_frame_ms=int(cache_cfg.get("trim_frame_ms", DEFAULT_TRIM_FRAME_MS)),
            digit_gap_ms=float(cache_cfg.get("digit_gap_ms", DEFAULT_DIGIT_GAP_MS)),
            group_gap_ms=float(cache_cfg.get("group_gap_ms", DEFAULT_GROUP_GAP_MS)),
        )

    # -- 路径与读写 -------------------------------------------------------- #

    @property
    def directory(self) -> Path:
        return self.root / self.backend

    def key(self, text: str) -> str:
        return hashlib.sha1(f"{text}{self.voice_key}".encode("utf-8")).hexdigest()

    def path(self, text: str) -> Path:
        return self.directory / f"{self.key(text)}.pcm"

    def has(self, text: str) -> bool:
        return self.path(text).is_file()

    def get(self, text: str) -> bytes | None:
        try:
            return self.path(text).read_bytes()
        except FileNotFoundError:
            return None

    def put(self, text: str, pcm: bytes) -> Path:
        path = self.path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pcm.tmp")
        tmp.write_bytes(pcm)
        tmp.replace(path)  # 原子替换：实时链路不会读到写了一半的文件
        return path

    # -- 号码复述 ---------------------------------------------------------- #

    def render_phone(self, number: str, *, prefix: str = PHONE_PREFIX, suffix: str = PHONE_SUFFIX) -> bytes:
        """前缀 + 逐位数字 + 后缀拼出号码复述音频。

        分组之间插 `group_gap_ms`，位之间插 `digit_gap_ms`；每段先裁掉首尾静音。
        缺片段时抛 `PhraseMissing`（先跑 prerender）。
        """
        digits = "".join(ch for ch in str(number) if ch.isdigit())
        if not digits:
            raise ValueError(f"phone number has no digits: {number!r}")

        digit_gap = silence(self.digit_gap_ms, sample_rate=self.sample_rate)
        group_gap = silence(self.group_gap_ms, sample_rate=self.sample_rate)

        parts = [self._require(PHONE_AFFIX_PHRASES[prefix])]
        for group_index, group in enumerate(phone_groups(digits)):
            if group_index:
                parts.append(group_gap)
            for digit_index, digit in enumerate(group):
                if digit_index:
                    parts.append(digit_gap)
                parts.append(self._require(DIGIT_READINGS[digit]))
        parts.append(self._require(PHONE_AFFIX_PHRASES[suffix]))
        return b"".join(parts)

    def _require(self, text: str) -> bytes:
        pcm = self.get(text)
        if pcm is None:
            raise PhraseMissing(
                f"missing pre-rendered phrase {text!r} in {self.directory} "
                f"(backend={self.backend}, voice={self.voice_key}); "
                "run `uv run python tools/prerender_phrases.py` first"
            )
        return trim_silence(
            pcm,
            threshold=self.trim_threshold,
            frame_ms=self.trim_frame_ms,
            sample_rate=self.sample_rate,
        )

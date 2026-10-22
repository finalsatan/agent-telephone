"""预合成话术与缓存（T06）。

设计文档第 4 次评审的结论：云端 TTS 首包太慢，T05 本地 CosyVoice3 的首包也在 1s 以上，
实时链路不能等 TTS 现合成。这里把高频、不含变量的句子提前合成成 PCM，实时链路直接播。

三部分：

1. 话术表 `PHRASES`：键为 id（T07 的 `reply_hint` 会引用这些 id），值可含 `{owner}`，
   机主称呼由 `config.yaml: owner.name` 填入（0.6），代码里不写死。
2. `PhraseCache`：按文本哈希读写 `cache/tts/<backend>/<sha1(版本+文本+voice)>.pcm`。
   缓存必须由**实时链路实际使用的同一后端、同一音色**生成，否则缓存句与现合成句音色不一致，
   因此目录按后端分、哈希里带音色标识（云端是 `tts.voice`，本地是 T05b 的 `voice_id`）。
3. **号码复述的兜底路径**（第 11 次评审：主方案改为 T07c 的整句现合成，这里只在现合成
   失败时用）：逐位数字 + 固定前后缀拼接（`render_phone`），手机号按 3-4-4 分组、
   座机按 区号-号码 分组，组间 150ms、位间 40ms 静音，每段首尾裁掉静音。
   数字片段不是单字合成的：一次合成整句（`DIGIT_SENTENCE`）再按停顿切成 10 段
   （T06b），这样一条句子里的语速与韵律一致，单字合成那种 170–620ms 的抖动会消失。
   其他含变量的复述（姓名、来意）仍走现合成，不做拼接。

话术表 `PHRASES` 只有这一份：`core/dialog.py` 从这里导入（T06b 的第 9 次评审要求），
任何一条文本与状态机不一致，模板话术就永远命不中缓存。

**导入约束（第 11/12 次评审）**：本模块只能依赖 `core.types`、`core.normalize`、标准库和 numpy，
不得导入 `core.prompts` 或 `core.dialog`——`dialog` 会导入本模块，导入 `prompts` 就会成环
（T07c 已把意图判定拆到 `core.intent`，`prompts` 不再导入 `dialog`，但约束仍然保留）。
号码分组 `phone_groups` 自 T07c 起放在 `core/normalize.py`，本模块只导入复用。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from core.normalize import phone_groups  # noqa: F401  (re-export: T06 的调用方从这里取)
from core.types import State

# 机主称呼的兜底值（0.6：真正的称呼由 config.yaml: owner.name 提供）。
# 这里自己定义、**不导入 core.prompts**：`dialog` 导入 `phrases`，而 `prompts` 导入 `dialog`，
# 走 prompts 会形成 `dialog → phrases → prompts → dialog` 的循环导入（第 12 次评审实测报错）。
DEFAULT_OWNER_NAME = "机主"

OWNER_PLACEHOLDER = "{owner}"

# 话术表：键为 id，值为文本模板。每个状态至少有一条（见 STATE_PHRASES）。
PHRASES: dict[str, str] = {
    # 文本以 T07（main 上的 core/dialog.py）为准：状态机按这些 id 播放，
    # 任何一条与 dialog 里的不一致，缓存命中就会永远失败。
    "greet": "您好，这里是{owner}的电话助理，他暂时不方便接听。请问您是哪位？",
    "ask_name": "请问您是哪位？",
    "ask_purpose": "请问您有什么事？",
    "ask_callback": "请问您的回拨电话是多少？",
    "ask_time": "请问具体是什么时间？",
    "repeat": "抱歉，我没听清，请您再说一遍。",
    "please_wait_confirm": "好的，我核对一下，请您稍等。",
    # 复述号码的前后缀（T07 用 PHRASES[...] 访问，见 PHONE_AFFIX_PHRASES）。
    "confirm_phone_prefix": "您的回拨电话是",
    "confirm_phone_suffix": "，对吗？",
    # T06b 新补：整句确认语的后半句，以及 CONFIRM 里"哪一项不对"的追问（T07c 使用）。
    "confirm_question": "对吗？",
    "ask_which_wrong": "请问是哪一项不对？",
    "forward": "好的，我会转告{owner}，请您留一下联系方式。",
    "goodbye": "好的，我会转告{owner}，再见。",
}

# 复述号码用的固定前后缀（第 4 次评审新增：号码不能现合成）。
# 文本只在 PHRASES 里定义一处；这里给出另一种访问方式（T06 内部用），
# T07 的 `core/dialog.py` 直接用 PHRASES[PHONE_PREFIX] / PHRASES[PHONE_SUFFIX]。
PHONE_PREFIX = "confirm_phone_prefix"
PHONE_SUFFIX = "confirm_phone_suffix"
PHONE_AFFIX_PHRASES: dict[str, str] = {
    PHONE_PREFIX: PHRASES[PHONE_PREFIX],
    PHONE_SUFFIX: PHRASES[PHONE_SUFFIX],
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

# 数字片段的合成顺序（与 DIGIT_READINGS 的键无关，句子就按这个顺序念：
# 幺、二、三、四、五、六、七、八、九、零）。切分后按同一顺序对回数字。
DIGIT_ORDER = "1234567890"
# T06b：一次合成整句再按停顿切分（单字合成的片段时长在 170–620ms 之间抖动）。
DIGIT_SENTENCE = "，".join(DIGIT_READINGS[digit] for digit in DIGIT_ORDER) + "。"

DEFAULT_CACHE_ROOT = Path("cache/tts")
DEFAULT_SAMPLE_RATE = 16000
# 数字片段首尾静音裁剪的默认阈值（int16 RMS，约 -44dBFS）。TTS 输出即使"静音"也有底噪，
# 所以不能只裁全零样本；阈值可在 config.yaml: tts.phrase_cache.trim_threshold 调。
DEFAULT_TRIM_THRESHOLD = 200
DEFAULT_TRIM_FRAME_MS = 10
DEFAULT_DIGIT_GAP_MS = 40
DEFAULT_GROUP_GAP_MS = 150
# 整句数字切分时，短于该值的停顿算同一个字内部的间隙，不切（T06b）。
DEFAULT_DIGIT_SPLIT_MIN_GAP_MS = 80
# 整句合成偶尔多出的一小段呼吸/尾音：又短（<= 该值）又弱（< 中位峰值的该比例）才忽略。
DEFAULT_DIGIT_IGNORE_MAX_MS = 80
DEFAULT_DIGIT_IGNORE_PEAK_RATIO = 0.25

# 缓存键版本：T06b 起键里加了分隔符（第 10 次评审指出的"文本+音色"拼接歧义），
# 同时换版本号让 T06 的旧缓存自动失效（数字片段改为整句切分，内容也不同了）。
CACHE_KEY_VERSION = "2"

# 本地后端拿不到 voice_id 时的占位（T05b 合入后由客户端从 /health 提供）。
# 占位值是固定的，所以不会把不同参考音色的缓存混在一起，只是换参考音频时不会自动失效。
LOCAL_VOICE_PLACEHOLDER = "local-reference-unknown"


class PhraseMissing(RuntimeError):
    """缓存里缺某个片段：还没跑 tools/prerender_phrases.py，或换了音色/后端。"""


class PhraseSplitError(RuntimeError):
    """整句数字合成后切分失败（段数不是 10）。

    不要静默退回单字合成（T06b）：段数不对说明 TTS 的读法或停顿变了，
    退回去会悄悄用回 170–620ms 的长片段，问题更难发现。
    """


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


def phrase_texts(owner: str | None = None) -> dict[str, str]:
    """话术表里每一条的最终文本（机主称呼已填好，前后缀也在内）。"""
    return {phrase_id: fill_owner(text, owner) for phrase_id, text in PHRASES.items()}


def digit_texts() -> dict[str, str]:
    """数字 → 读法文本（复述号码的片段，按 DIGIT_ORDER 排列）。"""
    return {digit: DIGIT_READINGS[digit] for digit in DIGIT_ORDER}


def cache_texts(owner: str | None = None) -> dict[str, str]:
    """需要预合成的全部文本：话术（含号码前后缀）+ 10 个数字片段。

    数字片段的 key 是 `digit_<数字>`，值是读法文本；缓存里实际存的是整句切出来的片段，
    但查找仍按读法文本哈希（`PhraseCache.get(digit_text("8"))`）。
    """
    texts = phrase_texts(owner)
    texts.update({f"digit_{digit}": text for digit, text in digit_texts().items()})
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


def _frame_rms(samples: np.ndarray, frame: int, n_frames: int) -> np.ndarray:
    """按帧算 RMS。末尾不足一帧的补 0（补的静音不会被当成发声）。"""
    buf = np.zeros(n_frames * frame, dtype=np.float64)
    buf[: samples.size] = samples.astype(np.float64)
    return np.sqrt(np.mean(buf.reshape(n_frames, frame) ** 2, axis=1))


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
    loud = np.nonzero(_frame_rms(samples, frame, n_frames) >= threshold)[0]
    if loud.size == 0:
        return b""
    start = int(loud[0]) * frame
    end = min((int(loud[-1]) + 1) * frame, samples.size)
    return samples[start:end].tobytes()


def split_digit_clips(
    pcm: bytes,
    *,
    expected: int = len(DIGIT_ORDER),
    min_gap_ms: float = DEFAULT_DIGIT_SPLIT_MIN_GAP_MS,
    ignore_max_ms: float = DEFAULT_DIGIT_IGNORE_MAX_MS,
    ignore_peak_ratio: float = DEFAULT_DIGIT_IGNORE_PEAK_RATIO,
    threshold: int = DEFAULT_TRIM_THRESHOLD,
    frame_ms: int = DEFAULT_TRIM_FRAME_MS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[bytes]:
    """把整句数字（`DIGIT_SENTENCE`）按停顿切成单字片段，返回 `expected` 段。

    判定方式与静音裁剪一致：按 `frame_ms` 帧算 RMS，低于 `threshold` 的帧算静音；
    两段之间的静音短于 `min_gap_ms` 时视为同一个字内部的间隙（辅音、爆破造成的凹陷），
    不切。

    单次合成偶尔会在句尾多出一小段呼吸/尾音（实测 50ms、峰值 RMS 1365，约 1/6 的概率）。
    **又短又弱**的段（时长 < `ignore_max_ms` 且峰值 RMS < 中位峰值的 `ignore_peak_ratio` 倍）
    直接忽略；正常的数字段都在 110ms 以上，不会被这条规则误伤。

    段数仍不是 `expected` 时抛 `PhraseSplitError`，由调用方决定怎么办（T06b 要求
    不要静默退回单字合成）。
    """
    samples = _to_samples(pcm)
    if samples.size == 0:
        raise PhraseSplitError("digit sentence produced no audio")
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    n_frames = -(-samples.size // frame)
    rms = _frame_rms(samples, frame, n_frames)
    loud = rms >= threshold

    runs: list[list[int]] = []
    start: int | None = None
    for index, voiced in enumerate(loud):
        if voiced and start is None:
            start = index
        elif not voiced and start is not None:
            runs.append([start, index - 1])
            start = None
    if start is not None:
        runs.append([start, n_frames - 1])

    # 合并"字内凹陷"：间隔帧数少于 min_gap_frames 的不算停顿。
    min_gap_frames = max(1, int(round(min_gap_ms / frame_ms)))
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 < min_gap_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # 去掉又短又弱的呼吸/尾音（见 docstring）。
    if len(merged) > expected:
        peaks = [float(rms[run_start : run_end + 1].max()) for run_start, run_end in merged]
        reference = float(np.median(peaks)) if peaks else 0.0
        kept = [
            run
            for run, peak in zip(merged, peaks)
            if not (
                (run[1] - run[0] + 1) * frame_ms <= ignore_max_ms
                and peak < reference * ignore_peak_ratio
            )
        ]
        merged = kept

    if len(merged) != expected:
        gaps = [
            round((run[0] - merged[index][1] - 1) * frame_ms) for index, run in enumerate(merged[1:])
        ]
        durations = [round((run[1] - run[0] + 1) * frame_ms) for run in merged]
        raise PhraseSplitError(
            f"expected {expected} digit segments, got {len(merged)} "
            f"(durations_ms={durations}, gaps_ms={gaps}, min_gap_ms={min_gap_ms}, "
            f"threshold={threshold})"
        )
    return [
        samples[run_start * frame : min((run_end + 1) * frame, samples.size)].tobytes()
        for run_start, run_end in merged
    ]


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
        # 用 \x1f 分隔各段：直接拼接时 ("ab","c") 与 ("a","bc") 会撞成同一个键。
        payload = f"{CACHE_KEY_VERSION}\x1f{text}\x1f{self.voice_key}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

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
        """前缀 + 逐位数字 + 后缀拼出号码复述音频（**兜底路径**，T07c 起主方案是整句现合成）。

        第 11 次评审：用户试听发现逐位拼接的韵律对不齐、连贯性差，所以号码复述改成由代码
        把号码写成汉字、和确认语一起交给 TTS 现合成（T07c）。本方法保留，只在现合成失败时用。

        分组之间、前缀与第一位之间、最后一位与后缀之间都插 `group_gap_ms`，
        位之间插 `digit_gap_ms`；每段先裁掉首尾静音。缺片段时抛 `PhraseMissing`
        （先跑 prerender）。
        """
        digits = "".join(ch for ch in str(number) if ch.isdigit())
        if not digits:
            raise ValueError(f"phone number has no digits: {number!r}")

        digit_gap = silence(self.digit_gap_ms, sample_rate=self.sample_rate)
        group_gap = silence(self.group_gap_ms, sample_rate=self.sample_rate)

        parts = [self._require(PHONE_AFFIX_PHRASES[prefix]), group_gap]
        for group_index, group in enumerate(phone_groups(digits)):
            if group_index:
                parts.append(group_gap)
            for digit_index, digit in enumerate(group):
                if digit_index:
                    parts.append(digit_gap)
                parts.append(self._require(DIGIT_READINGS[digit]))
        parts.append(group_gap)
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

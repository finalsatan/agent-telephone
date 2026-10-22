"""T06 / T06b 离线测试：话术表、缓存读写、号码复述拼接与静音裁剪（全部用 fake TTS，不联网）。

T06b 新增：话术表只保留 `core/phrases.py` 一份并与 `core/dialog.py` 一致、数字片段改为
整句合成后按停顿切分、前后缀两侧也插组间静音。
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pytest

from core import dialog as dialog_module
from core.phrases import (
    DEFAULT_DIGIT_GAP_MS,
    DEFAULT_DIGIT_SPLIT_MIN_GAP_MS,
    DEFAULT_GROUP_GAP_MS,
    DEFAULT_SAMPLE_RATE,
    DIGIT_ORDER,
    DIGIT_READINGS,
    DIGIT_SENTENCE,
    LOCAL_VOICE_PLACEHOLDER,
    PHRASES,
    PHONE_AFFIX_PHRASES,
    PHONE_PREFIX,
    PHONE_SUFFIX,
    STATE_PHRASES,
    PhraseCache,
    PhraseMissing,
    PhraseSplitError,
    backend_name,
    cache_texts,
    digit_text,
    digit_texts,
    fill_owner,
    phrase_text,
    phrase_texts,
    phone_groups,
    resolve_voice_key,
    silence,
    split_digit_clips,
    trim_silence,
)
from core.types import State
from tools import prerender_phrases

# ---- 工具函数 -------------------------------------------------------------- #


def tone(amplitude: int = 8000, ms: float = 100, *, freq_hz: int = 220) -> bytes:
    """一段正弦波（int16），用于区分不同片段。"""
    n = int(DEFAULT_SAMPLE_RATE * ms / 1000)
    t = np.arange(n) / DEFAULT_SAMPLE_RATE
    samples = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype("<i2")
    return samples.tobytes()


def silence_ms(ms: float) -> int:
    """ms 对应的静音字节数（用于断言拼接间隔）。"""
    return len(silence(ms))


def digit_sentence_pcm(
    *,
    syllable_ms: float = 250,
    gap_ms: float = 150,
    lead_ms: float = 200,
    tail_ms: float = 200,
) -> bytes:
    """构造"10 个音节 + 逗号停顿"的整句，用来测切分（间距 > min_gap 才能切开）。"""
    parts = [silence(lead_ms)]
    for index in range(10):
        if index:
            parts.append(silence(gap_ms))
        parts.append(tone(amplitude=5000, ms=syllable_ms, freq_hz=200 + 20 * index))
    parts.append(silence(tail_ms))
    return b"".join(parts)


class FakeTts:
    """只实现 T06 用到的 synth 接口，按文本产出确定性 PCM。

    `DIGIT_SENTENCE` 返回 10 个音节（间隔 150ms），其它文本返回一段正弦波。
    """

    sample_rate = DEFAULT_SAMPLE_RATE

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        digit_pcm: bytes | None = None,
        digit_pcms: list[bytes] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._voice_id = voice_id
        self.digit_pcms = list(digit_pcms) if digit_pcms is not None else None
        if self.digit_pcms is None:
            self.digit_pcms = [digit_pcm if digit_pcm is not None else digit_sentence_pcm()]
        self._digit_index = 0

    @property
    def voice_id(self) -> str | None:
        return self._voice_id

    async def synth(self, text: str, cancel: asyncio.Event):
        self.calls.append(text)
        await asyncio.sleep(0)  # 模拟异步产出
        if text == DIGIT_SENTENCE:
            pcm = self.digit_pcms[min(self._digit_index, len(self.digit_pcms) - 1)]
            self._digit_index += 1
            yield pcm
            return
        yield tone(amplitude=4000 + len(text), ms=50)

    async def aclose(self) -> None:
        pass


def make_cache(tmp_path, *, backend: str = "local", voice: str = "voice-a", **kwargs) -> PhraseCache:
    return PhraseCache(backend, voice, root=tmp_path / "cache", **kwargs)


def put_clip(cache: PhraseCache, text: str, pcm: bytes) -> None:
    cache.put(text, pcm)


# ---- 话术表 ---------------------------------------------------------------- #


def test_every_state_has_a_template_phrase():
    for state in State:
        ids = STATE_PHRASES.get(state)
        assert ids, f"{state} 没有模板话术"
        for phrase_id in ids:
            assert phrase_id in PHRASES, f"{state} 引用了不存在的话术 {phrase_id}"


def test_greet_text_matches_spec():
    assert (
        phrase_text("greet", "张先生")
        == "您好，这里是张先生的电话助理，他暂时不方便接听。请问您是哪位？"
    )


def test_required_phrases_exist():
    for phrase_id in (
        "greet",
        "ask_purpose",
        "ask_callback",
        "ask_time",
        "ask_name",
        "repeat",
        "please_wait_confirm",
        "forward",
        "goodbye",
    ):
        assert phrase_id in PHRASES
    assert PHRASES["repeat"] == "抱歉，我没听清，请您再说一遍。"
    # T06b：文本以 T07（状态机）为准
    assert PHRASES["ask_name"] == "请问您是哪位？"
    assert PHRASES["ask_purpose"] == "请问您有什么事？"
    assert PHRASES["ask_callback"] == "请问您的回拨电话是多少？"
    assert PHRASES["forward"] == "好的，我会转告{owner}，请您留一下联系方式。"
    assert PHRASES["ask_time"] and PHRASES["please_wait_confirm"]


def test_owner_placeholder_is_filled_from_config_name():
    texts = cache_texts("张先生")
    for text in texts.values():
        assert "{owner}" not in text
    assert "张先生" in texts["greet"]
    assert "张先生" in texts["goodbye"]
    assert "张先生" not in texts["ask_name"]
    assert fill_owner("这里是{owner}的电话助理", "王女士") == "这里是王女士的电话助理"
    # 空称呼退回通用称呼，不生成“这里是 的电话助理”
    assert fill_owner("这里是{owner}的电话助理", "  ") == "这里是机主的电话助理"
    assert fill_owner("这里是{owner}的电话助理", None) == "这里是机主的电话助理"


def test_cache_texts_cover_phrases_digits_and_affixes():
    texts = cache_texts("张先生")
    for phrase_id in PHRASES:
        assert texts[phrase_id] == phrase_text(phrase_id, "张先生")
    # 前后缀只定义一处：两种访问方式拿到同一批文本
    assert PHONE_AFFIX_PHRASES == {PHONE_PREFIX: PHRASES[PHONE_PREFIX], PHONE_SUFFIX: PHRASES[PHONE_SUFFIX]}
    assert texts[PHONE_PREFIX] == "您的回拨电话是"
    assert texts[PHONE_SUFFIX] == "，对吗？"
    assert texts[PHONE_PREFIX] == PHONE_AFFIX_PHRASES[PHONE_PREFIX]
    for digit, reading in DIGIT_READINGS.items():
        assert texts[f"digit_{digit}"] == reading
    assert len(texts) == len(PHRASES) + 10
    # 数字片段按整句合成顺序排列
    assert list(digit_texts()) == list(DIGIT_ORDER)
    assert "，".join(digit_texts()[d] for d in DIGIT_ORDER) + "。" == DIGIT_SENTENCE


def test_t06b_added_phrases_for_t07c():
    # 第 12 次评审：整句确认语的后半句与"哪一项不对"的追问，供 T07c 使用
    assert PHRASES["confirm_question"] == "对吗？"
    assert PHRASES["ask_which_wrong"] == "请问是哪一项不对？"
    # T12 对比实验：确认语开头预合成
    assert PHRASES["confirm_lead"] == "好的，跟您确认一下。"
    texts = cache_texts("张先生")
    assert texts["confirm_question"] == "对吗？"
    assert texts["ask_which_wrong"] == "请问是哪一项不对？"
    assert texts["confirm_lead"] == "好的，跟您确认一下。"


def test_phrases_module_does_not_import_prompts_or_dialog():
    """第 12 次评审的循环导入约束：phrases 只能依赖 types / normalize / 标准库 / numpy。"""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "core" / "phrases.py"
    modules = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    core_modules = {m for m in modules if m.startswith("core.")}
    assert core_modules <= {"core.types", "core.normalize"}, core_modules
    assert {"core.prompts", "core.dialog"}.isdisjoint(modules)


def test_core_modules_import_without_cycle():
    # 第 12 次评审的阻塞项：原先是 dialog → phrases → prompts → dialog。
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import core.dialog, core.phrases, core.prompts"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_digit_readings_use_yao_for_one():
    assert digit_text("1") == "幺"
    assert digit_text("0") == "零"
    assert [DIGIT_READINGS[str(d)] for d in range(10)] == [
        "零",
        "幺",
        "二",
        "三",
        "四",
        "五",
        "六",
        "七",
        "八",
        "九",
    ]
    with pytest.raises(ValueError):
        digit_text("十")


# ---- 后端与音色键 ---------------------------------------------------------- #


def test_backend_name_reads_config():
    assert backend_name({"tts": {"backend": "local"}}) == "local"
    assert backend_name({"tts": {"backend": " Cloud "}}) == "cloud"
    assert backend_name({}) == "cloud"


def test_voice_key_resolution():
    config = {"tts": {"backend": "cloud", "voice": "FunAudioLLM/CosyVoice2-0.5B:alex"}}
    assert resolve_voice_key(config) == "FunAudioLLM/CosyVoice2-0.5B:alex"
    # 显式参数优先
    assert resolve_voice_key(config, voice="explicit") == "explicit"

    local = {"tts": {"backend": "local", "voice": ""}}
    # T05b 的客户端从 /health 取 voice_id
    assert resolve_voice_key(local, tts=FakeTts(voice_id="abc123")) == "abc123"
    # 客户端还没提供时，允许 config 固定
    configured = {"tts": {"backend": "local", "voice": "", "local_voice_id": "cfg-id"}}
    assert resolve_voice_key(configured, tts=FakeTts()) == "cfg-id"
    # 都没有时用占位值（固定值，不会把不同音色混在一起）
    assert resolve_voice_key(local, tts=FakeTts()) == LOCAL_VOICE_PLACEHOLDER


def test_from_config_reads_phrase_cache_settings(tmp_path):
    config = {
        "tts": {
            "backend": "local",
            "voice": "",
            "phrase_cache": {
                "dir": str(tmp_path / "cache"),
                "trim_threshold": 320,
                "digit_gap_ms": 30,
                "group_gap_ms": 120,
            },
        },
        "audio": {"sample_rate": 16000},
    }
    cache = PhraseCache.from_config(config, tts=FakeTts(voice_id="v1"))
    assert cache.backend == "local"
    assert cache.voice_key == "v1"
    assert cache.root == tmp_path / "cache"
    assert cache.directory == tmp_path / "cache" / "local"
    assert cache.trim_threshold == 320
    assert cache.digit_gap_ms == 30
    assert cache.group_gap_ms == 120


# ---- 缓存读写 -------------------------------------------------------------- #


def test_cache_key_uses_text_and_voice(tmp_path):
    cache = make_cache(tmp_path, voice="voice-a")
    other_voice = make_cache(tmp_path, voice="voice-b")
    assert cache.path("您好") != other_voice.path("您好")
    assert cache.path("您好") != cache.path("您好。")
    assert cache.path("您好").name.endswith(".pcm")

    assert cache.get("您好") is None
    written = cache.put("您好", b"\x01\x00" * 10)
    assert written == cache.path("您好")
    assert cache.get("您好") == b"\x01\x00" * 10
    assert cache.has("您好")
    assert other_voice.get("您好") is None


def test_cache_key_has_separator_between_text_and_voice(tmp_path):
    # 第 10 次评审的非阻塞建议：直接拼接时 ("ab","c") 与 ("a","bc") 会撞键。
    a = PhraseCache("local", "c", root=tmp_path)
    b = PhraseCache("local", "bc", root=tmp_path)
    assert a.key("ab") != b.key("a")
    # 版本号也在键里：换版本后旧缓存自动失效
    assert PhraseCache("local", "c", root=tmp_path).key("您好") == a.key("您好")


def test_cache_is_written_atomically(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("您好", tone())
    assert [p.name for p in cache.directory.iterdir()] == [cache.path("您好").name]


# ---- 预合成 ---------------------------------------------------------------- #


async def test_prerender_generates_everything_then_skips(tmp_path):
    cache = make_cache(tmp_path)
    tts = FakeTts(voice_id="v1")
    texts = phrase_texts("张先生")

    stats = await prerender_phrases.prerender(cache, tts, texts)
    assert stats == {"generated": len(texts), "skipped": 0, "failed": 0}
    assert tts.calls == list(texts.values())
    for text in texts.values():
        assert cache.get(text)

    again = FakeTts(voice_id="v1")
    stats = await prerender_phrases.prerender(cache, again, texts)
    assert stats == {"generated": 0, "skipped": len(texts), "failed": 0}
    assert again.calls == []


async def test_prerender_force_regenerates(tmp_path):
    cache = make_cache(tmp_path)
    texts = phrase_texts("张先生")
    await prerender_phrases.prerender(cache, FakeTts(), texts)

    tts = FakeTts()
    stats = await prerender_phrases.prerender(cache, tts, texts, force=True)
    assert stats["generated"] == len(texts)
    assert tts.calls == list(texts.values())


async def test_changed_text_regenerates_only_the_changed_phrases(tmp_path):
    cache = make_cache(tmp_path)
    await prerender_phrases.prerender(cache, FakeTts(), phrase_texts("张先生"))

    # 换了机主称呼：只有含 {owner} 的三句话术变了（greet / forward / goodbye），
    # 其余照旧命中缓存——这就是"文本改了自动重新生成"的机制。
    texts = phrase_texts("王女士")
    tts = FakeTts()
    stats = await prerender_phrases.prerender(cache, tts, texts)
    assert stats["generated"] == 3
    assert tts.calls == [texts["greet"], texts["forward"], texts["goodbye"]]
    assert cache.get(texts["greet"]) is not None
    assert cache.get(texts["ask_name"]) is not None


async def test_prerender_reports_failures_without_stopping(tmp_path):
    cache = make_cache(tmp_path)

    class FlakyTts(FakeTts):
        async def synth(self, text: str, cancel: asyncio.Event):
            if text == PHRASES["ask_name"]:
                raise prerender_phrases.TtsError("boom")
            async for chunk in super().synth(text, cancel):
                yield chunk

    stats = await prerender_phrases.prerender(cache, FlakyTts(), phrase_texts("张先生"))
    assert stats["failed"] == 1
    assert stats["generated"] == len(phrase_texts("张先生")) - 1


# ---- 数字片段：整句合成后切分（T06b） ------------------------------------- #


def test_digit_sentence_reads_in_order():
    assert DIGIT_SENTENCE == "幺，二，三，四，五，六，七，八，九，零。"
    assert len(DIGIT_ORDER) == 10


def test_split_digit_clips_splits_on_pauses():
    pcm = digit_sentence_pcm()
    clips = split_digit_clips(pcm)
    assert len(clips) == 10
    for clip in clips:
        # 每段首尾都是发声帧，长度接近一个音节（250ms ± 一帧）
        length_ms = len(clip) / 2 / DEFAULT_SAMPLE_RATE * 1000
        assert 250 - 10 <= length_ms <= 250 + 10


def test_split_digit_clips_keeps_short_dips_inside_a_syllable():
    # 每个音节内部有 60ms 凹陷（< 默认 80ms 的停顿阈值），不应被切开
    parts = [silence(100)]
    for index in range(10):
        if index:
            parts.append(silence(150))
        parts.append(tone(ms=100))
        parts.append(silence(60))
        parts.append(tone(ms=100))
    parts.append(silence(100))
    clips = split_digit_clips(b"".join(parts))
    assert len(clips) == 10
    assert DEFAULT_DIGIT_SPLIT_MIN_GAP_MS == 80


def test_split_digit_clips_raises_when_segment_count_is_wrong():
    two_clips = silence(100) + tone(ms=200) + silence(200) + tone(ms=200) + silence(100)
    with pytest.raises(PhraseSplitError, match="expected 10"):
        split_digit_clips(two_clips)
    with pytest.raises(PhraseSplitError):
        split_digit_clips(b"")


async def test_prerender_digits_writes_ten_clips_then_skips(tmp_path):
    cache = make_cache(tmp_path)
    tts = FakeTts(voice_id="v1")

    stats = await prerender_phrases.prerender_digits(cache, tts)
    assert stats == {"generated": 10, "skipped": 0, "failed": 0}
    assert tts.calls == [DIGIT_SENTENCE]  # 只合成一次
    for digit in DIGIT_ORDER:
        clip = cache.get(digit_text(digit))
        assert clip and len(clip) / 2 / DEFAULT_SAMPLE_RATE * 1000 <= 300

    again = FakeTts(voice_id="v1")
    stats = await prerender_phrases.prerender_digits(cache, again)
    assert stats == {"generated": 0, "skipped": 10, "failed": 0}
    assert again.calls == []


async def test_prerender_digits_overwrites_partial_or_stale_clips(tmp_path):
    cache = make_cache(tmp_path)
    # 模拟 T06 的单字旧缓存：片段又长又不一致，且 10 个都在
    for digit in DIGIT_ORDER:
        cache.put(digit_text(digit), tone(amplitude=3000, ms=600))

    tts = FakeTts(voice_id="v1")
    stats = await prerender_phrases.prerender_digits(cache, tts, force=True)
    assert stats["generated"] == 10
    for digit in DIGIT_ORDER:
        clip = cache.get(digit_text(digit))
        assert len(clip) / 2 / DEFAULT_SAMPLE_RATE * 1000 <= 300

    # 缺一个片段时（不是 --force）也整批重做
    cache.path(digit_text("5")).unlink()
    stats = await prerender_phrases.prerender_digits(cache, FakeTts(voice_id="v1"))
    assert stats["generated"] == 10


def test_split_digit_clips_ignores_short_quiet_breath_at_the_end():
    # 实测（take 3）：句尾多出 50ms、峰值 RMS 1365 的尾音；正常数字段 >=110ms
    pcm = digit_sentence_pcm() + silence(400) + tone(amplitude=900, ms=50)
    clips = split_digit_clips(pcm)
    assert len(clips) == 10


def test_split_digit_clips_keeps_a_long_quiet_segment():
    # 又长又弱 => 是正常数字（例如轻声的"二"），不能当尾音丢掉
    pcm = digit_sentence_pcm() + silence(400) + tone(amplitude=900, ms=150)
    with pytest.raises(PhraseSplitError):
        split_digit_clips(pcm)


def test_split_digit_clips_still_raises_when_extra_segment_is_loud():
    pcm = digit_sentence_pcm() + silence(400) + tone(amplitude=8000, ms=200)
    with pytest.raises(PhraseSplitError, match="got 11"):
        split_digit_clips(pcm)


async def test_prerender_digits_picks_the_fastest_take(tmp_path, capsys):
    cache = make_cache(tmp_path)
    slow = digit_sentence_pcm(syllable_ms=400)
    fast = digit_sentence_pcm(syllable_ms=180)
    tts = FakeTts(voice_id="v1", digit_pcms=[slow, fast, slow])

    stats = await prerender_phrases.prerender_digits(cache, tts, takes=3)
    assert stats["generated"] == 10
    assert tts.calls == [DIGIT_SENTENCE] * 3
    out = capsys.readouterr().out
    assert "picked digit take 1" in out
    for digit in DIGIT_ORDER:
        clip_ms = len(cache.get(digit_text(digit))) / 2 / DEFAULT_SAMPLE_RATE * 1000
        assert 180 - 10 <= clip_ms <= 180 + 10


async def test_prerender_digits_retries_when_one_take_fails_to_split(tmp_path):
    cache = make_cache(tmp_path)
    tts = FakeTts(voice_id="v1", digit_pcms=[tone(ms=800), digit_sentence_pcm()])
    stats = await prerender_phrases.prerender_digits(cache, tts, takes=2)
    assert stats == {"generated": 10, "skipped": 0, "failed": 0}


async def test_prerender_digits_fails_loudly_when_split_fails(tmp_path, capsys):
    cache = make_cache(tmp_path)
    # 连续正弦波：切不出 10 段
    tts = FakeTts(voice_id="v1", digit_pcm=tone(ms=800))

    stats = await prerender_phrases.prerender_digits(cache, tts)
    assert stats == {"generated": 0, "skipped": 0, "failed": 10}
    # 不静默退回单字合成：缓存里不应出现任何数字片段
    for digit in DIGIT_ORDER:
        assert cache.get(digit_text(digit)) is None
    assert "切分失败" in capsys.readouterr().err


def test_parse_args_defaults():
    args = prerender_phrases.parse_args([])
    assert args.force is False
    assert args.backend is None
    assert args.phone is None
    assert args.verify is False


# ---- 与状态机（T07）的一致性 ----------------------------------------------- #


def test_phrases_live_only_in_core_phrases():
    # T06b：只保留一份话术表，dialog 从 phrases 导入
    assert dialog_module.PHRASES is PHRASES


def test_every_dialog_template_id_is_cached_with_identical_text():
    """Dialog 可能返回的每个 template_id（confirm_phone 除外）都要能命中缓存。"""
    owner = "张先生"
    texts = cache_texts(owner)
    ids = set(dialog_module.ASK_TEMPLATES.values()) | {
        "greet",
        "repeat",
        "goodbye",
        "forward",
    }
    for phrase_id in sorted(ids):
        assert phrase_id in texts, f"{phrase_id} 不在话术缓存里"
        assert phrase_id in dialog_module.PHRASES, f"{phrase_id} 不在状态机的话术表里"
        expected = dialog_module.PHRASES[phrase_id].format(owner=owner)
        assert texts[phrase_id] == expected
        assert PHRASES[phrase_id] == dialog_module.PHRASES[phrase_id]

    # 号码复述用的前后缀（Dialog 以 PHRASES[...] 访问）也要在缓存里
    for phrase_id in (PHONE_PREFIX, PHONE_SUFFIX):
        assert texts[phrase_id] == PHRASES[phrase_id]
        assert dialog_module.PHRASES[phrase_id] == PHRASES[phrase_id]


def test_all_states_map_to_cached_template_phrases():
    owner = "张先生"
    texts = cache_texts(owner)
    for state in State:
        for phrase_id in STATE_PHRASES[state]:
            assert texts[phrase_id] == phrase_text(phrase_id, owner)


# ---- 号码复述 -------------------------------------------------------------- #


def test_phone_groups_mobile_landline_and_fallback():
    assert phone_groups("13800138000") == ["138", "0013", "8000"]
    assert phone_groups("01012345678") == ["010", "12345678"]
    assert phone_groups("021-87654321") == ["021", "87654321"]
    assert phone_groups("0755-82345678") == ["0755", "82345678"]
    assert phone_groups("82345678") == ["8234", "5678"]
    assert phone_groups("1234567") == ["1234", "567"]
    with pytest.raises(ValueError):
        phone_groups("无")


def test_render_phone_order_and_gaps(tmp_path):
    cache = make_cache(tmp_path)
    prefix, suffix = PHONE_PREFIX, PHONE_SUFFIX
    clips = {
        "prefix": tone(freq_hz=100),
        "suffix": tone(freq_hz=110),
        "1": tone(freq_hz=200),
        "3": tone(freq_hz=300),
        "8": tone(freq_hz=800),
        "0": tone(freq_hz=1000),
    }
    put_clip(cache, cache_texts()["confirm_phone_prefix"], clips["prefix"])
    put_clip(cache, cache_texts()["confirm_phone_suffix"], clips["suffix"])
    for digit, pcm in clips.items():
        if len(digit) == 1:
            put_clip(cache, digit_text(digit), pcm)

    # 13800138000 -> 138 | 0013 | 8000；片段本身没有静音，裁剪不改内容
    # T06b：前缀与第一位之间、最后一位与后缀之间也插 group_gap
    expected: list[bytes] = [clips["prefix"], silence(DEFAULT_GROUP_GAP_MS)]
    for group_index, group in enumerate(["138", "0013", "8000"]):
        if group_index:
            expected.append(silence(DEFAULT_GROUP_GAP_MS))
        for digit_index, digit in enumerate(group):
            if digit_index:
                expected.append(silence(DEFAULT_DIGIT_GAP_MS))
            expected.append(clips[digit])
    expected.extend([silence(DEFAULT_GROUP_GAP_MS), clips["suffix"]])
    assert cache.render_phone("13800138000") == b"".join(expected)

    # 间隔长度与配置的量级对得上：组间 150ms、位间 40ms
    assert silence_ms(DEFAULT_GROUP_GAP_MS) == len(silence(150))
    assert silence_ms(DEFAULT_DIGIT_GAP_MS) == len(silence(40))
    assert DEFAULT_GROUP_GAP_MS > DEFAULT_DIGIT_GAP_MS


def test_render_phone_trims_clip_silence(tmp_path):
    cache = make_cache(tmp_path)
    lead = silence(30)
    tail = silence(50)
    speech = tone(amplitude=6000, ms=60)
    noisy_lead = b"\x10\x00" * 800  # 底噪（RMS 16 < 阈值 200）
    cache.put(cache_texts()["confirm_phone_prefix"], lead + noisy_lead + speech + tail)
    cache.put(cache_texts()["confirm_phone_suffix"], speech)
    cache.put(digit_text("8"), lead + speech + tail)
    # 只用一位数字，避免其它片段缺失
    pcm = cache.render_phone("8")
    # 三段的首尾静音都被裁掉（只剩 speech），前缀→数字、数字→后缀各有一段组间静音
    assert pcm == speech + silence(DEFAULT_GROUP_GAP_MS) + speech + silence(DEFAULT_GROUP_GAP_MS) + speech
    assert len(pcm) < len(lead + noisy_lead + speech + tail) + len(speech) + len(lead + speech + tail) + 2 * len(
        silence(DEFAULT_GROUP_GAP_MS)
    )


def test_render_phone_rejects_empty_number(tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(ValueError):
        cache.render_phone("  ")


def test_render_phone_missing_clip_raises_with_hint(tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(PhraseMissing, match="prerender_phrases.py"):
        cache.render_phone("13800138000")


# ---- 静音裁剪 -------------------------------------------------------------- #


def test_trim_silence_removes_edges_only():
    speech = tone(amplitude=6000, ms=100)
    pcm = silence(40) + speech + silence(40)
    trimmed = trim_silence(pcm)
    assert trimmed == speech
    assert np.frombuffer(trimmed, dtype="<i2").max() == np.frombuffer(speech, dtype="<i2").max()


def test_trim_silence_threshold_is_configurable():
    quiet = tone(amplitude=100, ms=50)  # RMS≈71
    pcm = silence(20) + quiet + silence(20)
    # 默认阈值 200：整段都被当静音裁掉
    assert trim_silence(pcm) == b""
    # 阈值调低后保留
    assert trim_silence(pcm, threshold=10) != b""


def test_trim_silence_handles_empty_and_odd_length():
    assert trim_silence(b"") == b""
    assert trim_silence(silence(20)) == b""
    # 奇数长度（不该出现）截掉半个样本，不抛异常
    assert trim_silence(tone() + b"\x01") != b""


def test_silence_length():
    assert silence(40) == b"\x00\x00" * int(DEFAULT_SAMPLE_RATE * 0.04)
    assert silence(0) == b""


def test_write_phone_wav_roundtrip(tmp_path):
    cache = make_cache(tmp_path)
    texts = cache_texts()
    cache.put(texts["confirm_phone_prefix"], tone(amplitude=3000, ms=30))
    cache.put(texts["confirm_phone_suffix"], tone(amplitude=3000, ms=30))
    for digit in "38":
        cache.put(digit_text(digit), tone(amplitude=3000, ms=30))
    pcm = cache.render_phone("38")

    import soundfile as sf

    out = tmp_path / "phone.wav"
    prerender_phrases.write_phone_wav(out, pcm, cache.sample_rate)
    data, rate = sf.read(str(out), dtype="int16")
    assert rate == DEFAULT_SAMPLE_RATE
    assert len(data) == len(pcm) // 2


def test_run_writes_cache_and_phone_wav(tmp_path, monkeypatch):
    config = {
        "siliconflow": {"base_url": "http://127.0.0.1:1/v1"},
        "owner": {"name": "张先生"},
        "tts": {
            "backend": "local",
            "voice": "",
            "local_url": "http://127.0.0.1:1",
            "phrase_cache": {"dir": str(tmp_path / "cache")},
        },
        "audio": {"sample_rate": DEFAULT_SAMPLE_RATE},
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    tts = FakeTts(voice_id="v1")
    monkeypatch.setattr(prerender_phrases, "build_tts", lambda cfg, backend: tts)
    phone_wav = tmp_path / "phone.wav"
    args = argparse.Namespace(
        config=config_path,
        backend=None,
        voice_id=None,
        owner=None,
        cache_dir=None,
        force=False,
        phone="13800138000",
        phone_wav=str(phone_wav),
        verify=False,
        digit_takes=1,
    )
    assert asyncio.run(prerender_phrases.run(args)) == 0
    assert (tmp_path / "cache" / "local").is_dir()
    assert phone_wav.is_file()
    assert len(list((tmp_path / "cache" / "local").glob("*.pcm"))) == len(cache_texts("张先生"))

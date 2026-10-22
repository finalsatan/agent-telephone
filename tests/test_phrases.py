"""T06 离线测试：话术表、缓存读写、号码复述拼接与静音裁剪（全部用 fake TTS，不联网）。"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pytest

from core.phrases import (
    DEFAULT_DIGIT_GAP_MS,
    DEFAULT_GROUP_GAP_MS,
    DEFAULT_SAMPLE_RATE,
    DIGIT_READINGS,
    LOCAL_VOICE_PLACEHOLDER,
    PHRASES,
    PHONE_PREFIX,
    PHONE_SUFFIX,
    STATE_PHRASES,
    PhraseCache,
    PhraseMissing,
    backend_name,
    cache_texts,
    digit_text,
    fill_owner,
    phrase_text,
    phone_groups,
    resolve_voice_key,
    silence,
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


class FakeTts:
    """只实现 T06 用到的 synth 接口，按文本产出确定性 PCM。"""

    sample_rate = DEFAULT_SAMPLE_RATE

    def __init__(self, *, voice_id: str | None = None) -> None:
        self.calls: list[str] = []
        self._voice_id = voice_id

    @property
    def voice_id(self) -> str | None:
        return self._voice_id

    async def synth(self, text: str, cancel: asyncio.Event):
        self.calls.append(text)
        await asyncio.sleep(0)  # 模拟异步产出
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
    assert PHRASES["forward"] == "我会转告，请您留下联系方式。"


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
    assert texts[PHONE_PREFIX] == "您的回拨电话是"
    assert texts[PHONE_SUFFIX] == "，对吗？"
    for digit, reading in DIGIT_READINGS.items():
        assert texts[f"digit_{digit}"] == reading
    assert len(texts) == len(PHRASES) + 2 + 10


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


def test_cache_is_written_atomically(tmp_path):
    cache = make_cache(tmp_path)
    cache.put("您好", tone())
    assert [p.name for p in cache.directory.iterdir()] == [cache.path("您好").name]


# ---- 预合成 ---------------------------------------------------------------- #


async def test_prerender_generates_everything_then_skips(tmp_path):
    cache = make_cache(tmp_path)
    tts = FakeTts(voice_id="v1")
    texts = cache_texts("张先生")

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
    texts = cache_texts("张先生")
    await prerender_phrases.prerender(cache, FakeTts(), texts)

    tts = FakeTts()
    stats = await prerender_phrases.prerender(cache, tts, texts, force=True)
    assert stats["generated"] == len(texts)
    assert tts.calls == list(texts.values())


async def test_changed_text_regenerates_only_the_changed_phrases(tmp_path):
    cache = make_cache(tmp_path)
    await prerender_phrases.prerender(cache, FakeTts(), cache_texts("张先生"))

    # 换了机主称呼：只有含 {owner} 的两句话术变了，其余照旧命中缓存
    texts = cache_texts("王女士")
    tts = FakeTts()
    stats = await prerender_phrases.prerender(cache, tts, texts)
    assert stats["generated"] == 2
    assert tts.calls == [texts["greet"], texts["goodbye"]]
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

    stats = await prerender_phrases.prerender(cache, FlakyTts(), cache_texts("张先生"))
    assert stats["failed"] == 1
    assert stats["generated"] == len(cache_texts("张先生")) - 1


def test_parse_args_defaults():
    args = prerender_phrases.parse_args([])
    assert args.force is False
    assert args.backend is None
    assert args.phone is None


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
    expected: list[bytes] = [clips["prefix"]]
    for group_index, group in enumerate(["138", "0013", "8000"]):
        if group_index:
            expected.append(silence(DEFAULT_GROUP_GAP_MS))
        for digit_index, digit in enumerate(group):
            if digit_index:
                expected.append(silence(DEFAULT_DIGIT_GAP_MS))
            expected.append(clips[digit])
    expected.append(clips["suffix"])
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
    # 前缀/数字/后缀的静音都被裁掉；按 10ms 帧对齐，因此允许一个帧的余量
    assert len(pcm) < len(lead + noisy_lead + speech + tail) + len(speech) + len(lead + speech + tail)
    assert pcm.startswith(speech[:100])
    assert pcm.endswith(speech[-100:])


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
    )
    assert asyncio.run(prerender_phrases.run(args)) == 0
    assert (tmp_path / "cache" / "local").is_dir()
    assert phone_wav.is_file()
    assert len(list((tmp_path / "cache" / "local").glob("*.pcm"))) == len(cache_texts("张先生"))

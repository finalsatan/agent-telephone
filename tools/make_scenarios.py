"""生成 T12 的场景集：来电者语音用 TTS 合成，每轮一个 wav。

用法：

    # 生成缺失的场景（已存在的跳过）；默认用云端 TTS 的另一个音色当来电者
    uv run python tools/make_scenarios.py

    # 全部重生成（换了音色或改了脚本）
    uv run python tools/make_scenarios.py --force

    # 只生成某几条
    uv run python tools/make_scenarios.py --only s01_delivery,s06_correction

产物：

    scenarios/<id>.json         场景描述、期望槽位、每轮 {"text","wav","gap_ms"}
    scenarios/<id>/turn_01.wav  第 1 轮来电者音频（16kHz 单声道 16-bit）
    scenarios/<id>/turn_02.wav  ...

来电者音色必须与 AI 不同：AI 走本地 CosyVoice3（单参考音色），这里默认用硅基流动
CosyVoice2 的另一个预设音色（`config.yaml: scenarios.caller_voice`，默认 `:anna`）。

另外写入 5 个真人录音的占位 json（`real_01.json` … `real_05.json`）：把录音放到
`scenarios/real_01.wav`，再把每轮文本填进 `turns`（或保留 `wav` 用整条模式）。
占位场景 `pending=true`，评估脚本会跳过，直到用户填好。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.phrases import silence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT / "scenarios"
DEFAULT_CALLER_VOICE = "FunAudioLLM/CosyVoice2-0.5B:anna"
DEFAULT_GAP_MS = 800.0
DEFAULT_SAMPLE_RATE = 16000

# 每条场景：turns[].say 可以是字符串，也可以是 [(文本, 停顿毫秒), ...]（同一条 wav 内拼接）。
# text 只用于场景 json 里的人类可读基准，不参与回放（回放用 wav）。
SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "s01_delivery",
        "desc": "快递送货：自报姓名、来意，报 11 位手机号",
        "expected_slots": {
            "name": ["李", "李先生"],
            "purpose": ["送快递", "送包裹"],
            "callback_number": "13800138000",
            "time": ["今天下午"],
        },
        "turns": [
            {"say": "喂，你好，我找一下张先生。", "gap_ms": 800},
            {"say": "我姓李，是送快递的，有个包裹今天下午送到。", "gap_ms": 800},
            {"say": "我的回拨电话是一三八零零一三八零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s02_food_delivery",
        "desc": "外卖：报身份、来意，报号码",
        "expected_slots": {
            "name": ["外卖员", "送外卖的", "外卖"],
            "purpose": ["送外卖"],
            "callback_number": "13900139000",
        },
        "turns": [
            {"say": "你好，请找一下张先生。", "gap_ms": 800},
            {"say": "我是送外卖的，他点的餐已经到楼下了。", "gap_ms": 800},
            {"say": "电话是幺三九零零幺三九零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s03_appointment",
        "desc": "预约确认：带时间，只记录不承诺",
        "expected_slots": {
            "name": ["王", "王先生"],
            "purpose": ["确认预约", "装修预约", "确认装修预约"],
            "callback_number": "13700137000",
            "time": ["明天上午十点", "明天上午10点"],
        },
        "turns": [
            {"say": "喂，张先生在吗？", "gap_ms": 800},
            {"say": "我姓王，想跟他确认一下明天上午十点的装修预约。", "gap_ms": 800},
            {"say": "回拨电话幺三七零零一三七零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s04_sales",
        "desc": "陌生推销：介绍理财产品，留号码",
        "expected_slots": {
            "name": ["赵", "赵先生"],
            "purpose": ["介绍理财产品", "推销理财", "理财"],
            "callback_number": "13600136000",
        },
        "turns": [
            {"say": "您好，我这边是安泰保险公司的。", "gap_ms": 800},
            {"say": "想跟张先生介绍一下我们的理财产品。", "gap_ms": 800},
            {"say": "我姓赵，电话是幺三六零零一三六零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s05_message",
        "desc": "一般留言：同事让机主回电话",
        "expected_slots": {
            "name": ["陈", "老陈", "陈先生"],
            "purpose": ["回电话", "让张先生回电话", "同事留言"],
            "callback_number": "13500135000",
        },
        "turns": [
            {"say": "你好，张先生在吗？", "gap_ms": 800},
            {"say": "我是他同事老陈，他回来让他给我回个电话。", "gap_ms": 800},
            {"say": "号码是幺三五零零一三五零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s06_correction",
        "desc": "纠正：先把时间说错，再改成后天下午三点",
        "expected_slots": {
            "name": ["刘", "刘先生"],
            "purpose": ["周末聚会", "聚会"],
            "callback_number": "13400134000",
            "time": ["后天下午三点", "后天下午3点"],
        },
        "turns": [
            {"say": "你好，我找张先生。", "gap_ms": 800},
            {"say": "我姓刘，想问一下周末聚会的事。", "gap_ms": 800},
            {"say": "时间是明天下午三点，不对，是后天下午三点。", "gap_ms": 800},
            {"say": "电话幺三四零零一三四零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s07_long_number",
        "desc": "长号码分段报：座机 010-8888-6666",
        "expected_slots": {
            "name": ["周", "周先生"],
            "purpose": ["办公室租约", "租约"],
            "callback_number": "01088886666",
        },
        "turns": [
            {"say": "喂，你好，我找张先生。", "gap_ms": 800},
            {"say": "我姓周，想问下办公室租约的事。", "gap_ms": 800},
            {"say": "我的座机是零幺零，八八八八，六六六六。", "gap_ms": 800},
        ],
    },
    {
        "id": "s08_address",
        "desc": "含地址：送货地址保留原话",
        "expected_slots": {
            "name": ["吴", "吴先生"],
            "purpose": ["送东西", "送货", "送到"],
            "callback_number": "13300133000",
            "location": ["朝阳区建国路88号", "朝阳区建国路八十八号"],
        },
        "turns": [
            {"say": "你好，张先生在家吗？", "gap_ms": 800},
            {"say": "我姓吴，想把东西送到朝阳区建国路八十八号。", "gap_ms": 800},
            {"say": "电话幺三三零零一三三零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s09_thinking_pause",
        "desc": "思考停顿：一条 wav 里有长停顿，AI 不能把半句话当一轮",
        "expected_slots": {
            "name": ["孙", "孙先生"],
            "purpose": ["周末聚会", "聚会"],
            "callback_number": "13200132000",
        },
        "turns": [
            {
                "say": [
                    ("呃，我想一下……", 900),
                    ("我姓孙，想问下周末的聚会。", 0),
                ],
                "gap_ms": 800,
            },
            {"say": "电话是幺三二零零一三二零零零。", "gap_ms": 800},
        ],
    },
    {
        "id": "s10_early_hangup",
        "desc": "提前挂断：对方表示没事，直接告别",
        "expected_slots": {},
        "turns": [
            {"say": "喂，你好。", "gap_ms": 800},
            {"say": "没事了，就这样吧，再见。", "gap_ms": 800},
        ],
    },
]

REAL_PLACEHOLDERS: list[dict[str, Any]] = [
    {
        "id": f"real_{index:02d}",
        "desc": f"真人录音 {index:02d}（待填）：把 wav 放到 scenarios/real_{index:02d}.wav，"
        "再填写 expected_slots 与 turns（或保留 wav 用整条模式）",
        "expected_slots": {},
        "wav": f"real_{index:02d}.wav",
        "turns": [],
        "pending": True,
    }
    for index in range(1, 6)
]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def turn_text(turn: Mapping[str, Any]) -> str:
    """把 `say`（字符串或 [(文本, 停顿)]）拼成基准文本。"""
    say = turn.get("say") or ""
    if isinstance(say, str):
        return say
    return "".join(str(text) for text, _pause in say)


def write_wav(path: Path, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)


async def synth_text(tts: Any, text: str, *, attempts: int = 3) -> bytes:
    """把一段文本合成成 16k 单声道 PCM（流式读完）。

    云端 TTS 偶尔会返回 200 但没有音频（短句、语气词更容易触发）；这里重试几次，
    全部失败才报错，避免一条场景毁掉整批生成。
    """
    import asyncio

    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        chunks: list[bytes] = []
        try:
            async for chunk in tts.synth(text, asyncio.Event()):
                if chunk:
                    chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001 - 重试后仍失败才向上抛
            last_error = exc
        if chunks:
            return b"".join(chunks)
        if attempt + 1 < attempts:
            await asyncio.sleep(1.0 + attempt)
    detail = f"：{last_error}" if last_error else "（返回 200 但 0 字节）"
    raise RuntimeError(f"TTS returned no audio for {text!r}{detail}")


async def synth_turn(tts: Any, turn: Mapping[str, Any], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """合成一轮的音频；`say` 是列表时在片段之间插入静音（思考停顿）。"""
    say = turn.get("say") or ""
    if isinstance(say, str):
        return await synth_text(tts, say)

    parts: list[bytes] = []
    for item in say:
        text, pause_ms = item if isinstance(item, (list, tuple)) else (item, 0)
        pcm = await synth_text(tts, str(text))
        if not pcm:
            raise RuntimeError(f"TTS returned no audio for caller text: {text!r}")
        parts.append(pcm)
        if pause_ms:
            parts.append(silence(float(pause_ms), sample_rate=sample_rate))
    return b"".join(parts)


def scenario_payload(spec: Mapping[str, Any], turn_wavs: Sequence[str]) -> dict[str, Any]:
    """生成场景 json 的内容（与 `tools/replay.load_scenario()` 的格式一致）。"""
    default_gap = float(spec.get("default_gap_ms", DEFAULT_GAP_MS))
    return {
        "id": str(spec["id"]),
        "desc": str(spec.get("desc") or ""),
        "expected_slots": dict(spec.get("expected_slots") or {}),
        "turns": [
            {
                "text": turn_text(turn),
                "wav": wav,
                "gap_ms": float(turn.get("gap_ms", default_gap)),
            }
            for turn, wav in zip(spec.get("turns") or [], turn_wavs)
        ],
    }


async def write_scenario(
    spec: Mapping[str, Any],
    scenarios_dir: Path,
    tts: Any,
    *,
    force: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bool:
    """写一条场景；返回是否真的合成了音频（已存在且未 force 时返回 False）。"""
    scenario_id = str(spec["id"])
    json_path = scenarios_dir / f"{scenario_id}.json"
    audio_dir = scenarios_dir / scenario_id
    turns = list(spec.get("turns") or [])
    wav_names = [f"turn_{index:02d}.wav" for index in range(1, len(turns) + 1)]
    payload = scenario_payload(spec, wav_names)
    missing = [name for name in wav_names if not (audio_dir / name).is_file()]

    if not force and json_path.is_file() and not missing:
        return False

    # json 已存在时不覆盖（保护用户手改的 expected_slots）；--force 才重写。
    if force or not json_path.is_file():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for index, (turn, wav_name) in enumerate(zip(turns, wav_names), start=1):
        wav_path = audio_dir / wav_name
        if wav_path.is_file() and not force:
            continue
        print(f"  [{scenario_id}] 合成 {wav_name}：{turn_text(turn)[:30]}")
        pcm = await synth_turn(tts, turn, sample_rate=sample_rate)
        if not pcm:
            raise RuntimeError(f"TTS returned no audio for {scenario_id} {wav_name}")
        write_wav(wav_path, pcm, sample_rate)
    return True


def write_real_placeholders(scenarios_dir: Path) -> None:
    for spec in REAL_PLACEHOLDERS:
        path = scenarios_dir / f"{spec['id']}.json"
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_caller_tts(config: Mapping[str, Any], voice: str) -> Any:
    """来电者语音用云端 TTS（本地服务只有一个参考音色，无法与 AI 区分）。"""
    from services.tts import TtsClient

    tts_cfg = config.get("tts") or {}
    return TtsClient(
        base_url=config["siliconflow"]["base_url"],
        model=str(tts_cfg.get("model") or "FunAudioLLM/CosyVoice2-0.5B"),
        voice=voice,
        sample_rate=int((config.get("audio") or {}).get("sample_rate", DEFAULT_SAMPLE_RATE)),
    )


async def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    scenarios_dir = Path(args.scenarios_dir)
    voice = (
        args.caller_voice
        or (config.get("scenarios") or {}).get("caller_voice")
        or DEFAULT_CALLER_VOICE
    )
    ai_voice = str((config.get("tts") or {}).get("voice") or "")
    if voice == ai_voice:
        print(f"来电者音色不能与 AI 相同：{voice}", file=sys.stderr)
        return 2

    only = {item.strip() for item in (args.only or "").split(",") if item.strip()} or None
    specs = [spec for spec in SCENARIOS if only is None or spec["id"] in only]
    if only:
        unknown = only - {spec["id"] for spec in SCENARIOS}
        if unknown:
            print(f"未知场景：{', '.join(sorted(unknown))}", file=sys.stderr)
            return 2

    scenarios_dir.mkdir(parents=True, exist_ok=True)
    write_real_placeholders(scenarios_dir)
    tts = build_caller_tts(config, voice)
    print(f"caller voice: {voice}（AI 音色：{ai_voice or '本地参考音色'}）")
    created = 0
    try:
        for spec in specs:
            print(f"[{spec['id']}] {spec.get('desc', '')}")
            if await write_scenario(spec, scenarios_dir, tts, force=args.force):
                created += 1
    finally:
        aclose = getattr(tts, "aclose", None)
        if aclose is not None:
            await aclose()

    print(f"完成：{created} 条场景生成/更新，{len(specs) - created} 条已存在跳过")
    print(f"真人录音占位：{scenarios_dir}/real_01.json … real_05.json")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T12 场景集生成（来电者语音）")
    parser.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS))
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--caller-voice", default=None, help="默认取 scenarios.caller_voice")
    parser.add_argument("--only", default=None, help="只生成这些场景 id（逗号分隔）")
    parser.add_argument("--force", action="store_true", help="重新合成并覆盖 json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

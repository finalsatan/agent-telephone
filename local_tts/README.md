# 本地 CosyVoice3 TTS 服务（T05）

在 WSL2 + RTX 4060 Ti 16G 上部署 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，包成
FastAPI 流式服务，作为实时链路的主 TTS 后端（`config.yaml: tts.backend: local`）。

> **T05 结论（详见 `docs/progress.md`）**
> 服务可用，25/25 次压测无失败。首包 P50 3500ms / P95 5278ms，比云端 T04
> （4358 / 8835ms）明显更稳；但**未达到任务的目标参考“短句首包 P50 ≤ 0.8s”**。
> 原因在模型侧：首块至少要攒够一个 token hop（约 1s 音频），短句实际要等整句生成完；
> 4060 Ti 上 fp16 的实时率约 0.95–1.3×。详见文末“实测与结论”。

## 1. 目录与文件

```
local_tts/
  README.md     本文件
  server.py     FastAPI 服务（模型常驻 + 启动预热）
  bench.py      5 句 × n 次压测：首包/总耗时/实时率/GPU 显存
services/tts_local.py       客户端（与云端 TtsClient 同一接口与超时/取消语义）
tests/test_tts_local.py     客户端离线测试
tests/live/test_tts_local_live.py  连已启动服务的 live 测试（可选）
```

模型、CosyVoice 官方仓库、conda 环境都在仓库外，不入库：

- 官方仓库：`~/code/ai/CosyVoice`
- conda 环境：`~/miniconda3/envs/cosyvoice`（Python 3.10）
- 模型：`~/code/ai/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B`（约 9.1GB）

## 2. 安装（实际执行过的步骤）

```sh
# 1) 官方仓库 + 子模块（GitHub 在本网络走 SSH over 443）
cd ~/code/ai
git clone --depth 1 git@github.com:FunAudioLLM/CosyVoice.git CosyVoice
cd CosyVoice
git submodule update --init --recursive --depth 1

# 2) 按官方 README 建 conda 环境
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice

# 3) 先装 torch，再装其余依赖（见下面“安装问题”）
pip install "setuptools<81" wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.3.1 torchaudio==2.3.1 \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 Cython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wheel_stub -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-build-isolation -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 验证 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.3.1+cu121 True
```

模型下载（ModelScope，国内较快）：

```sh
cd ~/code/ai/CosyVoice
python - <<'EOF'
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
EOF
```

### 安装问题与处理

| 问题 | 处理 |
|---|---|
| `pip install -r requirements.txt` 在 `openai-whisper` 报 `No module named 'pkg_resources'` | 新版 setuptools 移除了 pkg_resources；先装 `setuptools<81`，再 `--no-build-isolation`。openai-whisper 只用于训练/评测，推理不需要。 |
| `pyworld` 元数据生成报 `No module named 'numpy'` | `--no-build-isolation` 下构建依赖不会自动装；先装 `numpy==1.26.4` 和 `Cython`。 |
| `tensorrt-cu12` 元包报 `Cannot import 'wheel_stub.buildapi'` | 装 `wheel_stub` 后再装。TensorRT 只有 `load_trt=True` 才用，本服务未启用。 |
| `torch` 从 PyTorch 源下载慢 | 走清华 PyPI 主源 + `download.pytorch.org/whl/cu121` 作为 extra index；PyPI 的 2.3.1 Linux wheel 本身带 cu121。 |
| wetext 首次初始化需要下载 FST 资源 | 首次启动会自动下载到 `~/.cache`，之后不再联网；没有网络时会退化为不做文本正则化。 |

官方 `requirements.txt` 里的 `deepspeed`、`tensorrt-*`、`gradio`、`openai-whisper`
推理都不需要，但为了与官方安装保持一致仍然装上了（不做裁剪）。

## 3. 启动服务

```sh
conda activate cosyvoice
cd /path/to/agent-telephone   # 本仓库（worktree 也可以）
python local_tts/server.py --fp16
# INFO: Uvicorn running on http://127.0.0.1:50000
```

- 默认 `--fp16`（T05 实测比 fp32 快约 30–50%）；`--no-fp16` 关闭。
- 模型目录默认 `$COSYVOICE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B`，
  `COSYVOICE_ROOT` 默认 `~/code/ai/CosyVoice`。
- 参考音色默认用官方 `asset/zero_shot_prompt.wav` + 对应提示文本，
  启动时通过 `add_zero_shot_spk()` 缓存，避免每次请求重新编码提示音频。
- 启动流程：加载模型 → 预热一次 `"您好。"` → 开始接受请求。本机实测
  `/health` 变为 `ok` 约需 **57s**（含首次 wetext 资源下载；之后约 20s）。

冒烟测试：

```sh
curl -s http://127.0.0.1:50000/health
# {"status":"ok","sample_rate":16000,"model_sample_rate":24000,"load_s":57.31,"fp16":true}

# 保存为 16kHz/16-bit/单声道 PCM（raw，不是 wav）
curl -s -X POST http://127.0.0.1:50000/synth \
  -H 'Content-Type: application/json' \
  -d '{"text":"您好，请问您是哪位？"}' -o /tmp/tts.pcm
```

### 接口

`POST /synth`

```json
{"text": "您好，请问您是哪位？"}
```

返回 `application/octet-stream`，裸 PCM：**16kHz、单声道、16-bit 小端**，
响应头带 `X-Sample-Rate: 16000`。空文本返回 400，超过 500 字返回 413。
服务内按请求串行推理（一把进程内锁），T05 只要求单路实时合成。

模型内部是 24kHz。服务用 `soxr.ResampleStream` 做**流式**重采样（保留状态），
不是每块独立重采样，避免块边界出现咔哒声。

## 4. 客户端

```python
from services.tts_local import LocalTtsClient

client = LocalTtsClient.from_config(config)   # 读 tts.local_url / audio.sample_rate
cancel = asyncio.Event()
async for pcm in client.synth("您好，请问您是哪位？", cancel):
    ...  # 16kHz PCM 块
```

`LocalTtsClient` 继承 `services.tts.TtsClient`，因此 T04/T04b 的语义原样保留：

- 首包/总时限只包住单次网络等待，`yield` 不在超时范围内；
- `cancel` 置位后立即结束，退避期间被打断也不泄漏内部异常；
- 未产出音频前才重试 429/5xx/传输错误，已开始输出不重试；
- 空文本不发请求。

本地服务不鉴权：`LocalTtsClient` 显式传 `api_key=""`，不会发 `Authorization` 头。

## 5. 压测

先启动服务，再在仓库根目录：

```sh
uv run python local_tts/bench.py --n 5
```

`bench.py` 会先等 `/health` 就绪、预热一次，然后对 5 句不同长度的话各测 5 次，
打印每句与总体的首包/总耗时 P50、P95，实时率（合成耗时/音频时长），
以及后台采样的整卡 GPU 显存峰值。

## 6. 实测与结论

环境：WSL2 + RTX 4060 Ti 16G，`--fp16`，2026-10-21 18:15 CST。
本地服务与 T04 的云端数据在同一台机器、同一网络下测得（云端是不同时段）。

| 句长（字） | 本地首包 P50 / P95 (ms) | 云端首包 P50 (ms) | 本地总耗时 P50 (ms) | 音频 P50 (ms) | 实时率 P50 |
|---|---|---|---|---|---|
| 3 | **1098 / 1465** | 3593 | 1116 | 840 | 1.33 |
| 7 | **1917 / 2643** | 3644 | 1938 | 1840 | 1.05 |
| 18 | **3500 / 3673** | 5833 | 3529 | 3720 | 0.95 |
| 25 | **4513 / 5278** | 8748 | 5452 | 5680 | 0.96 |
| 49 | 5010 / 5598 | 3475* | 10535 | 10600 | 0.99 |
| **总体** | **3500 / 5278**（min 988 / max 5714） | 4358 / 8835 | **3529 / 10624** | — | — |

\* 云端这一句恰好赶上一次快速突发；云端同一时段整体很不稳定（T04 有 12/50 次超时）。

- 本地 25/25 次成功，0 超时；GPU 显存 baseline 9298 MiB、peak 9765 MiB
  （整卡采样，含其它进程；增量约 467 MiB，模型常驻约 4.9GB allocated）。
- **是否比云端快至少 1 秒**：短句和中等长度是（3 字 −2.5s、7 字 −1.7s、18 字 −2.3s、
  25 字 −4.2s），整体首包 P50 只快 **0.86s**（未到 1s），P95 快 **3.56s**。
  49 字这一句本地反而慢 1.5s，但云端那次是突发，不代表稳定水平。
- **未达到“短句首包 P50 ≤ 0.8s”**。原因不是网络，而是模型/实现：
  CosyVoice3 的流式推理至少要攒够一个 `token_hop_len`（25 个 token，约 1s 音频）
  才产出第一块；短句（3–7 字）的 token 总数接近这个阈值，等于要等整句生成完。
  4060 Ti 上 fp16 的实时率约 0.95–1.3×，首包与音频长度基本成正比。
- **已核实的替代方案（同一台机器、fp16）**：整句非流式合成的实测总耗时为
  3 字 1161ms、7 字 2117ms、18 字 3505ms、25 字 4489ms、49 字 8618ms；
  对短句与流式首包几乎一样，对长句反而更差（要等整句），因此**不能**用它替代流式。
- 其他可选路径（本次未实现/未验证，供评审决定）：启用 TensorRT（仓库自带
  `flow.decoder.estimator.fp32.onnx` 可现场编译 plan）或 JIT；把 `token_hop_len`
  调小（官方注释要求与训练时的 static_chunk_size 一致，需先验证音质/重复推理）；
  在更强的 GPU 上部署；或保留云端 TTS 并用 T06 预合成覆盖高频话术。
  CosyVoice2-0.5B 本地部署也是候选，但本次未完成下载与压测，不能声称已核实。

## 7. 已知限制

- 单进程串行推理；多路并发会排队（T10 是一路通话，够用）。
- 客户端断开后，服务端不保证立刻中断已开始的 GPU 推理（会跑完当前请求）；
  客户端侧 `cancel` 会立即停止读取并关闭连接。
- 未安装 `ttsfrd`，文本正则化用 wetext；号码读法由 T06 的数字片段拼接解决，
  不依赖 TTS 自己读数字。
- 参考音色固定为官方示例音色；换音色可传 `--prompt-wav` / `--prompt-text`。

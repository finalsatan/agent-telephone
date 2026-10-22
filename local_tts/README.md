# 本地 CosyVoice3 TTS 服务（T05）

在 WSL2 + RTX 4060 Ti 16G 上部署 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，包成
FastAPI 流式服务，作为实时链路的主 TTS 后端（`config.yaml: tts.backend: local`）。

> **T05/T05b 结论（详见 `docs/progress.md`）**
> 服务可用。T05b 修复了上游 `token_hop_len` 在请求之间不复位的问题：首包
> P50 从 3500ms 降到 **2281ms**，P95 从 5278ms 降到 **3220ms**，长句改善明显
> （18/25/49 字从 3500/4513/5010ms 降到 2300/2301/2798ms）。
> 但短句没有变快（3 字 1098→1485ms、7 字 1917→2075ms），总耗时反而上升
> （分块更细、每块都有 flow/hift 开销，RTF 1.06–1.64），
> **仍未达到“短句首包 P50 ≤ 0.8s”的目标**。
> 另外，打断后立刻重说时，下一句首包实测 **P50 10.9s**（正常 2.1s），
> 因为服务端要等上一句跑完才释放锁；T10 的插话打断不能依赖当前实现。

## 1. 目录与文件

```
local_tts/
  README.md     本文件
  server.py     FastAPI 服务（模型常驻 + 启动预热）
  bench.py      5 句 × n 次压测：首包/总耗时/块数/实时率/GPU 显存与利用率/打断重说
services/tts_local.py       客户端（与云端 TtsClient 同一接口；含 voice_id）
tests/test_tts_local.py     客户端离线测试
tests/test_local_tts_server.py    服务端离线测试（fake engine）
tests/live/test_tts_local_live.py 连已启动服务的 live 测试（可选）
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

本次使用的官方仓库 commit：`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`。
T05b 的 hop 复位依赖该 commit 的 `CosyVoice2Model.tts()` 行为（它会把
`self.token_hop_len` 从 25 逐块翻倍到 100 且不复位）；升级官方仓库后需重新验证。

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
- 启动流程：加载模型 → 缓存参考音色 → 预热一次 `"您好。"` → 开始接受请求。
  本机实测 `/health` 变为 `ok` 约需 **28–57s**（首次要下载 wetext 资源时更久）。
- 启动日志会打印参考音色的 token 数与 pad。当前官方音色：`prompt_tokens=87`、
  `initial_hop=25`、`pad=13`，即首块门槛约 38 个 token（首块音频约 452ms）。
- 每次请求开始前会把 `engine.model.token_hop_len` 复位为加载时的初始值（T05b）。
- `--vllm` 开关已加（`AutoModel(..., load_vllm=True)`），但本次没有完成 vLLM
  环境安装与实测，见文末。

冒烟测试：

```sh
curl -s http://127.0.0.1:50000/health
# {"status":"ok","sample_rate":16000,"model_sample_rate":24000,"load_s":57.31,"fp16":true,
#  "voice_id":"ba97985b9879","initial_token_hop_len":25,
#  "prompt_token_count":87,"prompt_token_pad":13,"vllm":false}

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

`voice_id`（T05b，供 T06 缓存键使用）：参考音频内容 + 提示文本的 sha1 前 12 位。
首次 `synth()` 前会自动从 `/health` 取；也可以显式：

```python
await client.fetch_voice_id()
print(client.voice_id)   # 例：ba97985b9879
```

## 5. 压测

先启动服务，再在仓库根目录：

```sh
uv run python local_tts/bench.py --n 5
```

`bench.py` 会先等 `/health` 就绪、预热一次，然后对 5 句不同长度的话各测 5 次，
打印每句与总体的首包/总耗时 P50、P95、块数、首块音频时长、实时率
（合成耗时/音频时长），以及后台采样的整卡 GPU 显存与利用率。

- `--wav-dir /tmp/t05b_wavs`：每句第一次合成的音频存成 wav，便于试听/ASR 检查。
- 最后会自动跑“打断后重说”：49 字句收到第一块后 cancel，紧接着合成 7 字句，
  记录后者的首包（5 次）。服务端要等上一句跑完，这个数会明显高于正常值。

## 6. 实测与结论

环境：WSL2 + RTX 4060 Ti 16G，`--fp16`。T05 数据采集于 2026-10-21 18:15，
T05b 数据采集于 2026-10-22 09:38（服务端为新代码，专用端口 50001，避免与 T06
同时使用 50000 的请求互相干扰）。两轮都包含 5 句 ×5 次。

### 6.1 T05b 首包修复：hop 复位

上游 `CosyVoice2Model.tts()` 在流式循环里把实例属性 `token_hop_len` 从 25
逐块翻倍到 100，并且请求结束后不复位。服务常驻时，第一句长话之后所有请求的
首块门槛都变成 `100 + pad`，这就是 T05 首包 3.5–5.0s 的主因。T05b 在每次请求
开始前把它复位为加载时的初始值（25），不再硬编码。

| 句长（字） | T05 首包 P50 | T05b 首包 P50 / P95 | T05b 总耗时 P50 / P95 | 块数 | 首块音频 | 音频 P50 | RTF P50 |
|---|---|---|---|---|---|---|---|
| 3 | 1098 | **1485 / 1621** | 1507 / 1649 | 3 | 452 | 920 | 1.64 |
| 7 | 1917 | **2075 / 2452** | 2615 / 2900 | 6 | 452 | 1880 | 1.39 |
| 18 | 3500 | **2300 / 2931** | 4851 / 5560 | 8 | 452 | 4040 | 1.20 |
| 25 | 4513 | **2301 / 3337** | 6874 / 7188 | 10 | 452 | 5720 | 1.20 |
| 49 | 5010 | **2798 / 3188** | 10775 / 11549 | 16 | 452 | 10200 | 1.06 |
| **总体** | 3500 | **2281 / 3220**（min 1094 / max 3574） | **4851 / 11133** | — | — | — | — |

- 首包总体 P50 下降 **1219ms（35%）**，P95 下降 **2058ms**；18/25/49 字分别
  下降 1200/2212/2212ms。修复有效。
- 短句反而略慢：3 字 +387ms、7 字 +158ms。因为 hop 复位后短句也会分成
  3/6 块，而 T05 时 hop 已被上一次请求抬到 100，短句只出 1 块。
  “块多”带来的 flow/hift 开销让总耗时上升约 35–40%，RTF 从 0.95–1.33
  升到 1.06–1.64。
- 任务预计“18 字以上首包约 1.7s”没有达到，实测 2.3s：除 38 个 token 的门槛外，
  4060 Ti 上每个 chunk 的 flow/hift 仍有固定开销（块数 8–16 个）。
- **仍未达到“短句首包 P50 ≤ 0.8s”**（总体 2281ms，最好 1094ms）。

### 6.2 打断后重说（T10 风险）

49 字句收到第一块后 cancel，紧接着合成“请问您是哪位？”，5 次：

```
10830ms  10974ms  11633ms  9929ms  10917ms
P50 10917ms  P95 11501ms；正常 7 字首包 P50 2075ms
```

- 打断后首包比正常值多约 **8.8s**。原因：客户端 cancel 只是关闭连接；
  服务端为了让上游 LLM 线程收尾，会把当前 49 字句跑完（约 10.8s）才释放锁。
- 不做这个收尾会怎样：实测 cancel 后立刻放锁，下一请求的 LLM 与上一请求的
  LLM 线程并发使用同一模型，会触发 `probability tensor contains either inf, nan`
  并永久卡住服务（T05b 开发中复现 2 次）。T05b 暂时选择“跑完再放锁”，
  稳定但打断响应很慢。**T10 必须设计真正的服务端取消/请求排队，不能依赖当前实现。**

### 6.3 GPU 占用

- `nvidia-smi --query-compute-apps` 在压测时只列出本机启动的 python 服务进程，
  没有其它计算进程；T05 记录的 baseline 里多出的约 4.4GB 是当时另一个已加载的
  模型进程，T05b 时已不存在。
- 单服务 baseline 约 8227 MiB（无服务时约 1819 MiB，模型+CUDA 上下文约 6.4GB）；
  T05b 最终压测同时加载了两个服务（50000/50001），baseline 14683 MiB、
  peak 15072 MiB、增量 389 MiB。
- 压测期间 GPU 利用率 max 100%、avg 34%，说明不是持续满载；
  LLM 逐 token 生成与每块 flow/hift 之间存在串行空档，RTF 仍在 1.1–1.6。

### 6.4 音频检查（替代人工试听）

无法人耳试听，改用波形检查 + ASR 回译，每句第一次合成的 wav 在
`/tmp/t05b_wavs/`（不入库）：

| 文件 | 时长 | 峰值 | 削波 | 最大样本跳变 |
|---|---|---|---|---|
| 1_3chars.wav | 0.88s | 0.63 | 0% | 0.481 |
| 2_7chars.wav | 1.88s | 0.65 | 0% | 0.409 |
| 3_18chars.wav | 3.36s | 0.60 | 0% | 0.510 |
| 4_25chars.wav | 5.12s | 0.62 | 0% | 0.406 |
| 5_49chars.wav | 9.88s | 0.99 | 0% | 0.664 |

ASR 回译（`Qwen/Qwen3-ASR-1.7B`）与输入文本逐句对照：无丢字、无重复；
只有标点差异（如“你好” vs “您好”）和一处“记下来，转告他”的逗号插入。
样本跳变在正常语音瞬态范围内；**块边界是否有轻微咔哒声无法用波形判定，
本次未做人工听感确认**。

### 6.5 vLLM 尝试（未完成）

按要求在独立环境尝试（不破坏 `cosyvoice`）：

```sh
conda create -n cosyvoice_vllm --clone cosyvoice -y
conda activate cosyvoice_vllm
pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

结果：环境已 clone，但 `pip install` 需要连带升级 torch 2.8 和一批 CUDA 库，
下载量很大（磁盘占用从 99GB 涨到 141GB），运行数分钟后被中断，`vllm` 未安装成功，
**没有 vLLM 的实时率/首包数据**。服务端的 `--vllm` 开关已经加上，待环境装好后可直接试：

```sh
conda activate cosyvoice_vllm
python local_tts/server.py --vllm --port 50002
uv run python local_tts/bench.py --n 5 --url http://127.0.0.1:50002
```

TensorRT（flow 部分）按评审意见排在 vLLM 之后，本次未做。

## 7. 已知限制

- 单进程串行推理；多路并发会排队（T10 是一路通话，够用）。
- 客户端断开后，服务端会把当前句跑完再释放锁（T05b 为防止并发 NaN 的临时措施），
  因此“打断后重说”会等 9–11s；T10 需要真正的取消/排队设计。
- 未安装 `ttsfrd`，文本正则化依赖 wetext（本机启动时从 ModelScope 下载资源，
  失败时退化为不做正则化）；号码读法由 T06 的数字片段拼接解决。
- 参考音色固定为官方示例音色；换音色可传 `--prompt-wav` / `--prompt-text`，
  `voice_id` 会自动变化（T06 缓存随之失效）。
- vLLM 未验证；TensorRT/JIT 未尝试。

## 8. T05c 可取消（第 13 次评审）

T05b 的“打断后重说”要等整句跑完（10.9s）。T05c 改成：

1. 加载后包装 `engine.model.llm.inference`：逐个 token 转发前检查当前请求的
   `threading.Event`，置位就关闭上游生成器并 `return`。上游 `llm_job()` 会正常
   结束并置 `llm_end_dict=True`，`tts()` 随后只做一次 finalize。
2. `/synth` 改为异步流：推理仍在独立线程里跑，HTTP 层用无界队列交接；同时启动
   `request.is_disconnected()` 监视器，客户端一断开就置位停止开关，不再等下一次
   写 socket 失败。
3. `iter_pcm` 捕获 `GeneratorExit` 时先置位停止开关，再把上游剩余收尾跑完（不 yield），
   保留 T05b 的“收尾完成后再放锁”保护。

正式服务固定用 `50000`（`config.yaml: tts.local_url`）；T05c 的验证在 `50001`
隔离端口进行，避免与 T06 的请求互相干扰。

### 8.1 首包（`bench.py --n 5 --stress 20`，50001，2026-10-22 11:07）

| 句长（字） | T05b 首包 P50 | T05c 首包 P50 / P95 | T05c 总耗时 P50 / P95 | RTF P50 |
|---|---|---|---|---|
| 3 | 1485 | 1354 / 1621 | 1355 / 1623 | 1.78 |
| 7 | 2075 | 2364 / 3063 | 2916 / 3672 | 1.55 |
| 18 | 2300 | 2638 / 3198 | 4330 / 5924 | 1.22 |
| 25 | 2301 | 2725 / 3754 | 7021 / 8759 | 1.30 |
| 49 | 2798 | 2730 / 3487 | 12958 / 13670 | 1.28 |
| **总体** | 2281 / 3220 | **2615 / 3565** | **4330 / 13221** | — |

T05c 的总体首包比 T05b 略高（+334ms），主要是异步队列/断线监视的额外开销与
宿主机负载；仍远低于 T05 的 3500ms。

### 8.2 打断后重说（目标：正常 7 字首包 + 700ms）

5 次：`2460 / 3565 / 2910 / 2868 / 3367ms`，P50 **2910ms**；
正常 7 字首包 P50 2364ms，目标 3064ms → **PASS**。
每次额外等待 96 / 1201 / 547 / 504 / 1004ms。

`--stress 20`：20/20 成功、0 失败、无 NaN、无卡死；短句首包 P50 2939ms、
P95 3396ms，额外等待 0–1624ms。比 T05b 的 10.9s 明显改善；额外等待的波动
来自取消时正在计算的那一块大小（0.45–2s 音频），不是锁没释放。

### 8.3 voice_id 容错

首次 `synth()` 取 `/health` 失败只记 warning，合成继续；下次 `synth()` 会再次尝试。
显式 `await fetch_voice_id()` 失败仍然抛 `TtsError`。T06 在取不到 `voice_id` 时
应把它当作“暂时未知”，不要因此中断播放。

### 8.4 vLLM（本轮未安装）

磁盘检查：WSL 内部 115G 已用 / 124G 可用；Windows C 盘只剩 **16G**；
`ext4.vhdx` 已实际占用 158G 且非稀疏，扩容会直接吃 C 盘。
vLLM 连同其 torch/CUDA 约需 10–15GB，装下去会让 C 盘接近耗尽，因此本轮
**没有安装**，也没有 vLLM 数据。`--vllm` 开关保留，待压缩/迁移 WSL 磁盘后再试。

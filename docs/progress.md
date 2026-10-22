# 进度记录

格式：每个任务完成时追加一节，含任务号、完成情况、遇到的问题、实测数据。

## 环境基线（T00 建立）

| 项 | 值 |
|---|---|
| 系统 | WSL2 + Ubuntu 22.04.3 LTS，内核 6.18.54.1-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 4060 Ti，16380 MiB，驱动 610.88，`nvcc` CUDA 12.6 |
| Python | 3.12.15（uv 管理的 CPython，项目内 `.venv`），非 conda base 的 3.12.7 |
| 环境管理 | uv 0.13.4；`uv run <cmd>` 运行，无需 activate；创建 venv 时用 `--python-preference only-managed` 避免选中 conda 的解释器 |
| PyPI 源 | 清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`（写在本机 `~/.config/uv/uv.toml`，不入库） |
| 密钥 | `SILICONFLOW_API_KEY` 通过环境变量提供，当前 shell **尚未设置**，live 测试暂不可跑 |
| 网络 | 中国大陆；直连 PyPI 下载约 300 KB/s，走清华镜像为满速 |

---

## T00 环境与骨架

**完成情况**：完成。

**产物**

- `requirements.txt`：`httpx pyyaml pytest pytest-asyncio numpy soundfile fastapi uvicorn[standard] websockets`
- `config.yaml`：`siliconflow.base_url`、`asr.model`、`llm.realtime_model`、`llm.summary_model`、`tts.model`、`tts.voice`、`audio.sample_rate`、`interrupt.min_speech_ms`、`interrupt.ack_max_ms`，取值按任务书。
- `.env.example`：只有 `SILICONFLOW_API_KEY=` 占位。
- `pytest.ini`：注册 `live` 标记，`addopts = -m "not live"`（默认不跑联网测试），`asyncio_mode = auto`（后面大量 async 服务测试用），`testpaths = tests`。
- 包骨架：`core/ services/ tools/ web/ adapters/ adapters/pipecat/ adapters/livekit/ tests/ tests/live/`，各含空 `__init__.py`。`local_tts/`、`scenarios/` 留到 T05、T12 再建。
- `.gitignore`：在原 4 条基础上补 `cache/`、`report/`、`__pycache__/`、`*.py[cod]`、`.venv/`、`venv/`、`.pytest_cache/`、`*.egg-info/`。

**验收结果**

```
$ uv run python -c "import yaml, httpx"        → OK
$ uv run python -c "import numpy, soundfile, fastapi, uvicorn, websockets" → OK
$ uv run pytest                                 → collected 0 items, no tests ran（符合预期）
$ nvidia-smi                                    → RTX 4060 Ti, 16380 MiB, 驱动 610.88, CUDA 12.6
$ soundfile.__libsndfile_version__              → 1.2.2（wheel 自带，无需系统 libsndfile）
```

**遇到的问题与处理**

1. **默认 Python 是 conda base 的 3.12.7，与任务书的 3.11 不符。**
   处理：改用 uv 管理项目环境。**版本最终改为 3.12**（见下面第 4 条），conda base 环境不动。
   **conda 暂不卸载**，原因：① 卸掉后 `python3` 会退回系统 3.10.12，影响本机其它脚本；② T05 的 CosyVoice3 官方 README 通常要求 conda + Python 3.10，留退路；③ 省下的只有约 1GB。
   （顺带发现 `envs/langchain_rag` 已是坏 env，无 `lib/` 目录，无保留价值。）

2. **直连 PyPI 极慢**：索引页 8s 才响应，包下载约 300 KB/s，装 T00 依赖 5 分钟未完成。
   处理：本机 `~/.config/uv/uv.toml` 配清华默认源，同一批依赖 **12.7s** 装完。此文件是本机网络配置，不提交。

3. **`.gitignore` 原本缺 Python 惯例条目**，`cache/`（T06 预合成 PCM）和 `report/`（T12 报告）也是后续产物，一并补上，避免以后误提交。

**待办（不属 T00 验收，但会阻塞后续 live 测试）**

- 需要在运行 live 测试的 shell 里 `export SILICONFLOW_API_KEY=...`，否则 T02/T03/T04 的 `pytest -m live` 无法验收。

### 变更：Python 3.11 → 3.12（同日，T00 内）

**原因**：任务书写的 3.11 是文档约定，不是依赖约束；3.11 已偏旧，用户要求升级。

**核实方式**：用 `uv pip compile --python-version` 对 `pipecat-ai silero-vad onnxruntime av torch numpy soundfile fastapi uvicorn[standard] websockets httpx pyyaml` 做真实解析，结果 3.11 / 3.12 / 3.13 全部可解析（89–90 个包），关键版本一致（torch 2.14.1、onnxruntime 1.24.4）。3.12 上这些包都有预编译 wheel。

**选择 3.12 而非 3.13 的理由**：3.12 与 conda base 版本一致、wheel 齐备；3.13 虽可解析，但 pipecat/aiortc/onnxruntime 这条 C 扩展链在 3.13 上的运行期验证不足，首版不冒这个险，M2 后可再评估。

**已同步改动**：`AGENTS.md` 规则 6 改为「Python 3.12」。

**踩到的坑**：`uv venv --python 3.12` 会优先选 PATH 上的解释器，实际选中了 conda 的 3.12.7（`.venv/pyvenv.cfg` 的 `home` 指向 `~/miniconda3/bin`）。改用 `uv venv --python 3.12 --python-preference only-managed`，现在 `home` 指向 uv 管理的 `cpython-3.12-linux-x86_64-gnu`，`python -V` 为 3.12.15。

**T05 不受影响**：CosyVoice3 是独立进程（`local_tts/`），官方 README 若要求 3.10，单独给它建环境即可，与主环境版本无关。

---

## T01 类型与事件

**完成情况**：完成。

**产物**

- `core/types.py`：`AsrResult`、`TurnResult`、`SlotValue`、`State`（0.2 原样落地）。另加常量 `SLOT_NAMES`、`REQUIRED_SLOTS`、`OPTIONAL_SLOTS`、`SLOT_STATUSES`，把 0.2 里「槽位名固定为……」这条约定固化到代码里，避免 T07/T10 各自硬编码。`SlotValue.__post_init__` 校验 status 取值。
- `core/events.py`：`EventLogger(call_dir, *, clock=None)`、`EVENT_TYPES`（19 种白名单）、`monotonic_ms()`、`read_events(call_dir)`。
  - 每行 `{"call_id", "t", "type", "data"}`，`ensure_ascii=False`（中文不转义，日志可直接看）。
  - 写入即 `flush()`，进程被强杀不丢已写事件。
  - 白名单外类型抛 `ValueError`，且在写文件之前抛出，不会留下半条事件。
  - `t` 用 `time.monotonic()*1000`；即使注入的时钟回退，也取 `max(上一条 t, 当前)`，保证单调不减。
  - `clock` 参数只为测试注入假时钟用，默认走系统单调时钟。
- `tests/test_events.py`：27 个用例全通过。

**验收结果**

```
$ uv run pytest tests/test_events.py -v     → 27 passed in 0.05s
```

对应验收项：
| 验收要求 | 用例 |
|---|---|
| 每行都是合法 JSON | `test_every_line_is_valid_json` |
| `t` 单调不减 | `test_t_is_non_decreasing_with_rollback_clock`、`test_t_uses_monotonic_ms_by_default` |
| 类型不在白名单抛 `ValueError` | `test_unknown_type_raises_and_writes_nothing`、`test_all_whitelisted_types_accepted`（19 种逐项） |
| `call_id` 目录自动创建 | `test_call_dir_created_recursively` |

另补：立即 flush（不 close 也能读到）、跨实例追加（两次运行写同一目录）、中文不转义、`call_id` 取目录名。

**需要说明的两点**

1. `read_events()` 严格说超出 T01 验收范围，但 T11（摘要）、T12（评估）、T13（web 历史页）都要读事件流，且测试本身也需要读回校验，故一并实现，仅 5 行。
2. `State` 用 `str, Enum` 而非 `StrEnum`，与 0.2 一致；后续 JSON 序列化时可直接当字符串用。

**未决**：`data` 里各事件类型的字段名（如 `asr_final` 用 `text` 还是 `transcript`）任务书未规定，T10 落 `CallSession` 时统一，届时若需要固化再补进 `docs/`。

### 补：T01 测试补齐（types）

`core/types.py` 起初只用一次性 `python -c` 验证，没有测试文件。补 `tests/test_types.py`（10 个用例）：
字段名与顺序、`parse_ok` 默认值、`SlotValue` 默认 `empty` 与四种 status、
非法 status 报错、槽位常量自洽（必填∪选填=全部且不相交）、`State` 的取值与 str 行为。

覆盖率（`uv run --with pytest-cov pytest --cov=core`，cov 不写入 requirements.txt，按需临时用）：

```
core/events.py     39 stmts   0 miss   100%
core/types.py      33 stmts   0 miss   100%
TOTAL              72 stmts   0 miss   100%
37 passed
```

**工作方式调整**：此后每个任务的代码与测试同批完成、先跑绿再提交，不再用一次性脚本代替测试；
提交前一律跑 `uv run pytest`（全量，不只是本任务文件）确认没有回归。

---

## T02 LLM 客户端 ★

**完成情况**：完成。验收命令全部通过，含两轮真实压测。

**产物**

- `core/prompts.py`：系统提示（身份/权限/风格/槽位/输出格式五段）+ `build_user_message()` + `build_messages()`。
  用户消息带「当前状态（含该状态的指示）、已记录槽位、缺失必填槽位、追问次数、对方这一轮的原话」。
- `services/llm.py`：`parse_turn_output()` 纯函数、`LlmClient.turn()`（SSE 流式）、`LlmClient.complete()`（非流式摘要）、`LlmError`。
- `tools/bench_llm.py`、`tests/test_llm_parse.py`、`tests/test_llm_client.py`、`tests/test_prompts.py`、`tests/live/test_llm_live.py`。

**关键实现选择**

1. **JSON 提取**：从头逐个 `{` 用 `json.JSONDecoder().raw_decode()` 试解，成功即返回。这样代码围栏、前后多余解释、甚至「前面先出现一个非 JSON 的花括号」都能正确处理，无需正则剥围栏。
2. **失败判定**：拿不到 JSON 对象，或 `reply` 缺失/非字符串/空白 → `parse_ok=False`、`reply=""`，由调用方用兜底话术。`slots` 缺失只当本轮无更新，仍算 `parse_ok=True`（避免因为模型省略空槽位就整轮丢弃）。
3. **槽位过滤**：非 `SLOT_NAMES` 的键、非字符串值、空串一律丢弃。
4. **`callback_number` 一律不从 LLM 取**（见下面的问题 2）。
5. **重试**：429 与 5xx 指数退避（0.5s、1s）最多 2 次；4xx 不重试；**已经开始输出内容后不重试**（重发会重复播报）；超时 15s。
6. `enable_thinking: false` 对流式与非流式都显式传，实测 `reasoning_tokens=0`。

**验收结果**

```
$ uv run pytest tests/test_llm_parse.py -q      → 21 passed（现 22）
$ uv run pytest -q                              → 89 passed（默认排除 live）
$ uv run --with pytest-cov ... --cov=core --cov=services
    core/events.py  100% | core/prompts.py 100% | core/types.py 100% | services/llm.py 100%
$ source ~/.bashrc && uv run pytest -m live tests/live/test_llm_live.py -q → 4 passed in 11.70s
```

`tests/test_llm_parse.py` 覆盖任务书要求的五类：正常 JSON、前后带多余文字、带代码围栏、非法 JSON、缺字段；
另加：非白名单槽位、非字符串值、嵌套对象、空 delta、心跳行、半截 SSE 帧、`[DONE]`。

**实测数据（`uv run python tools/bench_llm.py --n 10`，两轮，公网访问 api.siliconflow.cn）**

第一轮：

| 指标 | P50 | P95 | min | max | mean |
|---|---|---|---|---|---|
| 首 token | 595ms | 4896ms | 411ms | 8035ms | 1380ms |
| 总耗时 | 1871ms | 20899ms | 1276ms | 31462ms | 5643ms |

第二轮（紧接着重跑）：

| 指标 | P50 | P95 | min | max | mean |
|---|---|---|---|---|---|
| 首 token | 638ms | 2192ms | 324ms | 2953ms | 931ms |
| 总耗时 | 3387ms | 10670ms | 1217ms | 12775ms | 4634ms |

29 次成功样本、0 失败、JSON 解析 20/20 成功。

- 首 token 的中位数与 M1 预核实一致（约 0.54–0.59s）。
- **长尾明显**：第一轮出现一次首 token 8.0s / 总 31.5s，而同一输入的另一轮只要 411ms，判断为服务端排队抖动，不是输入长度导致。第二轮没有复现，但 P95 仍达 2.2s。任务书要求端到端 P95 ≤ 3.5s，LLM 这一段就要占掉相当一部分。
- **总耗时方差比首 token 大得多**（P50 1.9s vs 3.4s）：模型有时把 reply 写得偏长。真实链路里 TTS 只需第一句，因此 **T10 起必须按短句切分、首句一到位就开始合成**，不能等 `llm_done`，否则总耗时直接进关键路径。

**遇到的问题与处理**

1. **提示词约束不住越权请求**（live 测试抓到）。
   对方说「我跟他说好了今天下午三点见，你帮我确认一下」，`Qwen/Qwen3-8B` 回的是
   「请问您是李先生的朋友吗？您需要我帮您确认和李先生的见面安排吗？」——把越权请求当成可选项抛回去，
   违反设计文档第 4 节「统一回应会转告，请留下联系方式」。
   处理：提示词明确禁止反问「需要我帮您确认吗」与任何代办暗示，并加一条正例（对方说……你就回……）。改后通过。
   **说明 8B 模型的指令跟随偏弱**，后续 T07/T12 需要靠状态机模板话术兜底，而不是指望 LLM 每轮都守规矩。

2. **模型把提示语当成槽位值回传**：`slots.callback_number = "请说一遍您的回拨号码"`。
   因为 `callback_number` 在白名单里，原实现会把它当有效槽位收下。
   处理：`parse_turn_output` 直接丢弃 `callback_number`，理由见设计文档第 4 节「号码与地址以 ASR 最终文本和规则校验为准，不依赖 LLM 听写」。已加用例。

3. **测试自身写错两次**（不是实现问题，记录以示区分）：
   ① 用非 JSON 文本断言 reply 解析成功；② 用字符串切分拼 SSE 样本导致 chunk 为空。
   改为复用 `sse()` 辅助函数后修正。

**未决**

- `total_ms` 里包含了完整 JSON 的输出时间，而链路真正需要的是「第一个完整短句」的时间。T10 要在流式回调里按标点切句并记 `llm_first_sentence_ms`，届时可能需要新增一类事件或复用 `llm_done` 的 data 字段。

---

## 网络与推送（环境备忘，非任务）

`github.com:443` 在本网络被阻断（`git push` 卡 133s 后超时），但 `ssh.github.com:443` 可达。处理：

- 生成 `~/.ssh/id_ed25519`，公钥加到 GitHub 账号；
- `~/.ssh/config` 把 `Host github.com` 指向 `ssh.github.com:443`（本机配置，不在仓库里）；
- `origin` 改为 `git@github.com:finalsatan/agent-telephone.git`。

因此 **所有 git 操作走 SSH over 443**，不要用 https 远端。

另外记录：用户启动代理后，WSL 内可见代理端口 `127.0.0.1:33210`（7890/7897/1080/10809/2080 均不通）。
若日后 SSH 也不通，可退回 HTTPS + 该代理端口（`git config http.proxy http://127.0.0.1:33210`）。

---

## T02b LLM 客户端修订（第 1 次评审）★

**完成情况**：完成。评审列出的 6 条要求全部落地，验收命令全部通过。

**产物**

- `core/types.py`：新增 `ReplyChunk`（`kind/text/result/t_ms`）；`TurnResult` 增加 `first_sentence_ms: float | None = None`。
- `services/llm.py`：
  - `LlmClient.turn_stream()`：流式产出 `ReplyChunk(kind="sentence")`，流结束产出 `done`。
  - `LlmClient.turn()` 改为消费 `turn_stream()`，签名增加 `attempts`。
  - `_ReplyExtractor`：增量识别 reply 字符串，支持 `\" \\ / \b \f \n \r \t \uXXXX`，转义序列可跨 SSE 块。
  - `split_reply_sentences()`：完整 reply 的切句规则，兜底与增量共用同一组终止符（`。！？；` 与换行）。
  - `LlmTimeout(LlmError)`；两层截止时间用 `asyncio.timeout` 实现，首 token 2.5s / 整轮 8s，均可由 config 覆盖；触发时不重试。
  - 请求体带 `max_tokens`（默认 150）。
- `core/prompts.py`：`SYSTEM_PROMPT` 常量改为 `build_system_prompt(owner_name)`；`build_messages()` 增加 `attempts`、`owner_name` 参数；输出格式要求 reply 为第一个键。
- `config.yaml`：新增 `owner.name: 张先生`，`llm.first_token_timeout_s/total_timeout_s/max_tokens`。
- `tests/test_llm_client.py`：新增 15 个 T02b 用例（见下）。
- `tools/bench_llm.py`：打印首句耗时，统计 `LlmTimeout` 次数。
- `tests/live/test_llm_live.py`：增加 `turn_stream()` 的 live 用例；功能验收用 10s/20s 宽松时限，避免公网抖动导致不稳定（默认 2.5s/8s 的触发率由 bench 统计）。

**接口变更记录（AGENTS.md 规则 2）**

- `TurnResult` 增加 `first_sentence_ms`，默认 `None`，旧调用方不受影响。
- `ReplyChunk` 为新增类型。
- `build_messages()` 的 `attempts`、`owner_name` 为 keyword-only 且有默认值；`turn()` 的 `attempts` 有默认值。契约变化均来自评审文档 T02b，已同步更新全部调用方与测试。

**关键实现选择**

1. **reply 必须是 JSON 的第一个键**。增量识别器只在对象开头接受 `"reply"`；若遇到 `"slots"` 在前则立即放弃（`abandoned=True`），不报错，流结束后用完整解析结果切句兜底。这样既能边收边播，又不会把 slots 值里出现的 `"reply"` 误认成键。
2. **兜底只在“一句都没产出”时触发**，避免与已播出的句子重复。
3. **兜底时 `first_sentence_ms = total_ms`**：句子在流结束时才可用，保证 `first_sentence_ms <= total_ms` 的单调关系。
4. **总时长超时保留已产出的句子**，只是不再产出 `done`，由调用方播兜底话术（T10 使用）。
5. `turn_stream` 在 `finally` 里 `aclose()` 底层生成器，保证超时/提前退出时连接被释放。

**验收结果**

```
$ uv run pytest -q
    109 passed, 5 deselected
$ uv run --with pytest-cov pytest -q --cov=core --cov=services
    core/events.py 100% | core/prompts.py 100% | core/types.py 100% | services/llm.py 95%
$ source ~/.bashrc && uv run pytest -m live tests/live/test_llm_live.py -q
    5 passed in 16.07s
$ uv run python tools/bench_llm.py --n 20   （见下表）
```

T02b 用例覆盖（任务书要求逐项）：

| 验收要求 | 用例 |
|---|---|
| reply 在前时首句在流结束前产出 | `test_first_sentence_emitted_before_stream_finishes` |
| 转义字符正确（含跨块） | `test_reply_escapes_decoded_even_when_split_across_chunks`、`test_extractor_survives_char_by_char_chunks` |
| reply 无标点时作为一句在结束时产出 | `test_reply_without_punctuation_flushes_as_one_sentence` |
| slots 在前时退回到结束后切句 | `test_slots_before_reply_falls_back_to_split_after_stream` |
| 首 token 超时抛 `LlmTimeout` | `test_first_token_timeout_raises_and_does_not_retry` |
| 总时长超时在已产出句子之后抛 `LlmTimeout` | `test_total_timeout_after_sentence_keeps_sentence_but_no_done` |
| `max_tokens` 与 `attempts` 出现在请求/提示词 | `test_max_tokens_attempts_and_owner_reach_the_request` |
| 系统提示含 config 中的机主称呼 | `test_from_config_reads_t02b_settings`、`test_system_prompt_uses_owner_name` |

另补：逐字符分块（真实模型按 token 返回）、reply 词出现在 slots 值里不被误判、畸形 reply 头、连续标点、空白。

**实测数据（`uv run python tools/bench_llm.py --n 20`，公网 `api.siliconflow.cn`）**

| 指标 | P50 | P95 | min | max | mean |
|---|---|---|---|---|---|
| 首 token | 490ms | 757ms | 372ms | 914ms | 544ms |
| 首句 | 1033ms | 3509ms | 642ms | 3519ms | 1559ms |
| 总耗时 | 1867ms | 7115ms | 915ms | 7724ms | 2975ms |

成功 19/20，`LlmTimeout` 1 次（总时长 8s 触发，已产出 1 句），JSON 解析 19/19 成功。

**遇到的问题与处理**

1. **增量识别器在真实流上从未生效（最重要）**。
   第一次跑 bench 时 4/20 总时长超时且“emitted 0 sentence(s)”，成功样本的 `first_sentence_ms` 又全部等于 `total_ms`——说明每轮都走了“流结束后兜底切句”。
   定位：真实模型逐 token 返回，`":` 与值的引号 `"` 常被拆到不同 SSE 块；原状态机把「跳空格 → 期待冒号 → 跳空格 → 期待引号」放在同一个 `seek_colon` 状态里，在跳空格处 `break` 后，下一块会重新从期待冒号开始，看到引号便误判为畸形 JSON 而放弃。
   处理：拆成 `seek_colon` 与 `seek_value` 两个状态（见 `services/llm.py` 注释）；补逐字符分块的回归用例。
   修复后首句 P50 从「等于总耗时」降到 1.0s，总耗时超时从 4 次降到 1 次。
2. **总时长 8s 仍会切掉长尾**。本轮 20 次里有 1 次模型流式输出超过 8s（已播出 1 句）。这是预期内的设计取舍：实时链路不能等完整 JSON，超时后由 T10 播兜底话术。`max_tokens=150` 后总耗时的 P95 仍有 7.1s，T10 必须依赖首句而非 `done`。
3. **首句 P95 仍为 3.5s**，高于首 token P95（0.76s）。原因是部分回复的第一句本身较长、逐 token 流式输出耗时；T10 的端到端预算里要给“首句完整”留出时间，必要时把系统提示进一步限制为“第一句 ≤ 15 字”。
4. **提示词仍偶尔出现越权反问**：本轮案例 9 出现“请问需要我转告张先生吗？”。与 T02 记录的 8B 指令跟随偏弱一致，T07 的状态机模板话术必须能覆盖这条路径。

**未决**

- 首句过长导致的额外延迟，是否在 T10 用「首句长度上限 + 标点切分」进一步压，待 T10 实测端到端后再定。

---

## T02c LLM 截止时间与首句修订（第 2 次评审）★

**完成情况**：完成。评审列出的 4 条要求全部落地，验收命令全部通过。

**产物**

- `services/llm.py`：
  - `turn_stream()` 不再用 `asyncio.timeout` 包住循环，改为请求开始时算两个绝对截止时刻，只用 `asyncio.timeout_at()` 包住每一次 `await anext(stream)`；**`yield` 全部在超时范围之外**。
  - 首 token 截止从请求开始计时，空白块不会重置时限。
  - 超时时关闭底层流并抛 `LlmTimeout`，消息区分 `first-token timeout` / `total timeout`。
  - 新增 `_SentenceSplitter`：增量流与 `split_reply_sentences()` 共用同一状态机；首句在累计非空白字符 ≥ `first_sentence_min_chars` 后允许在 `，`、`、` 处提前切出，第一句之后不再按逗号切。
- `config.yaml`：新增 `llm.first_sentence_min_chars: 8`。
- `tests/test_llm_client.py`：新增 5 个 T02c 回归用例。
- `tools/bench_llm.py`：统计首句平均字数。
- `docs/progress.md`：本节。

**接口变更记录（AGENTS.md 规则 2）**

- `LlmClient.__init__` 新增 `first_sentence_min_chars`（有默认值），`from_config` 从 `llm.first_sentence_min_chars` 读取。
- `split_reply_sentences()` 新增 keyword-only 参数 `first_sentence_min_chars`（有默认值），旧调用不受影响。
- 对外契约（`turn` / `turn_stream` / `ReplyChunk` / `TurnResult`）不变。

**关键实现选择**

1. **截止时间的语义**：`total_timeout_s` 是从请求开始的墙钟上限。消费者的耗时不会被计入“等待 LLM”，但会让下一次取句更早超时——若消费者已耗时超过总时限，下一次 `await anext()` 会立即抛 `LlmTimeout`。这符合评审给出的定义，也保证 T10 调 TTS 时不会被本模块取消。
2. **超时的异常归属**：`asyncio.timeout_at()` 只包住 `anext`，`TimeoutError` 在生成器内部转成 `LlmTimeout`；`yield` 不在任何超时上下文内，因此消费者的 `await` 不会收到 `CancelledError`，也不会产生未被接收的异常。
3. **首句提前切分只放宽一次**：`_SentenceSplitter` 的 `_first_done` 一旦置位，逗号不再作为句末。阈值按“累计非空白字符数”计算，逗号本身计入累计。
4. **两条路径一致性由同一个类保证**：`split_reply_sentences()` 只是把完整文本逐字符喂给 `_SentenceSplitter`，不再维护第二套切句逻辑。

**验收结果**

```
$ uv run pytest -q
    114 passed, 5 deselected
$ uv run --with pytest-cov pytest -q --cov=core --cov=services
    core/events.py 100% | core/prompts.py 100% | core/types.py 100% | services/llm.py 98%
$ source ~/.bashrc && uv run pytest -m live tests/live/test_llm_live.py -q
    5 passed in 6.16s
$ uv run python tools/bench_llm.py --n 20   （见下表）
```

T02c 用例与评审要求对应：

| 验收要求 | 用例 |
|---|---|
| 消费者 sleep 不被取消；随后取句抛 `LlmTimeout`；无未接收异常 | `test_consumer_await_is_not_cancelled_by_total_timeout`（用 `sys.unraisablehook` 检查） |
| 空白块不重置首 token 截止 | `test_whitespace_deltas_do_not_reset_first_token_deadline`（300 块 / 3s，断言 < 1s 内超时） |
| 首句在逗号处提前切出 | `test_first_sentence_splits_at_comma_after_min_chars` |
| 第一句之后的逗号不再切 | 同上第二条断言 |
| 兜底路径与增量路径切分一致 | `test_incremental_and_fallback_split_agree_for_comma_rule` |
| 阈值可配置 | `test_first_sentence_min_chars_is_configurable`、`test_from_config_reads_t02b_settings` |

**实测数据（`uv run python tools/bench_llm.py --n 20`，公网 `api.siliconflow.cn`）**

| 指标 | T02b | T02c | 变化 |
|---|---|---|---|
| 首 token P50 / P95 | 490ms / 757ms | 414ms / 604ms | 基本持平 |
| 首句 P50 / P95 | 1033ms / 3509ms | **785ms / 1278ms** | P95 降 64% |
| 总耗时 P50 / P95 | 1867ms / 7115ms | 1396ms / 1934ms | P95 降 73% |
| 超时次数 | 1/20 | **0/20** | — |
| 首句平均字数 | 未统计 | **12.3**（min 7 / max 22） | — |

说明：两轮 bench 之间服务端状态不同，总耗时的下降不能全部归因于本次改动；但**首句 P95 从 3509ms 降到 1278ms** 与“首句在逗号处提前切分”的预期一致。首句 min 7 字是句号自然结束的短句（如“请问您是哪位？”），不受 8 字阈值限制。

**遇到的问题与处理**

1. **逗号提前切分会改变切句结果，需要同步更新旧测试的隐含假设。** T02b 的用例里没有“第一句含逗号”的样本，因此未出现回归；新增了“兜底路径与增量路径一致”的用例，防止以后只改一侧。
2. **首 token 与总时限的优先级**：当两个截止时刻同时到达时，以总时限为准（先判 `loop.time() >= total_deadline`），避免把总时限误报成首 token 超时。
3. **超时终止时的清理**：`finally` 里 `aclose()` 底层流；关闭异常被吞掉，避免覆盖原始 `LlmTimeout`。

**未决**

- 本轮 20 次无超时，但 T02b 曾出现 1 次整轮 >8s。T10 仍必须按“首句一到位就送 TTS”设计，不能依赖 `done`。
- 8B 模型仍偶有越权反问（T02b 已记录），继续由 T07 模板话术兜底。

---

## T03 ASR 客户端与号码规范化 ★

**完成情况**：完成。验收命令全部通过，含两轮真实压测。

**产物**

- `services/asr.py`：`AsrClient.transcribe(pcm16k) -> AsrResult`。
  - 把 PCM 封成 WAV（`wave` 模块，16kHz/单声道/16-bit）后 multipart 上传到
    `POST {base_url}/audio/transcriptions`，字段 `file`（`audio.wav`）+ `model`（取 `asr.model`）。
  - 解析响应 `text`（`strip()`），`latency_ms` = 请求发出到解析完成，`audio_ms` = `len(pcm)/32`。
  - 超时 20s；重试策略与 T02 一致：429 与 5xx 指数退避最多 2 次，4xx 不重试，传输错误重试。
  - 空 PCM 直接返回空结果、不发请求（省配额，也不需要重试）。
  - `AsrError`（重试用尽／响应不可用／缺密钥）；`AsrClient.from_config` 读
    `siliconflow.base_url`、`asr.model`、`audio.sample_rate`。
- `core/normalize.py`：`cn_digits_to_arabic()`、`extract_phone()`、`normalize_time_expr()`（占位返回原文）。
- `tools/bench_asr.py`：`--file xxx.wav --n 10`，打印每次耗时与实时率，末行给 P50/P95/min/max/mean，
  并打印文本样例与 `extract_phone` 结果（顺带确认规则后处理可用）。
- `scenarios/asr_01.wav`（8.76s，整段来电）、`scenarios/asr_02.wav`（2.52s，单轮短句）：
  用硅基流动 TTS 合成的 16kHz 单声道 16-bit wav，供 live 测试与 bench 共用（T12 会换成正式场景集）。
- `tests/test_normalize.py`（42 例）、`tests/test_asr_client.py`（`httpx.MockTransport`，11 例）、
  `tests/live/test_asr_live.py`（3 例）。

**接口与范围说明（AGENTS.md 规则 2、8）**

- `services/asr.py` 只多了一个辅助函数 `load_wav_pcm(path)`（读任意 wav → 16kHz 单声道 16-bit PCM），
  被 `tools/bench_asr.py` 与 live 测试共用，避免两处各写一份解码/重采样。0.3 的 `transcribe` 契约未变。
- `scenarios/` 目录任务书列在目录结构里（T03 文件清单没写），但 T03 的 live 验收要求“用 `scenarios/`
  里的一条样例 wav”，因此本次先放两条样例音频；正式场景集仍由 T12 生成。
- `core/normalize.py` 不在 0.3 的共享类型里，是 T03 的交货物，T07 的 `SlotStore.update_phone_from_text()`
  会直接复用它。

**关键实现选择**

1. **只转“连续数字串”，不动数量级和小数**：`一三八零零一三八零零零`（11 字连续）→ 11 位数字；
   `一百二十三`、`三四百万`、`三点一四`、单个 `一`/`两` 都保持原样。判据是
   「连续数字字符长度 ≥ 2」且「前后不紧邻 `十百千万亿` / `点.`」。这样避免为了转号码把普通语气
   （“我一个人”“两点再说”）改坏，也就避免了 `extract_phone` 的假阳性。
2. **`extract_phone` 的顺序**：NFKC（全角数字、全角空格）→ 汉字数字转阿拉伯 → 合并数字间分隔符
   （空格、`-`、`、`、括号等）→ 先匹配 11 位手机号（`1[3-9]` + 9 位），再匹配 7–8 位座机号。
   前后都加 `(?<!\d)`/`(?!\d)`，所以 10 位或 12 位数字串既不会被当成手机号，也不会被截成座机号。
3. **不对空音频发请求**：链路里 VAD 可能给来极短片段（或调用方传空 buffer），这种情况不需要往返一次；
   返回 `latency_ms=0` 不会污染时延统计。
4. **`_HttpStatusError`/`_is_retryable` 与 `services/llm.py` 同形但不跨模块复用**：两个服务各自独立，
   以后调重试策略（比如 ASR 想放宽次数）不会牵动 LLM。

**验收结果**

```
$ uv run pytest tests/test_normalize.py tests/test_asr_client.py -q
    53 passed
$ uv run pytest -q
    167 passed, 8 deselected
$ uv run --with pytest-cov pytest -q --cov=core --cov=services
    core/normalize.py 100% | services/asr.py 100% | 其余同 T02c（llm 98%）
$ source ~/.bashrc && uv run pytest -m live tests/live/test_asr_live.py -q
    3 passed
$ source ~/.bashrc && uv run pytest -m live -q
    8 passed（T02 的 5 个 LLM live 用例 + T03 的 3 个）
```

任务书要求的用例逐项对应：

| 验收要求 | 用例 |
|---|---|
| `"我的电话是一三八零零一三八零零零"→13800138000` | `test_cn_digits_mobile`、`test_converts_digit_runs` |
| `"幺三八零零一三八零零零"→13800138000` | `test_yao_digits_mobile` |
| `"13800138000"` 原样 | `test_arabic_mobile_passthrough` |
| `"二零二六年"` 不被当成手机号 | `test_year_is_not_a_phone` |
| 不足 11 位返回 `None` | `test_fractional_length_mobile_is_not_a_phone` |
| live：样例 wav 得到非空文本 | `test_transcribe_returns_non_empty_text` |

另补：全角数字、`138-0013-8000`/`138 0013 8000`/`138、0013、8000` 分隔符、7–8 位座机号、
10/12 位数字串不误判、`load_wav_pcm` 的单声道/立体声降混/8k→16k 重采样、
multipart 里确实带 WAV 容器与原始 PCM、429/5xx 重试与 4xx 不重试、非 JSON/非对象/缺 `text` 的响应、
空音频不发请求、`from_config` 与缺密钥报错。

**实测数据（`uv run python tools/bench_asr.py`，公网 `api.siliconflow.cn`，`Qwen/Qwen3-ASR-1.7B`）**

`scenarios/asr_01.wav`（8.76s 整段来电，含号码）

| 轮次 | P50 | P95 | min | max | mean | 实时率 P50 |
|---|---|---|---|---|---|---|
| 第 1 次 | 533ms | 757ms | 420ms | 758ms | 567ms | 0.06 |
| 第 2 次 | 382ms | 661ms | 307ms | 705ms | 435ms | 0.04 |

`scenarios/asr_02.wav`（2.52s 单轮短句，更接近真实的一轮输入）

| 轮次 | P50 | P95 | min | max | mean | 实时率 P50 |
|---|---|---|---|---|---|---|
| 第 1 次 | 179ms | 492ms | 156ms | 622ms | 239ms | 0.07 |
| 第 2 次 | 172ms | 247ms | 152ms | 300ms | 182ms | 0.07 |

10/10 成功，两轮共 40 次调用 0 失败；转写文本每次完全一致
（`喂，你好，我找一下张先生，我是送快递的，有个包裹要送过来，我的电话是一三八零零一三八零零零。`），
`extract_phone` 每次都能还原 `13800138000`。

**结论（对时延预算的影响）**

- **ASR 比设计文档 16.1 的预估好得多**：短句 P50 约 0.17s（预估 0.6s），8.8s 长句 P50 0.38–0.53s。
  也就是说 T03 这一段在当前网络下只占端到端预算的 5%–15%，瓶颈仍在 TTS 首包（预估 1.2–3.7s）。
- 长音频的耗时约为短音频的 2–3 倍，但实时率仍 < 0.1：**整段上传没有“按音频时长线性排队”的问题**。
- 首个请求有冷启动（第 1 次 622–758ms 的长尾出现在首轮），T10 起链路第一条 ASR 请求前可以先用
  一段静音预热一次连接（是否值得等 T10 实测再定）。
- 号码识别稳定：10 次转写全是汉字数字且 11 位正确，与 16.1 的结论一致；规则转换（T03）已能 100% 还原，
  因此“号码不过 LLM”这条设计成立。

**遇到的问题与处理**

1. **任务书要求的“`scenarios/` 里的样例 wav”还不存在**（`scenarios/` 是 T12 的产物，当前仓库没有音频）。
   处理：用硅基流动 TTS 直接合成两条 16kHz 单声道 16-bit 的样例（`asr_01.wav` 整段来电、
   `asr_02.wav` 单轮短句），提交进仓库供 live 测试与 bench 复用；T12 生成正式场景集时替换。
   合成脚本没有进仓库（只是临时 `/tmp` 脚本，避免提前做 T04 的 `services/tts.py`）。
2. **`bench_asr.py` 第一版把实时率按整数打印**，0.04–0.09 全显示成 `0`，看起来像是没测到数据。
   处理：耗时用 `:.0f` 毫秒、实时率用 `:.2f` 分开格式化。
3. **8.8s 样例的首轮 P95 偏高**（757ms，min 420ms）：首轮含连接建立与 TLS 握手，后续轮次降到 661ms。
   与设计文档 15.2「连接加 TLS 约 0.16s」的观察一致，不算异常。

**未决**

- **带区号的座机**（`010-12345678`）当前不识别：合并分隔符后有 11 位且首位是 0，既不匹配手机号，
  也不匹配 7–8 位座机号。真实来电里对方报座机的比例不高，且 T07 会用复述确认兜底；若 T12 的场景集
  出现这类号码再补规则（需要一套区号白名单，不能靠长度猜）。
- `normalize_time_expr()` 目前是占位返回原文，时间表达仍会原样进 `time` 槽位；T07 的状态机与
  T11 的摘要需要时间结构化时再实现。
- 首轮冷启动的预热（发一段静音）是否值得做，待 T10 端到端实测后决定。

---

## T03b ASR 修订（第 4 次评审）★

**完成情况**：完成。评审列出的 4 条要求全部落地，验收命令全部通过。

**产物**

- `core/normalize.py`
  - `extract_phone(text, *, allow_landline: bool = False)`：手机号规则不变、始终可用；座机号（7–8 位，或 `0`＋2–3 位区号＋7–8 位）只在 `allow_landline=True` 或号码前 6 个字以内出现关键词时才接受。关键词表与窗口宽度写成模块常量 `LANDLINE_CONTEXT_KEYWORDS`、`LANDLINE_CONTEXT_WINDOW`。
  - `cn_digits_to_arabic()` 的 docstring 明确写"仅供号码提取使用，不得用来改写显示或摘要用的转写文本"，并给出"两三个人"→"23个人"的例子。
- `services/asr.py`
  - `attempt_timeout_s`（默认 3s）与 `total_timeout_s`（默认 5s）；每次尝试用 `asyncio.timeout_at(min(本次尝试截止, 总截止))` 包住；超过总截止抛 `AsrTimeout`，不再重试。
  - `latency_ms` 从第一次请求发出开始计，包含重试与退避。
  - `AsrTimeout(AsrError)`；单次尝试超时视为可重试，总截止到点则直接抛。
- `config.yaml`：`asr.attempt_timeout_s: 3`、`asr.total_timeout_s: 5`、`asr.recheck_attempt_timeout_s: 20`、`asr.recheck_total_timeout_s: 60`（后两个供 T11 构造客户端时传入）。
- `tests/test_normalize.py`：新增 `TestLandlineContext`（11 例，含评审给的全部正反例）。
- `tests/test_asr_client.py`：新增 `TestTimeouts`（4 例）、`TestTimeoutSemantics`（2 例）、`TestFromConfig` 增加 3 例（T03b 时限从 config 读取、缺省值、仓库 config 断言）。

**接口变更记录（AGENTS.md 规则 2）**

- `extract_phone()` 增加 keyword-only 参数 `allow_landline`（默认 `False`，即默认行为变严格）；T07 的 `update_phone_from_text(text, allow_landline=...)` 按评审要求透传。已有调用方（`tools/bench_asr.py`、live 测试）行为不变，因为它们的样例都是手机号。
- `AsrClient.__init__` 增加 `attempt_timeout_s`、`total_timeout_s`；`from_config` 从 `asr.*` 读取。
- 新增 `AsrTimeout(AsrError)`，仍然是 `AsrError` 的子类，T10 现有 `except AsrError` 不受影响。
- **`timeout` 参数默认值由 `20.0` 改为 `None`**：为 `None` 时取 `max(attempt_timeout_s, total_timeout_s)`。原因是 T11 的离线校正用 60s 总截止，若 httpx 仍是 20s 读超时，会先于我们自己的截止时间触发并把结果变成"传输错误重试"，语义混乱。显式传 `timeout` 仍然生效。

**关键实现选择**

1. **两种候选文本**：先匹配"合并了数字间分隔符"的文本（`138 0013 8000` 这类分组读法），再退回原样文本。原因：合并分隔符会把相邻的两个号码连成一个 19 位串而丢掉手机号（`"13800138000 87654321"`），退回原文即可取到手机号。手机号的两种候选都先试完，再生效座机规则，保证"手机号优先"不被 `allow_landline` 影响。
2. **座机上下文窗口 6 个字**，按号码起点往前数（含标点）。窗口是启发式：`"我的电话是82345678"` 认，"电话我已经说过了，那就是八二三四五六七八" 不认——后者宁可漏，也不要让整段话里任何一个 8 位数被"电话"带进槽位；T07 的复述确认会兜住漏掉的情况。
3. **关键词只放 8 个**（电话、号码、座机、手机、联系、回拨、打、拨），单字"打""拨"是有意放宽的（"请打八二三四五六七八找我"），也正因为单字容易误中，才配了 6 字窗口。
4. **单次尝试超时算可重试**。第一版把 `AsrTimeout` 当成不可重试，结果"第 1 次尝试超时后永不重试"，是测试 `test_attempt_timeout_is_retried_within_total` 抓出来的（评审要求的"单次尝试超时后在总时限内重试成功"正是这条）。
5. **退避最多等到总截止**：`await asyncio.sleep(min(delay, max(0, 总截止 - now)))`，下一轮循环顶部统一报 `AsrTimeout`。这样"退避会越过总截止"的路径不会多打一次请求，也不用写一个几乎无法覆盖的防御分支。

**验收结果**

```
$ uv run pytest -q
    193 passed, 8 deselected
$ uv run --with pytest-cov pytest -q --cov=core --cov=services
    core/normalize.py 100% | services/asr.py 100%（其余同 T03：llm 98%）
$ source ~/.bashrc && uv run pytest -m live -q
    8 passed
$ uv run pytest tests/test_asr_client.py::TestTimeouts -q   # 连跑 3 次
    4 passed（每次 0.42s，无抖动）
```

评审给的正反例逐项对应：

| 场景 | 用例 | 结果 |
|---|---|---|
| `"预约在2026-10-21上午"` | `test_no_landline_without_context` | `None` |
| `"订单号是12345678"` | 同上 | `None` |
| `"明天下午三点，房间号八零二一二三四"` | 同上 | `None` |
| `"一共一二三四五六七块"` | 同上 | `None` |
| `"我家座机是八二三四五六七八"` | `test_keyword_context_accepts_landline` | `82345678` |
| `extract_phone("八二三四五六七八", allow_landline=True)` | `test_allow_landline_accepts_bare_digits` | `82345678` |
| `"座机零一零一二三四五六七八"` | `test_area_code_landline` | `01012345678` |
| 原有手机号用例 | 全部保留 | 通过 |
| 单次尝试超时后在总时限内重试成功 | `test_attempt_timeout_is_retried_within_total` | 通过 |
| 总时限到达抛 `AsrTimeout` 且不再重试 | `test_total_timeout_raises_asr_timeout_and_stops_retrying`、`test_backoff_past_total_deadline_does_not_retry` | 通过（断言最后一次尝试的发起时刻 ≤ 总截止） |
| `latency_ms` 含退避 | `test_latency_ms_includes_retries_and_backoff` | 通过（退避 50ms，断言 ≥ 45ms） |

**实测数据（改完后重跑，确认 3s/5s 的时限不影响正常链路）**

`uv run python tools/bench_asr.py`，公网 `api.siliconflow.cn`，`Qwen/Qwen3-ASR-1.7B`：

| 样例 | P50 | P95 | min | max | 实时率 P50 | 失败 |
|---|---|---|---|---|---|---|
| `asr_01.wav`（8.76s） | 346ms | 491ms | 304ms | 518ms | 0.04 | 0/10 |
| `asr_02.wav`（2.52s） | 203ms | 314ms | 180ms | 386ms | 0.09 | 0/10 |

20 次调用 0 失败、0 超时；与 T03 的数据（0.17–0.53s）在同一水平，说明 3s 单次尝试、5s 总截止对当前网络有三倍以上余量。
`extract_phone` 仍然还原 `13800138000`（`no context` 收紧只影响座机，手机号不受影响）。

**遇到的问题与处理**

1. **合并分隔符会把相邻两个号码粘成一个 19 位串**，手机号因此匹配失败（`"13800138000 87654321"`）。这暴露了 T03 的一个隐患。
   处理：改为"合并版 + 原文"两种候选文本依次匹配，先手机号后座机。
2. **`AsrTimeout` 最初被算作不可重试**，导致单次尝试超时后立刻放弃（与评审要求相反）。由新增测试抓出，`_is_retryable()` 现在把 `AsrTimeout` 视为可重试（总截止时是直接 raise，不经过它）。
3. **httpx 的 20s 读超时会先于 T11 的 60s 总截止触发**，把超时伪装成"传输错误重试"。处理：`timeout` 默认改为 `None`，取 `max(attempt, total)`。
4. 改写时留下一个"`remaining <= 0` 当场抛超时"的防御分支，实测任何情况都会被 `asyncio.timeout_at` 先打断在总截止处，属于不可达代码且拉低覆盖率。改为把退避截断到总截止、由循环顶部统一判定。

**未决**

- 座机上下文仍是启发式：`"电话我已经说过了，那就是八二三四五六七八"`（关键词超窗）会漏；`"打12345678"` 这类前缀会误收。真实场景由 T07 的复述确认兜底，T12 用场景集统计漏收/误收率后再调窗口与关键词表。
- 不带区号的座机在 `allow_landline=False` 且无关键词时一律不收，即使对方说的是 8 位座机并已在其他话里提到"电话"。这是有意取舍，记录备查。

---

## T04 云端 TTS 客户端 ★

**完成情况**：完成。接口、取消、两层截止时间、重试边界、空文本与压测均已实现，验收命令全部通过。

**产物**

- `services/tts.py`：`TtsClient`（实现 0.3 的 `TTS` 接口，鸭子类型）与 `TtsError` / `TtsTimeout`。
  - `synth(text, cancel)` 异步迭代出 16kHz、单声道、16-bit 小端 PCM 块。
  - POST `{base_url}/audio/speech`，JSON：`model`、`input`、`voice`、`response_format:"pcm"`、`sample_rate`、`stream:true`；`Authorization: Bearer $SILICONFLOW_API_KEY`。
  - 空文本/纯空白不发请求；返回空音频时抛 `TtsError("tts returned no audio")`。
- `tools/bench_tts.py`：5 句由短到长，每句测 `--n` 次；首包由调用方包装器在收到第一块时记录，输出每句与总体的 P50/P95、音频时长与实时率。
- `tests/test_tts_client.py`：16 个离线用例（MockTransport）。
- `tests/live/test_tts_live.py`：2 个 live 用例。
- `config.yaml`：`tts.first_chunk_timeout_s: 10`、`tts.total_timeout_s: 30`（取值理由见下）。

**接口/配置变更记录（AGENTS.md 规则 2）**

- 新增 config 键 `tts.first_chunk_timeout_s`、`tts.total_timeout_s`。`core/types.py` 与既有服务接口未改。
- `TtsClient` 是新增类，不改变 0.3 约定的 `synth(text, cancel)` 与 `sample_rate`。

**关键实现选择**

1. **超时范围不跨越 `yield`（同 T02c）**：请求开始时算 `first_deadline = now + first_chunk_timeout_s`、`total_deadline = now + total_timeout_s`；用 `asyncio.timeout_at()` 只包住 `client.send()`、每一次 `anext(response.aiter_bytes())` 和错误响应体读取。产出音频的 `yield` 在超时范围之外，消费者自己的 await 不会被本模块取消；`total_timeout_s` 是墙钟语义，超过后下一次取块立即超时。
2. **取消可打断网络等待**：每次网络等待都是“任务 + `cancel.wait()` 二选一”（`asyncio.wait(FIRST_COMPLETED)`），因此 cancel 置位时不必等下一块数据到达；退避重试的 `sleep` 也可被 cancel 立即打断。取消是安静结束，不抛异常。
3. **重试边界**：只有“尚未产出任何音频”时的 429/5xx/`httpx.TransportError` 才按 0.5s、1s 退避重试（最多 2 次）；已经开始 output 的请求不重试，避免调用方重复播报。截止时间触发不重试。
4. **首包从请求开始计时**：建连与等响应头使用同一个 `first_deadline`（实测服务端会把响应头压到首块数据一起返回，二者相差约 2ms）。
5. **实测确认 API 形态**：`stream:true` 返回 `content-type: application/octet-stream` 的裸 PCM（非 SSE/base64），块大小约 1–2KB；全部音频通常在一次突发里到达，之后按块拆开。

**验收结果**

```
$ uv run pytest -q
    130 passed, 7 deselected
$ uv run pytest -m live tests/live/test_tts_live.py -q
    2 passed in 7.66s
$ uv run python tools/bench_tts.py --n 10   （见下表）
```

| 用例 | 覆盖内容 |
|---|---|
| `test_synth_sends_expected_payload_and_yields_chunks` | 请求路径、鉴权、全部 JSON 字段与 PCM 块顺序 |
| `test_from_config_reads_tts_and_audio_settings` / `..._kwargs_override_config` | 模型、音色、采样率、两个截止时间从 config 读取 |
| `test_empty_text_makes_no_request` | 空文本/空白不发请求 |
| `test_cancel_before_start_makes_no_request` / `test_cancel_between_chunks_stops_within_200ms` | 取消不抛异常且 200ms 内结束 |
| `test_first_chunk_timeout` / `test_total_timeout_after_first_chunk` | 首包超时与总时限的消息可区分 |
| `test_consumer_await_is_not_cancelled_by_total_timeout` | 消费者 sleep 不被取消；随后取块抛 `TtsTimeout`；`sys.unraisablehook` 检查无未接收异常 |
| `test_no_retry_after_partial_output` | 已产出音频后不重试，已送出块保持有效 |
| `test_retries_on_429_then_succeeds` / `test_retries_on_transport_error_before_audio` / `test_client_error_not_retried` | 重试边界 |
| `test_empty_audio_response_raises` / `test_missing_api_key_raises` | 异常路径 |

live 用例：合成“您好”得到非空 PCM（> 0.3s），PCM 字节数为偶数；长文本收到第一块后置位 cancel，生成器在 200ms 内结束。

**实测数据（`uv run python tools/bench_tts.py --n 10`，公网 `api.siliconflow.cn`，2026-10-21 17:05 CST）**

最终配置（首包 10s / 总 30s）：

| 句长（字） | 首包 P50 / P95 (ms) | 总耗时 P50 / P95 (ms) | 成功/次数 | 音频 P50 (ms) | 实时率 P50 |
|---|---|---|---|---|---|
| 3 | 3593 / 5840 | 3603 / 5846 | 10/10 | 917 | 0.25× |
| 7 | 3644 / 8124 | 3943 / 8540 | 10/10 | 1697 | 0.43× |
| 18 | 5833 / 8558 | 6879 / 10870 | 7/10 | 3317 | 0.48× |
| 25 | 8748 / 9269 | 12494 / 12810 | 3/10 | 4917 | 0.39× |
| 49 | 3475 / 8233 | 4261 / 8605 | 9/10 | 9137 | 2.14× |
| **总体** | **4358 / 8835**（min 883 / max 9519） | **5145 / 11522**（min 891 / max 12845） | **38/50**，超时 12 | — | — |

同一时段先用 6s/20s 跑过一轮：成功 32/50、超时 18，首包 P50 4148 / P95 5833，总耗时 P50 4260 / P95 6899。两轮都远慢于设计文档 16.2 记录的“首包 p50 2.5s”。

说明与结论：

- 首包 ≈ 总耗时：服务端在响应头之前就把整段音频生成完（单次诊断请求 headers 与首块相差 2ms），因此“首包慢”就是“生成完才返回”，不是本客户端缓冲造成的。
- 长句的实时率可以到 2.1×（音频 9.1s，总耗时 4.3s），说明生成速度本身够快，瓶颈在排队/启动。
- 本次时段服务波动极大（0.9–13s），12 次超过 10s；这不是客户端能修的，T10 必须依赖 T06 预合成常用话术与 T07 模板兜底，不能等云端 TTS。

**遇到的问题与处理**

1. **默认超时值按实测调整**：最初按设计文档的 p95 取 6s，实测 36% 请求超时。改为首包 10s、总 30s（覆盖实测 p95 并留余量），并把理由写进 `services/tts.py` 与 config。设得更紧只会把本来能播的句子丢掉，压延迟要靠预合成，不靠超时。
2. **响应头延迟**：`client.stream()` / `send(stream=True)` 在服务端返回 PCM 流时并不比首块早多少，因此把建连与等头也纳入首包截止；否则“首包超时”会变成“建连之后才开始计时”，与 T02c 的口径不一致。
3. **取消时不要留下 pending 任务**：用 `asyncio.wait` + 显式取消子任务，并在 `finally` 里 `gather(..., return_exceptions=True)` 收掉结果；否则会出现 “Task exception was never retrieved”（回归用例用 `sys.unraisablehook` 检查）。
4. **worktree 的 `.venv`**：新 worktree 默认没有虚拟环境，临时软链到主 worktree 的 `.venv` 跑测试（`.venv` 不入库，提交前删除）。**不影响仓库内容**。

**未决**

- 云端 TTS 首包 4–9s、p95 8.8s，远超设计文档的目标。T06 的预合成命中率、T07 的模板话术是后续能否把端到端压下来的关键；T10 定稿前需要用 T06 的结果重测“命中缓存 + 未命中”两条路径。
- 本次没有测慢速网络/代理差异对首包的影响；如果 T10 的公网表现与 bench 不一致，先查代理与 HTTP 版本。

---

## T04b TTS 修订（第 4 次评审）★

**完成情况**：完成。评审指出的“退避等待期间被打断，内部异常 `_Cancelled` 泄漏给调用方”已修复，回归用例在旧代码上能稳定复现、修复后通过。

**问题与修复**

- 旧实现：`synth()` 在 `except (_HttpStatusError, httpx.TransportError)` 分支里调用 `self._pause(...)`；`_pause` 在 cancel 置位时抛 `_Cancelled`。该异常产生在 except 分支内部，同级的 `except _Cancelled` 接不住，最终泄漏给调用方。
- 修复：`_pause()` 改为返回 `bool`（`True` 表示等待期间被取消），不再抛 `_Cancelled`；`synth()` 据此安静 `return`。现在 `_Cancelled` 只可能从 `_stream_pcm()` 的 `async for` 抛出，位于 `synth()` 的 try 主体内，必被 `except _Cancelled` 接住，**任何路径都不会离开 `synth()`**。
- 未改接口、config 与重试语义（仍是未产出音频前最多重试 2 次、0.5s/1s 退避）。

**回归用例**（`tests/test_tts_client.py::test_cancel_during_retry_backoff_ends_quietly`）

- 场景与评审复现一致：服务端一直返回 429、`retry_base_delay=1.0`、0.2s 后置位 cancel。
- 断言：`synth()` 以 `StopAsyncIteration` 安静结束（不抛 `_Cancelled`）；从 cancel 置位到结束 < 200ms；只发过一次请求。
- 已验证该用例在修复前的代码上失败，异常类型为 `services.tts._Cancelled`（即评审复现的泄漏），修复后通过。

**验收结果**

```
$ uv run pytest -q
    131 passed, 7 deselected
$ uv run pytest tests/test_tts_client.py -q
    17 passed
```

**说明**

- 分支已合并 `origin/main` 上的第 4 次评审提交（`ccc32ce`，含 T03b/T04b 任务书与后续计划调整），本次只实现 T04b，未改动 T03 相关文件。

---

## T08 电话信道模拟

**完成情况**：完成。`tools/channel_sim.py` 实现 `simulate(pcm16k: bytes) -> bytes`，
把 16kHz PCM 依次做「低通 + 2× 抽取 → G.711 μ-law 编码/解码 → 零插值 + 2× 插值」，
输出仍是 16kHz 单声道 16-bit 小端 PCM，且**与输入等长（误差 0 个样本，任务书允许 20ms）**。
`tests/test_channel_sim.py` 32 个用例全通过。

**实现要点**

- **不用 `audioop`**（Python 3.13 已移除）：低通、重采样、μ-law 全部用 numpy 实现。
  `requirements.txt` 未新增依赖（`numpy` 已就位）。
- **抗混叠低通**：63 抽头 Blackman 窗 sinc FIR，截止 3400Hz（G.711 采样率 8kHz，
  话音频段约 300–3400Hz），直流增益归一化为 1。降采样与升采样共用同一条滤波器；
  零插值后乘 2 补偿幅度减半。两端用边缘值延拓，避免首尾瞬态被当成语音。
- **G.711 μ-law**：按 ITU-T G.711（正值码字 MSB=1，`+0=0xFF`、`-0=0x7F`），
  段表用 `np.searchsorted` 查表，避免浮点 `log2` 的边界误差；编码前 `CLIP=32635` 削顶，
  不让 int32 溢出环绕。
- **静音不做抖动（dither）**：μ-law 零码字解码为精确 0，全零输入输出全零。
  这样 VAD／能量阈值测试不会被本模块引入的底噪干扰。

**验收结果**

```
$ uv run pytest tests/test_channel_sim.py -q      → 32 passed in 0.16s
$ uv run pytest -q                                 → 242 passed, 10 deselected in 2.52s
```

| 验收要求 | 用例 |
|---|---|
| 输出长度（误差 ≤ 20ms） | `test_output_length_matches_input`（1/2/3/159/160/161/1599/1600/8000 点，逐点等长）、`test_empty_input_returns_empty` |
| 8kHz 以上频率能量明显衰减 | `test_frequencies_above_4khz_are_attenuated`（5k/10k/12kHz，要求 > 20dB）、`test_4khz_sits_in_transition_band_and_is_suppressed` |
| 静音输入输出仍为静音 | `test_silence_stays_silent`、`test_no_noise_added_to_digital_silence` |
| 语音带不被破坏（自加） | `test_speech_band_is_preserved`（300/500/1k/2k/3kHz，0.7 < 能量比 < 1.3） |
| μ-law 自洽（自加） | 零值精确、符号位约定、解码单调、量化误差上界、削顶不环绕 |
| 其它（自加） | 奇数字节报错、输出 `<i2` 且不贴 int16 rails/无爆音环绕、确定性、不改写入参 |

**实测数据**（`amplitude=0.5` 的 16kHz 正弦波、0.5s，输出/输入 RMS 比）

| 输入频率 | 能量比 | 衰减 |
|---|---|---|
| 300 Hz | 1.002 | +0.02 dB |
| 500 Hz | 0.999 | −0.01 dB |
| 1 kHz | 1.003 | +0.03 dB |
| 2 kHz | 1.023 | +0.20 dB |
| 3 kHz | 0.901 | −0.91 dB |
| 3.4 kHz（截止） | 0.253 | −11.9 dB |
| 4 kHz | 0.0094 | −40.5 dB |
| 5 kHz | 0.0051 | −45.9 dB |
| 10 kHz | 0.0029 | −50.8 dB |
| 12 kHz | 0.0094 | −40.5 dB |

8kHz 恰好是 8kHz 采样的奈奎斯特频率，输入正弦全部落在零点上，不列入表格。

处理开销（同一台机器，冷启动后重复 20/50 次）：

| 输入 | 耗时 P50 | 最大 |
|---|---|---|
| 1 秒音频（16000 点） | 0.8 ms | 0.9 ms |
| 32ms 帧（512 点） | 0.079 ms | 0.162 ms |

32ms 帧 < 0.2ms，实时链路（T10/T13）可以逐帧调用，不需要单独线程。

**遇到的问题与决定**

1. **满量程输入会有轻微削顶**：μ-law 解码峰值 32124，插值滤波器约 2% 过冲，
   `amplitude=1.0` 时最低样本会贴到 −32768。这是真实线路也有的削顶，不做额外处理；
   正常说话电平（0.5–0.9 满量程）不会触发，测试用 0.8 满量程断言"不贴 rails"。
2. **3.4kHz（截止频率）衰减 −11.9 dB**：单条 63 抽头滤波器在截止点本来就有 −6dB，
   降采样 + 升采样走两遍即 −12dB。若 T12 听感评估觉得语音太闷，可把截止抬到
   3600Hz 或增加到 95 抽头（代价是延迟从 31 样本增到 47 样本，仍 < 3ms），当前按
   任务书的"先低通、再 8kHz、再升回"实现，未提前调参。
3. **未加线路噪声/电平衰减/回声**：任务书只要求低通 + μ-law + 重采样。真实的
   丢包、抖动、回声在 T09/T10 的缓冲与打断逻辑里另有处理，等 T12 有数据再决定
   是否补充；避免现在把 VAD 阈值调到适配不存在的人工噪声。

**未决**：`simulate()` 目前是整段函数，T13 需要在页面上做「信道模拟开关」并逐帧
调用；届时若需要按帧保留滤波器状态（当前每帧两端边缘延拓，帧间会有极小的不连续），
再增加流式接口 `ChannelSimulator.feed(frame)`，不影响现有 `simulate()` 调用方。

---

## T09 VAD 与轮次判断

**完成情况**：完成。`core/turn.py` 提供 `TurnDetector.feed()` 与 `segment_wav()`；13 个新用例全过，全量 `pytest` 223 passed。

**文件**

- `core/turn.py`：`SileroVad`（ONNX）、`SmartTurn`（v3.2 ONNX）、`TurnDetector`、`segment_wav()`。
- `tests/test_turn.py`：12 个 fake 模型用例 + 1 个真实模型用例。
- `config.yaml`：新增 `vad` 段（见下表）。
- `requirements.txt`：新增 `pipecat-ai==1.13.0`。

**模型与依赖的选择（与任务书措辞的偏差，需评审确认）**

- **没有安装 `silero-vad` 包**：该包即使只用 ONNX 也硬依赖 `torch`（约 GB 级）。改用 pipecat 1.13 自带的 `pipecat/audio/vad/data/silero_vad.onnx`（与 silero-vad 官方 ONNX 模型同一份）和 `SileroOnnxModel` 包装（含 64 采样 context 与 recurrent state 处理），逐帧 32ms / 512 采样。
- **Smart Turn 直接复用 pipecat 的 `LocalSmartTurnAnalyzerV3`**（`smart-turn-v3.2-cpu.onnx` + `_whisper_features.py` 的 numpy log-mel），本项目调用它的 `_predict_endpoint()`。这是私有方法，因此 `requirements.txt` 把 `pipecat-ai` 固定为 `1.13.0`；升级前要确认方法还在。
- 任务书里的 `Event` 在实现中命名为 `TurnEvent`（避免和 `core/events.py` 的日志事件混淆）；`t_ms` 是**音频时间轴**毫秒（32ms 帧累加），不是墙钟，便于 T10 按它切 wav；在线链路由调用方用 `EventLogger` 记 `time.monotonic()`。
- `feed()` 接受任意长度 PCM，内部按 1024 字节切帧，末尾不足一帧的留到下次；不用调用方凑整。

**参数（`config.yaml: vad`）**

| 参数 | 值 | 含义 |
|---|---|---|
| `threshold` | 0.5 | Silero VAD 判为语音的概率阈值 |
| `speech_end_silence_ms` | 200 | 连续静音达到该值发 `speech_end`，并立刻跑一次 Smart Turn |
| `stop_secs` | 3.0 | Smart Turn 判为没说完时，静音超过该值兜底发 `turn_end` |
| `pre_speech_ms` | 500 | 送 Smart Turn 的预卷；`segment_wav` 的每轮起点也前移这么多（不早于 0／上一轮结束） |
| `max_segment_s` | 8.0 | 单轮送模型的最长音频（模型本身只取最后 8s） |

**验收结果**

```
$ uv run pytest tests/test_turn.py -q
    13 passed
$ uv run pytest -q
    223 passed, 10 deselected
```

用例覆盖：纯静音无事件；一段语音后 `speech_end` + `turn_end`；96ms 短停顿不触发 `turn_end`；停顿 416ms 触发 `speech_end` 但 Smart Turn 判没说完时继续同一轮、重新开口只补 `speech_start`；连续判没说完在 `stop_secs` 兜底；预卷被送进 Smart Turn；任意切分的 feed 与整块 feed 事件一致；`reset()`；`segment_wav` 两轮边界与末尾截断；`from_config`；真实模型跑 `scenarios/asr_02.wav`。

**实测数据（本机 WSL2 CPU，2026-10-21）**

- VAD 单帧（32ms）：273 帧，P50 0.11ms、P95 0.16ms、max 0.41ms。
- Smart Turn 单次（3.2s 输入）：20 次，P50 83ms、P95 102ms、max 102ms。
- 整段 `scenarios/asr_02.wav`（2.52s，追加 1s 静音）：`speech_start@320ms` → `speech_end@2528ms` → `turn_end@2528ms`（`reason=smart_turn`，prob 0.98）；`segment_wav` 返回 1 段 `(0, 2517)`。

**遇到的问题**

1. **`scenarios/asr_01.wav`（T03 的 8.8s TTS 样例）被切成 4 轮**：TTS 在数字之间留了 224–320ms 停顿，超过 200ms 的 `speech_end` 阈值，Smart Turn 每次都以 0.90–0.99 概率判为“说完”。这是 200ms 阈值（任务书规定）加 TTS 合成节奏导致的，不是状态机错误；真人语速下是否合适要用 T12 的真人录音校准。已实测记录，暂不调参。
2. **`pipecat` 导入时的告警**：`importlib.resources.path` 的 `DeprecationWarning` 与 `audioop` 的 `DeprecationWarning`（Python 3.13 移除）。前者来自 pipecat 的模型路径解析，后者来自它未走到的 `audioop` 兼容层；本项目在 3.12 上运行，不影响功能，未改第三方源码。
3. `_predict_endpoint()` 是私有 API；已在 `core/turn.py` 顶部和 requirements 注释里写明版本约束。

**给 T10 的接口**

```python
det = TurnDetector.from_config(config)          # 真实模型；测试可注入 vad=/smart_turn=
events = det.feed(pcm_chunk)                    # TurnEvent(kind, t_ms, speech_ms, probability, reason)
segments = segment_wav(pcm, detector=det)        # [(start_ms, end_ms), ...]，读 wav 后直接切轮
```

---

## T05 本地 CosyVoice3 服务（Windows 试验 → 本地主路径）★

**完成情况**：服务部署成功、可流式合成 16kHz PCM、压测跑完。**未达到任务的目标参考“短句首包 P50 ≤ 0.8s”**，原因与可选方案见下。任务要求的“失败时如实记录、不硬凑结果”按此执行。

**产物**

- `local_tts/server.py`：FastAPI `POST /synth` + `GET /health`；模型常驻、启动预热；24kHz→16kHz 用 `soxr.ResampleStream` 流式重采样；按请求串行推理。
- `local_tts/bench.py`：5 句 × n 次，首包/总耗时 P50/P95、实时率、后台采样 GPU 显存峰值。
- `local_tts/README.md`：实际安装步骤、安装冲突与处理、启动/冒烟/压测命令、实测结论与限制。
- `services/tts_local.py`：`LocalTtsClient`，与云端 `TtsClient` 同一接口；复用 T04/T04b 的超时、取消、重试语义；本地服务不鉴权。
- `tests/test_tts_local.py`：11 个离线用例。
- `tests/live/test_tts_local_live.py`：连已启动服务的 live 用例（服务未启动时跳过）。
- `config.yaml`：`tts.backend: local`、`tts.local_url: http://127.0.0.1:50000`。

**环境（仓库外，不入库）**

| 项 | 值 |
|---|---|
| 官方仓库 | `~/code/ai/CosyVoice`（含 Matcha-TTS 子模块） |
| conda 环境 | `~/miniconda3/envs/cosyvoice`，Python 3.10.22 |
| 模型 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，`pretrained_models/Fun-CosyVoice3-0.5B`，9.1GB |
| 推理 | `AutoModel(..., fp16=True)`；参考音色用官方 `asset/zero_shot_prompt.wav` + 提示文本，启动时 `add_zero_shot_spk()` 缓存 |
| 启动耗时 | `/health` 变 ok 约 57s（含首次 wetext 资源下载；之后约 20s） |

**接口/配置变更记录（AGENTS.md 规则 2）**

- `services/tts.py`（T04 代码）小重构：新增类属性 `endpoint`、`_build_payload()`；`api_key` 显式传空字符串时表示该后端不需要鉴权（不发 `Authorization`）。云端行为不变，`LocalTtsClient` 通过子类复用流式/超时/取消/重试。
- 新增 config 键 `tts.backend`、`tts.local_url`。`core/types.py` 与 0.3 的接口约定未改。
- `TtsClient` 的构造签名与对外方法未变。

**安装问题与处理**

1. `pip install -r requirements.txt` 在 `openai-whisper` 报 `No module named 'pkg_resources'`：新版 setuptools 移除 pkg_resources。处理：先装 `setuptools<81`，再 `--no-build-isolation`。openai-whisper 仅训练/评测用到，推理不需要。
2. `pyworld` 在 `--no-build-isolation` 下缺 numpy：先装 `numpy==1.26.4` + Cython。
3. `tensorrt-cu12` 元包报 `Cannot import 'wheel_stub.buildapi'`：装 `wheel_stub` 后重试。TensorRT 只有 `load_trt=True` 才用，本服务未启用。
4. PyTorch 下载慢：主源用清华 PyPI，`download.pytorch.org/whl/cu121` 作 extra index；最终 `torch 2.3.1+cu121`，`torch.cuda.is_available()=True`。

**关键实现选择**

1. **fp16**：官方 `AutoModel(fp16=True)`。实测比 fp32 明显快（短句首包约 1.3s vs 2.2–3.0s），显存约 4.9GB allocated，16G 足够。
2. **流式重采样**：模型输出 24kHz，下游要求 16kHz。用 `soxr.ResampleStream` 保留重采样状态，避免逐块独立重采样在块边界产生咔哒声。
3. **参考音色缓存**：`add_zero_shot_spk()` 把提示音频的 token/feature 缓存进 spk2info，避免每次请求重新编码 3.5s 的参考音频。
4. **客户端复用而非复制**：`LocalTtsClient` 只覆盖 `endpoint` 与 `_build_payload()`，T04b 的“退避期间被打断不泄漏 `_Cancelled`”等语义自动继承；离线用例也验证了这些语义在本地客户端成立。
5. **服务端串行**：一把进程内锁，T05 只要求单路实时合成；多路并发会排队。

**验收结果**

```
$ conda run -n cosyvoice python local_tts/server.py --fp16
    ... local tts ready: model=.../Fun-CosyVoice3-0.5B sr=24000 load=57.3s
$ curl -s http://127.0.0.1:50000/health
    {"status":"ok","sample_rate":16000,"model_sample_rate":24000,"load_s":57.31,"fp16":true}
$ uv run python local_tts/bench.py --n 5
    requests=25  ok=25  timeouts=0  failures=0
$ uv run pytest -q
    142 passed, 9 deselected
$ uv run pytest -m live tests/live/test_tts_local_live.py -q
    2 skipped（当时服务未启动；bench 已在服务运行时完成同样的合成与取消验证）
```

**实测数据（`uv run python local_tts/bench.py --n 5`，2026-10-21 18:15 CST，fp16）**

| 句长（字） | 首包 P50 / P95 (ms) | 总耗时 P50 / P95 (ms) | 音频 P50 (ms) | 实时率 P50 |
|---|---|---|---|---|
| 3 | 1098 / 1465 | 1116 / 1485 | 840 | 1.33 |
| 7 | 1917 / 2643 | 1938 / 2664 | 1840 | 1.05 |
| 18 | 3500 / 3673 | 3529 / 3696 | 3720 | 0.95 |
| 25 | 4513 / 5278 | 5452 / 6409 | 5680 | 0.96 |
| 49 | 5010 / 5598 | 10535 / 10951 | 10600 | 0.99 |
| **总体** | **3500 / 5278**（min 988 / max 5714） | **3529 / 10624** | — | — |

GPU 显存（整卡采样，含其它进程）：baseline 9298 MiB，peak 9765 MiB，增量 467 MiB；
模型常驻约 4.9GB allocated。

**与 T04 云端对比（同一台机器，不同时段）**

| 指标 | 云端 T04 | 本地 T05 | 变化 |
|---|---|---|---|
| 首包 P50 / P95 | 4358 / 8835ms | **3500 / 5278ms** | P50 −0.86s，P95 −3.56s |
| 总耗时 P50 / P95 | 5145 / 11522ms | 3529 / 10624ms | P50 −1.62s，P95 −0.90s |
| 成功/次数 | 38/50（12 次超时） | 25/25 | — |
| 短句（3 字）首包 P50 | 3593ms | 1098ms | −2.50s |
| 7 字首包 P50 | 3644ms | 1917ms | −1.73s |
| 18 字首包 P50 | 5833ms | 3500ms | −2.33s |
| 25 字首包 P50 | 8748ms | 4513ms | −4.24s |
| 49 字首包 P50 | 3475ms（快速突发） | 5010ms | +1.54s |

**结论：自部署首包是否比云端快至少 1 秒？**

- 短句与中等长度：**是**，快 1.7–4.2s，且不受公网抖动影响（云端 12/50 超时，本地 0/25）。
- 整体首包 P50：**否**，只快 0.86s；整体 P95 快 3.56s。
- 长句：本地首包 5.0s 比云端那次突发的 3.5s 慢，但云端同句在其它时段明显更慢；不能把一次突发当稳定值。
- **未达到 0.8s 目标**：CosyVoice3 流式推理至少要攒够一个 `token_hop_len`（25 token≈1s 音频）才出第一块；短句 token 数接近该阈值，实际要等整句生成完。4060 Ti fp16 的实时率约 0.95–1.3×，首包与音频长度基本成正比。这是模型/实现限制，不是网络或客户端缓冲。

**已核实的替代方案**

- **非流式整句合成**（同机、fp16，3 次平均总耗时）：3 字 1161ms、7 字 2117ms、18 字 3505ms、25 字 4489ms、49 字 8618ms。短句与流式首包几乎一样，长句更差（要等整句），不能替代流式。
- 未完成、不能声称已核实：CosyVoice2-0.5B 本地部署（本次下载被中断）；TensorRT/JIT（仓库提供 ONNX，可现场编译 plan）；调小 `token_hop_len`（官方注释要求与训练 static_chunk_size 一致，需先验证音质）。这些留给评审决定是否继续投入。

**遇到的问题与处理**

1. **wetext 首次初始化需要下载资源**，首次启动曾因 403/网络失败退化为“无文本正则化”；本次启动时下载成功，服务日志显示正常。无网络时会退化，但不影响 PCM 输出；号码读法后续由 T06 数字片段拼接解决。
2. **启动慢（57s）**：模型加载 11s + 首次预热 2.4s + 首次 wetext 资源下载/初始化；之后重启约 20s。bench 会等 `/health`，不会把加载时间算进首包。
3. **GPU 显存采样包含其它进程**：baseline 9298MiB 中约 4.4GB 是别的进程/常驻模型，peak 9765MiB；报告里给出 baseline、peak 和增量，避免把整卡占用当成模型占用。
4. **服务端取消**：客户端 `cancel` 会立即停止读取并关闭连接；服务端不保证中断已开始的 GPU 推理（会跑完当前请求）。T05 未要求服务端中断，已在 README 限制里写明。
5. **CosyVoice2 对比测试中途停止**：为评估替代方案尝试下载 `iic/CosyVoice2-0.5B`，因用户要求先交付而中断；进度里明确写“未核实”，不拿未完成的数据下结论。

**未决**

- 0.8s 目标未达成。若评审认为本地 TTS 必须继续压首包，建议优先级：① TensorRT 编译 flow decoder（仓库自带 ONNX）；② 验证更小的 token hop 对音质/重复推理的影响；③ 换更强 GPU 或 vLLM；④ 本机跑 CosyVoice2-0.5B 对照。
- T06/T07 应按 `tts.backend` 选择后端，并把高频话术预合成；本地 TTS 的 3.5s 中位首包仍然偏慢，模板命中率决定 T10 的端到端表现。
- 本次未在 pytest 里自动拉起服务；`tests/live/test_tts_local_live.py` 需要先手动启动服务（README 已写清），服务未启动时跳过。

---

## T07 槽位与状态机（纯逻辑）★

**完成情况**：完成。`core/slots.py`（`SlotStore`）与 `core/dialog.py`（`Dialog` + `ReplyHint`）
落地，44 个用例全过，两个模块覆盖率 100%，全量 `pytest` 310 passed。

**产物**

- `core/slots.py`
  - `SlotStore`：5 个槽位的值与状态。`update()` 新值 → `unconfirmed`、改写已有值 → `corrected`；
    `confirm()` / `confirm_all()` → `confirmed`；`missing_required()` 返回没值且没被放弃的必填槽位；
    `abandon()` 标 `unconfirmed`（值为空）并跳过；`as_dict()` 返回副本给提示词与事件日志。
  - `update_phone_from_text(text, allow_landline=False)`：对 ASR 原话调用
    `core.normalize.extract_phone()`，提取到才写入 `callback_number`（T03b 的 `allow_landline` 透传）。
  - `update()` 里出现的 `callback_number`、白名单外的键、非字符串/空白值一律忽略（不抛异常）。
- `core/dialog.py`
  - `Dialog.from_config(config)`、`greeting() -> ReplyHint`、
    `on_user_turn(text, slots_update=None, parse_ok=True) -> (State, ReplyHint)`。
  - 只读属性：`state`、`slots`（`dict[str, SlotValue]`，可直接喂 `llm.turn_stream`）、
    `attempts`（>0 的追问次数，直接喂提示词）、`asked_slot`（上一句在问哪个槽位，T10 判断
    “是否在追问号码”用）、`slot_store`。
  - `ReplyHint(kind, template_id, text, phone)` + `use_llm` / `use_cache`：
    模板话术带 id 与填好 `owner` 的文本，`phone` 非空时调用方应改用 T06 的
    `PhraseCache.render_phone(phone)`。
  - 话术表 `PHRASES`（id 与 T06 任务书对齐：`greet` / `ask_name` / `ask_purpose` /
    `ask_callback` / `repeat` / `confirm_phone_prefix` / `confirm_phone_suffix` /
    `forward` / `goodbye`），关键词表 `END_INTENT_KEYWORDS`、`AFFIRMATIVE_KEYWORDS`、
    `NEGATION_KEYWORDS` 都在模块顶部。
- `config.yaml`：新增 `dialog.max_asks: 2`（AGENTS.md 规则 4：阈值不硬编码在代码里）。
- `tests/test_dialog.py`：44 例（`TestSlotStore` 15 例 + `TestDialogFlow` 29 例），全离线。

**接口决定（需 T06/T10 知晓，非 0.3 共享契约变更）**

1. `ReplyHint` 是本任务新定义的类型，放在 `core/dialog.py`（不进 `core/types.py`，因为只有
   Dialog 与 T10 用）。T10 按 `hint.kind` 分流：`template` → 查缓存 / `render_phone`，`llm` → 播 LLM 句子。
2. **`PHRASES` 暂放在 `core/dialog.py`**：T06（`core/phrases.py`）尚未实现，而 T07 需要给出
   “可预合成的模板 id + 文本”。T06 落地时请把话术表迁到 `core/phrases.py` 并由 `dialog` 引用
   （或让 `phrases` 直接 `from core.dialog import PHRASES`），**不要各写一份**——同一个 id 的文本
   不一致会让预合成缓存与现合成句不是同一句话。T06 任务书里还有 `ask_time`、`please_wait_confirm`
   两个 id 本任务没用到（选填槽位不主动追问、确认复述直接交给 LLM），T06 自己补即可。
3. `attempts` 的语义是**已经问出口的次数**（不是失败次数）。问过回拨号码而本轮没拿到号码，
   下一轮 `_advance()` 再计一次，因此两次空手而归后即 `abandon`，与设计文档第 4 节
   “同一槽位最多追问 2 次”一致。问候语里的身份询问不计入（否则姓名只有 1 次追问机会）。
4. `SlotValue(None, "unconfirmed")` 表示“问过但没拿到、已跳过”，与“有值但未确认”
   （`value` 非空 + `unconfirmed`）靠 `value` 区分；`SlotStore.is_abandoned()` 显式记录。

**关键实现选择**

1. **号码提取先于 `parse_ok` 判断**：号码来自 ASR 原文 + 规则，与 LLM 是否解析成功无关，
   因此 LLM 输出坏掉的一轮也照样能记录号码（有用例）。
2. **`allow_landline` 只看上一句问的是不是号码**：`asked_slot == "callback_number"` 时传 True，
   其余情况保持 T03b 的严格模式，所以不在追问号码时“预约在 2026-10-21”不会写进槽位（有用例）。
3. **`CONFIRM` 里“带改动的肯定”按纠正处理**：`changed` 非空就不确认、直接重新复述，
   否则“对，我姓王”会把刚改过的姓名误标成 `confirmed`。手机号用值是否变化判断改动，
   因为 `update_phone_from_text()` 的返回值只表示“提取到了号码”。
4. **否定判定先于肯定判定**：「不是」「不对」里含「是」「对」，顺序反了会把否定当肯定。
5. **`parse_ok=False` 的一轮不推进状态、不计追问失败**，只播 `repeat`；避免 LLM 抖动把
   追问次数耗光。
6. **全部槽位被放弃时用 `forward` 收尾**：什么也没记到就不再复述（复述空内容没有意义），
   改用“我会转告，请留联系方式”并进入 `CLOSING`。
7. **选填槽位（`time` / `location`）从不主动追问**，只在对方主动说出时记录（T07 任务书只要求
   追问必填项；避免把“顺便问一下时间/地址”变成额外的 TTS 轮次）。

**验收结果**

```
$ uv run pytest tests/test_dialog.py -q                      → 44 passed
$ uv run --with pytest-cov pytest tests/test_dialog.py -q --cov=core.slots --cov=core.dialog
    core/slots.py 100% | core/dialog.py 100%
$ uv run pytest -q                                           → 310 passed, 12 deselected
```

任务书要求的用例逐项对应：

| 验收要求 | 用例 |
|---|---|
| 正常流程走完 | `test_normal_flow_reaches_closing` |
| 追问 2 次后跳过 | `test_two_asks_then_skip_name`、`test_callback_number_skipped_after_two_asks`、`test_max_asks_is_configurable` |
| 中途结束意图 | `test_end_intent_in_any_state`、`test_end_intent_during_confirm` |
| 纠正流程（“不是今天，是明天下午三点” → `time` 为 `corrected`） | `test_correction_in_confirm_updates_status_and_reconfirms`、`test_confirm_affirmative_after_correction` |
| `parse_ok=False` | `test_parse_not_ok_keeps_state_and_repeats`、`test_parse_not_ok_still_extracts_phone` |
| 非法号码被忽略 | `test_invalid_phone_is_ignored` |
| 汉字数字号码从原话提取 | `test_phone_from_chinese_digits_is_extracted` |
| `slots_update` 里的 `callback_number` 被忽略 | `test_invalid_phone_is_ignored`（第二段）、`TestSlotStore.test_callback_number_in_update_is_ignored` |
| 不在追问号码时“预约在 2026-10-21”不写入 | `test_date_not_stored_when_not_asking_phone` |
| 在追问号码时只报“八二三四五六七八”能写入 | `test_landline_accepted_while_asking_phone` |

另补：只有号码时的复述走 `confirm_phone` 模板（`test_phone_only_confirm_uses_render_phone_template`）、
全部放弃时 `forward` 收尾、`max_asks=0` 报错、`from_config` 读 `owner.name` 与 `dialog.max_asks`、
`attempts`/`slots`/`asked_slot` 暴露给 T10、选填槽位只记录不追问、断言话术表里没有写死机主名字。

**实测数据**

本任务无网络调用，没有时延数据。逻辑开销可忽略（44 例含覆盖统计 0.21s）。
端到端表现要等 T10 把这些模板接到 T06 缓存后统计命中率。

**遇到的问题与处理**

1. **`test_turn.py` 的真模型用例在 main 上就失败**（`ModuleNotFoundError: pipecat`）：
   共享的 `.venv` 里没有装 `requirements.txt` 中的 `pipecat-ai==1.13.0`（之前 T09 是在别的
   环境里跑的）。**不是本次改动引起的**：在 main 上 `uv run pytest tests/test_turn.py::test_real_models_on_sample_wav`
   同样失败。处理：`uv pip install pipecat-ai==1.13.0`（只改 `.venv`，`.venv` 不入库），
   之后全量 310 passed。提醒后续任务在新 worktree 里软链共享 `.venv` 时注意依赖是否齐全。
2. **纠正流程的状态归属**：任务书写“否定或纠正 → 回到 FILL_MISSING”，但纠正后若必填项都已齐，
   再停在 `FILL_MISSING` 会让提示词按“追问缺失项”生成的话术变成无内容可问。实现为
   “先置 `FILL_MISSING` 再走一遍 `_advance()`”：有缺口就追问（`FILL_MISSING`），没有缺口
   就重新复述（`CONFIRM`），对应设计文档“被纠正的槽位更新后重新复述”。
3. **关键词是子串匹配的启发式**：「我没有听清」不会被判为否定（关键词是「没有」而非「没」），
   但「我没有」会。真实文本下先按这个保守表跑，T12 的场景集统计误判率后再调。

**给 T10 的接口**

```python
dialog = Dialog.from_config(config)
hint = dialog.greeting()                       # ReplyHint(kind="template", template_id="greet", text=...)
state, hint = dialog.on_user_turn(
    asr_text, result.slots_update, result.parse_ok,
)                                              # -> (State.CONFIRM, ReplyHint(...))
llm.turn_stream(state, dialog.slots, history, asr_text, attempts=dialog.attempts)
```

**未决**

- `PHRASES` 与 T06 `core/phrases.py` 的合并方式（见“接口决定”第 2 条），T06 落地时必须处理。
- `render_phone` 的调用时机由 T10 负责：`hint.phone` 非空时优先
  `cache.render_phone()`，缓存未命中再退回 `hint.text` 现合成。
- 关键词表（结束意图、肯定/否定）目前是中文子串匹配，T12 用场景集统计漏判后再调。

---

## T07b 确认与结束意图修订（第 9 次评审）★

**完成情况**：完成。评审背景表里的 11 行全部有用例，判定改为“整句意图”，号码复述一律走
`render_phone`，`CONFIRM` 有了出口。`pytest tests/test_dialog.py tests/test_prompts.py` 114 passed，
三个模块覆盖率 100%，全量 366 passed。

**产物**

- `core/dialog.py`
  - 新增 `classify_confirm_reply(text) -> "affirmative" | "negative" | "unclear"`（整句意图判定）；
    常量 `AFFIRMATIVE_KEYWORDS` / `NEGATIVE_PATTERNS`（正则）/ `QUESTION_MARKERS` /
    `POLITENESS_PHRASES` / `HEDGE_PHRASES` / `END_CONTINUATION_MARKERS`。
  - `Dialog.__init__` 增加 `max_confirm_rounds`（默认 2）、`max_parse_failures`（默认 2）；
    `from_config` 读 `dialog.max_confirm_rounds`。
  - `ReplyHint.phone` 语义按评审要求写清：`template`+`phone` 是纯号码复述；
    `llm`+`phone` 是先播 LLM 复述再接 `render_phone`。
- `core/prompts.py`
  - `CONFIRM_GUIDE_WITH_PHONE`（号码由系统播报，LLM 只复述姓名/来意、不要以问句结尾）。
  - `CONFIRM_GUIDE_NEGATIVE`（对方否定时让 LLM 问哪一项不对，不要原样复述）。
  - `STATE_GUIDE` 接口不变，选择逻辑放在 `_state_guide()`，由 `build_user_message` 调用。
- `config.yaml`：`dialog.max_confirm_rounds: 2`。
- `tests/test_dialog.py`：新增 `TestClassifyConfirmReply`、`TestConfirmIntent`、`TestEndIntent`、
  `TestParseFailures`，共 70 例（原 44 例保留，仅把 `from_config` 用例补上 `max_confirm_rounds`）。
- `tests/test_prompts.py`：新增 6 例（两种 CONFIRM 说明的选择、否定说明、不串到其它状态）。

**判定规则（要求 1）**

顺序固定，先疑问、再否定、最后肯定：

1. 规范化后为空 → `unclear`；
2. 含疑问标记（`？`/`?`/`吗`/`什么`/`哪`/`是不是`/`怎么`/`多少`/`如何`）→ `unclear`
   （“你刚才说的是什么？”“对了吗？”“是不是这样”）；
3. 删掉客套话（“对不起”“不好意思”“抱歉”“谢谢”“没关系”“不客气”）与含糊语
   （“好像”“似乎”“大概”“印象中”）和标点；
4. 命中否定句式 → `negative`：`不对`、`不是`、`不准确`、`有误`、`不可以`、`不行`、
   `要改`、`改一下`、`麻烦改`、`少[位一了\d]`、`多[位一了\d]`、`缺`、`漏`、
   `(?<![没不])错`（“没错”“不错”因此不算否定）；
5. 命中肯定词 → `affirmative`：`没问题`、`没有问题`、`没错`、`正确`、`对的`、`是的`、
   `可以`、`好的`、`行`、`对`、`是`、`好`、`ok`；
6. 都不命中 → `unclear`（“我没听清”“嗯”）。

背景表逐行实测（`_at_confirm()` 后喂一句，括号里是实际返回的 `ReplyHint`）：

| 状态 | 对方说 | 修正后结果 |
|---|---|---|
| CONFIRM | “对不起，我没听清” | `CONFIRM`，`llm`+`phone`（重新复述），槽位不确认 |
| CONFIRM | “好像号码少了一位” | `CONFIRM`，`llm` 且 `phone=None`（追问哪一项不对） |
| CONFIRM | “你刚才说的是什么？” | `CONFIRM`，`llm`+`phone`（重新复述） |
| CONFIRM | “对，没有问题” | `CLOSING`，全部 `confirmed` |
| CONFIRM | “不好意思，是对的” | `CLOSING`，全部 `confirmed` |
| CONFIRM | “对，就这样吧” | `confirm_all()` 后 `CLOSING` |
| CONFIRM | 连说 4 次“不对” | 前 2 次追问哪一项不对，第 3 次起 `CLOSING`+`goodbye`，槽位保持未确认 |
| CONFIRM | “不对，是八二三四五六七九” | 号码更新为 `82345679`（`corrected`），重新复述 |
| 通话初期 | “也没什么事，就是想问下快递放哪” | 继续（`ASK_IDENTITY`） |
| 通话初期 | “我就这样跟你说吧，合同的事很急” | 继续 |
| 通话初期 | “不麻烦的话让他回个电话” | 继续 |

**结束意图（要求 2）**

三个条件同时成立才挂断：① 本轮没有任何新槽位值；② 命中 `END_INTENT_KEYWORDS`；
③ 关键词后面没有接着说事——先看 `END_CONTINUATION_MARKERS`（就是/但是/不过/可是/还有/
另外/想/让他/麻烦他/的话），再把尾部剩下的标点、语气词、客套与肯定用语、其它结束关键词
剥掉，仍有内容就认为对方还在说事（`_substantive_tail()`）。
所以“先这样吧”“没事了，谢谢”照旧挂断，“就这样，我还有件事”“我就这样跟你说吧，合同的事很急”
不会挂断。`CONFIRM` 里“结束意图＋肯定”先 `confirm_all()` 再 `CLOSING`
（“对，就这样吧”“好的，没事了，拜拜”）；不带肯定的“先这样吧”照旧保持未确认。

**CONFIRM 出口（要求 3）**

- 否定且没给新值 → `ReplyHint(kind="llm")`，追问哪一项不对；
- 无法判定 → 重新复述（`llm`+`phone` 或纯号码模板）；
- 两者合计每轮 +1，超过 `dialog.max_confirm_rounds`（默认 2）→ `CLOSING`+`goodbye`，槽位保持未确认；
- 有进展（收到新槽位值或肯定）时计数归零。新增只读属性 `confirm_rounds` 供测试/日志用。

**其它要求**

- 要求 4：`CONFIRM` 中调用 `update_phone_from_text` 时 `allow_landline=True`
  （刚复述过号码，对方常直接报正确的号），座机纠正例已覆盖。
- 要求 5：号码有值时 `ReplyHint.phone` 一律填上；除号码外没有其它槽位时才用
  `template_id="confirm_phone"` 的纯号码模板，其余情况都是 `llm`+`phone`，由 T10 接着播
  `render_phone`。唯一留空的是“追问哪一项不对”——那一轮不该再念旧号码。
- 要求 6：`parse_ok=False` 连续超过 2 次后不再播 `repeat`，改为重播当前追问槽位的模板
  （`CONFIRM` 中重新复述）；解析成功即归零。
- 要求 7：没有新建 `core/phrases.py`，`PHRASES` 仍留在 `core/dialog.py` 等 T06 迁移。

**验收结果**

```
$ uv run pytest tests/test_dialog.py tests/test_prompts.py -q   → 114 passed
$ uv run --with pytest-cov pytest tests/test_dialog.py tests/test_prompts.py -q \
      --cov=core.dialog --cov=core.prompts --cov=core.slots
    core/dialog.py 100% | core/prompts.py 100% | core/slots.py 100%
$ uv run pytest -q                                              → 366 passed, 12 deselected
```

**遇到的问题与处理**

1. **`CONFIRM` 的提示词会把“追问哪一项不对”改回“复述”**。要求 5 的说明是
   “复述姓名与来意、不要以问句结尾”，而要求 3 要 LLM 问“哪一项不对”，两条要求依赖同一段说明。
   若为此给 `build_messages()` 加参数，`services/llm.py` 的 `turn_stream` 也得跟着改，
   而 T07b 的文件清单不含 `services/llm.py`。处理：`build_user_message` 本来就收 `user_text`，
   于是在 `_state_guide()` 里用同一个 `classify_confirm_reply()` 判一下这轮是不是否定，
   是就用 `CONFIRM_GUIDE_NEGATIVE`。**不新增接口**，`STATE_GUIDE` 的键值与含义不变，
   `prompts.py` 因此 `from core.dialog import ...`（无循环依赖：`dialog` 不 import `prompts`）。
2. **“否定追问”这一轮 `ReplyHint.phone` 留空**，与“号码有值时一律填上”字面有张力。
   理由是这一轮不该再念旧号码（对方刚说不对），填上会让 T10 播完“哪一项不对”又接一遍旧号码。
   已在 `ReplyHint` 的文档字符串里写明这是唯一例外。
3. **否定正则的边界**：`错` 用 `(?<![没不])错` 排除“没错”“不错”；“不好”从否定表里删掉，
   因为“不好意思”更常见，且“不好”作为单独否定在真实对话里几乎总带上下文；
   “没有”也不再算否定（“没有问题”是肯定）。
4. **`_substantive_tail()` 会把肯定词从尾部剥掉**（“没事了，好的，谢谢”仍算挂断），
   但只处理尾部，不影响正文判定；尾部剥完还有字就不挂断，宁可多问一句。
   这个规则偏保守：出现漏挂断（对方说完“再见”还在等）时，只需把更多“无意义词”加进
   `AFFIRMATIVE_KEYWORDS`/`POLITENESS_PHRASES` 的剥离表。

**未决**

- 判定表仍是规则集合，“好”“行”“是”这类单字肯定有误判风险（例如“行不通”含“行”）。
  已用“否定先于肯定”挡住主要情况，剩下的等 T12 场景集统计误判率再调。
- `max_parse_failures` 目前是模块常量（评审只要求新增 `max_confirm_rounds` 到 config）；
  若 T12 显示需要按链路调，再进 `config.yaml`。

---

## T06 预合成话术缓存

**完成情况**：完成。`core/phrases.py`（话术表、`PhraseCache`、`render_phone`）、
`tools/prerender_phrases.py`（预合成工具）、`tests/test_phrases.py`（27 个用例，含 fake TTS）；
全量 `pytest` 293 passed。真实本地 TTS 生成全部 21 条缓存成功，号码复述经 ASR 复验读法正确。

**文件**

- `core/phrases.py`：`PHRASES`（9 条话术，`{owner}` 占位）、`STATE_PHRASES`（每个状态至少一条）、
  数字读法 `DIGIT_READINGS`、前后缀 `PHONE_AFFIX_PHRASES`、`PhraseCache`
  （`get` / `put` / `render_phone` / `path`）、`phone_groups`、`trim_silence`、`silence`。
- `tools/prerender_phrases.py`：`--backend` / `--voice-id` / `--owner` / `--cache-dir` /
  `--force` / `--phone` / `--phone-wav`。可用 `--phone 13800138000 --phone-wav /tmp/x.wav` 试听。
- `tests/test_phrases.py`：27 个用例（话术表完整性、owner 填充、音色键解析、缓存读写与原子写、
  首次生成／重复跳过／文本变更重生成／失败不中断、号码分组与拼接顺序与间隔、静音裁剪、wav 导出）。
- `config.yaml`：新增 `tts.phrase_cache`（`dir`、`trim_threshold`、`trim_frame_ms`、`digit_gap_ms`、`group_gap_ms`）。

**设计与取舍**

1. **缓存键与目录**：`cache/tts/<backend>/<sha1(文本 + 音色标识)>.pcm`。`backend` 取
   `config.yaml: tts.backend`，音色标识按优先级解析：工具参数 `--voice-id` > 客户端 `voice_id`
   （T05b）> `tts.local_voice_id`（预留的手动固定值）> 占位常量。云端用 `tts.voice`。
   换了参考音频或音色，哈希变化，缓存自动失效（占位值不会自动失效，见"未决"）。
2. **本地 `voice_id` 的来源**：T05b 尚未合入 `main`，`LocalTtsClient` 还没有 `voice_id` 属性，
   因此工具在本地区后端下直接 GET `tts.local_url/health` 取 `voice_id`（当前 `ba97985b9879`），
   并打印来源。这样缓存键与 T05b 合入后客户端提供的值一致，不需要重建缓存。
   探测只写在工具里，`core/phrases.py` 不发 HTTP（实时链路将来由客户端属性提供，两者都不碰）。
3. **号码拼接**：前缀 + 逐位数字 + 后缀；手机号 3-4-4，带区号座机按 区号-号码（010/02x 为 3 位，
   03xx–09xx 为 4 位），其它按每 4 位一组。组间 150ms、位间 40ms 静音。**前缀与第一位、
   最后一位与后缀之间不额外插静音**（任务书只规定组间与位间）；拼接边界无跳变（见下）。
4. **静音裁剪**：按 10ms 帧算 RMS，低于 `trim_threshold`（默认 200，约 −44dBFS）的帧从首尾去掉。
   TTS 的"静音"有底噪，不能用全零判定。只在 `render_phone()` 里裁，`get()` 返回缓存原样。
5. **失败不中断**：`prerender()` 单条失败只记 `failed` 并继续，退出码 1；成功过的条目下次运行
   直接跳过，所以重跑即补齐。
6. **写文件用临时文件 + `replace`**：实时链路不会读到写了一半的 `.pcm`。

**验收结果**

```
$ uv run pytest tests/test_phrases.py -q
    27 passed
$ uv run pytest -q
    293 passed, 12 deselected
$ uv run python tools/prerender_phrases.py            # 第二次运行
    0 generated, 21 skipped, 0 failed in 0.0s
$ uv run python tools/prerender_phrases.py --phone 13800138000 --phone-wav /tmp/phone_t06.wav
    21 generated ... in 123.1s
    phone demo: 13800138000 -> /tmp/phone_t06.wav (7.50s)
```

**实测数据（本地 CosyVoice3，2026-10-21，voice_id `ba97985b9879`）**

| 条目 | 音频时长 | 条目 | 音频时长 |
|---|---|---|---|
| greet | 6.28s | ask_callback | 3.36s |
| ask_name | 1.68s | ask_time | 2.44s |
| ask_purpose | 2.40s | repeat | 3.20s |
| please_wait_confirm | 2.80s | forward | 2.64s |
| goodbye | 2.64s | 前缀／后缀 | 1.31s／0.28s（裁剪后） |

- 数字片段裁剪后：零 610、幺 560、二 170、三 230、四 200、五 200、六 620、七 200、八 330、九 300（ms）。
- 21 条预合成总耗时 123.1s（单条 3.1–8.6s）。这是离线一次性开销；注意当时机器上另有进程在用 GPU。
- `render_phone("13800138000")` = 7.50s。其中数字本身约 5.3s（平均 0.48s/位），
  间隔（7×40ms + 2×150ms = 0.58s）只占 8%。
- 客观检查：峰值 27672（未削顶），组间间隔 150/155ms、位间 40/45ms（与配置一致），
  拼接边界处样本从 0 平滑起音（起点幅值 2–307/32768），相邻样本最大跳变出现在语音内部，
  说明裁剪边界没有产生咔哒声。
- **读法复验**：把 `render_phone` 的输出交给 T03 的 ASR 转写，得到
  `您的回拨电话是幺三八零零幺三八零零零，对吗？`（latency 782ms）——11 位数字顺序、数量与前后缀都对，
  没有漏读或重读。

**遇到的问题**

1. **首次预合成 21 条全部"All connection attempts failed"**：本地 TTS 服务在我探测 `/health` 成功之后
   被另一个会话重启（正在做 T05b），整个 `--force` 前的那次运行期间端口没有监听。等 `/health` 恢复后
   重跑成功。真实链路里这类失败由 T10 的兜底话术处理；工具本身按"失败不中断 + 下次跳过已成功项"设计，
   GPU 服务重启是常态，不需要手工清理缓存。
2. **数字片段时长不均匀**（二 170ms vs 零/六 610ms）：单字文本合成的语速不稳定，40ms 位间静音在
   0.6s 的字面前显得很紧。读法本身正确（ASR 复验过），但 11 位号码念完要 7.5s，偏慢；
   真人复述一般 3–4s。留给 T12 听感评估时决定是否对数字片段做时间压缩或换更短的参考音色。
3. **占位音色键的风险**：如果本地服务连不上、`/health` 也没有 `voice_id`，缓存键会退化成
   `local-reference-unknown`，此后换参考音频不会自动失效（工具会打印 warn）。T05b 合入后客户端
   自带 `voice_id`，这条路径只在服务未启动时才会走到。
4. **未真实"人工试听"**：coding agent 不能主观听音，上面用波形客观指标 + ASR 复验代替；
   `/tmp/phone_t06.wav` 保留供评审方人工试听，若听感有问题（例如数字之间太赶、单字音高飘）
   请在评审意见里指出，我再调位间静音或改用整句合成方案。

**给 T07／T10 的接口**

```python
from core.phrases import PhraseCache, phrase_text, PHONE_PREFIX, PHONE_SUFFIX
cache = PhraseCache.from_config(config, tts=tts)      # tts 用于取 voice_id（本地）
pcm = cache.get(phrase_text("ask_callback", config["owner"]["name"]))   # 命中返回 bytes，否则 None
pcm = cache.render_phone("13800138000")               # 缺片段抛 PhraseMissing（先跑 prerender）
groups = phone_groups(number)                          # 3-4-4 / 区号-号码
```

---

## T06b 话术表合并与号码复述提速（第 10 次评审）★

**完成情况**：完成。合并最新 `main`（T07），`PHRASES` 只保留 `core/phrases.py` 一份、`core/dialog.py`
改为导入（`tests/test_dialog.py` 一行未改仍通过）；数字片段改为整句合成后按停顿切分；
`render_phone("13800138000")` 从 7.50s 降到 **5.21s**（目标 ≤ 4.5s 未达到，原因见"问题与偏差"）。
全量 `pytest` 348 passed。

**文件**

- `core/phrases.py`：话术表文本改为 T07 的版本（`ask_name`／`ask_purpose`／`ask_callback`／`forward`），
  `ask_time`／`please_wait_confirm` 保留；`confirm_phone_prefix`／`confirm_phone_suffix` 只在 `PHRASES`
  里定义一次，`PHONE_AFFIX_PHRASES` 由它派生（T07 的 `PHRASES[...]` 与 T06 的 `PHONE_AFFIX_PHRASES`
  两种访问方式都可用）；新增 `DIGIT_ORDER`、`DIGIT_SENTENCE`、`split_digit_clips()`、
  `PhraseSplitError`、`phrase_texts()`、`digit_texts()`；缓存键加版本号与分隔符；
  `render_phone()` 在前后缀两侧也插 `group_gap_ms`。
- `core/dialog.py`：删掉自己的 `PHRASES`，改为 `from core.phrases import PHRASES`（只有这一处改动 + 注释）。
- `tools/prerender_phrases.py`：`prerender_digits()`（整句合成 → 切分 → 写 10 个片段）、
  `--digit-takes`（默认 3，取语速最快的一版）、`--verify`（用 ASR 逐段复验，见下）。
- `tests/test_phrases.py`：43 个用例（原 27 个 + T06b 新增 16 个）。

**关键实现选择**

1. **单表迁移**：`core/dialog.py` 只把 `PHRASES` 换成 import，其它逻辑一行未动；
   `tests/test_dialog.py` 未改动、全部通过。一致性用例做两件事：
   （a）`core.dialog.PHRASES is core.phrases.PHRASES`（只保留一份）；
   （b）把 `Dialog` 可能返回的每个 id（`ASK_TEMPLATES` 的值 + `greet`/`repeat`/`goodbye`/`forward`，
   `confirm_phone` 除外）与 `cache_texts()` 逐条比对文本。以后 T07b 改话术表，缓存侧会自动跟上。
2. **整句合成再切分**：`DIGIT_SENTENCE = "幺，二，三，四，五，六，七，八，九，零。"`，一次合成，
   按 10ms 帧 RMS 找发声段，间隔 < 80ms 视为字内凹陷（辅音、爆破）不切。
   切分后段数必须是 10，否则抛 `PhraseSplitError`，工具报错退出、**不退回单字合成**。
3. **句尾呼吸音**：实测约 1/6 的概率在句尾多出一小段 50ms、峰值 RMS 1365 的尾音（正常数字段
   110–370ms、峰值 RMS 2000 以上）。规则是**又短（≤80ms）又弱（< 中位峰值 25%）**才忽略；
   又长又弱的段（如轻声的"二"）仍然保留，多出响亮的段仍然报错。
4. **语速抖动**：同一句话连合成 6 次，10 个数字合计 1.88–2.82s（差 50%）。因此工具默认
   `--digit-takes 3`，取切分成功且总时长最短的一版（离线一次性开销，约多 2 次合成）。
5. **前后缀两侧补静音**：`前缀 → 150ms → 数字 → 150ms → 后缀`，避免"……是幺三八"和
   "零零零，对吗"黏在一起。
6. **缓存键**：`sha1(版本 + \x1f + 文本 + \x1f + 音色)`。加版本号让 T06 的旧缓存（单字片段、
   旧话术文本）自动失效；加分隔符解决第 10 次评审指出的 `("ab","c")` / `("a","bc")` 撞键问题。
7. **没做 `digit_max_ms` 截断**（第 4 条备选方案）：整句切分后最长片段 290ms，`0`/`5`/`1`
   分别是 290/270/230ms，全部低于 320ms，截断会白掉字尾，因此不启用。

**验收结果**

```
$ git merge origin/main            # 合并 T07，docs/progress.md 冲突按两边都保留
$ uv run pytest tests/test_phrases.py -q     -> 43 passed
$ uv run pytest -q                           -> 348 passed, 12 deselected
$ uv run python tools/prerender_phrases.py --config /tmp/t06b_config.yaml \
      --digit-takes 3 --phone 13800138000 --phone-wav /tmp/phone_t06b.wav --verify
  21 generated, 0 skipped, 0 failed in 63.2s
  phone demo: 13800138000 -> /tmp/phone_t06b.wav (5.21s)
```

**实测数据（2026-10-22，本地 CosyVoice3，voice_id `ba97985b9879`）**

数字片段（`take 0`，10 段合计 2020ms；T06 的单字合成是 170–620ms、合计 3570ms）：

| 数字 | 片段 | 数字 | 片段 | 数字 | 片段 |
|---|---|---|---|---|---|
| 0 | 290ms | 4 | 140ms | 8 | 150ms |
| 1 | 230ms | 5 | 270ms | 9 | 190ms |
| 2 | 180ms | 6 | 180ms | | |
| 3 | 180ms | 7 | 210ms | | |

`render_phone` 总时长与构成：

| 号码 | 总时长 | 说明 |
|---|---|---|
| 13800138000（11 位手机） | **5.21s** | 数字 2570ms + 间隔 880ms + 前缀 1330ms + 后缀 390ms |
| 13912345678（11 位手机） | 4.78s | 数字少一个"零" |
| 01012345678（带区号座机） | 4.88s | 区号-号码两段 |
| 82345678（8 位座机） | 3.87s | |
| 133（3 位） | 2.69s | 前缀 1.33s + 后缀 0.39s 占大头 |

与 T06 对比：11 位手机号 **7.50s → 5.21s（−31%）**，数字片段平均 0.48s/位 → 0.23s/位。
实测语速（约 0.19s/字）下，前缀"您的回拨电话是"（7 字）本身就占 1.33s，加后缀 0.39s 共 1.72s
（33%）；间隔 0.88s（17%）是 T06 任务书规定的 40/150ms；数字本身 2.57s（49%）。

**ASR 复验**

- **整段号码**：`您的回拨电话是幺三八零零幺三八零零零，对吗？` —— 11 位顺序、数量、前后缀全对。
- **逐段复验**（`--verify`，每段单独送 ASR）：1 幺、2 二、4 四、5 五、6 六、7 七、0 零 全部正确；
  3 被识别成"在"、8 被识别成"吧"、9 被识别成"叫"。这三段只有 150–190ms，孤立音节没有上下文，
  ASR（Qwen3-ASR）会按同音字猜；整句复验证明读法本身没问题。逐段复验的局限记在这里，
  不把它当成"读音错误"。
- 试听文件：`/tmp/phone_t06b.wav`（5.21s）。agent 不能主观听音，客观检查：峰值 27672 未削顶，
  拼接边界样本从 0 平滑起音，组间 150ms、位间 40ms 与配置一致。

**问题与偏差**

1. **目标 ≤ 4.5s 未达到（5.21s）**。构成见上表：前缀+后缀 1.72s、间隔 0.88s、数字 2.57s。
   要压到 4.5s 需要再省 0.7s，可选（未做，留给评审决定）：
   ① 前缀也对同样的话术做 best-of-N 取语速更快的一版（预计省 0.2–0.3s，但话术缓存会随音色变化重做）；
   ② 把前缀改成更短的说法（如"回拨电话是"，按 0.19s/字约省 0.38s）；
   ③ 换语速更快的参考音色（T05b 的参考音频试验可以一起评估）。
   保持现状的理由：数字部分已达"真人 0.25–0.3s/位"的目标，剩余时间在固定话术和任务书规定的间隔上。
2. **逐段 ASR 复验对孤立音节不可靠**（见上）。整段号码复验是有效验收项。
3. **本机 50000 端口的本地 TTS 服务在测量时卡死**（`/health` 正常、`POST /synth` 返回 200 但 0 字节，
   挂到 120s 超时）。该服务是并行的 T05b 会话启的，它另起了 50001 端口（同模型、同 `voice_id`）在做
   vLLM/bench 试验。本次预合成与测量都用 50001 完成（临时 config 副本，未改 `config.yaml` 的
   `tts.local_url`）。T10 前需要确认 50000 恢复正常，或让 T05b 把正式端口固定下来。
4. **没做 `digit_max_ms` 截断**，原因见"关键实现选择"第 7 条。
5. **T07b 尚未合入**：本次按当前 `main`（T07）的话术文本对齐。T07b 只改 CONFIRM/结束意图的判定
   与提示词，不新增话术 id（否定时交给 LLM 生成），因此一致性用例在 T07b 合入后仍然适用。

**给 T07／T10 的接口（T06b 后）**

```python
from core.phrases import PhraseCache, phrase_text, PHONE_PREFIX, PHONE_SUFFIX, DIGIT_SENTENCE
cache = PhraseCache.from_config(config, tts=tts)      # 本地后端用 tts.voice_id 做音色键
pcm = cache.get(phrase_text("ask_callback", config["owner"]["name"]))
pcm = cache.render_phone("13800138000")               # 5.21s；缺片段抛 PhraseMissing
groups = phone_groups(number)                          # 3-4-4 / 区号-号码
```

### T06b 第 12 次评审修订（循环导入 + 补话术）

**完成情况**：完成。合并 `main`（T07b）后按评审意见修掉循环导入，补了 `confirm_question`、
`ask_which_wrong` 两句话术并重新预合成。全量 `pytest` 412 passed；
`python -c "import core.dialog, core.phrases, core.prompts"` 通过。

**改了什么**

1. **循环导入（阻塞项）**：原先是 `dialog → phrases → prompts → dialog`。`core/phrases.py` 不再
   导入 `core.prompts`，`DEFAULT_OWNER_NAME = "机主"` 在本模块自己定义；模块可以依赖的范围
   限定为 `core.types`、`core.normalize`、标准库与 numpy（`phone_groups` 按修订后的任务书
   暂留在本模块，由 T07c 再移到 `core.normalize`）。
2. **合并 T07b**：`core/dialog.py` 的冲突按"以 `main` 的 T07b 逻辑为准、只把 `PHRASES`
   换成 import"解决——文件其余部分（`classify_confirm_reply`、结束意图判定、
   `max_confirm_rounds`、连续解析失败改播模板）原样保留；`tests/test_dialog.py` 未改动。
3. **补话术**：`confirm_question`（"对吗？"）、`ask_which_wrong`（"请问是哪一项不对？"），
   供 T07c 组整句确认语与"哪一项不对"的追问用；`cache_texts()` 自动包含它们。
4. **数字拼接停止优化**：按第 11 次评审，号码复述主方案改为 T07c 的整句现合成，
   `render_phone` 降级为兜底路径（代码、数据保留，5.21s）；4.5s 的目标不再要求。
   `render_phone` 与模块 docstring 都写明了这一点。
5. **新增测试**：
   - `test_core_modules_import_without_cycle`：子进程执行 `import core.dialog, core.phrases, core.prompts`，
     在修复前会失败（原复现输出就是评审给的那条 `ImportError: cannot import name 'NEGATIVE'`）；
   - `test_phrases_module_does_not_import_prompts_or_dialog`：AST 检查 `phrases.py` 的
     `core.*` 依赖只能是 `core.types` / `core.normalize`；
   - `test_t06b_added_phrases_for_t07c`：两句话术的文本与缓存条目。

**验收结果**

```
$ uv run pytest -q                     -> 412 passed, 12 deselected
$ python -c "import core.dialog, core.phrases, core.prompts" -> imports ok
$ uv run python tools/prerender_phrases.py --config /tmp/t06b_config.yaml \
      --phone 13800138000 --phone-wav /tmp/phone_t06b.wav --verify
  synth confirm_question         对吗？ (1720ms, 音频 560ms)
  synth ask_which_wrong          请问是哪一项不对？ (3600ms, 音频 2360ms)
  2 generated, 21 skipped, 0 failed in 5.3s
  phone demo: 13800138000 -> /tmp/phone_t06b.wav (5.21s)
  verify phone 13800138000 -> '您的回拨电话是幺三八零零幺三八零零零，对吗？'
```

- 话术文本没变，只有新加的两条没命中缓存（`2 generated, 21 skipped`），未用 `--force`；
  缓存键含版本号与分隔符，T06 的旧文件全部失效。
- `render_phone("13800138000")` 仍是 5.21s（数字片段没有重新合成），`/tmp/phone_t06b.wav` 已更新。
- 本地 TTS 的 50000 端口在本次运行前**仍然卡死**（`/health` 200、`POST /synth` 返 200 但 0 字节），
  本次仍旧用并行的 T05b 会话起的 50001 端口完成（临时 config 副本，`config.yaml` 未改）。
  该现象已由第 12 次评审转给 T05b 排查（疑似 `SYNTH_LOCK` 未释放）。

**待 T07c 接手的接口**

```python
from core.phrases import PHRASES, phrase_text, phone_groups
phrase_text("confirm_question", owner)      # "对吗？"
phrase_text("ask_which_wrong", owner)       # "请问是哪一项不对？"
phone_groups(number)                        # 仍在 core/phrases.py，T07c 移到 core/normalize.py
render_phone(number)                        # 兜底路径，主方案是整句现合成
```

---

## T05b 本地 TTS 首包修订（第 8 次评审）★

**完成情况**：hop 复位、离线测试、voice_id、bench 扩展与重测已完成；打断后重说、GPU 占用、音频检查已记录。**vLLM 尝试未完成**（安装被中断），如实记录。首包仍未达到 0.8s 目标。

**产物**

- `local_tts/server.py`：
  - 加载时从 `engine.model.token_hop_len` 读出初始值保存为 `engine.initial_token_hop_len`，每次请求持锁后复位（不硬编码 25），修复上游实例属性在请求间不复位的问题。
  - 参考音色日志打印 token 数与 pad；`/health` 增加 `voice_id`、`initial_token_hop_len`、`prompt_token_count`、`prompt_token_pad`、`vllm`。
  - 新增 `compute_voice_id()`：参考音频内容 + 提示文本的 sha1 前 12 位。
  - 新增 `--vllm` 开关（`AutoModel(..., load_vllm=True)`），未验证。
  - 客户端断开时把当前句剩余部分跑完再释放锁，避免上游 LLM 线程与下一请求并发。
- `local_tts/bench.py`：输出每句块数、首块音频时长；新增“打断后重说”5 次；GPU 采样增加利用率；`--wav-dir` 保存每句 wav。
- `services/tts_local.py`：新增只读 `voice_id` 与 `async fetch_voice_id()`；首次 `synth()` 前自动从 `/health` 获取。
- `tests/test_local_tts_server.py`（新增，6 个离线用例）；`tests/test_tts_local.py` 增至 14 个用例。
- `local_tts/README.md`：补充 hop 修复、上游 commit、T05b 数据、打断数据、GPU、音频检查、vLLM 失败原因。

**上游版本**

- 官方仓库 commit：`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`。
- T05b 依赖该 commit 的 `CosyVoice2Model.tts()` 行为：把 `self.token_hop_len` 从 25 逐块翻倍到 100 且请求间不复位。升级官方仓库后必须重新验证。

**接口/配置变更记录（AGENTS.md 规则 2）**

- `LocalTtsClient` 新增只读属性 `voice_id`（取到前为 `None`）和 `fetch_voice_id()`；`synth()` 语义不变，只是首次调用前多一次 `/health`。
- 本地服务 `/health` 新增字段：`voice_id`、`initial_token_hop_len`、`prompt_token_count`、`prompt_token_pad`、`vllm`。
- 新增服务启动参数 `--vllm`。`config.yaml` 与 `core/types.py` 未改。

**验收结果**

```
$ uv run pytest -q
    274 passed, 12 deselected, 1 failed
    # 唯一失败：tests/test_turn.py::test_real_models_on_sample_wav
    # ModuleNotFoundError: No module named 'pipecat'（本 worktree 的 .venv 没装 pipecat，
    # 属合并 main 后的环境差异，与 T05b 无关；main 评审环境为 266 passed）

$ uv run pytest tests/test_local_tts_server.py tests/test_tts_local.py -q
    20 passed
```

离线用例覆盖评审要求：连续两次 `iter_pcm` 第二次从初始 hop 开始；产出一块后 `close()` 锁已释放；空文本 400；`voice_id` 只在首次 synth 前取一次；缺 `voice_id` 报错。

**实测数据（`bench.py --n 5`，专用端口 50001，2026-10-22 09:38，fp16）**

| 句长（字） | T05 首包 P50 | T05b 首包 P50 / P95 | T05b 总耗时 P50 / P95 | 块数 | 首块音频 | 音频 P50 | RTF P50 |
|---|---|---|---|---|---|---|---|
| 3 | 1098 | 1485 / 1621 | 1507 / 1649 | 3 | 452 | 920 | 1.64 |
| 7 | 1917 | 2075 / 2452 | 2615 / 2900 | 6 | 452 | 1880 | 1.39 |
| 18 | 3500 | 2300 / 2931 | 4851 / 5560 | 8 | 452 | 4040 | 1.20 |
| 25 | 4513 | 2301 / 3337 | 6874 / 7188 | 10 | 452 | 5720 | 1.20 |
| 49 | 5010 | 2798 / 3188 | 10775 / 11549 | 16 | 452 | 10200 | 1.06 |
| **总体** | 3500 / 5278 | **2281 / 3220**（min 1094 / max 3574） | **4851 / 11133** | — | — | — | — |

- 首包 P50 下降 1219ms（35%），P95 下降 2058ms；18/25/49 字分别下降 1200/2212/2212ms。**hop 复位确实有效，T05 把主因归为“模型限制”是不准确的。**
- 短句反而略慢（3 字 +387ms、7 字 +158ms）：修复前 hop 已被上一句抬到 100，短句只出 1 块；修复后短句分成 3/6 块，每块 flow/hift 有固定开销。
- 总耗时上升 35–40%，RTF 1.06–1.64（T05 为 0.95–1.33）。
- 任务预计“18 字以上首包约 1.7s”未达到（实测 2.3s）；**仍未达到短句首包 P50 ≤ 0.8s**（总体 2281ms）。

**打断后重说（5 次，49 字 → cancel → 7 字）**

```
10830ms  10974ms  11633ms  9929ms  10917ms
P50 10917ms  P95 11501ms；正常 7 字首包 P50 2075ms
```

- 比正常值多约 **8.8s**。原因：客户端 cancel 只关闭连接；服务端为防并发会把当前 49 字句跑完（约 10.8s）才放锁，下一句才能开始。
- 开发中复现了更严重的问题：若 cancel 后立刻放锁，下一请求的 LLM 与上一请求残留的 LLM 线程并发使用同一模型，会抛 `probability tensor contains either inf, nan or element < 0`，随后永久卡住（两次）。因此当前实现选择“跑完再放锁”，稳定但打断响应不可用。
- **T10 必须实现真正的服务端取消/排队**，不能依赖当前本地服务。

**GPU 占用排查**

- 压测时 `nvidia-smi --query-compute-apps` 只有本机 python 服务进程，无其它计算进程。T05 记录的 baseline 中额外约 4.4GB 是当时另一个已加载的模型进程，T05b 时已不存在。
- 单服务 baseline 约 8227 MiB（无服务时约 1819 MiB；模型+CUDA 上下文约 6.4GB）。
- 最终压测同时加载了两个服务（50000 被 T06 使用、50001 为压测专用）：baseline 14683 MiB、peak 15072 MiB、增量 389 MiB；GPU 利用率 max 100%、avg 34%。
- 利用率 avg 只有 34%，说明不是持续满载；LLM 逐 token 生成与每块 flow/hift 之间有串行空档，RTF 仍约 1.1–1.6。

**音频检查（替代人工试听）**

wav 存在 `/tmp/t05b_wavs/`（不入库）。5 句时长 0.88/1.88/3.36/5.12/9.88s，峰值 0.60–0.99，削波 0%，最大样本跳变 0.406–0.664（正常语音瞬态范围）。
ASR 回译（Qwen3-ASR）与输入逐句对照：无丢字、无重复；只有标点差异和一处逗号插入。
**块边界是否有轻微咔哒声无法用波形判定，本次无人耳试听条件，未做听感确认。**

**vLLM 尝试（未完成）**

- 已 `conda create -n cosyvoice_vllm --clone cosyvoice`，然后 `pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4`。
- 安装需连带升级 torch 2.8 和大量 CUDA 库，下载量很大（磁盘占用 99GB→141GB），运行数分钟后被中断，`vllm` 未装成，**没有 vLLM 数据**。
- 服务端 `--vllm` 已加；环境装好后用 `python local_tts/server.py --vllm --port 50002` + `bench.py --url ...` 即可复测。
- TensorRT（flow）按评审意见排在 vLLM 之后，未做。

**遇到的问题与处理**

1. **T06 同时在用 50000 端口**，第一次 T05b bench 的请求与 T06 的请求互相排队，数据被污染（同一时段出现了本任务没有的句子）。处理：另起一个专用服务在 50001 重测，最终数据来自 50001。
2. **cancel 后并发导致 NaN/卡死**：见上。处理：`iter_pcm` 捕获 `GeneratorExit` 后把当前句跑完（不再 yield）再放锁；bench 的打断用例超时从 10s 放宽到 30s，才拿到 9–11s 的真实数值。
3. **压测波动大**：同机不同时段 3 字首包出现过 1.3s 和 4.4s（后者有外部请求干扰/后台负载）。最终数据取隔离端口的一轮，并在 README 注明环境。
4. **全量测试 1 个失败**：`pipecat` 未装（合并 main 后 T09 的测试），与 T05b 无关。T05b 相关 20 个用例全部通过。

**未决**

- 0.8s 目标仍未达成。hop 复位后单块门槛约 38 token（首块 452ms 音频），但 4060 Ti 上每块的 flow/hift 固定开销使首包 1.5–2.8s、RTF 1.1–1.6。
- 打断后重说 10.9s，T10 必须解决；可选方向：服务端可取消推理、请求排队与丢弃过期请求、把上一句生成与下一句隔离到不同 CUDA stream/进程。
- vLLM 未验证；TensorRT/JIT 未尝试。
- 参考音频 pad=13（87 token）。按评审建议可裁静音/换音频使 pad=0，首块门槛可降到 28 token（约 1.1s 音频），本次未做；换音色会改变 `voice_id`，T06 缓存会自动失效。

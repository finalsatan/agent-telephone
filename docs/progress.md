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

# T15 / T15b：Pipecat 适配笔记

记录日期：2026-10-26（T15b 同日修订）
Pipecat 版本：`pipecat-ai==1.13.0`（`requirements.txt` 已固定）
代码：`adapters/pipecat/{stt,tts,llm,pipeline,run}.py`，测试 `tests/test_pipecat.py`、
`tests/test_playback.py`，live 测试 `tests/live/test_pipecat_live.py`。

## 1. 接进来的形态

```
浏览器 WS /ws/call（与 T13 同一协议）
   │ 16kHz PCM 二进制帧 / 控制 JSON
   ▼
PipecatCall.feed()  → PipelineWorker.queue_frame(InputAudioRawFrame)
   ▼
VADProcessor(SileroVADAnalyzer)           ← config.yaml: vad.threshold / speech_end_silence_ms
   ▼                                         start_secs = interrupt.min_speech_ms（T15b）
SiliconFlowSTTService (SegmentedSTTService) → services.asr.AsrClient
   ▼
UserTurnProcessor(TurnAnalyzerUserTurnStopStrategy(ConfiguredSmartTurn))
   ▼
TurnEventLogger                            ← 把 vad_start / vad_end / turn_end 写成 T10 事件
   ▼
BargeInMonitor                             ← 抢话：用播放队列模型记 interrupted(tail=True)
   ▼
DialogProcessor                            ← core.dialog.Dialog + services.llm.LlmClient
   ▼
SiliconFlowTTSService (TTSService)         ← core.phrases.PhraseCache + services.tts[_local]
   ▼
WsAudioSink → 浏览器二进制帧
```

- 启动：`uv run python -m adapters.pipecat.run`（默认 8001 端口，避免和 T13 的 8000 冲突）；
  也可以把 `config.yaml: web.adapter` 改成 `pipecat` 后用 `uv run python -m web.server` 挂在 8000。
- 与 T13 的对接方式：`web/server.py` 的 `AppState.start_call()` 按 `web.adapter` 选择
  `RealtimeCall`（cascade）或 `PipecatCall`（pipecat），两者对外接口相同
  （`start` / `feed` / `takeover` / `finish` / `finished` / `events` / `services`），
  页面、HTTP 路由、事件转发、`slots.json` / 分轨 wav 全部复用。
- 参数不写死：VAD 阈值、静音确认时长、Smart Turn 阈值、`stop_secs`、`pre_speech_ms`、
  `max_segment_s` 全部从 `config.yaml: vad` 读；模型名、base_url 走 `services/`。
  T15b 起 `interrupt.min_speech_ms` 映射到 `VADParams.start_secs`。

## 2. 接入工作量

T15 实际投入约 **1.5 天**（在 2 天限时内）；T15b 修订另计 **约 1 天**（见第 9 节）。

| 阶段（T15） | 用时 |
|---|---|
| 读 Pipecat 1.13 源码（transport / STT / TTS / turn / worker） | 约 4h |
| 写 STT / TTS 适配与单测 | 约 3h |
| 写 DialogProcessor（LLM + 状态机 + 打断） | 约 3h |
| 写管线、worker 生命周期、T13 对接 | 约 3h |
| live 联调（顺序、转写竞态、打断重复事件、参数阈值） | 约 4h |
| 文档与数据整理 | 约 2h |

其中“踩坑—定位—修复”占了近一半时间，主要来自 1.13 相对旧文档的 API 变化（见第 4 节）。

## 3. 能直接复用 vs 必须自写

| 部分 | 结论 |
|---|---|
| ASR（`services/asr.py`） | **直接复用**。`SegmentedSTTService.run_stt()` 里一次 `transcribe()`，`wants_wav_segments=False` 拿裸 PCM。 |
| TTS（`services/tts.py` / `tts_local.py`） | **直接复用**。`TTSService.run_tts()` 里 `async for chunk in synth(text, cancel)`。 |
| 预合成话术缓存（`core/phrases.py`） | **直接复用**。在 `run_tts()` 里先 `cache.get(text)`，命中直接产出 PCM，首包 0。 |
| 对话状态机（`core/dialog.py`） | **直接复用**。但接不进 Pipecat 的 `LLMService`：本项目 LLM 输出固定 JSON、流程由代码决定，Pipecat 的 LLM 服务假设 OpenAI 式“上下文进、流式文本出”。所以写成自定义 `FrameProcessor`。 |
| 一轮的判断逻辑（`core/call.py`） | **T15b 起抽成共用函数**：`stream_llm_turn()`（LLM 流式 + 事件）、`choose_reply_text()`、`is_partial_phone()`、`template_hint()`；`CallSession` 与 `DialogProcessor` 都调它们，避免两条链路行为不一致。 |
| LLM（`services/llm.py`） | **直接复用** `turn_stream()`。 |
| 事件流（`core/events.py`） | **直接复用** `EventLogger`；复用 `web/server.py` 的 `ForwardingEventLogger`。 |
| 信道模拟（`tools/channel_sim.py`） | **直接复用**。在 `feed()` 里逐段 `ChannelSimulator.feed()`。 |
| 浏览器播放队列模型（`core/playback.py`） | **T15b 起抽成共用类** `PlaybackTracker`，T13b 的 `RealtimeCall` 与 `PipecatCall` 共用，保证尾音抢话行为一致。 |
| VAD / Smart Turn | **用框架的**：`SileroVADAnalyzer` + `LocalSmartTurnAnalyzerV3`（与 T09 的 `core.turn` 是同一批模型）。 |
| 轮次事件（`vad_start` / `vad_end` / `turn_end`） | **必须自写**：Pipecat 内部不写 T10 格式的事件，加 `TurnEventLogger` 把框架帧翻成事件。 |
| 打断的 `[被打断]` 标记与 `interrupted` 事件 | **必须自写**：用 `PlaybackTracker` 判断浏览器里是否还在播，`BargeInMonitor` 记事件并标历史。 |
| 离线回放（`tools/replay.py` / `evaluate.py`） | **框架不便回放**：Pipecat 面向实时流，没有“把一条 wav 按轮喂进去”的现成入口。见第 6 节。 |

## 4. 踩到的坑（按影响排序）

1. **1.13 的 API 与旧文档/示例不一致**。`PipelineTask` 已弃用，改为 `PipelineWorker` +
   `WorkerRunner`（`runner.add_workers(worker)` → `runner.run(auto_end=False)`）；VAD 不再挂在
   transport 参数上，而是独立的 `VADProcessor`；轮次判断从旧的 `TurnAnalyzer` 参数改为
   `UserTurnProcessor` + `UserTurnStrategies`。任务书里提到的 `SegmentedSTTService` /
   `TTSService` / `LocalSmartTurnAnalyzerV3` 都还在，但周边接线全变了。
2. **`UserTurnProcessor` 必须放在 STT 之后**。`TurnAnalyzerUserTurnStopStrategy` 默认
   `wait_for_transcript=True`，要等一个 `finalized=True` 的 `TranscriptionFrame` 才发
   `UserStoppedSpeakingFrame`。一开始放在 STT 之前，它收不到转写，只能等 STT 超时兜底，
   每轮凭空多 4–5s。改成 `vad → stt → turn`（与官方级联模板 `input → stt → user_aggregator`
   一致）后恢复正常。代价是 STT 要保留 `audio_passthrough=True`，把音频继续交给 turn 跑
   Smart Turn。
3. **转写帧比轮次结束帧晚到**。`SegmentedSTTService` 在后台任务里转写，实测 `asr_final`
   可能比 `UserStoppedSpeakingFrame` 晚几十毫秒。最初的 `_schedule_turn()` 在轮次结束时
   发现 `_user_text` 为空就直接返回，导致这一轮被吞、文本顺延到下一轮才处理。加了
   `_turn_pending` 标志：轮次结束时置位，转写帧到达时若还没起 LLM 任务就补起。
4. **`LocalSmartTurnAnalyzerV3` 没有概率阈值**。它的 `_process_speech_segment()` 只看模型的
   布尔预测（内部按 0.5 判），而 T13c 把阈值调到 0.85 才把误切率压到 10% 以内。
   用 `ConfiguredSmartTurn` 子类覆写 `_predict_endpoint()`，按 `config.yaml` 的
   `vad.smart_turn_threshold` 再判一次（与 `core.turn.SmartTurn` 同一做法）。
5. **打断事件写重**。`InterruptionFrame` 由 `UserTurnProcessor` 在用户开口时广播：TTS 会
   取消 `run_tts()`，`DialogProcessor` 也会处理。两边都写 `interrupted` 会出现两条。
   T15b 起统一由 `PipecatCall` 的 `BargeInMonitor` 写（它能判断“在播 / 已播完”并给历史
   打标记），`DialogProcessor` 只在 LLM 阶段（还没播）写 `interrupted(stage="llm")`。
6. **`PipelineWorker` 的 `StartFrame` 由框架发**。问候语挂在 `DialogProcessor` 的
   `StartFrame` 处理里，`PipecatCall.start()` 要等 `on_pipeline_started` 事件后再让
   调用方开始喂音频。
7. **`ttfs_p99_latency` 未设置**。启动时会 warning，`TurnAnalyzerUserTurnStopStrategy`
   用它做 STT 兜底超时。当前 ASR P95 约 0.9s，默认 1.0s 够用；要更准可以在
   `SiliconFlowSTTService` 上暴露 `STTMetadataFrame`。
8. **`STTSettings` / `TTSSettings` 的 `language` 未初始化**，启动时有一条 ERROR 日志
   （不影响运行）。显式填 `language=None` 可消掉。

## 5. 打断与行为对齐（T15b，第 26 次评审）

T15 的框架接线可用，但打断与一轮处理有缺陷；T15b 逐条修：

### 5.1 浏览器还在播时抢话要有效

T15 直接把 PCM 推给浏览器，没有按实时节奏发送：2 秒的缓存话术 0.3ms 就推完并回调
“播完”，之后抢话不产生 `interrupted`，浏览器也不清队列。

**选用的方案**：复用 T13b 的排队模型（`core/playback.PlaybackTracker`），而不是按实时
节奏发送。理由：

- 按实时节奏发送要接管 Pipecat 的输出 transport 时钟，改动大、容易和框架的
  `TTSStoppedFrame` / 打断取消语义打架；
- T13b 的排队模型已经过 T14b 的浏览器实测验证，两条链路共用同一实现，行为最容易对齐；
- 服务端只需在 `play_start` 和每块音频送出时更新队列结束时刻，代价小。

`BargeInMonitor` 在 `InterruptionFrame` 到达时查 `PlaybackTracker.current_playing()`：
浏览器里还有没播完的音频就记 `interrupted(reason="barge_in", tail=True, played_ms=...,
unplayed=[...])` 并把历史最后一条 assistant 标“（被对方打断）”，与 T13b/T14b 一致。
LLM 阶段（还没播）由 `DialogProcessor` 记 `interrupted(stage="llm")`。

### 5.2 LLM 阶段被抢话不丢文本

T15 直接 `task.cancel()`，顺延文本的代码执行不到，来电者的前半句会丢。改为只置位取消
开关、不取消任务：`stream_llm_turn()` 在下一个 chunk 处退出，`_run_turn()` 跑到
“回滚状态机 + 把文本塞回 `_user_text` + 记 `interrupted(stage="llm")`”的分支。下一轮
开始前会先 `await` 上一轮任务（`_run_turn_after`），文本按顺序拼接，不丢也不重复追问。

### 5.3 打断阈值生效

T15 没用上 `interrupt_min_speech_ms`，Pipecat 默认 VAD 约 200ms 就判开口并打断。
T15b 把页面参数映射到 `VADParams.start_secs`：VAD 连续检测到语音达到该时长才发
`VADUserStartedSpeakingFrame`，短噪声不会触发打断。同时把默认的
`TranscriptionUserTurnStartStrategy` 去掉（它会在转写到达时再打断一次，绕过阈值），
`start` 只用 `VADUserTurnStartStrategy`。参数从页面 `interrupt_min_speech_ms` 一路传到
`build_vad_params()`，改一次只对下一通生效（与 T13 相同）。

### 5.4 一轮的处理与 `CallSession` 对齐

`core/call.py` 抽出 `stream_llm_turn()` / `choose_reply_text()` / `is_partial_phone()` /
`template_hint()`，`CallSession` 与 `DialogProcessor` 共用，行为一致：

- 号码没报完：实时链路没有音频时间轴，按“超时”处理（与 T13 的 `RealtimeCall` 一致），
  记 `error(incomplete_phone_timeout)`、不把半个号码写进座机、计一次追问失败；
- ASR 出错 / 转写为空：`SegmentedSTTService` 产出 `AsrFailedFrame`，`DialogProcessor`
  播 `repeat` 兜底话术（`CallSession._reply_with_fallback` 的同一分支）；
- CONFIRM 合成失败：`DialogProcessor` 推的 `SpeakHintFrame` 带 `phone`，`TTSService`
  一块音频都没播出时退回 `PhraseCache.render_phone()` 的号码片段拼接，并回调
  `on_tts_fallback` 改历史（事件与 `CallSession._play_text` 对齐）；
- `tts.confirm_lead`：`DialogProcessor` 把确认语拆成预合成的“好的，跟您确认一下。”和
  现合成的后半句，分别推帧。

### 5.5 事件补全

- `turn_end.reason`：`ConfiguredSmartTurn` 记录最后一次 COMPLETE 来自模型
  （`smart_turn`）还是静音超时（`stop_secs`），`TurnEventLogger` 写进事件；
- `call_end`、`takeover` 带 `turns`（`DialogProcessor.turn_count`）；
- 接管后来电者音频照常写进 `caller.wav`（`feed()` 在接管后仍累计 `caller_pcm`，只是
  不再进管线）；
- 页面时延面板：Pipecat 的 `asr_final` 在 `turn_end` 之前发出，前端 `buildTurnRows()`
  现在把 `turn_end` 之前的 ASR 标记带到这一轮，ASR 一栏不再为空。

## 6. 离线回放：用 T10 路径作基线

Pipecat 是实时流框架，没有“把一条场景 wav 按轮喂进去、按模拟时钟推进”的入口；
硬做需要自己写一套输入 transport 并接管它的时钟，性价比低。按任务书允许的替代方案：

- **时延/槽位基线仍用 T10 的离线路径**（`tools/replay.py` + `tools/evaluate.py`），
  这样两框架对比的是“同一批模型 + 同一批场景”，差异只来自编排。
- **Pipecat 路径用 T13 的 `tools/ws_client.py` 做实时回放**：它按 32ms 帧、实时速度把
  每轮 wav 发到 `WS /ws/call`，服务端换不换适配器对客户端透明。
- 事件流与 T10 **同格式**：`tools/evaluate.py:extract_turn_metrics()` 直接读 Pipecat 的
  `events.jsonl` 能得到每轮的 `e2e_ms / asr_ms / llm_ms / tts_first_ms / parse_ok`。
  唯一缺的是 `smart_turn` 的推理耗时事件（模型在框架内部线程里跑，拿不到），
  `evaluate.py` 不依赖它。

## 7. 实测（T15b，2026-10-26，本地 CosyVoice3 + 硅基流动 ASR/LLM）

T15 只跑了 3 条场景就下结论“与 T13c 量级一致”，这是不对的。T15b 用 `ws_client.py --all`
把 15 条场景（44 轮）在两条链路上各跑一遍，同一批模型、同一批 wav、同一份 `config.yaml`：

| 指标 | 级联（T13/T14b，`web.server` 8000） | Pipecat（T15b，`adapters.pipecat.run` 8001） |
|---|---|---|
| 回复数 | 45 | 见第 7.1 节 |
| 总等待 P50 / P95 | 见下 | 见下 |
| 轮次判断 P50 / P95 | 600 / 1952 ms | 见下 |
| ASR P50 / P95 | 446 / 1287 ms | 见下 |
| LLM P50 / P95 | 1407 / 2067 ms | 见下 |
| TTS 首包 P50 / P95 | 0 / 2524 ms | 见下 |
| `turn_end` 原因 | smart_turn 39 / stop_secs 6 | 见下 |

### 7.1 数据

两条链路用 `tools/turn_eval.py --breakdown`（读 `events.jsonl`，同一口径）拆同一批 15 条场景：

| 指标（P50 / P95，ms） | 级联（T13/T14b） | Pipecat（T15b） | 差值 |
|---|---|---|---|
| 回复数 | 45 | 44 | |
| 总等待 | **3147 / 6516** | **3515 / 10469** | +368 / +3953 |
| 轮次判断 | 600 / 1952 | 685 / 2636 | +85 / +684 |
| ASR | 446 / 1287 | 480 / 1631 | +34 / +344 |
| LLM | 1407 / 2067 | 1384 / 4682 | −23 / +2615 |
| TTS 首包 | 0 / 2524 | 0 / 2715 | 0 / +191 |
| `turn_end` 原因 | smart_turn 39 / stop_secs 6 | smart_turn 39 / stop_secs 5 | |

`tools/ws_client.py --all` 的客户端汇总（含网络与客户端排队）：级联 44 轮
**P50 3391.83 / P95 6741.93 ms**；Pipecat 44 轮 **P50 3678.77 / P95 10626.45 ms**。

两轮里 LLM 接口失败次数（从事件流统计）：

| | LLM timeout | LLM parse | 兜底 `repeat` 播放 |
|---|---|---|---|
| 级联 | 0 | 1 | 0 |
| Pipecat | 4 | 2 | 5 |

抢话变体（`scenarios/barge_in/s03b_barge_in.json`）5 次：本轮的 CONFIRM 现合成异常慢
（首包 2.4s、整句 11.1s），客户端因 Pipecat 不发 `turn_done` 而按“服务端空闲”等到确认语
播完才开口，**没有触发抢话**（`被抢话轮次 0`）。抢话逻辑改由单测覆盖：
`tests/test_playback.py`（排队模型）+ `tests/test_pipecat.py::test_pipecat_barge_in_uses_playback_tracker`
（缓存话术播放期间抢话 → `interrupted(tail=True, unplayed=...)`）。

### 7.2 结论

- **P50 差距不大**：3147 → 3515 ms（+368ms）。分段能解释的只有 +96ms（轮次判断 +85、
  ASR +34、LLM −23），剩下约 270ms 是 Pipecat 管线各阶段之间的调度间隙。**不是**第 26 次
  评审里写的“T15 5.3–6.0s vs T14b 2.87s”——那个 3 条场景的数字与本次全量同口径数据不可比。
  notes 里“量级一致”的说法要修正为：**P50 接近（同批场景差约 0.4s）**。
- **P95 差距主要来自 LLM 接口波动，不是框架**：Pipecat 这轮 3 个 >8s 的回复全是
  `抱歉，我没听清，请您再说一遍。`，对应 4 次 LLM 超时 + 2 次解析失败；级联这轮只 1 次
  解析失败、0 次兜底。同一份 `services/llm.LlmClient`，属于接口侧波动。但这也暴露一个
  共性问题：LLM 失败时 `repeat` 兜底要等满超时（首包 2.5s / 总 8s）才出声，**P95 会被
  LLM 失败直接放大**，T17 汇总时要单列。
- **打断**：排队模型修复后，AI 在浏览器里还在播（含缓存话术）时的抢话会记
  `interrupted(tail=True, unplayed=...)` 并标历史；LLM 阶段抢话不再丢文本。两者都有单测。

## 8. 已知限制 / 未做

- **麦克风实测未做**：本机没有麦克风输入，`tests/live/test_pipecat_live.py` 用场景 wav
  代替；页面用麦克风打一通需要真人操作。
- **`call.wav`**：Pipecat 路径只写 `caller.wav` / `ai.wav`，没写双声道对齐的 `call.wav`
  （Pipecat 不等播放，时间轴对齐没有意义）。
- **`smart_turn` 单次推理耗时**：模型在框架内部线程里跑，拿不到 `smart_turn` 事件。
- **`ttfs_p99_latency` 未设置**：见第 4 节第 7 条。

## 9. T15b 工作量（第 26 次评审修订）

| 阶段 | 用时 |
|---|---|
| 读 Pipecat 源码，定位打断两处缺陷（播放进度、`cancel()` 丢文本） | 约 2h |
| 抽 `PlaybackTracker`、改两套链路共用 | 约 1.5h |
| 抽 `stream_llm_turn` 等共用函数，`DialogProcessor` 对齐 `CallSession` | 约 2h |
| 打断阈值（VAD `start_secs`）、ASR 兜底、CONFIRM 兜底、`confirm_lead` | 约 2h |
| 事件补全（`turn_end` 原因、`turns`、接管录音、前端 ASR 栏） | 约 1h |
| 单测与同口径对比数据 | 约 1.5h |
| 文档 | 约 0.5h |

合计约 **10.5h（约 1.3 天）**。其中“浏览器播放进度”和“LLM 阶段文本顺延”是框架没提供、
必须在框架之上自己补的部分，属于 T17 框架对比里的一项工作量。

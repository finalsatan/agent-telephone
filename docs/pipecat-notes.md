# T15 / T15b / T15c：Pipecat 适配笔记

记录日期：2026-10-26（T15c 同日修订）
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
InterruptVADProcessor(InterruptSilero)           ← config.yaml: vad.threshold / speech_end_silence_ms
   ▼                                         start_secs = vad.start_secs（默认 Pipecat 0.2s）
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
  T15c 起开口与抢话阈值分离：`start_secs` 使用 Pipecat 默认值或 `vad.start_secs`；
  抢话阈值只在有处理任务或待播音频时按 Silero 有声采样累计检查。

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

T15b 曾把抢话阈值映射到 `VADParams.start_secs`，导致短句无法开口（第 27 次评审）。
T15c 改为 `VADUserTurnStartStrategy(enable_interruptions=False)`：框架照常标开口，
打断由 `InterruptVADProcessor` 独立广播。`InterruptSilero` 在同一次推理里按置信度和
音量累计实际有声采样，静音清零；达到页面阈值且 LLM/ASR/TTS 正在处理或浏览器有尾音时
才广播。默认开口 0.2s，可用 `vad.start_secs` 单独配置，页面抢话阈值不会改变它。

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

**T15b 历史数据**：以下旧数字只记录当时本地 CosyVoice3 的对照；T15c 云端实测结果见本文第 11 节。旧原始事件流未取得，无法追溯补算 T15b 的 `no_done_chunk`。

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
- **P95 归因待复核**：旧统计没有单列 `no_done_chunk`，第 27 次评审已证明取消竞态
  也会导致缺少 done 并播放 repeat。因此撤回“P95 差距主要来自 LLM 接口波动，不是框架”
  的确定归因；现有数据无法区分接口失败与管线竞态的贡献。修复后需重跑全量与 5 次抢话，
  保留原始事件流，分别列 timeout / parse / no_done_chunk / repeat，再分析长尾。
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


## 10. T15c 帧顺序修复（第 27 次评审）

本轮未修改 `core/` 数据类、服务接口或共享调用约定。实现依据项目安装的 Pipecat 1.13
包源码：`SegmentedSTTService` 的后台串行队列、`UserTurnProcessor` 开口广播、
`VADAnalyzer._run_analyzer()`、`TTSService._handle_tts_speak()` 与音频上下文队列。

| 要求 | 修复 | 真实管线测试（`tests/test_pipecat.py`） |
|---|---|---|
| 1 多段转写 | 每段发声分配编号；轮次结束冻结编号集合；全部返回文本或失败终态才调 LLM；移除 0.5s 宽限。登记/完成采用 SystemFrame，避免抢话清空数据队列造成永等 | `test_pipeline_waits_for_all_asr_segments`（文本、空转写、异常 3 档，最后一段在 turn_end 后阻塞超过 0.6s） |
| 2 独立打断阈值 | 开口保留默认 0.2s；抢话按实际有声采样和忙碌状态检查，不累计静音，覆盖 TTS 等首包阶段 | `test_pipeline_short_reply_is_recognized_at_any_interrupt_threshold`、`test_pipeline_interrupt_threshold_only_gates_barge_in`、`test_pipeline_short_noise_does_not_interrupt_cached_audio` |
| 3 LLM 竞态 | 每轮冻结输入、持有自己的 cancel；保留旧任务引用，新任务先 await 旧任务再读取顺延文本 | `test_pipeline_llm_barge_in_serializes_carry_and_keeps_shared_dialog`（新一轮结束后才放行旧首包） |
| 4 dialog 分叉 | LLM 返回前未提交任何 Dialog 变更，取消时只顺延文本，不替换 Dialog 对象 | 同上：检查对象恒等、下一轮 CONFIRM、页面槽位、slots.json、call_end |
| 5 号码兜底 | Speak 的任务上下文绑定号码，创建 context_id 时建立映射，完成/打断清理；无关帧不能覆盖 | `test_pipeline_confirm_phone_survives_system_and_audio_frames`（合成阻塞时注入 urgent 帧与 PCM，检查号码拼接、confirm_lead 和历史） |
| 6 turn_done | 无音频回复发 skipped；有回复的完成帧经 TTS 音频队列到 sink 后发，完成帧不可被打断队列清掉 | `test_pipeline_turn_done_allows_next_replay_turn`、`test_pipeline_turn_done_skipped_after_takeover`、`test_pipeline_tts_wait_can_be_interrupted_without_losing_turn_done` |

以上用例都启动真正的 `PipecatCall`，ASR/LLM/TTS 用 fake；Silero、分段 STT、轮次处理器、
TTS 基类和 sink 为真实组件，只替换 Smart Turn 的预测。场景 WAV 按 20ms 实时送入
`feed()`，先送 0.6s 通话静音用于框架音量检测预热；短句直接用 s03b 的 0.757375s“不对。”。
此外 `test_pipeline_failed_asr_completes_with_repeat` 验证整轮空转写/失败也会正常播 repeat 并结束。

### 第 7 条补测与独立统计（第 27 次评审时的状态）

当时真实 API 测量尚未执行；`tools/turn_eval.py --breakdown` 已扩展为按通话和总计分别输出
`timeout`、`parse`、`no_done_chunk`、`repeat`，测试为 `test_llm_failure_counts_separates_no_done_chunk`
与 `test_breakdown_cli_exports_llm_failures`。当时记录的待测命令和本地 TTS 计划已被本文件第 11 节
的云端实测结果取代。

### T15c 工作量（本次会话估计，包含测试等待）

| 阶段 | 用时 |
|---|---|
| 读任务、评审与包源码，切分支合并文档，准备 Python 3.12 环境 | 约 5 分钟 |
| 段编号、独立抢话阈值、取消串行化、共享 Dialog、TTS 上下文与 turn_done | 约 6 分钟 |
| 真实管线回归、统计脚本与全量验收 | 约 8 分钟 |
| 文档与提交核查 | 约 3 分钟 |

合计约 **22 分钟（0.37h，按本次会话墙钟估算）**，不含未执行的云端/本地 TTS 实测。
这部分是框架帧顺序和项目逻辑之间的额外适配工作，T17 对比应单独计入。


最终验收：`uv run pytest` **634 passed、13 deselected、2 warnings（81.88s）**；
`node --test tests/js/*.test.js` **21 passed**。Pipecat 测试共 34 项，其中新增 15 个真实
管线参数化用例；独立统计脚本新增 2 项。默认排除 live 测试，以上不是云端 API 实测。


### 第 1 轮评审补充（T15c）

复核 Pipecat 1.13 的 `SegmentedSTTService`：在 `_handle_user_stopped_speaking()` 中，服务
`not is_usable` 会在队列前返回；`_segment_task_handler()` 只对 truthy audio 调用 `run_stt()`，
因此空段也可能绕过识别。适配层将 VAD 打开的编号和已排队编号分开：不可用/空段当场产出失败
终态，只有实际放进分段队列的编号交给 `run_stt()`；意外异常同样产出终态，屏障不会悬挂。
`cancel()` 会在 teardown 丢队列，通话同时结束并取消 DialogProcessor，属整通关闭路径。

轮次串行器现吞并前一轮的普通异常（写 `error(stage="turn")` 后继续）或非本轮取消产生的
`CancelledError`；当前任务自身取消才向上退出。每轮 LLM 抢话都把用户文本放进 `_carried_text`，
即使无分段输入路径已冻结下一轮文本，也会在等待旧任务结束后与顺延文本合并。旧的直接逻辑
单测调用仍兼容 `_user_text`，正式 PipecatCall 使用 SegmentedSTTService 的分段路由。

新增管线用例：`test_pipeline_next_turn_runs_after_previous_turn_exception`、
`test_pipeline_unusable_stt_segment_closes_barrier_without_shifting_ids`。


评审修订验收：`uv run pytest` **636 passed、13 deselected、2 warnings（90.55s）**；
Node **21 passed**。定向 `tests/test_pipecat.py tests/test_turn_eval.py` **50 passed**。


## 10. T15c 实测工作量（2026-10-26）

| 阶段 | 用时 |
|---|---:|
| 云端话术预合成、检查缓存及两条服务配置 | 约 5 分钟 |
| Cascade `--all`（15 场景/44 轮） | 约 23 分钟 |
| Pipecat `--all`（15 场景/44 轮） | 约 21 分钟 |
| 抢话变体（两链路各 5 次） | 约 15 分钟 |
| 事件复核、数据整理、全量验收与提交 | 约 10 分钟 |
| **第 7 条累计（含首次作废和沙箱禁网重试）** | **约 1.2 小时** |

## 11. T15c 第 7 条云端实测（2026-10-26）

### 环境与数据有效性

首次回放使用了 `config.yaml` 的 `tts.backend: local`，整批数据作废，不纳入本节。先前云端预合成出现
`[Errno 8] nodename nor servname provided, or not known` 是 workspace-write 沙箱禁网所致，并非真实网络/API 故障。
提升权限后访问恢复，`tools/prerender_phrases.py --backend cloud` 完成 21 条生成、3 条跳过、0 失败，耗时 37.2s，
缓存位于 `cache/tts/cloud`。

本次进程配置在 `web.server.create_app()` 中覆盖为 `tts.backend=cloud`，不改 `config.yaml`。级联服务 8000、
Pipecat 服务 8002 的 `/api/config` 都返回 `tts_backend=cloud`；8001 已被占用，未干预。50000、50004 只做了
只读健康探测，未停止、重启或操作。密钥仅从 `.env` 加载到进程环境，未打印或写入记录。

### 同条件实测

两条链路均在同一 Mac 上使用相同 ASR/LLM 模型、云端 TTS 和 `cache/tts/cloud`。误切按每通输入轮数对照
`vad_start` 数：出现额外分段的通话数 / 额外分段总数。抢话变体成功只统计 5 次 `s03b_barge_in` 中收到
`interrupted` 的次数。阶段耗时直接取原始 `events.jsonl` 的延迟字段，包含缓存 TTS 的 0ms。

| 链路 | 误切通话 / 额外 VAD 段 | 抢话变体 | `ws_client --all` P50/P95（ms） | ASR P50/P95（n） | LLM 首 token P50/P95（n） | TTS 首包 P50/P95（n） | LLM timeout / parse | no_done_chunk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cascade，15 场景/44 轮 | 5/15 / 6 | 2/5 | 6191 / 11175（42 个时延样本） | 463 / 1735（46） | 1419 / 4094（43） | 0 / 2174（59） | 9 / 1 | 9 |
| Pipecat，15 场景/44 轮 | 6/15 / 8 | 0/5 | 5696 / 10152（44） | 419 / 806（52） | 1230 / 4800（44） | 0 / 2191（60） | 11 / 1 | 8 |
| T14b 历史参考：WSL2 + RTX 4060 Ti，本地 CosyVoice3 | 3/29（s01–s10 子集） | 5/5 | 3027 / 6223（44） | 396 / 564 | 未单列 TTFT；LLM 全段 1475 / 1909 | 0 / 2528 | 旧流不可复核 | 未知 |

T14b 行机器和 TTS 后端都不同，仅作历史参考。全量回放中的客户端抢话轮次另为 Cascade 6/44、Pipecat 3/44；
该指标不是 5 次显式抢话变体的成功数。Cascade 另有 3 次 ASR timeout，Pipecat 没有 ASR 错误。

原始事件目录（gitignore）：`calls/t15c_live2_cascade/` 与 `calls/t15c_live2_pipecat/`；主报告：
`report/t15c_live2_cascade_all.json`、`report/t15c_live2_pipecat_all.json`；抢话逐次报告：
`report/t15c_live2_{cascade,pipecat}_barge_1..5.json`。

### P95 与 `no_done_chunk` 事件核查

在两链路 15 通原始 JSONL 中，Cascade 有 9 个 LLM timeout、1 个 parse、3 个 ASR `AsrTimeout`；
Pipecat 有 11 个 LLM timeout、1 个 parse、0 个 ASR 错误。`no_done_chunk` 逐条按 LLM error 的 `message` 含
`no done chunk` 统计，为 Cascade 9 次、Pipecat 8 次；它们都是上述 LLM timeout 的子集（事件 `reason` 仍为 `timeout`），
不能重复加进 timeout 总数。举例：

- Cascade call `web_20261026_222421_cace3f`（s03_bank）末轮 `turn_end.reason=stop_secs`，轮次判断约 2024ms、
  ASR 813ms，之后 LLM error 原文 `turn exceeded 8s (total timeout); emitted 1 sentence(s), no done chunk`，
  接着播缓存 repeat；该回复总等待 10843ms。
- Pipecat call `web_20261026_223737_7b3748`（h02_property）首轮轮次判断约 2738ms、ASR 539ms，随后同类
  LLM timeout/no_done_chunk 并播 repeat；该回复总等待 10743ms。
- 也有非 LLM 主导的长尾：Cascade `web_20261026_223306_e74f28` 第三轮 LLM 全段 6766ms，确认语 TTS 首包
  4329ms，总等待 12311ms；Pipecat `web_20261026_224527_d6349f` 确认回复 LLM 全段 1366ms，云端 TTS 首包
  7088ms，总等待 9862ms。

客户端端到端 P95 为 Cascade 11175ms、Pipecat 10152ms；`turn_eval` 从事件流拆出的响应 P50/P95 分别为
Cascade 5833/10804ms、Pipecat 5014/9663ms。LLM 超时/no_done_chunk 确实造成若干长尾，但同一 P95 范围内也有
云端 TTS 首包和 `stop_secs` 轮次兜底耗时。Pipecat 的 LLM 首 token P95（4800ms）高于 Cascade（4094ms），而
Pipecat 总 P95 低约 1024ms。事件证据不足以把链路间的 P95 差值单独归因给 LLM 接口波动。

抢话变体各 5 次：Cascade 2/5 次收到 `interrupted`（其余 3 次未等到匹配的“对吗”）；Pipecat 0/5 次收到
`interrupted`，5 次均未等到匹配语句，客户端给出对应 warning。逐次 P50/P95 与 call id 保存在上述 JSON 报告。


全量验收：提升权限的 `uv run pytest` **636 passed、13 deselected、2 warnings（90.49s）**；
`node --test tests/js/*.test.js` **21 passed**。普通沙箱中 fake WebSocket 测试受本地 bind 权限限制，
提升权限后全量通过；`git diff --check` 通过。

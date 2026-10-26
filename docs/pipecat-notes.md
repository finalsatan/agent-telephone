# T15：Pipecat 适配笔记

记录日期：2026-10-26
Pipecat 版本：`pipecat-ai==1.13.0`（`requirements.txt` 已固定）
代码：`adapters/pipecat/{stt,tts,llm,pipeline,run}.py`，测试 `tests/test_pipecat.py`、
live 测试 `tests/live/test_pipecat_live.py`。

## 1. 接进来的形态

```
浏览器 WS /ws/call（与 T13 同一协议）
   │ 16kHz PCM 二进制帧 / 控制 JSON
   ▼
PipecatCall.feed()  → PipelineWorker.queue_frame(InputAudioRawFrame)
   ▼
VADProcessor(SileroVADAnalyzer)           ← config.yaml: vad.threshold / speech_end_silence_ms
   ▼
SiliconFlowSTTService (SegmentedSTTService) → services.asr.AsrClient
   ▼
UserTurnProcessor(TurnAnalyzerUserTurnStopStrategy(ConfiguredSmartTurn))
   ▼
TurnEventLogger                            ← 把 vad_start / vad_end / turn_end 写成 T10 事件
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

## 2. 接入工作量

实际投入约 **1.5 天**（在 2 天限时内）：

| 阶段 | 用时 |
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
| LLM（`services/llm.py`） | **直接复用** `turn_stream()`。 |
| 事件流（`core/events.py`） | **直接复用** `EventLogger`；新增一个 `ForwardingEventLogger` 的同类（复用 `web/server.py` 的实现）。 |
| 信道模拟（`tools/channel_sim.py`） | **直接复用**。在 `feed()` 里逐段 `ChannelSimulator.feed()`。 |
| VAD / Smart Turn | **用框架的**：`SileroVADAnalyzer` + `LocalSmartTurnAnalyzerV3`（与 T09 的 `core.turn` 是同一批模型）。 |
| 轮次事件（`vad_start` / `vad_end` / `turn_end`） | **必须自写**：Pipecat 内部不写 T10 格式的事件，加 `TurnEventLogger` 把框架帧翻成事件。 |
| 打断的 `[被打断]` 标记与 `interrupted` 事件 | **必须自写**：在 `DialogProcessor` 里根据 `_playing_text` 判断，避免和 TTS 的取消重复写事件。 |
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
   现在只由 `DialogProcessor` 写（它能判断“在播 / 已播完”并给历史打标记），TTS 只在
   `finally` 里补 `play_end`（带已送出的时长）。
6. **`PipelineWorker` 的 `StartFrame` 由框架发**。问候语挂在 `DialogProcessor` 的
   `StartFrame` 处理里，`PipecatCall.start()` 要等 `on_pipeline_started` 事件后再让
   调用方开始喂音频。
7. **`ttfs_p99_latency` 未设置**。启动时会 warning，`TurnAnalyzerUserTurnStopStrategy`
   用它做 STT 兜底超时。当前 ASR P95 约 0.9s，默认 1.0s 够用；要更准可以在
   `SiliconFlowSTTService` 上暴露 `STTMetadataFrame`。
8. **`STTSettings` / `TTSSettings` 的 `language` 未初始化**，启动时有一条 ERROR 日志
   （不影响运行）。显式填 `language=None` 可消掉。

## 5. 打断的取消顺序

设计文档第 5 节要求：**停止播放 → 清空待播缓冲 → 取消 TTS → 取消 LLM → 转听**。
Pipecat 的做法是 `UserTurnProcessor` 在用户开口时广播一个 `InterruptionFrame`，它同时
向上下游传播：

- 下游的 `SiliconFlowTTSService` 收到后取消当前 `run_tts()` 任务、丢弃排队的音频上下文
  （等价于“停播 + 清缓冲 + 取消 TTS”）；
- 下游的 `DialogProcessor` 收到后取消正在跑的 LLM 任务（等价于“取消 LLM”）；
- 浏览器收到 `interrupted` 事件后清空本地待播队列（T14 前端）。

TTS 在 `DialogProcessor` 的下游，两者都会收到广播，但**框架不保证两个处理器的处理
先后**。功能上两者都会在下一帧前停掉，实测从来电者开口到 `interrupted` 约 464ms，
与 T13 手写顺序（431ms）同量级；如果要严格复现“先 TTS 后 LLM”的顺序，需要在
`DialogProcessor` 里延迟一个事件循环再取消 LLM，当前没做（收益很小）。

## 6. 离线回放：用 T10 路径作基线

Pipecat 是实时流框架，没有“把一条场景 wav 按轮喂进去、按模拟时钟推进”的入口；
硬做需要自己写一套输入 transport 并接管它的时钟，性价比低。按任务书允许的替代方案：

- **时延/槽位基线仍用 T10 的离线路径**（`tools/replay.py` + `tools/evaluate.py`），
  这样两框架对比的是“同一批模型 + 同一批场景”，差异只来自编排。
- **Pipecat 路径用 T13 的 `tools/ws_client.py` 做实时回放**：它按 32ms 帧、实时速度把
  每轮 wav 发到 `WS /ws/call`，服务端换不换适配器对客户端透明。实测见第 7 节。
- 事件流与 T10 **同格式**：`tools/evaluate.py:extract_turn_metrics()` 直接读 Pipecat 的
  `events.jsonl` 能得到每轮的 `e2e_ms / asr_ms / llm_ms / tts_first_ms / parse_ok`
  （见第 7 节数据）。唯一缺的是 `smart_turn` 的推理耗时事件（模型在框架内部线程里跑，
  拿不到），`evaluate.py` 不依赖它。

## 7. 实测（2026-10-26，本地 CosyVoice3 + 硅基流动 ASR/LLM）

启动：`uv run python -m adapters.pipecat.run`（8001）；用 `tools/ws_client.py` 回放。

| 场景 | 轮数 | e2e P50 / P95（ws_client，含轮次判断） | 备注 |
|---|---|---|---|
| s01_delivery | 3 | 5895 / 5940 ms | 第 2、3 轮 TTS 现合成 |
| s03_appointment | 3 | 5337 / 6947 ms | |
| s03b_barge_in | 4 | 6008 / 7352 ms | 第 4 轮确认语播到 464.81ms 被打断 |

从 `events.jsonl` 用 `extract_turn_metrics()` 拆出的分段（s01）：

| 轮 | e2e | ASR | LLM | TTS 首包 |
|---|---|---|---|---|
| 1 | 1698 ms | 628 ms | 1067 ms | 0（缓存命中） |
| 2 | 5172 ms | 852 ms | 1913 ms | 2403 ms |
| 3 | 5141 ms | 714 ms | 1799 ms | 2624 ms |

- 与 T13c 的级联链路量级一致（T13c P95 约 5.5s，主因同样是 CONFIRM 轮的现合成 TTS）。
- 抢话：`interrupted(reason="barge_in")` 在来电者开口后 **464.81ms** 到达，与 T13b 的
  431ms 同量级（Pipecat 的 VAD 确认 + 广播比手写链路多约 30ms）。
- 接管：`takeover` 事件之后 **0 个 `play_start`**，AI 静默。
- 缓存命中：问候语、`ask_name`、`ask_purpose`、`repeat` 命中预合成缓存，首包 0。

## 8. 已知限制 / 未做

- **麦克风实测未做**：本机没有麦克风输入，`tests/live/test_pipecat_live.py` 用场景 wav
  代替；页面用麦克风打一通需要真人操作。
- **接管后的摘要字段**：`takeover` 事件带 `turns=None`，`core/summary.py` 的接管情况
  按事件存在与否判断，不受影响；但 `call_end.turns` 没有统计轮数（Pipecat 路径）。
- **`interrupted(stage="llm")`**：LLM 还在跑就被抢话时，Pipecat 直接取消任务，不会走
  `_run_turn()` 里写 `interrupted(stage="llm")` 的分支；T13 会写。对摘要影响很小。
- **`call.wav`**：Pipecat 路径只写 `caller.wav` / `ai.wav`，没写双声道对齐的 `call.wav`
  （Pipecat 不等播放，时间轴对齐没有意义）。

# T16：LiveKit Agents 适配笔记

记录日期：2026-10-27
LiveKit Agents 版本：`livekit-agents==1.8.9`（`requirements.txt` 已固定）
代码：`adapters/livekit/`，测试 `tests/test_livekit.py`、`tests/live/test_livekit_live.py`。

---

## 1. 先查文档与源码的结论（动手前）

依据：`livekit-agents==1.8.9` 的包内源码（`.venv/lib/python3.12/site-packages/livekit/agents/`）
与 docs.livekit.io 的 Agents 1.x 文档。任务书要求先查清下面几点。

### 1.1 AgentSession / Agent 生命周期；能否不连 livekit-server

生命周期：`initializing → listening → thinking → speaking → listening`，由
`agent_state_changed` 事件暴露；`session.start(agent=...)` 进入 listening，
`session.shutdown()` 收尾（`close` 事件）。

**可以不连 livekit-server**（`voice/agent_session.py:877-1160`）：

```python
session.input.audio = my_audio_input     # io.AudioInput 子类，async iterator[rtc.AudioFrame]
session.output.audio = my_audio_output   # io.AudioOutput 子类，实现 capture_frame()
await session.start(agent=my_agent)      # room 参数可省略
```

`start()` 里只有 `elif is_given(room)` 才建 `RoomIO`；不传 `room` 时直接
`asyncio.create_task(self._forward_audio_task())` 消费 `input.audio`。
`get_job_context(required=False)` 返回 `None` 也不影响（只是少了录制与 job 遥测）。
**结论：管线级用例放默认测试（`tests/test_livekit.py`），不需要 livekit-server。**
`livekit-server` 只在浏览器真机联调时需要（见第 8 节）。

### 1.2 非流式 STT / 自定义 TTS / 固定 JSON 的 LLM 怎么接

- **非流式 STT**：`stt.StreamAdapter(stt=<非流式 STT>, vad=<VAD>)`（`stt/stream_adapter.py`）。
  它按 VAD 的 END_OF_SPEECH 切段，对每段调一次 `stt.recognize()`，产出
  `FINAL_TRANSCRIPT`。也可以不包：`Agent.default.stt_node()` 在发现
  `stt.capabilities.streaming == False` 时会自动用 `StreamAdapter` 包一层
  （`voice/agent.py:510`），只要 session 上挂了 VAD。**我们用后者（少一层自写）。**
- **自定义 TTS**：`tts_node(self, text: AsyncIterable[str], model_settings)` →
  `AsyncIterable[rtc.AudioFrame]`（`voice/agent.py:596`）。可以直接用本项目的
  `services/tts` + `core/phrases.PhraseCache` 产出 PCM 帧，不必实现 LiveKit 的
  `tts.TTS` 抽象。另一种是 `session.say(text, audio=<AsyncIterable[AudioFrame]>)`
  直接送预合成音频，但那样绕开了分段与 `FlushSentinel`，我们用 `tts_node`。
- **固定 JSON 的 LLM**：`llm_node(self, chat_ctx, tools, model_settings)` →
  `AsyncIterable[ChatChunk | str | FlushSentinel]`（`voice/agent.py:572`）。
  本项目回复由 `Dialog` 用代码决定，所以不用 LiveKit 的 `llm.LLM`：
  在 `Agent.on_user_turn_completed(turn_ctx, new_message)` 里拿到本轮用户文本、
  跑 `Dialog` + `services.llm.turn_stream()`，把要播的文本存起来，`llm_node`
  只负责把它按段 yield 出去。`FlushSentinel` 用来切段（confirm_lead 的前后两句）。

#### 收尾：整轮 ASR 的接法（livekit-agents 1.8.9 源码）

`AgentSession.commit_user_turn()` 调用音频识别路径并等待最终转写（`voice/agent_session.py`、
`voice/audio_recognition.py`）；非流式 STT 的默认节点会以 VAD 包装为 StreamAdapter，
因而不能保证一轮只识别一次。`generate_reply(user_input=...)` 直接安排回复，
不会调用 `on_user_turn_completed`（`voice/agent_activity.py`）。因此关闭框架 STT，
由 TurnDetector 的时间戳截取整轮音频，调用项目 ASR，显式调用对话 hook 后使用
`generate_reply` 安排播放；不再调用等待框架转写的 `commit_user_turn()`。
ASR 失败或空文本仍由对话层播 repeat。离线复核继续使用 `core.summary.recheck_slots`。

### 1.3 轮次检测的选项，能否用本项目的 Smart Turn

`turn_handling.turn_detection` 取值（`voice/turn.py:96`）：
`"stt"` / `"vad"` / `"realtime_llm"` / `"manual"` / 自定义 detector 对象。

自定义 detector 有两种协议（`voice/turn.py:36` / `:80`）：

| 协议 | 输入 | 用途 |
|---|---|---|
| `_TurnDetector` | `chat_ctx`（文本） | 文本型 EOU 模型（LiveKit 自带的 mini 模型） |
| `_StreamingTurnDetector` | `push_audio(frame)` + `predict() → Future[TurnDetectionEvent]` | **音频型 EOU**，正是 Smart Turn 的位置 |

框架对音频型 detector 的用法（`voice/audio_recognition.py:1460-1530`、`:1570-1900`）：

1. VAD `START_OF_SPEECH` → `stream.cancel_inference()` 并清掉 prediction future；
2. 说话中 `raw_accumulated_silence >= MIN_SILENCE_DURATION_MS(=200ms)` → 调一次
   `stream.predict()`（future 在推理完时 resolve）；
3. VAD `END_OF_SPEECH` → 跑 EOU：取 future 的概率与 `unlikely_threshold()` 比较，
   低于阈值（negative EOT）就把等待时间从 `endpointing.min_delay` 换成 `max_delay`
   （**正好是 stop_secs 兜底**）；
4. 等够延迟后 `on_end_of_turn()` 提交本轮。

**结论：能接，但接上去会丢后半句，所以主路径不用它。** 原因是框架的 EOU 由 VAD 的
`END_OF_SPEECH` 触发，而本项目的 ASR 是非流式的（`StreamAdapter` 正是在这一刻才发起识别），
转写永远比 VAD 事件晚到：

- 单段发声：框架在 `_run_eou_detection()` 里发现还没有转写就先返回，等
  `FINAL_TRANSCRIPT` 到了再用 `trigger="stt"` 补跑一次 EOU，能补上；
- **一轮多段发声**（Smart Turn 判第一段没说完、对方停顿后接着说）：第二段的
  `END_OF_SPEECH` 会用**第一段的旧转写**跑 EOU 并提交本轮（正 EOT 时 `min_delay`
  几乎不等待），后半句被挤到下一轮——与 T15c 第 1 条是同一个问题。
  实测：喂 `turn_01.wav` → 0.7s 静音 → 再喂一遍，框架提交了两次
  `llm_request`（`"我找"`、`"张先生"`），而不是一次 `"我找 张先生"`。

**实际采用：`turn_detection="manual"` + 自己提交轮次。** `LiveKitCall.feed()` 里跑
`core.turn.TurnDetector`（Silero + Smart Turn + stop_secs，与级联 T13 是同一套、同一份
`config.yaml` 参数），判出 `turn_end` 后按时间轴截取整轮并转写一次，再显式调用
`on_user_turn_completed()` 和 `generate_reply()`，见 1.2 的收尾说明。
代价是框架的 `interruption.min_duration` 不再生效，打断改由 `TurnSpeechCounter`
（`core.turn` 里的同一份实现）判定后调 `session.interrupt()`——这正好是任务书要求 4
里“按 `TurnSpeechCounter` 的语义累计有声时长”的写法，行为与级联／T15 一致。

LiveKit 自带的 turn detector（`inference.TurnDetector()`）是云端推理，按任务书只作为
选做对照，没有接入。

### 1.4 打断的参数；框架怎么知道 AI 播到哪

`turn_handling.interruption`（`voice/turn.py:150`，默认值在 `_INTERRUPTION_DEFAULTS`）：

- `enabled`（默认 True）、`mode`（`"adaptive"` / `"vad"`；不写时按环境自动选，
  **生产模式下自动选 `"vad"`，dev/hosted 才用 adaptive**，见
  `voice/agent_activity.py:5316`。我们显式写 `"vad"`，避免任何云端推理）；
- `min_duration`（默认 0.5s）：**只管打断**。`on_vad_inference_done` 里
  `active_speech = ev.speech_duration >= min_duration` 才 `_interrupt_by_audio_activity()`；
- `min_words`（默认 0）：按转写词数再收一道；
- `resume_false_interruption` / `false_interruption_timeout`（默认 True / 2.0s）：
  误打断后从断点继续播，并 emit `agent_false_interruption`；
- `discard_audio_if_uninterruptible`。

**开口与打断天然分离**：VAD 的 `min_speech_duration`（默认 0.05s）决定开口，
`interruption.min_duration` 只决定是否打断。所以任务书要求 2（"对""好的"在任何阈值下
都能识别、阈值只管打断）在 LiveKit 上是框架原生行为，不需要像 Pipecat 那样自己补。
实际手动模式由 `TurnSpeechCounter` 用 `interrupt.min_speech_ms` 判定。

**播放进度**（`voice/io.py:147`）：`AudioOutput` 有
`playback_started` / `playback_progressed(started_at, offset, duration)` /
`playback_finished(playback_position, interrupted)` 事件，还有
`_AudioSinkProxy` 会在切换 sink 时把"已播到哪"算出来；`SpeechHandle.interrupted`
是布尔。**播放进度优先用框架的**（`AudioOutput.on_playback_progressed`），
`core/playback.PlaybackTracker` 只在框架给不出时兜底——实际接的时候会确认
（见第 5 节）。

### 1.5 会不会默认调用 LiveKit Cloud

不会，前提是：

- 只用自托管 `livekit-server --dev` 连房间；
- 不碰 `livekit.agents.inference.*`（`inference.STT/LLM/TTS/TurnDetector` 都是云端网关）；
- 不装云端插件（`livekit-plugins-*` 里我们只装 `silero`，纯本地 ONNX 推理）；
- `interruption_detection` 显式写 `"vad"`，否则 dev 模式下会拉
  `inference.AdaptiveInterruptionDetector()`；
- `preemptive_generation` 关闭（任务书要求：回复要先由 `Dialog` 更新状态再决定）。

### 1.6 其它要点（会影响实现）

- **`rtc.AudioFrame`**：`data`(bytes) / `sample_rate` / `num_channels` / `samples_per_channel`。
  本项目固定 16kHz 单声道 int16 小端，`samples_per_channel = len(data)//2`。
- **STT 分段**：`StreamAdapter` 用 VAD 切段，一轮多段发声会产出**多条**
  `FINAL_TRANSCRIPT`，框架把它们累加到 `_audio_transcript`
  （`_process_stt_event` 里 `self._audio_transcript += " " + transcript`）。
- **`rtc.AudioFrame.data` 是 int16 的 `memoryview`**：`len(frame.data)` 是**采样数**，
  字节数要看 `frame.data.nbytes` / `len(bytes(frame.data))`。写错会把 20ms 帧当 10ms。
- **`AgentSession` 有两处硬性检查**：`AgentActivity.on_end_of_turn()` 里
  `elif self.llm is None: return`（没有 LLM 就不生成回复），`AgentActivity.say()` 里
  没有 `tts` 就抛 `RuntimeError`。我们用 `llm_node` / `tts_node` 自己产出文本与音频，
  所以挂了两个占位实现（`adapters/livekit/nodes.py`）满足检查。
- **`rtc.EventEmitter` 用 `self._events` 存事件名集合**：自定义 `stt.STT` 子类里
  不要再把日志器存成 `self._events`，会把它覆盖掉（踩过一次，见第 4 节）。
- **`Agent` 子类不要用 `self._llm` / `self._tts` / `self._stt` / `self._vad` 存自己的东西**：
  那是 `Agent` 自己的字段（`Agent.llm` 等属性直接读它们）。踩过一次：`self._llm`
  存了 `services.llm`，导致 `isinstance(self.llm, llm.LLM)` 为假，框架静默跳过整条回复。
- **`_generate_reply()` 会 emit `speech_created`**，可以据此拿到 `SpeechHandle`，
  `await handle.wait_for_playout()` 就是“这一轮播完/被抢话”的收尾点。
- **`session.commit_user_turn()` 会取消上一次未完成的提交**：两轮挨得近时必须自己串行化。
- **`session.interrupt()`** 是显式打断入口（返回 future）；`SpeechHandle.interrupt()` 同理。
- `MIN_SILENCE_DURATION_MS = 200`（`inference/eot/base.py:32`）。

---

## 2. 接入工作量

T16 实际投入约 **1.5 天**（在 2 天限时内），其中“查文档/源码 + 管线级用例”占大头，
真正写适配层约 6h，其余是定位框架行为（见第 4 节）。

| 阶段 | 用时 |
|---|---|
| 读 LiveKit Agents 1.8 文档与源码（AgentSession / 节点 / 轮次 / 打断 / 播放事件） | 约 5h |
| 写 STT / TTS / 占位 LLM·TTS / 音频 IO | 约 2h |
| 写 `CallAgent`（Dialog + 一轮 + 打断 + `turn_done`） | 约 3h |
| 写 `LiveKitCall`、与 T13 接口对齐 | 约 2h |
| 管线级用例（15 个）与定位问题 | 约 5h |
| 文档与数据整理 | 约 1h |
| 修首轮问候语误判抢话（本机复现 + 回归用例 + 复测） | 约 1h |

| 收尾：整轮 ASR / 阈值与管线回归 | 约 0.5h |
| 收尾：问候语等待与真实房间测试 | 约 0.3h |
| 收尾：三链路串行实测与整理（含实时等待） | 约 1.4h |

选做 LiveKit 自带 turn detector：**未做，原因：限时**。先前表内约 19h，收尾后已超过约两天工作时限。

## 3. 能直接复用 / 必须自写

| 部分 | 结论 |
|---|---|
| ASR（`services/asr.py`） | **直接复用**。每个 `turn_end` 按时间戳截取整轮并调用一次 `transcribe()`；不启用框架 STT。 |
| TTS（`services/tts.py` / `tts_local.py`） | **直接复用**。`CallAgent.tts_node()` 里 `async for chunk in synth(text, cancel)`。 |
| 预合成话术缓存（`core/phrases.py`） | **直接复用**。`ReplySpeaker.synth()` 先查缓存，命中首包 0。 |
| 对话状态机（`core/dialog.py`） | **直接复用**。接不进 LiveKit 的 `llm.LLM`（固定 JSON + 代码决定流程），写成 `Agent.llm_node()`。 |
| 一轮的判断逻辑（`core/call.py`） | **直接复用** `stream_llm_turn()` / `choose_reply_text()` / `is_partial_phone()` / `template_hint()`。 |
| 历史操作 | **T16 起抽成 `core/history.py`**，`CallSession` / `DialogProcessor` / `CallAgent` 共用（同一份“（被对方打断）”与“从未播出的句子”逻辑）。 |
| 轮次判断（`core/turn.py`） | **直接复用** `TurnDetector`（Silero + Smart Turn + stop_secs）。框架自带的轮次判断在非流式 ASR 上会丢后半句（第 1.3 节），所以用 `turn_detection="manual"` + 自己提交。 |
| 打断计数（`TurnSpeechCounter`） | **直接复用**，`LiveKitCall._maybe_barge_in()` 用它判 300ms 阈值。 |
| 信道模拟（`tools/channel_sim.py`） | **直接复用**。在 `feed()` 里逐段 `ChannelSimulator.feed()`。 |
| 事件流（`core/events.py`） | **直接复用** `EventLogger`；复用 `web/server.py` 的 `ForwardingEventLogger`。 |
| VAD / Smart Turn 模型 | 复用**同一批 ONNX 模型**，但走 `core/turn.py` 而不是框架的插件（框架的 VAD 只用于 `user_state_changed`）。 |
| 文本分段合成（`FlushSentinel`） | **用框架的**：`confirm_lead` 拆句就是两个段，框架保证“下一段在上一段播完前开始合成”，被打断时已播完的段保留。 |
| 每段播放进度 | **用框架的**：`AudioOutput` 的 `playback_started` / `playback_progressed` / `playback_finished(playback_position, interrupted)`。`core/playback.PlaybackTracker` 没用到（框架给得出，见第 5 节）。 |
| 音频 IO | **必须自写**：`QueueAudioInput` / `PacedAudioSink`（不连 livekit-server 时用；房间路径把 sink 写到 `rtc.AudioSource`）。 |
| 占位 LLM / TTS | **必须自写**：`nodes.py`，只为过框架的两处前置检查。 |
| 浏览器音频（WebRTC） | **已实现**：面板与事件流复用 T14，只有采集/播放按 LiveKit JS 客户端单独一份。 |

### 与 Pipecat 的差异（收尾）

LiveKit 的原生 StreamAdapter 会按框架 VAD 将一轮切成多段分别转写；Pipecat 适配器
由自己的轮次判断控制识别音频。LiveKit 现在同样在 `turn_end` 时整轮转写，绕过框架
STT 分段，并手动调用对话 hook 后交给框架生成和播放。源码核对、改接线及管线回归
本次合计约 0.5 小时。播放进度、合成取消仍由框架提供；轮次、累计有声打断判断、
LLM 阶段顺延文本、号码兜底和 `turn_done` 仍由适配层实现。

## 4. 踩到的坑（按影响排序）

1. **框架 STT 分段破坏整轮识别并可能触发 5 秒等待**：语音停顿切出多段且前缀重叠，
   数字串容易截断。现在用项目轮次边界做一次整轮识别，绕过框架 STT。

2. **`Agent` 的 `self._llm` 被自己的属性覆盖**。`CallAgent.__init__` 里写
   `self._llm = services.llm`，而 `Agent.llm` 属性直接返回 `self._llm`，于是
   `_generate_reply()` 的 `elif isinstance(self.llm, llm.LLM)` 为假，整条回复被静默跳过
   （没有任何报错，只有 `speech_created` 之后再无 `llm_node`）。改名 `_llm_client` 解决。
3. **自定义 `stt.STT` 子类里的 `self._events` 覆盖了 `rtc.EventEmitter._events`**。
   日志器一存进去，`STT.on("metrics_collected", ...)` 就抛
   `TypeError: argument of type 'ForwardingEventLogger' is not iterable`。改名 `_event_log`。
4. **框架的轮次判断配非流式 ASR 会丢一轮的后半句**（第 1.3 节），现在由项目轮次判断和整轮识别处理。
5. **`say()` 需要 session 上有 `tts`，生成回复需要 `llm`**。用 `llm_node` / `tts_node`
   绕开它们还不够，必须挂占位实现（`nodes.py`），否则问候语抛 `RuntimeError`、
   `on_user_turn_completed()` 之后静默返回。
6. **`rtc.AudioFrame.data` 的 `len()` 是采样数**，不是字节数（第 1.6 节）。
7. **抢话窗口要覆盖“TTS 首块音频之前”**。只按“正在播 / 正在提交轮次”判会漏掉这段：
   `session.commit_user_turn()` 返回时 TTS 可能才刚开始合成。用
   `agent.busy`（`_turn_in_flight or _awaiting_turn_end`）补上。
8. **`commit_user_turn()` 会取消上一次未完成的提交**：两轮挨得近（例如阈值调高、对方在
   AI 说话时说话）时必须自己用 `asyncio.Lock` 串行化，否则前一轮被取消、转写丢掉。
9. **`interrupt.min_duration` 在手动模式下不生效**：`on_vad_inference_done` 对
   `manual` 直接 return。打断要自己判（`TurnSpeechCounter`）后调 `session.interrupt()`。

## 5. 打断与行为对齐（要求 4 逐条）

| 要求 | LiveKit 上怎么做到 | 用例 |
|---|---|---|
| 一轮多段发声不丢后半句 | `turn_detection="manual"` + 整轮音频一次转写 | `test_pipeline_multi_segment_turn_keeps_second_half` |
| 打断阈值只管打断、不管开口 | 开口由项目 VAD 决定，打断阈值只用于 `_maybe_barge_in()` | `test_pipeline_short_reply_recognised_regardless_of_interrupt_threshold` |
| AI 还在播时抢话 | `PacedAudioSink` 的 `playback_position` → 框架算 `partial`/`skipped`；`CallAgent.on_play_end()` 记 `interrupted(tail=True)` 并给历史打标记 | `test_pipeline_barge_in_while_playing_marks_tail` |
| LLM 阶段抢话 | `_turn_in_flight` + 顺延文本 + `interrupted(stage="llm")` | `test_pipeline_barge_in_during_llm_carries_text` |
| TTS 首块之前被抢话 | 这段 `playback_position == 0` → `_never_played()`：记 `interrupted(reason, text)`、历史里去掉这句 | `test_pipeline_barge_in_before_first_tts_chunk_drops_sentence` |
| `confirm_lead` 拆句 | `llm_node` 用 `FlushSentinel` 切成两段；框架按段播放、按段报进度 | `test_pipeline_confirm_lead_keeps_only_played_first_sentence` |
| 号码兜底 | `ReplySpeaker` 合成失败且一块都没播出时用 `PhraseCache.render_phone()`；`on_tts_fallback` 改历史 | `test_pipeline_confirm_synth_failure_falls_back_to_phone_parts` |
| ASR 失败 / 空转写播兜底 | 提交轮次拿到空转写 → `CallAgent.speak_fallback()` 播 `repeat` | `test_pipeline_asr_failure_plays_fallback` |
| `dialog` 只有一个对象 | `LiveKitCall.dialog` 就是 `CallAgent` 用的那一个，`slots.json` / 页面快照都读它 | `test_pipeline_greeting_and_one_turn` |
| 事件与 T10 一致 | `vad_start` / `vad_end` / `smart_turn` / `turn_end` / `asr_*` / `llm_*` / `tts_*` / `play_start` / `play_end` / `interrupted` / `state_change` / `slot_update` / `call_end` 全部走同一个 `EventLogger` | 上面各条 + `tools/evaluate.py` |
| `turn_done` | `speech_created` 拿到 handle，`wait_for_playout()` 之后发；`skipped` / `interrupted` / `reply` | 上面各条 |
| 接管 | `takeover()` 记事件 + `session.interrupt()`，来电者音频照常录进 `caller.wav` | `test_pipeline_takeover_stops_replies_and_records_caller` |

**播放进度用框架的还是 `PlaybackTracker`**：用框架的。`PacedAudioSink` 按实时节奏播出，
每帧调 `on_playback_progressed()`，段结束/被清空时 `on_playback_finished(playback_position, interrupted)`；
框架的 `forward_generation()` 据此把每段判成 `full` / `partial` / `skipped`，我们只需要
“这段有没有开播、播了多久”，不再需要 `core/playback.PlaybackTracker`（T13/T15 用它是因为
两条链路都把音频一次性推给浏览器，LiveKit 这边框架自己会算）。

---

## 6. 自托管 livekit-server、浏览器与场景客户端

### 6.1 livekit-server（二进制不进仓库）

```bash
brew install livekit
livekit-server --dev --bind 127.0.0.1 --node-ip 127.0.0.1
```

`--dev` 的默认密钥是 `devkey` / `secret`（服务端启动日志里会打印）。这两个值只写在这里，
代码与 `config.yaml` 都不写；后端从环境变量 `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` 读
（新增环境变量已按 AGENTS 规则 2 记进 `docs/progress.md`）。`config.yaml: livekit.url` 默认
`ws://127.0.0.1:7880`。

### 6.2 浏览器（要求 5）

- `web/static/livekit-audio.js`：**唯一新增**的音频模块。
  - AI 音频：`RoomEvent.TrackSubscribed` → `track.attach()` 挂 `<audio>` 播放；
  - 麦克风：`createLocalAudioTrack()`，回声消除／降噪／自动增益与 T14 的 `getUserMedia`
    完全一致（都由页面上的“回声消除”开关控制）；
  - 场景回放／上传 wav：把 16kHz PCM 推进一个 AudioWorklet（代码内联，Blob URL），
    实时输出到 `MediaStreamAudioDestinationNode` 再发布成音轨，
    这样 T14 的 `streamScenarioTurns()` 回放节奏逻辑不用改。
- JS 客户端**放进仓库**：`web/static/vendor/livekit-client.umd.js`，版本 **2.23.0**
  （`https://unpkg.com/livekit-client@2.23.0/dist/livekit-client.umd.js`，610KB）。
  选仓库内而不是 CDN：本机测试不依赖外网、版本与实测一致。
- 面板代码只做了最小改动：`call_started` 多带一个 `livekit: {url, room, token}`；
  `startMic()` / `sendPcmBuffer()` / `stopAudioInput()` 各加一个 livekit 分支；
  `play_start` 也当作“AI 已开口”（livekit 模式下没有 WS 二进制帧）。
- `web.adapter: livekit` 时后端在 `start_call()` 里建房间、签两张令牌（浏览器 `caller`、
  后端 `assistant`），并把 `livekit` 段随 `call_started` 发给页面。
  后端忽略 WS 二进制帧（`LiveKitCall.accepts_ws_audio = False`），音频只走房间。
- **真机麦克风由用户补测**（本机没有可用的麦克风/浏览器自动化）。

### 6.3 场景回放客户端（要求 6）

`tools/lk_client.py`：`LiveKitScenarioClient` 继承 `ws_client.ScenarioRunner`，
只实现 `open()/close()/hangup()/_send_frame()` 四个传输钩子，**测量、节奏、抢话统计与
报告格式全部共用** `ws_client.py` 的那一份（`--out` 的 JSON 结构一致，多一个
`transport` 字段）。事件与 `start`/`hangup` 仍走同一个 `WS /ws/call`，
房间 URL 与令牌从 `call_started` 里取，所以客户端不需要 API 密钥。

WebRTC 比 WebSocket 多出的固定时延：Opus 编解码 + 接收侧抖动缓冲 + `AudioSource` 的
`queue_size_ms=200`。它只影响“客户端听到 AI 开口”的时刻（`latency_ms` 的右端）；
服务端事件里的分段时延（`asr_*` / `llm_*` / `tts_*` / `play_start`）两条链路完全同口径，
所以同机对比时用 `ws_client.py` 的 `latency_ms` 减出差值即可。

**WebRTC 音轨是连续的**：不说话时也在送静音帧，所以客户端要把静音帧丢掉
（`SILENCE_RMS=30`），否则“收到音频 = AI 在说话”的假设不成立——踩过：
首轮时延量成 10ms、每轮都等不到服务端空闲。同理，`gap_ms` 的等待改成
“最后一次真实音频 / `play_end` 之后过了 gap_ms”（云端 TTS 比实时慢时一句话中间会有
超过 `gap_ms` 的空档，只看“没有音频”会在句子中间抢话）。

**`_wait_for_gap()` 的 `target` 必须在循环里重算**（修复）：问候语是一小段一小段到的，
开头只有约 50ms 的音频（RMS 66~274）后面跟着 300ms 空档。`target` 只算一次的话，
这一小段 + `gap_ms` 就把开口时刻定在问候语说完之前，第 1 轮被当成抢话把问候语打断
（旧 artifacts `calls/web_20261027_143*` 里 `duration_ms` 只有 160 / 1060 / 1940 / 2000ms）。
改成每轮重算后，AI 每来一段新音频就把开口时刻往后推，问候语完整播出。
离线回归用例在 `tests/test_lk_client.py`（修复前会失败）。

### 6.4 早期单场景实测（历史记录）

同机跑通了一条场景（`scenarios/s01_delivery.json`，3 轮，云端 TTS、`channel_sim: 开`）：

| 项 | LiveKit 路径 |
|---|---|
| 事件流 | `call_start` / `vad_*` / `smart_turn` / `turn_end` / `asr_*` / `llm_*` / `tts_*` / `play_*` / `turn_done` / `call_end` 全齐，与 T10 同格式 |
| 一轮多段 | 房间路径实测把 `"喂。"` 与 `"张先生在吗？"` 合成**一次** `llm_request` |
| 端到端 P50/P95 | 5.3s / 8.7s（云端 TTS 冷缓存，`cache/` 为空，与 T05 的云端首包 P50 4.4s 同量级） |

**历史状态**：当时还未完成第 8 条要求的三链路同口径对比表（级联 / Pipecat / LiveKit 各跑 `--all`
与抢话变体 5 次）。

**首轮问候语误判抢话（已修）**：客户端 `_wait_for_gap()` 的 `target` 只在循环外算了一次，
问候语开头的一小段音频就能把开口时刻定在问候语说完之前，第 1 轮因此把问候语打断
（`interrupted(tail=true)`，`duration_ms` 只有 160 / 1060 / 1940 / 2000ms）。改成循环内
重算后（与 `ws_client.py` 同写法），本机连跑两次问候语都是完整的 5680ms，
第 1 轮 `vad_start` 落在问候语播完 + `gap_ms`，首轮 `latency_ms` 6706 / 7815ms。
细节与对照表见 `docs/progress.md` 的“首轮问候语误判抢话的修复”。

### 6.5 收尾中的修复（2026-10-27）

问候语由 `LocalTrackPublication.wait_for_subscription()` 等待首个订阅；源码中
`room.py` 的 `local_track_subscribed` 事件完成该 future。`on_enter` 独立调度，
所以 `call_started` 能先发送给客户端；等待超过配置的
`livekit.greeting_subscription_timeout_s` 时记错误并继续播放。无房间时不等待。

之前 LiveKit 的 StreamAdapter 把一轮切成多个重叠段，数字串识别质量差，且手动提交
可能等满 5 秒；现在框架 STT 已关闭，由同一个 TurnDetector 决定整轮边界后一次转写。
管线级 FakeAsr 验证整轮时长、一次请求和 turn_end 到 llm_request 小于 1 秒。


### 6.6 本机三链路最终对比

macOS ARM64、云端 TTS、同一 `cache/tts/cloud`、信道模拟开启；每链路串行 15 通 44 轮及抢话 5 次。详细方法和数据路径见 `docs/progress.md`。

| 链路 | 误切通话 / 额外段 | 抢话成功（tail） | 客户端端到端 P50/P95（ms） | ASR P50/P95（ms，样本） | LLM 首 token P50/P95（ms，样本） | TTS 命中数 / 未命中首包 P50/P95（ms，样本） | LLM timeout / parse / no_done_chunk |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cascade | 5/15；6 | 3/5 | 4408 / 10917（41） | 415 / 762（48） | 570 / 1243（37） | 48；5858 / 8711（8） | 6 / 0 / 0 |
| Pipecat | 6/15；8 | 4/5 | 4303 / 9556（44） | 451 / 766（52） | 655 / 1334（35） | 50；1943 / 2972（9） | 9 / 0 / 0 |
| LiveKit | 5/15；7 | 3/5 | 4343 / 11581（44） | 421 / 707（48） | 616 / 1101（36） | 53；2880 / 8552（8） | 12 / 1 / 0 |

每条链路 15 通、44 个输入轮；客户端时延括号为有效样本数。误切按每通 `vad_start` 超过场景输入轮数计，缓存命中首包为 0。抢话只计 `reason=barge_in, tail=true`；挂断不计。LLM 首 token 取 `llm_done.data.first_token_ms`；超时无 done 的轮次不进入该样本。事件 `llm_first_token` 在流式模式可能记录首句，不直接当成首 token。TTS 未到首包的请求不进入首包样本。

按 `tools.turn_eval.latency_breakdown` 拆服务端事件（估计语音结束为 `vad_end - 600ms`；P50/P95，ms；分位数不可逐列相加）：

| 链路 | 轮次判断 | ASR | LLM 完成 | TTS 首包 | 总等待 | 回复数 / 重做 | smart_turn / stop_secs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cascade | 600 / 2044 | 446 / 790 | 1445 / 5826 | 0 / 7077 | 3922 / 10520 | 41 / 4 | 35 / 6 |
| Pipecat | 650 / 2730 | 461 / 777 | 1496 / 6385 | 0 / 2117 | 3955 / 9235 | 44 / 4 | 39 / 5 |
| LiveKit | 600 / 1973 | 424 / 707 | 1367 / 5697 | 0 / 4047 | 3713 / 11369 | 46 / 2 | 41 / 5 |

**超过 500ms 的差距**：客户端 P95 LiveKit 比 Pipecat 高 2025ms、比 Cascade 高 664ms。服务端总等待 P95 分别为 11369、9235、10520ms；并非同一组轮次，不能把各段分位数相减后当成单一因果。LiveKit 相比 Pipecat 的轮次判断 P95 反而低 757ms、ASR 低 70ms、已完成 LLM 低 688ms，TTS 首包 P95 高 1930ms；LiveKit 有 12 次 LLM 超时，Pipecat 9 次、Cascade 6 次。具体长尾：LiveKit 的 h02_property 为 15639ms，其中轮次判断 1978、ASR 791、LLM 3127、TTS 首包 9740ms；s02_food_delivery 为 13944ms，其中 600、445、6549、6346ms。Cascade 的 s04_sales 为 11346ms，其中 600、462、1541、8742ms；Pipecat 的 h04_lease 为 9736ms，其中 2730、643、5106、1899ms。超时轮缺少 `llm_done`，不混入已完成 LLM 的分位数。云端模型长尾、缓存命中差异、轮次判断和接收链路都影响结果，样本量小且三批串行，不能将差异全部归因框架。

LiveKit 48 个 `turn_end` 对应 48 次 ASR 请求；`turn_end → asr_request` P50/P95 1.5/998.3ms、最大 2649.1ms（相邻轮等待前轮串行结束），`asr_final → llm_request` P50/P95 0.2/0.4ms、最大 0.5ms。原先框架转写固定 5s 等待不再存在。

WebRTC 固定开销在本机单独测一次：服务端 `play_start` 的单调时钟到同机客户端首个 RMS≥30 的有声音频帧为 **354.9ms**（`web_20261027_214347_d64aec`）。路径包含 `AudioSource.queue_size_ms=200`、Opus 编解码、接收抖动缓冲及首帧阈值，无法从这一次测量拆出各自份额；首段静音和调度也会影响。表中保留原始客户端数字、不作精确扣除；若需粗略对照，可将 LiveKit 客户端结果标注约减 355ms，不能把单次值当稳定校准。

LiveKit 抢话逐次：第 1、2 次手机号整轮识别正确，但该轮 LLM 8s 总超时，播 `repeat`，客户端等 40s 未等到确认句后发送“不对”；之后才出现确认句，挂断事件不计抢话。第 3、4、5 次均是确认句期间 `interrupted(reason=barge_in, tail=true)`，成功 3/5，与 Cascade 3/5 同量级，Pipecat 4/5。第 3、4 次抢话轮客户端的 19.53/3.46ms “时延”可能采到确认句残余音频，不用于三链路端到端表。

旧 WSL2 实测（机器不同，不可和本 Mac 数字直接比较）：Cascade 4801/9964ms、Pipecat 3976/9584ms、LiveKit 5953/12916ms，抢话 2/5、4/5、0/5。

最终验证：`uv run pytest` 669 passed、15 deselected；`node --test tests/js/*.test.js` 21 passed；真实服务和房间 `tests/live/test_livekit_live.py` 2 passed；`git diff --check` 通过。

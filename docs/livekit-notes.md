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
`config.yaml` 参数），判出 `turn_end` 后调 `session.commit_user_turn()`：框架会把 STT
冲干净、**等这一轮所有段的转写都回来**，再调 `on_user_turn_completed()`。
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
我们把 `interrupt.min_speech_ms`(300ms) 映射到 `interruption.min_duration`。

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

## 3. 能直接复用 / 必须自写

| 部分 | 结论 |
|---|---|
| ASR（`services/asr.py`） | **直接复用**。`stt.STT._recognize_impl()` 里一次 `transcribe()`，框架自动用 `StreamAdapter` 按 VAD 切段。 |
| TTS（`services/tts.py` / `tts_local.py`） | **直接复用**。`CallAgent.tts_node()` 里 `async for chunk in synth(text, cancel)`。 |
| 预合成话术缓存（`core/phrases.py`） | **直接复用**。`ReplySpeaker.synth()` 先查缓存，命中首包 0。 |
| 对话状态机（`core/dialog.py`） | **直接复用**。接不进 LiveKit 的 `llm.LLM`（固定 JSON + 代码决定流程），写成 `Agent.llm_node()`。 |
| 一轮的判断逻辑（`core/call.py`） | **直接复用** `stream_llm_turn()` / `choose_reply_text()` / `is_partial_phone()` / `template_hint()`。 |
| 历史操作 | **T16 起抽成 `core/history.py`**，`CallSession` / `DialogProcessor` / `CallAgent` 共用（同一份“（被对方打断）”与“从未播出的句子”逻辑）。 |
| 轮次判断（`core/turn.py`） | **直接复用** `TurnDetector`（Silero + Smart Turn + stop_secs）。框架自带的轮次判断在非流式 ASR 上会丢后半句（第 1.3 节），所以用 `turn_detection="manual"` + 自己提交。 |
| 打断计数（`TurnSpeechCounter`） | **直接复用**，`LiveKitCall._maybe_barge_in()` 用它判 300ms 阈值。 |
| 信道模拟（`tools/channel_sim.py`） | **直接复用**。在 `feed()` 里逐段 `ChannelSimulator.feed()`。 |
| 事件流（`core/events.py`） | **直接复用** `EventLogger`；复用 `web/server.py` 的 `ForwardingEventLogger`。 |
| VAD / Smart Turn 模型 | 复用**同一批 ONNX 模型**，但走 `core/turn.py` 而不是框架的插件（框架的 VAD 只用来给 `StreamAdapter` 切段与 `user_state_changed`）。 |
| 文本分段合成（`FlushSentinel`） | **用框架的**：`confirm_lead` 拆句就是两个段，框架保证“下一段在上一段播完前开始合成”，被打断时已播完的段保留。 |
| 每段播放进度 | **用框架的**：`AudioOutput` 的 `playback_started` / `playback_progressed` / `playback_finished(playback_position, interrupted)`。`core/playback.PlaybackTracker` 没用到（框架给得出，见第 5 节）。 |
| 音频 IO | **必须自写**：`QueueAudioInput` / `PacedAudioSink`（不连 livekit-server 时用；房间路径把 sink 写到 `rtc.AudioSource`）。 |
| 占位 LLM / TTS | **必须自写**：`nodes.py`，只为过框架的两处前置检查。 |
| 浏览器音频（WebRTC） | **待做**：面板与事件流复用 T14，只有采集/播放按 LiveKit JS 客户端单独一份。 |

## 4. 踩到的坑（按影响排序）

1. **`Agent` 的 `self._llm` 被自己的属性覆盖**。`CallAgent.__init__` 里写
   `self._llm = services.llm`，而 `Agent.llm` 属性直接返回 `self._llm`，于是
   `_generate_reply()` 的 `elif isinstance(self.llm, llm.LLM)` 为假，整条回复被静默跳过
   （没有任何报错，只有 `speech_created` 之后再无 `llm_node`）。改名 `_llm_client` 解决。
2. **自定义 `stt.STT` 子类里的 `self._events` 覆盖了 `rtc.EventEmitter._events`**。
   日志器一存进去，`STT.on("metrics_collected", ...)` 就抛
   `TypeError: argument of type 'ForwardingEventLogger' is not iterable`。改名 `_event_log`。
3. **框架的轮次判断配非流式 ASR 会丢一轮的后半句**（第 1.3 节），改用 `manual` 提交。
4. **`say()` 需要 session 上有 `tts`，生成回复需要 `llm`**。用 `llm_node` / `tts_node`
   绕开它们还不够，必须挂占位实现（`nodes.py`），否则问候语抛 `RuntimeError`、
   `on_user_turn_completed()` 之后静默返回。
5. **`rtc.AudioFrame.data` 的 `len()` 是采样数**，不是字节数（第 1.6 节）。
6. **抢话窗口要覆盖“TTS 首块音频之前”**。只按“正在播 / 正在提交轮次”判会漏掉这段：
   `session.commit_user_turn()` 返回时 TTS 可能才刚开始合成。用
   `agent.busy`（`_turn_in_flight or _awaiting_turn_end`）补上。
7. **`commit_user_turn()` 会取消上一次未完成的提交**：两轮挨得近（例如阈值调高、对方在
   AI 说话时说话）时必须自己用 `asyncio.Lock` 串行化，否则前一轮被取消、转写丢掉。
8. **`interrupt.min_duration` 在手动模式下不生效**：`on_vad_inference_done` 对
   `manual` 直接 return。打断要自己判（`TurnSpeechCounter`）后调 `session.interrupt()`。

## 5. 打断与行为对齐（要求 4 逐条）

| 要求 | LiveKit 上怎么做到 | 用例 |
|---|---|---|
| 一轮多段发声不丢后半句 | `turn_detection="manual"` + `commit_user_turn()`（等所有段转写） | `test_pipeline_multi_segment_turn_keeps_second_half` |
| 打断阈值只管打断、不管开口 | 开口由 VAD 的 `min_speech_duration`（0.05s）决定，阈值只用于 `_maybe_barge_in()` | `test_pipeline_short_reply_recognised_regardless_of_interrupt_threshold` |
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

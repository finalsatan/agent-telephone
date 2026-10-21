# AI 电话代接：技术调研与验证方案

整理日期：2026-10-21  
适用目标：开源模型、自有服务部署、中文电话代接、文字记录与通话总结。

本文汇总本次讨论及已查阅的一手资料。模型能力、价格和框架接口可能变化；本文尚未包含真实部署、电话接入或同条件性能实测。文中的架构选择、工时和性能目标属于工程建议，不能视为厂商承诺。

## 1. 需求与当前建议

目标是让 AI 在用户不方便接听时：

- 接听来电，询问来电者身份和来意。
- 按用户预设处理常见问题，记录留言、时间、地点与回拨号码。
- 支持对方插话、修改信息，以及用户接管。
- 生成区分来电者与 AI 的文字对话记录。
- 挂机后输出通话摘要、待办事项及未确认的信息。

部署偏好：本地 GPU 较弱，优先采用可下载权重的模型，在租用 GPU 上部署；目前重点考察 fal.ai。

**第一版建议采用流式 ASR → 文本 LLM → 流式 TTS 的级联方案。** 实时对话链路负责交互，记录持续保存，总结在通话结束后异步生成。端到端语音模型作为体验对照，暂不要求第一版具备所有自然全双工能力。

首先验证电话接入方式，再投入模型对比：浏览器语音 Demo 跑通，不代表已经具备代接手机 SIM 卡来电的能力。

## 2. 电话接入是独立工作

### 2.1 两种产品路径

| 路径 | 主要工作 | 当前状态 |
|---|---|---|
| 直接代接本人手机 SIM 来电 | 接听与挂机控制、读取通话音频、向通话回传 AI 语音、用户接管 | 未确定手机型号、系统版本及可用接口 |
| 通过电话线路／网关代接 | 号码与线路接入、呼叫路由、SIP／媒体流、转接和接管 | 可用网关及运营商条件待核实 |

Android 官方文档说明，读取 `VOICE_CALL`、`VOICE_UPLINK`、`VOICE_DOWNLINK` 音频需要 `CAPTURE_AUDIO_OUTPUT`，该权限保留给系统组件，普通第三方 App 不能直接取得。因此，手机厂商的系统内置代接能力不能等同于普通应用的可实现范围。读取音频与回传音频也需要分别验证。[Android 音频来源文档](https://developer.android.com/reference/android/media/MediaRecorder.AudioSource#VOICE_CALL)

Asterisk 等电话平台能提供另一条验证路径，但不自动包含可用电话号码、运营商线路或个人手机号呼叫转移。应先确认这些接入条件。[Pipecat 的 Asterisk 集成](https://docs.pipecat.ai/api-reference/server/services/serializers/asterisk)

## 3. 小米 AI 代接的公开技术信息

### 3.1 可以确认的事实

| 资料／版本 | 官方披露 | 能支持的判断 |
|---|---|---|
| 2022 年“小爱通话（原 AI 通话）”数据共享说明 | 系统语音引擎 API 提供 NLP、ASR、TTS，涉及 PCM、16kHz 音频 | 早期产品包含可区分的识别、语义和合成组件 |
| 2023 年小爱通话隐私政策 | 语音转文字、通话展示、自定义接听文本／录音、音色设置 | 存在文字辅助接听和预设回复路径 |
| 2026 年超级小爱隐私政策 | AI 帮接识别身份与来意、回复常见来电、生成摘要 | 新版已覆盖主要代接业务能力 |
| HyperOS 官方产品页 | 自动代接一段时间内的来电、总结重点、结合习惯建议回复；超级小爱 2.0 基于 MiMo | 可以确认产品功能及整体模型体系，不能确认每个电话环节的模型 |

来源：[旧版数据共享说明](https://privacy.mi.com/AI-Call-Assistant-sh/zh_CN/)、[旧版隐私政策](https://privacy.mi.com/AI-Call-Assistant/zh_CN/)、[新版隐私政策](https://privacy.mi.com/xiaomihyperxiaoai/zh_CN/)、[HyperOS 官方介绍](https://os.mi.com/)。

### 3.2 尚未确认的内容

- 新版 AI 帮接具体使用哪个 ASR、对话和 TTS 模型。
- 各环节在端侧还是云端执行，以及具体路由方式。
- 新版是否采用原生端到端、全双工语音模型。

旧版政策中的“自定义设置保存在本地”，不能据此推断全部语音推理都在本地。公开的超级小爱整体模型介绍，也不足以确定电话代接的完整实现。

小米开源了 MiMo-Audio-7B 系列，可作为额外研究对象；目前没有找到足够证据证明它就是手机 AI 帮接所使用的模型。[MiMo-Audio 官方仓库](https://github.com/XiaomiMiMo/MiMo-Audio)

## 4. 实时语音架构选择

需要区分三种能力：

- **流式输出**：内容未生成完即可开始播放。
- **支持打断**：用户开口后停止生成和播放，转入下一轮。
- **原生全双工**：模型说话时仍持续理解输入，并管理重叠对话。

端到端语音模型不自动等于原生全双工；WebSocket 双向传输也不自动意味着模型具备持续听说能力。

| 架构 | 优点 | 主要取舍 | 当前定位 |
|---|---|---|---|
| ASR → 文本 LLM → TTS | 中间文本可检查；便于记录、规则控制、工具调用；组件可替换 | 需优化轮次判断和多环节流水线 | 第一版主方案 |
| 端到端语音 | 可保留更多声音信息，具有自然表达潜力 | 文本、已播内容与打断状态需要额外管理 | 对照实验 |
| 语音前台＋后台 Agent | 交互与复杂业务分工，可复用已有 Agent | 需同步任务状态和上下文 | 后续复杂业务扩展 |

端到端模型也可能同时输出文字，例如 Qwen3-Omni Instruct。文字记录需求并不强制采用级联；当前选择级联的主要原因是电话任务的可控性、可检查性和组件独立调优能力。[Qwen3-Omni 官方说明](https://github.com/QwenLM/Qwen3-Omni)

商业 API 调研中还涉及 GPT-Live／Realtime、Gemini Live、豆包 Seeduplex、Qwen 实时语音等。这些可作为体验参照，当前不作为开源部署主线。云端 API 能力不能直接套用到同品牌开放权重上。[OpenAI 语音架构](https://developers.openai.com/api/docs/guides/voice-agents)、[Gemini Live](https://ai.google.dev/gemini-api/docs/live-api)、[Seeduplex](https://seed.bytedance.com/en/blog/introducing-seed-full-duplex-speech-llm-attentive-listening-robust-interference-suppression-enabling-more-natural-interaction)、[Qwen 语音选型](https://help.aliyun.com/en/model-studio/s2s-model)

## 5. 开放权重模型比较

### 5.1 重点候选

| 模型 | 能力与部署要点 | 对电话代接的定位 |
|---|---|---|
| GLM-4-Voice-9B | 中英语音理解与生成；流式输出；支持 BF16／INT4 | 较小规模端到端语音基线 |
| Qwen3-Omni-30B-A3B-Instruct | 音频／视频／文本输入，文字和语音输出，包含 Thinker＋Talker | 较大模型的理解与回答质量对照 |
| MiniCPM-V | 主要是图像／视频／文本理解 | 纯电话场景不优先 |
| MiniCPM-o 4.5，9B | 中英实时语音、音色配置、全双工实时 Demo | 优先验证的自然语音交互候选 |
| Step-Audio 2 mini，8B | 中英语音理解与生成，有流式推理和工具调用方案 | 后续补充对照 |

#### GLM-4-Voice

完整系统包含 Tokenizer、9B 语言模型、语音 Decoder 三部分，不能只计算语言模型本身的资源。官方 Demo 是提交录音后生成流式回复，尚不能据此确认原生持续流式输入和完整双工能力。电话服务还需增加轮次控制、取消生成、音频缓冲清理及会话管理。

官方没有给出可直接用于所有模式的完整系统最低显存承诺；可先在 H100 上验证，不预先承诺 24GB 显卡的运行表现。官方还列出 Gradio 流式播放不稳定的问题，应区分模型输出与演示播放器的影响。

代码采用 Apache-2.0；权重采用专门的 glm-4-voice 协议，商业使用需登记，并有协议附带及标识要求。[官方仓库](https://github.com/THUDM/GLM-4-Voice)、[参考代码](https://github.com/THUDM/GLM-4-Voice/blob/main/web_demo.py)、[权重许可](https://huggingface.co/zai-org/glm-4-voice-9b/blob/main/LICENSE)

#### Qwen3-Omni-30B-A3B-Instruct

Instruct 包含 Thinker 与 Talker，可生成文字和语音；Thinking 版本主要输出文字，不可直接替代语音输出版本。A3B 表示激活参数规模，不代表按普通 3B 模型准备显存即可。

官方显存表的一个参考条件是：BF16、Transformers、15 秒视频输入、完整 Instruct，理论最低约 78.85GB。这不是纯电话音频的最低门槛，不能直接据此保证一张 80GB H100 在长上下文或并发下可用。

vLLM-Omni 已提供双工服务适配，但其 Qwen 配置明确区分了“并行输入、可打断的 VAD 轮次”与原生持续流式输入 KV。测试时应分别验证模型能力与运行时实现。权重按指定模型的 Apache-2.0 授权信息核对。

来源：[官方仓库与显存表](https://github.com/QwenLM/Qwen3-Omni)、[模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)、[双工配置](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_duplex.yaml)。

#### MiniCPM-V 与 MiniCPM-o

同一个仓库包含不同系列。V 系列主要面向视觉语言任务；o 4.5 包含语音和全双工能力，应作为电话交互候选。

o 4.5 官方 PyTorch Demo 要求至少 28GB 显存；llama.cpp-omni 量化路径部分模式列出至少 12GB 显存的要求。这些是特定实现与模式的门槛，不代表并发容量。官方列出的限制包括全双工模式发音错误、中英混杂等。当前项目说明代码和权重采用 Apache-2.0。

来源：[V 系列](https://github.com/OpenBMB/MiniCPM-V)、[o 4.5 部署与限制](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_o4dot5_en.md)、[许可说明](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/project_en.md#model-license)。

### 5.2 其他候选

- **Step-Audio 2 mini**：8B，Apache-2.0，有 vLLM 流式部署路径；全双工交互效果仍需验证。[官方仓库](https://github.com/stepfun-ai/Step-Audio2)
- **PersonaPlex 7B**：英语全双工、角色与声音控制；代码与权重许可不同，中文不作为首选。[模型卡](https://huggingface.co/nvidia/personaplex-7b-v1)
- **NemotronLabs VoiceChat 11B**：全双工及工具调用候选，官方说明为英语模型；不纳入首轮中文主线。[模型卡](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
- **Moshi**：全双工语音研究与部署参考，不作为首轮中文产品主线。[官方仓库](https://github.com/kyutai-labs/moshi)

## 6. 建议的级联组件

| 环节 | 首轮候选 | 验证重点 |
|---|---|---|
| 电话入口 | Asterisk＋可用电话线路／网关 | 接听、挂机、双向媒体、接管 |
| 实时编排 | Pipecat；LiveKit Agents 作为备选 | 会话、取消、缓冲、日志、电话适配 |
| 语音活动检测 | Silero VAD | 噪声、静音、插话 |
| 轮次判断 | 本地 Smart Turn | 中文停顿、犹豫、说完判断 |
| ASR | FunASR／Paraformer-zh-streaming | 部分结果、最终结果、号码与地址 |
| 文本 LLM | 按资源选择的小型 Qwen 文本模型 | 短回复、按规则代接、结构化输出 |
| TTS | Fun-CosyVoice3；Qwen3-TTS 作备选 | 真正的文本输入流、音频输出流、首包延迟 |
| 总结 | 复用文本 LLM，挂机后执行 | 来意、确认事项、待办、未确认信息 |

这是候选组合，不代表所有组件已有无需开发的直接适配。尤其要核实 ASR／TTS 服务接口、取消机制和音频采样格式。

FunASR 工具代码与各模型权重分别授权，不能把工具许可当作所有权重的许可。CosyVoice3 与所讨论的 Qwen3-TTS 模型卡标注为 Apache-2.0。

来源：[FunASR](https://github.com/modelscope/FunASR)、[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)、[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[CosyVoice3 模型卡](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)、[Qwen3-TTS 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)。

## 7. 文字记录与总结

实时记录至少包括：通话 ID、发言方、文本、时间、结果状态及必要的音频对应关系。

- ASR 增量结果可用于字幕，最终结果到达后修正。
- AI 的“生成文本”和“已发送／已播放文本”分别管理。
- AI 被打断时，记录打断位置；摘要不能默认整段生成内容都已告知对方。
- 优先分别记录来电者输入与 AI 输出，减少混合音频分离的需要。
- 姓名、号码、地址、时间通过复述确认；必要时挂机后做转写校正。
- 总结基于已确认的完整记录异步生成，不阻塞下一句回复。

建议的结构化摘要：

| 字段 | 内容 |
|---|---|
| 来电者 | 身份／姓名，未知时保留未知 |
| 来意 | 本次电话的目的 |
| 已确认事项 | 双方明确确认的内容 |
| 待办 | 行动及时间要求 |
| 回拨需求 | 是否需要、号码、时间 |
| 未确认信息 | 尚缺或存在歧义的内容 |
| 接管情况 | 用户是否接管、何时接管 |

## 8. 级联时延优化

### 8.1 让各环节重叠工作

1. **流式 ASR**：边听边识别，避免挂机式整段上传。
2. **语义轮次判断**：结合 VAD、语气或文本语义，避免固定等待一两秒，也避免过早抢话。
3. **提前生成**：收到稳定识别结果后预先生成候选；确认轮次结束再播放，输入变化时取消候选。
4. **首个完整短句启动 TTS**：不等待整段回答；避免逐字随意切块导致韵律或数字读法错误。
5. **双流式 TTS**：核实服务既接收增量文本，也能持续返回音频。
6. **固定话术预合成**：问候、询问来意等固定音频可缓存；不能把无信息的占位回复算作任务已响应。
7. **保持模型预热、复用连接**：各组件尽量部署在同地域，减少跨公网往返。
8. **短回复与轻量推理**：代接以简短确认、询问和记录为主；复杂任务另行处理。
9. **总结与校正异步执行**：不放入实时响应关键路径。
10. **清理打断缓冲**：同时停止生成、TTS 和电话网关待播音频，并修正会话历史。

LiveKit Agents 已提供提前生成，且可选择提前 TTS；候选被丢弃会浪费额外计算，收益应以实测为准。[提前生成说明](https://docs.livekit.io/agents/multimodality/audio/#preemptive-generation)

### 8.2 成熟框架与私有化边界

- **Pipecat＋Asterisk**：有双向媒体适配、流控封装和测试示例，适合作为私有化电话原型。[集成文档](https://docs.pipecat.ai/api-reference/server/services/serializers/asterisk)
- **Pipecat Smart Turn**：开源、本地 CPU 推理，支持中文，可结合 Silero VAD 使用。[文档](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview)、[模型仓库](https://github.com/pipecat-ai/smart-turn)
- **LiveKit Agents＋SIP**：有电话接入、轮次处理、提前生成与会话管理。需显式核对所用轮次检测、降噪等组件的运行位置；自托管 Agent 不等于所有默认功能都在本地。[电话文档](https://docs.livekit.io/telephony/)、[轮次检测](https://docs.livekit.io/agents/logic/turns/turn-detector/)、[调优指南](https://docs.livekit.io/agents/logic/turns/tuning/)

### 8.3 测量与目标

首轮可将普通短问答的目标设为：用户停止说话到首段有效回复，P50 约 1 秒、P95 不超过 2 秒。这是建议目标，不是实测或性能保证。

分开测量：轮次等待、ASR 收尾、LLM 首个可播短句、TTS 首包、电话传输与播放缓冲。同时记录抢话率、误打断率、任务完成率和结构化信息准确性。

先做单会话，再做并发。同一张 GPU 上的组件资源争抢、跨地域网络和电话网关缓冲都可能抬高尾延迟。电话采样与编码应按真实线路验证；重采样不能恢复原音频已经丢失的信息。

## 9. fal.ai 部署可行性

### 9.1 产品选择

| 产品 | 特征 | 当前建议 |
|---|---|---|
| Compute | 专用实例、SSH、手动安装和调试、按小时计费 | 有权限且报价合适时，优先用于模型验证 |
| Serverless | 自定义容器／应用、WebSocket、自动扩缩容、按 runner 存活时间计费 | 后续服务封装或替代验证路径 |

官方 Compute 文档列出的单卡实例为 1×H100-SXM，80GB 显存、200GB 内存、1TB SSD；具体账户权限、容量和单价需控制台确认。Serverless 自定义部署需按账户审核开通。

来源：[Compute](https://fal.ai/docs/documentation/compute)、[Compute 接入](https://fal.ai/docs/documentation/compute/quickstart)、[Serverless](https://fal.ai/docs/documentation/serverless)。

Serverless 支持自定义容器、持久 WebSocket，以及直接处理原始 WebSocket 的端点。可将自有模型封装成实时音频服务；模型加载、冷启动、会话时限、断线恢复仍需实测。[实时端点](https://fal.ai/docs/documentation/development/realtime)、[伸缩参数](https://fal.ai/docs/documentation/deployment/scale-your-application)

**在 fal 上运行自有模型属于云端部署，音频会传到 fal，不是完全本地处理。** fal 提供推理资源，不自动提供电话线路和 SIM 来电接入。

### 9.2 初期实验方式

- 本地电脑仅负责界面、录音和播放，不依赖本地强 GPU。
- 第一阶段用一张 GPU，依次运行各方案，避免同时争抢资源影响对比。
- 保持输入录音、问题、上下文和测试指标一致，优先比较原始精度，再研究量化。
- GLM-4-Voice 与 MiniCPM-o 可优先在 H100 上分别验证；级联先用小型文本模型。
- Qwen3-Omni 单独核实显存与部署配置，必要时用更大显存或多卡，不保证首轮单 H100 配置通用。
- 现成的 fal PersonaPlex 实时接口可做英语网络与音频路径预验证，不能替代中文模型评测。[接口文档](https://fal.ai/models/fal-ai/personaplex/realtime/api)

## 10. GPU 预算

### 10.1 单价与计费口径

以下是调研时公开的 Serverless 标准价，不采用有额外条件的最低折扣价。**Compute 的实际租赁单价需在控制台确认，不能直接套用下表。**

| GPU | 显存 | Serverless 标准价／小时 |
|---|---:|---:|
| RTX PRO 6000 | 96GB | $4.00 |
| H100 | 80GB | $4.50 |
| H200 | 141GB | $6.00 |

Serverless 的模型加载、空闲保温和运行等状态都可能计费；Compute 实例运行期间也持续收费。GPU 小时指所有实例计费时间之和，不等于实际对话时长。[公开价格](https://fal.ai/pricing)、[Serverless 计费](https://fal.ai/docs/documentation/serverless/pricing)、[Compute 计费](https://fal.ai/docs/documentation/compute/pricing)

### 10.2 首轮估算

按 H100 参考价 $4.50／小时，单张 GPU、方案依次验证估算：

| 范围 | GPU 时间估算 | 参考 GPU 费用 | 人民币预算换算 |
|---|---:|---:|---:|
| 跑通 MiniCPM-o 并体验中文对话 | 3–5 小时 | $13.50–22.50 | 约 95–160 元 |
| MiniCPM-o＋级联，完成基本对比 | 10–18 小时 | $45–81 | 约 315–570 元 |
| 增加反复调试、长对话、简单并发 | 20–30 小时 | $90–135 | 约 630–945 元 |

时间为工程估算；人民币仅按预算汇率 1 美元＝7 元折算，不是实时汇率报价。上表不含人工、电话线路／网关、税费及可能另收的存储和流量费用，也不包含所有额外模型的完整验证。

**建议首轮预留 $100，约 700 元预算折算。** 先用 $20–25 验证网络、模型加载和基本对话，再投入余下预算做级联和对比。若 Compute 小时价为 P，参考费用为 `P × 实际实例运行小时＋其他费用`。

测试结束后确认资源已释放、计费已停止累计。以 $4.50／小时计算，单 H100 连续保留 24 小时约 $108。

## 11. 分阶段验证与验收

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| 0：电话入口 | 确认 SIM／电话网关路线、获取双向音频、挂机及接管 | 可完成受控测试通话 |
| 1：网络与音频 | 检查目标访问网络、重采样、音频回传、缓冲 | 连续音频稳定，延迟可测 |
| 2：级联主线 | ASR、短回复 LLM、流式 TTS、记录及摘要 | 基本代接任务闭环 |
| 3：交互优化 | 语义轮次、打断、提前生成、话术缓存 | 时延下降且不明显增加抢话／错误 |
| 4：模型对照 | MiniCPM-o、GLM-4-Voice；后续 Qwen3-Omni／Step-Audio | 同样条件下比较任务、体验与成本 |
| 5：稳定性 | 长通话、噪声、网络中断、并发、用户接管 | 明确容量、恢复行为及成本 |

建议的中文测试集覆盖：

- 快递、外卖、预约、陌生来电与一般留言。
- “不是今天，是明天下午三点”等纠正和否定。
- 姓名、长号码、地址、口音、背景噪声。
- 思考停顿、插话、双方同时说话。
- 用户接管、对方提前挂机、网络断开。
- AI 只说完部分回复即被打断，记录和摘要仍准确。
- 信息不足时追问或标记未知，避免擅自确认事项。

## 12. 待确认事项

1. 最终代接本人 SIM 卡来电，还是通过线路／网关接入？
2. 手机型号、系统版本、运营商及呼叫转移条件是什么？
3. fal 账户是否有 Compute／Serverless 权限，实例报价和可用地域是什么？
4. 目标并发、每日通话量、平均通话时长及常驻需求是多少？
5. AI 可以代表用户确认哪些事项，哪些必须转交用户？
6. 记录、音频和摘要保存多久，存储位置及访问方式如何设置？
7. 各指定权重的最终许可、服务适配及部署版本是否已核对？

上述问题会影响实现路径与预算，不能由 GPU 模型选型单独决定。

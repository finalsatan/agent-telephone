# T17 对比与报告：级联自写 / Pipecat / LiveKit Agents

本报告汇总 T05（本地 vs 云端 TTS）、T06（话术缓存）、T12（场景评估）、T13c（轮次参数与实时时延）、
T15／T15b／T15c／T15d（Pipecat）与 T16（LiveKit）的实测数据，给出选型建议、框架对比与下一步。

**每个数字后面的括号里是出处**，格式为 `（progress：小节名）`、`（pipecat-notes：第 N 节）`、
`（livekit-notes：第 N 节）` 或 `（report/xxx.json）`。数字与 `docs/progress.md` 不一致时以 progress
的最终值为准，并在文中注明取的是哪一次。

`report/` 是 `.gitignore` 的，原始 JSON 不在本仓库：

- 主仓库（T05/T12/T13c/T14b/T15c 的批跑）：`/Users/murphy/code/ai/agent-telephone/report/`
- T16 的 Mac 三链路：`/Users/murphy/code/ai/agent-telephone-t16/report/`（`t16_mac_*`）

本文只引用这两处已有的 JSON，不重新计算新数字。

---

## 1. 结论先行

**三条链路的时延几乎一样，选型不该只看时延；差别在“框架替你做了多少”和“接非 WebRTC 信道有多难”。**

同机同口径（T16 Mac + 云端 TTS）的三链路客户端端到端 P50/P95：级联 **4408 / 10917ms**、
Pipecat **4303 / 9556ms**、LiveKit **4343 / 11581ms**（`report/t16_mac_compare.json`）。
P50 三条差不到 110ms；P95 的 2s 差距主要来自上游 LLM 超时次数（6 / 9 / 12 次）和云端 TTS 首包长尾，
不是链路本身（progress：T16 收尾·三链路同机实测）。

关键依据（每条都能在 progress / notes 里找到来源）：

1. **同机 P50 差异 < 110ms**，P95 差异 2s 但可拆到 LLM timeout 与 TTS 首包，不是单一原因
   （`report/t16_mac_compare.json`）。
2. **框架能省的事不一样**：播放进度和“打断阈值 vs 开口阈值分离”，LiveKit 原生就有
   （`livekit-notes：1.4`）；Pipecat 两样都得自己补——T15b 里抽 `PlaybackTracker` 约 1.5h，
   打断阈值与各类兜底合在一起约 2h（`pipecat-notes：第 9 节`）。
3. **轮次判断两个框架都要自己接**：LiveKit 框架自带的 EOU 配非流式 ASR 会丢一轮后半句，
   主路径改成 `turn_detection="manual"` + 项目 `TurnDetector`；Pipecat 用框架的
   `LocalSmartTurnAnalyzerV3`，但要自己补 0.85 阈值（`livekit-notes：1.3`、`pipecat-notes：第 4 节第 4 条`）。
4. **工作量**：Pipecat T15–T15d 约 31.5h、4 轮评审；LiveKit T16 约 21.2h、1 轮评审；
   级联自写的工时没有逐任务记录（见第 8 节）。
5. **槽位准确率与框架无关**：默认 `minimal` 模式在留出场景 39/42（92.9%）、全部 15 条 115/120（95.8%）
   （progress：T12c）。

**选型建议**

| 场景 | 建议 | 理由 |
|---|---|---|
| 要接电话线路（SIP）、要完全掌控轮次与状态机、团队不想背框架依赖 | **级联自写** | 三条链路时延同量级；接 SIP 时框架的 WebRTC/WS transport 用不上，反而要绕开（第 10 节）。代价是播放进度、打断、取消竞态都得自己写并自己测（T13b/T13c 的返工就出在这里）。 |
| 要浏览器/多端 WebRTC、想少写音频链路，但愿意自己补播放进度与打断语义 | **LiveKit Agents** | 播放进度、打断阈值、分句合成是框架原生（`livekit-notes：3`）；代价是要处理框架轮次判断与非流式 ASR 的冲突（T16 最大返工项）。 |
| 要框架的管线抽象、P95 想压到三条里最低 | **Pipecat** | 同机 P95 9556ms，三条里最低（`report/t16_mac_compare.json`）；代价是播放进度、打断阈值、TTS 取消语义都要自己补，且经历了 4 轮评审返工。 |
| 只做原型验证、模型和场景还会大改 | **级联自写** | 槽位准确率、轮次参数的瓶颈都在模型和提示词上（T12c），框架对比在模型定型前收益有限。 |

---

## 2. 测试环境与口径

### 2.1 两套环境（数字不能混着比）

| 环境 | 机器 / TTS | 覆盖的任务 |
|---|---|---|
| A：WSL2 + RTX 4060 Ti + 本地 CosyVoice3 | WSL2 + Ubuntu 22.04、RTX 4060 Ti 16GB（progress：环境基线、T05） | T05/T05b/T05c、T06、T09、T12/T12b/T12c、T13b、T14b 的实时补测 |
| B：Mac（macOS ARM64）+ 云端 TTS | SiliconFlow ASR/LLM + 云端 TTS，同一份 `cache/tts/cloud`（24 条预合成），`channel_sim: 开`（progress：T16 收尾） | T15c 第 7 条（级联 vs Pipecat）、T16 Mac 三链路 |

- **可直接比**：T16 Mac 三链路（同一台机器、同一批 15 场景 44 轮、同一 TTS 后端、同一缓存、串行跑完，
  progress：T16 收尾）；T15c 第 7 条 Mac 云端里的级联与 Pipecat 之间（同一批、同一口径，
  progress：T15c 第 7 条）。但 T15c 与 T16 是**两批不同时间**的实测，跨批的差值不能归因给链路。
- **不能直接比**：WSL2 那批（T14b 级联 P50 3027 / P95 6223ms，progress：T14b；T13b P50 2242.6 / P95 9669.6ms，
  progress：T13b）——机器和 TTS 后端都不同；本地 TTS 那批（T05/T05b/T05c 的首包）与云端 TTS 的首包也不能直接比
  （progress：T14b 明确写了“T14b 的 2.87s 是 WSL2 + 本地 CosyVoice3，两者不可直接比”）。

### 2.2 场景集与抢话变体

- 15 条按轮场景：`s01`–`s10`（原场景）+ `h01`–`h05`（T12c 新增的留出场景），共 **44 个输入轮**
  （progress：T12c、T16 收尾）。
- 抢话变体：`scenarios/barge_in/s03b_barge_in.json`，`turn_01`–`03` 复用 `s03_appointment`，
  `turn_04` 是“不对。”的纠正音频；客户端等含“对吗”的确认句开播后再等 `barge_in_ms=1000` 开口
  （progress：T13b、T13c）。
- 信道模拟：`tools/channel_sim.py`（低通 + G.711 μ-law + 2× 抽取/插值），三链路实测都开
  （progress：T08、T16 收尾）。
- 真人录音 `scenarios/real_01`–`real_05` **仍是占位**，评估自动跳过（progress：T12）。

### 2.3 指标定义

| 指标 | 定义 | 出处 |
|---|---|---|
| 端到端（客户端，`ws_client.py` / `lk_client.py`） | 该轮 wav 的“语音结束点”（20ms 帧 RMS 阈值估出）→ 客户端收到第一块 AI 音频 | `tools/ws_client.py` 文件头；progress：T13b |
| 端到端（离线，`evaluate.py`） | `vad_end → play_start` | `tools/evaluate.py` 文件头；progress：T12 |
| 端到端（服务端拆分，`turn_eval.py --breakdown`） | 估计语音结束（`vad_end − 600ms`）→ 本轮第一个 `play_start` | progress：T16 收尾 |
| 轮次判断 | `vad_end → turn_end` | progress：T14 |
| ASR / LLM 首 token / TTS 首包 | `asr_final.latency_ms` / `llm_done.first_token_ms` / `tts_first_audio.latency_ms`（缓存命中为 0） | progress：T16 收尾 |
| 误切（离线） | 一条 wav（一句话）出现超过一次 `turn_end` | progress：T13c |
| 误切（实时） | 每通 `vad_start` 数超过场景定义的输入轮数；同时列“额外 VAD 段 / 输入轮数” | progress：T16 收尾 |
| 抢话成功 | **只计 `interrupted(reason=barge_in, tail=true)`**；挂断 `hangup` 不计 | progress：T16 收尾 |
| 缓存命中 | `PhraseCache.get()` 命中次数；命中时 TTS 首包恒为 0ms | progress：T16 收尾 |
| P50/P95 样本数 | 表里括号中的 n；未取到首包的请求不进入该样本 | progress：T16 收尾 |

> 注意：客户端端到端与服务端“总等待”不是同一起止点，也不能逐列相减（progress：T16 收尾）。

---

## 3. 时延分段分布

### 3.1 同机三链路（T16 Mac，云端 TTS，44 轮）

客户端（含网络与客户端排队）：

| 链路 | 端到端 P50 / P95（样本） | ASR P50 / P95（样本） | LLM 首 token P50 / P95（样本） | TTS 命中数 / 未命中首包 P50 / P95（样本） | LLM timeout / parse / no_done_chunk |
|---|---|---|---|---|---|
| 级联 | 4408 / 10917（41） | 415 / 762（48） | 570 / 1243（37） | 48；5858 / 8711（8） | 6 / 0 / 0 |
| Pipecat | 4303 / 9556（44） | 451 / 766（52） | 655 / 1334（35） | 50；1943 / 2972（9） | 9 / 0 / 0 |
| LiveKit | 4343 / 11581（44） | 421 / 707（48） | 616 / 1101（36） | 53；2880 / 8552（8） | 12 / 1 / 0 |

（`report/t16_mac_compare.json`；progress：T16 收尾·三链路同机实测）

服务端事件拆分（`tools.turn_eval.latency_breakdown`，P50 / P95，ms）：

| 链路 | 轮次判断 | ASR | LLM 完成 | TTS 首包 | 总等待 | 回复数 / 重做 | smart_turn / stop_secs |
|---|---|---|---|---|---|---|---|
| 级联 | 600 / 2044 | 446 / 790 | 1445 / 5826 | 0 / 7077 | 3922 / 10520 | 41 / 4 | 35 / 6 |
| Pipecat | 650 / 2730 | 461 / 777 | 1496 / 6385 | 0 / 2117 | 3955 / 9235 | 44 / 4 | 39 / 5 |
| LiveKit | 600 / 1973 | 424 / 707 | 1367 / 5697 | 0 / 4047 | 3713 / 11369 | 46 / 2 | 41 / 5 |

（`report/t16_mac_compare.json`；progress：T16 收尾·三链路同机实测）

要点：

- **P50 几乎一致**：端到端 4303–4408ms，服务端总等待 3713–3955ms。
- **P95 的长尾来源不是单一的**（progress：T16 收尾）：
  - LLM 超时/无 done：级联 6 次、Pipecat 9 次、LiveKit 12 次（`report/t16_mac_compare.json`）。
  - 云端 TTS 首包：未命中 P95 级联 8711ms、Pipecat 2972ms、LiveKit 8552ms（同上）。
  - 轮次兜底：`stop_secs` 5–6 次/链路，每次约 2s（同上，`vad.stop_secs=2.0`，config.yaml）。
  - 具体长尾（progress：T16 收尾）：LiveKit `h02_property` 15639ms（轮次判断 1978 + ASR 791 + LLM 3127 + TTS 首包 9740）；
    LiveKit `s02_food_delivery` 13944ms（600 / 445 / 6549 / 6346）；级联 `s04_sales` 11346ms（600 / 462 / 1541 / 8742）；
    Pipecat `h04_lease` 9736ms（2730 / 643 / 5106 / 1899）。
  - 超时轮缺少 `llm_done`，不混入“已完成 LLM”的分位数（progress：T16 收尾）。

### 3.2 T15c 第 7 条：Mac 云端两链路（早一批）

| 链路 | 端到端 P50 / P95（样本） | ASR P50 / P95（n） | LLM 首 token P50 / P95（n） | TTS 命中 / 未命中首包 P50 / P95（n） | LLM timeout / parse / no_done_chunk |
|---|---|---|---|---|---|
| 级联 | 6191 / 11175（42） | 463 / 1735（46） | 1419 / 4094（43） | 52；1990 / 4682（7） | 9 / 1 / 9 |
| Pipecat | 5696 / 10152（44） | 419 / 806（52） | 1230 / 4800（44） | 52；2122 / 6356（8） | 11 / 1 / 8 |

（`report/t15c_live2_cascade_all.json`、`report/t15c_live2_pipecat_all.json`；progress：T15c 第 7 条）

服务端拆分：级联响应 P50/P95 = 5833 / 10804ms，Pipecat 5014 / 9663ms
（`report/t15c_live2_breakdown.json`；progress：T15c 第 7 条）。这一批的 `no_done_chunk` 单列了
（9 / 8），T16 那批为 0（progress：T15c 第 7 条、T16 收尾）——说明上游接口状态在不同批次差别很大，
跨批比较要谨慎。

### 3.3 离线基线（不含轮次判断，供对照）

`evaluate.py` 的离线回放里“轮次判断”恒为 0（音频预先切好），只有 T13 实时链路才有真实值
（progress：T12）。默认 `minimal` 模式：端到端 P50 2001.2 / P95 5365.3ms，CONFIRM 轮 P50 4398.7 / P95 9899.9ms
（progress：T12c）。CONFIRM 轮是离线 P95 的主要来源，因为确认语是现合成的长句
（progress：T12c、T13c、T14b）。

### 3.4 WebRTC 固定开销怎么看

LiveKit 路径本机单独测过一次：服务端 `play_start` 的单调时钟 → 同机客户端首个 RMS≥30 的有声音频帧 =
**354.9ms**（progress：T16 收尾；`livekit-notes：6.6`，call `web_20261027_214347_d64aec`）。
路径包含 `AudioSource.queue_size_ms=200`、Opus 编解码、接收侧抖动缓冲和首帧阈值，**单次测量拆不出各自份额**。
所以：

- 三链路表的客户端数字**保持原样、不精确扣除**（progress：T16 收尾）；
- 需要粗略对照时，把 LiveKit 客户端结果**标注约减 355ms**，但不能当成稳定校准（progress：T16 收尾）；
- 服务端事件（`asr_*` / `llm_*` / `tts_*` / `play_start`）三条链路同口径，不受这项影响
  （`livekit-notes：6.3`）。

---

## 4. TTS：本地 vs 云端

### 4.1 首包与总耗时（同一台 WSL2 机器，不同时段）

| 指标 | 云端 T04 | 本地 T05 | 本地 T05b（hop 复位） | 本地 T05c（可取消） |
|---|---|---|---|---|
| 首包 P50 / P95 | 4358 / 8835ms | 3500 / 5278ms | **2281 / 3220ms** | 2615 / 3565ms |
| 总耗时 P50 / P95 | 5145 / 11522ms | 3529 / 10624ms | 4851 / 11133ms | 4330 / 13221ms |
| 成功 / 次数 | 38/50（12 次超时） | 25/25 | — | — |
| RTF P50 范围 | 0.25–2.14×（按句长） | 0.95–1.33× | 1.06–1.64× | 1.22–1.78× |

（progress：T04 实测、T05 实测、T05b 实测、T05c 实测；句长维度见各节表格）

分句对比（首包 P50，progress：T05 与 T04 对比表）：3 字 1098ms（本地）vs 3593ms（云端）、
7 字 1917 vs 3644、18 字 3500 vs 5833、25 字 4513 vs 8748、49 字 5010 vs 3475（云端那次是突发）。
结论：短句与中等长度本地快 1.7–4.2s；整体首包 P50 只快 0.86s（progress：T05）。

### 4.2 可取消性（打断体验）

| 阶段 | 打断后重说 | 出处 |
|---|---|---|
| T05b（服务端 cancel 不中断推理，跑完才放锁） | P50 **10917ms** / P95 11501ms（正常 7 字首包 P50 2075ms，多约 8.8s） | progress：T05b |
| T05c（服务端逐 token 检查停止开关） | P50 **2910ms** / P95 3525ms，额外等待 96–1201ms | progress：T05c |
| T05c `--stress 20` | 20/20 成功，首包 P50 2939 / P95 3396ms，额外等待 0–1624ms | progress：T05c |

T05b 期间还复现过更严重的问题：cancel 后立刻放锁会让两个请求并发使用同一模型，
抛 `probability tensor contains either inf, nan or element < 0` 并永久卡住（progress：T05b）。
T05c 的 `--stress 20` 无卡死（progress：T05c）。

### 4.3 部署与成本代价

- 本地 CosyVoice3：模型 `Fun-CosyVoice3-0.5B` 9.1GB，conda 环境 Python 3.10，
  启动到 `/health` ok 约 57s（progress：T05）；单服务 baseline 约 8227 MiB，
  模型 + CUDA 上下文约 6.4GB（progress：T05b）；服务端串行一把锁（progress：T05）。
- 云端：无本地 GPU 依赖，但首包 4–9s、P95 8.8s（progress：T04）；按最初 6s 首包超时跑的那一轮
  有 36% 请求超时（38/50 成功），最终把首包/总时限放宽到 10s/30s（progress：T04）。
- **API 单价没有实测数据**，本报告不写成本数字。
- 号码复述时长：T06 `render_phone("13800138000")` 7.50s → T06b 5.21s（−31%，progress：T06b）；
  T07c 起主方案改为整句现合成，`render_phone` 降级为兜底（progress：T06b 第 12 次评审修订）。

---

## 5. 槽位准确率与场景评估

### 5.1 三种 reply 模式（T12b，每条场景 3 遍，共 30 通/模式）

| 项 | full | minimal | slots_only |
|---|---|---|---|
| 必填槽位 | 72/78（92.31%） | **78/78（100%）** | 75/78（96.15%） |
| name | 21/24 | 24/24 | 24/24 |
| purpose | 24/27 | 27/27 | 24/27 |
| callback_number | 27/27 | 27/27 | 27/27 |
| time（选填） | **9/9** | 7/9 | 5/9 |
| location（选填） | 3/3 | 2/3 | 3/3 |
| LLM 总耗时 P50 / P95 | 1845.3 / 2918.5 | 1371.2 / 2159.0 | **1248.9 / 1742.5** |
| 端到端 P50 / P95 | 2335.0 / 5409.6 | 1864.9 / 4805.0 | **1685.7 / 4480.2** |
| 解析失败率 | 0/87 | 0/87 | 1/87（1.15%） |

（progress：T12b 三种模式实测）

### 5.2 去例句泄漏后的结果（T12c，15 条 × 3 遍 = 45 通/模式）

| 项 | full | minimal（默认） |
|---|---|---|
| 必填槽位（全部 15 条） | 110/120（91.7%） | **115/120（95.8%）** |
| 原场景 s01–s10 必填 | 72/78（92.3%） | 76/78（97.4%） |
| **留出场景 h01–h05 必填** | 38/42（90.5%） | **39/42（92.9%）** |
| name（原 / 留出） | 20/24 + 11/12 | 24/24 + 12/12 |
| purpose（原 / 留出） | 25/27 + 12/15 | 25/27 + 12/15 |
| callback_number（原 / 留出） | 27/27 + 15/15 | 27/27 + 15/15 |
| time（选填，原 / 留出） | 6/9 + 5/9 | 4/9 + 2/9 |
| location（选填，原 / 留出） | 1/3 + 3/3 | 2/3 + 2/3 |
| 解析失败率 | 5/132（3.8%） | **0/132** |
| 缓存命中率 | 135/177（0.763） | 135/177（0.763） |

（progress：T12c 实测）

- 默认 `reply_mode: minimal`（config.yaml），留出场景 92.9% > full 的 90.5%，解析失败 0 vs 3.8%
  （progress：T12c）。
- T12b 的 100% 偏乐观：去掉提示词例句与场景原文的重叠后，原 10 条降到 full 92.3% / minimal 97.4%
  （progress：T12c）。
- 主要失败点（minimal，progress：T12c）：`h01_aircon` purpose 0/3、time 0/3；`h02_property` time 0/3；
  `h04_lease` time 1/3、location 1/3。模糊时间（“月底前”“明天傍晚”）和非“上午/下午+点”的说法最不稳。
- 早期 T12 的三种实验（full 21/27、minimal 19/27、confirm_lead 23/27，progress：T12）已被 T12b/T12c
  取代；报告以 T12c 为最终值。
- 摘要 + 复核 ASR 是批跑里最慢的一段：单条通话合计 **30.25s**（progress：T12），建议只对含姓名/号码的场景做。

---

## 6. 轮次与打断

### 6.1 参数扫描（T13c，44 条场景 wav）

基线（静音确认 200ms / Smart Turn 阈值 0.5，改动前的默认值）：

| 信道模拟 | 静音确认 | 阈值 | 误切 | 误切率 | 兜底次数 | 结束延迟 P50 | P95 |
|---|---|---|---|---|---|---|---|
| 关 | 200 | 0.5 | 31/44 | 70.5% | 7 | 312ms | 3091ms |
| 开 | 200 | 0.5 | 35/44 | 79.5% | 7 | 316ms | 3103ms |
| 关 | 600 | 0.85 | **3/44** | **6.8%** | 8 | 694ms | 3099ms |
| 开 | 600 | 0.85 | **4/44** | **9.1%** | 7 | 700ms | 3099ms |

（progress：T13c 参数扫描表）

- 误切主因是静音确认时长：场景 wav 逗号处停顿多在 400–600ms，Smart Turn 在这些地方给 0.9–0.99 的
  “已说完”概率，**只调阈值压不住**（200ms 档阈值 0.5→0.85，信道开时误切率只从 79.5% 降到 61.4%，
  progress：T13c）。
- 没有任何组合同时满足“误切率 ≤10%（开、关都要）且 P50 比基线多不超过 300ms”；600/0.85 超出预算约 80ms
  （progress：T13c）。
- 阈值 0.85 的副作用：句尾概率 0.74–0.82 的会被拒、改走兜底，兜底次数从 5 增到 8（信道关，
  progress：T13c）。
- **结束延迟 P95 ≈ 3.1s 全部来自 `stop_secs=3.0` 兜底**；`stop_secs` 2.0 → P95 2107ms、1.5 → 1595ms
  （progress：T13c）。

### 6.2 当前取值（config.yaml 最终值）

| 参数 | 值 | 来源 |
|---|---|---|
| `vad.speech_end_silence_ms` | 600 | progress：T13c 方案 C |
| `vad.smart_turn_threshold` | **0.85** | progress：T13c 方案 C（最终值，没有采用过 0.90） |
| `vad.stop_secs` | **2.0** | progress：T14b（第 23 次评审建议，从 3.0 改） |
| `interrupt.min_speech_ms` | 300 | config.yaml；progress：T13b |

### 6.3 实时误切与抢话

| 项 | T13b（WSL2+本地，旧参数 200/0.5/stop 3.0） | T14b（同机，600/0.85/stop 2.0） |
|---|---|---|
| 被 VAD 切多段／触发抢话 | 22/29 | **3/29** |
| 时延 P50 | 2242.6ms | 2866.2ms（+623.6） |
| 时延 P95 | 9669.6ms | **5491.4ms（−43%）** |

（progress：T13b、T14b 实测）

抢话（只计 `barge_in, tail=true`）：

| 场景 | 结果 | 出处 |
|---|---|---|
| T13b/T14b 抢话变体（WSL2+本地） | T14b 5/5，开口→`interrupted` 431.77 / 431.24 / 431.63 / 431.58 / 430.89ms | progress：T14b |
| T15c 第 3 轮 Pipecat（云端，短音频） | 2/5，延迟 414.28 / 419.65ms | progress：T15c 第 3 轮 |
| T15c 第 2 轮长音频变体 | 级联 1/5、Pipecat 4/5 | progress：T15c 第 2 轮 |
| T16 Mac 三链路 | 级联 **3/5**、Pipecat **4/5**、LiveKit **3/5** | `report/t16_mac_compare.json`；progress：T16 收尾 |

- 抢话变体成功率主要受“目标确认句是否及时出现”限制，不是适配层的抢话逻辑问题
  （progress：T15c 第 2/3 轮、T16 收尾）。
- 误打断（短噪声不打断）有管线级用例覆盖：级联 `test_ws_short_noise_does_not_interrupt`（progress：T13）、
  Pipecat 稀疏噪声不跨段累计（progress：T15c 第 4 轮）、LiveKit 播放中 150ms 短声不打断
  （progress：T16 第 30 次评审修订）。
- 误切率验收用例的上限是 5/44（`int(44*0.10)+1`），留一条余量（progress：T14b）。

---

## 7. 缓存命中率

| 批次 | 命中 / 总数（命中率） | 出处 |
|---|---|---|
| T12 full | 32/39（0.821） | progress：T12 |
| T12 minimal | 30/39（0.769） | progress：T12 |
| T12 confirm_lead | 39/47（0.830） | progress：T12 |
| T12b（三种模式各 30 通） | 90/117 | progress：T12b |
| T12c（两种模式各 45 通） | 135/177（0.763） | progress：T12c |
| T16 Mac 级联 / Pipecat / LiveKit | 48 / 50 / 53 次（未命中首包 P50/P95 见 3.1 表） | `report/t16_mac_compare.json` |

- 缓存命中时 TTS 首包恒为 0ms（progress：T16 收尾）。
- 未命中即现合成，是 CONFIRM 轮和 P95 长尾的主要来源（progress：T12c、T13c、T14b）。
- 预合成开销：T06 首次 21 条 123.1s；T06b 63.2s；T10 23 条 75.5s；T15c 云端 21 条 37.2s
  （progress：T06、T06b、T10、T15c）。

---

## 8. 开发工作量

### 8.1 级联自写（T00–T14b）

progress 只按任务追加小节，**没有逐任务的小时数**，因此这里给可核对的规模指标，不编工时：

| 项 | 值 | 出处 |
|---|---|---|
| 任务/修订小节数（T00–T14b，不含 T15 起的框架适配） | 29 个 | progress 的二级标题 |
| 提交数（`T*:` 前缀，T00–T14b） | 44 个 | `git log --format=%s` 按前缀统计 |
| 有记录的实测耗时 | 预合成 63.2–123.1s、复核 ASR 单条 9.04–30.25s、场景生成约 3 分钟 | progress：T06/T06b/T10/T11/T12 |

### 8.2 Pipecat（T15–T15d，4 轮评审）

| 阶段 | 用时 | 出处 |
|---|---|---|
| T15 接入（读源码 4h、STT/TTS 3h、DialogProcessor 3h、管线与 T13 对接 3h、live 联调 4h、文档 2h） | 约 19h（约 1.5 天） | `pipecat-notes：第 2 节` |
| T15b 修订（播放进度 1.5h、共用函数 2h、打断阈值与兜底 2h、事件 1h、单测与对比 1.5h、定位 2h、文档 0.5h） | 约 10.5h（约 1.3 天） | `pipecat-notes：第 9 节` |
| T15c 帧顺序修复 | 约 0.37h（会话墙钟） | `pipecat-notes：第 10 节` |
| T15c 第 1 轮评审修订 | 约 7 分钟 | progress：T15c 第 1 轮评审修订 |
| T15c 第 7 条云端实测 | 约 1.2h | `pipecat-notes：第 10 节`（实测工作量） |
| T15d 首块前抢话记录 | 约 0.45h | `pipecat-notes：第 12 节` |
| **合计** | **约 31.5h** | 上表相加 |
| 评审轮数 | 4 轮（第 26、27、28、29 次） | 开发任务拆分.md 评审记录 |

### 8.3 LiveKit（T16，1 轮评审）

| 阶段 | 用时 | 出处 |
|---|---|---|
| 读文档与源码 5h、STT/TTS/占位节点 2h、CallAgent 3h、LiveKitCall 2h、管线级用例 5h、文档 1h、首轮问候语修复 1h | 约 19h | `livekit-notes：第 2 节` |
| 收尾：整轮 ASR / 阈值与管线回归 0.5h、问候语等待与真实房间测试 0.3h、三链路串行实测与整理 1.4h | 约 2.2h | `livekit-notes：第 2 节` |
| **合计** | **约 21.2h** | 上表相加 |
| 评审轮数 | 1 轮（第 30 次） | 开发任务拆分.md 评审记录 |

> `livekit-notes：第 2 节`的表头写“约 1.5 天”，与分阶段表相加（约 19h）口径不一致；
> progress：T16 收尾用的是“先前工作量表已约 19 小时”，本报告取 **19h**。
> 第 30 次评审修订（恢复历史、补用例、删 `stt.py` 等）的工时没有记录。

### 8.4 框架原生 / 必须自写对照（要求 4 的核心）

| 能力 | 级联自写 | Pipecat 1.13 | LiveKit Agents 1.8 | 记录的时间 |
|---|---|---|---|---|
| 轮次判断（Silero + Smart Turn + `stop_secs`） | 自写 `core/turn.py` | 用框架 `LocalSmartTurnAnalyzerV3`，但 0.85 阈值要自补（子类覆写 `_predict_endpoint`） | 框架有 EOU 但配非流式 ASR 丢后半句 → 改用 `turn_detection="manual"` + 项目 `TurnDetector` | LiveKit 整轮 ASR 接线约 0.5h（`livekit-notes：第 2 节`） |
| 播放进度 | 自写 `core/playback.PlaybackTracker` | **必须自写**：复用 `PlaybackTracker` | **框架原生**：`AudioOutput` 的 `playback_started/progressed/finished` | Pipecat 抽 `PlaybackTracker` 约 1.5h（`pipecat-notes：第 9 节`） |
| 打断阈值 vs 开口阈值分离 | 自写 `TurnSpeechCounter` + 页面阈值 | **必须自写**：T15c 用独立 `InterruptVADProcessor` | **框架原生**：开口 `min_speech_duration` 与打断 `interruption.min_duration` 分离；手动模式下仍需自己判 | 计入 Pipecat T15b/T15c；LiveKit 未单列 |
| 一轮多段发声不丢后半句 | 整轮音频一次 ASR（原生） | **必须自写**：段编号屏障，等全部转写/失败 | 框架 `StreamAdapter` 会分段导致数字串截断 → 改成整轮 ASR（同级联） | LiveKit 约 0.5h；Pipecat 计入 T15c 22min |
| LLM 阶段抢话文本顺延 | 自写 | **必须自写**：每轮独立 cancel + `_carried_text` | **必须自写**：`_turn_in_flight` + 顺延 | Pipecat 抽共用函数 2h 的一部分 |
| TTS 首块音频之前被抢话 | 自写 | **必须自写**：`started_audio` + `on_tts_unplayed` | **框架给进度**：`playback_position == 0` 判“从未播出” | Pipecat T15d 0.45h |
| 文本分段合成（`confirm_lead`） | 自写拆句 | 自写推两个 `Speak` 帧 | **框架原生** `FlushSentinel` | 未单列 |
| 事件流（T10 格式） | 原生 | **必须自写** `TurnEventLogger` | **必须自写**（走同一个 `EventLogger`） | 未单列 |
| `turn_done` 语义 | 自写 | **必须自写** | **必须自写** | 未单列 |
| 音频 IO / transport | 自写 WS | 框架给 WS 管线 | 框架给 WebRTC；不连 server 时自写 `QueueAudioInput` / `PacedAudioSink` | LiveKit 2h（STT/TTS/音频 IO 的一部分） |
| 场景回放客户端 | 自写 `ws_client.py` | 复用 `ws_client.py`（服务端换适配器对客户端透明） | **必须自写** `lk_client.py`（继承 `ScenarioRunner`，只写 4 个传输钩子） | 未单列 |

---

## 9. 遇到的问题（按影响排序）

1. **云端 TTS 首包 4–9s、P95 8.8s**（按 6s 首包超时跑的那一轮 36% 请求超时，最终放宽到 10s/30s），
   不能等云端 TTS，必须靠预合成 + 模板兜底（progress：T04）。
2. **本地 CosyVoice3 首包 3.5s、0.8s 目标未达**；hop 复位后降到 2.28s，但短句仍 1.5–2.8s
   （progress：T05、T05b）。
3. **本地 TTS 打断后重说一度 10.9s**（cancel 不中断推理、跑完才放锁）；T05c 改成服务端可取消后
   降到 P50 2.9s，但仍有 0–1.6s 波动（progress：T05b、T05c）。
4. **上游 LLM 超时/长尾是 P95 的主要来源之一**：T16 Mac 三链路 6/9/12 次 timeout，超时后播 `repeat`
   （`report/t16_mac_compare.json`；progress：T16 收尾）。
5. **轮次误切**：200ms 静音确认下逗号停顿被当句末（70–80%），只调阈值压不住，只能加大静音确认
   （progress：T13c）。
6. **`stop_secs` 兜底造成约 3s 长尾**（Smart Turn 给低概率的句尾要等满）；降到 2.0 后 P95 明显下降
   （progress：T13c、T14b）。
7. **场景 wav 被 VAD 按内部停顿切成多段**，与离线“按轮回放”不一致，22/29 轮触发抢话，文本顺延后
   LLM 一次要处理更长输入（progress：T13b）。
8. **Pipecat 的帧时序与自写逻辑接缝**：一轮多段转写晚到、开口阈值与打断阈值耦合、LLM 取消竞态、
   `dialog` 对象分叉、号码上下文被无关帧清空、不发 `turn_done`——6 个阻塞项，全部要自己补
   （progress：T15c 第 27 次评审）。
9. **Pipecat TTS 等首包时被抢话不记 `interrupted`**，历史却记成整句已说（progress：T15d）。
10. **LiveKit 框架 STT 分段把一轮切细、数字串截断**，还连带把 `commit_user_turn()` 拖到 5s 上限；
    改成整轮 ASR 后 48 个 `turn_end` 对应 48 次 ASR，`asr_final → llm_request` P95 0.4ms
    （progress：T16 收尾；`livekit-notes：6.5`）。
11. **LiveKit 的 `Agent._llm` 被自己的属性覆盖**，框架静默跳过整条回复、没有任何报错
    （progress：T16 遇到的问题 1）。
12. **抢话变体成功率受“目标确认句是否及时出现”限制**，不是抢话逻辑问题：T16 Mac 级联 3/5、
    Pipecat 4/5、LiveKit 3/5（progress：T15c 第 2/3 轮、T16 收尾）。
13. **本地 TTS 服务 50000 端口卡死**（`/health` 正常、`POST /synth` 返 200 但 0 字节），
    疑似取消后锁未释放，需 `kill -9` 重启（progress：T06b 第 12 次评审修订、T10）。
14. **Smart Turn 单次推理 90–150ms**，每次都超过 20ms 阈值，必须 `asyncio.to_thread` 转线程，
    否则卡事件循环（progress：T13c、T13b）。
15. **WebRTC 固定开销约 355ms**（单次实测，拆不出份额），客户端数字不能与 WS 链路精确对齐
    （progress：T16 收尾）。

---

## 10. 下一步（按优先级）

1. **接电话线路（SIP）**（最高）：三条链路目前都以 16kHz PCM 为界，接 SIP 网关后都能复用
   （`core/call.py` 的 `feed()` / `LiveKitCall.feed()`）。为什么优先：这是产品的真实入口，
   也决定后续并发方案；框架的 WebRTC/WS transport 在 SIP 场景用不上，反而要绕开。
2. **并发与多路通话**：现在 `AppState` 单通话、本地 TTS 服务端一把串行锁（progress：T05、T13）。
   为什么：真机上线必须多路；先做请求队列 + 过期丢弃，再考虑多实例。
3. **真人录音（T14c）与轮次参数复核**：`real_01`–`real_05` 仍是占位，600ms / 0.85 有过拟合 TTS
   停顿的风险（progress：T13c、T14c 任务书）。为什么：误切率与 `stop_secs` 截断都还没在真人语速上验证。
4. **回声消除 / 真机麦克风补测**：T14b、T15、T16 都没做真人麦克风与外放回声验收
   （progress：T14b 要求 8 f、T15/T16 未做）。为什么：外放会让 AI 被自己的声音打断；
   LiveKit 路径由 `getUserMedia({echoCancellation,...})` 控制，需要实测确认。
5. **抢话变体测试方法**：改为“按实际播放的句子触发”，而不是等固定 `barge_in_ms`
   （progress：T15c 第 28 次评审非阻塞项）。为什么：确认语时长会变，固定等待会落空。
6. **上游模型与提示词优化**：
   - 号码复述 5.21s 仍偏慢（T06b），确认语整句现合成是 CONFIRM 轮 P95 的主要来源
     （progress：T12c、T13c、T14b）；可考虑先播固定开头、分句合成。
   - `h01_aircon` 的 purpose、模糊时间（“月底前”“明天傍晚”）提取不稳（progress：T12c）。
7. **LiveKit 自带 turn detector（选做）**：限时未做，同一批场景多跑一行误切率和结束延迟
   （progress：T16 收尾）。为什么：可能改善轮次判断，但它是云端推理，与“不调 LiveKit Cloud”的约束冲突，需先确认。
8. **LiveKit Cloud / Egress 录制**：T16 明确不做（progress：T16 任务书“不做”）。为什么：产品化需要录音与多端接入，
   但要先确认是否接受云端依赖与成本。
9. **`ttfs_p99_latency` 与 `smart_turn` 事件**：Pipecat 路径拿不到 `smart_turn` 推理耗时，
   启动时有 `ttfs_p99_latency` warning（`pipecat-notes：第 4 节第 7/8 条`、第 8 节）。为什么：影响 STT 兜底超时精度与可观测性。
10. **报告自身的复现**：`report/` 不入库，三链路批跑的命令与参数只在 progress/notes 里；
    如果要做长期趋势，建议把汇总 JSON 单独归档一份（不进 git 也可放共享盘）。

---

## 附：主要来源索引

| 内容 | 位置 |
|---|---|
| 任务进度与所有实测数字 | `docs/progress.md` |
| Pipecat 接入、坑、工作量 | `docs/pipecat-notes.md` |
| LiveKit 接入、坑、工作量 | `docs/livekit-notes.md` |
| T16 Mac 三链路原始汇总 | `/Users/murphy/code/ai/agent-telephone-t16/report/t16_mac_compare.json` |
| T16 Mac 各链路客户端报告 | `/Users/murphy/code/ai/agent-telephone-t16/report/t16_mac_{cascade,pipecat,livekit}_{all,barge_1..5}.json` |
| T15c Mac 云端两链路 | `/Users/murphy/code/ai/agent-telephone/report/t15c_live2_{cascade,pipecat}_all.json`、`t15c_live2_breakdown.json` |
| T15c 抢话逐次报告 | `/Users/murphy/code/ai/agent-telephone/report/t15c_{review2,review3}_*` |
| 评估口径 | `tools/evaluate.py`、`tools/turn_eval.py`、`tools/ws_client.py` |

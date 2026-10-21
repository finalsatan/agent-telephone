# AGENTS.md（给开发用的 coding agent）

项目：AI 电话代接的级联方案（VAD → ASR → LLM → TTS）原型。先读 `级联方案实现设计.md`，再按 `开发任务拆分.md` 的顺序逐个任务实现。

## 工作规则
1. **一次只做一个任务**（T00、T01…）。做完并通过该任务的验收命令后再提交，提交信息以任务号开头，如 `T03: add qwen3-asr client`。
2. **不要擅自改动接口**。`core/types.py` 里的数据类和 `services/` 的抽象接口是各任务之间的约定；确需修改，先在 `docs/progress.md` 记录原因，再改，并同步更新所有调用方和测试。
3. **密钥只从环境变量读取**：`SILICONFLOW_API_KEY`。不得写进代码、配置、日志、测试或提交。`.env` 已在 `.gitignore`。
4. 配置（模型名、base_url、阈值）放 `config.yaml`，代码里不硬编码。
5. 每个任务都要有测试。不调用网络的单元测试用 fake／mock；需要真实 API 的放在 `tests/live/`，用 `pytest -m live` 才运行，默认不运行。
6. Python 3.12（用 uv 管理 `.venv`，`uv run <cmd>` 执行），依赖写入 `requirements.txt`，用 `pytest` 测试。遇到不确定的第三方 API 用法，先查官方文档或包源码，不要凭记忆猜；查不到就在 `docs/progress.md` 记录问题并跳过该子项。
7. 每完成一个任务，在 `docs/progress.md` 追加：任务号、完成情况、遇到的问题、实测数据（如果任务要求）。
8. 不要创建超出任务要求的文件或功能（例如不要提前做 LiveKit、不要做账号系统）。
9. 事件时间一律使用 `time.monotonic()` 换算的毫秒；音频一律 16kHz、单声道、16-bit PCM 小端，除非任务另有说明。

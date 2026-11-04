# 比赛球员库与事件管理

接收上游轨迹、人物特征和事件证据，建立并持续维护比赛内球员身份，提供按球员检索的事件时间线。
同一场比赛可以有任意多批片段、逐步出现的球员、晚到的事件和人工修正。

当前状态：**契约模型、配置读取、sample_test 只读输入适配、核验 CLI 和身份核心算法层 v1 已实现；
normalized 适配、业务工作流、数据库与展示仍待实现。身份算法只输出诊断及计划，尚无已校准策略。**

## 先看业务，再看目录

1. [流水线](docs/pipeline.md)：你的输入、初始化、增量更新、事件关联与输出。
2. [架构](docs/architecture.md)：每一步放在哪个模块。
3. [数据契约](docs/data-contracts.md)：比赛、批次、身份段、特征空间和事件。
4. [实施计划](docs/implementation-plan.md)：逐个闭环开发和验收。
5. [AI 协作约定](AGENTS.md)：后续修改要保持的边界。
6. [身份算法与归人决策](docs/identity-policy.md)：v1 距离、评分、拒识和人工维护规则。
7. [身份核心调用](docs/identity-core.md)：纯算法输入、结果、人工上下文和校准边界。

## 输入边界

本模块接收上游计算结果，不要求重新运行追踪、ReID 或动作模型。
规范接入使用标准 EvidenceBatch；不同同伴文件格式由 adapters 转换。
人物证据和事件可以分别提交，事件类别不限定为进球。

身份作用域是 match_id；一次交接使用 batch_id。换批次沿用该比赛已有球员 ID。
跨比赛永久身份合并不作为默认能力。

## 样例与运行目录

`../test/` 是第一份验收样例，不是产品范围或内部数据格式。
样例说明在 [examples/sample-test.md](examples/sample-test.md)，接入配置在 examples/sample-test.toml。
`../repository-review/` 是只读上游审查快照，不是运行依赖。
所有数据库、预览、日志和导出写 var/，不进入 Git。

## 环境

使用 Python 3.11 和 uv，依赖以 uv.lock 为准：

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

config.example.toml 定义通用配置，examples/sample-test.toml 是样例配置；两者均可由
`app.settings.load_settings` 读取。所有相对路径以配置文件所在目录解析，配置加载不会创建目录、
打开数据库或修改输入，也不要求输入目录已经存在。

在项目根目录可运行：

```bash
uv run python - <<'PY'
from app.contracts.evidence import EvidenceBatch
from app.settings import load_settings

settings = load_settings("config.example.toml")
print(settings.paths.input_root)

batch = EvidenceBatch(match_id=settings.match.match_id, batch_id="handoff-1", producer="example")
assert EvidenceBatch.model_validate_json(batch.model_dump_json()) == batch
PY
```

契约使用 Pydantic 2；字典/JSON 读取入口为 `model_validate` / `model_validate_json`，
输出为 `model_dump(mode="json")` / `model_dump_json`，字段及边界见数据契约。
上述示例只读取配置并构造模型，不执行导入。

只读核验同伴输入：

```bash
uv run python -m app.presentation.cli validate --config examples/sample-test.toml --batch-id sample-check
```

可重复提供 `--clip-id` 选择片段，例如 `--clip-id 20 --clip-id 68`。
CLI 仅调用配置和适配器，输出数量及警告，不创建数据库、球员或运行目录。
文件缺失、哈希不符、数组与索引错行、非法范围和越界路径会使核验失败并返回非零退出码。
Python 调用为 `load_sample_test(settings, batch_id="sample-check", clip_ids=None)`，返回现有
`EvidenceBatch`；完整规则见 [样例说明](examples/sample-test.md)。

身份分支按已确定的 v1 策略实施：人工确认种子，KPR 多样本/多模板推荐，证据不足保留待复核。
纯算法链已实现，正式归人提交与校准尚未完成；当前配置继续关闭自动归人。
可运行 `uv run python -m examples.identity_preview` 查看无文件输入的合成预览，输出 JSON，
不会建库或创建球员。实际输入调用见 [身份核心调用](docs/identity-core.md)。

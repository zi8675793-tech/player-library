# player-library 协同开发指南 (Agent & Vibe Guidelines)

本文件是 `player-library` 项目的核心开发纪律与交互规范。进入本项目的任何 AI 开发者或工程师必须严格遵守。

---

## 1. 唯一事实来源 (Single Source of Truth)

- **当前状态与基准**：一切工程进度、已验证能力与基准跑分**唯一记录在 `docs/current-state.md`**。历史旧文档和测试日志不代表当前达标状态。
- **数据结构与契约**：所有跨模块传递的对象必须严格遵照 `docs/data-contracts.md`。**严禁在修改算法时顺便修改契约字段**。
- **任务规划**：所有执行任务拆解见 `docs/implementation-plan.md`，按阶段分工推进。

---

## 2. 核心架构与技术栈

- **语言与环境**：Python 3.11，使用 `uv` 管理依赖与虚拟环境。
- **架构主线**：
  - 上游输入适配器 (`app/adapters/`)：读取 `test/track/` 轨迹、人物抠图与视频。
  - 双流特征抽取 (`app/features/`)：球衣号码识别 (OCR) + 运动人体 ReID。
  - 核心身份引擎 (`app/identity/`)：时空硬约束网络 (Cannot-Link) + 多原型图库比对 + 增量三态决策。
  - 工作流与事务 (`app/workflows/`)：Revision R 状态机演进与原子写操作。
  - 仓储层 (`app/storage/`)：SQLite 驱动，不可变操作与审计日志。
  - 独立评分防火墙 (`app/evaluation/`)：冻结产物核验后，独立运行评分。

---

## 3. Vibe Coding 研发纪律（防止死循环与破坏式修改）

为了防止大项目中“改动 A 破坏 B、越改越乱”的死循环，执行任务时必须严格执行以下原则：

1. **单轮单一职责 (Single Responsibility per Turn)**：
   - 每次对话只聚焦一个明确的模块（如：只写 OCR 特征提取器，或者只优化时空排他矩阵）。
   - 严禁一次性提出“重构整个流水线并把评分提到 80%”的跨多层复合任务。
2. **测试牢笼原则 (Test-Driven Safety)**：
   - 编写或修改算法前，先锁定对应的单元测试或冒烟测试。
   - 交付条件必须包含：`uv run pytest <target_test_file>` 全绿通过，并且 `uv run ruff check .` 无报错。
3. **推理零作弊 (Zero Ground Truth During Inference)**：
   - `test/annotations.csv` 和 `test/target/` 是期末考试的秘密答案，算法推理代码严禁导入、读取或反向引用这些路径。
   - 特征中的球衣号码必须由算法模型（如 OCR）从 `test/track/*/samples/*.png` 自动推断。
4. **产物留痕与交接 (Immutable Artifacts & Handoff)**：
   - 所有实验运行、数据库生成的产物必须存放于 `var/<task-id>/run-<timestamp>/` 目录，严禁覆盖历史运行。
   - 任务完成后，在对应运行目录下生成 `handoff.txt`，写明交付内容、复现命令、测试结果及下一个任务入口。
5. **Git 存盘点纪律**：
   - 每次任务通过测试后必须立刻 Commit 记录。
   - 若当前轮次 AI 出现幻觉或过度重构，果断回滚到上一稳定节点，严禁顺着错误思路打补丁。

---

## 4. 常用验证命令

```bash
# 运行单元测试
uv run pytest tests/

# 运行代码规范检查
uv run ruff check .
uv run ruff format --check .

# 执行增量工作流与独立评分 (按 implementation-plan.md 规范)
uv run python -m app.presentation.cli --help
```

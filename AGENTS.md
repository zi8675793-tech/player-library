# player-library 协同开发指南

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

## 3. 常用验证命令

```bash
# 运行单元测试
uv run pytest tests/

# 运行代码规范检查
uv run ruff check .
uv run ruff format --check .

# 执行增量工作流与独立评分 (按 implementation-plan.md 规范)
uv run python -m app.presentation.cli --help
```

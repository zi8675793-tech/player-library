# 阶段二 任务卡：时空硬排他与多原型增量匹配引擎 (T2)

请直接阅读并实施，完成核心算法、单元测试与交接。

## 1. 任务背景与目标
接收 T1 输出的 `DualStreamFeature`，基于篮球物理常识与带 Cannot-Link 互斥约束的最大相容团（Maximal Compatible Clique）挖掘算法，做出稳健的增量身份归属决策：
1. **时空硬排他 (Cannot-Link)**：同一机位、同一物理帧出现的两人，距离强制设为 $+\infty$；
2. **号码一票否决**：两段轨迹若识别出不同号码，物理排他；
3. **多原型比对**：与已有球员的典型视角原型（正面/背面带号/侧面）比对，要求满足距离与竞争边距（$\Delta \ge 0.04$）；
4. **最大相容团挖掘与三态决策**：在待分配池中以未分配最高质量节点为 Seed 贪心成团，严格满足跨至少 2 个独立视频见证（`len(videos) >= 2`），输出 `matched_existing`、`created_new_identity` 或 `insufficient_evidence`（延期待定），严禁将整个连通分支级联剔除。

## 2. 职责边界与规范
- **允许修改/新建**：
  - `app/identity/constraints.py`
  - `app/identity/matching.py`
  - `tests/test_constraints_and_matching.py`
- **严禁修改**：
  - `docs/data-contracts.md`
  - `app/features/`
- **契约依赖**：严格接收 `DualStreamFeature`，产出 `IncrementalPlan`。

## 3. 验收要求
1. 单元测试 `uv run pytest tests/test_constraints_and_matching.py` 全部通过（当前基准：100 项全绿）；
2. 边界用例覆盖：同帧空间分离测试、跨镜头号码相符测试、号码冲突一票否决测试、单次孤立段待定测试、连通分支内共现 Cannot-Link 最大相容团分离测试；
3. 代码规范 `uv run ruff check app/identity/` 与格式检查无报错。

# 阶段四 任务卡：独立黄金评测与达标验收 (T4)

请直接阅读并实施，完成评测防火墙核验、基准跑分与最终验收报告。

## 1. 任务背景与目标
运行独立评分工具，对 T3 生成的完整冻结产物进行无偏评估：
1. **哈希固化门禁**：先核验 `artifact-hashes.json`，确保预测结果在评分前未被篡改；
2. **独立打开考卷**：读取 `test/annotations.csv`，在固定 70 分母上独立打分；
3. **输出基准账本**：27 号覆盖率（合格线 $\ge 57 / 70$）、已知异人（10号、15号）混入数（必须为 0）、物理冲突数（必须为 0）。

## 2. 职责边界与规范
- **允许修改/新建**：
  - `app/evaluation/`
  - `tests/test_evaluation_harness.py`
  - `docs/current-state.md`（更新跑分指标）
- **严禁修改**：
  - `app/features/`、`app/identity/`、`app/workflows/`（严禁为提分而篡改推理代码）
- **原则**：严禁缩小 70 分母，严禁挑选有利样本。

## 3. 验收要求
1. 跑通评分流水线并输出 `var/benchmarks/run-<timestamp>/scoring_summary.json`；
2. 逐样本归因分析：详细列出每一项未命中的原因（漏检、排他阻断或特征距离超标）；
3. 更新 `docs/current-state.md` 中的跑分表格，出具正式验收结论。

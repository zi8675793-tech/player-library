# 阶段一 任务卡：双流特征提取器构建 (T1)

请直接阅读并实施，完成代码、测试、产物与交接，不要停留在方案讨论。

## 1. 任务背景与目标
针对篮球场同队球衣完全同色的问题，传统行人 ReID 无法区分 10 号与 27 号。本任务目标是构建双流特征提取器：
- 强物理证据流：从 `test/track/<clip_id>/samples/*.png` 人体切片中提取球衣号码识别结果（带时序投票）；
- 外观表征流：提取 512 维深度外观特征向量。
两者封装为符合 `docs/data-contracts.md` 中定义的 `DualStreamFeature` 契约对象。

## 2. 职责边界与规范
- **允许修改/新建**：
  - `app/features/` 目录下的所有文件
  - `tests/test_dual_stream_features.py`
- **严禁修改**：
  - `docs/data-contracts.md`（契约冻结）
  - `test/` 文件夹（只读）
- **绝制作弊**：严禁以任何形式读取 `test/annotations.csv` 或 `test/target/`。

## 3. 验收要求
1. 单元测试 `uv run pytest tests/test_dual_stream_features.py` 全部通过；
2. 代码检查 `uv run ruff check app/features/` 无报错；
3. 将运行产物或缓存特征导出到 `var/features/run-<timestamp>/` 并生成 `handoff.txt`。

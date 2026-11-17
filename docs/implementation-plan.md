# 四阶段实施计划 (Implementation Plan)

本计划将系统建设拆解为四个互不干扰、逐层演进的独立开发任务（T1 $\rightarrow$ T2 $\rightarrow$ T3 $\rightarrow$ T4）。四个阶段均已交付验收，本文档保留各阶段的目标、实现要点与验收命令，作为算法演进的历史依据与回归基准。

---

## 阶段 1：双流特征提取器 (T1)

### 任务目标
实现 `app/features/` 模块，读取切片图片（`test/track/*/samples/*.png`）提取球衣号码识别特征，并与外观特征打包成符合 `DualStreamFeature` 契约的标准输出。

### 实现要点
1. 采用 OCR / 轻量场景文本识别模型读取球衣号码；
2. 实现时序置信度投票 (Temporal Voting)，同一条轨迹内多帧结果投票得到整段号码；
3. 外观流输出 512 维 L2 归一化向量；
4. 绝不以任何方式读取 `test/annotations.csv` 或 `test/target/`。

### 验收命令
```bash
uv run pytest tests/test_dual_stream_features.py
uv run ruff check app/features/
```

---

## 阶段 2：时空硬排他与多原型增量匹配引擎 (T2)

### 任务目标
实现 `app/identity/` 模块中的时空排他网络构建与多原型增量三态决策器。

### 实现要点
1. 构建 Cannot-Link 排他图：同一视频源同一物理帧出现的两人，距离强制设为 +inf；
2. 号码一票否决：若候选号码与已有球员号码不同（如 10 vs 27），绝对禁止匹配；
3. 多原型相容匹配：计算候选段与已有球员典型原型的距离，要求通过距离门槛且满足竞争边距 (margin >= 0.04)；
4. 最大相容团挖掘：采用带 Cannot-Link 互斥约束的最大相容团启发式算法，以最高质量未分配节点为 Seed 贪心成团；只有跨至少 2 个片段获得独立见证 (`len(videos) >= 2`) 的新集团才允许建立新档案；单次孤立段一律标记为 insufficient_evidence (待定)，严禁级联剔除整个连通分支。

### 验收命令
```bash
uv run pytest tests/test_constraints_and_matching.py
uv run ruff check app/identity/
```

---

## 阶段 3：流水线组装与 SQLite 版本化原子存储 (T3)

### 任务目标
将 T1 特征提取器、T2 匹配引擎串联进 `app/workflows/incremental.py`，实现分批到达、SQLite 事务提交、留一法重估与幂等中断续跑。

### 实现要点
1. 独占写事务 (BEGIN IMMEDIATE)，写操作完全基于快照隔离；
2. 实现留一法重估 (Leave-one-out Reassessment)：对机器暂定归属排除自身后重估，支持版本化重分配；
3. 幂等性：中断后重放或重复提交请求，保持数据完全一致 (changed=False)；
4. 产物留痕于 `var/runs/`，生成不可变 handoff.txt。

### 验收命令
```bash
uv run pytest tests/test_incremental_pipeline_e2e.py
uv run ruff check app/workflows/
```

---

## 阶段 4：独立评分防火墙与黄金评测达标 (T4)

### 任务目标
运行独立评分工具，核验 T3 生成的冻结输出，在固定 70 分母上独立计算 27 号覆盖率与纯度指标。

### 实现要点
1. 双进程物理隔离：只有在产物哈希核验完全通过后，评分工具才允许打开人工标注；
2. 在固定 70 个测试样本上评分，严禁缩小分母或挑选有利样本；
3. 报告 27 号命中率（目标 >= 57/70，即 >80%）；
4. 报告已知异人（10号、15号）混入数（目标必须为 0）；
5. 报告空间分离物理冲突（目标必须为 0）；
6. 结果汇总写入 `docs/current-state.md`。

### 验收命令
```bash
uv run python -m app.evaluation.run_benchmark --run-dir <latest_run_dir>
uv run pytest tests/test_evaluation_harness.py
```
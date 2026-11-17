# 独立评测防火墙与黄金基准规范 (Evaluation Harness)

本文档定义系统如何进行**可审计、防作弊、客观公正的独立基准评分**。

---

## 1. 评测防火墙核心原则 (Firewall Principles)

1. **绝对防作弊隔离 (Zero Ground-Truth Leakage)**：
   - 算法推理全流程（特征提取、图库维护、增量匹配、存储提交）**物理隔离**于任何人工标签。
   - `test/annotations.csv`、`test/target/` 仅在评测阶段被独立读取，用于考卷核对。
2. **两阶段门禁（先冻结，后出卷）**：
   - 第一阶段（`prepare`）：固化推理产物、轨迹几何账本与代码指纹，生成 `artifact-hashes.json`；
   - 第二阶段（`score`）：校验指纹未篡改后，打开考卷，打出分数。若指纹不匹配，程序直接崩溃拒绝评分。
3. **固定 70 分母（严禁挑选有利样本）**：
   - 27 号球员在 10 个测试片段中固定有 **70 个评测样本点**。
   - 任何由于漏检、未处理、歧义、拆档或错误分配导致的损失，**一律留在 70 分母中作为惩罚**，绝对不允许缩小分母。

---

## 2. 评测指标定义与合格线

| 指标项 | 计算方式 | 合格底线 (Pass Line) | 优秀目标 (Target) |
| :--- | :--- | :---: | :---: |
| **27 号正确样本覆盖** | $\frac{\text{归入 27 号全局档案的正确样本数}}{70}$ | **$\ge 57 / 70$ (81.4%)** | $\ge 62 / 70$ (88.6%) |
| **已知异人混入数** | 绿衣 10 号、15 号等被错误归入 27 号档案的样本数 | **0 项**（零容忍） | 0 项 |
| **空间分离物理冲突** | 同一视频源同一真实帧内，两个物理空间分离的人体被同时标为 27 号 | **0 项**（物理不可能） | 0 项 |
| **碎片化拆档数量** | 70 个 27 号样本被撕裂归入的不同多余档案数 | $\le 2$ 个碎档 | 0 碎档 |
| **真实有效观测时长** | 27 号档案在真实视频帧中的有效曝光秒数之和 | $\ge 50.0$ 秒 | $\ge 55.0$ 秒 |

---

## 3. 一键统一端到端评测闭环 (推荐)

开发者只需执行一行命令，即可从原始轨迹出发，完整跑通：摄入 → 抽取 → 纯自动冷启动匹配 → SQLite 提交 → 产物固化 → 独立子进程防火墙打分 的全链路闭环：

```bash
# 全流程无作弊一键评测（默认运行全量 10 个片段并生成带时间戳账本与控制台报表）
uv run python run_e2e_benchmark.py

# 或对既有已冻结推理产物进行快速独立打分
uv run python run_e2e_benchmark.py --run-dir var/runs/run-20261115T033000-bug7
```

---

## 4. 独立评分分步命令与流水

评测工具统一位于 `app/evaluation/`，各步骤亦可分步单独调用：

```bash
# 步骤 1: 冻结产物核验与几何对齐 (不读标注)
uv run python -m app.evaluation.prepare_benchmark \
  --run-dir var/runs/latest \
  --test-data ../test \
  --out-dir var/benchmarks/latest/geometry

# 步骤 2: 打开标准答案独立评分
uv run python -m app.evaluation.score_benchmark \
  --geometry-dir var/benchmarks/latest/geometry \
  --annotations ../test/annotations.csv \
  --out-dir var/benchmarks/latest/scoring
```

运行后自动输出：
- `scoring_summary.json`：汇总指标卡；
- `confusion_matrix.csv`：多球员混档矩阵；
- `audit_failure_cases.jsonl`：逐个样本失败归因（漏检、误判、排他阻断）。

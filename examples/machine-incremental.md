# 机器增量分批更新基线记录 (Machine Incremental Baseline)

> [!NOTE] 历史演进与最新基线导航
> 本文档记录 2026-11-06 早期开放集合增量分批运行（`incremental-20261106-run1` 与 `run2-reversed`）。
> 在 2026-11-09 ~ 2026-11-10 的身份修复计划（阶段2专项漏洞修复）中，增量模式进一步实现了短段增量匹配门槛支持（`min_match_buckets=2`）、短段建档隔离（`can_create_new_identity=False`）、校准档案一致性绑定（R1）、漏斗全局多批累积单调性恢复（R3）以及部分复制硬过滤（R4）。
> 最新增量运行数据（Forward 样本覆盖率 8.02%，Reverse 样本覆盖率 10.92%，ARI 达 0.883，零物理违规且跨片 FP=0）请参见 [开放集合增量身份更新规范](../docs/incremental-identity.md) 与 `var/identity-repair/stage-2/handoff.txt`。

## 1. 运行命令（2026-11-06 历史命令）

```bash
# 正向分批运行 (batch_size=3, 共 4 批次: [11,20,3], [30,37,43], [50,6,63], [68])
uv run python -m app.presentation.cli baseline \
  --config examples/machine-discovery.toml \
  --run-id incremental-20261106-run1 \
  --incremental-batches

# 逆向分批运行 (评估顺序敏感性: [68], [50,6,63], [30,37,43], [11,20,3])
uv run python -m app.presentation.cli baseline \
  --config examples/machine-discovery.toml \
  --run-id incremental-20261106-run2-reversed \
  --incremental-batches \
  --reverse-batches
```

---

## 2. 冻结产物与关键指标

### 2.1 正向运行 (`incremental-20261106-run1`)
产物目录：`var/identity-discovery/incremental-20261106-run1/`

- **输入数据规模**：10 个视频片段、158 条轨迹、75,465 条逐帧观测、1,759 个特征样本/行。
- **分批处理推进**：
  - Batch 1 (`[11, 20, 3]`): 139 个段均因单次出现/无跨片强支持保持 `unassigned`（Revision 0 -> 1，重放幂等验证通过）；
  - Batch 2 (`[30, 37, 43]`): 发现 1 个强支持新身份，关联合并 2 个段（Clip 37 Track 11 与 Clip 43 Track 0），准入 6 个自动实验模板（Revision 1 -> 2，重放幂等验证通过）；
  - Batch 3 (`[50, 6, 63]`): 发现 1 个强支持新身份，关联合并 2 个段（Clip 6 Track 4 与 Batch 1 中累积的 Clip 20 Track 3），准入 6 个自动实验模板（Revision 2 -> 3，重放幂等验证通过）；
  - Batch 4 (`[68]`): 新增段无跨片强支持，保持 `unassigned`（Revision 3 -> 4，重放幂等验证通过）。
- **最终图库与身份**：
  - 生成 2 个稳定匿名球员（`pl_a31044da39075185bc08cc0f824f0cad`, `pl_4ad3b5c11b9152c780cbf5f19f0ebcf1`）；
  - 12 个活跃自动实验模板（`review_status="unreviewed"`, `admission="automatic_experimental"`）；
  - 4 个提议分配（`status="proposed"`），859 个待定段（`status="unassigned"`, `reason="unknown"`）；
  - 人工干预数：0；人工批准数：0。

### 2.2 逆向运行 (`incremental-20261106-run2-reversed`)
产物目录：`var/identity-discovery/incremental-20261106-run2-reversed/`

- **分批处理推进**：
  - Batch 1 (`[68]`): 235 个段保持 `unassigned`；
  - Batch 2 (`[50, 6, 63]`): 220 个段保持 `unassigned`（Clip 6 Track 4 进入未分配池，此时 Clip 20 尚未输入）；
  - Batch 3 (`[30, 37, 43]`): 发现 1 个强支持新身份，合并 2 个段（Clip 6 Track 4 与 Clip 37 Track 10），准入 6 个模板；
  - Batch 4 (`[11, 20, 3]`): 139 个段保持 `unassigned`。
- **最终图库与身份**：
  - 生成 1 个稳定匿名球员（`pl_4c621218d78c5ed380c760198cff4cf6`）；
  - 6 个活跃自动实验模板；
  - 2 个提议分配，861 个待定段。

---

## 3. 顺序敏感性与对比审计 (Order Sensitivity Audit)

| 维度 | 正向运行 (`run1`) | 逆向运行 (`run2-reversed`) | 差异机制分析 |
| :--- | :--- | :--- | :--- |
| **最终发现人数** | 2 名球员 | 1 名球员 | 批次输入时序改变了候选池中段的可见性与组合空间 |
| **Clip 37 与 43** | 成功合并 (`pl_a31044...`) | 阻断待定 (`incompatible_positive_neighbors_bridge`) | 逆向运行时，未分配池中已存在 Clip 6/63 的候选段，三方形成非全互连接桥接，触发安全防御阻断 |
| **Clip 6 与其它** | 与 Clip 20 合并 (`pl_4ad3b5...`) | 与 Clip 37 合并 (`pl_4c6212...`) | Clip 6 的到达顺序决定了它是先与 Clip 20 聚类还是先与 Clip 37 聚类 |
| **幂等性与审计** | 4 个批次均 100% 幂等重放 | 4 个批次均 100% 幂等重放 | 批次内部严格使用当前 Revision 固定快照，无状态级联泄漏 |

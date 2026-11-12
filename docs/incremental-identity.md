# 开放集合增量身份更新与图库维护规范

## 1. 架构目标与设计原则

同一场比赛分批输入新视频片段时，在**无人工身份答案**（零种子）的前提下，实现身份状态的自动演进：
1. **自动延续已有身份**（`matched_existing`）：当新输入证据与已有图库中的活跃模板存在强流形近距离支持且满足全局时空硬约束与边界裕度时，提议关联到已有 `player_id`，绝不重复建档。
2. **发现有证据支持的新身份**（`created_new_identity`）：当多个未分配段跨越独立片段（至少 2 个不同片段）形成完全支持集团，且与所有已有球员保持充分分离时，自动建立新的持久匿名 `player_id`。
3. **证据不足保持待定**（`insufficient_evidence`）：单次出现段、跨视角对照不完整、存在少数异类矛盾或与候选球员距离较远/边界模糊的段，严格标记为待定（`status="unassigned"`, `unassigned_reason="unknown"`）。**无匹配或距离远绝不单独作为建立新人的证据**。
4. **固定快照批次隔离**：当前批次的匹配与归人完全基于批次开始时读取的固定图库快照（Revision $R$），批次内部的新决策在提交原子事务并推进版本到 Revision $R+1$ 前对批内其它段不可见，杜绝同批次内的错误级联与串联污染。
5. **严谨的自动模板准入与退场**：自动准入模板标记为 `admission="automatic_experimental"`, `review_status="unreviewed"`, `approval=None`，绝不伪造人工批准；必须具有独立多样本见证（raw multi-witnesses）及 `sample_sources` 来源版本绑定；模板支持对必须完整解析至已支持的 `DiscoveryPair`；当人工复核拒绝或重映射时，自动模板立即退场失效。
6. **人工优先级与审计**：人工确认归属、人工拒绝或人工约束（`block_automatic`, `forbid_player`, `cannot_link`）具有不可侵犯的绝对最高优先级，自动算法只能作用于未受人工约束阻断的段。
7. **语义边界**：空库重放幂等与增量批次重放幂等仅证明状态迁移与事务序列化的一致性，不能等同于免除了无真值情况下的独立校准。所有自动归属一律标记为 `proposed/automatic`，校准状态为 `unvalidated`，不报告无真值身份准确率，不计入正式 `confirmed` 事件。

---

## 2. 增量更新工作流与事务隔离 (`app/workflows/incremental.py`)

```
                  +--------------------------------+
                  |  Incoming Batch at Rev R       |
                  +---------------+----------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| Atomic SQLite Transaction (Exclusive)                             |
|                                                                   |
| 1. Snapshot Read: GallerySnapshot at expected revision R          |
| 2. Replay Idempotency Check:                                      |
|    If exact batch already ingested and no business changes,       |
|    return changed=False with unchanged revision R.                |
| 3. Raw Evidence & Features Persistence:                           |
|    Store new batch sources, tracks, segments, observations.       |
| 4. Plan Incremental Identities:                                   |
|    (Pure algorithmic evaluation against snapshot R)               |
|    Step 1: Match incoming segments against existing player tpls   |
|    Step 2: Pool unassigned segments for cross-clip discovery      |
|    Step 3: Deduplicate discovered groups vs existing players      |
|    Step 4: Prepare atomic decisions, templates & discovery report |
| 5. Atomic State Writes:                                           |
|    - Put new players & proposed assignments (Revision R+1)        |
|    - Put new automatic experimental templates (Revision R+1)      |
|    - Put immutable DiscoveryReport with sample bindings           |
| 6. Revision Advance:                                              |
|    Advance match revision from R to R+1 with structured audit     |
| 7. Consistency Checks:                                            |
|    validate_evidence_context, validate_templates, segment_frames  |
+-------------------------------------------------------------------+
```

---

## 3. 三态判定与匹配策略 (`app/identity/incremental.py`)

### 3.1 已有身份匹配 (`matched_existing`)
- **多代表元比对**：将输入段按 `bucket_s` 划分为非重复组件代表元集合，两两比对图库活跃模板。
- **短段匹配门槛（阶段2修复）**：
  - 传统段要求至少 `policy.min_support_buckets`（3 桶）；
  - 阶段2解耦匹配资格与建档资格：时间桶为 2 桶的短段标记为 `short_segment_limited_match_support`（`can_match_existing=True`, `can_create_new_identity=False`）；
  - 在 Step A 匹配已有球员时，允许使用 `seg_min_support_buckets = policy.min_match_buckets`（默认 2）以及可选的独立专属门槛 `policy.short_segment_match_threshold`（未配置时使用 `match_threshold`），解除 3 桶对短段增量匹配的硬阻断。
- **物理排他与距离裕度检验**：
  - 时空硬排他：若候选球员在输入片段的时间范围内已存在物理重合段或观测，直接一票否决。
  - 距离门槛与竞争者裕度：输入段与候选球员的代表距离必须 $\le \tau$ (0.18) 且与次优候选球员拉开至少 `policy.rival_margin` (0.04) 的差距；若两个已有球员距离过于接近产生歧义，触发 `ambiguous_competing_existing_players` 保守保持待定。
- **图库扩充**：匹配已有球员后，仅当该球员活跃模板数未达 `max_templates_per_group` 且能补充全新独立视角组件时，方可按准入规则申请添加新模板。

### 3.2 支持建立新身份 (`created_new_identity`)
- **发现池组成**：当前批未匹配且合格的段 + 历史图库中累积的合格未分配段（排除受人工约束阻断的段）。
- **短段建档严格拦截（铁律）**：
  - 未匹配的短段（`can_create_new_identity=False`）**绝对禁止**进入 Step B 跨片段聚类发现池；
  - 保持其未归属状态并赋予 `deferred_reasons=["short_segment_limited_match_support"]`，坚决杜绝因短段特征单一、视角不全而建立孤立碎片化新球员档案。
- **完全图约束与桥接隔离**：
  - 集团内任意两段均须具备双向支持（`status="supported"`）。
  - 若某段连接两个互不兼容的集团，触发 `incompatible_positive_neighbors_bridge` 隔离该桥接段，防止误合并。
- **防止同人反复建档**：
  - 发现集团形成后，将集团成员样本与已有所有球员模板计算中位数距离；
  - 若中位数距离 $\le \tau$，则判定为已有球员的新视角补充，直接复用既有 `player_id`，不建立新球员。
  - 仅当集团跨越 $\ge 2$ 个独立视频片段且与全部已有球员保持充分分离时，才调用 UUIDv5 派生新匿名 `player_id`。

### 3.3 证据不足 (`insufficient_evidence`)
- 单次出现的孤立段（无独立跨片段见证）。
- 存在内部运动跳变、混人风险、采样质量差或被阻断的桥接段。
- 距离现有图库较远但又缺乏跨片段共现支持的段。
- 结果一律保持 `status="unassigned"`, `player_id=None`, `unassigned_reason="unknown"`。延期原因结构化保存在 `deferred_reasons` 供完整审计。

---

## 4. 批次顺序敏感性分析 (Order Sensitivity)

在对真实 10 片段数据进行正序与倒序分批运行时，揭示了流形增量聚类的固有顺序敏感特征：

1. **历史未分配池的异步可见性**：
   - 先到达的片段若为单次出现，必须按契约保守进入未分配待定池；
   - 正序与倒序到达使得未分配池的片段交接顺序不同，导致先配对的跨片段锚点存在次序差异。
2. **桥接阻断与时序上下文**：
   - 某段在正序下先行与另一段成对闭合；但在倒序下，未分配池中已存在先前批次输入的弱关联段，形成了三方不可闭合的三角形，触发了 `incompatible_positive_neighbors_bridge` 防御性阻断，导致该段在倒序下选择保守保持待定。
3. **工程应对原则**：
   - 严格禁止把“正序与逆序完全相等”作为系统硬断言；
   - 无论何种输入顺序，核心硬约束（物理时空零违例、unknown 永不归人、模板支持链完整、重放幂等）均必须严格成立。

---

## 5. 阶段 4 端到端验收与三次运行实际指标

分别使用全新独立输出目录对真实数据进行无标注增量 baseline 运行，验证可变批次大小与正反序输入：

### 5.1 增量运行配置与结果对比（阶段2完成 2026-11-10 专项漏洞加固后最新基线）

基于固定分母（10 个比赛片段机器证据输入，158 原始轨迹，1,759 样本，75,465 观测，1,346.35 秒时长）在阶段2最新冻结产物（`var/identity-repair/stage-2/`）：

| 运行标识 (Run ID) | 批大小 | 批数 | 到达顺序 | 最终 Rev | 发现球员数 | 活跃模板数 | 分配段数 | 样本覆盖率 | 时长覆盖率 | 跨片段对偶精度 | 物理碰撞 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stage-2/cold` (全量发现对照) | - | 1 | - | 2 | 4 | 13 | 23 段 / 107 样本 | 6.08% (107/1759) | 3.20% (43.12s) | 100.0% (FP=0) | **0** |
| `stage-2/forward` (正序增量) | 3 | 4 | 正序 (Forward) | 5 | 4 | 16 | 37 段 / 141 样本 | 8.02% (141/1759) | 4.12% (55.45s) | 100.0% (FP=0) | **0** |
| `stage-2/reverse` (逆序增量) | 3 | 4 | 倒序 (Reverse) | 5 | 4 | 16 | 50 段 / 192 样本 | 10.92% (192/1759) | 5.60% (75.38s) | 100.0% (FP=0) | **0** |

*注：覆盖率指标基于固定分母（1,759 样本 / 1,346.35 秒），严格禁止与真实身份准确率混淆。Forward 与 Reverse 之间跨批对齐 F1 达 0.883，ARI 达 0.883，Jaccard 达 0.553。*

### 5.2 核心安全、漏斗单调性与契约核验结果

1. **增量漏斗全批累积单调性恢复（2026-11-10 专项修复 R3）**：在 baseline 增量运行中汇总全部批次的 segmentation reports，构建全局 `combined_seg_report` 注入漏斗评测，彻底修复了原先仅取最后一批报告导致的局部层倒挂缺陷。Forward 漏斗各级修正为：452 (Stage 3) >= 166 (manifest cumulative) >= 141 (final assigned)；Reverse 漏斗为：452 >= 192，严格恢复单调递减物理定律。
2. **输入白名单完整受控**：`input-manifest.json` 经校验仅包含 `track/`、`video/`、`identity/` 目录下的合法视频、轨迹与特征文件，未引入任何标注、答案或人工先验。
3. **物理时空硬约束零违例**：全场 75,465 条逐帧观测穿透排查，三次运行中同一球员在同一片段内的同屏重叠数严格为 0。
4. **未知永不继承身份**：无任何 `status="unassigned"` 或未通过局部一致性支持检验的段被赋予 `player_id`。
5. **模板支持证据链完整**：所有自动生成的模板均包含非空见证样本（`witness_sample_ids`），其 `support_pair_ids` 均 100% 成功解析至已支持的 `DiscoveryPair`。
6. **Revision 与幂等严格一致**：每一次批次提交原子递增 1 个版本（冷启动 rev=2；批大小为 3 的增量运行共 4 个批次，从初始空库推进至 rev=5）；同批次重复提交核验均触发幂等拦截，`changed=False` 且版本完全不变。
7. **延期原因清晰可审计**：所有待定段均保留结构化排除原因（如 `segment_not_supported`, `insufficient_independent_buckets`, `appearance_conflict_including_minority` 等），杜绝静默无理由拒识。
8. **漏洞专项加固闭环（R1, R2, R4）**：运行时严格断言绑定的校准档案参数一致性（R1）；校准候选过滤缺失框与低可见度特征（R2）；部分复制序列在非共享组件基数与多见证硬约束下无法通过增量聚类建组（R4）。

---

## 6. 剩余限制与后续建议

1. **增量保守性与覆盖率落差**：相比于全量一次性聚类（Phase 3 `manifold-delivery-001` 活跃段归人率 97.22%），增量分批更新在严格的固定快照隔离与防串联污染要求下，单次出现的段必须先行待定，只有在后续批次出现明确跨片段对应证据时才得以激活，因此阶段覆盖率较为保守（37% ~ 45% 支持段覆盖）。
2. **无真值精度未校准**：当前所有结果均为 `automatic/proposed/unvalidated`，未经过独立标注真值的精确率/召回率评估。
3. **人工复核协同**：建议将增量输出的高风险待定段和边界段推入审校队列，通过人机协同机制形成正式 `confirmed` 身份。

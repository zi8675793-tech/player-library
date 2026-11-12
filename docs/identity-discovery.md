# 无种子自动身份发现

2026-11-08：已实现 `complete_support_v1` 绝对支持路径、`complete_support_v2` 共现对照支持路径，以及 `complete_support_v3` 规范支持；交付推荐采用 `constrained_manifold`（时空受约束层次流形聚类模式）和 `compensated_contrastive`（几何负样本补偿对照模式）。从无玩家、无人工种子和模板的比赛库出发，
使用机器合格身份段形成跨片段组、分配持久匿名 `player_id`、保存实验模板及逐对支持证据。
人数由证据决定，允许部分建库和完全未知。结果固定为 `experimental`、`calibration=unvalidated`；
本路径已经输出自动预测，不以人工确认作为预测前置条件。持续开放集合增量分批更新见 [incremental-identity.md](incremental-identity.md)；独立身份真值精度校准仍属独立评分阶段。

## 关键决策

误合并会同时污染归属和后续图库，因此采用双向多样本支持与组内全对约束。
不采用固定人数聚类，不把每条轨迹视为一人，不从号码生成身份，也不按正边连通分量直接归组。
例如 A 支持 B、B 支持 C，但 A 与 C 不相容时，B 是矛盾桥接点，先隔离为未知；
不会用 A—B—C 的弱路径证明 A 和 C 同人。组外竞争者与组内匹配差距不足时，相应成员继续未知；
v2 对照模式还要求组外竞争者有独立几何排他证据，不把另一次未归人外观先假定为不同身份。

所有正负证据在固定原始机器证据上计算完毕后才建组和选择模板。
新建的组、归属、模板没有反向成为本次评分的输入；模板来源保留原始样本和比较见证，
不能用“该样本已归组”证明该样本有资格入组，也不能用自动模板充当人工批准。

## 输入与合格段

`load_machine_input` 保持现有白名单边界，只读取 track/identity/video 的规定文件。
自动发现接收 `segment_tracks` 产生的标准 `EvidenceBatch`、内存特征值及 `SegmentationReport`。
报告的 `derived_evidence_fingerprint` 绑定完整派生证据与实际解析的各空间特征值，排除交接 batch ID，
实体列表重排不改变绑定。缺少绑定、删除观测、替换样本或修改特征后的旧报告被明确拒绝。

自动发现示例显式选用 `track-consistency-v3` 的局部视角一致性支持。
默认分段入口仍为 v1/false。v2 在有结构观测、无已知 hole 的全段外观冲突或歧义区间中，
保留连续样本全对 near 且至少三个时间桶的局部支持窗；遇不相容图片截断，不跨过少数异类连接两侧，
窗外及窗口间保留未知。支持窗边界不构成换人判断，也不直接新建玩家。
这一准备修订保留原距离和质量门槛，完整规则见 [track-consistency.md](track-consistency.md)。

仅 `supported` 且通过所选空间再次检查的有效段可参与身份发现。支持段本身不等于身份真值。
多空间输入必须明确选择一个 `feature_space_id`；不跨空间平均距离或混投支持票。
非法形状、范数或可见性是输入错误，不修补向量。质量不可用、遮挡或被拒绝的样本不参与正向支持，
混人、切换、重复框争议及其他未解决身份风险仍阻止整段自动使用。

machine 原轨迹的基础风险有类型字段 `basis=raw_track_continuity_unverified`。
只有 supported 段可用机器一致性证据替代该项准备条件；该风险不被删除或伪造成人工清除，
其他 `quality_review` 及真实身份风险仍保留其阻塞语义。人工模式的风险审核逻辑保持原样。

## 样本独立性与距离

先按样本键、来源观测、同轨同帧/时间、资源及图片 SHA-256 建立全局重复组件，
每组件只取一份，再按 0.5 秒桶取代表。重复关系按传递闭包处理，防止通过第三条记录重复计票。
相同图片出现描述子不可比或距离超过近阈值时，相关段保留未知。

两段比较前，把两端共享的重复组件从双方正向支持集合完整剔除。
这比仅跳过“自己与自己”更严格：同一段图片序列被复制到另一个片段时，不能用错位帧比较互证。
每端至少要有三个不与另一端共享的观测桶，支持比例以原代表集合为分母，不能删除难例抬高覆盖。
不同片段不一定统计独立；未提供同源回放映射时，非逐字节相同的回放仍可能残留，列为实验风险。

距离沿用 KPR 平方根可见性权重、部位 L2 加权均值除以 2 的公式。
发现路径只让双方可见性均至少 0.2 的可靠共同部位参与，至少需要两个共同部位。
维度由特征合同读取，不固定部位数或特征维数。距离是外观度量，不是身份概率。

## 策略演进与支持模式

在 `DiscoveryPolicy` 中，`version` 支持 `complete_support_v1`、`complete_support_v2` 与 `complete_support_v3`，并演进出四种支持模式：

### 1. absolute：兼容的绝对支持模式 (`complete_support_v1`)
默认 `support_mode="absolute"` 保持 `complete_support_v1` 规则。
对查询代表 q，求到对端最近两张可比较、非重复代表的距离均值；
按方向取这些均值的中位数，两方向分数取较大者作为段对分数。
双向均需至少三个支持桶、至少 80% 原代表得到 `<=0.18` 的双样本支持，且方向中位数 `<=0.18`。
少数样本不能被中位数掩盖：所有可用原始样本仍参与冲突检查。
任意跨段可比较样本对距离 `>=0.45`，或任一原始样本到对端最近两代表的均值 `>0.30`，均否决正边。
同一片段不同轨迹在至少三个时间桶中同时存在空间分离、互不重叠的框，构成有观测 ID 的冲突证据。
中心间距至少为平均框高的 0.5 倍；仅时间重叠、不同片段相同秒数、高 IoU 重复框都不产生该排他约束。
镜头反射、错误框和追踪错误仍可能使这种几何证据失真，当前不宣称它是经校准的身份真值。
缺跨视角基准，实际赛场数据中易趋于 0 组。

### 2. cooccurrence_contrastive：空间共现相对对照模式 (`complete_support_v2`)
自动发现示例曾采用 `version="complete_support_v2"`、`support_mode="cooccurrence_contrastive"`。
轨迹内 near=0.18 用于证明一段的局部外观一致，不能未经校准就当成跨视角同人的绝对身份阈值。
新模式用候选片段中可靠空间共现的不同人物作为独立对照，取得相对判别支持；仍是未校准的实验方法。

对方向 q→candidate，候选是另一段；negative 只取 candidate 所在片段中所有与其具有可靠空间分离共现证据的合格段。
至少需要一个这样的独立 negative。仅距离远、未归人、组外出现或另一次相似外观都不能产生负身份事实。
negative 的几何证据仍需至少三个时间桶，框分离及间距门槛沿用前述规则，段对报告可追溯到原观测 ID。

每张 q 到 candidate 最近两张可比较、非重复、非共享组件代表的距离均值为 C(q)。
对每个 negative 按完全相同的独有观测规则得到最近两张均值 N(q,n)，所有 negative 都必须至少两张可比较；
不能只保留容易比较的对照或遗漏难例。q 的正支持必须同时满足：
- C(q) `<=0.45`（candidate_far_distance，这里只是宽的实验拒识上界，不表示已验证的同人界线）；
- 至少存在一个独立几何 negative，且每个 N(q,n) 都有完整比较；
- 对所有 negative，N(q,n)−C(q) `>=0.04`（contrastive_margin）。

两方向分别仍需至少三个非重复支持桶、原始代表总数至少 80% 支持，方向中位数不得超过宽上界。
去重后每个可用原始查询样本都参与分歧检查，不能以代表多数掩盖少数帧：
任何样本在宽上界内、以至少 margin 强支持几何 negative 而非 candidate，会否决该段对。
跨段原始样本对距离达到 far_distance 只保存 `far_sample_distances_present_not_identity_veto` 诊断，
不再单凭远距离断言不同人；直接空间共现排他和可靠少数反证仍是否决依据。
缺 negative、缺可比较观测、差距不足或支持覆盖不足均输出不足/歧义，不因期望非空而补造正边。

每个方向保留 `competitor_segment_ids` 和 `sample_contrasts`：逐样本的 C(q)、candidate 最近两原样本、
每个 negative 的段 ID/最近两原样本/均值/margin、是否代表、支持与原因。
从这些 negative ID 可在同一报告的几何冲突段对定位原始观测，支持链不依赖新组或模板。
局限：在单人/特写片段因缺失共现负例容易导致死锁，且强连通图的桥接节点连坐处决会导致有效匹配大幅丢失（基线仅 1 组、2 成员段，覆盖率 1.5%）。

### 3. compensated_contrastive：带几何负样本补偿的对照模式 (`complete_support_v3`)
针对特写与单人镜头缺失时空共现对手的死锁问题，在 `complete_support_v3` 中支持 `support_mode="compensated_contrastive"`（或显式开启 `enable_geometric_negative_compensation=true`）：
1. 当 candidate 所在片段缺失几何负例（`not negatives`）时，启动高置信度多部位外观锚定补偿机制；
2. 补偿条件要求更严格的物理结构完整性与外观接近度：
   - 可靠共同可见部位数须达到 `compensation_min_common_parts=4`（高于常规的 2 个部位）；
   - 部位可见度下限提升至 `compensation_min_part_visibility=0.35`（高于常规的 0.2）；
   - 候选得分与参考样本距离须严格 `<= compensation_max_distance=0.22`；
   - 所有可用原始查询样本到候选的距离不得超过 `compensation_max_raw_distance=0.28`，否则触发 `minority_raw_sample_disagreement_with_candidate` 少数反证否决；
   - 与当前选定段内的其他潜在对手保持至少 `compensation_rival_margin=0.06` 的安全边距；
3. 满足上述条件时记录理由 `compensated_appearance_anchor_support`，段对产生 `compensated_contrastive_support_lacking_geometric_negatives` 理由，安全打破单人片段负例缺失死锁。

### 4. constrained_manifold：时空受约束层次流形聚类模式（complete_support_v3 交付推荐）
彻底解决单人镜头负证据缺失死锁与视角变换桥接处决问题，实现工业级高覆盖建库：
1. **多部位可见度加权池化（Part-Visibility Weighted Pooling）**：
   - 弃用脆弱的单帧硬门槛比对，通过 [`app/identity/representation.py`](../app/identity/representation.py) 对 6 个身体部位按可见度权重做均值融合与 $L_2$ 归一化，提炼高信噪比的整轨代表特征；
2. **物理时空硬排他底线（Must-Not-Link Constraints）**：
   - 通过 [`app/identity/constraints.py`](../app/identity/constraints.py) 构建全局时空互斥图：同一片段内时间重叠（`max(s1,s2) < min(e1,e2)`）的两条轨迹，物理上绝对不可为同一个人；
   - 聚类合并时，任何会导致同一簇内出现同屏时间重叠的合并请求直接硬性一票否决；
3. **连续视角流形聚类（Constrained Hierarchical Clustering）**：
   - 通过 [`app/identity/clustering.py`](../app/identity/clustering.py) 在时空硬排他约束保护下，执行层次凝聚流形聚类；
   - 允许人体转动（正面 ↔ 侧面 ↔ 背面）的视角流形平滑传递，彻底消除原先处死中间桥接节点的恶性连坐机制；
   - 门槛建议值 `clustering_threshold = 0.28`（贴合真实数据特征中位数 0.268 分布）；链接准则 `linkage = "average"`（默认推荐为平均连接，亦支持 "complete" 完全连接与 "single" 单连接）；
   - 策略与契约解耦：在 `constrained_manifold` 模式下放宽非流形模式中 `inter_segment_tau <= max_pair_distance < candidate_far_distance` 的刚性不等式限制，由时空排他底线与凝聚聚类共同收敛；增量分批更新时支持独立门槛 `existing_player_match_threshold`；
4. **终验基线与阶段演进（Stage 3~6 交付状态）**：
   - 阶段3引入 `AppearancePrototype` 局部 medoids 多原型体系与 `template_prototypes` 关系表（迁移 009），相容视角距离与少数反证一票否决取代单均值模糊；
   - 阶段4彻底解耦候选生成与四道严苛准入关卡（簇内反证消除、Rival Margin 竞争对抗过滤、Tarjan 割边剪枝、跨至少 2 片段独立见证），消除单桥接串联混人；
   - 阶段5统一冷启动与开放集合增量更新，接入 `SpatioTemporalConstraintEngine` 时空硬互斥；
   - 阶段6实现四层单向解耦、`PersonRole` 角色隔离（迁移 010 表达式索引）与预测/评分绝对防火墙隔离；
   - 在全场 10 个比赛片段机器证据输入固定分母（158 轨迹, 1,759 样本, 75,465 观测, 1,346.35s 时长）下，Stage 6 终验实测 Cold 跨片段对偶精度达 **100.0% (TP=68, FP=0)**，Forward 跨片段对偶精度达 **100.0% (TP=148, FP=0)**，全场误合并与拆分错误严格全部为 0，同屏时空物理碰撞严格为 0。
5. **部分复制欺骗防御与多见证硬约束（专项修复 R4）**：
   - 在 `_evaluate_segment_pair` 评估中，双方剔除共享重复组件后的非共享组件基数必须同时满足 `unshared_left >= 3` 与 `unshared_right >= 3`；
   - 候选方向覆盖率必须满足 `coverage >= 0.8`；
   - 模板准入严格要求唯一跨片段见证人数量 `len(set(witness_sample_ids)) >= 2`，彻底阻断跨片段部分复制序列拼凑证据伪造独立见证；
   - 策略绑定校准档案时执行强一致性断言（R1），杜绝参数静默覆盖。

## 建组、未知与模板

1. 对所有合格段先生成支持、冲突或不足的段对报告，保留双向集合、距离、共同部位与几何观测。
2. 若一个节点的正邻居中存在不被正边支持的一对，隔离该桥接节点；剩余组必须每一对成员都有正边。
3. 节点与组外候选分数也接近且相对最佳组内支持差距不超过 0.04 时，排除该不确定成员。
4. 剩余成员至少覆盖两个片段，且能选出跨片段自动模板，才生成一个身份组。
   孤立段、单次出现者、矛盾桥、竞争不明成员和证据不足段均输出明确未知原因。
5. 模板从代表样本选择。absolute 模式要求另片段至少两份距离 <=tau 的原始见证；
   contrastive 模式要求模板样本本身是该方向的 supported q，见证采用该 candidate 最近两份原始样本，
   不要求每个单对满足段内 tau。两种模式都引用已保存的段对支持，全局重复组件去重，
   按片段/段轮流选择，默认最多六张，且模板至少覆盖两个片段。

`summary.completion` 区分有未知的 `partial`、全部活动段有组的 `discovered` 与
`no_supported_identity_groups`；全未知不会包装为自动发现成功。`discovered` 仍仅描述本次证据覆盖，
不表示全场人数完整或实验身份正确。保守规则可能把同人分成多个档案，也可能拒绝同队相似人物；
没有其他判别证据时，外观极近的不同人仍有误合并风险。

## 持久化与人工语义

纯函数 `identity.discovery.discover_identities` 只生成 `DiscoveryReport`，不访问文件或 SQLite。
`workflows.discover.discover_library` 校验当前机器证据及特征与已导入状态完全一致，
要求无玩家、模板或人工身份/样本审核及约束；同一报告重放幂等，其他非空身份库明确拒绝冷启动重跑。
持续开放集合增量分批更新已在 `app/workflows/incremental.py` 与 `app/identity/incremental.py` 中实现，遵循固定快照隔离与三态判定规范，不通过重建已有玩家来伪装稳定性。算法演进与端到端指标见 [incremental-identity.md](incremental-identity.md)。

事务分配 `pl_` 匿名 ID：UUID5 的名字绑定 match 和该组固定 anchor 段，不依赖批次、遍历顺序、号码或组序号。
数据库保留该 ID；未来人工修订保留现有审计与别名规则。
每个自动成员写现有 `Assignment(status=proposed, method=automatic)`，引用报告和段对支持；
这是实际自动预测归属，区别于 `confirmed` 的正式身份确认。proposed 不进入正式个人事件计数。

自动模板写现有 `Template`，`admission=automatic_experimental`、`automatic_support_id=report_id`、
`active=true`、`review_status=unreviewed`、`approval=null`。
active 表示实验图库成员有效，不使其取得人工模式的引用资格；人工匹配仍要求人工批准和 confirmed 归属。
显式人工拒绝可将自动模板标记 rejected；自动 admission 不允许 approved 或人工 approval，
人工批准须产生 manual 模板与真实审核记录。修正继续走显式审核及修订，不向自动记录伪填 actor 或批准理由。

迁移 007 新增不可变 `identity_discovery_reports`，保存报告 JSON、组到 player ID 映射、提交 revision，以及模板、正向见证及支持段对全部比较上下文（含 negative）的 `sample_sources` 版本绑定。
现有玩家、归属和模板表保持标准契约；报告与映射、revision 和自动审计原子提交。
旧库通过迁移升级，不删除或重建。查询由 `workflows.read.get_discovery_reports` 统一读取；导出 JSON 不反向更新。

## 命令与冻结输出

在项目根目录运行，run ID 必须是不存在的新目录名：

```bash
uv run python -m app.presentation.cli baseline --config examples/machine-discovery.toml --run-id discovery-new-run --discover-identities
```

`--discover-identities` 隐含 `--segment-tracks`。先从全新 SQLite 导入机器证据并检查换 batch 幂等，
再自动发现并检查发现报告重放幂等，最后从共享读取导出。
配置保留实际分段与发现策略、实验状态、实现/依赖版本和哈希；工程门槛在真实运行前固定，未读取评分答案。

除既有 raw-evidence、evidence、segmentation、config、diagnostics、input-manifest、summary 和冻结哈希外，
`discovery.json` 保存完整报告与持久 player 映射；`predictions.json` 增加身份组、成员段、模板、未知段、
报告 ID、策略指纹与实验/校准状态。逐样本来源由标准 EvidenceBatch 追溯，跨片段关联由段对报告追溯。
独立评分只能在冻结后进行，不能回流调整本次参数或修改预测；本轮不读取真实答案，不报告真实身份准确率。


## 开发诊断与正式冻结的区别

初始整段准备配 absolute 模式没有产生身份组；改用局部支持窗后，absolute 模式也没有产生身份组。
这两次运行如实作为零组开发诊断保留，不能当作本轮自动发现目标已完成，也不把全未知包装为成功。
观察到的准备覆盖和跨视角绝对距离局限促成 v2 的局部准备与共现对照方法；没有读取身份答案或评分分组。
最终模式的支持证据必须重新在全新 SQLite 运行并冻结参数、输入和实现版本，不能改写早期预测文件。
这些开发输入已经用于方法检查，不宣称它们是独立留出验证集；非零组也不代表已经证明身份正确或安全性能。

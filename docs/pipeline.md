# 业务流水线

## 模块定位

你的职责是把上游的“某片段某条轨迹及其特征”转成持久球员档案，持续维护归属，
再把独立提供的事件关联到当前球员，供检索、展示和下游剪辑使用。
片段数、球员数、导入批次数可变。跨比赛合并不默认开启。
具体距离、候选评分和归人分支固定在 [identity-policy.md](identity-policy.md)，避免各入口自定算法。

当前已实现 machine 无答案输入适配、sample_test 历史人工样例适配（含特征装配）、只读核验、身份核心纯算法、SQLite 存储层
以及工作流业务闭环 v1（workflows.initialize、update、ingest_events、review、read 与 events 关联解析）。
存储接口见 [storage.md](storage.md)，纯算法调用见 [identity-core.md](identity-core.md)。调用 `load_sample_test(settings, batch_id=..., clip_ids=None)`
得到标准 EvidenceBatch，调用 `load_sample_test_input` 同时获得就绪特征值，`match_id` 来自配置，`batch_id` 由调用者提供。
validate CLI 只展示该证据包的计数与警告，不创建球员或数据库；可运行 `examples/workflow_v1.py` 查看完整业务闭环。

## 无标注基线入口（上一阶段默认模式）

运行 `baseline --config examples/machine-baseline.toml --run-id <新名称>`：

1. machine adapter 只从 track/、identity/、video/ 的明确文件白名单发现片段与轨迹，
   核对观测、样本图片、特征行与合同，生成现有 EvidenceBatch 和内存特征映射。
2. 在项目 var 下新建运行目录及 SQLite；不连接任何旧库，不提供人工种子或模板批准。
3. 调用 update_library 的共享 ingest → prepare → associate → resolve → gallery 路径；
   同证据换 batch 重交核验幂等，不应新增证据或 revision。
4. 经共享读取导出预测、诊断、配置与输入清单，保存各输入/输出哈希。
   预测冻结后，未来独立评分才可读取答案，评分代码不得成为推理依赖。

缺少自动分段时，每条原轨迹只生成暂定 Segment。观测缺帧、原始时间、图片、
特征 row_index 与来源位置保留；暂定范围不保证段内同人，不产生身份确认或模板批准。
默认模式使用现有 v1 空图库规则，所有身份待定是有效基线结果；该模式本身不执行自动归人。
默认模式不做自动分段；可选的一致性分段和无种子发现模式见后文。开放集合增量发现与真实身份精度校准仍未实现。

## 轨迹内一致性准备（当前阶段：track-consistency-v3）

`baseline --segment-tracks`（或配合 `--segmentation-version track-consistency-v3`）在上述步骤 1 与 2 之间调用
`identity.consistency.segment_tracks(raw_batch, feature_values, policy=...)`。
输入是原始包、全部结构化 TrackObservation 及已有特征；只用机器证据，不解析外部审核文字。

在阶段2实现的 `track-consistency-v3` 中：
1. **局部连续外观支持窗**：替代旧版全轨全对 near 易误杀正常跑动/转身的弊端。窗口内要求：
   - 相邻样本外观连续（`relation(s_i, s_{i+1}) == "near"`，距离 $\le 0.18$）；
   - 零 far 强反证硬约束（窗口内任意样本对距离 $< 0.45$，绝不用多数票或均值掩盖少数异类）；
   - 中心样本（medoid anchor）支撑（窗口内存在至少一个样本 $m$，与窗口内所有其他样本均 near）；
   - 结构连续性（相邻样本时间间隔 $\le 1.5$s，观测缺失间隔 $\le 0.25$s）。
2. **视角平滑过渡区**：在外观渐变边界提取 `kind="viewpoint_transition_zone"` 与 `reason="viewpoint_transition_uncertainty"` 的未知过渡段，避免将正常转身误判为剧烈断裂或伪造确定性切换帧。
3. **资格解耦**：将“可参与已有身份匹配”（`can_match_existing`）与“足够支持建立新身份”（`can_create_new_identity`）在契约上解耦；2 桶短段标记为 `short_segment_limited_match_support`，允许增量匹配已有图库，但严禁独立聚类建档。
4. **疑似切换判定**：持续外观分离（跨侧全 far）须与局部运动突变或重叠框共同支持才报告 `suspected_identity_switch`；不确定边界保留为 unknown。

新包保留停用父段、派生有效段、原始观测与来源，再调用既有 ingest 路径。
导出原/派生证据、分段前后对照、逐对距离、风险、排除原因、不确定范围与证据不足比例。
`supported` 只表示通过本次一致性门槛，不能直接转成正式身份或模板；
仅分段运行不创建球员；无种子自动发现或增量分批更新需显式开启后述对应模式。
完整策略、CLI 和下游接口见 [track-consistency.md](track-consistency.md)。
针对需要反复调优聚类或匹配参数的多轮实验，支持提供 `--fixed-segmentation <path>` 选项：直接复用已有的 `segmentation.json` 报告（及其同目录下的 `evidence.json`），避免在多轮实验中重复执行耗时的轨迹分段，显著提升消融实验迭代效率。

`sample_test` 仍用于历史人工选择样例和人工复核兼容测试，不是上述无答案入口；
不能为了运行它而把工作区 annotations.csv 移回 test 或放宽读取根。

## 无种子自动身份发现（实验与流形聚类交付）

`baseline --discover-identities` 隐含分段，调用 `workflows.discover.discover_library`：
1. **历史探索模式**：默认 absolute 兼容旧绝对门槛；complete_support_v2/cooccurrence_contrastive 需同片段独立几何共现人物作为负例，但受限于单人镜头负例缺失与非完全图桥接连坐处决，实测仅覆盖 1.5%。
2. **2026-11-08 交付推荐模式（constrained_manifold）**：
   - 采用轨迹级分部位可见度加权池化（`representation.py`），提取高信噪比整轨表征；
   - 构建全局物理时空互斥图（`constraints.py`）：同机位同时刻出现的多框物理上绝对不可为同一人，建立 Must-Not-Link 铁律；
   - 带排他硬约束的层次流形聚类（`clustering.py`）：支持正面 ↔ 侧面 ↔ 背面的连续视角过渡，聚类阈值设为 0.28（完全连接）；
   - 历史运行 `manifold-delivery-001` 下识别 21 位核心跨视频球员，全场 75,465 条观测物理同屏碰撞严格为 0（注：早期报告中 97.22% 活跃段归人率系基于排除全部未知段的浮动分母得出，现已由阶段0统一为固定分母 7 级漏斗评测标准）。

满足证据门槛的组写持久匿名 Player、proposed/automatic Assignment 以及有跨片段原始见证的自动模板；
报告和 player 映射通过迁移 007 保存并与 revision/自动审计原子提交。
自动模板 active/unreviewed，准入标记 automatic_experimental，不能冒充人工批准，
也不回流证明本次归属。结果明确 experimental/unvalidated，proposed 不进入正式个人事件计数。
未知并非“必须先人工确认才能预测”，而是本次证据规则的实际自动拒识输出。

该工作流要求尚无身份和人工审核的库；同一报告重放保留 ID 且不增加 revision，
已有其他发现或人工身份时拒绝单次冷启动重跑。分批到达与非空库更新统一由下述增量更新路径处理。
导出 discovery.json、身份组、成员、模板来源、段对支持和未知原因，连同配置、输入与实现哈希冻结。
规则、工程参数、风险与命令见 [identity-discovery.md](identity-discovery.md)。

## 开放集合增量分批更新（当前阶段：短段匹配与资格解耦）

`baseline --incremental-batches [--batch-size N] [--reverse-batches] [--calibration-profile <path>]` 支持比赛视频片段分批到达时的持续更新：

1. **分批读取与轨迹一致性准备**：自动将机器片段分割为序列批次（默认每批 3 个片段），每批独立生成逐帧观测一致性分段报告（支持 `--segmentation-version track-consistency-v3`）。
2. **冻结校准档案绑定**：支持传入 `--calibration-profile`（由阶段1产出的 `distance-calibration-profile-v1`），在运行时强制绑定特征空间距离门槛（near=0.18, far=0.45），杜绝静默 fallback 到未校准空间。
3. **固定快照匹配与更新**：调用 `workflows.incremental.update_incremental`。算法在单一事务内基于当前 Revision $R$ 的固定图库快照比对，批次内部决策隔离；当批次提交时，原子写入新球员、归属与模板并推进到 Revision $R+1$。
4. **三态判定与短段匹配门槛修复**：
   - `matched_existing`：与图库中已有球员模板达成强共现对照或近距离匹配，直接关联已有 `player_id`，杜绝重复建档。在阶段2中修复了短段增量匹配门槛：具备 2 个时间桶的短段（标记为 `short_segment_limited_match_support`）在满足 `min_match_buckets=2` 与 `short_segment_match_threshold` 时，允许匹配已有图库球员；
   - `created_new_identity`：未分配池跨片段聚类，多段完全图且与已有球员充分分离，生成持久稳定匿名 `player_id`。**重要安全约束**：未匹配的短段严格受 `can_create_new_identity=False` 拦截，绝不独立聚类建档，杜绝单视角/短段碎片化污染图库；
   - `insufficient_evidence`：单次出现或证据不足段保持待定，无匹配或距离远绝不单独建立新人。
5. **自动模板准入与退场**：标记为 `admission="automatic_experimental"`，附带多原始样本见证和来源绑定；人工复核或重新分配时模板自动失效。
6. **重放幂等与重排测试**：每批导入后自动提交重放副本，核验 revision 不变且 `changed=False`；支持 `--reverse-batches` 评估时序到达顺序对未分配池和桥接阻断的影响。
7. **增量漏斗累积单调性修复（2026-11-10 专项修复 R3）**：在 baseline 增量运行中汇总全批次分段报告，构建全局 `combined_seg_report` 注入评测漏斗，彻底修复原先仅取最后一批分段报告导致的漏斗倒挂缺陷（修复前 63 -> 166，修复后 452 -> 166 -> 141，保持严格物理单调）。
8. **部分复制与校准一致性防御（2026-11-10 专项修复 R1, R2, R4）**：运行强制校验校准档案数值一致性断言（R1）；校准证据严格过滤低可见度部位（`min_part_visibility`）、缺失观测/框以及显式混人风险（R2）；段对评估激活非共享组件基数（$\ge 3$）、双向覆盖率（$\ge 0.8$）及唯一见证人（$\ge 2$）硬约束，阻断跨片部分复制序列欺骗（R4）。
规则、接口与诊断见 [incremental-identity.md](incremental-identity.md)。

## 共同入口和分支

```text
同伴A：轨迹 + 人物图 + ReID特征 + 可选队伍/号码
同伴B：动作/进球等事件 + 时间 + 主体轨迹引用
                       ↓ 可分别到达
adapters：外部文件 → 标准 EvidenceBatch
                       ↓
workflows.ingest：格式/来源/版本/重复提交检查
            ┌──────────┴──────────┐
            ↓                     ↓
     人物身份证据分支          事件证据分支
     prepare 清理/分段         validate 校验时间与主体
     associate 查已有身份      登记事件原始事实
     resolve 确定归属/新建/待定  允许暂时没有球员归属
     gallery 更新可用模板           │
            └──────────┬──────────┘
                       ↓
      保存证据 + 身份状态 + 图库 + 修订/审计
                       ↓
events.associate：按当前身份段解析每个事件主体
                       ↓
workflows.read：档案 / 事件时间线 / 统计 / 待复核项
                       ↓
API / CLI / 网页 / 下游剪辑查询
```

各分支只在有相应输入时执行。事件-only 批次不做 ReID，人物-only 批次不要求动作结果。
所有待写内容先计算和校验，最终在版本检查后的一个事务内提交；失败不留下半批正式状态。
事件当前关联由共享读取规则派生；不另外维护一份独立、可能过期的 event.player_id。

机器适配保留全部实际轨迹观测帧和原文件引用，缺帧不补观测。
历史 sample_test 适配的人工选择语义单独保留：
人工目标选择只产生待确认的身份段，不等于整段身份或模板资格已确认。
人工给定的进球时刻转换为 made 点事件，号码留在事件主体，不向整轨迹或样本传播。

## 人物分支的四步

| 步骤 | 输入 | 产物 | 代码 |
| --- | --- | --- | --- |
| prepare | 轨迹、样本、质量与时间 | 可比较身份段、合格模板候选、风险项 | identity/prepare.py |
| associate | 新身份段 + 当前图库快照 | 每段已有球员候选、距离、共同可见部位 | identity/associate.py |
| resolve | 候选 + 队伍号码等证据 + 人工约束 | 关联已有 / 新身份提案 / 待定 / 拒绝 | identity/resolve.py |
| gallery | 已接受的身份决策 + 合格样本 | 模板增加、替换或退出计划 | identity/gallery.py |

低质量样本退出图库不等于人物不存在。未匹配不是“发现新球员”；
新建身份需要可信的独立支持或人工确认。模板更新在当前批匹配完成后生效。
混人轨迹先隔离/拆段；无法定位切换边界时保留待复核，不凭猜测自动切分。
v1 新建身份由人工明确确认，图库只激活人工批准的合格模板；自动归人的样本不自动入图库。
候选与正式确认分开，未校准时仍可推荐候选，但不自动确认或据此计算个人进球数。

## 初始化

入口：workflows.initialize.initialize_library（已实现）。

1. 创建或读取比赛作用域，检查尚未建立初始身份锚点；已有事件或待定证据不阻止初始化。
2. 走共同的接入、校验和 prepare 路径。
3. v1 根据人工确认的身份段和干净模板建立第一批档案，构造固定的初始图库快照。
4. 其余证据只与该快照进行一致性判断；本批自动关联得到的模板不反向用于本批匹配。
   接受明确决定，其余保留待定。
5. 保存第一版球员 ID、模板和待复核项，关联已到达的事件。

初始化允许只识别出一部分球员，不要求凑齐上场人数。
重复提交同一证据幂等；已有初始化状态时不清库，新的证据走增量入口。

## 增量更新

入口：workflows.update.update_library（已实现）。

1. 读取同一 match_id 下的当前 revision 和图库快照。
2. 接收新的 EvidenceBatch；先识别重复证据，再 prepare。
3. 与已有身份关联，形成已有球员归属、新身份提案或待复核项。
4. 接受满足策略的决定；保留旧 ID、人工修订和所有历史证据。
5. 更新合格模板与修订，当前事件归属随映射一起反映到查询。

新增球员、补充已知球员的新视角、拒绝污染模板、晚到的人物证据都属于更新。
初始化和增量调用同一组身份规则，只有是否已有锚点及身份状态的前置条件不同。

## 独立事件导入

入口：workflows.update.ingest_events（已实现）。
只进行适配、事件校验、来源去重和登记，不强制跑人物分支。
事件先到则保存未关联的原主体引用；人物证据到达后，同一事件自动显示当前归属。
晚到片段揭示事件本身越界时，保留证据并进入复核，不阻止合法人物接入、不计正式统计。
不变的历史事件跨批重交保持幂等；新增/更新事件仍按全部已知上下文严格校验。
进球结果来自上游或人工证据；普通投篮动作默认不能推断命中。

## 人工维护

入口：workflows.review.apply_review（已实现）。
支持确认种子/匹配、新建档案、拆段、合并、改归属、模板批准/拒绝和可追溯恢复。
维护路径：读取当前版本 → 生成变更计划 → 检查约束 → 原子保存 → 查询当前事件与统计。
重新匹配不覆盖人工决定；拒绝模板只影响图库资格，身份与事件按各自证据保留。
恢复从目标修订的有效状态生成新修订，统一恢复段的生效与替换关系、归属、模板、
约束和审核；反复恢复使用实际状态清单，完整保留中间历史，失败原子回滚。
清洁审核绑定具体样本来源版本；样本升级不转授批准，人工段级身份继续保留。

## 稳定输出

- 球员档案及稳定 player_id、代表图片、当前模板和来源。
- 每个身份段当前属于谁、依据和待复核状态。
- 按球员、动作、结果、片段或原视频时间过滤的事件及媒体定位。
- 当前 revision、输入处理报告和修改审计。

这些输出既供网页使用，也能供其他同伴的剪辑模块调用。

历史早期探索中曾采用 constrained_manifold 聚类识别 21 位跨视频球员（早期基于浮动分母报告 97.22% 覆盖率，现已由阶段0纠正为固定分母漏斗标准）。当前最新基线遵循阶段0~2完整修复闭环（包含 2026-11-10 专项漏洞加固），基线运行报告见 `var/identity-repair/stage-2/handoff.txt` 与 [docs/incremental-identity.md](incremental-identity.md)。

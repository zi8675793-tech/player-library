# 通用模块实施计划

## 当前核心目标与差距（2026-11-06 更新）

用户明确要求：没有人工身份答案、从空库自动跨视频归人；人工标注只用于预测完成后的结果检查。
原有里程碑主要验证证据管理与人机协同流程；本轮已补无种子冷启动实验发现，不能等同于真实身份精度已校准。
现有 v1 人工种子/人工模板策略是待扩展的实现边界；后续设计须同步修改身份策略、契约与调用者。

本次审查确认：

- `annotations.csv` 已移到工作区根目录。旧 sample_test 适配器仍强制从 test 中读取它，现有 validate 入口实际因缺少该文件失败；不能通过移回文件或扩大输入根解决。
- test/target 仍含人工目标选择、人工帧段与筛选后的样本/特征，不是独立的无答案输入。过去演练只处理 11 段、118 个目标样本；track/identity 中实际有 158 条保留轨迹、1,759 个特征样本，覆盖 10 个片段。
- 只读取 track/identity、全新内存库、零人工种子/复核的结构探测，得到 0 个球员、0 个活跃模板、0 个候选、158 个待定归属。探测按原轨迹首尾生成暂定段，不证明段内身份连续，也没有验证自动风险检测或识别精度。输入文件哈希和结果见 `var/reviews/no-annotation-gap-audit-20261105.json`。该临时探测不等于已实现生产级原始数据适配器。
- eval_session_1 验证零种子时保留待定；eval_session_2 使用人工已知的两人图库进行距离诊断和模拟阈值扫描；eval_session_3 使用人工种子、归属确认和模板批准。三者均不是无人工身份答案的自动归人验收。
- 当前 annotations.csv 只有 10 条事件/目标轨迹记录，person_gt_id 和 team_id 均为空；它可用于有限范围的结果核查，不能直接提供全部 158 条轨迹的逐段身份真值。已反复用于诊断的样例不作为未经接触的最终留出集。

## 上一阶段交付：无答案入口与可复现基线（2026-11-06）

已实现 machine 适配器，自动发现 track/identity/video 的机器片段、全部轨迹与样本，
通过明确文件白名单核对原观测、图片、特征索引、合同和来源 SHA-256。
输出仍为 EvidenceBatch 与现有 InlineFeature 内存映射，不建立另一套数据模型。
原轨迹首尾只产生暂定 Segment，不保证段内同人，不充当身份确认或模板批准。

`baseline` CLI 在项目 var 下创建全新且不可覆盖的运行目录和 SQLite，
调用 update_library 的既有共享工作流，无种子、无人工审核与旧图库加载。
它导出标准证据、Segment/Assignment 预测、诊断、配置、输入清单、摘要与冻结哈希，
并换 batch 重复导入核验幂等。无标注可运行、规模/维度可变、索引哈希、路径隔离和
答案变化不影响预测由必要测试验证。真实输入规模和本次结果随运行产物保存。
全部待定是空库基线的如实结果，不能称为自动归人成功或识别准确率。

该阶段只完成下列优先项 1 的机器接入、冻结和评分接口预留，未实现答案读取或评分器；
当时自动分段、零种子身份发现与自动图库、开放集合更新、真实自动识别验收四个阶段未启动。
旧 sample_test/eval_session_1/2/3 仅作历史流程和人工复核演练，不再作为自动识别验收。

本次真实运行：10 个片段、158 条轨迹/暂定段、75,465 个观测、1,759 个样本/特征行、
1,829 个来源文件；0 球员/模板/候选，158 段全部 pending，人工干预为 0。
同证据换 batch 重交 changed=false，revision 仍为 1。详见 [机器基线记录](../examples/machine-baseline.md)。

按核心目标优先补齐：

1. **无答案输入与独立评分**：接入机器生成的 video/track/identity，明确输入白名单；推理不读 target、标注、旧库或硬编码分组。冻结预测与配置后评分，评分中的匿名 ID 对齐不回流推理；覆盖标注缺失、答案变更不改变预测、批次重排等验收行为。
2. **自动身份段准备（当前已实现保守首版）**：从原始轨迹检查混人、身份切换、重复检测及样本质量；自动提出有证据的拆段或隔离未知范围，不使用人工 target_frame_ranges，不能假定整条 track 属于同一人。合成规则验证不等于真实分段精度已验证，调用与局限见 [track-consistency.md](track-consistency.md)。
3. **零种子身份发现与自动图库（本轮已实现冷启动实验首版）**：在人数未知时形成跨片段身份组并建立匿名稳定 player_id；为自动支持证据和模板准入定义真实来源及质量门槛，不能伪装成人工批准，也不能仅打开现有 auto_assign_enabled 开关。
4. **开放集合增量更新**：自动区分已有人、新人及证据不足；控制误合并、重复建档和模板污染，维持跨批稳定 ID，保留修正审计。当前批匹配仍使用固定快照。
5. **真实自动识别验收**：明确目标标注覆盖范围，报告跨片段同人关联、误合并、误拆分、自动覆盖与未知比例，单列人工干预数量；加入更多人物、新出场者、同队相似球员和混人段，使用独立留出数据评估泛化。若验证事件归属，事件主体也须来自独立上游，不能把人工射手轨迹和进球结果混算成自动识别成果。

## 当前交付：轨迹内一致性准备（2026-11-06）

优先项 2 已实现保守的机器证据路径：结构化逐帧 TrackObservation 及 SQLite 迁移 006、
可解释全对样本比较、时间支持、可见性与质量排除、重复检测候选、观测间断/运动突变检查、
supported/unknown 派生 Segment 和不确定区间。`baseline --segment-tracks` 接入同一工作流，
冻结分段前后证据、报告、风险、排除和配置；不读取身份答案，不生成姓名、号码或 Player。
验收覆盖同人换视角、真正混人、少数异类、遮挡、重复框、稀疏采样和无特征等合成情况。
工程门槛未经过真实身份校准，supported 不表示已证明同人；真实误拆分与漏切换仍需独立检查。
完整定义和提供给优先项 3 的接口见 [track-consistency.md](track-consistency.md)。
优先项 3 的冷启动实验路径与优先项 4 的持续开放集合增量分批更新现已补齐，详见后文及 [incremental-identity.md](incremental-identity.md)；独立真实精度验收仍属独立评分阶段。生成多个段本身不作为自动识别成绩。

## 当前交付：无种子跨片段身份发现（2026-11-06）

新增 `complete_support_v1` absolute 及 `complete_support_v2` 共现对照固定证据算法、类型化发现契约、settings.discovery、
空库发现工作流、迁移 007 的不可变支持报告及 CLI `--discover-identities`。
默认 v1 分段保留；自动发现配置显式使用 v2 连续局部支持窗，从全段外观失败中保留合格局部证据，
不跳过少数异类，窗外未知、不推断换人，段内距离门槛保持不变。
跨视角对照模式使用候选片段的可靠空间共现人物作为独立 negative，
最近两原始样本相对所有 negative 达到 margin 才投支持票，缺 negative 不强行输出；
段内 tau 不再冒充跨视角身份阈值，实验宽上界与支持集合逐样本保存。
以双向多样本、全局重复组件剔除、少数矛盾否决、组内全对约束、桥接隔离与竞争差距控制误合并。
人数由证据决定；跨片段组写持久匿名 player_id，单次出现及不确定段保留未知。
模板准入保留跨片原始样本见证，自动选择与人工批准明确区分，防止自比较及模板反馈循环。

自动预测保存为 proposed/automatic，支持报告与配置明确 experimental/unvalidated，
不伪造独立校准，也不要求先人工批准才输出身份组。人工模式和正式事件计数语义保留。
验收覆盖可变人数、同队相似人物、单次出现、重复图片/整段复制、桥接矛盾、少数异类和陈旧报告；
真实运行在参数固定后从新目录/新 SQLite 完成，结果与命令单独记录，未读取身份评分答案。
初始整段/absolute 与局部窗/absolute 的零组运行保留为开发诊断，均不作为完成目标；
最终对照模式须另跑新 SQLite 冻结，不读取答案、不宣称独立留出精度。
详细接口与局限见 [identity-discovery.md](identity-discovery.md)。

## 当前交付：开放集合增量身份更新与图库维护（2026-11-06）

优先项 4 已全面实现并交付：
1. **纯算法增量编排** (`app/identity/incremental.py`)：明确三态判定（`matched_existing`, `created_new_identity`, `insufficient_evidence`）。无匹配或距离远绝不单独建立新人；未分配池跨批次累积，支持未知证据逐批累积后形成独立跨片段强支持新身份；已发现集团与图库现有球员做中位数比对，避免同一人反复建档。
2. **固定快照与原子事务更新** (`app/workflows/incremental.py`)：每次更新基于 Revision $R$ 的固定图库快照比对，批次内部决策隔离，批次完成后在单一 SQLite 事务内原子写入 assignments、players、templates 和不可变 DiscoveryReport，并推进版本至 Revision $R+1$。
3. **自动模板准入与退场审计**：标记为 `admission="automatic_experimental"`, `review_status="unreviewed"`, `approval=None`，杜绝伪造人工批准；跨片段多原始样本见证绑定；人工复核或重映射时关联模板自动退场。
4. **人工优先级与事件动态解析**：人工确认归属与人工约束（`block_automatic`, `forbid_player`, `cannot_link`）具有绝对优先级；事件读取机制保持不变，身份状态变更后事件主体动态绑定最新确认身份，没有独立事件输入时不补造事件。
5. **十项关键场景验证**：覆盖重复批次幂等、批次重排、旧人回归、新人发现、未知项补充、新视角、混人段、模型特征空间不兼容阻断、错误模板撤销及事务回滚（`tests/test_incremental_workflow.py`，11 项全部通过）。
6. **分批实际运行与顺序敏感性分析**：基于机器真实输入（10 片段、158 轨迹、75,465 观测、1,759 样本）完成批大小为 3 的正向分批运行（`incremental-20261106-run1`）与逆向分批运行（`incremental-20261106-run2-reversed`），冻结全套产物并深入分析桥接阻断与时序池可见性带来的顺序敏感差异，详见 [incremental-identity.md](incremental-identity.md)。

当前独立真实身份精度校准（优先项 5）、HTTP 接口和前端展示（M6）仍需后续实现。

当前已实现：
- M1：契约、配置、machine 无答案白名单适配及 baseline CLI；sample_test 历史人工样例适配和核验 CLI；
- M2：SQLite 存储层（迁移 001–008、来源版本幂等、事务原子提交、一致快照读取）；
- M3/M4：身份核心纯算法（prepare、distance、associate、resolve、gallery plan）；
- 工作流业务闭环 v1：workflows（ingest、initialize、update、review、read）与 events（validate、associate），串联冷启动初始化建档、增量归人、人工复核、模板维护与当前事件绑定查询，并通过全套跨模块集成测试；
- 运行示例：examples/workflow_v1.py；
- 历史人工目标材料的零种子接入演练（对话 1，非自动识别验收）：通过 `eval_session_1.py` 使用真实数据完成零种子冷启动与存储核验（TC-1.1 ~ TC-1.4 全部通过，产物位于 `var/test_session_1.sqlite3`，审计报告见 `examples/sample-test.md`）；
- 历史人工目标特征空间几何诊断与模拟阈值扫描（对话 2，非自动识别验收、未完成独立身份校准）：通过 `eval_session_2.py` 完成 11x11 轨迹段两两矩阵诊断、类内紧凑性/类间分离度评估（Margin=+0.0593）、Clip 68/3/30 病理诊断与网格扫描（报告位于 `var/session_2_diagnostic_report.md`）；
- 人机协同增量建库与事件解析闭环业务演练（对话 3，非自动识别验收）：通过 `eval_session_3.py` 完成以 Clip 20（27号）与 Clip 11（15号）为种子的建库初始化、分批增量导入与算法候选推荐核验、apply_review 协同复核与模板图库维护、Clip 68 双段死锁解耦与质检防御，最终闭环验证 Player-27 拥有完整 6 条进球、Player-15 拥有完整 4 条进球（产物位于 `var/session_3_delivery.sqlite3`，报告见 `var/session_3_delivery_report.md` 与 `/Users/ahs/Documents/agc/session_3_delivery_report.md`）。

仍待完成：
- M1 中 normalized 适配器的标准外部输入边界；
- M6 接口和展示层（HTTP FastAPI、原生前端页面和下游剪辑导出）。

## M1 标准证据和输入适配

实现contracts、settings、adapters/normalized和adapters/sample_test的读取校验边界，
提供presentation/cli的只读核验入口。身份与事件都允许单独提交。

- 已完成：`app/contracts/` 的证据、媒体、特征空间、身份、事件和接口模型及局部校验；
  `app/settings.py` 的 TOML 类型、必需项、数值范围与配置目录相对路径解析。
- 局部校验增强：拒绝实体列表内重复主键（相同内容也拒绝），并检查同包已知的
  Sample 对 Segment/Track 的作用域与定位范围、Clip 时间/帧界限、事件及显式主体引用的本地时间和媒体类型。
  同 file_id 的嵌套 SourceFile 检查信息兼容：path 相同，sha256/size_bytes 双方已知时不得冲突；
  null 与具体值兼容，且不回填或修改原声明。sources 列表内重复 file_id 仍拒绝；
  IdentityEvidence 独立使用时也检查其内部文件引用。
  未知/晚到目标继续允许；Segment 选择范围不要求被 Track 实际观测首尾包含，
  不校验样本帧是否在 observed_frame_indices 中，也不在模型中强制 CFR 帧时等式；
  省略主体引用 time 时不提前执行事件归属或缺帧定位。
- 本次完成：`adapters/sample_test.py` 只读接入 CSV、target 元数据/轨迹/样本/NPZ、
  引用图片与原视频，检查根目录边界、哈希、数组和 JSONL 对齐及帧/时间范围；
  生成稳定来源与业务 ID，保留观测缺帧、质量警告、待确认身份和事件级号码。
  `presentation/cli.py validate` 只调用配置和适配器，支持全部或部分片段，输出计数与警告。
- 已覆盖：`tests/test_contracts.py`、`tests/test_settings.py` 中的 JSON 往返、独立/混合证据、
  晚到引用、可变特征形状、非法时间/数值和不同工作目录下的路径一致性；
  `tests/test_contract_integrity.py` 覆盖批内重复主键、同包已知引用冲突与范围、
  跨片段本地坐标及来源文件信息兼容性。
- 本次覆盖：`tests/test_sample_test.py` 的样例统计、子集与稳定 ID、临时输入的文件/数组/
  路径失败检查及真实输入只读核对；`tests/test_cli.py` 的参数转交、摘要与失败退出。
- 待完成：`adapters/normalized.py` 的标准证据读取边界。M1 未整体完成；
  此段描述的是 M1 交付范围；其后的身份核心和 SQLite 存储进度见上文及 M2。

验收：一般证据包支持可变片段数、可变球员候选数、每片多个事件或无事件；
人物/事件缺省不表示删除。样例adapter报告examples/sample-test.md中的数量和风险，
通用模块不出现样例目录、号码、固定6/128布局。

## M2 持久证据和比赛作用域

storage 迁移、仓储、来源幂等、原子事务及一致读取已实现，调用见 [storage.md](storage.md)。
workflows/ingest 和 read 已实现；自动发现报告也通过共享读路径导出。
区分match、batch、clip、identity segment、feature space和event subject。

验收：跨batch重复提交也幂等；同match补新片段保留旧状态；跨match不误连；
同源同版本被改写拒绝，新版本显式登记；异常不留下半批结果，输入只读。

## M3 初始化与人工档案维护

实现identity/prepare、gallery以及initialize/review，人工确认种子可先跑通。
支持可变人数、未知段、样本批准/拒绝、拆段和稳定player_id。
先将质量风险及分数语义结构化，禁止通用身份规则解析 adapter 的 notes 文本。
v1 人工建立身份锚点和新球员；仅人工批准且归属已确认的模板进入正式图库。

验收：初始化可只发现部分球员，继续更新不重置ID；已有库initialize不清库；
轨迹混人可以隔离和拆分，样本模糊不等于身份错误；人工修改有审计和版本检查。

## M4 增量匹配和模板更新

实现identity/associate、resolve、distance和workflows/update。
初始化与增量共享规则；先用明确种子，再向同场持续导入新证据。
顺序为部位距离 → 多模板/多样本候选及诊断 → 保守决策 → 预览和人工确认闭环。
未标定时保持 auto_assign_enabled=false；完成按片段隔离的校准和验证后，再开放自动确认已有球员。
距离远不自动新建；单候选、缺模板、证据覆盖不足和强支持分歧均保留复核。

验收：已有/新身份提案/待定/拒绝分开；未知不自动建人；新出场球员可经确认加入；
固定图库快照匹配，合格模板随后更新；模型变更不重编号，空间不兼容不比较。
按独立身份标注报告关联精度、召回、误合并和待复核比例，不按固定人数评成绩。

## M5 一般事件关联与当前时间线

实现events/validate、associate、update.ingest_events和共享读取。
支持点/区间、多角色、未知命中结果、事件先到和晚到。

验收：事件-only不执行重识别；先到事件可未关联保存，身份到达后同一事件得到归属；
拆段/合并后查询同步变化；射手和助攻者可不同人；区间跨身份切换保留歧义。
样例68只登记一次进球，20保留3.5秒；有无全场来源映射均能正确定位。

## M6 接口和展示

实现presentation/api、main和static，共享workflows。
展示任意数量档案、模板、事件、统计、来源与复核入口；支持按球员和事件属性筛选。

验收：UI/CLI/导出查询口径一致；受控媒体播放；两场比赛和多批更新不串档；
样例最终经人工身份核验后，可展示预期两人及6/4条给定进球记录。

## 后续独立需求

精剪和合集渲染、自动命中识别、跨比赛永久球员身份、整场容量优化分别立项。
本模块始终提供稳定的身份、事件与媒体定位接口，不要求重新实现上游视频理解模型。

## AI开发节奏

每次给出一个用例、输入和验收行为，先读AGENTS及契约，再修改对应流程模块。
改契约同步所有调用者；必要测试通过后保存Git提交。先跑业务闭环，再接展示。
不以建立更多目录或补满空文件作为功能完成标准。


## 2026-11-05 补充验收修复

本轮修复拆段恢复、恢复到恢复版本、晚到片段揭示事件越界、晚到轨迹下的重复事件幂等、
新版样本继承清洁审核五项问题。恢复使用追加修订与有效状态清单，不删除历史审核；
迁移与旧库限制见 [storage.md](storage.md)。回归测试以正确身份、统计、审核与原子性为断言。
三份演练隔离重跑结果见 `var/reviews/fixes-20261105/`。
6/4 是固定人工身份确认后的流程结果，不能视为自动识别准确率；仍有 109 项模板、4 项风险待复核。


2026-11-06 历史冷启动基线运行：1 个实验身份组（片段 20/T1 ↔ 6/T5）、2 个成员段、861 个未知段，见 examples/machine-discovery.md。

2026-11-08 历史真实冷启动运行 `manifold-delivery-001` 存档说明：
该版本为时空受约束层次流形聚类（`constrained_manifold`，聚类阈值 0.28）的早期实验交付，历史报告见 `/Users/ahs/Documents/agc/manifold_identity_discovery_final_delivery.md`。2026-11-09 复测（`var/reviews/acceptance-20261109/acceptance-report.txt`）确认其 97.22% 覆盖率系浮动分母所致。

## 2026-11-09 身份修复计划：阶段0（统一覆盖漏斗、基线修正与工具缺陷修复）已完成

本阶段落实了第一轮身份修复计划中的阶段0，已全部通过验证并冻结交付：

1. **统一 7 级覆盖漏斗与多维损失归因**（`app/evaluation/funnel.py`）：
   - 确立统一固定分母漏斗：`raw_tracks` (158条, 1346.35s) $\to$ `usable_samples` (1759个) $\to$ `locally_consistent_segments` (51段, 222样本) $\to$ `cross_clip_comparable_segments` (42段) $\to$ `candidate_matches` (30段, 128样本) $\to$ `final_assigned_identities` (30段, 128样本) $\to$ `templates_admitted` (21模板)。
   - 提供按 clip/track/reason（带共现追踪）/batch 的精细化损失归因，运行自动导出 `funnel.json` 并整合入 `summary.json`。
2. **修复评测工具已知缺陷**：
   - 严格半开时间区间 $[start\_s, end\_s)$（`app/evaluation/units.py`）。
   - F1 边界公式修正：采用 $2\text{TP}/(2\text{TP}+\text{FP}+\text{FN})$，TP=0 且分母>0 时严格返回 0.0，杜绝零预测冒充 100% 精确（`app/evaluation/metrics.py`）。
   - 身份评估统一对齐持久 `player_id`，`group_id` 仅作实验分组统计。
   - 混合 `track_id` 稳定排序（`_track_sort_key`），解决数字与字母字符串混用时的 `TypeError`。
   - 剔除完全复制图片序列的伪独立支持（组件级哈希集合去重，100% 重合时置为 `insufficient_evidence` 并拒绝生成虚假模板 witness）。
3. **验收入口重塑**：
   - 改造 `tests/test_manifold_delivery_acceptance.py`，解耦旧静态数据库硬断言，增加 7 项可变规模（2/3/4片）、零物理碰撞与幂等性的动态业务测试。
4. **固定分母基线重跑与独立审计**：
   - 固定以 10 个片段机器证据（158 轨迹, 1759 样本, 75465 观测, 1346.35s）为唯一固定分母；
   - 复跑 Cold（样本覆盖 7.28%, 时长覆盖 3.59%）、Forward（批次大小3，样本覆盖 6.08%）、Reverse（批次大小3，样本覆盖 4.83%）；
   - 时空同屏碰撞严格为 0；Target 已分配样本准确率 78.57%；
   - 详细产物、无真值机器审计报告及独立评分见 `var/identity-repair/stage-0/`。
5. **交接与产物**：
   - 阶段0详细交接文档位于 `var/identity-repair/stage-0/handoff.txt`。

## 2026-11-09 身份修复计划：阶段1（空间绑定距离校准机制与正负样本离散度诊断）已完成

本阶段落实了第一轮身份修复计划中的阶段1，已全部通过验证并冻结交付：

1. **实现特征空间绑定的距离校准档案契约 (`DistanceCalibrationProfile`)**：
   - 数据契约（schema: `distance-calibration-profile-v1`，`app/contracts/calibration.py`），绑定 `profile_id`、`feature_space_id`、`calibration_version`、`distance_semantics`、`status`、`near_distance`、`far_distance`、`evidence_source` 与统计分位数。
   - 实现 SHA-256 `fingerprint` 属性，支持跨运行指纹校验。
   - 确立空间绑定铁律：禁止跨特征空间静默 fallback，遇到未校准空间显式抛出 `FeatureValidationError`。
2. **四态距离语义关系与判定工具**：
   - 在 `app/identity/distance.py` 中导出 `DistanceRelation` 枚举（`near`、`far`、`ambiguous`、`uncomparable`）；实现纯函数 `evaluate_feature_relation`。
   - 明确 near 不等于同人真值，单个 far 观测不能直接判为换人。
3. **可复现候选校准证据提取与结构化排除引擎**（`app/identity/calibration.py`）：
   - 候选正例（同一轨迹 $\Delta t \le 1.0$s 高置信观测）：排除时间跨度过大、重复图片/描述子、空间跳跃、遮挡混人及部件不可比；
   - 互斥物理负例（同片段同帧不同轨迹检测框）：通过 IoU $\ge 0.2$ 过滤双检框排除，排除重叠、空间不足、重复图片及部件不可比；
   - 结构化排除审计：每对被排除样本均记录 `CandidatePairExclusion`。
4. **真实开发集特征空间诊断与正负样本离散度实证**：
   - 候选正例（1,113对）：min=0.0224, p50=0.1958, p90=0.3642, max=0.5757, mean=0.2091；
   - 物理负例（890对）：min=0.1652, p10=0.3067, p50=0.3717, max=0.6211, mean=0.3774；
   - 实证发现正负样本在区间 [0.1652, 0.5757] 严重重叠！同队不同球员球衣相近时最小负例仅 0.1652。如实保留 `status="uncalibrated"`，保持工程门槛 near=0.18, far=0.45，坚决不盲目调高全局 near，不调整聚类门槛（0.28），不捏造已校准成功。
5. **策略冻结、基线复跑与独立评分**：
   - 产物固化为 `var/identity-repair/stage-1/calibration-profile.json` 与 `calibration-diagnostic.json`。
   - 统一绑定到 cold、forward、reverse 运行；执行 `machine_audit.py` 与 `score_frozen_answers.py`；
   - 阶段1详细交接文档位于 `var/identity-repair/stage-1/handoff.txt`。

## 2026-11-09 身份修复计划：阶段2（重做轨迹内局部视角一致性 v3 与增量资格解耦）已完成

本阶段落实了第一轮身份修复计划中的阶段2（第一轮修复的收官阶段），已全面通过验证并冻结交付：

1. **核心问题实证与肯定答复**：
   - 核心问题：“在不让明确混人通过的前提下，能否从 unknown 恢复正常同人局部证据？”
   - 实证结论：**明确成立**。在相同固定分母下，阶段2局部一致段样本覆盖率从阶段1的 12.6% (222) 恢复至 25.7% (452，+103.6%)，观测覆盖率从 6.7% 恢复至 14.0% (+108.6%)，有效时长覆盖率从 6.5% 恢复至 13.3% (+106.4%)；同时坚持零 far 强反证硬约束（窗口内只要存在一对距离 $\ge 0.45$ 即一票否决，绝不用多数票或均值掩盖），9 项合成对抗场景 100% 拦截混人，独立评分 Target 跨片段对偶精度保持 100.0%（FP=0，零误合并）。
2. **实现第三代局部连续外观一致性 (`track-consistency-v3`, `app/identity/consistency.py`)**：
   - 局部连续外观支持窗：相邻样本外观连续（$\le 0.18$）、零 far 强反证一票否决、中心样本（medoid anchor）支撑（存在至少一个样本与窗口内所有其他样本均 near）、采样与观测时间连续；
   - 平滑视角过渡区（`enable_transition_zones`）：在渐变边界提取 `viewpoint_transition_zone` 与 `viewpoint_transition_uncertainty`，避免将渐进转身误判为剧烈断裂；
   - 风险局部收敛（`scope_risks_to_breaks`）：运动跳变等质量风险收敛于边界局部，干净支持子段不再连带继承父段跳动风险；
   - 相对时间分桶（`ensure_relative_buckets`）：段内相对分桶对齐，消除绝对时间抖动；
   - 疑似身份切换（`suspected_identity_switch`）：持续外观分离结合运动突变或重叠框，不确定边界完整保留为 unknown，不伪造精确切换帧。
3. **修复漏斗第4级跨片段可比段 (`cross_clip_comparable_segments`) 统计缺陷**：
   - 修复前 `app/evaluation/funnel.py` 中误用负向条件导致所有未决 unknown 段均被计入第4级，样本与时长虚假回弹至 100%；
   - 修复后采用正向白名单校验未分配原因，严格限定为通过单段合格检查的 43 段（208 样本 / 85.82 秒），漏斗完全恢复物理单调递减（1759 $\to$ 1759 $\to$ 452 $\to$ 208 $\to$ 140 $\to$ 140 $\to$ 20）。
4. **资格解耦与短段增量匹配门槛修复**：
   - 契约解耦：`SegmentProposal` 增加 `can_match_existing` 与 `can_create_new_identity`；
   - 2 桶短段标记为 `short_segment_limited_match_support`，严禁其独立建档，杜绝图库碎片化；
   - `DiscoveryPolicy` 增加 `min_match_buckets`（默认 2）与 `short_segment_match_threshold`；允许短段增量匹配已有球员，解除原 3 桶硬阻断；
   - 增量表现：Forward 增量样本覆盖率提升至 8.02%~9.44%，Reverse 增量样本覆盖率提升至 10.52%~10.92%，批次间 Jaccard 一致性提升至 0.553，ARI 达 0.883。
5. **2026-11-10 专项漏洞加固（闭环 4 项关键审查缺陷）**：
   - **R1 校准档案强一致性断言**：在 discovery, segmentation, consistency, incremental, baseline 等入口建立硬性数值断言，杜绝参数静默旁路与指纹脱钩；
   - **R2 校准证据严格准入过滤**：在 `kpr_distance` 中引入 `min_part_visibility` 硬过滤，排斥低于门槛的部位；并在校准对构建阶段对 explicit_identity_risk (mixed_identity, identity_conflict, suspected_identity_switch) 以及缺失 observation/bbox 的样本实施硬排除；
   - **R3 增量漏斗多批分段报告累积合并**：在 baseline 增量运行中汇总全批次 segmentation reports，构建真实的全局 `combined_seg_report` 注入漏斗评测，彻底修复增量模式下漏斗倒挂缺陷（修复前 63 -> 166，修复后 452 -> 166 -> 141，保持严格单调递减）；
   - **R4 部分复制独立组件门槛与覆盖率激活**：激活 segment pair 评估中的 `unshared_left >= 3` 与 `unshared_right >= 3` 独立组件基数硬约束，激活 `directional coverage >= 0.8` 约束，严格要求 template unique witnesses $\ge 2$，阻断跨片段部分复制样本欺骗。
6. **测试套件与基线复跑**：
   - 累计新增 6 项专项回归测试（覆盖漏斗第4级跨片比较、增量漏斗单调性、短段增量匹配与建档门控、校准档案篡改旁路防御、校准证据严格准入过滤、部分复制阻断）；
   - 全部 701 项测试通过（700 passed, 1 skipped in 11.8s），Ruff 检查与格式化 138 文件 0 错误；
   - 全量复跑 cold、forward、reverse 三组基线，运行 `machine_audit.py` 与 `score_frozen_answers.py`，产物冻结于 `var/identity-repair/stage-2/`。
   - 阶段2详细交接文档位于 `var/identity-repair/stage-2/handoff.txt`。

## 第一轮修复总结与第二轮（阶段3）准入依据

第一轮修复（阶段0 $\to$ 阶段1 $\to$ 阶段2 及 2026-11-10 漏洞闭环）在阶段2达成目标并圆满收官：
- **准入条件完全成立**：在不接纳已知混人的前提下，成功从 unknown 恢复了正常同人局部证据（局部一致段覆盖率翻倍，跨片对偶误合并 FP=0）；
- **当前状态冻结**：所有产物冻结于 `var/identity-repair/stage-2/`，不提前调整 `clustering_threshold`（保持 0.28），不提前重写多原型表示。
- **下一阶段**：阶段3（多原型特征表示与必要接口适配），属于第二轮计划，待用户下发新指令后开启。

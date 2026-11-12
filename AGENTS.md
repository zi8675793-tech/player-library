# 项目协作约定

## 当前核心验收目标（2026-11-05 用户更新）

- 核心目标是没有人工身份答案、从空库自动跨视频归人；同一比赛的球员数、片段数和批次数均未知且可变。下文的人工种子与人工模板规则描述现有 v1 的实现边界，不代表该目标已经完成，也不能作为后续只做人机协同验收的理由。
- 真实数据自动归人验收的推理输入不得包含人工身份标注、人工选择的目标轨迹/帧段、人工种子/模板批准、硬编码真值分组，或历史人工复核后的数据库/图库。当前 `../annotations.csv` 只供预测完成后的独立评分；不得移回输入目录或扩大输入根来恢复旧测试。
- `../test/target/` 是人工筛选的派生材料，其中 selection、裁切范围、目标样本和特征也属于答案侧材料，不能因仍位于 test 中而当作无标注输入。候选机器输入为 `video/`、`track/`、`identity/`，须明确读取白名单并核验来源；附件和 README 中的预期分组不进入算法。
- 自动验收先在全新状态上运行并冻结预测、配置和输入清单，再由独立评分阶段读取答案；不得根据本次验收答案选择阈值、种子、切换边界或模板。需要调参时另设开发数据，正式留出验证与开发片段/同源回放隔离。
- 人工复核功能可以继续保留并使用合成数据测试，但其通过率、给定进球的 6/4 统计、全部待定的零种子接入，均不能作为自动身份识别成绩。当前样例答案仅覆盖部分事件/目标轨迹，不足以评估全部人物的完整身份精度。

## 目标和当前状态

- 本项目是通用的比赛内球员库与事件管理模块。人数、片段数、导入批次数不固定。
- 同一比赛分批建库和更新，player_id 持久保留；不同比赛默认不自动归并。
- `../test/` 只是一个样例；样例细节仅属于 `adapters/sample_test.py`、examples 和样例测试。
- 当前已实现契约、配置、machine 无答案白名单适配与全新状态 baseline CLI、sample_test 历史人工样例适配与特征装配、核验 CLI、身份核心算法层 v1（纯诊断与计划）、SQLite 存储层、工作流业务闭环 v1（workflows.initialize/update/review/read 与 events.validate/associate）以及增量分批更新（workflows.incremental：开放集合分批到达演进、固定快照隔离、三态判定与幂等原子递增）；normalized 适配器、HTTP 服务、网页、视频剪辑导出尚未实现；球员合并与追加修订式历史恢复已实现（旧实现删除过的审核历史不可凭空恢复）。不得用固定分组或假成功响应充当实现。
- 上一阶段无答案基线见 examples/machine-baseline.md：默认按机器轨迹生成暂定段。当前已增加 `baseline --segment-tracks` 的轨迹内一致性检查，保留完整逐帧观测，提出 supported/unknown 派生 Segment、风险与不确定范围；supported 不表示身份真值。规则与下一阶段接口见 docs/track-consistency.md。
- 阶段修复计划与当前进度（2026-11-09 ~ 2026-11-10 更新）：
  * 固定分母基准：全场 10 个比赛片段机器证据输入（158 条原始轨迹、1,759 个特征样本、75,465 条逐帧观测、1,346.35 秒时长），杜绝基于浮动分母的虚假覆盖率。
  * 阶段0已完成：建立了统一的 7 级覆盖漏斗与全场固定分母；修复了评测工具中已知的时间严格半开区间 [start_s, end_s)、F1 边界 (TP=0 时为 0.0)、统一持久 player_id 评判、混合 track_id 稳定排序，以及完全复制图片序列伪独立支持剔除；解耦旧静态数据库断言。基线产物与详细交接文档见 `var/identity-repair/stage-0/handoff.txt`。
  * 阶段1已完成：实现了与 feature_space_id 绑定的距离校准档案契约（DistanceCalibrationProfile，schema: `distance-calibration-profile-v1`）；确立四态距离语义关系（near/far/ambiguous/uncomparable）；提取候选正例与互斥物理负例进行实证离散度诊断，证实二者在 [0.1652, 0.5757] 区间存在严重重叠；如实保留 status="uncalibrated"（保持工程门槛 near=0.18, far=0.45），不提前修改聚类门槛（clustering_threshold 保持 0.28），不伪造校准成功。基线产物与交接文档见 `var/identity-repair/stage-1/handoff.txt`。
  * 阶段2已完成：实现了第三代局部连续外观一致性（track-consistency-v3），引入连续局部支持窗、相邻近距离连续、零 far 强反证一票否决、中心样本（medoid anchor）支撑及视角平滑过渡区，成功在不接纳已知混人的前提下将局部一致段样本覆盖率从 12.6% (222) 恢复至 25.7% (452 / 137 段)、观测覆盖率从 6.7% 恢复至 14.0%、时长覆盖率从 6.5% 恢复至 13.3%，跨片段对偶精度保持 100%（FP=0）；解耦建档资格与匹配资格（can_match_existing 与 can_create_new_identity）；修复漏斗第4级统计缺陷与短段增量匹配门槛（min_match_buckets=2 与 short_segment_match_threshold）。并在 2026-11-10 专项完成 4 项关键漏洞加固（R1 空间校准档案硬绑定断言、R2 校准候选严格准入与 min_part_visibility 过滤、R3 增量漏斗全批分段合并与单调性恢复、R4 部分复制证据独立组件与双见证硬约束）。第一轮在此圆满收官，全部 700 项测试通过（700 passed, 1 skipped），基线产物与交接文档见 `var/identity-repair/stage-2/handoff.txt`。
  * 阶段3已完成：实现了 Segment 与 Player 的多原型身份表示（AppearancePrototype），以高质量原始样本/局部 medoid 替代单均值向量，完整保留 sample、segment、时间桶、可靠可见部位与独立跨片见证；距离决策实现相容视角查找与少数异类强反证（far >= 0.45）一票否决；扩展契约与 SQLite 迁移（009_multi_prototype_templates.sql 增设 template_prototypes 关系表）；合成测试验证转身同人相容恢复、混人严格否决与重复组件防刷；全场固定分母基准实测 Cold 跨片段对偶精度达 100%（TP=83, FP=0, F1=0.0556，历史首次实现冷启动跨片段正向召回），误合并严格为 0，全部 708 项测试通过（707 passed, 1 skipped），交接文档位于 `var/identity-repair/stage-3/handoff.txt`。
  * 阶段4已完成：实现了层次聚类候选生成与四道严苛准入关卡集团验收的彻底解耦（簇内强反证消除、Rival Margin 竞争对抗过滤、Tarjan 割边单桥接串联剪枝、跨至少 2 片段独立见证与模板校验）；明确区分已有球员匹配与新建身份建档门槛；输出可追溯审查字段（prototype_witnesses, rival_margins, rejected_member_ids, member_evidence）；全场固定分母基准实测跨球员误合并（Merge Errors）严格全部为 0，Forward 增量对偶精度达 100%（TP=80, FP=0），Cold vs Forward 联合归属 ARI 达 1.0000；全部 714 项测试通过（713 passed, 1 skipped），交接文档位于 `var/identity-repair/stage-4/handoff.txt`。
  * 阶段5已完成：彻底消解冷启动与增量更新的代码重复与阈值分歧，严格落实 Revision R 统一 7 步闭环更新流水线；消除增量更新“先遍历者先获利”时序漏洞，接入 SpatioTemporalConstraintEngine 时空物理互斥，实现历史延期证据（segmentation_proposals）联合代表元重构；统一输出 matched_existing、created_new_identity、insufficient_evidence 三态互斥决策；全场固定分母基准实测 Forward vs Reverse 顺序稳定性 ARI 达到 0.9236（较阶段 4 提升 +12.4%），Cold vs Forward 联合归属 ARI 达 1.0000，跨球员误合并严格全部为 0，全部 727 项测试通过（726 passed, 1 skipped），交接文档位于 `var/identity-repair/stage-5/handoff.txt`。
  * 阶段6已完成：角色边界解耦、人工复核队列契约完善与独立终验闭环。实现上游 machine、核心算法、下游展示与事件四层彻底单向解耦；引入 PersonRole 枚举与匿名档案隔离；建立预测层与评分层绝对隔离防火墙（assert_no_ground_truth_in_evidence, assert_frozen_before_scoring）；升级 Review Queue 数据契约，为视频 3 重叠框（duplicate_box）等 21 项关键风险段赋予 critical 优先级与一键操作建议；全场固定分母基准实测跨片对偶精度 100%（FP=0, TP=68 Cold, TP=148 Forward），跨球员误合并与拆分错误严格全部为 0，时空物理碰撞为 0，顺序稳定性 ARI 达到 0.9264 / 0.9823 / 0.8830；全部 740 项测试全数通过（739 passed, 1 skipped），交接文档位于 `var/identity-repair/stage-6/handoff.txt`。至此，7 阶段身份修复计划全案圆满结项！
  * 7阶段计划整体交付状态：阶段0至阶段6全部高质量交付闭环，基线产物与审查审计记录完整归档。
- 开工先读 README 和 docs 下的架构、流水线、数据契约、实施计划，以及开放集合增量身份更新（docs/incremental-identity.md）、机器轨迹内一致性检查（docs/track-consistency.md）与 SQLite 存储层（docs/storage.md）。
- 身份算法与归人决策遵循 docs/identity-policy.md；纯函数调用见 docs/identity-core.md。策略尚未经过真实身份校准，不得宣称已完成正式建库。
- 用户新指令优先，日常实现自主推进；本约定不引入额外审批步骤。

## 模块边界

- Python 3.11、FastAPI/Pydantic 2、NumPy、SQLite、原生 HTML/CSS/JS，依赖由 uv.lock 锁定。
- contracts 定义唯一字段、枚举和类型；adapters 只把上游格式转换为标准 EvidenceBatch。
- identity 按 prepare → associate → resolve → gallery 处理身份，产生可审查计划，不写库。
- events 校验事件并按角色解析当前球员；不调用重识别模型，不推断未提供的进球结果。
- workflows 编排初始化、增量、复核与读取。ingest 是共享处理路径；避免复制两套规则。
- storage 负责 SQLite、SQL 和迁移；事务及提交由 workflows 的写用例控制。
- presentation 只接 HTTP/CLI/网页输入并调用 workflows，不重算身份、统计和图库。
- identity 不依赖 events；跨业务组合由 workflows 完成。禁止出现新的万能 service.py。
- settings.py 是配置入口；路径以配置文件所在目录解析，不依赖命令执行目录。
- 禁止从 repository-review 动态导入、改 sys.path 或写入私人绝对路径。

## 必须保持的数据语义

- match_id 是身份作用域；batch_id 只是交接记录，不能作为球员库分区或幂等依据。
- EvidenceBatch 可以只交人物证据或只交事件；字段缺省、空列表均不表示删除旧数据。
- 原始证据与模型合同不可静默改写；新版本显式登记，人工修正不回写输入文件。
- 原始轨迹不等于身份段；有效归属粒度为片段内的轨迹段，支持拆分、未知和其他人物。
- 号码、跟踪质量、模板资格、身份确认和事件结果分别保存，禁止互相当作真值。
- 没有匹配结果不等于新球员；新增身份必须有独立支持或人工确认。
- 点事件与区间事件显式区分；多主体按角色分别解析，不能把所有轨迹强归一个球员。
- 片段时间、原视频时间、比赛计时分开；缺少来源映射时全局时间留空。
- 特征、模板、距离和阈值绑定 feature_space_id；换模型不重置 player_id。
- SQLite 是唯一正式状态源；所有查询经 workflows/read.py，派生 JSON 不反向更新。
- 重复证据跨批次提交也不新增记录或 revision；新增片段不删除旧片段。
- 身份修正后，事件和统计根据当前映射重新解析；事件先到允许保存为未关联。
- 人工确认优先；自动重匹配不覆盖确认、拒绝、拆段或样本复核。
- 模板更新使用最终决策，当前批匹配读取固定图库快照，防止本批错误串联。
- v1 人工种子建档、新建和拆段；人工模式的自动确认仍要求校准，匹配只使用人工批准的有效模板。无种子发现独立输出 experimental/unvalidated 的 proposed 自动归属与自动模板，不伪造人工批准，不进入 confirmed 事件统计；空库与增量分批更新均保证重放幂等与持久稳定 ID，持续开放集合更新遵循固定快照隔离与保守待定策略；独立真实精度校准仍属独立评分阶段。
- 写入核对 expected_revision，证据、映射、图库计划、revision 和审计原子提交。
- 资源通过 asset_id 提供，禁止任意本机路径读取接口。

## 修改和交付

- 一次实现一个可验收行为；改契约时同步调用者、文档和必要测试。
- 改库增加 storage/migrations 中的迁移，不以删库重建处理升级。
- 改依赖同时更新 pyproject.toml 和 uv.lock；改策略集中在 identity 和配置。
- 新输入格式增加 adapter，通用身份/事件模块不得识别 target/、CSV 列名或样例号码。
- 样例特征 [N,6,128] 不是通用维度常量；支持范围由特征空间和后端决定。
- 不预设一次初始化能发现全场所有球员，也不以固定人数作为聚类规则。
- 默认只读 ../test/ 和 ../repository-review/；生成产物写 var/，样例配置放 examples/。
- 避免为纯文案改动增加镜像测试；业务测试覆盖规则、失败路径和跨模块行为。
- 检查：uv sync --locked；uv run ruff check .；uv run ruff format --check .。
- 有业务测试后运行 uv run pytest。报告真实完成状态，不把给定标注当作识别成绩。

# 按业务流水线组织代码

## 范围与状态

通用的比赛内球员档案、增量身份关联、图库维护、事件关联与查询模块。
当前已实现 contracts 的 Pydantic 2 模型和 settings 的 TOML 读取、类型与路径校验。
machine 的无答案输入适配、sample_test 的历史人工样例适配及文件/数组校验已实现。
无标注 baseline CLI 从全新 SQLite 状态导入机器证据，复用现有工作流并冻结导出；
空库基线全部待定也如实保存，不表示已完成自动归人。
当前另有可选的 `baseline --segment-tracks`：身份纯算法层从结构化逐帧观测与人物特征
提出保守的一致性支持段和未知范围，再交同一共享工作流保存；不创建球员或批准模板。
算法、版本化报告与建库接口见 [track-consistency.md](track-consistency.md)。
默认 v1 保持全段检查；发现配置显式选择 v2 连续局部支持窗，保留全段失败中的合格局部证据，窗外未知且不推断换人。
新增 `baseline --discover-identities` 从合格段自动形成跨片段组，保存稳定匿名 ID、proposed 自动归属、
自动实验模板和不可变支持报告。默认 absolute 保持兼容，示例显式采用 complete_support_v2 的共现人物对照支持；
距离、negative 最近两原样本和空间排他来源均保存，缺独立对照时未知。算法与边界见 [identity-discovery.md](identity-discovery.md)。
结果未独立校准，不进入 confirmed 事件；持续开放集合增量分批更新见 [incremental-identity.md](incremental-identity.md)。
身份核心算法层 v1 已实现纯样本准备、KPR 距离、候选诊断、归人决策和图库变更计划。
SQLite 存储层及工作流业务闭环 v1（workflows 的 ingest、initialize、update、review、read 与 events 的 validate、associate）已实现并经过集成测试；normalized 输入适配与展示层（HTTP/页面）留待后续实现。
测试样例的片段数、号码、人数和事件类别不属于通用业务规则。
身份算法与决策以 [identity-policy.md](identity-policy.md) 的 v1 规范为准；调用见 [identity-core.md](identity-core.md)，真实身份校准尚未完成。

## 结构

```text
app/
  main.py                     # 组装配置、存储、用例和HTTP应用
  settings.py                 # 统一配置和路径
  contracts/                  # 内部统一语言
    common.py                 # 标量、来源键与半开时间/帧区间
    evidence.py               # EvidenceBatch、媒体、样本、特征空间
    identity.py               # Player、Segment、Assignment、Template
    segmentation.py           # 一致性策略、分段提案和可追溯报告
    discovery.py              # 自动发现策略、段对、组、模板支持和未知
    calibration.py            # 距离校准档案、四态距离语义与离散度诊断契约
    events.py                 # 点/区间事件、按角色区分的主体
    api.py                    # 请求、响应、复核命令
    views.py                  # PlayerView、ReviewQueueItem 业务查询视图
    workflows.py              # WorkflowReport 工作流执行报告
  adapters/                   # 上游文件格式边界
    _reading.py               # 根目录约束、只读读取、来源文件与记录指纹
    _machine_reading.py       # 机器白名单读取与结构化解析支撑
    normalized.py             # 标准证据接口
    machine.py                # track/identity/video 的无答案白名单适配
    sample_test.py            # 历史人工选择样例转换，非自动识别验收
  identity/                   # 人物身份分支：无数据库写入
    prepare.py                # 样本质量、身份段、可比较性
    representation.py         # 轨迹级分部位可见度加权池化与归一化
    constraints.py            # 物理时空绝对排他互斥图 (Must-Not-Link)
    clustering.py             # 带排他硬约束的层次凝聚流形聚类器
    consistency.py            # 轨迹内一致性提案（v1全对、v2连续窗、v3局部连续外观支持与中心样本支撑）
    calibration.py            # 空间绑定校准档案构建、四态距离判定与正负样本离散度诊断
    discovery.py              # 固定机器证据上的双向支持、冲突、流形聚类与模板计划（含部分复制硬过滤）
    contrastive.py            # 空间共现对照、几何负例与边际支持
    incremental.py            # 开放集合增量身份更新、短段匹配门槛与建档隔离、去重防反复建档纯算法
    fingerprints.py           # 派生证据和解析特征的顺序无关绑定
    associate.py              # 与已有图库比较，生成候选
    resolve.py                # 已有/新建/待定/拒绝决策
    gallery.py                # 模板增加、替换、退出计划
    distance.py               # 特征空间对应的距离，KPR（含可见性门槛过滤与四态语义）
    _eligibility.py           # 资格判定与建档/匹配能力分离辅助函数
  events/                     # 事件分支：不重新跑动作模型
    validate.py               # 点/区间、结果、来源与主体校验
    associate.py              # 将每个事件主体解析到当前球员
  workflows/                  # 可调用用例，也是事务边界
    ingest.py                 # 共同接入与处理路径
    initialize.py             # 首次建立人工身份锚点与图库
    discover.py               # 空库实验发现、稳定ID、自动证据与原子提交
    incremental.py            # 开放集合分批增量更新、固定快照隔离与事务推进
    update.py                 # 增量身份更新、独立事件导入
    review.py                 # 人工维护与审计
    read.py                   # 统一档案、事件、统计、资源查询
    _consistency.py           # 工作流一致性分段协调
    _review_ops.py            # 审核操作底层原语
    _restore.py               # 历史版本可追溯恢复原语
  storage/
    database.py               # 连接与迁移执行
    repositories.py           # 参数化 SQL 查询与仓储
    discovery.py              # 发现报告与分组持久化
    features.py               # 特征向量 BLOB 编解码与字节序保证
    relations.py              # 跨表关系与业务外键校验
    revision_states.py        # 不可变版本状态清单记录
    snapshot.py               # 统一图库快照读取
    workflow_plans.py         # 工作流计划暂存
    migrations/               # 001–008 增量迁移脚本
  evaluation/                 # 离线与自动化评测子系统：固定基准、无监督漏斗与双轨度量
    funnel.py                 # 统一 7 级覆盖漏斗与多维损失归因（样本/观测/时长）
    units.py                  # 固定评测单元提取、投影与清单封装 EvaluationManifest
    metrics.py                # 双轨评测指标计算：无监督多层漏斗与有监督聚类指标，DualTrackEvaluationReport
    extract_units.py          # 评测基准单位提取工具
  presentation/
    api.py                    # HTTP → workflows
    cli.py                    # 只读 validate / 无标注 baseline 命令
    baseline.py               # 全新运行、工作流调用与冻结导出
    static/                   # 网页只消费API结果
examples/                     # 样例配置与样例说明
tests/                        # 通用规则、工作流和adapter测试
docs/                         # 流水线、架构、契约、实施计划
var/                          # 不提交的数据库、预览、日志和导出
```

这些目录对应数据语言、输入边界、两个业务分支、编排、持久化和展示。
不再使用含全部业务的 service.py；不额外复制另一套查询层。

contracts 只校验提供对象中的局部事实，不读取文件或数据库；目标引用可尚未到达。
settings 使用 Python 3.11 的 tomllib，以配置文件目录解析相对路径，加载不创建运行资源。
machine adapter 仅打开明确白名单文件，核对文件内容、特征合同、观测和索引；
sample_test adapter 保留历史人工样例的独立边界；
后端可执行性由后续 identity 检查，读取现有特征不运行模型；
storage 提供跨批来源幂等检查以及有效段重叠、别名环的显式检查原语；workflows 负责决定何时检查和处理。

## 依赖和事务

```text
presentation → workflows → adapters / identity / events / storage
adapters / identity / events / storage → contracts
main → settings + 各组件装配
```

底层不反向调用 workflows。identity 与 events 无相互依赖，跨领域组合由 workflows 完成。
identity 输出计划而不提交，events.associate 接收当前映射上下文而不擅自修改身份。
HTTP、CLI 和后续剪辑导出均通过 workflows；前端不再统计或认人。
只读 `validate` 是输入边界核验，直接调用 settings 和 adapter，不执行业务用例或写入状态。
`baseline` 仅接受 machine 输入，在项目 var 下创建不存在的新运行目录与新数据库，
经 update_library 接入，再经 workflows/read 读取预测与诊断。它没有旧库加载、人工种子或答案参数。
开启 `--segment-tracks` 时先调用纯函数 `segment_tracks`，保留原包并提交派生 EvidenceBatch。
SQLite 迁移 006 保存全部 TrackObservation 及来源版本；观测不因身份修订恢复而倒退。
开启 `--discover-identities` 隐含分段，再调用 discover_library；迁移 007 保存不可变发现报告与组到玩家映射；
迁移 008 保存机器分段提案与状态索引，在开放集合增量分批更新中持久化保留各段一致性资格（supported/unknown）。
报告通过派生证据指纹绑定全部输入与解析特征；自动模板没有人工批准，不作为本次发现的反馈输入。
推理模块不导入 sample_test 或独立评分代码；答案文件只允许未来独立评分在冻结后读取。
独立评测子系统 `app/evaluation/` 承担单因子实验与方法探索的固定评测基准职责，将全场原始样本与观测时间原子化为不可变评测单元（`EvaluationManifest`），提供不受下游分段切分差异影响的全局固定分母；通过无监督多层漏斗（样本覆盖率、时长覆盖率、未知原因分布）与双轨度量（`DualTrackEvaluationReport`），确保算法优化与消融实验过程不依赖人工标注且具有严格统一的对照标尺。

写流程在一致的图库/身份版本下计算计划，事务内再次核对 expected_revision；
证据、决策、图库、修订与审计一次提交。storage 辅助函数不得自行 commit。
发生并发版本变化时返回冲突，重读后重新计划，不接受过期候选。

## 数据和规模

SQLite 作为第一版正式状态源，保存按 match_id 隔离的比赛、原始证据、身份段、
档案、各特征空间图库、事件主体和修订审计。batch 记录交接，不隔离球员。
向量可用带 shape/dtype 的BLOB保存；图片与视频保存受控资源引用和哈希。
JSON、网页缓存与导出清单都是派生结果。

数据库迁移只能改变存储实现，不改变 player_id。模型升级新增特征空间或显式迁移，
不靠新建比赛或重编号球员解决。无可比较模板时返回未知。
不保证任意规模吞吐；实现后通过整场基准测量容量，再决定索引或存储优化。
业务契约不设十片段、两人、固定部位数等样例上限。

## 目标用例入口

| 入口 | 作用 |
| --- | --- |
| discover.discover_library | 无玩家/人工审核的机器证据库冷启动实验发现；同报告幂等 |
| initialize.initialize_library | 尚无初始身份锚点的比赛库建档；可已存在事件或待定证据 |
| update.update_library | 同场新人物证据和可选事件的增量处理 |
| update.ingest_events | 只有事件的提交，允许依赖人物证据尚未到达 |
| review.apply_review | 确认、拆分、合并、模板复核等变更 |
| read.get_players / get_events / get_review_queue | 当前档案、主体归属、统计和待复核查询 |

上述 Python 工作流已实现；HTTP 路由仍待实现，调用者使用统一 contracts。
候选预览调用相同身份规则但不写库；接受经有类型的review命令并核对版本。
API围绕match而非样例dataset：/api/matches/{match_id}/players、/events、/reviews、
/imports、/initialize、/matching/preview；资源为/api/assets/{asset_id}。

## 技术与决策记录

- Python 3.11 + FastAPI/Pydantic 2 + NumPy + 标准库sqlite3 + 原生网页；uv.lock锁定依赖。
- 初期不引入消息队列、动态插件框架、独立向量库或跨比赛永久身份系统。
  独立人物/事件导入不依赖异步基础设施。
- 2026-11-03：将上一版样例原型定位修正为通用模块，样例布局仅由adapter处理。
  将平铺职责骨架调整为上述分支与工作流；没有业务实现需要迁移。
- 2026-11-03：实现契约与配置层。特征空间描述与文件校验分离，原始事件与当前主体解析分离；
  只作局部校验以允许分批证据和先到事件，适配器与持久化仍按后续里程碑实现。
- 2026-11-03：实现 sample_test 只读输入边界和核验 CLI；复用现有模型与 load_settings。
  文件引用解析后必须在 input_root 内，NPZ 禁止 pickle，逐行核对 JSONL、数组和图片哈希；
  来源文件哈希与记录指纹分开，业务 ID 和特征空间 ID 不依赖批次或遍历顺序。
  为保留来源定位、实际观测帧、事件级号码和警告作最小契约补充，不引入 normalized 新协议。
- 2026-11-03：确定身份策略 v1：人工种子、KPR 部位距离、双模板均值与段内中位数、
  覆盖/支持/分歧门槛、人工新建与图库批准。阈值经独立身份标注校准前不启用自动确认。
  当时只确定设计，不表示数据库或事件关联已实现。
- 2026-11-04：实现身份核心算法层 v1；复用原有身份模型，增加结构化风险、精确空间策略、
  校准指纹、固定图库快照和逐样本诊断。纯算法不读文件、不写库、不生成正式 player_id，
  模板准入与身份确认独立；同批匹配不扩张图库。实现与合成测试不表示完成真实数据校准。
- 2026-11-05：实现 SQLite 存储连接、增量迁移、类型化仓储、来源版本幂等、修订审计与一致快照。
  调用与边界见 [storage.md](storage.md)。
- 2026-11-05：实现工作流业务闭环 v1，打通 ingest、initialize、update、review、read 与 events 关联解析。
  支持冷启动种子建档、增量归人、人工作用域审核、模板激活与撤销、未关联事件晚到归属解析。
  提供 examples/workflow_v1.py 运行示例及 11 类集成测试，HTTP 服务与页面留待后续。
- 上游参考33613/basket_cut，提交a7fb52ffbd92b22c2dc5d02f4e8376ca5c6b6880。
  参考pipeline/identity/cross_clip.py的KPR公式，新项目按已确定公式独立实现并用合成数据验证，未复制上游代码。
  复制前登记许可证和对应测试；运行不能依赖审查快照。

## 无答案输入入口（2026-11-06）

片段从机器目录发现，不从标注枚举；输入规模与特征维度来自实际机器证据。
白名单包括 track/<clip>/ 的 tracks.csv、tracks.jsonl、video_meta.json 和 samples/*.png，
identity/<clip>/ 的 feature_contract.json、features.npz、samples.jsonl，以及 video/<clip>.mp4。
读取前检查路径及链接，禁止 annotations.csv、target、历史审核数据库成为输入。
NPZ 禁止 pickle，文件 SHA-256 与单条记录/观测集合指纹分别保存。

默认原轨迹暂定段仍使用现有 Segment；可选一致性检查也沿用同一 Segment，
将父段停用、派生段通过 supersedes_segment_id 回指，并记录未知范围和支持依据。
两种模式均不声称身份已确认。自动身份段准备通过 EvidenceBatch / IdentityEvidence 提交，
不新增第二套身份模型；报告另存机器检查诊断，不反向充当数据库正式状态。
预测导出沿用现有 Segment、Assignment，并保留样本和来源关联；JSON 是 SQLite 的派生结果。
评测基准解耦方面，`app/evaluation/` 提取不可变的原子样本与观测单元清单（`EvaluationManifest`），作为单因子实验与方法探索的固定评测基准。不论轨迹分段策略如何演进，均以原始机器证据为不可变分母，双轨计算无监督多层漏斗与有监督聚类指标（`DualTrackEvaluationReport`），确保算法优化过程不依赖人工标注且具有严格的对照标尺。


2026-11-06 历史运行：1 个实验身份组（片段 20/T1 ↔ 6/T5）、2 个成员段、861 个未知段，见 examples/machine-discovery.md。

2026-11-08 历史实验运行 `manifold-delivery-001`：采用时空受约束层次流形聚类（`constrained_manifold`），识别 21 位核心跨视频球员，全场 75,465 条逐帧观测保持物理同屏零违规。早期报告中的 97.22% 活跃段覆盖率系基于排除全部未知轨迹的浮动分母得出。

2026-11-09 ~ 2026-11-10 身份修复计划（第一轮已完成：阶段0、阶段1、阶段2）：
- 固定分母基准：全场 10 个比赛片段机器证据（158 轨迹, 1759 样本, 75465 观测, 1346.35s 时长）的统一固定分母与 7 级覆盖漏斗；修复时间严格半开区间、F1 边界公式与持久身份对齐，消除浮动分母产生的虚假覆盖（`var/identity-repair/stage-0/handoff.txt`）。
- 阶段0：确立 7 级覆盖漏斗与多维损失归因，解耦旧静态数据库断言。
- 阶段1：实现与特征空间绑定的 `DistanceCalibrationProfile` 契约（schema: `distance-calibration-profile-v1`）；实证诊断 1,113 对候选正例与 890 对物理负例在 [0.1652, 0.5757] 的严重重叠，如实保持 `status="uncalibrated"`（near=0.18, far=0.45），不提前修改聚类门槛（0.28）（`var/identity-repair/stage-1/handoff.txt`）。
- 阶段2：实现第三代局部连续外观一致性（`track-consistency-v3`），以连续外观支持窗、相邻近距离连续、零 far 强反证否决、中心样本（medoid anchor）支撑与视角过渡区替代易误杀同人的全轨全对检查；解耦 `can_match_existing` 与 `can_create_new_identity`；修复漏斗第4级跨片比较段统计缺陷与短段增量匹配门槛（`min_match_buckets=2` 与 `short_segment_match_threshold`）。在固定分母下将局部一致段样本覆盖率从 12.6% 恢复至 25.7% (452 样本 / 137 段)、有效时长从 6.5% 恢复至 13.3%，同时独立评分保持 0 误合并（FP=0，精确率 100%）。
- 2026-11-10 专项修复（4 项漏洞闭环）：
  * R1：建立校准档案强一致性断言，杜绝参数静默旁路；
  * R2：实现校准证据严格准入过滤与 `min_part_visibility` 硬过滤；
  * R3：基线增量运行汇总全批次分段报告，恢复漏斗单调递减（452 -> 166 -> 141）；
  * R4：部分复制证据独立组件基数（$\ge 3$）、覆盖率（$\ge 0.8$）与双见证硬约束激活。
- 第一轮修复在此圆满收官（`var/identity-repair/stage-2/handoff.txt`），通过全部 701 项测试（700 passed, 1 skipped）。
- 当前待执行阶段为第二轮阶段3（多原型特征表示与必要接口适配，尚未启动）。

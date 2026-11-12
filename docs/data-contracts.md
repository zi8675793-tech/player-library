# 标准数据契约 v2

字段和类型由 app/contracts 各模块唯一实现；本文固定语义，并与代码同次修改。
v2修正上一版以样例dataset为作用域、仅支持进球点事件的限制。
当前已实现 Pydantic 2 模型及局部校验、machine 无答案适配器、sample_test 历史人工样例适配器、
核验与基线 CLI，以及身份核心、SQLite 和身份/事件工作流。normalized 适配器与 HTTP 服务仍待实现。
无答案接入与导出继续使用下述契约，不以适配器专用结构替代标准业务模型。

模型统一 snake_case，未知字段报错，标识和号码保持字符串；数值不接受布尔值或数字字符串，
所有浮点字段拒绝 NaN/Inf。JSON 使用 `model_validate_json` / `model_dump_json` 往返，
枚举输出其字符串值。号码 `"00"` 不转换成整数。本文的可选字段可为 null；
实际默认值和完整类型以 `app/contracts/` 为准。

## 作用域和标识

| 标识 | 语义 |
| --- | --- |
| match_id | 持久比赛作用域，身份、图库、事件、revision归属此处 |
| batch_id | 一次交接/导入记录；不是身份分区，也不是幂等唯一依据 |
| source_video_id | 可选原始长视频标识；样例未提供时不虚构 |
| clip_id | 比赛内片段标识；重切片段产生新来源ID |
| track_id | 标准化字符串，片段内有效；外部整数由adapter转换并保留来源 |
| observation_id | 比赛内逐帧机器观测标识；保留框、实际时间、可选帧号和来源版本 |
| sample_id | 比赛内稳定样本标识；保留producer和原样本键供追溯 |
| segment_id | 绑定片段、轨迹与不可变时间/帧范围的身份段标识 |
| player_id | 比赛内持久pl_标识，不由号码、批次、模型生成 |
| feature_space_id | 模型、布局、预处理、归一化与距离共同定义的可比较空间 |
| event_id / subject_id | 持久事件和事件中单个角色主体标识 |
| revision | 比赛内非负整数；有效业务变更才增加 |

同一match的多个batch共享球员。不同match不自动合并。
不同片段内的相同秒数不表示同时；显式映射到同一source_video_id的source_time后才可比较。

## EvidenceBatch

统一交接对象包含：schema_version、match_id、batch_id、producer、来源清单，
以及可选的clips、identity_evidence、events。两类业务证据可以分别到达。
缺省表示未提交；空列表也不删除旧数据。删除、停用、替换采用显式有审计的操作。

| 模型 | 已实现字段与局部规则 |
| --- | --- |
| EvidenceBatch | schema_version 为字符串 `"2"`；match_id、batch_id、producer 必需；sources/assets/warnings 默认空列表；clips、identity_evidence、events 可缺省 |
| IdentityEvidence | tracks、observations、segments、samples、feature_spaces、feature_files、risks 各自默认空列表，不要求同时提供 |
| Track | 原始范围含 match_id、clip_id、track_id、start_s/end_s、可选成对帧范围及 source；observed_frame_indices 可选；不等同于 Segment，也不含球员归属 |
| TrackObservation | observation_id、match_id、clip_id、track_id、time_s、bbox_xyxy、source 必需；frame_idx 与 detection_score 可空；框为有限坐标且 x1<x2、y1<y2；检测分在 [0,1]，不表示身份概率 |
| Sample | match_id、sample_id、clip_id、track_id、time_s、source 必需；frame_idx、segment_id、image_asset_id、tracking_quality、appearance 可选；features 默认空列表 |
| TrackingQuality / AppearanceEvidence | 跟踪 score、score_semantics、遮挡/混人、usability、clean_review、risks 与可选队伍、号码、置信度分别表达，不确立身份或模板资格 |

批内提供的 clip、track、observation、segment、sample、event 的 match_id 必须与包一致。
各实体列表的主键必须唯一：sources 用 file_id，assets 用 asset_id，clips 用 clip_id，
events 用 event_id，身份分支的 segments、samples、feature_spaces、feature_files 分别用其 ID；
tracks 使用 (clip_id, track_id) 组合键。同一事件内的 subject_id 唯一，跨事件可重复。
observations 使用 observation_id；不因多个框位于同一时刻就推断不同身份。
即使内容完全相同，同一实体列表内重复主键仍报错，不静默覆盖或去重。
不同样本可以来自相同 track_id/frame_idx，样本主键 sample_id 仍须不同。

引用目标不要求同时出现；目标已在同包提供时，模型检查已知事实的一致性：

- Sample 引用已知 Segment 时，clip_id、track_id 必须匹配，时间须落在该段范围内；
  样本和目标双方均提供帧定位时，再检查帧号范围。
- Sample 引用已知 Track 时，时间及双方已提供的帧定位须位于轨迹范围内。
  Segment 保留的选择范围可以宽于 Track 的实际观测首尾范围，二者分别检查 Clip 界限，
  不要求 Segment 被 Track 的观测范围包含，避免把首尾缺帧误判为非法。
- TrackObservation 引用已知 Track 时检查作用域及已提供的时间/帧范围，
  已知 Clip 时检查片段时长与帧数；不在通用契约中推算外部 FPS、补缺失观测或推断遮挡。
- Clip 提供 duration_s/frame_count 时，该片段中的轨迹、身份段、样本和事件不能越过已知界限。
  时间点与帧号须小于右端点，区间的结束端点可以等于右端点；不补造未提供的时长、帧数或定位。
- Event 的时间按 event.clip_id 的本地坐标校验。主体引用显式提供 time 时，按所引用 Track
  或 Segment 所属片段的本地坐标与已知目标范围校验，不套用事件所在片段的时长。
- 已知 Clip.asset_id 必须指向 video 类型，已知 Sample.image_asset_id 必须指向 image 类型。

未知或晚到引用仍保留，不因目标未随本批提供而拒绝人物-only、事件-only或补交特征。
主体引用省略 time 时，定位留给后续关联流程；模型不据事件时刻是否落入当前支持轨迹、
观测缺帧或未来可能 unmapped 的结果拒绝原始事件。
样本范围校验不要求 frame_idx 出现在 Track.observed_frame_indices 列表中；
模型也不按 CFR 推算并强制 frame_idx/fps 与 time_s 相等，实际逐行观测一致性由输入适配器核对。

identity_evidence描述轨迹/身份段候选、人物样本、图片引用、向量和可选号码队伍证据。
events为一般事件列表，不能要求每个片段恰有一次进球。
先到事件可以引用尚未登记的轨迹；保留外部引用，状态待关联，而不是拒绝整次提交。

来源最小键：producer + record_type + source_record_id + evidence_version。
`SourceKey` 实现该键，evidence_version 是不作数字排序的非空字符串。
`Provenance` 在此基础上增加可选 content_hash、supersedes 和默认空的 locations；supersedes 是旧 `SourceKey`，
不能直接指向当前来源键。content_hash 和文件 sha256 均使用 64 位小写十六进制 SHA-256 字符串。
模型只校验格式，不计算哈希、检查源文件或查询旧版本。

`SourceLocation(file_id, line_number?, record_key?)` 引用来源清单中的文件；行号从 1 开始，
record_key 保留外部记录键或对象内定位。它是可审查位置，不属于来源去重键。
locations 可同时指向 CSV、selection 或 JSONL；未建模的原始字段通过这些位置保留可追溯性。
content_hash 是规范化单条记录或确定性记录集合的指纹；SourceFile.sha256 是整个文件的哈希。
CSV 行顺序改变可以改变文件哈希和行号，不应改变记录键、业务 ID 或单条记录指纹。

`EvidenceWarning(code, message, clip_id?, locations=[])` 保存质量风险、缺帧和身份未确认等说明。
警告不代替致命错误，不代表身份已确认，也不删除原始证据。模型只检查字段局部事实；
来源文件真实存在性、位置可解析性及外部输入的完整性由输入适配器负责。
模型中的同包已知引用校验不读取文件，也不查询尚未提供的目标或数据库。

指纹校验内容；同来源、同版本、同内容跨batch重复交付也不重复登记。
同版本内容变更报409。新证据版本显式新增并声明supersedes关系，保留旧证据与审计。
版本切换不能默默覆盖人工决定；需要重算的映射进入明确复核/迁移计划。

## 媒体与来源定位

`SourceFile(file_id, path, sha256?, size_bytes?)` 是输入定位及校验信息；
`MediaAsset(asset_id, media_type, file)` 区分 video/image。它们不是可任意读取文件的公开接口。
`EvidenceBatch.sources` 是 SourceFile 清单；各条业务证据的来源键单独保存在 source 中。
同一 SourceFile 可在 sources、MediaAsset.file、FeatureFile 的 values/index/visibility_file
等不同嵌套位置重复引用。同 file_id 的 path 必须相同；sha256、size_bytes 分别在双方均为
非 null 时比较，不相等则冲突。null 表示未知，与具体值兼容；校验不回填或修改任何原声明。
该规则在 IdentityEvidence 独立使用时同样检查其内部 FeatureFile 引用，
不需要包装为 EvidenceBatch 才生效。
sources 列表内相同 file_id 仍拒绝重复；嵌套引用兼容不放宽实体列表的主键唯一要求，
也不要求不同 file_id 指向不同路径。

Track 的范围只描述包围范围，不宣称其中每帧均有观测。提供 observed_frame_indices 时，
它必须排序、唯一，且落在已提供的帧范围内；缺失的帧保留为缺失，不插值或补造。
null 表示未提供逐帧存在性信息，空列表表示明确未提供任何实际观测帧。

`Clip` 包含 match_id、clip_id，以及可选 asset_id、duration_s、frame_count、fps、source_mapping；
frame_timing 为 cfr/vfr/unknown，默认 unknown；明确 cfr 时必须提供正 fps。
`SourceMapping(source_video_id, source_start_s, time_scale=1.0)` 使用显式仿射关系：

```text
source_time_s = source_start_s + clip_time_s * time_scale
```

source_start_s 非负，time_scale 为有限正数。没有 mapping 不生成 source_video_id 或全局时间；
VFR 时间定位仍需后续适配器提供实际时间戳，不能因为存在 fps 就按 CFR 处理。
采用可选映射以保留只有短片的输入，不虚构整场顺序。

## 身份与模板

- Segment：match_id、clip_id、track_id、start_s、end_s、可选frame_start/frame_end_exclusive、
  active、supersedes_segment_id。原始轨迹与可修正身份段分开。
- 同一原轨迹的当前有效段不得重叠；混人需拆段或隔离，不能把整条track当作身份真值。
- Assignment：segment_id、player_id可空、status、method、evidence、revision。
  status=unassigned/proposed/confirmed；候选不自动算作确认。
  实际模型另含 match_id；unassigned 必须 player_id=null，proposed/confirmed 必须提供 player_id。
  仅 unassigned 可带 unassigned_reason=unknown/other_person/ambiguous/rejected，
  明确区分未知球员、其他人物、身份歧义与拒绝，不靠号码或模板审核状态推断。
  method=manual/automatic/seed/imported；AssignmentEvidence 记录 kind 和可选参考、描述、空间、距离，
  有 distance 时必须指明 feature_space_id。
- Player：player_id、match_id、display_name、可选team_id/jersey_number、alias_of。
  球员数量不固定，号码是字符串或null，保留00。合并别名不得成环。
- Sample的跟踪质量、身份归属和模板复核分别保存；模板复核为unreviewed/approved/rejected。
  错人样本应修正身份段；模糊样本拒绝入模板不影响该人的事件归属。
- Template绑定player_id + feature_space_id + sample_id，准入取决于已接受归属和样本质量。
  模型另有 template_id、match_id、review_status、active、revision；
  review_status 不代替 Assignment.status 或 TrackingQuality。
- resolve输出match_existing/create_new_proposal/pending/rejected；
  未匹配或不可比较一律不能单独作为新建球员的充分依据。
  Candidate 保存绑定特征空间的 distance、common_parts、gallery_revision 和 template_ids；
  Resolution 中仅 match_existing 携带 player_id，新建提案必须有非空支持证据。

上述模型的 v1 执行策略见 [identity-policy.md](identity-policy.md)：人工种子和新建确认，
KPR 多样本、多模板候选，校准后才允许自动确认；仅人工批准的合格模板用于后续匹配。
纯算法现已补充有类型诊断、质量风险、人工约束和校准配置（见下文）；
不能把 Candidate.distance 当作身份概率，或把 Resolution.match_existing 直接等同于已确认归属。

新增契约继续沿用上述 Sample、Segment、Assignment、Candidate、Resolution 和 Template：

- TrackingQuality.score_semantics=detection/tracking/unknown（缺省 unknown），usability=
  unknown/usable/unusable，clean_review=unreviewed/approved/rejected；外部 accepted 不等于人工批准。
  IdentityEvidence.risks 和 TrackingQuality.risks 保存 IdentityRisk，以 match 和段/样本/轨迹及可选时间范围
  定位；kind 区分混人、切换、重复框、待质量复核和身份冲突。notes 仅保留原始说明，通用算法不解析。
- RiskReview 保存 risk_id、match_id、segment_id、actor/reason/revision、resolution=cleared/confirmed；
  清除风险必须匹配人工审核作用域。SampleExclusion 保存 sample_id、reason、可选 duplicate_of。
- PreparedSample 保留原 Sample 副本、所选空间的 InlineFeature、时间桶、前中后区间、可见部位信息及
  clean_approved；PreparedSegment 保留所有可用样本、代表样本 ID、排除、风险和实际时间覆盖。
  PreparedEvidence 记录 match、空间、策略版本及指纹；文件特征由调用者提供内存映射，不在算法内读取。
- GallerySnapshot 包含 match_id、revision、全部 Player、当前 Assignment、Template、IdentityEvidence、
  assets 和 risk_reviews，校验本地 scope、重复声明与修订上界。它是固定输入，不会被本批计划扩张。
- DistanceResult 仅在 comparable=true 时携带有限 distance；不可比较返回 reason，非法特征抛错。
  SampleDiagnostic → SamplePlayerScore → TemplateComparison 保留逐样本/球员/模板距离与共同部位及失败原因。
  AssociationResult 记录所有候选、共同查询集合及覆盖、第一/第二名、可空 margin、支持率、强分歧、
  版本/模板清单及决策阻塞原因。local_candidates 单独标明局部或 single_template_preview 预览。
- Resolution 新增 assignment、reasons、requires_review、automatic_confirmation_eligible 和版本定位。
  assignment 的 status 明确 proposed/confirmed；新身份提案仍不带正式 player_id。
- ManualConstraint 是带审计的 forbid_player/cannot_link/block_automatic；NewIdentityRequest 是独立人工
  新身份意图和干净支持样本；它们都不是可执行的 ReviewCommand。新 ManualTemplateReview 绑定
  sample_id、player_id、feature_space_id、状态、actor/reason/revision 和可选 template_id。
- Template.approval 可保存 ReviewRecord；新 GalleryPlan 只列 activate/deactivate/await_review/
  replacement_required 的 GalleryChange、人工审核及替换选择，不写库、不自动分配 template_id。
  归属失效引起的 deactivate 可跨特征空间；其余准入判断只针对本次选定空间。

时间范围均非空；成对帧范围要求 frame_end_exclusive > frame_start。
Segment 自替换、Player 自别名及同包已知目标的一致性可在本地拒绝；数据库中的引用存在性、
跨批历史有效段重叠、跨对象别名环及支持证据是否足以确认新球员均需后续工作流检查，
不在模型层查库或推断。同包身份段的历史生效策略也不由来源证据模型决定。

## 特征空间与格式

通用特征定义通过feature_space合同指定后端、模型/配置哈希、预处理/颜色约定、
布局、维数、归一化、可见性和距离。合同允许描述其他后端，当前运行时仅实现 KPR 部位距离，
不支持的后端明确报错，不运行 ReID 模型。
KPR为float32 [N,P,D]及visibility [N,P]，P/D由合同决定，不固定为6/128。
可见部位有限、L2归一化；visibility在[0,1]，共同部位数由匹配策略检查。

`FeatureSpace` 的 feature_space_id、backend、model_hash、preprocessing、layout、shape、
normalization、distance 必需；config_hash、visibility 可选，dtype 当前仅 float32。
preprocessing 包含 version、color_space 和可选 config_hash。
shape 表示**单样本**形状：vector 为 `[D]`、parts 为 `[P,D]`，tensor 支持其他秩；
各维为正整数。KPR 要求 parts、l2，visibility 合同的 shape 为 `[P]`，并显式描述 semantics。

样本 features 采用 kind 区分：inline 保存按 shape 的行优先顺序展平的 values 及可选 visibility；
file 保存 feature_file_id 与非负 row_index，两者均绑定 feature_space_id。
`FeatureFile(feature_file_id, feature_space_id, sample_count, values_file, index_file, visibility_file?)`
保存每文件数量与文件引用；相同空间允许不同文件具有不同 sample_count 和哈希。
因此文件校验信息不会造成同一模型空间因批次大小或文件内容变化而被误判不兼容。

同一对象提供了对应合同时，校验 inline 长度/可见性、文件行下标与空间一致性；
同包已知空间声明 visibility 时（包括 KPR），FeatureFile 必须提供 visibility_file；
未声明 visibility 时不得提供该文件。
embeddings 与 visibility_scores 存在同一个 NPZ 时，values_file 与 visibility_file 可以是同一个 SourceFile。
合同或文件记录尚未提交时先保留引用，不因本层缺少数据库上下文而拒绝。
模型不读取数组、不校验实际文件哈希或向量范数；真实 dtype、有限值、L2 约定与安全读取
仍须在后续输入边界检查。

每文件sample_count、数组/索引/图片哈希是来源校验，不属于跨文件必须相等的模型合同。
样本ID、轨迹、帧、向量行逐一对应；track_id和frame_idx允许在不同样本中重复。
图片路径解析后须在允许数据根内；np.load禁用pickle。
模型变化可新增特征空间，同一球员可在不同空间拥有模板，但空间之间不能比较距离或复用阈值。
没有新空间模板时保持未知并补充锚点，不重置球员ID或比赛。

## 距离校准档案与四态距离契约（阶段1）

为了实现与 `feature_space_id` 严格绑定的距离校准与判定体系，禁止跨特征空间静默 fallback：

| 模型 | 字段与约束 |
| --- | --- |
| `DistanceSemantics` | 显式数学与后端度量语义：`backend="kpr"`, `layout="parts"`, `normalization="l2"`, `distance_metric="euclidean_sqrt_visibility_mean_div2"`, `part_weighting="sqrt_visibility"` |
| `DistanceQuantiles` | 经验距离分布与统计分位数：`count`, `min`, `max`, `mean`, `std`, `p01`, `p05`, `p10`, `p25`, `p50` (median), `p75`, `p90`, `p95`, `p99` |
| `CandidatePairExclusion` | 候选正例/物理负例抽取时的结构化排除记录：`pair_type` ("candidate_positive" 或 "physical_negative"), `left_sample_id`, `right_sample_id`, `clip_id`, `reason`, `details` |
| `DistanceCalibrationProfile` | 模式版本固定 `distance-calibration-profile-v1`；绑定 `profile_id`, `feature_space_id`, `calibration_version`, `status` ("uncalibrated", "unvalidated", "provisional", "calibrated")；`near_distance`（工程值 0.18）, `far_distance`（工程值 0.45，严格大于 near）；`min_common_parts=2`, `min_part_visibility=0.2`；`evidence_source`, `evidence_fingerprint`, `config_fingerprint`；可选的 `candidate_positives_summary`, `physical_negatives_summary`, `exclusion_counts`, `notes`；提供基于确定性 JSON 的 SHA-256 `fingerprint` 属性 |
| `CalibrationDiagnosticReport` | 模式版本固定 `distance-calibration-diagnostic-v1`；包含 `profile`, 提取参数, 正负样本统计, 门槛重叠分析及 exclusions 列表 |

四态距离关系语义（`DistanceRelation` 枚举）：
- `"near"`：局部外观一致支持（距离 $\le \text{near\_distance}$），用于单轨迹局部连续支持或候选匹配支持，绝不等于全局同人身份真值；
- `"far"`：强反证（距离 $\ge \text{far\_distance}$），窗口内存在任意 far 对即触发一票否决；单个 far 观测不能直接判为换人；
- `"ambiguous"`：介于 near 与 far 之间的不确定区域，既无充分支持也无一票否决；
- `"uncomparable"`：共同可靠部位数 $< \text{min\_common\_parts}$、可见度过低或特征空间不匹配。

空间校验与准入铁律（2026-11-10 专项修复 R1 与 R2）：
- **R1 校准档案强一致性断言**：在策略（`SegmentationPolicy`, `MatchingPolicy`, `DiscoveryPolicy`）显式绑定 `DistanceCalibrationProfile` 时，系统建立硬断言校验。若策略中的 `near_distance`、`far_distance`、`min_common_parts`、`min_part_visibility` 与绑定的档案声明数值不一致，或 `calibration_fingerprint` 脱钩，显式抛出 `FeatureValidationError`，严禁通过策略参数静默覆盖或旁路校准档案；`evaluate_feature_relation` 优先采用策略绑定的校准门槛，竞争 margin（delta）与物理距离 far 明确解耦。
- **R2 校准候选证据严格准入过滤**：在 `kpr_distance` 计算中引入 `min_part_visibility` 硬门槛过滤，排斥低于该门槛的不可靠部位；在 `extract_candidate_calibration_evidence` 中，严格要求互斥物理负例必须具备真实 `TrackObservation` 与有效 `bbox_xyxy`，且两框 IoU < 0.2；对包含显式混人风险（`mixed_identity`、`identity_conflict`、`suspected_identity_switch`）的样本按其作用时间范围实施硬排除；当候选证据为空时如实返回未校准诊断，杜绝空索引异常崩溃。

## 时间与事件

所有规范片段时间使用秒，区间统一[start_s,end_s)。帧区间如果提供也使用左闭右开。
外部含结束帧仅由adapter转换一次。CFR可用frame/fps；VFR需使用实际时间戳/映射，不能盲除fps。
可选来源映射给出source_video_id及时间变换；缺失则source_time留空。
节次与比赛计时单独保存，不等于视频播放时间。

事件时间为显式联合类型：

- point：time_s；没有伪造的结束时刻。
- interval：start_s/end_s，要求end>start。

Event包括event_id、source_event_key、可选canonical_event_id、clip_id、event_type、time、
outcome（适用时made/missed/unknown）、来源、原始分数和subjects。
投篮不默认命中，未知分值/动作边界/比赛时钟不猜测。
实际字段另含 match_id、可选非负 points 和 game_clock；source 为 Provenance，
raw_scores 为有限数值字典。outcome 默认 null（未提供或不适用），显式未知结果可用 unknown。
GameClock(period, clock_s, direction?) 将比赛节次/计时与视频定位分开，direction 可为 count_up/count_down。
source_event_key由上游稳定提供，或由adapter按明确的输入格式生成，不按批次/CSV行序生成。

每个Subject独立包含subject_id、role和该角色的支持轨迹/身份段引用。
可选 event_jersey_number 和 event_team_id 只表达该事件角色的人工/上游标注，
不能扩展为整轨迹或其所有样本的身份事实，也不自动创建 Player；来源随 Event.source 保留。
role、event_type 为可扩展非空字符串，不限定动作类别或角色数量；同事件 subject_id 必须唯一，
允许多个主体具有相同 role。references 是按 kind 区分的 TrackReference(clip_id, track_id)
或 SegmentReference(segment_id)，每个引用可带自己的 time 点/区间。
省略引用 time 时由后续关联规则使用事件时间；跨片段引用需在其片段坐标下定位，
不能把不同片段的同秒数当作同一时刻。
同包已知目标的显式 time 越界属于契约错误；省略 time 时不提前执行上述定位，
也不把缺帧、未确认或无法关联当作原始事件非法。
射手、传球者、助攻者可为不同人；同一主体的多个支持引用才需要指向同一个人。
点事件在主体的定位时刻查当前有效段；区间事件需支持范围内归属一致，
跨切换、空档、未确认或冲突则该主体unmapped，不随意选最近轨迹。
状态为mapped/unmapped，player_id为当前派生值，不另存一份可漂移的正式归属。

原始 `Event` 不含 player_id。查询模型 `ResolvedEvent(event, subjects, source_time?, validation_reasons=[])` 单独携带
`ResolvedSubject(subject_id, role, status, player_id?, reason?)`；mapped 必须有 player_id，
unmapped 必须为空，解析结果必须逐一对应原事件的 subject_id/role。
可选 SourceEventTime(source_video_id, time) 仅表示已有来源映射时的派生定位，
保持原事件 point/interval 类型。此拆分避免把后续人工身份修正后过期的球员归属固化到原始证据。

`validation_reasons` 是当前上下文揭示的事件级问题。晚到 Clip 揭示原事件时间越界时，
保留 Event 原文，输出 `event_outside_clip`，主体保持 unmapped，事件进入复核且不参与正式统计。
事件自身时间与主体显式引用时间分别校验；合法引用不能消除事件时间越界。
历史重复证据以完整来源键和业务指纹识别，不因晚到目标而作为新事件重新拒绝；
新来源版本、新事件及同版本内容改写仍执行严格校验或报来源冲突。

进球计数按适用事件语义、made结果和shooter/scorer主体计算，每个canonical事件最多一次。
重复交付通过来源键去重；不同来源或回放是否同一次动作需显式canonical_event_id/复核，
不能仅凭相似画面自动宣称完成语义消重。未确认重复时展示证据和状态。
播放/剪辑窗口是派生区间，与事件识别边界分开。

## 修订、读写和一致性

- 写用例核对expected_revision；数据、图库变更、revision、审计原子提交，失败全部回滚。
- 计算使用固定图库快照；当前批自动关联后被接受的模板在后续匹配中生效，避免链式污染。
  初始化特例是先以已确认种子构造固定初始快照，再处理其余证据；该快照不随本批自动关联扩张。
- 初始化是建立第一版锚点，不是重置；重复来源幂等，已有库新证据走update。
- 自动更新不覆盖人工确认、拒绝、拆段、合并或模板复核。
- 事件可以早到/晚到；身份修订后共享关联规则反映到查询和统计，不重复创建事件。
- 恢复生成新审计和更大revision，不倒退版本号。身份段及替换关系、归属、模板、别名、
  约束与审核的有效状态按目标修订还原，并为恢复后的归属、模板和审核追加新版本；
  不删除后续审核来模拟恢复。原始证据和稳定 player_id 保留；共享一致性校验失败则全部回滚。
- 样本清洁批准绑定被审核的完整 SourceKey；样本 supersedes 新版本后须重新审核，
  不继承旧版清洁批准。身份段上的人工身份确认与样本清洁批准分别保留。
- 身份记录按match持久保存，batch缺少某人不删除该档案或历史事件。
- 正式状态源是SQLite。HTTP、CLI、网页、后续导出均通过workflows/read共享查询。

## API

统一snake_case。成功为{data: ..., revision: 整数}，资源流例外。
错误为{error: {code, message, details}}；格式422、缺失404、版本/来源冲突409。
资源使用asset_id，不向前端暴露可任意读取的本机路径。
有来源映射时支持原视频时间排序；无映射只提供片段内定位，不虚构整场顺序。

已实现的接口模型包括 `SuccessResponse[T]`、`ErrorResponse`、带 batch/expected_revision 的
ImportRequest 及初始化/更新/事件导入/匹配预览模型，另有 PlayerQuery、EventQuery、ImportReport、AssetReference。
ReviewRequest 包含 match_id、expected_revision、actor、非空 commands；命令按 kind 区分
确认归属 (`confirm_assignment`)、新建球员 (`create_player`)、拆段 (`split_segment`)、合并 (`merge_players`)、
改归属 (`set_assignment`)、模板审核 (`review_template`)、样本审核前复核 (`review_sample` / `ReviewSampleCommand`，用于在 template_id 建立前独立于身份复核样本模板资格)、
风险处理记录 (`review_risk` / `ReviewRiskCommand`，登记风险处置决议及 clean 样本白名单)、
人工约束设置 (`set_constraint` / `SetConstraintCommand`，针对段设置 forbid_player 或 cannot_link 硬互斥约束)
和恢复修订 (`restore_revision`)，并要求 reason。
改归属的 SetAssignmentCommand 也可在 player_id=null 时携带 unassigned_reason。
这些模型只描述输入输出，当前不执行命令、不提供 HTTP 路由，也不实现 404/409 的数据库判断。

## 配置契约

`load_settings(config_path)` 使用标准库 tomllib 读取 UTF-8 TOML，再进行严格类型校验。
原有六个 section 和所有列出字段均必需；segmentation、discovery 为可选 section，未知 section/字段报错：

| Section | 字段与范围 |
| --- | --- |
| paths | input_root、runtime_root：非空路径，相对配置文件所在目录解析为绝对路径 |
| input | adapter：normalized、machine 或 sample_test |
| match | match_id：非空字符串 |
| matching | min_common_parts、max_templates_per_player：正整数；auto_assign_enabled：布尔值；可选 policies 精确空间策略列表；可选 distance_calibration 绑定的校准档案 |
| playback | before_event_s、after_event_s：非负有限秒数 |
| server | host：非空字符串；port：1..65535 的整数 |
| segmentation（可选） | SegmentationPolicy，缺省为 track-consistency-v1 工程门槛；含近/远距离、可见性、时间桶（min_support_buckets 默认 3, min_match_buckets 默认 2）、采样/观测间隔、位移、重复框与镜头运动提示参数；version 支持 track-consistency-v1/track-consistency-v2/track-consistency-v3（v1/v2 下 local_support_windows 默认 false，v2 开启连续局部支持窗；v3 默认自动启用局部支持窗、连续视角过渡区 enable_transition_zones、过渡不确定阈值 transition_uncertainty_threshold、局部风险收敛 scope_risks_to_breaks 与相对时间分桶 ensure_relative_buckets）；可按 feature_space_id 覆盖距离门槛；支持绑定 calibration_profiles 与 strict_space_calibration |
| discovery（可选） | DiscoveryPolicy，缺省 complete_support_v1/absolute；version 支持 complete_support_v1/complete_support_v2/complete_support_v3；support_mode 支持 absolute、cooccurrence_contrastive、compensated_contrastive 与 constrained_manifold（交付推荐模式）；含流形聚类门槛 clustering_threshold（默认 0.28）、existing_player_match_threshold、linkage（complete/average/single，默认 average）；支持 min_match_buckets（默认 2）与 short_segment_match_threshold 短段匹配门槛；负样本补偿开关 enable_geometric_negative_compensation 及 compensation_* 系列参数；以及解耦的 near/far/margin 阈值字段；支持绑定 calibration_profile 与 calibration_fingerprint；calibration 固定 unvalidated |

不将字符串转换为数值、不接受以布尔值充当数字；拒绝 NaN/Inf。
读取或校验失败抛出含配置文件位置和原因的 SettingsError。
路径规则不受命令执行目录影响，不展开环境变量；加载不创建目录、写数据库、修改输入，
也不要求输入/运行目录已经存在。路径落点与允许数据根内的文件访问检查由后续适配器负责。

MatchingPolicy 集中保存 feature_space_id、policy_version、距离/聚合/模板准入版本、共同部位/容量、
自动开关、分桶间隔/最少桶数、共同覆盖/支持率、L2 验证容差、可见性极小值及可空 tau/delta。
所有距离阈值与同一空间策略绑定；tau/delta 无默认值且必须成对提供。
Calibration 包含 status、dataset_version、policy_fingerprint；validated 要求与准确策略指纹一致，
不表示本项目已完成真实标注。开启自动确认必须存在有效校准；未配置空间不得继承自动确认。
具体可调用例及人工/数据库责任边界见 [identity-core.md](identity-core.md)。

## 无答案机器接入和冻结结果

`load_machine_input(settings, batch_id=..., clip_ids=None)` 返回
`(EvidenceBatch, dict[(feature_file_id, row_index), InlineFeature])`；
`load_machine` 只返回同一标准包。clip_ids 是可选机器片段筛选，不从标注生成。
原始 track ID 在片段内规范化为字符串，样本 ID、段 ID、特征空间 ID 不依赖交接 batch。
每条原轨迹生成的暂定 Segment 不代表同人事实；实际观测帧、时间与来源指纹独立保留。
FeatureFile、FileFeatureReference 与内存 InlineFeature 一一对应；P/D 与 N 均由合同、数组和索引确定。
来源文件哈希覆盖实际读取的机器文件；任何答案侧文件不进入来源清单或记录指纹。

无标注 baseline 结果是现有契约的版本化导出：evidence.json 保存 EvidenceBatch，
predictions.json 保留 Segment 的片段/轨迹/范围与对应 Assignment（未知时 player_id=null），
diagnostics.json 保存工作流报告和当前待复核原因，config.json 保存有效配置，
input-manifest.json 保存 SourceFile 清单与来源哈希，summary.json 保存计数、幂等结果与能力边界。
新 SQLite 是本次正式状态；artifact-hashes.json 是输出文件名到 SHA-256 的映射，
包括本次数据库且不含清单自身，用于冻结、重放核对和后续独立评分。
config.json 另存 effective_policies、baseline_version、initial_state，以及实现文件哈希和环境版本。

predictions.json 的冻结封装为：

| 字段 | 契约 |
| --- | --- |
| schema_version | 固定 `baseline-predictions-v1` |
| match_id / revision | 本次工作流状态作用域与修订 |
| segmentation | 默认 `provisional_raw_tracks_not_identity_verified`；开启 --segment-tracks 时为 `machine_consistency_proposals_not_identity_verified` |
| segments | 现有 Segment 列表，按 segment_id 排序 |
| assignments | 现有 Assignment 列表，按 segment_id 排序；不对 player_id 重编号 |
| players | 现有 Player 列表；空库可能为空 |

input-manifest.json 的 schema_version 为 `machine-input-manifest-v1`，保留 input_root、
sources（现有 SourceFile，按 path 排序）和 fingerprint；fingerprint 为规范化 sources 的 SHA-256。
推理输入不含标注侧哈希；改变答案文件不应改变该清单、预测或诊断。

未来独立评分可按 match_id、clip_id、track_id、段半开范围和 sample_id 定位预测；
匿名 player_id 的评分对齐只发生在评分阶段，不回流到推理，不加载评分前的人工图库。
本轮仅预留该接口，不读取或生成身份答案，不报告识别准确率。

## 轨迹内一致性报告（当前阶段）

`segment_tracks` 返回 `SegmentationResult(batch, report)`；batch 仍为 EvidenceBatch，
不另建身份模型。原 Segment 停用，派生 Segment 用 supersedes_segment_id 记录父段，
完整划分父范围。原始 Track、TrackObservation、样本描述、特征引用和来源保留；
派生样本段关联可与 raw-evidence.json 对照。原始文件不变。

`SegmentationReport` 的 schema_version 为 `track-consistency-report-v1`，记录 policy、
policy_fingerprint、input_fingerprint、derived_evidence_fingerprint、逐轨迹 tracks、duplicate_candidates 与 summary。
每条 TrackConsistencyReport 包含 parent_segment、proposals、findings、comparisons、
exclusions 和现有 IdentityRisk。SegmentProposal 保存现有 Segment、parent_segment_id、
status=supported/unknown、reasons、support_sample_ids、source_observation_ids，
can_match_existing（StrictBool，默认 True）、can_create_new_identity（StrictBool，默认 True），
identity_verified 恒为 false。未知段通过风险契约保留阻塞原因。
阶段2引入资格解耦：时间桶不足 3 桶但达到 2 桶的短段标记为 short_segment_limited_match_support（can_match_existing=True, can_create_new_identity=False），允许增量匹配已有图库，但严禁独立聚类建档。

ConsistencyPair 按 feature_space_id 保存 sample 对、可比较性、共同部位、有限距离或失败原因；
不合并不同空间的距离。ConsistencyFinding 保留半开时间/可选帧范围、sample/observation ID 与证据详情。
疑似切换的 exact_boundary_known=false；不确定范围不是一个伪精确切换帧。
同帧高 IoU 框报告 duplicate_box_candidate、cannot_link=false，不直接形成身份排他约束。

开启 CLI 分段开关后新增 raw-evidence.json 与 segmentation.json；config.json 保存实际策略，
summary.json 与 predictions.json 保存分段摘要。证据不足时长比例分母为各原轨迹范围之和，
重叠轨迹分别计数；样本比例分母为全部输入样本，无样本时为 null。
支持段只是自动发现的候选输入，不自动等同于已确认归属或模板准入。
derived_evidence_fingerprint 绑定派生批证据与实际解析特征，发现拒绝缺少绑定或证据变更后的陈旧报告。

`SegmentationPolicy.version` 接受 `track-consistency-v1`、`track-consistency-v2`、`track-consistency-v3`，默认 v1；
`local_support_windows` 为严格布尔值，默认 false 保留旧行为。自动发现示例显式用 v2/true。
v2 仅在非 hole、具有结构观测且因外观冲突/歧义失败的区间提取连续全对 near、至少三个时间桶的支持窗；
窗外和不相容图片保持未知，不跳过中间样本，不降低近远距离门槛。
新增 finding=local_appearance_support_window，理由=local_all_pairs_consistent_with_independent_temporal_support，
保留 parent_interval_reason、identity_verified=false、switch_inferred=false；局部边界不表达 identity_switch。

v3（阶段2）重构了局部一致支持判定逻辑，用局部连续外观窗口与中心支撑替代容易误杀正常转身/跑动的全对 near 规则，并在模型校验中若 `version == "track-consistency-v3"` 自动默认开启各项优化：
- **局部连续外观窗口判定**：窗口内要求相邻样本连续（`relation(s_i, s_{i+1}) == "near"`）、零 far 强反证一票否决（任意样本对距离 $< \text{far\_distance}$，绝不用多数票或均值掩盖异类）、存在中心代表样本支撑（至少存在一个样本 $m$ 与窗口内所有其他样本均 near），以及观测间断 $\le 0.25$s 与采样间隔 $\le 1.5$s。
- `enable_transition_zones`（StrictBool，默认 False；v3 默认为 True）：在局部支持窗的前后边界，若存在非运动突变且未发生严重外观冲突的渐变间隙，提取 `kind="viewpoint_transition_zone"` 发现与 `status="unknown", reason="viewpoint_transition_uncertainty"` 的过渡段，避免将人体连续平滑旋转（如转身过程）误判为剧烈断裂；配合 `transition_uncertainty_threshold`（Probability，默认 0.45）控制过渡不确定度。
- `scope_risks_to_breaks`（StrictBool，默认 False；v3 默认为 True）：将父轨迹的运动跳变等质量风险（`quality_review`）严格限制在断点/切换边界局部，连续且一致的干净 supported 段不再被动连带继承父段的全局跳变风险，解决长轨迹局部跳动导致全轨不可用的过度阻断问题。
- `ensure_relative_buckets`（StrictBool，默认 False；v3 默认为 True）：在评估时间支持桶时，以段内首个样本时间为相对基准 `math.floor((s.time_s - samples[0].time_s) / bucket_s)` 进行对齐，消除跨片段或全局绝对时间偏移导致的离散分桶伪效应。
- `min_match_buckets`（PositiveInt，默认 2）：支持段允许参与已有身份匹配所需的最小非重复时间桶数。短于 min_support_buckets (3) 但达到 min_match_buckets (2) 的段赋予匹配资格，但阻断建档资格。

算法、工程门槛和局限见 [track-consistency.md](track-consistency.md)。


## 无种子自动发现契约

`contracts/discovery.py` 定义完整支持链。以下是机器自动证据，不是人工 ReviewRecord：

| 模型 | 语义 |
| --- | --- |
| DiscoveryPolicy | version=complete_support_v1/complete_support_v2/complete_support_v3，support_mode=absolute/cooccurrence_contrastive/compensated_contrastive/constrained_manifold（交付推荐模式），选定空间、流形聚类与补偿参数及解耦门槛；calibration 只能 unvalidated |
| DiscoveryComparison | 原始样本对、可空有限距离、共同部位数或不可比较原因 |
| DiscoveryDirection | 查询段、原代表查询及支持样本集合、支持覆盖和可空方向中位数；competitor_segment_ids 与逐查询 sample_contrasts 保存独立 negative、最近两原样本/均值/margin/支持原因 |
| DiscoveryPair | 稳定 pair_id、左右段、supported/conflict/insufficient/ambiguous、分数、原因、方向、原始比较与几何冲突观测 ID |
| AutomaticTemplateSupport | 模板 sample/segment/space、支持 pair ID、另片段的非重复原始 witness_sample_ids |
| DiscoveredGroup | group_id、固定 anchor_segment_id、成员段、全对支持及自动模板支持 |
| UnknownSegment | 未归人段及明确原因；不是“其他人物”或“已确认为不同人”的同义词 |
| DiscoveryReport | schema_version=identity-discovery-v1、report_id、match、status=experimental、策略及输入/分段指纹、空间、组、段对、未知、摘要 |

`DiscoveryPolicy` 集中管理冷启动无种子身份发现的核心参数，其 `version` 支持 `complete_support_v1`、`complete_support_v2` 与 `complete_support_v3`；`support_mode` 支持 `absolute`、`cooccurrence_contrastive`、`compensated_contrastive` 以及交付推荐模式 `constrained_manifold`。

各支持模式工程参数与解耦门槛规范如下：
1. **解耦的近/远/边距阈值体系**（通过模型验证器自动双向同步缺省值）：
   - **Near 系列**：`intra_segment_near_distance`、`inter_segment_tau`、`tau`（工程初值 0.18；只要显式提供任一者，其余自动同步对齐）。
   - **Far 与上界系列**：`candidate_far_distance`、`negative_evidence_distance`、`far_distance`（工程初值 0.45；自动同步对齐），以及段对最大距离 `max_pair_distance`（初值 0.30）。
   - **Margin 系列**：`contrastive_margin`、`counter_evidence_margin`、`rival_margin`、`margin`（工程初值 0.04；在 `complete_support_v3` 下默认同步提升至 0.06）。
   - **门槛严格校验**：当 `support_mode != "constrained_manifold"` 时，模型严格执行 `inter_segment_tau <= max_pair_distance < candidate_far_distance` 不等式校验；在 `constrained_manifold` 模式下放宽此项校验，交由全局时空互斥图与层次凝聚聚类决策。
   - 基础多部位可见性与时间支持：`min_common_parts=2`、`min_part_visibility=0.2`、`bucket_s=0.5`、`min_support_buckets=3`、`min_coverage=0.8`，默认模板容量 `max_templates_per_group=6`；同片段共现分离至少三桶、中心间距至少平均框高的 0.5 倍。
2. **流形聚类与已有球员匹配新增字段**（`constrained_manifold` 交付推荐）：
   - `clustering_threshold`（NonNegativeFloat，默认 0.28）：时空受约束层次凝聚流形聚类的切分阈值，与真实赛场特征距离中位数分布（0.268）精准吻合。
   - `existing_player_match_threshold`（NonNegativeFloat | None，默认 None）：增量模式下匹配已有球员图库时的独立距离门槛。
   - `linkage`（Literal["complete", "average", "single"]，默认 "average"）：层次凝聚聚类的簇间连接度量准则；支持平均连接（average，默认推荐）、完全连接（complete）与单连接（single）。
3. **几何负样本补偿新增字段**（`compensated_contrastive` 及 v3 补偿机制）：
   - `enable_geometric_negative_compensation`（bool，默认 False；在 `support_mode="compensated_contrastive"` 或 `version="complete_support_v3"` 时自动置为 True）：当特写或单人镜头缺失同屏共现负样本时，启用基于高置信度多部位外观锚点的保底补偿，打破传统对比模式因缺失 negative 对手导致的识别死锁。
   - `compensation_min_common_parts`（PositiveInt，默认 4）：补偿锚点所需的最小可靠共同可见部位数（要求更严苛的结构完整性）。
   - `compensation_min_part_visibility`（Probability，默认 0.35）：补偿计算所需的部位可见度最低门槛。
   - `compensation_max_distance`（Probability，默认 0.22）：候选样本平均加权距离上限。
   - `compensation_max_raw_distance`（Probability，默认 0.28）：原始全量样本距离上限，任何可用原始查询超出此阈值将触发少数样本否决。
   - `compensation_rival_margin`（NonNegativeFloat，默认 0.06）：与潜在对手之间的最小安全间距。
4. **短段增量匹配门槛与空间校准绑定新增字段**（阶段1与阶段2）：
   - `min_match_buckets`（PositiveInt，默认 2）：增量匹配已有图库所需的最少非重复时间支持桶数。允许 2 桶短段匹配已有球员，但未匹配短段严格受 `can_create_new_identity=False` 拦截，杜绝独立建档；
   - `short_segment_match_threshold`（NonNegativeFloat | None，默认 None）：短段匹配已有球员时的独立更严距离门槛；未提供时回退到 match_threshold；
   - `calibration_profile`（DistanceCalibrationProfile | None，默认 None）：绑定的冻结距离校准档案，自动同步 `feature_space_id`、`intra_segment_near_distance` 与 `far_distance`；
   - `calibration_fingerprint`（str | None，默认 None）：绑定的校准档案 SHA-256 指纹，确保跨批次实验可验证可重放。
5. **部分复制与独立见证硬约束**（2026-11-10 专项修复 R4）：
   - 在 `_evaluate_segment_pair` 中激活非共享独立组件基数校验：双方非共享组件必须同时满足 `unshared_left >= 3` 与 `unshared_right >= 3`；
   - 激活方向覆盖率硬门槛 `coverage >= 0.8`；
   - 模板准入严格要求唯一跨片段见证人数量 `len(set(witness_sample_ids)) >= 2`，阻断跨片部分复制序列欺骗建组与虚假多见证。

模式语义差异：
- `absolute`：保持旧绝对门槛，要求跨段样本距离严格低于 tau。
- `cooccurrence_contrastive`：tau 仅用于段内一致性；跨段支持要求 candidate 最近两独有样本均值 <=far_distance=0.45，并相对 candidate 所在片段中所有可靠空间共现排他段的最近两均值低至少 margin=0.04。每方向至少一个独立 negative，每个 negative 都须至少两份可比较独有代表，不足时不产生支持；每方向仍需至少三桶和 80% 原代表分母。所有非重复可用原始样本检查强支持 negative 的少数反证。
- `compensated_contrastive`：在共现对照基础上，若同片段缺失时空共现负例且开启补偿，通过严苛部位数、可见度与距离锚点提供正向支持理由 `compensated_contrastive_support_lacking_geometric_negatives`。
- `constrained_manifold`（交付推荐）：采用多部位加权池化特征与物理时空绝对排他（同屏时间重叠两轨迹绝对不可为同人）构建 Must-Not-Link 约束，执行时空受约束层次流形聚类；平滑传递视角变换流形，彻底消除桥接节点恶性连坐，在空库下实现 21 位核心跨视频球员识别，全场零同屏违例。

实际策略、指纹和实现哈希随每次运行保存，不能从结果或答案反向选本次阈值。
远原始样本对只保留诊断，不单独作为不同人事实；几何排他和可靠少数反证仍可否决。
contrastive 的组外竞争仅考虑有独立几何排他证据的组外段；模板必须是该方向 supported q，引用 candidate 最近两 raw witnesses，
不用段内 tau 限制每个跨视角单对。模板见证再按全局重复组件去重。
sample_contrasts 的每项保存 sample_id、representative、candidate_distance、candidate_sample_ids、
competitor_scores（segment_id/distance/sample_ids/margin）、supported、reason；几何 negative 可经段对报告追溯原观测。


沿用 Player、Assignment、Template 的正式存储模型，不另建一套当前身份状态。
自动成员写 status=proposed、method=automatic，evidence 引用自动报告与段对支持；
已分配 player_id 是持久匿名预测档案，未达到 confirmed 的正式确认语义，不能用于正式个人事件计数。
ID 由 workflow 以 match 和固定 anchor 段生成 UUID5，不依赖号码、批次或输出排序；同报告幂等。

Template 新增 admission=manual/automatic_experimental（默认 manual）和 automatic_support_id。
自动模板首次准入 active=true、review_status=unreviewed、approval=null，必须有报告支持；
显式人工拒绝后可为 rejected。自动 admission 禁止 approved 或伪造 approval；人工批准须形成 manual 模板与真实审核记录。
active 的自动模板仅是实验图库成员，不满足人工匹配引用资格，不反向证明本次身份发现。
IdentityRisk 新增 basis=unspecified/raw_track_continuity_unverified，默认 unspecified。
自动发现仅在 supported 段对后者的 quality_review 基础准备条件使用机器替代支持，其余风险继续阻塞。

SQLite 迁移 007 的 identity_discovery_reports 按 match/report 保存不可变报告 JSON、组到 player ID 映射和 revision，
与 Player、Assignment、Template 及自动审计原子提交。由 workflows.read.get_discovery_reports 查询，
恢复历史映射不会篡改原支持报告。单次冷启动发现支持空库初始化及同报告重放；持续开放集合增量分批更新则支持分批序列化到达与多版本演进。

`baseline --discover-identities` 隐含分段，导出 discovery.json（report、player_ids_by_group、committed_revision）。
predictions.json 另增 identity_groups（含 player_id 及成员段/模板支持）、unknown_segments、templates、
discovery_report_id、discovery_policy_fingerprint、实验和校准状态；summary.json 加入发现摘要及重放幂等结果。
冻结配置含实际 discovery 策略，完整原始来源仍由 evidence/input-manifest 提供。算法和限制见 [identity-discovery.md](identity-discovery.md)。

## 开放集合增量分批更新契约与产物（阶段 4）

`baseline --incremental-batches [--batch-size N] [--reverse-batches]` 支持比赛片段分批到达时的增量演进：

1. **批次更新请求与原子事务**：调用 `workflows.incremental.update_incremental`，核对 `expected_revision`，输入当前批次标准 `EvidenceBatch`、`SegmentationReport` 及内存特征。事务基于当前 Revision $R$ 的快照读取，批内完成已有匹配或新集团发现后原子递增至 $R+1$；每批报告作为独立记录存入 `identity_discovery_reports`。
2. **三态决策契约 (`IncrementalSegmentDecision`)**：
   - `category="matched_existing"`：提议关联到已有 `player_id`，assignment 标记 `status="proposed", method="automatic"`，evidence 类型为 `automatic_incremental_match`。
   - `category="created_new_identity"`：跨片段集团提议分配新匿名 `player_id`，assignment 标记 `status="proposed", method="automatic"`，evidence 类型为 `automatic_discovery_incremental`。
   - `category="insufficient_evidence"`：保持待定，assignment 标记 `status="unassigned", player_id=None`，并在 evidence 记录结构化 `deferred_reasons`（如 `insufficient_cross_clip_support`, `segment_not_supported`, `appearance_conflict_including_minority` 等）。
3. **指标与语义隔离**：
   - `summary.json` 输出 `matched_existing`, `created_new_identity`, `insufficient_evidence`, `supported_segment_coverage`, `all_segment_coverage` 及 `deferred_reasons` 字典。
   - 覆盖率仅代表段分配比例，附带 `coverage_note` 明确禁止与真实身份准确率混淆。所有产物状态一律标注 `calibration="unvalidated"`, `identity_status="experimental"`。算法与验收见 [incremental-identity.md](incremental-identity.md)。

## 统一覆盖漏斗与评测契约（阶段0）

基线实验（`baseline-discovery-experiment` 与 `baseline-incremental-experiment`）统一输出 `funnel.json`（schema: `player-library-coverage-funnel-v1`），并在 `summary.json` 中附带 `funnel` 和 `loss_summary` 摘要：

1. **7 级覆盖漏斗 (`CoverageFunnel`)**：
   - 必须基于固定分母（固定 10 个比赛片段原始机器证据输入：158 轨迹, 1759 样本, 75465 逐帧观测, 1346.35s 时长）：
     * Stage 1: `raw_tracks` (原始轨迹全量，158 轨迹 / 1,759 样本 / 1,346.35s)
     * Stage 2: `usable_samples` (tracking_quality.usability != 'unusable'，1,759 样本)
     * Stage 3: `locally_consistent_segments` (局部一致且状态为 supported 的派生段，452 样本 / 137 段 / 179.6s)
     * Stage 4: `cross_clip_comparable_segments` (单段合格且具有跨片时空互斥节点的比较段，修复统计缺陷后为 208 样本 / 43 段 / 85.8s)
     * Stage 5: `candidate_matches` (在聚类或增量图库匹配中具有正向距离支持的段)
     * Stage 6: `final_assigned_identities` (通过时空与见证约束最终赋 ID 的段)
     * Stage 7: `templates_admitted` (被选入身份图库快照的活跃模板样本)
   - 漏斗单调性与增量聚合规则（2026-11-10 专项修复 R3）：
     * 第4级严格按正向候选白名单 `cross_clip_unassigned_reasons` 筛选，杜绝未决 unknown 导致虚假回弹；
     * baseline 增量运行汇总全批次分段报告构建全局 `combined_seg_report`，确保增量各级严格满足单调递减物理定律（452 -> 166 -> 141）。
2. **多维损失归因 (`LossBreakdown`)**：
   - 按 `clip`、`track`、`reason`（带多标签重叠追踪，避免相加失真）以及增量 `batch` 提供样本、观测、时长流失明细。
3. **监督指标契约扩展 (`SupervisedMetrics`)**：
   - `merge_errors` 与 `split_errors` 严格对齐持久 `player_id`；
   - 扩展 `experimental_group_merge_errors` 与 `experimental_group_split_errors`，供无监督聚类分组实验独立统计，不与持久身份混为一谈。
   - 跨片段对偶 F1 公式固定为 $2\text{TP}/(2\text{TP}+\text{FP}+\text{FN})$，当 $\text{TP}=0$ 且分母 $>0$ 时严格为 0.0。

# 标准数据契约 v2

字段和类型由 app/contracts 各模块唯一实现；本文固定语义，并与代码同次修改。
v2修正上一版以样例dataset为作用域、仅支持进球点事件的限制。
当前已实现 Pydantic 2 模型及局部校验、sample_test 只读适配器与核验 CLI；
身份核心纯算法和计划已实现；normalized 适配器、持久化、正式身份/事件工作流和 HTTP 服务仍待实现。
下文中的幂等、事务和当前归属是后续工作流必须保持的语义，不表示这些能力已经实现。

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
| IdentityEvidence | tracks、segments、samples、feature_spaces、feature_files 各自默认空列表，不要求同时提供 |
| Track | 原始范围含 match_id、clip_id、track_id、start_s/end_s、可选成对帧范围及 source；observed_frame_indices 可选；不等同于 Segment，也不含球员归属 |
| Sample | match_id、sample_id、clip_id、track_id、time_s、source 必需；frame_idx、segment_id、image_asset_id、tracking_quality、appearance 可选；features 默认空列表 |
| TrackingQuality / AppearanceEvidence | 跟踪 score、score_semantics、遮挡/混人、usability、clean_review、risks 与可选队伍、号码、置信度分别表达，不确立身份或模板资格 |

批内提供的 clip、track、segment、sample、event 的 match_id 必须与包一致。
各实体列表的主键必须唯一：sources 用 file_id，assets 用 asset_id，clips 用 clip_id，
events 用 event_id，身份分支的 segments、samples、feature_spaces、feature_files 分别用其 ID；
tracks 使用 (clip_id, track_id) 组合键。同一事件内的 subject_id 唯一，跨事件可重复。
即使内容完全相同，同一实体列表内重复主键仍报错，不静默覆盖或去重。
不同样本可以来自相同 track_id/frame_idx，样本主键 sample_id 仍须不同。

引用目标不要求同时出现；目标已在同包提供时，模型检查已知事实的一致性：

- Sample 引用已知 Segment 时，clip_id、track_id 必须匹配，时间须落在该段范围内；
  样本和目标双方均提供帧定位时，再检查帧号范围。
- Sample 引用已知 Track 时，时间及双方已提供的帧定位须位于轨迹范围内。
  Segment 保留的选择范围可以宽于 Track 的实际观测首尾范围，二者分别检查 Clip 界限，
  不要求 Segment 被 Track 的观测范围包含，避免把首尾缺帧误判为非法。
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

原始 `Event` 不含 player_id。查询模型 `ResolvedEvent(event, subjects, source_time?)` 单独携带
`ResolvedSubject(subject_id, role, status, player_id?, reason?)`；mapped 必须有 player_id，
unmapped 必须为空，解析结果必须逐一对应原事件的 subject_id/role。
可选 SourceEventTime(source_video_id, time) 仅表示已有来源映射时的派生定位，
保持原事件 point/interval 类型。此拆分避免把后续人工身份修正后过期的球员归属固化到原始证据。

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
- 恢复生成新审计和更大revision，不倒退版本号。
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
确认归属、新建球员、拆段、合并、改归属、模板审核和恢复修订，并要求 reason。
改归属的 SetAssignmentCommand 也可在 player_id=null 时携带 unassigned_reason。
这些模型只描述输入输出，当前不执行命令、不提供 HTTP 路由，也不实现 404/409 的数据库判断。

## 配置契约

`load_settings(config_path)` 使用标准库 tomllib 读取 UTF-8 TOML，再进行严格类型校验。
以下六个 section 和所有列出字段均必需，未知 section/字段报错：

| Section | 字段与范围 |
| --- | --- |
| paths | input_root、runtime_root：非空路径，相对配置文件所在目录解析为绝对路径 |
| input | adapter：normalized 或 sample_test |
| match | match_id：非空字符串 |
| matching | min_common_parts、max_templates_per_player：正整数；auto_assign_enabled：布尔值；可选 policies 精确空间策略列表 |
| playback | before_event_s、after_event_s：非负有限秒数 |
| server | host：非空字符串；port：1..65535 的整数 |

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

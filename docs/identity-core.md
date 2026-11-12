# 身份核心算法层 v1 调用与边界

已实现纯计算链：标准 EvidenceBatch → prepare → associate → resolve → gallery plan。
函数不打开特征文件、不写数据库、不分配 player_id、不执行 ReviewCommand。
SQLite 存储层、业务工作流闭环（workflows.initialize/update/review/read）与事件关联（events.validate/associate）均已完整实现；目前仅 normalized 适配器、HTTP 服务与原生网页留待后续实现。
算法规范继续以 [identity-policy.md](identity-policy.md) 为准。

## 最小调用

```python
from app.identity import (
    prepare_evidence,
    associate_segments,
    resolve_associations,
    plan_gallery,
)
from app.settings import load_settings

settings = load_settings("config.example.toml")
policy = settings.matching.policy_for(feature_space_id)

# batch: EvidenceBatch；snapshot: GallerySnapshot，由调用者提供完整固定状态。
# 文件引用先在 adapter/读取边界安全加载；纯算法只接收已解析的内存值。
# InlineFeature 直接包含在 Sample.features 时不需要 feature_values。
feature_values = {}  # {(feature_file_id, row_index): InlineFeature(...)}
prepared = prepare_evidence(batch, policy, feature_values=feature_values)
associations = associate_segments(
    prepared,
    snapshot,
    policy,
    feature_values=feature_values,
)
resolutions = resolve_associations(prepared, snapshot, associations, policy)
gallery_plan = plan_gallery(
    prepared,
    snapshot,
    policy,
    final_assignments=[r.assignment for r in resolutions if r.assignment is not None],
    feature_values=feature_values,
)
print(gallery_plan.model_dump_json(indent=2))
```

同一批所有段调用 associate 时使用同一 snapshot。gallery_plan 仅提出后续修改；不可将它的
激活项回填为当前批匹配模板。每条 AssociationResult 记录 policy_fingerprint、空间、图库 revision、
模板清单、共同查询样本和逐样本比较。prepared 和结果与输入分离，不修改输入或快照。
`uv run python -m examples.identity_preview` 可运行一份无文件输入、无校准阈值的合成预览。

GallerySnapshot.evidence 需包含模板的 Sample、Segment 和 FeatureSpace，assignments 是当前有效
归属，players 列出比赛中所有未合并球员（缺模板的球员也必须列出），risk_reviews 是已接受的风险处理。
缺引用或缺少内存特征会保留排除/待定原因；非法数值、形状、可见性、归一化或后端合同明确报错。
不同 feature_space_id 不比较；同 ID 的语义合同在查询与图库不一致时明确报错。

## 结果解释

- PreparedSegment.samples 保留全部可用图片用于分歧检查；representative_sample_ids 才是计分代表。
  exclusions 包含不可用、缺特征、重复观测和桶内未选样本；覆盖记录前中后区间和实际观测首尾。
- AssociationResult.candidates 仅使用共同查询集合 Q。local_candidates 是局部诊断，
  `single_template_preview` 仅为单模板预览；不同局部查询集合的排名不能当自动确认依据。
  Candidate.common_parts 是该候选实际使用比较对的最小共同部位数。
- 距离不是概率。未设置 tau/delta 时保留距离，支持票为空；未校准时不会输出自动 match_existing。
  单候选 margin=null，没有无限候选差距；弃权样本仍在支持率分母中。
- Resolution 为 pending/rejected/create_new_proposal/match_existing。新身份提案不含最终 ID。
  校准有效但开关关闭时 match_existing.assignment 为 proposed；开关开启且所有门槛通过才为
  confirmed/method=automatic。assignment.revision 是计算所据快照版本，不代表已经提交新修订。
- GalleryPlan 为 activate/deactivate/await_review/replacement_required。确认身份与批准模板分开；
  只有当前 confirmed 归属、有效特征、无风险并有独立人工批准才提出激活。容量超限给出替换选择，
  不自动淘汰已有人工锚点。改归属、停用/替换身份段或模板审核失效会提出旧模板撤销。

## 人工上下文

这些函数接收已经接受、有 actor/reason/revision 的记录，不解析或执行 API ReviewCommand：

- prepare_evidence 的 risk_reviews：RiskReview；cleared 表示人工作用域内明确排除风险，
  confirmed 表示风险仍成立。仅批准若干图片不能清除整段混人风险。
- resolve_associations 的 manual_assignments：人工确认/拒绝等 Assignment 上下文；已有 manual/seed
  决定原样保留，新冲突另报。constraints 使用 ManualConstraint：forbid_player 拒绝特定目标，
  cannot_link 对明确作用段统一检查，block_automatic 阻止自动处理；不从片段同秒数推断互斥。
- new_identity_requests 使用 NewIdentityRequest，要求独立人工意图及本段 clean_approved 支持样本；
  风险/强分歧仍需处理，不执行新建档案。block_automatic 不否定独立人工提案路径。
- plan_gallery 的 final_assignments 是最终上下文增量（叠加快照），reviews 为 ManualTemplateReview，
  明确 sample_id、player_id、feature_space_id 和人工记录；旧归属的批准不能复用于另一人。
  批准记录早于最终归属修订时只给复核计划。Template.approval 可保留历史人工记录；现有快照的
  approved 状态视为调用者已接受的人工审核事实，新增激活计划必须带完整 review。

## 配置和校准

MatchingPolicy 是唯一算法参数合同，settings.matching.policy_for 按空间选择。
现有 min_common_parts、max_templates_per_player、auto_assign_enabled 仍为共享配置，
`[[matching.policies]]` 可补充空间、策略版本、时间桶、覆盖/支持率、数值容差及校准字段。
显式空间策略的三个共享字段必须与 matching 一致；省略时继承。默认抽样桶 0.5 秒、至少 3 桶、
共同覆盖与支持率至少 0.8；tau/delta 无默认值。

校准数据必须通过独立身份标注和片段隔离验证；当前项目没有已校准策略。
外部完成验证后，将 Calibration(status="validated", dataset_version=..., policy_fingerprint=...)
绑定到准确策略。可用未启用自动确认的 MatchingPolicy.fingerprint 计算待验证指纹。
指纹包含空间、版本、阈值、采样/聚合/准入及数值参数；不包含自动开关和校准记录本身。
变更参数使原指纹失效。人工填写 validated 只是一条外部声明，算法不能证明真实标注或验证质量。

开启 auto_assign_enabled 而缺少适用的有效校准时配置报错；未知空间也不会继承已开启的自动确认。

## 四态距离判定与校准档案绑定

纯函数 `evaluate_feature_relation(feature_a, feature_b, policy, profile=...)` 在 `app/identity/distance.py` 中导出：
- 返回 `DistanceRelation` 枚举（`near`, `far`, `ambiguous`, `uncomparable`）；
- 优先读取 `policy.calibration_profile` 绑定的冻结近/远门槛与可见度过滤参数；
- 运行时严格执行校准档案一致性断言（R1）：若策略参数与档案声明冲突，显式抛出 `FeatureValidationError`；
- 阶段3（第二轮）将基于阶段2产出的局部一致段，引入多原型身份表示与接口适配，当前核心接口与调用规范保持向前兼容。
未匹配、距离远、无模板或不可比较始终不能自动新建球员。

## 后续工作

SQLite 存储层现提供来源幂等、迁移、一致快照、revision 比较更新及审计原语，
并提供历史范围重叠与别名环检查原语；见 [storage.md](storage.md)。正式 ID 分配、人工命令执行、
何时应用检查及完整原子提交编排仍属 workflows；事件当前关联与统计仍属 events/workflows。
本层只验证传入对象可用事实，不声称完成正式建库、真实比赛精度或全场人数发现。

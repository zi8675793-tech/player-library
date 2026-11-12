# 标准数据契约规范 (Data Contracts Specification)

字段和数据类型由 `app/contracts/` 及相关领域模块唯一权威实现；本文固定所有结构的数据语义、校验约束及版本演进规则，与代码实现保持 100% 同步。

全仓通用规范：
- 模型统一采用 Pydantic 2 `BaseModel`（配置 `extra="forbid"`, `strict=True`）；
- 统一使用 `snake_case` 字段命名；未知额外字段一律抛出验证错误；
- 标识符（ID）与号码保持字符串类型；禁止布尔值或数字字符串相互隐式转换；
- 所有浮点数值字段默认拒绝 `NaN`/`Inf`（评测报告除外，当预测正样本为 0 时允许精确报告 `NaN`）；
- 序列化统一通过 `model_validate_json` 与 `model_dump_json` 实现无损往返。

---

## 1. 作用域与核心标识符

| 标识符 | 数据类型与格式 | 核心业务语义 |
| :--- | :--- | :--- |
| `match_id` | `StrictStr`, 非空 | 持久比赛作用域。所有球员、图库、事件、修订及审计均以此为最高隔离边界。 |
| `batch_id` | `StrictStr`, 非空 | 单次交接/导入批次标识。仅作为交接回执，不作为球员库分区依据。 |
| `clip_id` | `StrictStr`, 非空 | 比赛内视频片段标识。相同时间戳在不同片段内不代表物理同时。 |
| `track_id` | `StrictStr`, 非空 | 片段内轨迹标识。外部整数由适配器标准化为字符串并保留原始映射。 |
| `observation_id` | `StrictStr`, 非空 | 比赛内逐帧机器观测标识。承载原生边框坐标、实际时间点与来源版本。 |
| `sample_id` | `StrictStr`, 非空 | 比赛内稳定特征样本标识。保留原图抠图资产与提取特征映射。 |
| `segment_id` | `StrictStr`, 非空 | 绑定片段、轨迹与不可变时间/帧半开区间 $[start\_s, end\_s)$ 的身份段标识。 |
| `player_id` | `StrictStr`, `pl_` 前缀 | 比赛内持久匿名球员标识。由 UUID5 基于 match 与 anchor 段哈希生成，不依赖批次或遍历顺序。 |
| `feature_space_id` | `StrictStr`, 非空 | 特征空间唯一标识。由模型骨干、特征布局、预处理、归一化与距离函数共同定义。 |
| `event_id` | `StrictStr`, 非空 | 持久比赛技术事件标识。 |
| `revision` | `int`, 非负整数 | 比赛内版本状态计数器。仅发生有效业务变更（增量更新、人工审核）时原子递增。 |

---

## 2. 证据契约 (`app/contracts/evidence.py` & `common.py`)

### 2.1 EvidenceBatch (证据包契约)
统一交接顶层对象（schema_version: `"2"`）：
- `schema_version`: `StrictStr = "2"`
- `match_id`: `StrictStr`
- `batch_id`: `StrictStr`
- `producer`: `StrictStr`
- `sources`: `list[Provenance] = []`
- `assets`: `list[Asset] = []`
- `clips`: `list[Clip] = []`
- `identity_evidence`: `IdentityEvidence | None = None`
- `events`: `list[Event] = []`
- `warnings`: `list[str] = []`

### 2.2 TrackObservation (逐帧机器观测)
- `observation_id`: `StrictStr`
- `match_id`: `StrictStr`
- `clip_id`: `StrictStr`
- `track_id`: `StrictStr`
- `time_s`: `float`
- `frame_idx`: `int | None = None`
- `bbox_xyxy`: `list[float]`（长度为 4，满足 $x_1 < x_2$ 且 $y_1 < y_2$）
- `detection_score`: `float | None = None`（范围在 $[0.0, 1.0]$，不代表身份置信度）
- `source`: `Provenance`

### 2.3 Sample (特征样本)
- `sample_id`: `StrictStr`
- `match_id`: `StrictStr`
- `clip_id`: `StrictStr`
- `track_id`: `StrictStr`
- `time_s`: `float`
- `frame_idx`: `int | None = None`
- `segment_id`: `StrictStr | None = None`
- `image_asset_id`: `StrictStr | None = None`
- `tracking_quality`: `TrackingQuality | None = None`
- `appearance`: `AppearanceEvidence | None = None`
- `features`: `list[SampleFeature] = []`
- `source`: `Provenance`

---

## 3. 轨迹内一致性分段契约 (`app/contracts/segmentation.py`)

### 3.1 SegmentProposal (分段提案)
由 `track-consistency-v3` 纯算法计算产出并持久化于 `segmentation_proposals` 表：
- `segment`: `Segment`（派生有效身份段，绑定半开时间区间 $[start\_s, end\_s)$）
- `parent_segment_id`: `StrictStr`（被切分的原暂定段标识）
- `status`: `Literal["supported", "unknown"]`（一致性支持状态）
- `reason`: `StrictStr`（如 `local_continuous_appearance_window`, `viewpoint_transition_uncertainty`）
- `support_sample_ids`: `list[StrictStr] = []`
- `support_observation_ids`: `list[StrictStr] = []`
- `evidence_fingerprint`: `StrictStr`（绑定派生特征与观测上下文的防篡改指纹）
- `identity_verified`: `bool = False`（恒为 False，强调分段支持不等于身份确认）
- `can_match_existing`: `bool = True`（资格解耦：是否允许匹配已有图库）
- `can_create_new_identity`: `bool = True`（资格解耦：是否允许独立聚类建档，短段设为 False）

---

## 4. 距离校准档案契约 (`app/contracts/calibration.py`)

### 4.1 DistanceCalibrationProfile (距离校准档案)
用于在特征空间维度固化经验证的度量阈值与离散度边界（schema: `distance-calibration-profile-v1`）：
- `schema_version`: `StrictStr = "distance-calibration-profile-v1"`
- `feature_space_id`: `StrictStr`
- `status`: `Literal["calibrated", "uncalibrated"]`
- `distance_metric`: `StrictStr = "cosine"`
- `thresholds`: `DistanceThresholds`
  * `near_distance`: `float = 0.18`（外观近距离一致界限）
  * `far_distance`: `float = 0.45`（少数异类强反证一票否决界限）
  * `min_common_parts`: `int = 2`
  * `min_part_visibility`: `float = 0.2`
- `dispersion_report`: `DispersionReport | None = None`（正负例距离经验分布报告）
- `derived_fingerprint`: `StrictStr`

### 4.2 四态距离语义 (`DistanceRelation`)
- `NEAR` (`"near"`): $\text{dist} \le \text{near\_distance}$
- `FAR` (`"far"`): $\text{dist} \ge \text{far\_distance}$
- `AMBIGUOUS` (`"ambiguous"`): $\text{near\_distance} < \text{dist} < \text{far\_distance}$
- `UNCOMPARABLE` (`"uncomparable"`): 共同可见部位数 $< 2$ 或可见度 $< 0.2$

---

## 5. 多原型表征契约 (`app/identity/representation.py`)

### 5.1 AppearancePrototype (多原型结构)
废弃单均值模糊表示，保留局部原始细节与跨视角见证：
- `prototype_id`: `StrictStr`
- `sample_id`: `StrictStr`
- `segment_id`: `StrictStr`
- `clip_id`: `StrictStr`
- `track_id`: `StrictStr`
- `time_s`: `float`
- `time_bucket`: `int`（相对时间桶编号）
- `feature_space_id`: `StrictStr`
- `feature`: `np.ndarray`（加权池化特征向量，形状 $[D]$）
- `visibility`: `np.ndarray`（部位可见度向量，形状 $[P]$）
- `reliable_parts`: `list[int]`（可见度 $\ge 0.2$ 的部位索引列表）
- `quality_score`: `float`
- `risks`: `list[str] = []`
- `witness_sample_ids`: `list[str] = []`（跨片段独立见证样本列表）
- `source_component`: `StrictStr`（用于防复制去重的连通组件标识）

---

## 6. 身份、角色与图库契约 (`app/contracts/identity.py`)

### 6.1 PersonRole (角色枚举)
```python
class PersonRole(StrEnum):
    PLAYER = "player"  # 正式球员（计入技术统计）
    REFEREE = "referee"  # 裁判
    STAFF = "staff"  # 场边/工作人员
    UNKNOWN_ROLE = "unknown_role"  # 未确认角色（匿名档案，默认隔离）
```

### 6.2 Player (球员实体)
- `match_id`: `StrictStr`
- `player_id`: `StrictStr`（`pl_` 前缀）
- `created_revision`: `int`
- `role`: `PersonRole = PersonRole.UNKNOWN_ROLE`（默认隔离，迁移 010 表达式索引加速）
- `display_name`: `StrictStr | None = None`
- `jersey_number`: `StrictStr | None = None`
- `team_id`: `StrictStr | None = None`
- `representative_asset_id`: `StrictStr | None = None`

### 6.3 Assignment (身份归属)
- `match_id`: `StrictStr`
- `segment_id`: `StrictStr`
- `player_id`: `StrictStr | None = None`（未分配时为 None）
- `status`: `AssignmentStatus` (`"confirmed"`, `"proposed"`, `"rejected"`, `"unassigned"`)
- `method`: `AssignmentMethod` (`"seed"`, `"automatic"`, `"manual"`, `"unassigned"`)
- `revision`: `int`
- `evidence`: `AssignmentEvidence | None = None`
- `unassigned_reason`: `UnassignedReason | None = None`

### 6.4 Template (图库模板)
- `match_id`: `StrictStr`
- `template_id`: `StrictStr`
- `player_id`: `StrictStr`
- `feature_space_id`: `StrictStr`
- `sample_id`: `StrictStr`
- `segment_id`: `StrictStr`
- `admission`: `Literal["seed", "automatic_experimental", "manual"]`
- `active`: `bool = True`
- `review_status`: `TemplateReviewStatus` (`"unreviewed"`, `"approved"`, `"rejected"`)
- `prototypes`: `list[AppearancePrototype] = []`（关联多原型实例，映射迁移 009 `template_prototypes` 表）

---

## 7. 开放集合增量决策契约 (`app/identity/incremental.py`)

### 7.1 IncrementalSegmentDecision (增量分段三态决策)
- `segment_id`: `StrictStr`
- `clip_id`: `StrictStr`
- `track_id`: `StrictStr`
- `category`: `Literal["matched_existing", "created_new_identity", "insufficient_evidence"]`
- `player_id`: `StrictStr | None`
- `assignment`: `Assignment`
- `reasons`: `list[StrictStr]`
- `distance`: `float | None`
- `matched_existing_player_id`: `StrictStr | None`

### 7.2 IncrementalIdentityPlan (增量执行计划)
- `match_id`: `StrictStr`
- `feature_space_id`: `StrictStr`
- `decisions`: `list[IncrementalSegmentDecision]`
- `new_players`: `list[Player]`
- `new_templates`: `list[Template]`
- `deactivated_template_ids`: `list[StrictStr]`
- `discovery_report`: `DiscoveryReport`
- `player_map`: `dict[str, str]`
- `sample_sources`: `dict[str, dict]`
- `summary`: `dict`

---

## 8. 查询视图与人工复核契约 (`app/contracts/views.py` & `api.py`)

### 8.1 PlayerView 与 PlayerStatsView
- `PlayerView`:
  * `player`: `Player`
  * `confirmed_segment_ids`: `list[str]`
  * `templates`: `list[Template]`
  * `made_count`: `int`
  * `is_formal_player`: `bool`（当 `role == PersonRole.PLAYER` 时为 True）
  * `profile_type`: `str`（`"formal_player_profile"` 或 `"anonymous_identity_profile"`）
- `PlayerStatsView`（仅为正式球员生成）：
  * `match_id`: `StrictStr`
  * `player_id`: `StrictStr`
  * `role`: `PersonRole`
  * `display_name`: `StrictStr | None`
  * `jersey_number`: `StrictStr | None`
  * `confirmed_segments_count`: `int`
  * `approved_templates_count`: `int`
  * `official_made_count`: `int`
  * `is_formal_player`: `bool = True`

### 8.2 ReviewQueueItem 与建议操作
- `ReviewQueueItem`:
  * `kind`: `ReviewQueueItemKind` (`"unassigned_segment"`, `"conflicted_segment"`, `"unassociated_event"`, `"unreviewed_template"`, `"unresolved_risk"`)
  * `segment_id`: `StrictStr | None`
  * `clip_id`: `StrictStr | None`（时空定位）
  * `track_id`: `StrictStr | None`（时空定位）
  * `start_s`: `float | None`（时空定位）
  * `end_s`: `float | None`（时空定位）
  * `risk_kinds`: `list[StrictStr] = []`
  * `review_priority`: `Literal["critical", "high", "normal", "low"]`（针对 `duplicate_box` 等高风险段自动赋予 `"critical"`）
  * `candidates`: `list[Candidate] = []`
  * `thumbnail_asset_ids`: `list[StrictStr] = []`
  * `suggested_actions`: `list[ReviewActionSuggestion] = []`
  * `reasons`: `list[StrictStr] = []`
  * `details`: `dict = {}`
- `ReviewActionSuggestion`:
  * `action`: `Literal["create_player", "split_segment", "set_assignment", "review_risk"]`
  * `label`: `StrictStr`
  * `description`: `StrictStr`
  * `target_player_id`: `StrictStr | None`
  * `target_role`: `PersonRole | None`

### 8.3 SetPlayerRoleCommand
运营人员调整球员角色的轻量化命令：
- `player_id`: `StrictStr`
- `role`: `PersonRole`
- `reason`: `StrictStr = "operator_role_assignment"`
- `kind`: `Literal["set_player_role"] = "set_player_role"`

---

## 9. 独立评测与防火墙契约 (`app/evaluation/`)

### 9.1 Stage6BenchmarkMetrics (九大显式评估指标)
严格规定数据分母，严禁使用虚假 100% 掩盖无正类预测：
1. **正确覆盖 (Correct Coverage)**：
   - `correct_coverage_samples`: `float | None`
   - `correct_coverage_observations`: `float | None`
   - `correct_coverage_duration`: `float | None`
   - `correct_coverage_denominator_note`: `"有限标注范围（目标范围118样本/10段），全场未标注样本不计入真值分母"`
2. **混人率 (Merge Error Rate)**：`merge_error_rate: float = 0.0`
3. **同人拆档率 (Split Error Rate)**：`split_error_rate: float = 0.0`
4. **已归人纯度 (Assigned Purity)**：`assigned_purity: float | None`
5. **跨片段对偶召回与精度 (Cross-Clip Pairwise)**：
   - `cross_clip_pairwise_precision`: `float | None`（无正类预测时返回 NaN）
   - `cross_clip_pairwise_recall`: `float | None`
   - `cross_clip_pairwise_f1`: `float | None`
   - `cross_clip_pairwise_tp`: `int`
   - `cross_clip_pairwise_fp`: `int`
   - `cross_clip_pairwise_fn`: `int`
6. **延期率 (Deferred Rate)**：`deferred_rate: float = 0.0`
7. **增量重复建档率 (Incremental Duplicate Identity Rate)**：`incremental_duplicate_rate: float | None = 0.0`
8. **批次顺序稳定性 (Batch Order Stability)**：
   - `order_stability_ari`: `float | None`
   - `order_stability_agreement`: `float | None`
9. **非球员污染率 (Non-Player Role Intrusion Rate)**：
   - `non_player_pollution_rate: float | None = 0.0`
   - `non_player_pollution_status`: `Literal["zero_pollution", "uncomputable_missing_ground_truth"]`

### 9.2 防火墙断言函数 (`app/evaluation/guard.py`)
- `assert_no_ground_truth_in_evidence(sources: list) -> None`：拦截含有真值路径的来源。
- `assert_inference_isolation(caller_context: str = "") -> None`：拦截评分函数调用栈内嵌套推理。
- `assert_frozen_before_scoring(run_dir: str | Path) -> dict[str, str]`：打分前强制比对 `artifact-hashes.json`，存在任何缺失或改动抛出 `PrematureScoringError`。

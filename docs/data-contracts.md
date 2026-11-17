# 数据契约规范 (Data Contracts Specification)

本文档是系统内各模块交互的**强制性不可变契约 (Frozen Contracts)**。所有核心数据结构均在 `app/contracts/` 下以强类型（Python `dataclasses` 或 `pydantic`）显式定义。

> [!IMPORTANT]
> **契约保护纪律**：任何代码编写与算法重构，**必须在既有契约定义的字段与类型下运作**。严禁在未经架构评审的情况下随意新增、删除或重命名这些核心字段。

---

## 1. 输入层契约 (Input Domain)

### `TrackObservation` (单帧人物观测)
```python
@dataclass(frozen=True)
class TrackObservation:
    video_id: str  # 视频片段标识 (如 "3", "20", "68")
    local_track_id: int  # 上游局部跟踪器分配的轨迹 ID (如 6, 8, 13)
    frame_idx: int  # 0-indexed 真实视频帧号
    timestamp_s: float  # 片段内相对秒数 (以 60fps 计算)
    bbox_xyxy: Tuple[float, float, float, float]  # [x1, y1, x2, y2] 像素绝对坐标
    confidence: float  # 目标检测置信度 (0.0 ~ 1.0)
    crop_path: Optional[str] = None  # 对应 samples/*.png 的相对路径
```

### `TrackletSegment` (时序连续轨迹段)
```python
@dataclass(frozen=True)
class TrackletSegment:
    segment_id: str  # 全局唯一段 ID (格式: "{video_id}_{local_track_id}_{start_frame}")
    video_id: str
    local_track_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    observation_count: int
    observations: Tuple[TrackObservation, ...]
```

---

## 2. 特征层契约 (Feature Domain)

### `JerseyNumberPrediction` (球衣号码识别结果)
```python
@dataclass(frozen=True)
class JerseyNumberPrediction:
    number: Optional[int]  # 识别出的球衣号码 (如 27, 10, 15)；无法辨识为 None
    confidence: float  # 置信度 (0.0 ~ 1.0)
    legible: bool  # 画面是否满足可读性要求 (非模糊/背光/背向)
    votes: Dict[int, float]  # 轨迹内各候选号码的时序投票累积得分
    raw_text: Optional[str] = None  # OCR 模型原始解码字符串
```

### `AppearanceEmbedding` (外观深度表征)
```python
@dataclass(frozen=True)
class AppearanceEmbedding:
    vector: np.ndarray  # 形状严格限制为 (512,)，dtype=float32，L2 归一化
    quality_score: float  # 样本综合质量分 (清晰度、完整度)
    model_name: str  # 提取模型版本标识 (如 "sports-reid-v1")
```

### `DualStreamFeature` (双流联合特征)
```python
@dataclass(frozen=True)
class DualStreamFeature:
    segment_id: str  # 绑定的轨迹段 ID
    jersey: Optional[JerseyNumberPrediction]  # 号码强证据 (可为空)
    appearance: AppearanceEmbedding  # 外观向量
```

---

## 3. 约束与图库契约 (Identity Domain)

### `CannotLinkConstraint` (时空硬排他关系)
```python
@dataclass(frozen=True)
class CannotLinkConstraint:
    segment_a: str  # 轨迹段 A
    segment_b: str  # 轨迹段 B
    reason: str  # 排他原因: "same_frame_spatial_separation"
```

### `PlayerPrototype` (球员图库多视角原型)
```python
@dataclass(frozen=True)
class PlayerPrototype:
    prototype_id: str  # 原型唯一 ID (UUID)
    player_id: str  # 绑定的全局球员 ID
    view_type: str  # 视角分类: "frontal" | "dorsal_with_number" | "lateral" | "generic"
    feature: DualStreamFeature  # 对应的双流特征
    weight: float  # 该原型在聚类比对中的信任权重 (0.0 ~ 1.0)
    created_at_revision: int  # 准入时的数据库版本号
```

### `PlayerProfile` (持久化球员档案)
```python
@dataclass(frozen=True)
class PlayerProfile:
    player_id: str  # 全局唯一持久球员 ID (如 "player_001")
    display_name: str  # 呈现名称 (如 "27号球员" 或 "未命名球员_01")
    confirmed_jersey_number: Optional[int]  # 确立的球衣号码
    prototypes: Tuple[PlayerPrototype, ...]  # 活跃多原型列表
    status: str  # "active" | "merged" | "archived"
    created_at_revision: int
```

---

## 4. 增量决策与事务契约 (Workflow Domain)

### `IncrementalDecision` (单段归属决策)
```python
class DecisionOutcome(str, Enum):
    MATCHED_EXISTING = "matched_existing"  # 匹配到已有球员
    CREATED_NEW = "created_new_identity"  # 证据充分，新建球员档案
    INSUFFICIENT = "insufficient_evidence"  # 证据不足/歧义，保持待定延期


@dataclass(frozen=True)
class IncrementalDecision:
    segment_id: str
    outcome: DecisionOutcome
    target_player_id: Optional[str]  # MATCHED 时的目标球员 ID，其余为 None
    best_distance: float  # 与最佳原型的特征距离
    competitive_margin: float  # 与第二名的距离差 (Margin)
    evidence: Dict[str, Any]  # 决策详细机器证据 (支持度、投票得分、排他证明)
```

### `IncrementalPlan` (全批次执行计划)
```python
@dataclass(frozen=True)
class IncrementalPlan:
    batch_id: str
    from_revision: int  # 计算所基于的图库快照版本
    to_revision: int  # 提交后期望推进到的版本 (from_revision + 1)
    decisions: Tuple[IncrementalDecision, ...]
    new_players: Tuple[PlayerProfile, ...]
    updated_prototypes: Tuple[PlayerPrototype, ...]
    deferred_segments: Tuple[str, ...]  # 本轮被延期的轨迹段 ID 清单
```

---

## 5. 评测契约 (Evaluation Domain)

### `EvaluationBenchmarkResult` (黄金基准评分)
```python
@dataclass(frozen=True)
class EvaluationBenchmarkResult:
    target_player: str  # 评测目标代号 (如 "27号")
    total_denominator: int  # 固定分母 (70)
    correct_samples: int  # 正确归入目标身份的样本数
    coverage_rate: float  # correct_samples / total_denominator
    observed_exposure_seconds: float  # 真实有效观测秒数
    cross_identity_pollution: Dict[str, int]  # 异人混入账本 (如 {"10号": 0, "15号": 0})
    physical_conflicts: int  # 空间分离冲突数 (必须为 0)
    passed: bool  # 是否达到基准线 (>= 57 且 0 混入)
```

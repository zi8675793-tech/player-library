# SQLite 存储层

本层保存和还原已有契约事实；不执行身份匹配、生成正式 player_id、处理人工命令或解析事件归属。工作流、事件关联与统计已实现，UI 尚未实现。`connect(database_path(settings))` 才创建数据库和父目录；加载配置及只读 `validate` 不会打开数据库。数据库名集中为 `player-library.sqlite3`，位于 `settings.paths.runtime_root`。每个连接启用外键及锁等待，打开时按顺序执行校验过的增量迁移（001 至 010）；迁移整体失败回滚，不删除数据库或重编号球员。

## 模型与关系

比赛和当前 revision 为作用域；批次是交接回执，不是球员分区或幂等主键。媒体、片段、轨迹、身份段、样本、特征空间及文件索引、球员、归属、模板、结构化风险和人工记录、事件主体及引用各有表和可查询键；较复杂的契约内容以 JSON 保留。轨迹键包含比赛、片段和轨迹 ID。事件原始事实不保存正式 player_id；它可以先于目标片段、轨迹或身份段到达。正式归属和模板则需完整同场关系。人工归属和模板更改须显式 `replace=True`，历史版本仍可读取。

带 `Provenance` 的记录按比赛和 `producer`、`record_type`、`source_record_id`、`evidence_version` 识别。内部指纹基于业务内容，排除来源对象中的哈希和定位，故行号变化不会造成业务冲突；来源定位单独保留。同源同版本不同业务内容或同实体 ID 不兼容时抛 `StorageConflict`；新版本必须显式 `supersedes` 当前来源版本，历史证据保留，旧版本重交不会回退当前值。`batch_id` 相同也可有多个不同回执。文件观测的路径和整文件哈希另行留存，不代表单条业务事实变更。缺省或空列表不删除历史；重复业务证据仅记录回执和新定位，不应推进 revision。

## 事务、修订和检查

调用方必须用 `transaction(connection)` 包住写入；仓储操作不自行提交。事务默认 `BEGIN IMMEDIATE`，异常自动回滚。调用方在同一事务中组合证据、人工决定及特征值，发生业务变更时调用 `advance_revision(match_id, expected_revision, actor=..., reason=..., changes=...)`：数据库原子比较当前版本，过期抛 `RevisionConflict` 并使整个事务回滚；成功只前进一次并写审计。仅交接回执不调用它。调用方须按其业务用例判断实际变更及提供可追踪的 `changes`。

算法输出 `Assignment.revision` 是计算所据图库版本；`put_assignment(assignment, committed_revision=expected_revision + 1)` 明确记录提交修订，不能把算法版本当新修订。模板写入同理。仓储的 `check_segment_overlaps(match_id)` 和 `check_alias_cycles(match_id)` 返回检查结果，**不会自动阻止提交**；未来工作流必须在事务中检查并决定如何处理。正式归属、模板和人工记录的仓储写入会调用 `app.storage.relations` 中的关系检查；外键不能代替所有业务范围检查。

## 特征与一致读取

输入边界负责安全解析文件，storage 不解析 NPZ 或样例路径。文件引用本身不等于值已就绪：可在事务中先存入 `FeatureFile` 和 `FeatureSpace`，再调用 `save_feature_value(connection, match_id, feature_file, feature_space, row_index, inline_feature)` 保存已解析行。无值时快照映射中不出现该行，不补零。直接内联于样本的值随批次保存，但必须先提供相应空间合同；已用于模板的样本不能直接升级来源版本，需由未来复核流程显式处理。两者的 embeddings 和 visibility 都有带 dtype、shape、字节序及空间绑定的 BLOB，使用小端 float32 而非 pickle；形状来自特征空间，数值按 float32 表示精度往返。

`read_identity_snapshot(connection, match_id)` 在同一 SQLite 读事务取出 revision、`GallerySnapshot`、文件特征映射和人工约束/模板审核；没有模板的未合并球员也在 `gallery.players` 中。调用时已有事务则由调用方控制。快照还原事实，不决定模板资格、匹配或归人。

## 最小调用

下面示例只登记一条原始事件，并展示关闭后重新读取；`settings` 为 `load_settings(...)` 的结果。不要把本示例当作正式导入工作流。

```python
from app.contracts.common import Provenance
from app.contracts.events import Event, PointEventTime
from app.contracts.evidence import EvidenceBatch
from app.storage.database import connect, database_path, transaction
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot

path = database_path(settings)
batch = EvidenceBatch(
    match_id="match-1",
    batch_id="handoff-1",
    producer="example",
    events=[
        Event(
            match_id="match-1",
            event_id="event-1",
            source_event_key="source-1",
            clip_id="clip-not-yet-received",
            event_type="shot",
            time=PointEventTime(time_s=1),
            source=Provenance(
                producer="example",
                record_type="event",
                source_record_id="source-1",
                evidence_version="1",
            ),
        )
    ],
)
conn = connect(path)
try:
    repo = Repository(conn)
    with transaction(conn):
        repo.ensure_match(batch.match_id)
        expected = repo.revision(batch.match_id)  # 实际工作流应使用计算计划时读取的版本
        changed = repo.store_batch(batch)
        if changed:
            repo.advance_revision(
                batch.match_id,
                expected,
                actor="operator",
                reason="ingest",
                changes={"events": ["event-1"]},
            )
finally:
    conn.close()

conn = connect(path)
try:
    stored_events = Repository(conn).events("match-1")
    state = read_identity_snapshot(conn, "match-1")
    gallery = state.gallery  # GallerySnapshot
    feature_values = state.feature_values  # {(feature_file_id, row_index): InlineFeature}
    constraints = state.constraints
finally:
    conn.close()
```

对于已准备的待比较批次，可将 `gallery` 和 `feature_values` 传给 [身份核心调用](identity-core.md)；人工约束和模板审核分别作为相关算法调用的上下文。实际用例应在事务中组合全部变更、范围检查、revision 与审计，并在过期时重新读取和计算。


## 修订状态与旧库升级（迁移 004/005）

004 为人工审核增加有效选择标记，并增加 `revision_states`。每次业务修订原子保存有效
身份状态清单（身份段、球员别名、归属、模板、约束、审核和派生计划），不复制原始证据表及特征 BLOB。
恢复追加新的归属、模板和审核版本，历史行保留。退回拆段前时，撤销子段的当前替换边，
历史替换关系仍保存在状态清单与段审核历史中；恢复拆段版本可重新启用这些关系。
恢复不撤回原始输入，不删除后来新建的 player_id。所有写入与共享校验均处于同一事务。
状态清单会随修订和业务规模增长，尚未完成整场容量基准。

打开 v3 库时原地增量升级，不改变比赛 revision、player_id 或已有业务记录数量。
迁移为升级时的实际当前状态建立带 legacy 标记的检查点，不宣称修正旧实现已经写坏的状态。
没有执行过破坏性旧恢复的历史，可由归属/模板版本、段审核历史与命令记录重建；
旧派生计划没有完整历史，恢复该类旧版本时清空计划缓存，查询使用当前事实回退显示。
若旧恢复已删除审核，缺少可靠状态清单的旧目标会明确拒绝恢复并整体回滚，需人工依据原证据
重新审核。升级前已丢失的审核无法推断补造；升级后新修订支持恢复到曾经恢复过的修订。

005 给清洁审核补充被批准样本的 `source_key`。仅有一个来源版本的旧样本可无歧义补绑；
已有多版本而无法证明审核对象的旧批准保持历史记录，但不在算法输入中作为当前批准生效。
新审核自动绑定存储中的当前来源版本；同来源同版本内容不可改写，版本升级不继承审核。
段级人工归属不因此撤销。模板使用过的样本仍按已有规则要求显式复核才能升级。


## 逐帧机器观测持久化（迁移 006）

迁移 006 引入 `track_observations` 表，完整记录机器提取的每帧目标检测框坐标（`bbox_xyxy`）、实际时间（`time_s`）、帧索引（`frame_idx`）与检测置信度（`detection_score`）。结合 `(match_id, clip_id, track_id, time_s)` 复合索引，为轨迹内一致性检查（`track-consistency-v3`）和时空排他碰撞检测（`SpatioTemporalConstraintEngine`）提供不可变的原生逐帧物理证据支持。


## 自动身份支持（迁移 007）

`identity_discovery_reports` 按 `(match_id, report_id)` 保存不可变 DiscoveryReport JSON、
组到 player ID 的映射及提交 revision；不存入人工 review 表。仓储 `storage.discovery` 只保存和读取，
不能自行生成身份或提交事务。相同 ID 内容或映射发生变化时拒绝覆盖。
`workflows.discover` 在一次事务中写玩家、proposed 自动归属、自动实验模板、支持报告并推进自动审计。
已有人工身份、约束或审核的库不接受冷启动覆盖；相同报告重放不增加 revision 或更换 player_id。

自动模板沿用现有 templates JSON 契约，admission=automatic_experimental，
automatic_support_id 指向报告，初始 unreviewed/approval=null，不能伪装成人工批准。
人工批准须保留独立人工审核语义，显式拒绝仍可审计。正式匹配资格继续排除自动实验模板。
查询用 workflows.read.get_discovery_reports；导出只是数据库的派生副本。
旧库原地增加迁移，不删除既有记录或修订历史。算法和调用见 [identity-discovery.md](identity-discovery.md)。


## 机器分段提案与资格保持（迁移 008）

迁移 008（`app/storage/migrations/008_segmentation_proposals.sql`）为数据库引入机器分段提案与段级资格持久化表 `segmentation_proposals`，用于在存储层固化轨迹一致性算法产生的段划分及支持状态。

### 表结构与索引

表结构由比赛作用域和分段 ID 构成复合主键：
- `match_id TEXT NOT NULL REFERENCES matches(match_id)`：关联所属比赛；
- `segment_id TEXT NOT NULL`：分段唯一标识；
- `parent_segment_id TEXT NOT NULL`：被切分的原暂定段或父段标识；
- `status TEXT NOT NULL CHECK (status IN ('supported', 'unknown'))`：分段一致性资格状态，严格限制为支持或未知；
- `evidence_fingerprint TEXT`：对应的派生证据指纹，绑定产生该提案的观测与特征上下文；
- `json TEXT NOT NULL CHECK (json_valid(json))`：结构化 `SegmentProposal` 契约对象的完整 JSON 序列化（自动囊括 `can_match_existing` 与 `can_create_new_identity` 资格解耦字段）；
- 主键：`PRIMARY KEY (match_id, segment_id)`；
- 状态索引：`CREATE INDEX IF NOT EXISTS idx_segmentation_proposals_status ON segmentation_proposals(match_id, status)`，加速按比赛及资格状态的过滤检索。

### 增量分批更新中的资格保持机制

在开放集合增量分批更新（`workflows.incremental`）中，证据包按批次陆续到达。若随批次提交了机器分段报告（`SegmentationReport`），工作流在同一事务中通过 `Repository.put_segmentation_proposal` 将每个分段提案及其派生指纹入库；存在同名段时原子更新。

快照读取 `read_identity_snapshot` 会统一将当前比赛所有已存提案与指纹装配至 `IdentitySnapshot.proposals`。该机制的核心价值在于：
1. **跨批次资格固化**：分段一致性判定（特别是局部支持窗）是在具备原始逐帧观测上下文时计算得出的；一旦持久化，后续批次无需重复计算历史片段的切分规则；
2. **避免支持状态丢失**：跨批次增量关联时，历史分段的 `supported` / `unknown` 资格保持不变，避免因跨批次重新划分或历史局部观测上下文脱离而丢失局部支持状态，确保增量聚类与图库维护建立在稳定可信的段资格基石之上。


## 多原型模板证据与关系索引（迁移 009）

迁移 009（`app/storage/migrations/009_multi_prototype_templates.sql`）为数据库引入 `template_prototypes` 关系表，废弃历史遗留的单均值中心模糊向量，彻底实现多原型（`AppearancePrototype`）证据结构与关系持久化。

### 表结构与索引

```sql
CREATE TABLE IF NOT EXISTS template_prototypes (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    template_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    prototype_id TEXT NOT NULL,
    time_bucket INTEGER,
    source_component TEXT,
    reliable_parts_json TEXT,
    quality_score REAL,
    json TEXT,
    PRIMARY KEY (match_id, template_id, prototype_id)
);
CREATE INDEX IF NOT EXISTS idx_template_prototypes_player
    ON template_prototypes(match_id, player_id);
CREATE INDEX IF NOT EXISTS idx_template_prototypes_sample
    ON template_prototypes(match_id, sample_id);
```

### 核心设计特性

1. **复合主键隔离**：以 `(match_id, template_id, prototype_id)` 形成复合主键，确保一个模板内可以容纳多个来自不同时间桶与视角的代表样本（局部 medoid），并保证原型级幂等。
2. **溯源与可靠部位**：通过 `time_bucket`、`source_component`、`reliable_parts_json` 与 `quality_score` 记录原型的物理与特征属性，支持在跨视角比对时仅使用双方均可靠的共同部位进行精确 KPR 计算。
3. **高效双向索引**：建立 `idx_template_prototypes_player` 与 `idx_template_prototypes_sample`，支持毫秒级查询特定球员的所有活跃外观原型，或逆向查询某样本关联的所有模板。


## 球员角色解耦与表达式索引（迁移 010）

迁移 010（`app/storage/migrations/010_player_role.sql`）为数据库引入非侵入式 JSON 表达式索引，实现球员、裁判、工作人员与未确认角色的彻底解耦。

### 索引定义

```sql
CREATE INDEX IF NOT EXISTS idx_players_role ON players(match_id, json_extract(json, '$.role'));
```

### 核心设计特性

1. **表物理结构零变更**：维持 `players` 表历史经典的 4 列物理结构（`match_id`, `player_id`, `created_revision`, `json`）完全不变，100% 保持历史仓储代码、SQL 查询与单元测试的二进制级兼容。
2. **角色域过滤加速**：通过 SQLite 原生 `json_extract(json, '$.role')` 表达式索引，使 `PlayerQuery(formal_players_only=True)` 与 `get_formal_player_stats` 能够直接走 B-Tree 索引跳过非球员档案（如 `unknown_role` 匿名档案、裁判与工作人员），杜绝全表扫描。
3. **平滑升级与默认隔离**：历史旧记录的 JSON 未显式包含 role 字段时，Pydantic 契约反序列化默认解析为 `PersonRole.UNKNOWN_ROLE`（匿名档案），被下游正式球员统计与事件排行榜自动过滤，只有通过人工复核或特定策略确认为 `PersonRole.PLAYER` 后才计入正式榜单。

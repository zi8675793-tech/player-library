# 比赛球员库与事件管理

接收上游轨迹、人物特征和事件证据，建立并持续维护比赛内球员身份，提供按球员检索的事件时间线。
同一场比赛可以有任意多批片段、逐步出现的球员、晚到的事件和人工修正。

当前状态：**已完成 2026-11-09 ~ 2026-11-10 身份修复计划第一轮（阶段0、阶段1、阶段2）：**
- **固定分母基准**：以全场 10 个片段机器证据（158 条原始轨迹、1,759 个特征样本、75,465 条逐帧观测、1,346.35 秒有效时长）为不可变固定分母，彻底纠正了旧实验中因浮动分母（排除所有未知轨迹）产生的虚假 97% 覆盖率。
- **阶段0（统一覆盖漏斗与基线修正）**：确立 7 级覆盖漏斗；修复时间严格半开区间、F1 边界公式与持久身份对齐（`var/identity-repair/stage-0/handoff.txt`）。
- **阶段1（空间绑定距离校准机制）**：建立 `DistanceCalibrationProfile` 契约（schema: `distance-calibration-profile-v1`）与四态距离语义（near/far/ambiguous/uncomparable）；通过 1,113 对候选正例与 890 对物理负例实证离散度诊断证实二者在 [0.1652, 0.5757] 严重重叠，如实保持 `status="uncalibrated"`（保持 near=0.18, far=0.45, clustering_threshold=0.28），不伪造校准成功（`var/identity-repair/stage-1/handoff.txt`）。
- **阶段2（局部连续外观一致性 track-consistency-v3 与资格解耦）**：引入连续外观支持窗、相邻近距离连续、零 far 强反证一票否决、中心样本（medoid anchor）支撑与视角过渡区；解耦建档资格与匹配资格（`can_match_existing` 与 `can_create_new_identity`）；修复漏斗第4级跨片比较段统计与短段增量匹配门槛。实测在固定分母下将局部一致段样本覆盖率从 12.6% 恢复至 25.7% (452 样本 / 137 段)、观测覆盖率从 6.7% 恢复至 14.0%、有效时长从 6.5% 恢复至 13.3%，同时跨片段独立评分保持 0 误合并（FP=0，精确率 100%）。
- **2026-11-10 专项加固（4 项关键漏洞闭环）**：完成 R1（校准档案强一致性断言，杜绝参数静默旁路）、R2（校准证据严格准入过滤与 `min_part_visibility` 硬门槛）、R3（增量多批分段报告全局累积合并，漏斗恢复严格单调递减：452 -> 166 -> 141）、R4（部分复制证据非共享组件 $\ge 3$、双向覆盖率 $\ge 0.8$ 与多见证约束硬激活）。第一轮修复在此圆满收官（`var/identity-repair/stage-2/handoff.txt`）。
- **测试套件状态**：通过全部 701 项业务与回归测试（700 passed, 1 skipped in ~11.8s），`uv run ruff check .` 与 `uv run ruff format --check .` 138 个文件 0 错误。当前待执行阶段为第二轮阶段3（多原型特征表示与必要接口适配，尚未启动）。

## 先看业务，再看目录

1. [流水线](docs/pipeline.md)：你的输入、初始化、增量更新、事件关联与输出。
2. [架构](docs/architecture.md)：每一步放在哪个模块。
3. [数据契约](docs/data-contracts.md)：比赛、批次、身份段、特征空间和事件。
4. [实施计划](docs/implementation-plan.md)：逐个闭环开发和验收。
5. [AI 协作约定](AGENTS.md)：后续修改要保持的边界。
6. [身份算法与归人决策](docs/identity-policy.md)：v1 距离、评分、拒识和人工维护规则。
7. [身份核心调用](docs/identity-core.md)：纯算法输入、结果、人工上下文和校准边界。
8. [无种子自动身份发现](docs/identity-discovery.md)：证据约束、自动模板、未知、存储与实验边界。
9. [开放集合增量身份更新](docs/incremental-identity.md)：分批次到达演进、固定快照隔离与三态判定。
10. [机器轨迹内一致性检查](docs/track-consistency.md)：逐帧观测分析、局部支持窗与疑似切换判定。
11. [SQLite 存储层](docs/storage.md)：增量迁移、不可变报告、特征 BLOB 与原子事务。

## 输入边界

本模块接收上游计算结果，不要求重新运行追踪、ReID 或动作模型。
规范接入使用标准 EvidenceBatch；不同同伴文件格式由 adapters 转换。
人物证据和事件可以分别提交，事件类别不限定为进球。

身份作用域是 match_id；一次交接使用 batch_id。换批次沿用该比赛已有球员 ID。
跨比赛永久身份合并不作为默认能力。

## 样例与运行目录

`../test/` 是第一份验收样例，不是产品范围或内部数据格式。
机器接入配置在 examples/machine-baseline.toml。
历史人工选择样例说明在 [examples/sample-test.md](examples/sample-test.md)，配置在 examples/sample-test.toml；
该演练不是自动识别验收，工作区根目录 annotations.csv 与 test/target 均不得成为基线输入。
`../repository-review/` 是只读上游审查快照，不是运行依赖。
所有数据库、预览、日志和导出写 var/，不进入 Git。

## 环境

使用 Python 3.11 和 uv，依赖以 uv.lock 为准：

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

config.example.toml 定义通用配置，examples/sample-test.toml 是样例配置；两者均可由
`app.settings.load_settings` 读取。所有相对路径以配置文件所在目录解析，配置加载不会创建目录、
打开数据库或修改输入，也不要求输入目录已经存在。

在项目根目录可运行：

```bash
uv run python - <<'PY'
from app.contracts.evidence import EvidenceBatch
from app.settings import load_settings

settings = load_settings("config.example.toml")
print(settings.paths.input_root)

batch = EvidenceBatch(match_id=settings.match.match_id, batch_id="handoff-1", producer="example")
assert EvidenceBatch.model_validate_json(batch.model_dump_json()) == batch
PY
```

契约使用 Pydantic 2；字典/JSON 读取入口为 `model_validate` / `model_validate_json`，
输出为 `model_dump(mode="json")` / `model_dump_json`，字段及边界见数据契约。
上述示例只读取配置并构造模型，不执行导入。

在项目根目录运行无标注轨迹内一致性检查与分段：

```bash
uv run python -m app.presentation.cli baseline --config examples/machine-baseline.toml --run-id consistency-new-run --segment-tracks
```

每次使用不存在的新 run-id。产物写入 `var/machine-baseline/<run-id>/`，包括新数据库、
标准证据包、预测、诊断、配置、输入清单、摘要及 artifact-hashes.json 冻结哈希。该入口自动发现机器片段和轨迹，
只读取 track/identity/video 的明确文件白名单，不读取标注、target 或历史审核图库。
`--segment-tracks` 在导入前检查逐帧框、部位特征、可见性、质量与时间覆盖，提出有依据的范围，
将证据不足或边界不确定的范围保留未知；不会仅凭特征突变认定换人。重复框只作候选证据，
时间重叠不推断为不同人。省略该选项仍运行上一阶段原轨迹暂定段基线。
本阶段冻结结果与证据不足比例见 [机器一致性运行记录](examples/machine-consistency.md)。
分段运行另导出 raw-evidence.json、segmentation.json，与 evidence.json、predictions.json 形成前后对照；
报告保留逐对距离、排除原因、风险、提案来源与不确定区间。`supported` 不等于身份确认，
仅开启分段时，空库策略仍将身份归属保留待定，不新建球员或批准模板。
同证据换 batch 重交会验证幂等；预测冻结后才允许未来独立评分读取答案。
针对多轮实验与参数调优，支持提供 `--fixed-segmentation <path>` 选项：复用已有的 `segmentation.json` 报告（及其同目录下的 `evidence.json`），避免在多轮实验中重复执行耗时的轨迹分段。
该仅分段模式不进行身份发现；无种子发现请使用下述开关。开放集合增量发现和真实身份精度校准仍待完成。
使用方法、冻结产物和自动建库阶段接口见 [机器输入与分段记录](examples/machine-baseline.md)。

从空身份库运行无种子自动身份发现（推荐 constrained_manifold 模式）：

```bash
uv run python -m app.presentation.cli baseline --config examples/machine-discovery.toml --run-id discovery-new-run --discover-identities
```

该命令调用 `constrained_manifold`（时空受约束层次流形聚类）：
1. 提取每条轨迹的多部位可见度加权均值代表特征向量（`representation.py`）；
2. 构造全局物理时空互斥图，严格保证同一镜头同一时间重叠的多框绝对互斥（`constraints.py`）；
3. 执行带排他硬约束的层次凝聚聚类（完全连接，门槛 0.28），平滑传递正面 ↔ 侧面 ↔ 背面视角变化，彻底消除原有自杀式桥接处决；
4. 历史实验运行 `manifold-delivery-001` 下识别 21 位核心跨视频球员，全场零物理时空违规（注：该历史报告中 97.22% 活跃段归人率系基于排除全部未知段的浮动分母得出，阶段0已纠正为固定分母漏斗标准，见 `docs/implementation-plan.md` 及 stage-0/handoff.txt）。
5. 自动模板 active/unreviewed，准入标记 automatic_experimental，有跨片段见证和绑定。
完整规则及复现边界见 [自动身份发现](docs/identity-discovery.md) 与 [自动发现交付记录](examples/machine-discovery.md)。

运行开放集合增量分批更新：

```bash
uv run python -m app.presentation.cli baseline --config examples/machine-discovery.toml --run-id incremental-run --incremental-batches --batch-size 3 --calibration-profile var/identity-repair/stage-2/forward/calibration-profile.json
```

参数意义与增量更新特性：
- `--incremental-batches`：开启开放集合增量分批更新模式，将全部视频片段分割为序列批次逐步接入；
- `--batch-size <N>`：设置每个增量批次的片段数量（默认值为 3）；
- `--reverse-batches`：按相反顺序依次接入批次，用于评估时序到达顺序对未分配池累积与桥接阻断的影响，检验跨批增量归人的顺序鲁棒性；
- `--calibration-profile <path>`：传入冻结距离校准档案（如 `distance-calibration-profile-v1`），建立特征空间硬断言，禁止静默旁路；
- **增量更新特性**：
  * **快照隔离与单事务推进**：算法在单一事务内基于当前 Revision $R$ 的固定图库快照比对，批次内部决策隔离；提交时原子写入新球员、归属与模板并推进到 Revision $R+1$。
  * **三态判定与短段匹配**：`matched_existing`（与已有球员模板达成强共现对照或近距离匹配，直接关联已有 `player_id`，杜绝重复建档；短段受 `min_match_buckets=2` 支持）、`created_new_identity`（未分配池跨片段聚类，多段完全图且与已有球员充分分离，生成持久稳定匿名 `player_id`；未匹配短段受 `can_create_new_identity=False` 拦截）、`insufficient_evidence`（单次出现或证据不足段保持待定，无匹配或距离远绝不单独建立新人）。
  * **漏斗累积单调性保证**：汇总跨批分段报告构建全局漏斗，确保漏斗各级严格单调递减（452 -> 166 -> 141）。
  * **重放幂等验证**：每批导入后自动提交重放副本，严格核验 revision 不变且 `changed=False`。
完整设计与评测见 [增量身份更新](docs/incremental-identity.md) 与 [增量分批实验记录](examples/machine-incremental.md)。

只读核验机器输入可使用：

```bash
uv run python -m app.presentation.cli validate --config examples/machine-baseline.toml --batch-id machine-check
```

Python 入口 `load_machine_input(settings, batch_id="machine-import")` 返回现有 EvidenceBatch
及以 `(feature_file_id, row_index)` 为键的 InlineFeature 映射；省略 clip_ids 自动发现片段。
特征行、观测、时间、图片、文件哈希与来源位置均保留，后续分段仍提交同一标准契约。

历史人工样例只读核验（当前真实 test 缺少 annotations.csv，此命令不能用于无答案验收）：

```bash
uv run python -m app.presentation.cli validate --config examples/sample-test.toml --batch-id sample-check
```

可重复提供 `--clip-id` 选择片段，例如 `--clip-id 20 --clip-id 68`。
validate 仅调用配置和适配器，输出数量及警告，不创建数据库、球员或运行目录。
文件缺失、哈希不符、数组与索引错行、非法范围和越界路径会使核验失败并返回非零退出码。
Python 调用为 `load_sample_test(settings, batch_id="sample-check", clip_ids=None)`，返回现有
`EvidenceBatch`；完整规则见 [样例说明](examples/sample-test.md)。

身份分支按已确定的 v1 策略实施：人工确认种子，KPR 多样本/多模板推荐，证据不足保留待复核。
人工纯算法与工作流提交已实现；自动发现已增加独立实验路径，真实身份校准仍未完成。
人工模式的 auto_assign_enabled 继续关闭，不用打开此开关代替无种子发现。
可运行 `uv run python -m examples.identity_preview` 查看无文件输入的合成预览，输出 JSON，
不会建库或创建球员。实际输入调用见 [身份核心调用](docs/identity-core.md)。

SQLite 存储层的事务、来源版本、特征值和快照调用见 [docs/storage.md](docs/storage.md)。

工作流业务闭环 v1 已提供可直接运行的完整流程示例（初始化建档 → 导入新批次 → 人工确认 → 查询个人事件）：

```bash
uv run python -m examples.workflow_v1
```

支持传入 `--database var/workflow-v1.sqlite3` 保存数据库文件并使用 SQLite 客户端检查。该闭环串联了真实事务、版本推进、人工复核、模板管理与事件解析，并通过了覆盖 11 类集成场景的端到端自动化测试。


最新基线结果与复现：最新基线与交接报告见 `var/identity-repair/stage-2/handoff.txt` 与 [增量身份更新](docs/incremental-identity.md)；历史早期实验交付记录见 [无标注自动发现交付记录](examples/machine-discovery.md)（注：早期交付报告中 97.22% 活跃段归人率系基于排除全部未知段的浮动分母得出，当前已由阶段0统一为固定分母 7 级漏斗标准，历史 11-06 基线 1 组 2 段存档见文档内）。

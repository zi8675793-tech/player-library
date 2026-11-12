# 项目协作约定与维护指南 (Player Library Agent Guide & Maintenance Handbook)

## 1. 核心验收目标与交付状态（全案结项）

- **模块定位**：通用的比赛内球员档案建立、开放集合增量身份更新、多原型图库维护与事件关联模块。支持未知球员人数、变长片段与开放分批到达。
- **真值隔离铁律**：推理输入严禁包含人工身份标注（`annotations.csv`）、人工目标切分（`test/target/`）或人工种子/模板。真实验收必须在全新 SQLite 状态上运行并冻结预测、配置和输入清单后，才允许评分代码打开真值进行独立评分。
- **固定分母基准**：全场统一使用 10 个比赛片段机器输入的不可变原子事实（**158 条原始轨迹、1,759 个特征样本、75,465 条逐帧观测、1,346.35 秒时长**）作为评估分母，彻底废除浮动分母。
- **全阶段最终状态（7 阶段全案高质量交付）**：
  * **阶段 0（验收基线修正与 7 级漏斗）**：纠正浮动分母与 NaN 掩盖；确立全场固定分母基准；修复时间半开区间与 F1 边界（`var/identity-repair/stage-0/handoff.txt`）。
  * **阶段 1（空间绑定距离校准与离散度诊断）**：实现 `DistanceCalibrationProfile` 契约（schema: `distance-calibration-profile-v1`）；实证诊断余弦距离集中与正负例重叠区间，确立 near=0.18 与 far=0.45 双门槛（`var/identity-repair/stage-1/handoff.txt`）。
  * **阶段 2（轨迹内局部视角一致性 v3 与增量资格解耦）**：实现第三代局部连续外观支持窗与过渡区检测；解耦建档资格与匹配资格；闭环 R1~R4 四项漏洞（空间档案断言、准入过滤、增量单调性、部分复制防御）（`var/identity-repair/stage-2/handoff.txt`）。
  * **阶段 3（多原型身份表示与模板证据结构）**：废除单均值模糊表示，引入 `AppearancePrototype` 局部 medoids 多原型体系与相容视角匹配；增设迁移 `009_multi_prototype_templates.sql` 关系表；历史首次实现全场冷启动跨片段正向召回且精度达 100%（TP=83, FP=0）（`var/identity-repair/stage-3/handoff.txt`）。
  * **阶段 4（候选生成与四道严苛集团验收解耦）**：彻底解耦层次流形聚类候选生成与四道准入关卡（簇内强反证消除、Rival Margin 竞争边距、Tarjan 割边剪枝、跨至少 2 片段独立见证）；消除单桥接串联混人（`var/identity-repair/stage-4/handoff.txt`）。
  * **阶段 5（统一冷启动与开放集合增量更新）**：落实 Revision R 统一 7 步闭环更新流水线；消除“先遍历者先获利”时序漏洞；接入 `SpatioTemporalConstraintEngine` 时空物理硬互斥；输出三态互斥决策（`matched_existing`, `created_new_identity`, `insufficient_evidence`）；顺序稳定性 ARI 达到 0.9236（`var/identity-repair/stage-5/handoff.txt`）。
  * **工程微调（27与15号球员工程微调）**：闭环边缘特征对抗样本调优，保持全场零误合并与零拆分（`var/identity-repair/finetune-27-15/handoff.txt`）。
  * **阶段 6（角色边界解耦、复核契约完善与独立终验）**：实现四层架构单向解耦；引入 `PersonRole` 枚举与匿名档案隔离（迁移 010 表达式索引）；建立绝对防火墙；升级 Review Queue 数据契约（提供时空坐标、视频 3 重叠框 critical 优先级与一键操作建议）；全场固定分母基准实测跨片对偶精度 100%（TP=68 Cold, TP=148 Forward, FP=0），跨球员误合并与拆分错误严格全部为 0，时空物理碰撞为 0，顺序稳定性 ARI 达到 0.9264 / 0.9823 / 0.8830；全部 740 项测试全数通过（739 passed, 1 skipped）（`var/identity-repair/stage-6/handoff.txt`）。
- **当前工程交付版本**：`v1.0.0-identity-repair`。

---

## 2. 模块边界与单向依赖原则

```text
app/
  contracts/     # 内部统一数据语言（Pydantic 2 Strict 模型，无外部状态依赖）
  adapters/      # 上游输入边界（machine 只读白名单读取，sample_test 历史样例适配）
  identity/      # 核心身份算法（纯计算、无状态、不访问数据库、不读文件、不生成正式 player_id）
  events/        # 比赛技术事件分支（校验、幂等登记与当前主体延迟关联）
  workflows/     # 业务编排用例与事务边界（ingest, initialize, update, review, read, incremental）
  storage/       # SQLite 仓储、特征 BLOB 编解码与 001~010 增量迁移脚本
  evaluation/    # 独立评测子系统、固定分母基准与防火墙防泄漏断言
  presentation/  # CLI 命令行入口与视图接口
```

- **单向依赖规则**：
  `presentation` → `workflows` → `adapters` / `identity` / `events` / `storage`
  `adapters` / `identity` / `events` / `storage` → `contracts`
- 核心算法层（`app/identity/`）绝不反向导入 `app.storage` 或 `app.workflows`；
- 持久化仓储层（`app/storage/`）不自行执行 `commit` 或 `rollback`，所有写事务由 `workflows` 统一管理；
- 推理全链路绝不导入评测打分模块。

---

## 3. 开发者维护指南 (Maintenance Handbook)

### 3.1 环境初始化与代码质量门禁
本项目使用 Python 3.11 与 `uv` 管理依赖：
```bash
# 同步环境（严格根据 uv.lock 锁定依赖）
uv sync --locked

# 代码风格与静态类型检查（零错误门槛）
uv run ruff check .
uv run ruff format --check .

# 执行全量单元测试与集成测试套件（740 项用例）
uv run pytest
```

### 3.2 数据库模式演进与迁移规范 (Migrations 001 ~ 010)
- **迁移原子性**：所有数据库结构变更必须以递增编号脚本置于 `app/storage/migrations/`（如 `011_new_feature.sql`）；
- **向后兼容性**：
  * 禁止破坏性删除既有数据表或列；
  * `players` 表核心物理 4 列（`match_id`, `player_id`, `created_revision`, `json`）必须保持稳定，角色与扩展字段通过 `json_extract(json, '$.field')` 建立表达式索引（参考 `010_player_role.sql`）；
  * 关系变更使用独立关系表（如 `009_multi_prototype_templates.sql` 的 `template_prototypes` 表）。
- **事务包裹**：迁移执行函数 `app.storage.database.apply_migrations` 会在单个事务中按顺序执行未应用的脚本，任何失败立即触发原子回滚。

### 3.3 核心算法调优与门槛策略指南
1. **特征距离与四态语义 (`app/identity/calibration.py`)**：
   - 近距一致门槛：`near_distance = 0.18`（严格用于局部连续支持）；
   - 强反证一票否决门槛：`far_distance = 0.45`（只要出现一对可比较且距离 $\ge 0.45$ 的视角对，立即否决合并或匹配，杜绝以均值洗白）；
   - 歧义带：$[0.18, 0.45]$ 内必须结合时空排他与竞争边距共同决策。
2. **时空硬排他网底线 (`app/identity/constraints.py`)**：
   - 严禁关闭 `SpatioTemporalConstraintEngine`；
   - 同片段同时间半开区间 $[start\_s, end\_s)$ 存在分离框的轨迹绝对不可合并（Must-Not-Link），全场物理碰撞必须保持为 0。
3. **集团验收四道关卡 (`app/identity/discovery.py`)**：
   - 关卡 1：簇内强反证消除（内部无 far $\ge 0.45$ 对）；
   - 关卡 2：Rival Margin 竞争对抗过滤（$\Delta \ge 0.04$）；
   - 关卡 3：Tarjan 割边剪枝（剪除单桥接脆弱连接，消除 A-B-C 串联混人）；
   - 关卡 4：跨至少 2 片段独立双见证校验。
4. **门槛分流与三态决策 (`app/identity/incremental.py`)**：
   - 已有匹配：允许 2 桶短段以 $\tau=0.28$ 稳健匹配；
   - 新建建档：必须通过四道关卡、$\ge 3$ 桶、且跨 $\ge 2$ 片段；
   - 证据不足者一律输出 `insufficient_evidence` 并写入 `segmentation_proposals` 作为延期提案，后续批次联合重估。

### 3.4 角色管理与人工复核队列运营指南
1. **角色域分流机制**：
   - 系统内置 `PersonRole`：`PLAYER`、`REFEREE`、`STAFF`、`UNKNOWN_ROLE`；
   - 自动发现的实体默认为 `UNKNOWN_ROLE`（匿名档案），被 `get_formal_player_stats` 自动过滤；
   - 运营人员核实后，通过 `SetPlayerRoleCommand` 将其变更为 `PLAYER`，正式计入比赛榜单。
2. **Review Queue 关键风险项处置**：
   - 视频 3（clip_03）等重叠框高风险段（`duplicate_box`）自动标记为 `review_priority: "critical"`；
   - 每个条目携带 `suggested_actions`（`create_player`, `split_segment`, `set_assignment`, `review_risk`），运营人员可通过 REST API 执行一键拆分或关联。

### 3.5 防火墙断言与独立评测运行指南
- **防泄漏哨兵**：推理全流程受 `assert_no_ground_truth_in_evidence` 守护，严禁传入任何真实答案；
- **哈希冻结验证**：运行独立打分脚本前，必须确保产物哈希已生成且完好：
  ```bash
  # 1. 运行无标注基线并冻结产物
  uv run python -m app.presentation.cli baseline \
    --config examples/machine-discovery.toml \
    --run-id my-eval-run \
    --discover-identities

  # 2. 独立评分阶段（验证产物哈希一致性后方可读取答案）
  # 评分代码内部必须显式调用 assert_frozen_before_scoring(run_dir)
  ```
- **显式 9 项指标**：所有评测输出遵循 `Stage6BenchmarkMetrics`，严禁在无正类时虚报 100% 精度（无正类严格输出 `NaN`）。

### 3.6 下游技术事件与呈现集成
- **事件解耦导入**：技术事件（进球、犯规）调用 `workflows.update.ingest_events` 独立导入，不受视频轨迹是否已解析身份阻塞；
- **动态主体投影**：调用 `workflows.read.get_events` 时，系统依据当前 Revision 的最新身份映射动态解析主体，身份修订或恢复时事件主体自动实时对齐。

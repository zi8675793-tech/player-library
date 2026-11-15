# 项目当前状态 (Single Source of Truth)

更新日期：2026-11-15。本页面是当前研发进度、最新实测跑分基准与下一任务执行的唯一汇总入口。

---

## 1. 业务目标与评测靶标

- **核心评测目标**：在 `test/` 提供的 10 个篮球比赛进球片段中，27 号球员在固定的 70 个测试样本中，**至少 57 个正确归入同一个比赛内全局身份档案**（覆盖率 $\ge 80\%$）。
- **硬性约束与纯度指标**：
  1. **零异人混入**：同队同色球衣的已知球员（如绿衣 10 号、15 号）绝不允许混入 27 号档案；
  2. **零物理冲突**：同一机位、同一时间帧内绝对不允许出现两个归属于 27 号的人体；
  3. **真实观测时长**：累计有效曝光观测时长需达到并保持高覆盖。

---

## 2. 方案演进与历史评测结论

| 方案版本 | 算法内核 | 27号命中样本 / 70 | 真实核验结论 | 最终决策 |
| :--- | :--- | :---: | :--- | :--- |
| **v1.0 旧版 KPR** | 局部支持窗 + 关键部位特征 (KPR) | **25 / 70** (35.7%) | 能识别部分镜头，但缺乏跨片段强锚点，易拆档 | 保留为历史工程底座与对照基准 |
| **v2.0 OpenVINO OMZ 原版** | 零售安防 0013 检测 + 0277 ReID | **0 / 70** (真实0%)<br>*(账面 4/70)* | **严重失效**。复核证实系统选中的 4 个命中全都是同队的绿衣 10 号；大面积丢框（58/70 检测不对应）；混入 15 号 | **正式废弃下线**。安防模型无法胜任篮球激烈对抗和同色球衣场景 |
| **v3.0 体育原生双流增量架构 (T1+T2实测)** | **球衣号码识别 (OCR) + ResNet18 + 时空 Cannot-Link + 多原型图库** | **24 / 70** (34.3%) | **0 混入 (PASS)、0 物理冲突 (1,020 条排他边)**；OCR 置信稀释及 ImageNet 外观在同色球衣下分离度不足 (+0.0076)，待 T3/T4 接入专有 ReID 与置信校准 | **已完成 T1+T2 实测，推进 T3** |
| **v3.0 T3 冻结运行 / T4 独立终验** | 封板 EasyOCR + ImageNet ResNet18 + 默认 T2 + Revision 5；仅评测既有预测 | **0 / 70** (0.0%) | 58 个样本保持待定、12 个来自未处理片段；曝光 0.0 秒。混入/冲突/碎档均为 0，但为空图库计数，**未认证纯度** | **T4 评分完成；算法验收 FAIL** |
| **v3.0 目标终态 (T1~T4)** | **优化号码流 + 体育专有 ReID + 事务存储 + 独立终验** | **目标 $\ge 57 / 70$** ($\ge 81.4\%$) | 专为篮球场景设计，号码强证据 + 时空排他 + 高分离度外观 | **当前主线推进中** |

---

## 3. 新架构四阶段实施状态

新方案按四阶段（T1 $\rightarrow$ T2 $\rightarrow$ T3 $\rightarrow$ T4）串行推进：

| 阶段 | 核心任务 | 交付产物 | 当前状态 | 下一步动作 |
| :--- | :--- | :--- | :---: | :--- |
| **T1: 双流特征适配器** | 针对 `test/track/*/samples/*.png` 建立球衣号码提取 (OCR) 与外观特征抽取，输出标准契约特征 | `app/features/`、84项单元测试全绿、真实模型运行产物与交接 | **已验证** | 交付进入 T2：使用 [T2 任务卡](tasks/02-constraints-and-matching.md) |
| **T2: 时空排他与多原型匹配** | 构建 Cannot-Link 物理排他矩阵；实现多原型相容比对与增量三态决策 (`matched_existing` / `created_new` / `insufficient`)；重构最大相容团挖掘算法修复冷启动图聚类死锁 | `app/identity/matching.py`、时空引擎测试 (1,020 条真实排他边)、100 项纯逻辑用例全绿 (含 Cannot-Link 相容团专项) | **已验证（Bug 1 已修复）** | 交付进入 T3：使用 [T3 任务卡](tasks/03-workflow-and-storage.md) |
| **T3: 流水线与 SQLite 版本化存储** | 将双流特征与匹配引擎接入 Revision R 事务流，支持分批到达、幂等重放与中断续跑 | `update_dual_stream`、迁移 013、20 项端到端测试、真实 5 批数据库及冻结产物 | **已验证（工程验收）** | 进入 T4 独立评测；真实默认配置仍全部待定，不能据此宣称覆盖率达标 |
| **T4: 独立黄金评测与基准达标** | 在评测防火墙下独立运行，核验 70 分母达标情况，生成详细失败归因与纯度账本 | `var/benchmarks/run-20261114T131430-642308/`；25 项专项测试通过 | **评测已交付；算法未达标** | 另开推理质量任务后生成新运行再独立复评；禁止用本次答案回流调参 |

---

## 4. 关键指标基准一览 (Benchmark Baseline)

* **历史最高安全基准**：旧版 KPR 25/70 (8.0333秒观测曝光，0 物理冲突)。
* **T1+T2 历史实测记录**：24/70 (34.3%)，0 异人混入 27 号，0 物理冲突，全量 10 个视频建立 1,020 条 Cannot-Link 强排他边。本次 T4 未重新认证该历史记录。
* **最新 T4 独立基准**：**0/70 (0.0%)，曝光 0.0 秒，算法 FAIL**；已知异人混入/物理冲突/多余碎档均为 0，目标身份不存在，不能认证纯度。
* **达标红线**：$\ge 57 / 70$ (81.4%) 且已知异人混入（10号、15号）数量必须为 0。
* **评测纪律**：任何阶段的预测必须先完全冻结（产物哈希锁定），评测工具才能打开 `test/annotations.csv` 评分，杜绝任何信息泄露。


## 5. T3 工程验收与交接（2026-11-14）

- **正式双流入口**：`app.workflows.incremental.update_dual_stream(connection, *, match_id, batch_id, request_id, expected_revision, segments, extractor, matcher=None)`；输入为 T1 `TrackletSegment`，调用真实 `DualStreamExtractor` 与冻结 T2 `IncrementalMatcher`。原 `update_incremental` 保持旧版 `EvidenceBatch` 兼容，旧入口的两阶段提交不属于本次原子性保证。
- **原子边界**：整个快照读取、计划、历史重估、持久化、revision 与审计处于同一个 `BEGIN IMMEDIATE`。当前批新段只使用 Revision R 图库；历史待定段及机器暂定归属可在同一事务的更新后图库中留一复核，排除自身所有原型/归属，重分配时撤回旧活跃原型但保留历史记录。所有结果只提交一次 R+1。
- **存储**：新增迁移 013，未修改 001–012。`dual_stream_segments` 保存不可变几何/特征、原始切片字节指纹和精确小端 float32 BLOB；`prototypes`、`dual_stream_profiles`、`dual_stream_decisions` 保存原型及版本历史；业务投影进入现有 `players`/`assignments`，审计进入 `workflow_operations`。T2 UUID 原样保留，旧版 `PlayerId` 投影通过 `legacy_player_id` 增加 `pl_` 前缀。历史读取支持 `read_dual_stream_snapshot(..., revision=N)`。
- **幂等与恢复**：输入指纹覆盖完整几何和实际 PNG 字节，忽略批次名、列表顺序及文件定位；同请求或同内容重放 `changed=False`，返回当前库版本并保留原始提交回执。同请求异内容拒绝；抽取中原图变化、过期新请求或任意提交异常整体回滚。人工确认、拆段、合并、恢复等导致旧版投影与双流事实不一致时拒绝自动覆盖，需后续显式协调。
- **测试**：`uv run pytest tests/test_incremental_pipeline_e2e.py` **20 passed**；覆盖 `20 → 6 → 43 → 50 → 68`、原型 0→2→3→4→5、历史待定收敛、真实几何留一法改归、原型满额后的号码排他、历史版本恢复读取、并发写锁及故障回滚。全套 `uv run pytest tests/ -q` **1003 passed, 1 skipped**（旧版人工样例缺失，与自动身份验收无关）；专项旧版工作流/T1/T2/存储回归 **231 passed**。
- **代码检查**：`uv run ruff check app/workflows/ app/storage/ tests/test_incremental_pipeline_e2e.py` 及对应格式检查通过。全项目 `ruff check .` 仍有历史 `eval_t1_t2_real.py` 的 12 项既有问题，文件不在 T3 修改范围，未改动。
- **真实输入验证**：仅读取 `test/video/{20,6,43,50,68}.mp4` 及各自 `test/track/<clip>/tracks.jsonl`/`video_meta.json`；由原始帧按图像质量生成 240 张准确帧号切片，保留 80 条轨迹、38,292 次观测。离线缓存 EasyOCR + ImageNet ResNet18、默认 T2 参数，Revision 连续 **1→2→3→4→5**；原请求及异回执同内容重放均无变更；SQLite integrity/foreign-key 检查通过，已归属段物理冲突 0。
- **真实结果边界**：上述 80 段均为 `insufficient_evidence`，球员/活跃原型/已归属段均为 0；默认 T2 相容簇门控没有形成可准入身份。多原型扩展与改归由可控图像后端端到端用例验证，不能把真实空图库的零冲突当作纯度达标。T3 不调参、不改封板 T1/T2、不读取人工答案、不复算或重新认证历史 24/70；覆盖率、纯度及体育 ReID/号码校准留给后续专项与 T4 防火墙验证。
- **不可变交接目录**：`var/runs/run-20261114T060714-089939/`，包含真实数据库、分批回执/快照、带多原型和改归的测试数据库、输入/代码/产物指纹、复现脚本、测试日志及 `handoff.txt`。备份目录：`/Users/ahs/Documents/agc/player-library-t3-run-20261114T060714-089939/`。下阶段按 `docs/tasks/04-golden-evaluation.md`继续，先核对冻结产物，再独立评分，严禁回流人工答案。

## 6. T4 独立黄金终验结论（2026-11-14）

**签署状态：评测工具与账本已交付，算法达标验收 FAIL。** 本次仅新增独立评测实现与专项测试、更新本页跑分结论；未修改封板推理模块、契约、上游输入或 T3 运行。

| 指标 | 实测 | 合格线 | 验收结论 |
| :--- | :--- | :--- | :--- |
| 27号正确样本覆盖 | **0 / 70（0.0%）** | ≥57 / 70 | **FAIL** |
| 已知异人混入样本 | 10号 0；15号 0 | 0 | 空图库计数，不构成纯度认证 |
| 同源同帧物理冲突 | 0 | 0 | 空图库计数 |
| 多余碎档 | 0（目标档案本身也不存在） | ≤2 | 空图库计数 |
| 真实有效曝光 | **0.0 秒** | ≥50.0 秒 | **FAIL** |

- **固定样本范围**：冻结所有 10 个机器片段，评分读取完整人工选择范围，共 118 个样本（27号 70、15号 48）。27号样本实际分布于 6 个片段：20/3/43/50/6 各12、68为10；不能误解为每个片段均有27号。12个未处理样本全部来自片段3，其余58个有可靠几何对应但保持待定，全部留在70分母。标注没有10号样本，不能凭“10号混入0”宣称具备10号辨别能力。
- **防火墙**：阶段A使用活动文件读取屏障，拒绝人工标签、目标目录及其符号链接别名；核验 T3 全部1,087项哈希及数据库/导出一致性，冻结全量几何后退出。阶段B以另一个进程校验外部固定 SHA256，再首次读取答案，完成后再次核验全部冻结指纹。正式几何指纹为 `1d9f88d81ede10f5be2cefecc322c8d6a94f32e5569ab92cfb56029ad9e6f103`。
- **归因边界**：70条失败账本逐项保留冻结几何、决策与特征证据。58条为安全门禁下无可准入档案，12条为未处理；已处理目标段均无接受号码，其中15个样本所在段有低置信OCR票但未被接受。冻结记录没有逐对簇拒绝日志，不把“同色外观倒挂”或“Cannot-Link阻断”伪造为已证明因果。
- **验证**：`uv run pytest tests/test_evaluation_harness.py` **25 passed**；评测指标及历史几何回归 **31 passed**（合计56项）。`uv run ruff check app/evaluation/ tests/test_evaluation_harness.py` 与对应格式检查通过；全局 `uv run ruff check .` 仍为历史 `eval_t1_t2_real.py` 的12项既有问题，未越界修改。
- **正式交付**：`var/benchmarks/run-20261114T131430-642308/`，包含必需四项输出、完整样本/帧/纯度账本、指纹、验证日志、源码快照及 `handoff.txt`。备份：`/Users/ahs/Documents/agc/player-library-t4-run-20261114T131430-642308/`。首次标签开启记录保留于 `run-20261114T131220-989655/`；随后仅将相同字节的既有几何工具纳入独立T4模块以消除未提交依赖，预测、几何账本、阈值与评分未改变，详见正式交接。

## 7. T2 增量匹配引擎死锁修复与最大相容团重构（2026-11-15）

- **问题定位与死锁机理 (Bug 1)**：
  - 原 `app/identity/matching.py` 中的 `_discover()` 使用 BFS 搜索极大连通分支（Connected Component）；
  - 在篮球比赛同队同色球衣场景下，跨视频传递关系将多名球员的轨迹串联为一个巨大分支；
  - 但同一视频内共现轨迹存在 Cannot-Link 强排他（距离为 $+\infty$），导致分支内两两互斥，后续 `complete = all(neighbors[sid] >= component - {sid})` 完全团判定数学上恒为 `False`；
  - 判定失败后执行 `unseen -= component` 将整个分支几十条轨迹全部一次性丢弃，导致在无人工种子的冷启动场景下，新球员创建数恒为 0。
- **算法重构：带 Cannot-Link 互斥约束的最大相容团（Maximal Compatible Clique）启发式提取**：
  1. **相容图构建**：节点间连边充要条件为 `_pair_distance(queries[a], queries[b], engine) <= distance_threshold + _EPS`（严格满足 Cannot-Link 排他、球衣号码一致性与质量基线）；
  2. **贪心种子与相容成团**：以未分配节点中最高质量节点为 Seed，按 `(跨视频优先, 距种子距离升序, 质量得分降序, 字典序)` 确定性排序，贪心扩展与团内所有成员均相互相容的候选节点，形成完全相容团（Clique）；
  3. **独立跨视频见证门槛**：团内成员必须包含至少 2 个不同的 `video_id`（`len(videos) >= 2`）；
  4. **避免级联误杀**：成团成功移出团成员；未满足跨视频门槛仅将当前 Seed 移出保持待定延期（`insufficient_evidence`），不再级联抹杀整个连通分支。
- **验证结论**：
  - `tests/test_constraints_and_matching.py` 从 99 项增至 **100 passed**（新增 `test_maximal_compatible_clique_extracts_separate_players_with_cannot_link` 专项测试）；
  - 全链路增量 E2E 测试 `tests/test_incremental_pipeline_e2e.py` **20 passed**；
  - 代码规范 `uv run ruff check app/identity/` 与格式检查 100% 通过；
  - 契约严格遵循 `docs/data-contracts.md`，返回类型保持 `list[PlayerProfile]`。

## 8. T2 增量匹配引擎单视频冷启动建档受阻与物理号码双轨准入修复（Bug 2，2026-11-15）

- **问题定位与死锁机理 (Bug 2)**：
  - 原 `app/identity/matching.py` 中 `_discover()` 存在一刀切的硬编码门禁 `if len(videos) < 2: continue`；
  - 引发冷启动首批必然全空死锁：在 `update_dual_stream` 增量流水线中，视频分批串行送入，Batch 1（如单视频 `clip-20`）因 `len(videos) == 1` 数学上 100% 无法建立新档案，首批永远为空图库；
  - 引发物理证据被无视：当轨迹已具备清晰确凿的球衣号码（如 27、15 等）时，在物理常识上已具备确定性独立身份，却依然被强行要求跨视频外观见证而全量打入待定池。
- **双轨准入与绝对排他安全实现**：
  1. **轨道 A（强物理号码直通）**：团内有明确确认的球衣号码（`number is not None`）时，允许 `len(videos) >= 1`，要求种子轨迹质量 `quality_score >= self.min_quality` 且号码在当前图库及本批新建档案中未被占用；决策证据标注为 `"confirmed_jersey_number_seed"`；
  2. **轨道 B（无号码纯外观聚类）**：团内未检出球衣号码（`number is None`）时，严格保留原有跨视频独立见证原则（`len(videos) >= 2`），防止单视频内的低质外观碎片误建碎片化虚拟身份；
  3. **时空 Cannot-Link 绝对物理安全**：双轨均严格校验团内所有同视频成员在时间帧上绝对不重叠（`engine.pair_relation == "clear"`）；
  4. **单段团边界异常防御**：对单成员团支持距离计算引入 `default=0.0` 防御，避免 `max()` 空序列异常。
- **验证结论**：
  - `tests/test_constraints_and_matching.py` 增至 **101 passed**（包含无号单视频碎片待定保留测试与单视频有号建档专项测试）；
  - 全链路增量 E2E 测试 `tests/test_incremental_pipeline_e2e.py` 增至 **21 passed**（新增单视频有号冷启动首批建档端到端专项测试）；
  - 代码规范 `uv run ruff check app/identity/` 与格式检查 100% 通过。

## 9. T1 球衣号码提取器时序置信度被无号码帧稀释归零修复（Bug 3，2026-11-15）

- **问题定位与稀释机理 (Bug 3)**：
  - 原 `app/features/jersey_number.py` 中 `JerseyNumberExtractor.extract_track` 计算轨迹置信度时，公式为 `confidence = votes[best] / usable_frames`；
  - 在篮球激烈运动场景下，球员频繁转身弯腰，球衣号码（胸前/后背）通常仅在 2~4 个特定关键帧中清晰可见；
  - 将关键帧检出票数直接除以全轨迹总质量通过帧数（如 12 帧），导致高置信度多帧命中（如 $0.95 \times 2 = 1.90$）被稀释至 $1.90 / 12 = 0.158 \ll 0.85$；
  - 导致真实测试集 11 条目标轨迹的球衣号码 100% 被一票否决为 `None`，使下游强物理证据流与号码排他机制彻底瘫痪。
- **算法重构与多重安全门禁**：
  1. **检出关键帧平均置信度**：将主候选号码置信度分母由全轨迹帧数 `usable_frames` 修正为该号码被实际支持检测的帧数：
     $$\text{confidence} = \frac{\text{votes}[\text{best}]}{\text{supports}[\text{best}]}$$
     背身无号帧不再稀释正面或背面高质检出；
  2. **次优候选竞争边距**：次优候选（runner）同样计算其检出平均置信度（若无则为 0.0），严格保留竞争边距检查：`confidence - runner >= self.min_vote_margin`，防止多号码混淆；
  3. **防单帧噪点与质量门禁**：严格保留 `supports[best] >= self.min_support_frames`（默认至少 2 帧独立见证才允许确认）以及 `confidence >= self.min_track_confidence`（默认 $\ge 0.85$）。
- **验证结论**：
  - `tests/test_dual_stream_features.py` 从 84 项增至 **86 passed**（校准背身帧不稀释测试，新增多候选竞争边距测试）；
  - 全量测试套件 `uv run pytest tests/ -q` **1034 passed, 1 skipped**；
  - 增量流水线与回归测试 `tests/test_incremental_pipeline_e2e.py` **21 passed**；
  - 代码规范 `uv run ruff check app/features/` 与格式检查 100% 通过；
  - **真实切片复核通过**：针对 `test/track/20/samples/` 和 `test/track/6/samples/` 真实切片复核，高置信度 15 号（Clip 20 T00002 conf=0.980）、23 号（Clip 20 T00001 conf=0.973）、28 号（Clip 6 T00001 conf=0.917）、10 号（Clip 20 T00005 conf=0.983，Clip 6 T00008 conf=0.995）全部准确识别，彻底终结全量退化为 `None` 的致命缺陷。

## 10. T2 增量匹配引擎球衣号码正向加权与相容外观加权折减修复（Bug 4，2026-11-15）

- **问题定位与强先验架空机理 (Bug 4)**：
  - 原 `app/identity/matching.py` 中的 `_pair_distance` 仅实现了号码不符时的一票否决（`a != b -> math.inf`）；
  - 当号码完全相符时（`a == b`），代码原封不动返回原始余弦距离 `_cosine(left, right)`，没有任何加权或折减；
  - 同样在 `_player_score` 中，即时候选段号码与目标球员确认号码一致，若两者外观因姿态、背光或光照波动达到 0.28（略高于阈值 0.25），依然被判定为待定（`insufficient_evidence`）拒之门外；
  - 强物理证据退化为“只能用于否决，不能用于肯定”的单向死规矩，系统总体架构白纸黑字规定的“若号码完全相同且外观颜色空间高度相容，赋予极高匹配加权（一票通过强先验）”被彻底架空。
- **正向加权算法与安全防御体系**：
  1. **`_pair_distance` 相容号码加权折减**：
     当两段轨迹均具备确凿号码且相同（`a is not None and b is not None and a == b`）时，计算原始外观余弦距离 `raw_dist`；
     仅当处于相容外观区间（`raw_dist <= 0.45`）时激活正向折减：
     $$\text{dist} = \text{raw\_dist} \times 0.5$$
     若原始外观严重偏离（`raw_dist > 0.45`，如不同队服颜色或噪点），不予折减，保持原距离防冒进。
  2. **`_player_score` 多原型图库对齐与侧身视角融合**：
     若候选段号码与球员档案号码相符（`player.confirmed_jersey_number == _jersey(query)`），即使比对的历史原型因侧身或背面角度未检出号码（`_jersey(proto) is None`），在相容外观区间（`raw_dist <= 0.45`）同样享受 $0.5\times$ 距离折减，确保侧身/背身高质原型与新带号轨迹平稳融合；
  3. **物理绝对安全守住底线**：
     时空 Cannot-Link（同一视频同一帧共现）无论号码是否相同，均强制阻断为 `math.inf`；极差外观（`raw_dist = 1.0`）无法被折减穿透；候选段无号码时平滑降级至纯外观比对，不享受折减。
- **验证结论**：
  - `tests/test_constraints_and_matching.py` 从 101 项增至 **105 passed**（新增相容号码加权折减、档案下侧身无号原型加权融合、外观偏离防穿透、冷启动带号成团 4 项专项测试）；
  - 全量测试套件 `uv run pytest tests/ -q` **1038 passed, 1 skipped**；
  - 增量流水线与回归测试 `tests/test_incremental_pipeline_e2e.py` **21 passed**；
  - 代码规范 `uv run ruff check app/identity/` 与格式检查 100% 通过；
  - **逻辑验证确认**：两段号码均为 27、原始余弦距离为 0.28（超标 0.25）的轨迹，在新逻辑下折减为 0.14，稳健归入 27 号球员档案。


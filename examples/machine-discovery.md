# 无标注自动身份发现：历史实验与交付记录

> [!NOTE] 历史演进与评测基准说明
> 本文档记录项目早期探索阶段（2026-11-06 ~ 2026-11-09）的实验运行产物与交付记录。
> 早期报告中出现的“97.22% 活跃段覆盖率”系基于排除全部未知段的浮动分母得出；项目已在 2026-11-09 身份修复计划（阶段0~阶段2）中全面纠正为以全场固定分母（10 个比赛片段机器输入、158 条原始轨迹、1,759 个特征样本、75,465 条逐帧观测、1,346.35 秒时长）为基准的统一 7 级覆盖漏斗。
> 最新正式基线指标与修复产物请参见 `var/identity-repair/stage-2/handoff.txt`、[增量身份更新规范](../docs/incremental-identity.md) 与 [实施计划](../docs/implementation-plan.md)。

## 1. 增量流形分批更新实验（2026-11-09 早期探索运行）

支持比赛视频片段分批到达时的持续增量演进，在全新空库状态下执行端到端分批增量发现与更新。
所有结果状态标记为 `automatic/proposed/unvalidated`，覆盖率不代表真实身份识别准确率。

### 1.1 三次全新独立运行指标对比

| 运行目录 (Run ID) | 批次大小 (Batch Size) | 批次数量 (Batches) | 到达顺序 | 最终 Revision | 发现球员数 (Players) | 活跃模板数 (Templates) | 分配段数 (Assigned) | 待定段数 (Unassigned) | 活跃段覆盖率 | 支持段覆盖率 | 物理同屏违例 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `incremental-manifold-forward-v1` | 3 | 4 | 正序 (Forward) | 4 | 7 | 16 | 23 (3 已有 + 20 新建) | 840 | 2.67% (23/863) | 45.10% (23/51) | **0** |
| `incremental-manifold-reverse-v1` | 3 | 4 | 倒序 (Reverse) | 4 | 5 | 10 | 19 (4 已有 + 15 新建) | 844 | 2.20% (19/863) | 37.25% (19/51) | **0** |
| `incremental-manifold-bs2-v1` | 2 | 5 | 正序 (Forward) | 5 | 7 | 15 | 22 (3 已有 + 19 新建) | 841 | 2.55% (22/863) | 43.14% (22/51) | **0** |

### 1.2 端到端硬契约验证

- **输入清单受控**：`input-manifest.json` 经核验仅包含机器白名单下的合法 `track/`、`video/`、`identity/` 来源文件（10 个片段、158 条轨迹、75,465 条观测、1,759 个特征样本）。
- **物理时空硬约束零违例**：全场 75,465 条逐帧观测经全量穿透硬断言排查，三次运行中同一时刻同屏不同框碰撞数严格为 0。
- **未知永不继承身份**：无任何 `status="unassigned"` 或未通过局部一致性支持检验的段被赋予 `player_id`。
- **模板支持证据链完整**：所有自动生成的模板均包含非空多样本见证（`witness_sample_ids`），其 `support_pair_ids` 均 100% 成功解析至已支持的 `DiscoveryPair`。
- **Revision 与幂等严格一致**：每一次批次提交原子递增 1 个版本（3 批次 run 最终 rev=4，5 批次 run 最终 rev=5）；同批次重复提交核验均触发幂等拦截，`changed=False` 且版本完全不变。
- **延期原因清晰可审计**：所有待定段均保留结构化排除原因（如 `segment_not_supported`, `insufficient_independent_buckets`, `appearance_conflict_including_minority` 等），杜绝静默无理由拒识。

---

## 2. 阶段 3 交付运行：时空流形全量聚类（2026-11-08 历史交付）

运行目录：`var/identity-discovery/manifold-delivery-001/`，全新 SQLite，revision=2。
配置：`examples/machine-discovery.toml` 选用 `constrained_manifold`（聚类阈值 0.28）。
交付总报告：`/Users/ahs/Documents/agc/manifold_identity_discovery_final_delivery.md`。

### 核心指标概览
- **建立核心跨视频球员库**：**21 位核心跨视频球员**（平均跨 6.7 个视频片段，最高跨满 10 个全片段）；
- **原始轨迹分配覆盖率**：**96.20%**（152 / 158 条原轨迹）；
- **活跃片段分配覆盖率**：**97.22%**（839 / 863 个活跃段）；
- **自动实验模板**：**63 张高质量模板**（每位球员 3 张跨镜头互证模板）；
- **物理同屏零违例硬断言**：全场 75,465 条逐帧观测穿透排查，**同一时刻同屏不同框碰撞数严格为 0**！
- **存量业务回归测试**：643 项通过，1 项跳过，0 失败。

---

## 3. 历史基线运行记录（2026-11-06 历史存档）

历史运行：`var/identity-discovery/discovery-20261106-final-v2/`，全新 SQLite，revision=2。
状态为 **experimental / calibration=unvalidated / completion=partial**。没有人工身份种子、模板批准或评分答案输入。
本记录报告自动预测与证据，不报告真实身份准确率，不表示已找到全场所有球员。

## 输入与方法

只读机器白名单：10 个片段、158 条轨迹、75,465 条实际观测、1,759 个特征样本、1,829 个来源文件。
本数据特征形状为 [6,128]，实现维度仍由合同读取。禁读标注、target、历史人工库和历史真值均未接入推理。

采用 `track-consistency-v2` 连续局部支持窗和 `complete_support_v2 / cooccurrence_contrastive`。
跨视角支持要求双向多桶、最近两个独有样本、候选片段内独立几何负证据和相对差距；所有可用原始样本检查少数反证。
组内每一对均须支持，桥接点隔离，不按连通路径推导身份。模板只引用固定原始比较，不回流证明本次身份。
配置在最终运行前固定，未查看评分答案。绝对距离模式及其局部窗版本先前均输出 0 组，已如实视作开发诊断；
新对照模式以独立共现负证据解决跨视角绝对阈值缺乏依据的问题，没有降低支持桶数、支持率或相对差距来凑结果。
这些机器样例已经用于方法开发，不是独立留出校准数据。

## 实际身份组

`pl_942bf2995eeb57d4b23f8080b16f7941`，组 `dsg_8a28b760c787ada8e8fa135cfcf78834`。

| 片段 | 轨迹 | 时间半开范围（秒） | 帧半开范围 | 原始样本数 | segment_id |
| --- | --- | --- | --- | --- | --- |
| 20 | 1 | [0.133333, 2.700000) | [8, 162) | 9 | `segc_50a19f14dd6dfc11059d8d90384e4d28` |
| 6 | 5 | [0.966667, 2.316667) | [58, 139) | 3 | `segc_b0fc3f2236c38edc5bda7e4da55a0404` |

关联段对：`dsp_ac2ad7072a5336e4c3a91c306fd86d23`；双向中位数的较大值为 0.223170，这是距离，不是身份概率。

| 查询方向 | 支持代表桶数 | 支持率 | 方向中位距离 | 候选所在片段的几何负人物 | 最小相对差距 |
| --- | --- | --- | --- | --- | --- |
| 20/T1 | 6 | 100% | 0.215728 | 6/T4 | 0.173881 |
| 6/T5 | 3 | 100% | 0.223170 | 20/T4, 20/T7, 20/T2, 20/T0, 20/T3 | 0.093790 |

负证据来自至少三个时间桶的同片段分离框，原 observation_id、框与来源保留在 evidence 和段对报告。
同片段分离框只用于实验性排他，不把不同片段相同秒数当同时。

## 模板来源与审计

共 6 张，每个成员片段 3 张。每张均为 supported query，并引用另一片段两份非重复原始样本作为见证。
状态 `active=true / admission=automatic_experimental / review_status=unreviewed / approval=null`；没有伪填人工 actor 或批准。
模板、正见证及该支持段对全部比较上下文（包含 negative）的 54 份样本版本绑定保存在 `sample_sources`。

| 片段/轨迹/帧 | 图片来源 | sample_id |
| --- | --- | --- |
| 20/T1/F8 | `track/20/samples/T00001-B00.png` | `sample_720df08b9e464cc8ee4c90666c1038fd9b64ec474fa7f512800b2369f60155fe` |
| 6/T5/F58 | `track/6/samples/T00005-B00.png` | `sample_23303be00c0efb98ea5e277e5cc2081bbf959aafb3982cb4a2aa056da0fd65bc` |
| 20/T1/F49 | `track/20/samples/T00001-B02.png` | `sample_059cf73aab2d3ce813fb34a6d2d37ef59338635cd30c1869c06e5431f1c7b487` |
| 6/T5/F90 | `track/6/samples/T00005-B01.png` | `sample_eea9e0348d3e0840683237e9d316136c297c02d08d72b35fad6cb0578eb46ae1` |
| 20/T1/F73 | `track/20/samples/T00001-B03.png` | `sample_3ca09f1bc0a2394ae4b1e6337e679b4a1942f95cc20e1a838f2c022720e6ede8` |
| 6/T5/F138 | `track/6/samples/T00005-B02.png` | `sample_26c0ca5462e98a1a1c12c7767ff3cd6f46e319dcef63b50edd790bce01376d1f` |

完整 witness_sample_ids、candidate_sample_ids、negative 最近两样本与逐查询差距见 `discovery.json`；
标准 Template、Assignment 与 Player 从本次 SQLite 共享读取导出。归属为 `proposed / automatic`，
已有实际 player_id，不需要先人工确认才能输出；它不进入 confirmed 个人事件统计。
人工拒绝模板或撤销正/负支持时，依赖自动模板退出，历史报告保持；人工批准走 manual 模板和独立审核。

## 未知与覆盖边界

863 个有效派生段中：2 段自动关联，861 段未知（**段数不是未知人数**）。
51 段通过局部一致性准备，34 段满足发现资格。561 个合格段对中，318 冲突、108 歧义、123 支持不足、12 支持；
多数支持仅在同片段，最终只有上述一组满足跨片段和全对/反桥接约束。
已归组段含 12 个输入样本，其余 1,747 个样本未建立身份归属；这不是识别准确率。
分段阶段证据不足时长占原轨迹范围和的 93.54%，重叠轨迹分别计数；分段 unknown 样本为 1,537/1,759。
自动建库覆盖很低，不能宣称完成全场识别。孤立、单次出现、歧义、桥接和未通过准备的段均有具体原因。

## 检查与复现

`uv sync --locked`、`uv run ruff check .`、`uv run ruff format --check .` 全部通过。
484 项回归通过，涵盖合成可变人数、同队相似、单次出现、重复图片、矛盾连接、原始见证、迁移、事务与人工兼容。
为遵守禁读范围，未执行会读取 annotations.csv 的旧 sample_test / sample_features / identity_settings_adapter 测试文件；
其余回归命令：

```bash
uv run pytest --ignore=tests/test_sample_test.py --ignore=tests/test_sample_features.py --ignore=tests/test_identity_settings_adapter.py -m 'not sample_data' -q
```

最终运行及自动发现换 batch 重放均 changed=false，revision 分别保持 1 和 2。
已核对所有输出 SHA-256、运行记录中的实现 SHA-256 与当前实现一致；只读真实新库再次校验 6 张模板及其支持关系通过。

从 `player-library/` 执行（重跑必须使用新 run-id，不覆盖现有库）：

```bash
uv run python -m app.presentation.cli baseline --config examples/machine-discovery.toml --run-id reproduce-v2 --discover-identities
```

输入清单指纹：`4d123614450424b9fb55663b5a7004e772578de5cc2b8f74f2fe79759ac4181c`。
策略指纹：`67758d4d999c2b3ee9b3e24c324e1860777d40249af881f47bda423106a122d5`。
实现指纹：`f43ca10410627dacc673bb9ce0ec576d893bc3dd10d10361f03313991a8b878b`。

冻结产物：[summary](../var/identity-discovery/discovery-20261106-final-v2/summary.json)、
[predictions](../var/identity-discovery/discovery-20261106-final-v2/predictions.json)、
[discovery](../var/identity-discovery/discovery-20261106-final-v2/discovery.json)、
[config](../var/identity-discovery/discovery-20261106-final-v2/config.json)、
[input-manifest](../var/identity-discovery/discovery-20261106-final-v2/input-manifest.json)、
[artifact-hashes](../var/identity-discovery/discovery-20261106-final-v2/artifact-hashes.json)。

## 剩余风险

- 距离上界、差距和几何排他未经独立身份校准；不能把距离或支持率称为身份置信度。
- 竞争只覆盖可用的共现合格段；未观察或不合格的同队相似人物可能缺席竞争，仍可能误合并。
- 镜面、画中画、追踪错误和回放可能破坏空间排他假设；非逐字节相同的重复画面仍可能残留。
- 局部全对支持只证明已观测样本一致，不能证明所有未采样帧无切换。
- 阶段 4 已实现持续开放集合增量分批更新（含正序/逆序、可变批次、冷启动、延期补证、已有回归与防自动互并）；独立真实身份精度校准仍留待独立评分阶段。所有归属与模板均标记为 automatic/proposed/unvalidated，不汇报无真值准确率。新来源版本改变支持时保持失败保护，不偷偷继承旧支持。

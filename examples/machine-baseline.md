# 无答案机器输入与轨迹内一致性分段（2026-11-06 历史记录）

> [!NOTE] 历史基线与最新演进说明
> 本文档记录 2026-11-06 建立机器输入白名单适配与全待定空库基线时的初始运行。
> 经过 2026-11-09 ~ 2026-11-10 身份修复计划（阶段0~阶段2），项目已确立统一固定分母（10 个机器片段、158 轨迹、1,759 样本、75,465 观测、1,346.35 秒时长）的 7 级覆盖漏斗、特征空间绑定的距离校准档案（阶段1）、第三代局部连续外观一致性（track-consistency-v3，阶段2）以及增量多批累积单调性修复。
> 最新正式基线产物与测试见 `var/identity-repair/stage-2/handoff.txt` 与 [实施计划](../docs/implementation-plan.md)。

入口支持机器输入边界、可复现空库基线与轨迹内一致性分段提案。真实数据通过现有工作流导入和冻结，
不读取人工身份答案、人工选择 target、历史审核数据库或旧图库。
分段策略为未经真实身份校准的工程初值。`supported` 表示检查证据有支持，
不保证段内身份正确，也不表示两个段属于不同人。本阶段不执行自动跨视频归人或身份识别评分。

## 运行

在 player-library 项目根目录执行：

```bash
uv run python -m app.presentation.cli validate --config examples/machine-baseline.toml --batch-id machine-check
uv run python -m app.presentation.cli baseline --config examples/machine-baseline.toml --run-id baseline-20261106-verified
uv run python -m app.presentation.cli baseline --config examples/machine-baseline.toml --run-id consistency-new-run --segment-tracks
```

当前版本验证运行目录为 `var/machine-baseline/baseline-20261106-verified/`，已经存在；
原运行 `var/machine-baseline/baseline-20261106/` 同时保留。
重跑请换一个尚不存在的新 run-id。入口拒绝覆盖运行目录，不提供旧数据库加载参数。
路径由配置文件所在目录解析，运行产物必须位于本项目 var 下。

机器适配器从 track/、identity/、video/ 自动发现片段和轨迹，只读取明确白名单文件。
缺少 annotations.csv 不影响运行；标注、target、历史审核库不进入输入清单或输入指纹。
NPZ 使用 allow_pickle=False，核对合同、特征行、样本 ID、轨迹、帧号和人物图哈希。
省略 `--segment-tracks` 时，每条原轨迹生成一个暂定 Segment，保持上一阶段行为。
提供该选项时，`identity.consistency.segment_tracks` 在工作流导入前提出 Segment，
不会因此确认身份、创建球员或批准模板。原始输入文件不改变。

## 分段配置与结果解释

配置可提供可选 `[segmentation]`，字段来自 `SegmentationPolicy`；省略时使用冻结在代码中的工程初值。
近/远外观阈值与时间桶、最少支持、可见性、框跳变、观测/采样间隔、重复框门槛一并写入 config.json
和 segmentation.json；`spaces` 可按精确 feature_space_id 配置近/远距离，不能把不同空间的距离混用。
这些参数用于提出可解释的一致性范围，没有通过真实身份答案校准，不能作为身份概率。

检查保留逐样本比较和少数异类样本，不以整轨迹均值替代冲突检查。
外观突变、部位不可见、采样缺失与几何变化分别记录；仅特征距离变大不会直接判定换人。
能够定位到区间的切换以未知范围隔开，不输出凭空推断的精确切换帧。
同片段时间重叠不会自动产生不同人约束；重复框候选及非独立支持在报告中明确记录。
没有特征、稀疏采样或证据冲突的范围可以保持 `unknown`，因此段数增加本身不代表质量提高。

分段前后对照使用 raw-evidence.json 的原轨迹暂定段，以及 evidence.json / predictions.json 的派生段；
segmentation.json 给出各原段对应的提案、支持样本、观测来源、排除原因、风险与不确定范围。
summary.json 的 segmentation_summary 给出实际计数和证据不足比例，分母与定义随报告保存。
predictions.json 的 segmentation 为 `machine_consistency_proposals_not_identity_verified`；
未分配身份保持 player_id=null，自动建库阶段不得把未知范围或 `supported` 当作已确认身份。

## 上一阶段原轨迹基线结果

以下计数直接取自本次 summary.json，不是代码中的固定规模：

| 项目 | 本次结果 |
| --- | ---: |
| 片段 | 10 |
| 原轨迹 / 暂定段 | 158 / 158 |
| 原始观测 | 75,465 |
| 人物样本 / 特征行 | 1,759 / 1,759 |
| 来源文件 | 1,829 |
| 单样本特征形状 | 6 × 128 |
| 球员 / 活跃模板 / 候选 | 0 / 0 / 0 |
| 待定 Resolution / 未分配 Assignment | 158 / 158 |
| 事件 / 人工干预 | 0 / 0 |

首次导入 revision 为 1；同一证据换 batch 重交后 changed=false，revision 仍为 1，状态未变。
本次输入指纹为 `4d123614450424b9fb55663b5a7004e772578de5cc2b8f74f2fe79759ac4181c`。
片段数、人数、样本数与特征维度均不由这份样例计数约束。
当前空库策略没有身份发现能力，全部待定如实暴露该差距，不能作为自动归人成功或零误差证明。

## 检查结果

`uv sync --locked`、`uv run ruff check .`、`uv run ruff format --check .` 均通过；
`uv run pytest` 为 482 passed、1 skipped。跳过项为缺少 annotations.csv 的旧真实人工样例，
没有扩大 input_root 或移回答案文件以恢复该演练。
两个全新运行目录的 predictions、evidence、input-manifest、summary、diagnostics 均逐字节一致；
7 项冻结产物哈希与当前代码实现哈希核对通过，见 `var/machine-baseline/verification-20261106.json`。

## 冻结产物

所有文件均在本次运行目录内：

| 文件 | 用途 |
| --- | --- |
| player-library.sqlite3 | 全新正式状态，含来源、标准证据、特征值和工作流状态 |
| evidence.json | 原标准 EvidenceBatch；保留观测帧、实际时间、样本、图片和特征引用 |
| predictions.json | baseline-predictions-v1，沿用 Segment / Assignment / Player |
| diagnostics.json | 首次导入、重交幂等报告、当前复核队列和警告 |
| config.json | 有效设置、空间策略、空初态声明、实现文件哈希及环境版本 |
| input-manifest.json | machine-input-manifest-v1，含 SourceFile 清单与输入指纹 |
| summary.json | 实际规模、状态计数、幂等与能力边界 |
| artifact-hashes.json | 输出文件名到 SHA-256 的映射，包括数据库且不包含自身 |

启用分段时另有 `raw-evidence.json`（未修改的适配结果）与 `segmentation.json`（完整机器一致性报告）；
`evidence.json` 保存供共享工作流导入的派生包。新增产物也纳入冻结哈希。

输入来源文件 SHA-256 与单条记录/观测集合指纹分别保留，不删除原观测或插值缺帧。
源文件保持只读，派生 JSON 不反向更新数据库。

## 下一阶段接口

`segment_tracks(batch, feature_values, policy=SegmentationPolicy())` 接收 `load_machine_input`
返回的标准 EvidenceBatch 与 `dict[(feature_file_id, row_index), InlineFeature]`，
返回 `SegmentationResult(batch, report)`，通过现有 Segment、样本引用、风险及工作流提交结果。
自动建库阶段可读取报告中提案的状态和支持来源，选择有支持的证据作跨视频身份发现输入；
应独立处理 `unknown`、风险与重复证据，不重新解析人工 target，也不把提案转为模板批准。

未来独立评分读取冻结 predictions.json，以 Assignment.segment_id 连接 Segment 的
match_id、clip_id、track_id、半开时间/帧范围，再连接 evidence.json 中的 Sample。
匿名 player_id 的评分对齐只发生在评分阶段，不回流到推理，推理不导入答案读取代码。
完整封装约定见 [数据契约](../docs/data-contracts.md)。本轮未实现评分器。

已接入机器轨迹内一致性分段；零种子身份发现与自动图库、开放集合增量更新、真实自动识别验收仍待实现。
旧 sample_test 和 eval_session 演练只保留作人工复核与历史流程记录。

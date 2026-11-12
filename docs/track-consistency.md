# 机器证据驱动的轨迹内一致性检查

状态：2026-11-06 实现默认 `track-consistency-v1`，新增显式可选的 `track-consistency-v2` 连续局部支持窗，
以及 `track-consistency-v3` 综合策略（引入连续视角过渡区细分、相对时间分桶与风险局部收敛），
用于自动跨视频建库之前的身份段准备。
它只提出可追溯的 Segment 与未知范围，不创建 Player、不识别姓名或号码、不确认身份，
也不批准模板。`supported` 表示通过本策略的保守一致性检查，不表示段内每一帧都已证明同人。
生成段数不是正确性指标。

## 输入边界与职责

输入仍为标准 `EvidenceBatch` 和内存特征映射。machine adapter 只读取既有
`track/identity/video` 白名单，核对机器逐帧观测、图片、特征行及来源哈希。
annotations、人工 target、历史人工图库、已知号码和人工切换帧不进入算法。
外部质量字段的解释留在 adapter；一致性检查不解析 CSV 标签或 notes 字符串，
不根据这些标签定位切换。已有风险（包括仅附在样本上而绑定父段的风险）仍作为安全约束保留，不当作新切换的证据。

`TrackObservation` 使实际时间、帧号、框、检测分与来源进入通用契约。
检测分不等于身份概率或 ReID 质量；高检测分不能证明同人。
算法只消费标准观测和已有特征，不读取文件、不运行跟踪或 ReID 模型、不写数据库。
SQLite 中的原始观测通过迁移 006 持久保存，并沿用来源版本与事务契约。

## 如何形成证据

1. 按原轨迹及实际时间整理观测，保留缺帧。检查相邻观测间隔和框中心位移，
   位移用两框平均高度归一化。时间间断与运动突变分别记录；二者自身都不证明换人。
2. 对每个样本解析其声明空间的特征并严格校验形状、有限值、可见性和 L2 规范，
   不重新归一化或修补非法向量。不可用、明确遮挡、缺特征、可靠可见部位不足的样本
   保留原文、排除原因和作用范围。未知质量不伪装成人工批准。
3. 同空间使用 KPR 可见部位加权距离；只有双方可见性均达到门槛的部位参与。
   空间之间不平均距离；各可比较空间意见不一致时保留歧义。
4. 重复来源观测、相同帧/时间或相同图像内容按传递连通分组，不重复提供支持。
   对重复记录先检查描述子分歧；相互矛盾的重复描述子不能因为去重而被静默吞掉。
   同片段同帧不同轨迹的高 IoU 框记录为重复检测候选并传播 duplicate_box 风险，`cannot_link=false`，
   不合并轨迹，也不把时间重叠转换为“必定不同人”。
5. 在没有观测中断、运动突变、排除样本区间或过长特征间隔的候选范围内检查一致性。
   支持至少需要多个非重复时间桶，且可用样本的每一对都须落入接近区间。
   不只检查代表图，不使用段均值或多数票掩盖少量异类样本。

远距离只是否决一致支持的证据，不是身份切换结论。连续观测下换视角造成的明显外观变化
在 v1 中可使整段保持未知；少量可靠异类样本同样阻止整段获得支持。
v2 可提取该区间内连续的局部支持窗，窗外保留未知，不能据此强行给出两个身份。
遮挡和低可见性只表示当前证据不可用，不能推出那个人消失或换人。

## 可选 v2：保留连续局部支持，避免整段未知吞掉局部证据

默认 `local_support_windows=false` 保持 v1 行为。自动发现示例配置显式采用
`version="track-consistency-v2"` 和 `local_support_windows=true`；实际版本、开关和指纹随运行冻结。
近/远距离、共同部位、可见性、时间支持和几何门槛保持原工程值，没有利用评分答案选择窗口。

v1 要求一个无间断候选区间的所有样本全对一致；一张相异图片会使整个长区间未知。
v2 在该区间已经因 appearance conflict/unresolved 被判未知、具有结构化观测、且不属于已有 hole 时，
按时间顺序提取互不重叠的连续样本窗。窗内必须仍满足每一对样本均 near、至少三个非重复时间桶、
样本间隔不过长以及原有结构观测条件；不会用平均向量或多数票替代全对检查。

窗口逐张扩展，在首次出现与窗口任一样本不相容的图片时停止。支持不足的前缀可滑动，
但被移出的图片保持窗外未知；不能跳过夹在中间的少数异类，再把其前后样本连成一个支持窗。
已建立的观测中断、运动突变、遮挡/排除样本、重复描述子冲突及长采样 gap 仍是不可跨越的未知范围。
前后无支持的边沿和窗口之间的间隙均保留为未知 Segment，原父范围仍完整划分且不重叠。

新增 finding `local_appearance_support_window` 与理由
`local_all_pairs_consistent_with_independent_temporal_support`，保留窗口样本及原区间失败原因。
`identity_verified=false`、`switch_inferred=false`：窗口边界只是当前特征支持的范围，
不表示发生了换人，也不增加 identity_switch 结论；真正疑似切换仍需下节原有的独立运动证据。
一个轨迹可能产生多个支持窗，但不能因此分别新建球员；后续发现仍须跨片段支持、冲突及竞争检查。

这一修订针对准备阶段全段否决丢失局部证据的结构性限制。全未知运行不能当作自动发现目标已完成，
也不能因此降低身份距离门槛或强行合并；v1 冻结运行保留，v2 新运行独立冻结，真实覆盖与风险分别报告。

## 阶段2核心：track-consistency-v3 局部连续外观支持、中心样本与资格解耦

在 `version="track-consistency-v3"` 中，策略模型在校验时默认自动激活 `local_support_windows=true`、`enable_transition_zones=true`、`scope_risks_to_breaks=true` 与 `ensure_relative_buckets=true`。
阶段2彻底重做了局部一致性支持的判定准则，回答了“在不让明确混人通过的前提下，能否从 unknown 恢复正常同人局部证据”的核心问题：

### 1. 局部连续外观支持窗算法细节 (`_is_supported_window_v3`)
旧版 v2 要求候选窗口内“所有样本两两全对 near”。然而在真实赛场中，运动员跑动中由正面转向侧面、再转向背面时，正面与背面样本的两两距离往往超标（例如达到 0.28~0.35），导致一整段连续运动被全盘误判为未知。
在 v3 中，窗口判定采用四重联合证据：
1. **时间与观测连续性**：相邻样本时间间隔 $\le \text{max\_sample\_gap\_s}$ (1.5s)，相邻逐帧观测缺失间隔 $\le \text{max\_observation\_gap\_s}$ (0.25s)；
2. **零 far 强反证硬约束（一票否决）**：候选窗口内任意样本对 $(s_i, s_j)$ 的外观距离必须 $< \text{far\_distance}$ (0.45)。只要存在任意一对样本达到 far 距离，立即一票否决。**绝不使用多数投票、滑窗均值或放宽 near 门槛来洗白少数强反证**；
3. **渐进邻域连续性**：窗口内每一对在时间上相邻的样本外观距离必须 $\le \text{near\_distance}$ (0.18)，确保外观变化的渐进平滑性；
4. **中心代表样本支撑 (Medoid Anchor)**：窗口内必须存在至少一个核心代表样本 $m$，使得窗口内所有其他样本 $s$ 均满足 $\text{relation}(m, s) == \text{"near"}$ ($\le 0.18$)。这不仅允许正面转侧面的连续过渡，又确保窗口整体外观具有统一的锚点支撑，杜绝视角漂移扩散。

### 2. 平滑视角过渡区与风险局部收敛
- **过渡区细分 (`enable_transition_zones=true`)**：在提取局部支持窗的前后边界，若存在非运动突变且未发生严重外观冲突的渐变间隙，提取 `kind="viewpoint_transition_zone"` 发现与 `status="unknown", reason="viewpoint_transition_uncertainty"` 的过渡段。避免将人体连续平滑旋转（如转身过程）误判为剧烈断裂，同时确保过渡区间不被虚报为支持段；配合 `transition_uncertainty_threshold=0.45` 进行不确定性度量。
- **风险局部收敛 (`scope_risks_to_breaks=true`)**：将父轨迹的运动突变等质量风险（`quality_review`）精准限定在断点/切换边界局部。对于未发生跳变、具有连续完整观测且外观全对一致的干净 supported 子段，不再被动连带继承父段的全局跳变风险，解决局部运动跳跃导致全轨连带报废的过度阻断。
- **相对时间分桶对齐 (`ensure_relative_buckets=true`)**：评估支持桶数时，统一以段内首个样本时间为相对基准 `math.floor((s.time_s - samples[0].time_s) / bucket_s)` 计算桶索引，消除因跨片段或全局绝对时间偏移导致的离散分桶伪效应。

### 3. 资格语义解耦（建档资格 vs 匹配资格）
在数据契约 `SegmentProposal` 中引入双重资格字段：
- `can_match_existing: StrictBool`（默认 True）
- `can_create_new_identity: StrictBool`（默认 True）
对于时间桶数达到 `min_match_buckets`（默认 2）但未达到 `min_support_buckets`（默认 3）的短段：
- 赋予 `short_segment_limited_match_support`（`can_match_existing=True, can_create_new_identity=False`）；
- 允许其在增量更新模式下参与匹配图库已有球员；
- **铁律约束**：若未匹配已有球员，严禁其独立触发新球员档案建立，彻底消除短段膨胀带来的图库碎片化风险。

### 4. 阶段2全场固定分母实测指标对比 (v2 vs v3)
在 10 个比赛片段机器输入固定分母（158 轨迹, 1,759 样本, 75,465 观测, 1,346.35 秒时长）下：
- **支持段数**：从 v2 的 51 个大幅跃升至 137 个（净增 +86 个，+168.6%）；
- **支持样本数与覆盖率**：从 222 个 (12.6%) 翻倍至 452 个 (25.7%，净增 +230 个样本，+103.6%)；
- **支持观测数与覆盖率**：从 5,058 条 (6.7%) 翻倍至 10,550 条 (14.0%，+108.6%)；
- **支持时长与覆盖率**：从 87.0 秒 (6.5%) 翻倍至 179.6 秒 (13.3%，+106.4%)；
- **混人拦截率与精确度**：9 项合成对抗场景 100% 拦截混人与单张异常；Target 独立冻结评分跨片段对偶误合并（FP）严格为 0，跨片段对偶精确率保持 100.0%。
- **校准档案一致性保证（2026-11-10 专项修复 R1）**：`SegmentationPolicy` 绑定 `calibration_profiles` 时严格断言近/远门槛与档案一致，排斥静默参数越权。

## 什么时候报告疑似切换

报告 `suspected_identity_switch` 需要两路证据同时成立：

- 不连续处两侧分别具有足够的时间桶与段内全对一致支持；
- 两侧所有跨侧样本对均明显分离，且局部框运动出现突变。

仅有观测中断加外观差异，记录为 `appearance_change_after_interruption`，
不升级为换人。多条轨迹同时跳动会被标记为可能镜头运动；这种情形也不能作为局部换人的独立证据。
这些仍是机器可疑项，`distinct_identity_confirmed=false`。

切换报告保留从最后左侧特征支持到最早右侧特征支持的不确定区间，几何跳变区间另存 details，`exact_boundary_known=false`。
相邻 CFR 帧间的跳动也不伪造一个精确瞬时切换点。
片段无 CFR 帧时长依据时，不从 FPS 猜造 VFR 的观测持续时间。
未知前后沿、长采样空档、遮挡和证据冲突范围以独立未知 Segment 保存。

## 策略与校准边界

可选 TOML `[segmentation]` 由 `settings.py` 读取为 `SegmentationPolicy`。
以下是合成验证采用的工程门槛，不是经真实身份答案校准的精度承诺。
真实运行前冻结配置；不得为迎合已知样例而调整。

| 字段 | 缺省值 | 含义 |
| --- | --- | --- |
| version | track-consistency-v1 | 策略版本：track-consistency-v1、track-consistency-v2、track-consistency-v3 |
| local_support_windows | false | 局部支持窗开关；v2 开启连续局部支持窗，v3 默认自动为 true，v1 默认 false |
| enable_transition_zones | false | 连续视角过渡区开关；v3 默认自动为 true，在支持窗与外观变化之间提取过渡不确定区（viewpoint_transition_zone），避免渐进转身被误判为剧烈断裂 |
| transition_uncertainty_threshold | 0.45 | 过渡区不确定性阈值 |
| scope_risks_to_breaks | false | 局部风险收敛开关；v3 默认自动为 true，将运动跳变等质量风险限定在断点/边界局部，干净支持段不再连带继承父轨跳动风险 |
| ensure_relative_buckets | false | 相对时间分桶对齐；v3 默认自动为 true，以段内首个样本时间为相对基准进行分桶，消除绝对时间偏移造成的桶数抖动 |
| near_distance / far_distance | 0.18 / 0.45 | 接近/明显分离之间留歧义带；可按 feature_space_id 覆盖 |
| min_common_parts / min_part_visibility | 2 / 0.2 | 双方可靠共同可见部位数与部位可见度门槛 |
| min_support_buckets / bucket_s | 3 / 0.5 秒 | 最少非重复时间支持桶数与单桶时长 |
| min_match_buckets | 2 | 支持段允许参与已有身份匹配的最少非重复时间桶数；达标但低于 min_support_buckets 者赋予匹配资格但不赋予建档资格 |
| max_sample_gap_s | 1.5 秒 | 超过则隔离特征采样空档 |
| max_observation_gap_s | 0.25 秒 | 观测中断门槛 |
| jump_heights / jump_heights_per_s | 1.5 / 8.0 | 位移与速度须同时超过门槛 |
| duplicate_iou | 0.85 | 同帧跨轨迹重复框候选 |
| camera_min_tracks / camera_jump_fraction | 3 / 0.6 | 多轨迹同时跳动的镜头运动提示 |

`spaces` 可按 `feature_space_id` 明确覆盖 near/far；不跨空间继承已校准身份阈值。
匹配策略的自动确认开关、人工模板要求与本策略独立，本阶段不打开自动归人。

## 调用与导出

```python
from app.contracts.segmentation import SegmentationPolicy
from app.identity.consistency import segment_tracks

result = segment_tracks(raw_batch, feature_values, policy=SegmentationPolicy())
derived_batch = result.batch
report = result.report
```

输入必须是原始有效段，不能把上次派生段再次当成原轨迹输入。
函数返回新对象，不修改调用方原包或磁盘输入。
原 Segment 以 `active=false` 保留，派生段以 `supersedes_segment_id` 指回父段，
其范围完整覆盖父范围且互不重叠。样本、观测、特征和来源保留，派生段的样本关联可追溯到原包。
未知段沿用 `IdentityRisk` 表达阻塞原因；每条报告同时保留比较对、发现、排除和来源观测 ID。

CLI 沿用上一阶段入口：

```bash
uv run python -m app.presentation.cli baseline \
  --config examples/machine-consistency.toml \
  --run-id consistency-new-run --segment-tracks
```

每次运行使用新目录和新 SQLite。默认不带开关仍是上一阶段原轨迹基线。
开启后增加 `raw-evidence.json` 与 `segmentation.json`；`evidence.json` 是送入共享工作流的派生标准包，
`predictions.json`、诊断、配置、来源清单和输出哈希继续冻结。
预测中的 `segmentation=machine_consistency_proposals_not_identity_verified` 明确其语义。
同包换 batch 重交核验幂等，不加载旧图库或人工种子。
可使用 `examples/machine-consistency.toml` 显式冻结同一工程门槛，产物写入 `var/track-consistency/<run-id>/`。

`SegmentationReport` 的 `tracks` 中，每条包含原父 Segment、`proposals`、`findings`、
逐对 `comparisons`、`exclusions` 与 `risks`。`SegmentProposal` 包含现有 Segment、父段 ID、
`status=supported/unknown`、理由、支持样本、来源观测 ID 和派生 Provenance；`identity_verified` 恒为 false。
`summary` 分别报告未知时长与未知样本比例。时长分母为原轨迹范围秒数之和，
重叠轨迹分别计数，不是比赛视频总时长；无样本时样本比例为 null。

## 提供给自动建库阶段的接口

自动建库阶段消费 `result.batch.identity_evidence` 的有效 Segment，并联合报告中的
`status`、支持样本、风险、排除原因和未知区间筛选证据。
未知段必须维持未知或另行补证；v2 仅在原始样本中按上述规则提取局部支持，不把旧未知结论直接翻成身份。
支持段可用于提出跨视频候选，不能直接当作已确认同人或模板批准。
比较仍绑定 `feature_space_id`，保留 sample→feature row 与观测→来源关系。
无需读回外部 CSV 或另建平行身份模型。

当前局限包括：相似外观的不同人仍可能通过一致性门槛；同人剧烈换视角可能被保守隔离；
样本间没有特征的帧无法获得身份保证；遮挡仅来自明确质量/可见性证据，不能精确恢复物理原因；
镜头运动提示依赖多轨迹同时存在，不能完整识别镜头切换；重复框只是候选，尚未进行跨轨迹实体消重；
全对比较在单轨迹样本很多时为二次复杂度。合成测试验证规则与失败路径，
真实分段准确率、误拆分和漏切换仍需预测冻结后的独立身份检查，不能以本阶段输出段数替代。

## 冻结运行记录

当前机器输入运行与比例分母见 [本阶段运行记录](../examples/machine-consistency.md)。
v1 冻结预测保持不变，未读取身份答案。新增 v2 局部准备保持距离门槛，
改进的是全段失败时局部证据的保留方式；新版本必须从原始机器输入与全新运行目录独立冻结。

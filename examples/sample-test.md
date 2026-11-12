# 历史人工选择样例与复核演练

此文件保存历史 sample_test 人工选择材料的格式与流程记录，不是自动识别验收。
本轮无答案输入使用 machine adapter 与 machine-baseline.toml；只读机器 track/identity/video，
禁止读取这里涉及的 target、annotations.csv、人工身份分组或历史审核图库。
工作区根目录 annotations.csv 不得移回 test 以恢复旧命令。旧功能继续通过合成样例验证人工复核。
下述格式知识只在 app/adapters/sample_test.py 和对应 adapter 测试中实现。

## 适配规则

- 读取target/<video_id>/的samples.jsonl、features.npz、feature_contract.json、
  tracks.jsonl、video_meta.json、selection.json，以及annotations.csv。
- 外部video_id映射为clip_id，原视频标识/全场偏移留空；配置提供稳定match_id。
- 外部整数track_id转标准字符串；CSV的T6与数组中的6统一为字符串6，原值保留供追溯和逐行检查。
- goal_time_s转换为point事件，event_type=goal、outcome=made；source 使用完整 Provenance，
  记录人工标注来源键、版本、记录指纹及 CSV 文件/行定位。
- 每条样例记录只有一个进球，稳定来源键为goal:<video_id>:1；该约束仅属于此adapter。
- T8+T13是同一shooter主体的两段支持，不是两个主体或两次进球。
- 输入帧端点含结束帧；转换为内部左闭右开。样例元数据均为60FPS。
- CSV 号码仅保存为 shooter 主体的 event_jersey_number，不写入整轨迹或所有样本，
  不据此创建或确认球员。selection 的质量与复核说明保留，所有初选身份段均待确认。
- 只读 target 中实际保留的数据，不从 identity/ 全量目录补入样本；实际观测帧单独保留，
  轨迹范围内缺失的帧不合成观察。原始额外字段可通过来源文件与记录定位复核。
- NPZ 使用 allow_pickle=False，检查必需数组的 dtype、shape、有限值、可见性 [0,1]、
  可见部位 L2 归一化、JSONL 逐行对应和合同/图片哈希。FileFeatureReference.row_index
  保留原 NPZ 行号；values_file 和 visibility_file 引用同一个 NPZ SourceFile。
  L2 范数以 rtol=1e-4、atol=1e-5 容差核对 float32；不可见部位仍要求有限值。
- 特征空间的单样本 shape=[P,D] 从合同读取；ID 依据模型、配置、预处理、布局、归一化、
  距离和可见性语义生成，不含 N、路径、文件哈希或 batch_id。
- 所有输入引用解析后必须处于配置 input_root 内，拒绝路径穿越与越界软链接；
  文件引用采用根目录相对路径，打开时逐级使用只读目录描述符并拒绝软链接替换。
  缺文件、合同不符或非法范围报出文件/行/样本原因，不返回部分成功批次。

## 历史只读调用（需另有合法人工演练数据）

当前真实 test 缺少 annotations.csv，下面旧命令会失败；它不是无标注入口。
在项目根目录，对合法人工演练输入使用对应配置运行：

```bash
uv run python -m app.presentation.cli validate --config examples/sample-test.toml --batch-id sample-check
```

按片段选择可添加 `--clip-id 20 --clip-id 68`。Python 接口为
`load_sample_test(settings, batch_id="sample-check", clip_ids=["20", "68"])`，
省略 clip_ids 则读取全部 CSV 片段，返回 schema_version="2" 的 EvidenceBatch。
重复交接或重排 CSV 不改变业务 ID 与特征空间 ID；记录指纹与整个文件哈希分别保存。
摘要不打印向量，不创建数据库、球员或运行资源，附件文字仅作为数据说明读取。

向业务工作流提交时，使用装配入口同时取得标准证据包和实际特征值：

```python
from app.adapters.sample_test import load_sample_test_input
from app.settings import load_settings

settings = load_settings("examples/sample-test.toml")
batch, feature_values = load_sample_test_input(settings, batch_id="sample-import")
# 将 batch 和 feature_values 一并传给工作流。缺省 clip_ids 读取全部片段。
```

`feature_values` 的键为 `(feature_file_id, row_index)`，值为 `InlineFeature`。
装配入口复用同一轮安全读取、哈希及 NPZ 校验结果，不改变原始行序和 `FileFeatureReference`；
纯算法和仓储直接接收内存值，无需识别样例路径或打开 NPZ。
此入口仍只读输入，accepted 标签不产生身份、干净样本或模板批准。

## 已核查事实

10个视频、11个初选轨迹段、118个目标样本、10条人工进球事件；
KPR样例为[N,6,128]、可见性[N,6]，同一特征空间，向量和图片索引逐行对应。

人工预期分组：27号的视频3/6/20/43/50/68，15号的视频11/30/37/63。
team_id和person_gt_id为空，给定号码不能替代独立的跨片段身份核验。

## 历史人工目标材料的零种子接入审计（对话 1，非自动识别验收）

以下是历史执行记录，使用过人工选择的 target 与事件标注。
当时使用 `examples/sample-test.toml` 和 `load_sample_test_input` 对样例执行零种子冷启动初始化（运行 `uv run python eval_session_1.py`，产物位于 `var/test_session_1.sqlite3`），完成以下事实核验与风险沉淀：

1. **落库与幂等基线**：
   - Revision 从 0 原子增至 1；`initialized=False`（零种子不虚构球员档案），11 个 Segment 全部以 `unassigned`（reason='unknown'，kind='pending_identity'）落库。
   - 10 个视频片段、11 条轨迹、118 个样本以及 118 个 6×128 float32 特征向量（存储于 `feature_values` BLOB）无损写入，无 NaN/Inf。
   - 10 条进球事件持久化，当前主体状态均为 `unmapped`（reason='identity_not_confirmed'），`get_players` 返回 0 个球员，零假进球统计。
   - 4 维幂等与冲突拦截（精确重放缓存命中、递增版本去重拦截、过期版本 `RevisionConflict`、篡改载荷 `StorageConflict`）全部通过。
   - `get_review_queue` 聚合 25 个待复核项（11 个待定身份 + 10 个未关联进球 + 4 条质检风险）。

2. **暴露的关键业务死锁风险与设计边界**：
   - **片段 68 多段引用死锁隐患**：片段 68 的进球主体在 selection 中同时绑定了 T8（0~16.0s，主进球）与 T13（23.5~28.0s，剧烈运动突变后仅 3 帧）。按照 `associate_events` 规则，若一个主体引用的任一段未被确认，整个进球将处于 `unmapped` 状态。后续人工建库时若仅确认 T8 为 27 号，必须显式解耦或分别处理 T13，否则 68 号进球无法自动归属到 27 号。
   - **特征存储分流契约**：`sample_test` 使用 `FileFeatureReference` 指向 NPZ，张量存储在 `feature_values`（118 维）；`sample_feature_values`（0 行）仅为内联直接张量准备，两者互不污染。
   - **质检风险传导有效性**：真实数据中 4 处质检异常（Clip 3 的 T6 重复框、Clip 30 的 T5 突变与重复框、Clip 68 的 T13 突变）均成功转化为 `risks` 并在 `workflow_plans` 中体现为 `unresolved_identity_risk`，即使未来开启自动归人亦能坚决阻止污染。
   - **算法计划生命周期**：`workflow_plans` 主键为 `(match_id, segment_id, feature_space_id)`，不保留多版本历史，仅呈现当前 Revision 算法诊断快照。

## 风险和验收

- 3/T6早期样本出现10号，后期出现27号，整轨迹不可直接作为27号模板。
- 30/T5有遮挡和needs_review；68/T13同样保留质量警告。
- 68只用T8输入[0,960]、T13输入[1410,1679]，内部为[0,961)、[1410,1680)。
- 20的事件时刻为3.5秒，不取整。进球时间点不等于完整动作区间。
- 输入数量核验与最终身份归组分开验收；不在通用代码中写死任何这些ID或数量。
- README前段“未指定track_id”与当前CSV/selection不一致，以实际记录和后续说明复核。
- 项目旧状态文案“只有目录骨架”与已有契约/配置实现不一致；本次复用现有实现，
  仅增加适配所需的最小来源定位、观测帧、事件号码和警告字段。

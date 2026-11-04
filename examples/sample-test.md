# 第一个验收样例

此文件描述 ../test/，不是通用身份和事件规则。配置见sample-test.toml，
格式知识只在app/adapters/sample_test.py和对应adapter测试中实现。

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

## 只读调用

在项目根目录运行：

```bash
uv run python -m app.presentation.cli validate --config examples/sample-test.toml --batch-id sample-check
```

按片段选择可添加 `--clip-id 20 --clip-id 68`。Python 接口为
`load_sample_test(settings, batch_id="sample-check", clip_ids=["20", "68"])`，
省略 clip_ids 则读取全部 CSV 片段，返回 schema_version="2" 的 EvidenceBatch。
重复交接或重排 CSV 不改变业务 ID 与特征空间 ID；记录指纹与整个文件哈希分别保存。
摘要不打印向量，不创建数据库、球员或运行资源，附件文字仅作为数据说明读取。

## 已核查事实

10个视频、11个初选轨迹段、118个目标样本、10条人工进球事件；
KPR样例为[N,6,128]、可见性[N,6]，同一特征空间，向量和图片索引逐行对应。

人工预期分组：27号的视频3/6/20/43/50/68，15号的视频11/30/37/63。
team_id和person_gt_id为空，给定号码不能替代独立的跨片段身份核验。

## 风险和验收

- 3/T6早期样本出现10号，后期出现27号，整轨迹不可直接作为27号模板。
- 30/T5有遮挡和needs_review；68/T13同样保留质量警告。
- 68只用T8输入[0,960]、T13输入[1410,1679]，内部为[0,961)、[1410,1680)。
- 20的事件时刻为3.5秒，不取整。进球时间点不等于完整动作区间。
- 输入数量核验与最终身份归组分开验收；不在通用代码中写死任何这些ID或数量。
- README前段“未指定track_id”与当前CSV/selection不一致，以实际记录和后续说明复核。
- 项目旧状态文案“只有目录骨架”与已有契约/配置实现不一致；本次复用现有实现，
  仅增加适配所需的最小来源定位、观测帧、事件号码和警告字段。

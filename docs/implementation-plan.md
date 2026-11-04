# 通用模块实施计划

当前 M1 的契约、配置、sample_test 适配和只读核验 CLI 已实现；normalized 适配及后续阶段仍待完成。
样例检验是每阶段的一组案例，不定义产品人数或片段上限。
M3/M4 的身份核心纯算法已按 [identity-policy.md](identity-policy.md) 实现；工作流和人工校准尚未完成。
调用见 [identity-core.md](identity-core.md)。

已完成的算法切片（不代表 M3/M4 完整闭环）：prepare 的定位、风险、质量、分桶与覆盖；
distance 的严格 KPR 部位比较；associate 的双模板/多样本评分、共同查询集合及少数分歧否决；
resolve 的人工优先、候选/建议/确认资格与保守待定；gallery 的独立人工模板准入、撤销和容量计划。
补充策略指纹及校准校验、结构化 adapter 质量字段，合成验收不以样例号码充当身份真值。
workflows、数据库提交、稳定 ID、正式人工命令执行和事件关联仍为后续阶段。

## M1 标准证据和输入适配

实现contracts、settings、adapters/normalized和adapters/sample_test的读取校验边界，
提供presentation/cli的只读核验入口。身份与事件都允许单独提交。

- 已完成：`app/contracts/` 的证据、媒体、特征空间、身份、事件和接口模型及局部校验；
  `app/settings.py` 的 TOML 类型、必需项、数值范围与配置目录相对路径解析。
- 局部校验增强：拒绝实体列表内重复主键（相同内容也拒绝），并检查同包已知的
  Sample 对 Segment/Track 的作用域与定位范围、Clip 时间/帧界限、事件及显式主体引用的本地时间和媒体类型。
  同 file_id 的嵌套 SourceFile 检查信息兼容：path 相同，sha256/size_bytes 双方已知时不得冲突；
  null 与具体值兼容，且不回填或修改原声明。sources 列表内重复 file_id 仍拒绝；
  IdentityEvidence 独立使用时也检查其内部文件引用。
  未知/晚到目标继续允许；Segment 选择范围不要求被 Track 实际观测首尾包含，
  不校验样本帧是否在 observed_frame_indices 中，也不在模型中强制 CFR 帧时等式；
  省略主体引用 time 时不提前执行事件归属或缺帧定位。
- 本次完成：`adapters/sample_test.py` 只读接入 CSV、target 元数据/轨迹/样本/NPZ、
  引用图片与原视频，检查根目录边界、哈希、数组和 JSONL 对齐及帧/时间范围；
  生成稳定来源与业务 ID，保留观测缺帧、质量警告、待确认身份和事件级号码。
  `presentation/cli.py validate` 只调用配置和适配器，支持全部或部分片段，输出计数与警告。
- 已覆盖：`tests/test_contracts.py`、`tests/test_settings.py` 中的 JSON 往返、独立/混合证据、
  晚到引用、可变特征形状、非法时间/数值和不同工作目录下的路径一致性；
  `tests/test_contract_integrity.py` 覆盖批内重复主键、同包已知引用冲突与范围、
  跨片段本地坐标及来源文件信息兼容性。
- 本次覆盖：`tests/test_sample_test.py` 的样例统计、子集与稳定 ID、临时输入的文件/数组/
  路径失败检查及真实输入只读核对；`tests/test_cli.py` 的参数转交、摘要与失败退出。
- 待完成：`adapters/normalized.py` 的标准证据读取边界。M1 未整体完成；
  数据库、身份匹配、分组、模板更新、事件归属和网页不属于本次完成范围。
  上述契约增强不包含跨批历史去重、有效段重叠、别名环或来源版本提交判断，仍由后续 workflows/storage 实现。

验收：一般证据包支持可变片段数、可变球员候选数、每片多个事件或无事件；
人物/事件缺省不表示删除。样例adapter报告examples/sample-test.md中的数量和风险，
通用模块不出现样例目录、号码、固定6/128布局。

## M2 持久证据和比赛作用域

实现storage迁移、workflows/ingest、read和幂等原子登记。
区分match、batch、clip、identity segment、feature space和event subject。

验收：跨batch重复提交也幂等；同match补新片段保留旧状态；跨match不误连；
同源同版本被改写拒绝，新版本显式登记；异常不留下半批结果，输入只读。

## M3 初始化与人工档案维护

实现identity/prepare、gallery以及initialize/review，人工确认种子可先跑通。
支持可变人数、未知段、样本批准/拒绝、拆段和稳定player_id。
先将质量风险及分数语义结构化，禁止通用身份规则解析 adapter 的 notes 文本。
v1 人工建立身份锚点和新球员；仅人工批准且归属已确认的模板进入正式图库。

验收：初始化可只发现部分球员，继续更新不重置ID；已有库initialize不清库；
轨迹混人可以隔离和拆分，样本模糊不等于身份错误；人工修改有审计和版本检查。

## M4 增量匹配和模板更新

实现identity/associate、resolve、distance和workflows/update。
初始化与增量共享规则；先用明确种子，再向同场持续导入新证据。
顺序为部位距离 → 多模板/多样本候选及诊断 → 保守决策 → 预览和人工确认闭环。
未标定时保持 auto_assign_enabled=false；完成按片段隔离的校准和验证后，再开放自动确认已有球员。
距离远不自动新建；单候选、缺模板、证据覆盖不足和强支持分歧均保留复核。

验收：已有/新身份提案/待定/拒绝分开；未知不自动建人；新出场球员可经确认加入；
固定图库快照匹配，合格模板随后更新；模型变更不重编号，空间不兼容不比较。
按独立身份标注报告关联精度、召回、误合并和待复核比例，不按固定人数评成绩。

## M5 一般事件关联与当前时间线

实现events/validate、associate、update.ingest_events和共享读取。
支持点/区间、多角色、未知命中结果、事件先到和晚到。

验收：事件-only不执行重识别；先到事件可未关联保存，身份到达后同一事件得到归属；
拆段/合并后查询同步变化；射手和助攻者可不同人；区间跨身份切换保留歧义。
样例68只登记一次进球，20保留3.5秒；有无全场来源映射均能正确定位。

## M6 接口和展示

实现presentation/api、main和static，共享workflows。
展示任意数量档案、模板、事件、统计、来源与复核入口；支持按球员和事件属性筛选。

验收：UI/CLI/导出查询口径一致；受控媒体播放；两场比赛和多批更新不串档；
样例最终经人工身份核验后，可展示预期两人及6/4条给定进球记录。

## 后续独立需求

精剪和合集渲染、自动命中识别、跨比赛永久球员身份、整场容量优化分别立项。
本模块始终提供稳定的身份、事件与媒体定位接口，不要求重新实现上游视频理解模型。

## AI开发节奏

每次给出一个用例、输入和验收行为，先读AGENTS及契约，再修改对应流程模块。
改契约同步所有调用者；必要测试通过后保存Git提交。先跑业务闭环，再接展示。
不以建立更多目录或补满空文件作为功能完成标准。

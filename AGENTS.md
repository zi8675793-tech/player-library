# 项目协作约定

## 目标和当前状态

- 本项目是通用的比赛内球员库与事件管理模块。人数、片段数、导入批次数不固定。
- 同一比赛分批建库和更新，player_id 持久保留；不同比赛默认不自动归并。
- `../test/` 只是一个样例；样例细节仅属于 `adapters/sample_test.py`、examples 和样例测试。
- 当前已实现契约、配置、sample_test 只读适配、核验 CLI 和身份核心算法层 v1（纯诊断与计划）；normalized、业务工作流、持久化和展示尚未实现。不得用固定分组或假成功响应充当实现。
- 开工先读 README 和 docs 下的架构、流水线、数据契约、实施计划。
- 身份算法与归人决策遵循 docs/identity-policy.md；纯函数调用见 docs/identity-core.md。策略尚未经过真实身份校准，不得宣称已完成正式建库。
- 用户新指令优先，日常实现自主推进；本约定不引入额外审批步骤。

## 模块边界

- Python 3.11、FastAPI/Pydantic 2、NumPy、SQLite、原生 HTML/CSS/JS，依赖由 uv.lock 锁定。
- contracts 定义唯一字段、枚举和类型；adapters 只把上游格式转换为标准 EvidenceBatch。
- identity 按 prepare → associate → resolve → gallery 处理身份，产生可审查计划，不写库。
- events 校验事件并按角色解析当前球员；不调用重识别模型，不推断未提供的进球结果。
- workflows 编排初始化、增量、复核与读取。ingest 是共享处理路径；避免复制两套规则。
- storage 负责 SQLite、SQL 和迁移；事务及提交由 workflows 的写用例控制。
- presentation 只接 HTTP/CLI/网页输入并调用 workflows，不重算身份、统计和图库。
- identity 不依赖 events；跨业务组合由 workflows 完成。禁止出现新的万能 service.py。
- settings.py 是配置入口；路径以配置文件所在目录解析，不依赖命令执行目录。
- 禁止从 repository-review 动态导入、改 sys.path 或写入私人绝对路径。

## 必须保持的数据语义

- match_id 是身份作用域；batch_id 只是交接记录，不能作为球员库分区或幂等依据。
- EvidenceBatch 可以只交人物证据或只交事件；字段缺省、空列表均不表示删除旧数据。
- 原始证据与模型合同不可静默改写；新版本显式登记，人工修正不回写输入文件。
- 原始轨迹不等于身份段；有效归属粒度为片段内的轨迹段，支持拆分、未知和其他人物。
- 号码、跟踪质量、模板资格、身份确认和事件结果分别保存，禁止互相当作真值。
- 没有匹配结果不等于新球员；新增身份必须有独立支持或人工确认。
- 点事件与区间事件显式区分；多主体按角色分别解析，不能把所有轨迹强归一个球员。
- 片段时间、原视频时间、比赛计时分开；缺少来源映射时全局时间留空。
- 特征、模板、距离和阈值绑定 feature_space_id；换模型不重置 player_id。
- SQLite 是唯一正式状态源；所有查询经 workflows/read.py，派生 JSON 不反向更新。
- 重复证据跨批次提交也不新增记录或 revision；新增片段不删除旧片段。
- 身份修正后，事件和统计根据当前映射重新解析；事件先到允许保存为未关联。
- 人工确认优先；自动重匹配不覆盖确认、拒绝、拆段或样本复核。
- 模板更新使用最终决策，当前批匹配读取固定图库快照，防止本批错误串联。
- v1 人工种子建档、新建和拆段；自动归人仅在策略校准后开启。匹配只使用人工批准的有效模板，自动归人不自动批准模板。
- 写入核对 expected_revision，证据、映射、图库计划、revision 和审计原子提交。
- 资源通过 asset_id 提供，禁止任意本机路径读取接口。

## 修改和交付

- 一次实现一个可验收行为；改契约时同步调用者、文档和必要测试。
- 改库增加 storage/migrations 中的迁移，不以删库重建处理升级。
- 改依赖同时更新 pyproject.toml 和 uv.lock；改策略集中在 identity 和配置。
- 新输入格式增加 adapter，通用身份/事件模块不得识别 target/、CSV 列名或样例号码。
- 样例特征 [N,6,128] 不是通用维度常量；支持范围由特征空间和后端决定。
- 不预设一次初始化能发现全场所有球员，也不以固定人数作为聚类规则。
- 默认只读 ../test/ 和 ../repository-review/；生成产物写 var/，样例配置放 examples/。
- 避免为纯文案改动增加镜像测试；业务测试覆盖规则、失败路径和跨模块行为。
- 检查：uv sync --locked；uv run ruff check .；uv run ruff format --check .。
- 有业务测试后运行 uv run pytest。报告真实完成状态，不把给定标注当作识别成绩。

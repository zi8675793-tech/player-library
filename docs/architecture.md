# 按业务流水线组织代码

## 范围与状态

通用的比赛内球员档案、增量身份关联、图库维护、事件关联与查询模块。
当前已实现 contracts 的 Pydantic 2 模型和 settings 的 TOML 读取、类型与路径校验。
sample_test 的只读读取、文件/数组校验与 validate CLI 已实现。
身份核心算法层 v1 已实现纯样本准备、KPR 距离、候选诊断、归人决策和图库变更计划。
normalized 输入适配、业务编排、数据库与页面仍为职责骨架，尚未实现。
测试样例的片段数、号码、人数和事件类别不属于通用业务规则。
身份算法与决策以 [identity-policy.md](identity-policy.md) 的 v1 规范为准；调用见 [identity-core.md](identity-core.md)，真实身份校准尚未完成。

## 结构

```text
app/
  main.py                     # 组装配置、存储、用例和HTTP应用
  settings.py                 # 统一配置和路径
  contracts/                  # 内部统一语言
    common.py                 # 标量、来源键与半开时间/帧区间
    evidence.py               # EvidenceBatch、媒体、样本、特征空间
    identity.py               # Player、Segment、Assignment、Template
    events.py                 # 点/区间事件、按角色区分的主体
    api.py                    # 请求、响应、复核命令
  adapters/                   # 上游文件格式边界
    _reading.py               # 根目录约束、只读读取、来源文件与记录指纹
    normalized.py             # 标准证据接口
    sample_test.py            # 当前test文件夹的专用转换
  identity/                   # 人物身份分支：无数据库写入
    prepare.py                # 样本质量、身份段、可比较性
    associate.py              # 与已有图库比较，生成候选
    resolve.py                # 已有/新建/待定/拒绝决策
    gallery.py                # 模板增加、替换、退出计划
    distance.py               # 特征空间对应的距离，初期实现KPR
  events/                     # 事件分支：不重新跑动作模型
    validate.py               # 点/区间、结果、来源与主体校验
    associate.py              # 将每个事件主体解析到当前球员
  workflows/                  # 可调用用例，也是事务边界
    ingest.py                 # 共同接入与处理路径
    initialize.py             # 首次建立身份锚点与图库
    update.py                 # 增量身份更新、独立事件导入
    review.py                 # 人工维护与审计
    read.py                   # 统一档案、事件、统计、资源查询
  storage/
    database.py               # SQLite连接和迁移执行
    repositories.py           # 参数化SQL，不做身份决策
    migrations/               # 增量数据库迁移
  presentation/
    api.py                    # HTTP → workflows
    cli.py                    # 只读validate → adapter；后续业务命令 → workflows
    static/                   # 网页只消费API结果
examples/                     # 样例配置与样例说明
tests/                        # 通用规则、工作流和adapter测试
docs/                         # 流水线、架构、契约、实施计划
var/                          # 不提交的数据库、预览、日志和导出
```

这些目录对应数据语言、输入边界、两个业务分支、编排、持久化和展示。
不再使用含全部业务的 service.py；不额外复制另一套查询层。

contracts 只校验提供对象中的局部事实，不读取文件或数据库；目标引用可尚未到达。
settings 使用 Python 3.11 的 tomllib，以配置文件目录解析相对路径，加载不创建运行资源。
sample_test adapter 检查文件真实内容、允许数据根、特征合同与索引；
后端可执行性由后续 identity 检查，读取现有特征不运行模型；
跨批幂等、历史有效段重叠和别名环由 workflows/storage 在正式状态上检查。

## 依赖和事务

```text
presentation → workflows → adapters / identity / events / storage
adapters / identity / events / storage → contracts
main → settings + 各组件装配
```

底层不反向调用 workflows。identity 与 events 无相互依赖，跨领域组合由 workflows 完成。
identity 输出计划而不提交，events.associate 接收当前映射上下文而不擅自修改身份。
HTTP、CLI 和后续剪辑导出均通过 workflows；前端不再统计或认人。
只读 `validate` 是输入边界核验，直接调用 settings 和 adapter，不执行业务用例或写入状态。

写流程在一致的图库/身份版本下计算计划，事务内再次核对 expected_revision；
证据、决策、图库、修订与审计一次提交。storage 辅助函数不得自行 commit。
发生并发版本变化时返回冲突，重读后重新计划，不接受过期候选。

## 数据和规模

SQLite 作为第一版正式状态源，保存按 match_id 隔离的比赛、原始证据、身份段、
档案、各特征空间图库、事件主体和修订审计。batch 记录交接，不隔离球员。
向量可用带 shape/dtype 的BLOB保存；图片与视频保存受控资源引用和哈希。
JSON、网页缓存与导出清单都是派生结果。

数据库迁移只能改变存储实现，不改变 player_id。模型升级新增特征空间或显式迁移，
不靠新建比赛或重编号球员解决。无可比较模板时返回未知。
不保证任意规模吞吐；实现后通过整场基准测量容量，再决定索引或存储优化。
业务契约不设十片段、两人、固定部位数等样例上限。

## 目标用例入口

| 入口 | 作用 |
| --- | --- |
| initialize.initialize_library | 尚无初始身份锚点的比赛库建档；可已存在事件或待定证据 |
| update.update_library | 同场新人物证据和可选事件的增量处理 |
| update.ingest_events | 只有事件的提交，允许依赖人物证据尚未到达 |
| review.apply_review | 确认、拆分、合并、模板复核等变更 |
| read.get_players / get_events / get_review_queue | 当前档案、主体归属、统计和待复核查询 |

这些是待实现的目标接口名称，首次实现时在contracts和调用者中保持一致。
候选预览调用相同身份规则但不写库；接受经有类型的review命令并核对版本。
API围绕match而非样例dataset：/api/matches/{match_id}/players、/events、/reviews、
/imports、/initialize、/matching/preview；资源为/api/assets/{asset_id}。

## 技术与决策记录

- Python 3.11 + FastAPI/Pydantic 2 + NumPy + 标准库sqlite3 + 原生网页；uv.lock锁定依赖。
- 初期不引入消息队列、动态插件框架、独立向量库或跨比赛永久身份系统。
  独立人物/事件导入不依赖异步基础设施。
- 2026-11-03：将上一版样例原型定位修正为通用模块，样例布局仅由adapter处理。
  将平铺职责骨架调整为上述分支与工作流；没有业务实现需要迁移。
- 2026-11-03：实现契约与配置层。特征空间描述与文件校验分离，原始事件与当前主体解析分离；
  只作局部校验以允许分批证据和先到事件，适配器与持久化仍按后续里程碑实现。
- 2026-11-03：实现 sample_test 只读输入边界和核验 CLI；复用现有模型与 load_settings。
  文件引用解析后必须在 input_root 内，NPZ 禁止 pickle，逐行核对 JSONL、数组和图片哈希；
  来源文件哈希与记录指纹分开，业务 ID 和特征空间 ID 不依赖批次或遍历顺序。
  为保留来源定位、实际观测帧、事件级号码和警告作最小契约补充，不引入 normalized 新协议。
- 2026-11-03：确定身份策略 v1：人工种子、KPR 部位距离、双模板均值与段内中位数、
  覆盖/支持/分歧门槛、人工新建与图库批准。阈值经独立身份标注校准前不启用自动确认。
  当时只确定设计，不表示数据库或事件关联已实现。
- 2026-11-04：实现身份核心算法层 v1；复用原有身份模型，增加结构化风险、精确空间策略、
  校准指纹、固定图库快照和逐样本诊断。纯算法不读文件、不写库、不生成正式 player_id，
  模板准入与身份确认独立；同批匹配不扩张图库。实现与合成测试不表示完成真实数据校准。
- 上游参考33613/basket_cut，提交a7fb52ffbd92b22c2dc5d02f4e8376ca5c6b6880。
  参考pipeline/identity/cross_clip.py的KPR公式，新项目按已确定公式独立实现并用合成数据验证，未复制上游代码。
  复制前登记许可证和对应测试；运行不能依赖审查快照。

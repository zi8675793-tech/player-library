# 阶段三 任务卡：流水线组装与 SQLite 版本化原子存储 (T3)

请直接阅读并实施，完成端到端流水线接入、数据库持久化与中断重放测试。

## 1. 任务背景与目标
将 T1 特征提取器与 T2 增量匹配引擎完整组装至主执行工作流（`app/workflows/incremental.py`），实现多片段分批到达场景下的强一致性状态维护：
1. **SQLite 强事务隔离**：`BEGIN IMMEDIATE` 锁定写事务，版本核对推进至 Revision $R+1$；
2. **留一法版本化重估**：在新证据到达时，对历史机器暂定段进行留一法复核，支持版本化重分配；
3. **幂等重放与中断续跑**：支持异常中断后通过输入指纹无损恢复，避免脏写。

## 2. 职责边界与规范
- **允许修改/新建**：
  - `app/workflows/incremental.py`
  - `app/storage/`
  - `tests/test_incremental_pipeline_e2e.py`
- **严禁修改**：
  - `docs/data-contracts.md`
  - `app/features/`
  - `app/identity/`

## 3. 验收要求
1. 端到端集成测试 `uv run pytest tests/test_incremental_pipeline_e2e.py` 全部通过；
2. 验证多片段顺序输入（如 3 -> 20 -> 43 -> 50 -> 6 -> 68）后数据库状态完整；
3. 验证幂等性：重复提交同一批次返回 `changed=False`，Revision 保持不变；
4. 产物留痕于 `var/runs/run-<timestamp>/` 并生成 `handoff.txt`。

# 比赛球员库与事件管理系统 (Player Library)

本项目专为**篮球比赛视频**设计，实现比赛内球员身份管理、多视角图库增量建库、跨片段时序追踪与技术事件关联。

---

## 核心算法架构总览

针对篮球比赛中**同队球员完全同色、高速激烈对抗、多人挤压遮挡、视频分批到达、零人工标签**的核心挑战，本项目采用以下现代化闭环算法架构：

1. **双流特征提取 (Dual-Stream Feature)**：
   - **强物理证据流 (Jersey Number OCR / JNR)**：从人体切片中自动检测与识别球衣号码（如 27、10、15），实现同队球员在物理层面的绝对剥离（一票否决/一票通过）。
   - **外观表征流 (Visual Appearance ReID)**：提取 512 维姿态感知深度外观特征向量。
2. **时空硬排他约束网络 (Spatio-Temporal Cannot-Link Engine)**：
   - 物理不可能定理：同一时间、同一摄像机视角下出现的不同人物轨迹，距离再近也绝对不可能是同一人（边权重设为 $+\infty$），杜绝同帧混人。
3. **多原型动态图库与增量三态决策 (Multi-Prototype Incremental Gallery)**：
   - 弃用“单均值向量”，每个球员维护多个高质量视角原型（正面、背面带号码、侧面）。
   - 增量三态决策：`matched_existing`（归入已有档案）、`created_new_identity`（多片段独立见证建立新档案）、`insufficient_evidence`（待定延期，防止碎片化）。
4. **版本化原子存储与审计 (SQLite Revision R)**：
   - 强事务与幂等中断续跑，不可变审计日志。
5. **防火墙独立黄金评测 (Evaluation Harness)**：
   - 推理阶段绝对严禁读取任何人工标注（`annotations.csv` / `target/`）。
   - 预测和几何对应冻结后，由独立评测防火墙在固定 70 分母上独立打分。

---

## 核心文档导航

| 文档 | 作用 |
| :--- | :--- |
| [AGENTS.md](AGENTS.md) | **项目协同指南**：SSOT、架构分层与常用验证命令 |
| [docs/current-state.md](docs/current-state.md) | **唯一事实来源 (SSOT)**：当前最新进展、实测基准跑分与状态 |
| [docs/architecture.md](docs/architecture.md) | **系统总体架构**：5 层分层拓扑与核心交互设计 |
| [docs/pipeline.md](docs/pipeline.md) | **业务流水线规范**：端到端 8 阶段时序处理规范 |
| [docs/data-contracts.md](docs/data-contracts.md) | **数据契约规范**：冻结的强类型数据模型与 Schema |
| [docs/implementation-plan.md](docs/implementation-plan.md) | **实施计划**：四阶段目标、实现要点与验收命令 |
| [docs/evaluation-harness.md](docs/evaluation-harness.md) | **独立评测防火墙**：防作弊独立打分与 70 分母验收标准 |
| [docs/storage.md](docs/storage.md) | **SQLite 存储规范**：数据库表结构、迁移与 Revision R 规范 |

---

## 数据准备（仓库不含比赛数据）

本仓库**只包含算法代码与文档**，以下内容不随仓库分发：

- 上游比赛视频与轨迹数据（`test/`）、人工标注（`annotations.csv`）；
- 全部实验运行产物与数据库（`var/`，已 gitignore）；
- 评测答案（`test/target/`）——按评测防火墙纪律，推理代码严禁读取。

**无数据也可运行**：`uv run pytest tests/` 中的单元测试全部基于合成夹具；依赖真实运行证据的审计测试在证据缺失时自动跳过。

**接入真实数据**：将上游数据目录放置为仓库同级的 `../test/`（即与 `player-library/` 平级），目录契约如下：

```text
test/
├── video/<片段ID>.mp4              # 原始比赛视频
└── track/<片段ID>/
    ├── tracks.jsonl                # 逐帧人物框：video_id、track_id、frame_idx、timestamp_s、bbox_xyxy、score
    ├── video_meta.json             # 帧率、画面尺寸、帧数；input_path 相对于 test/
    └── samples/*.png               # 人物裁剪图（原图尺寸）
```

坐标与时间约定：`frame_idx` 从 0 开始；`timestamp_s` 为片段内秒数；`bbox_xyxy` 为原画面像素坐标。真实数据批处理与评测入口见 `run_real_batches.py`、`run_e2e_benchmark.py` 与 [docs/evaluation-harness.md](docs/evaluation-harness.md)。

**模型权重**：OSNet x1.0 (SportsMOT) 预训练权重已随仓库提供（`models/osnet_x1_0_sportsmot.pth`）；EasyOCR 的识别权重在首次运行时自动下载。

---

## 环境与常规命令

项目基于 Python 3.11 与 `uv` 管理：

```bash
# 安装依赖
uv sync

# 运行单元测试
uv run pytest

# 代码风格与质量检查
uv run ruff check .
uv run ruff format --check .
```

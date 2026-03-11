# RL Quantitative Trading Agent (A-Share)
# 基于强化学习的量化交易智能体

> 🚧 Work in Progress / 施工中
>
> This repository is currently under active development for my undergraduate thesis.
> 本项目为我的本科毕业论文代码仓库，目前正在持续开发和更新中。

---

## Introduction / 项目简介

Welcome! This is the repository for my undergraduate thesis: "Design and Implementation of a Reinforcement Learning-based Quantitative Trading Agent".

In simple terms, the current thesis direction is a deployable A-share prototype for retail-scale capital.  
The strategy now uses a fixed 40-stock universe and a `Top-K` selection policy (`K=5`), rebalancing every 5 trading days.
Instead of only predicting prices, the agent learns decision-making and position updates under practical market constraints.

欢迎！这是我的本科毕业论文《基于强化学习的量化交易智能体设计与实现》的代码仓库。

简单来说，当前毕设方向已调整为面向普通投资者资金规模的可部署A股原型系统。  
策略采用固定40只股票池 + `Top-K` 选股（`K=5`），每5个交易日调仓一次。  
与传统股价预测不同，项目核心是让智能体在动态市场中进行决策与仓位更新。

## Key Features / 核心内容

* Market: Chinese A-share market (daily data, T+1 constraints).
  * 市场：聚焦中国A股（日频数据，考虑T+1等交易约束）。
* Universe: Fixed 40-stock pool selected for liquidity, stability, and retail affordability.
  * 股票池：固定40只，兼顾流动性、稳定性与小资金可参与性。
* Strategy: `Top-K` rotation (`K=5`), rebalance every 5 trading days.
  * 策略：`Top-K` 轮动（`K=5`），每5个交易日调仓。
* Pipeline: Independent data fetch + processing script, then training/backtest pipeline.
  * 流程：数据抓取与清洗独立脚本化，训练与回测分离执行。
* Data Source: Tushare API.
  * 数据源：Tushare API。

## Latest Update / 最新进度
- Completed 4-year OOS evaluation (2022-2025) and rolling retrain experiments with 10 seeds; results and plots saved under docs/experiments.
  完成2022-2025四年OOS评估与10个随机种子的滚动重训实验，结果与图表已归档至docs/experiments。
- Selected the best-performing rolling model (seed7) as the candidate for upcoming live demo stage.
  选定滚动重训中表现最佳的seed7模型，作为后续实机演示候选。

## Project Structure / 项目结构

Here is how the project is organized. This structure is designed to separate configuration, core logic, and experimental results.
这里是项目结构安排。项目采用模块化结构设计，将配置、核心逻辑与实验结果分离，以便复现和扩展。
## Structure Updated in 2026-03-11 / 2026-03-11 最新项目结构快照
Current snapshot (aligned with fixed 40-stock pool and Top-K workflow):
当前结构（已对齐固定40股与Top-K流程）：

```text
RL_Quant_Trading/
├─ configs/                               # Global/project-level configs / 全局与项目配置
│  ├─ base_config.py                      # Paths, dates, indicators / 路径、时间窗、指标配置
│  ├─ agent/                              # Agent-related configs / 智能体参数配置
│  │  └─ ppo.py                           # PPO hyperparameters / PPO超参数
│  └─ experiment/                         # Experiment orchestration configs / 实验流程配置
│     └─ base_experiment.py               # Run switches and output dirs / 开关与输出目录
├─ data/                                  # Data assets / 数据资产
│  ├─ raw/                                # Raw market data / 原始行情数据
│  │  ├─ 40_pool.csv                      # Fixed 40-pool daily data / 固定40池日线数据
│  │  ├─ 40_pool_universe.csv             # Universe metadata / 股票池元数据
│  │  └─ China_A_share_Real_Data.csv      # Legacy raw file / 历史原始文件
│  ├─ processed/                          # Processed training data / 处理后训练数据
│  │  ├─ processed_40_pool.csv            # Current processed dataset / 当前处理后数据
│  │  └─ processed_data.csv               # Legacy processed file / 历史处理后文件
│  └─ snapshots/                          # Frozen data snapshots / 数据快照
│     └─ stage1_asof_2021-12-31/
│        ├─ manifest.md                   # Snapshot metadata / 快照元信息
│        └─ processed_data.csv            # Snapshot file / 快照数据文件
├─ docs/                                  # Reports and archives / 报告与历史归档
│  ├─ experiments/                        # Current run outputs / 当前实验输出
│  └─ experiments_legacy/                 # Historical experiment archives / 历史实验归档
│     ├─ 20260129_0014_base_experiment/
│     ├─ 20260303_2300_base_experiment/
│     ├─ 20260303_2311_base_experiment/
│     ├─ 20260303_2329_base_experiment/
│     ├─ 20260304_1559_base_experiment/
│     └─ 20260304_1603_base_experiment/
├─ experiments/                           # Stage-oriented experiment workspace / 分阶段实验工作区
│  ├─ stage1_primary_offline/
│  │  ├─ runs/                            # Per-run artifacts / 每次运行产物
│  │  ├─ summaries/                       # Aggregated summaries / 汇总统计结果
│  │  └─ registry.csv                     # Run registry / 运行登记表
│  ├─ stage2_long_oos/
│  ├─ stage3_rolling_2025/
│  └─ stage4_demo_2026/
├─ hyperparam_search/                     # Hyperparameter search workspace / 超参数搜索预留
├─ online/                                # Online module workspace / 在线模块预留
├─ src/                                   # Source code / 核心源码
│  ├─ data_processing/
│  │  └─ fetch_40_pool.py                 # Fetch + clean fixed 40-pool data / 固定40池抓取与清洗
│  ├─ envs/
│  │  └─ env_stocktrading.py              # Trading environment / 交易环境
│  ├─ models/                             # Model definitions (reserved) / 模型定义预留
│  ├─ training/
│  │  └─ train_agent.py                   # Train + backtest logic / 训练与回测逻辑
│  └─ utils/                              # Utility functions (reserved) / 工具函数预留
├─ .gitignore                             # Git ignore rules / Git忽略规则
├─ Development_log.md                     # Development notes / 开发记录
├─ exp_main.py                            # Main pipeline entry / 主流程入口
├─ test_oos.py                             # Stage 2 OOS evaluation / 阶段2长期OOS评估
├─ test_rolling.py                         # Stage 3 rolling retrain / 阶段3滚动重训
├─ selected_40_pool.xlsx                  # Selected 40-stock universe source / 40池选股源文件
├─ README.md                              # Project documentation / 项目文档
└─ requirements.txt                       # Python dependencies / Python依赖

```

## Stage Documentation Framework / 分阶段说明文档框架
### Stage 1: Primary Offline Test / 阶段1：离线主实验
- `Objective / 目标`:
  Validate cross-regime generalization on A-share market.
  在A股不同市场状态下验证策略泛化能力。
- `Data Scope / 数据范围`:
  Train `2010-01-01 ~ 2018-12-31`, Test `2019-01-01 ~ 2021-12-31`.
  训练与测试严格按上述时间切分。
- `Protocol / 实验协议`:
  Fixed split, multi-seed runs, unified metrics and benchmark.
  固定切分、多随机种子、统一指标与基准口径。
- `Outputs / 输出物`:
  Model checkpoints, account value table, action/trade/holding logs, comparison plots.
  模型权重、净值表、动作交易持仓表、对比图。
- `Acceptance Criteria / 通过标准`:
  Risk-adjusted metrics are stable across seeds; no data leakage.
  多种子风险调整指标稳定，且无未来信息泄露。

### Stage 2: Long Pure OOS / 阶段2：长期纯OOS验证
- `Objective / 目标`:
  Provide external validity on unseen period.
  在完全未见区间给出外部真实性证据。
- `Data Scope / 数据范围`:
  `2022-01-01 ~ 2025-12-31`.
- `Rules / 规则`:
  No model redesign, no reward tuning, no hyperparameter tuning.
  不改模型结构、不改奖励函数、不调超参数。
- `Outputs / 输出物`:
  OOS report, metric summary, failure-case notes.
  OOS评估报告、指标汇总、失败案例记录。

### Stage 3: Rolling Retrain / 阶段3：滚动重训验证
- `Objective / 目标`:
  Verify deployment feasibility under rolling update mechanism.
  验证滚动更新机制下系统可部署性。
- `Data Scope / 数据范围`:
  Rolling OOS over `2022-2025`, yearly windows with trailing 3-year training span.
  2022-2025滚动样本外评估，每年滚动一次，训练窗口为最近3年。
- `Protocol / 协议`:
  Multi-seed rolling retrain for robustness; keep logs and NAV continuity.
  多随机种子滚动重训验证稳健性，记录日志并拼接连续净值曲线。
- `Outputs / 输出物`:
  Quarterly model versions, deployment logs, rolling NAV path.
  季度模型版本、部署日志、滚动净值轨迹。
- `Boundary / 边界`:
  Rolling result is for engineering validation, not significance claim.
  滚动结果用于工程验证，不用于统计显著性结论。

### Stage 4: Live Demo / 阶段4：实时演示
- `Objective / 目标`:
  Demonstrate daily data input and strategy output loop.
  展示“每日输入数据-输出策略”闭环。
- `Protocol / 协议`:
  Use latest model trained from historical data before demo window.
  使用演示窗口前训练出的最新模型。
- `Outputs / 输出物`:
  Daily strategy log, realized paper-trading NAV, operation records.
  每日策略日志、模拟净值、运行记录。
- `Boundary / 边界`:
  Demo is not used for back-adjusting model design.
  演示结果不得反向用于模型设计调整。

# RL Quantitative Trading Agent (A-Share)
# 基于强化学习的量化交易智能体

> 🚧 Work in Progress / 施工中
>
> This repository is currently under active development for my undergraduate thesis.
> 本项目为我的本科毕业论文代码仓库，目前正在持续开发和更新中。

---

## Introduction / 项目简介

Welcome! This is the repository for my undergraduate thesis: "Design and Implementation of a Reinforcement Learning-based Quantitative Trading Agent".

In simple terms, I'm training AI agents (using RL) to find profitable trading strategies within the complex environment of the Chinese A-share market. Instead of just predicting prices, the agent is learning how to make decisions (buy/sell/hold) to optimize portfolio value over time.

欢迎！这是我的本科毕业论文《基于强化学习的量化交易智能体设计与实现》的代码仓库。

简单来说，我正在尝试利用强化学习（RL）方法，训练一个能在中国A股市场中制定交易策略的AI智能体。与传统股价预测不同，这个项目的核心在于让智能体学会在动态市场环境中做决策（买入、卖出或持有），从而实现资产增值。

## Key Features / 核心内容

* Market: Chinese A-Share market (T+1 trading rule, price limits, etc).
    * 市场：聚焦中国A股（考虑T+1、涨跌停限制等特有规则）。
* Method: Deep Reinforcement Learning (PPO algorithm focus).
    * 方法：深度强化学习（重点使用PPO等算法）。
* Evaluation: Focusing on Online Trading simulation, not just backtesting on training data.
    * 评估：重点在于模拟实盘决策，而不仅仅是历史数据回测。
* Data Source: Tushare API (Data is not included in this repo).
    * 数据源：Tushare API（原始数据文件不包含在仓库中）。

## Project Structure / 项目结构

Here is how the project is organized. This structure is designed to separate configuration, core logic, and experimental results.
这里是项目结构安排。项目采用模块化结构设计，将配置、核心逻辑与实验结果分离，以便复现和扩展。
## Structure Updated in 2026-03-03
Updated project structure snapshot (focus on the latest experiment outputs):
最新项目结构快照：
```text
RL_Quant_Trading/
├─ configs/
│  ├─ agent/ppo.py                        # PPO超参数配置
│  ├─ env/stock_env.py                    # 环境参数配置
│  └─ experiment/base_experiment.py       # 实验流程与目录管理
├─ data/
│  ├─ raw/China_A_share_Real_Data.csv     # 原始行情数据
│  └─ processed/processed_data.csv        # 清洗后训练/回测数据
├─ docs/
│  └─ experiments/
│     ├─ 20260303_2311_base_experiment/
│     └─ 20260303_2329_base_experiment/
│        ├─ checkpoints/
│        │  └─ ppo_agent_base.zip         # 模型权重
│        ├─ logs/                         # 训练日志
│        ├─ plots/
│        │  └─ performance_comparison.png # 策略与基准净值对比图
│        └─ tables/
│           ├─ account_value.csv          # Agent净值序列
│           ├─ benchmark_account_value.csv # 公平口径Benchmark净值序列
│           ├─ backtest_actions.csv       # 回测动作明细
│           ├─ backtest_trades.csv        # 回测成交与费用明细
│           └─ backtest_holdings.csv      # 回测持仓轨迹
├─ src/
│  ├─ data_processing/processor.py        # 数据抓取与特征工程
│  ├─ envs/env_stocktrading.py            # 自定义交易环境
│  └─ training/train_agent.py             # 训练与回测主逻辑
├─ exp_main.py                            # 实验主入口
├─ requirements.txt
└─ Development_log.md
```


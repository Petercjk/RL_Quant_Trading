# RL Quantitative Trading Agent (A-Share)
# 基于强化学习的量化交易智能体

> 🚧 Work in Progress / 施工中**
>
> This repository is currently under active development for my undergraduate thesis.
> 本项目为我的本科毕业论文代码库，目前正在持续开发和更新中。

---

## Introduction / 项目简介

Welcome! This is the repository for my undergraduate thesis: "Design and Implementation of a Reinforcement Learning-based Quantitative Trading Agent".

In simple terms, I'm training AI agents (using RL) to find profitable trading strategies within the complex environment of the Chinese A-share market. Instead of just predicting prices, the agent is learning how to make decisions (buy/sell/hold) to optimize portfolio value over time.

欢迎！这是我的本科毕业论文《基于强化学习的量化交易智能体设计与实现》的代码仓库。

简单来说，我正在尝试利用强化学习（RL）方法，训练一个能在中国A股市场里制定策略的AI智能体。与传统股价预测不同，这个项目的核心在于训练智能体学会在动态的市场环境中做决策（买入、卖出或持仓），从而实现资产增值。

## Key Features / 核心内容

* Market: Chinese A-Share market (T+1 trading rule, price limits, etc).
    * 市场： 聚焦中国 A 股（考虑 T+1、涨跌停限制等特有规则）。
* Method: Deep Reinforcement Learning (PPO algorithm focus).
    * 方法： 深度强化学习（PPO 算法等）。
* Evaluation: Focusing on Online Trading simulation, not just backtesting on training data.
    * 评估： 重点在于模拟实盘决策，不仅仅是历史数据的简单回测。
* Data Source: Tushare API (Data is not included in this repo).
    * 数据源：Tushare API（原始数据文件不包含在仓库中）。

## Project Structure / 项目结构

Here is how the project is organized. This structure is designed to separate configuration, core logic, and experimental results.
这里是项目结构安排。项目采用了模块化结构设计，将配置、核心逻辑与实验结果分离，以便于复现和扩展。

```text
RL_Quant_Trading/
│
├── configs/                   # Configuration files (YAML)
│   ├── env/                   # Environment settings (transaction cost, window size)
│   ├── agent/                 # Agent hyperparameters (PPO, A2C, etc.)
│   └── experiment/            # Experiment protocols
│
├── data/                      # Data storage (Ignored by Git)
│   ├── raw/                   # downloaded from Tushare
│   └── processed/             # Cleaned and feature-engineered data
│
├── src/                       # Core Source Code
│   ├── agents/                # RL Agent implementations
│   ├── envs/                  # Custom Trading Environments (Gym-compatible)
│   ├── models/                # Neural Network architectures (PyTorch)
│   ├── utils/                 # Utilities (Logger, Seed, etc.)
│   └── data_processing/       # Data cleaning and feature engineering scripts
│
├── train/                     # Training Scripts
│   └── train_agent.py         
│
├── online/                    # Online / Rolling Prediction (Core)
│   ├── online_trader.py       # Simulating real-world trading decisions
│   └── rolling_test.py        # Walk-forward analysis
│
├── hyperparam_search/         # Hyperparameter Tuning
│
├── experiments/               # Experiment Results (Logs, Plots, Checkpoints)
│
└── docs/                      # Documentation & Notes
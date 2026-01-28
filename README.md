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
├── configs/                   # 配置文件 (YAML)
│   ├── agent/                 # 智能体参数配置 (PPO, A2C 等)
│   │   └── __pycache__/      
│   ├── env/                   # 环境配置 (交易成本、窗口大小等)
│   │   └── __pycache__/      
│   ├── experiment/            # 实验配置(每次训练为单独一次实验)
│   │   └── __pycache__/      
│   └── __pycache__/           #
│
├── data/                      # 数据存储 (已在 .gitignore 中忽略)
│   ├── raw/                   # 原始数据 (Tushare 下载)
│   └── processed/             # 清洗和特征工程后的数据
│
├── docs/                      # 文档与笔记
│   └── experiments/           # 实验结果数据 (已在 .gitignore 中忽略)
│       └── 20260129_0014_base_experiment
│           ├── checkpoints   # 模型检查点
│           ├── logs          # 日志
│           │   └── PPO_1     # PPO 单独训练日志
│           ├── plots         # 绘图
│           └── tables        # 表格数据
│
├── hyperparam_search/         # 超参数搜索脚本和结果（未开始）
│
├── online/                    # 在线/滚动预测（未开始）
│   ├── online_trader.py       # 模拟实时交易决策
│   └── rolling_test.py        # 滚动回测分析
│
└── src/                       # 核心源代码
    ├── data_processing/       # 数据清洗和特征工程脚本
    │   └── __pycache__/      
    ├── envs/                  # 自定义交易环境（兼容Gym，参考FinRL定义）
    │   └── __pycache__/      
    ├── models/                # 神经网络模型（未开启）
    ├── training/              
    │   └── __pycache__/      
    └── utils/                 # 工具函数 (日志记录、随机种子等)

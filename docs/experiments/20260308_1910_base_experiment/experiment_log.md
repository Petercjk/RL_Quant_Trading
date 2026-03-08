# Experiment Log: 20260308_1910_base_experiment

## 1. 实验描述
- **task_name**: A-Share Base Training & Backtest
- **description**: 基础流程：数据对齐、PPO训练、2019-2021回测及结果绘图对比
- **indicators**: Standard FinRL Indicators (MACD, RSI, etc.)
- **benchmark**: Equal Weight + Top5 Momentum (with transaction costs)

## 2. 任务流配置
- do_preprocessing: DISABLED
- do_training: ENABLED
- do_backtesting: ENABLED
- do_plotting: ENABLED

## 3. 核心配置与超参数 (Hyperparameters)
- **Train_Window**: `2010-01-01 to 2018-12-31`
- **Test_Window**: `2019-01-01 to 2021-12-31`
- **Initial_Amount**: `10000`
- **Transaction_Costs**: `Buy: 0.001, Sell: 0.001`
- **PPO_Params**: `{'n_steps': 2048, 'batch_size': 256, 'learning_rate': 0.0003, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01, 'vf_coef': 0.5, 'max_grad_norm': 0.5, 'device': 'cpu', 'verbose': 1}`
- **Total_Timesteps**: `50000`
- **Rebalance_Window**: `5`
- **Top_K**: `5`

## 3. 回测性能指标
- **Seed_10_夏普比率**: 0.4276 (回撤: 22.84%) 
- **Seed_20_夏普比率**: 0.9966 (回撤: 20.11%) 
- **Seed_30_夏普比率**: 0.6067 (回撤: 18.64%) 
- **Seed_40_夏普比率**: 0.7976 (回撤: 15.87%) 
- **Seed_50_夏普比率**: 0.5370 (回撤: 21.36%) 
- **Seed_60_夏普比率**: 0.8751 (回撤: 20.84%) 
- **Seed_70_夏普比率**: 0.5497 (回撤: 22.30%) ⭐ (本轮代表)
- **Seed_80_夏普比率**: 0.4399 (回撤: 21.03%) 
- **Seed_90_夏普比率**: -0.2353 (回撤: 25.86%) 
- **Seed_100_夏普比率**: 0.7655 (回撤: 17.06%) 
- **---**: ---
- **汇总_RL_平均夏普**: 0.5760 ± 0.3245
- **汇总_RL_平均最大回撤**: 20.59%
- **汇总_选取基准模型**: Seed 70 (夏普 0.5497，最接近均值)
- **Top5动量Benchmark_期末总资产**: 14099.82
- **Top5动量Benchmark_年化收益率**: 12.59%
- **Top5动量Benchmark_夏普比率**: 0.5235
- **Top5动量Benchmark_最大回撤**: 26.64%
- **Top5动量Benchmark_年化波动率**: 33.23%
- **Top5动量Benchmark_累计换手率**: 11746.61%
- **Top5动量Benchmark_累计交易成本/初始资金**: 12.94%

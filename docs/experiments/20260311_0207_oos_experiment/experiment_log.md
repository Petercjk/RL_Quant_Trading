# Experiment Log: 20260311_0207_oos_experiment

## 1. 实验描述
- **task_name**: A-Share Long OOS Backtest
- **description**: 长期纯样本外测试：2022-01-01 ~ 2025-12-31
- **indicators**: Standard FinRL Indicators (MACD, RSI, etc.)
- **benchmark**: Equal Weight + Top5 Momentum (with transaction costs)

## 2. 任务流配置
- do_preprocessing: DISABLED
- do_training: DISABLED
- do_backtesting: ENABLED
- do_plotting: ENABLED

## 3. 核心配置与超参数 (Hyperparameters)
- **OOS_Window**: `2022-01-01 to 2025-12-31`
- **Model_Path**: `D:/AAA_Petercjk/RL_Quant_Trading\docs\experiments\20260308_1910_base_experiment\checkpoints\ppo_agent_median_20260308_1928_seed70.zip`
- **PPO_Params**: `{'n_steps': 2048, 'batch_size': 256, 'learning_rate': 0.0003, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.01, 'vf_coef': 0.5, 'max_grad_norm': 0.5, 'device': 'cpu', 'verbose': 1}`
- **Total_Timesteps**: `50000`
- **Environment_Core**: `Top_K=5, Rebalance=5 days, Lot_Size=100`
- **Reward_Shaping**: `Risk_Penalty=0.1, Turnover_Penalty=0.01, Scaling=1.0`
- **Trading_Costs**: `Buy: 0.001, Sell: 0.001, Initial_Amount: 10000`
- **State_Features**: `[40 cols] macd, rsi_30, cci_30, dx_30, boll_ub, boll_lb, close_30_sma, close_60_sma, log_return, return_20, momentum_60, momentum_120, momentum_252, volatility_20, volatility_annual, sma_5, sma_10, sma_20, sma_30, sma_60, bias_20, bias_60, amplitude, intraday_return, amount_20_mean, volume_20_mean, volume_ratio, high_20, low_20, position_20, turbulence, open, high, low, close, pre_close, change, pct_chg, volume, amount`
- **Market_Features**: `market_return, market_volatility_20, market_turbulence`
- **Processed_File**: `D:/AAA_Petercjk/RL_Quant_Trading\data\processed\processed_40_pool.csv`

## 4. 回测性能指标
- **RL_OOS_期末总资产**: 8777.33
- **RL_OOS_年化收益率**: -3.33%
- **RL_OOS_夏普比率**: -0.2406
- **RL_OOS_最大回撤**: 18.79%
- **RL_OOS_年化波动率**: 11.39%
- **RL_OOS_累计换手率**: 10976.50%
- **RL_OOS_累计交易成本/初始资金**: 10.98%
- **---**: ---
- **EqualWeight_Benchmark_期末总资产**: 11392.03
- **EqualWeight_Benchmark_年化收益率**: 3.45%
- **EqualWeight_Benchmark_夏普比率**: 0.2928
- **EqualWeight_Benchmark_最大回撤**: 17.96%
- **EqualWeight_Benchmark_年化波动率**: 15.92%
- **EqualWeight_Benchmark_累计换手率**: 598.11%
- **EqualWeight_Benchmark_累计交易成本/初始资金**: 0.59%
- **Top5Momentum_Benchmark_期末总资产**: 6833.54
- **Top5Momentum_Benchmark_年化收益率**: -9.43%
- **Top5Momentum_Benchmark_夏普比率**: -0.2895
- **Top5Momentum_Benchmark_最大回撤**: 45.64%
- **Top5Momentum_Benchmark_年化波动率**: 24.13%
- **Top5Momentum_Benchmark_累计换手率**: 17897.89%
- **Top5Momentum_Benchmark_累计交易成本/初始资金**: 12.96%

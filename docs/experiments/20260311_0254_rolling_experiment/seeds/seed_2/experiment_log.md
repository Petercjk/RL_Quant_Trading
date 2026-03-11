# Experiment Log: 20260311_0254_rolling_experiment_seed2

## 1. 实验描述
- **task_name**: A-Share Rolling Retrain Backtest
- **description**: 滚动重训机制验证：4年 (2022-01-01 ~ 2025-12-31) | Seed=2
- **indicators**: Standard FinRL Indicators (MACD, RSI, etc.)
- **benchmark**: Equal Weight + Top5 Momentum (with transaction costs)

## 2. 任务流配置
- do_preprocessing: DISABLED
- do_training: ENABLED
- do_backtesting: ENABLED
- do_plotting: ENABLED

## 3. 核心配置与超参数 (Hyperparameters)
- **Rolling_OOS_Window**: `2022-01-01 to 2025-12-31 (4 Years)`
- **Rolling_Timesteps**: `30000`
- **Rolling_Seeds**: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- **Base_Model_Path**: `D:/AAA_Petercjk/RL_Quant_Trading\docs\experiments\20260308_1910_base_experiment\checkpoints\ppo_agent_median_20260308_1928_seed70.zip`
- **Environment_Core**: `Top_K=5, Rebalance=5 days, Lot_Size=100`
- **Reward_Shaping**: `Risk_Penalty=0.1, Turnover_Penalty=0.01, Scaling=1.0`
- **Trading_Costs**: `Buy: 0.001, Sell: 0.001, Initial_Amount: 10000`
- **State_Features**: `[40 cols] macd, rsi_30, cci_30, dx_30, boll_ub, boll_lb, close_30_sma, close_60_sma, log_return, return_20, momentum_60, momentum_120, momentum_252, volatility_20, volatility_annual, sma_5, sma_10, sma_20, sma_30, sma_60, bias_20, bias_60, amplitude, intraday_return, amount_20_mean, volume_20_mean, volume_ratio, high_20, low_20, position_20, turbulence, open, high, low, close, pre_close, change, pct_chg, volume, amount`
- **Market_Features**: `market_return, market_volatility_20, market_turbulence`
- **Processed_File**: `D:/AAA_Petercjk/RL_Quant_Trading\data\processed\processed_40_pool.csv`
- **Roll_Schedule**: `[{'train_start': '2019-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31', 'window_name': 'Roll_1_Test_2022'}, {'train_start': '2020-01-01', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31', 'window_name': 'Roll_2_Test_2023'}, {'train_start': '2021-01-01', 'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31', 'window_name': 'Roll_3_Test_2024'}, {'train_start': '2022-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31', 'window_name': 'Roll_4_Test_2025'}]`
- **Rolling_Seed**: `2`

## 4. 分窗口回测指标
- **Roll_1_Test_2022**: 
- **Roll_1_Test_2022_期末总资产**: 8314.03
- **Roll_1_Test_2022_年化收益率**: -17.49%
- **Roll_1_Test_2022_夏普比率**: -1.4413
- **Roll_1_Test_2022_最大回撤**: 17.49%
- **Roll_1_Test_2022_年化波动率**: 12.77%
- **Roll_1_Test_2022_累计换手率**: 2339.60%
- **Roll_1_Test_2022_累计交易成本/初始资金**: 2.34%
- **Roll_2_Test_2023**: 
- **Roll_2_Test_2023_期末总资产**: 9909.62
- **Roll_2_Test_2023_年化收益率**: -0.94%
- **Roll_2_Test_2023_夏普比率**: -0.0558
- **Roll_2_Test_2023_最大回撤**: 8.48%
- **Roll_2_Test_2023_年化波动率**: 9.27%
- **Roll_2_Test_2023_累计换手率**: 2763.75%
- **Roll_2_Test_2023_累计交易成本/初始资金**: 2.76%
- **Roll_3_Test_2024**: 
- **Roll_3_Test_2024_期末总资产**: 9269.81
- **Roll_3_Test_2024_年化收益率**: -7.59%
- **Roll_3_Test_2024_夏普比率**: -0.6187
- **Roll_3_Test_2024_最大回撤**: 17.00%
- **Roll_3_Test_2024_年化波动率**: 11.66%
- **Roll_3_Test_2024_累计换手率**: 2481.90%
- **Roll_3_Test_2024_累计交易成本/初始资金**: 2.48%
- **Roll_4_Test_2025**: 
- **Roll_4_Test_2025_期末总资产**: 9454.24
- **Roll_4_Test_2025_年化收益率**: -5.65%
- **Roll_4_Test_2025_夏普比率**: -0.9245
- **Roll_4_Test_2025_最大回撤**: 7.86%
- **Roll_4_Test_2025_年化波动率**: 6.09%
- **Roll_4_Test_2025_累计换手率**: 2497.60%
- **Roll_4_Test_2025_累计交易成本/初始资金**: 2.50%

## 5. 全周期回测性能指标
- **RL_Rolling_OOS_期末总资产**: 7220.48
- **RL_Rolling_OOS_年化收益率**: -8.12%
- **RL_Rolling_OOS_夏普比率**: -0.7726
- **RL_Rolling_OOS_最大回撤**: 29.15%
- **RL_Rolling_OOS_年化波动率**: 10.28%
- **RL_Rolling_OOS_累计换手率**: 10082.85%
- **RL_Rolling_OOS_累计交易成本/初始资金**: 10.08%

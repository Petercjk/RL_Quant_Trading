# Experiment Log: 20260311_0220_rolling_experiment

## 1. 实验描述
- **task_name**: A-Share Rolling Retrain Backtest
- **description**: 滚动重训机制验证：4年 (2022-01-01 ~ 2025-12-31)
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
- **Rolling_Seed**: `42`
- **Base_Model_Path**: `D:/AAA_Petercjk/RL_Quant_Trading\docs\experiments\20260308_1910_base_experiment\checkpoints\ppo_agent_median_20260308_1928_seed70.zip`
- **Environment_Core**: `Top_K=5, Rebalance=5 days, Lot_Size=100`
- **Reward_Shaping**: `Risk_Penalty=0.1, Turnover_Penalty=0.01, Scaling=1.0`
- **Trading_Costs**: `Buy: 0.001, Sell: 0.001, Initial_Amount: 10000`
- **State_Features**: `[40 cols] macd, rsi_30, cci_30, dx_30, boll_ub, boll_lb, close_30_sma, close_60_sma, log_return, return_20, momentum_60, momentum_120, momentum_252, volatility_20, volatility_annual, sma_5, sma_10, sma_20, sma_30, sma_60, bias_20, bias_60, amplitude, intraday_return, amount_20_mean, volume_20_mean, volume_ratio, high_20, low_20, position_20, turbulence, open, high, low, close, pre_close, change, pct_chg, volume, amount`
- **Market_Features**: `market_return, market_volatility_20, market_turbulence`
- **Processed_File**: `D:/AAA_Petercjk/RL_Quant_Trading\data\processed\processed_40_pool.csv`
- **Roll_Schedule**: `[{'train_start': '2019-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31', 'window_name': 'Roll_1_Test_2022'}, {'train_start': '2020-01-01', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31', 'window_name': 'Roll_2_Test_2023'}, {'train_start': '2021-01-01', 'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31', 'window_name': 'Roll_3_Test_2024'}, {'train_start': '2022-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31', 'window_name': 'Roll_4_Test_2025'}]`

## 4. 分窗口回测指标
- **Roll_1_Test_2022**: 
- **Roll_1_Test_2022_期末总资产**: 8513.31
- **Roll_1_Test_2022_年化收益率**: -15.43%
- **Roll_1_Test_2022_夏普比率**: -1.1084
- **Roll_1_Test_2022_最大回撤**: 19.55%
- **Roll_1_Test_2022_年化波动率**: 14.20%
- **Roll_1_Test_2022_累计换手率**: 2566.85%
- **Roll_1_Test_2022_累计交易成本/初始资金**: 2.57%
- **Roll_2_Test_2023**: 
- **Roll_2_Test_2023_期末总资产**: 10447.06
- **Roll_2_Test_2023_年化收益率**: 4.66%
- **Roll_2_Test_2023_夏普比率**: 0.4743
- **Roll_2_Test_2023_最大回撤**: 9.32%
- **Roll_2_Test_2023_年化波动率**: 10.84%
- **Roll_2_Test_2023_累计换手率**: 3679.32%
- **Roll_2_Test_2023_累计交易成本/初始资金**: 3.68%
- **Roll_3_Test_2024**: 
- **Roll_3_Test_2024_期末总资产**: 11831.48
- **Roll_3_Test_2024_年化收益率**: 19.14%
- **Roll_3_Test_2024_夏普比率**: 1.4653
- **Roll_3_Test_2024_最大回撤**: 12.44%
- **Roll_3_Test_2024_年化波动率**: 12.49%
- **Roll_3_Test_2024_累计换手率**: 2925.11%
- **Roll_3_Test_2024_累计交易成本/初始资金**: 2.93%
- **Roll_4_Test_2025**: 
- **Roll_4_Test_2025_期末总资产**: 9911.04
- **Roll_4_Test_2025_年化收益率**: -0.92%
- **Roll_4_Test_2025_夏普比率**: -0.0688
- **Roll_4_Test_2025_最大回撤**: 7.25%
- **Roll_4_Test_2025_年化波动率**: 8.38%
- **Roll_4_Test_2025_累计换手率**: 3589.51%
- **Roll_4_Test_2025_累计交易成本/初始资金**: 3.59%

## 5. 全周期回测性能指标
- **RL_Rolling_OOS_期末总资产**: 10429.20
- **RL_Rolling_OOS_年化收益率**: 1.10%
- **RL_Rolling_OOS_夏普比率**: 0.1519
- **RL_Rolling_OOS_最大回撤**: 19.55%
- **RL_Rolling_OOS_年化波动率**: 11.70%
- **RL_Rolling_OOS_累计换手率**: 12760.79%
- **RL_Rolling_OOS_累计交易成本/初始资金**: 12.76%
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

# Experiment Log: 20260311_0254_rolling_experiment

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
- **Rolling_Seeds**: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- **Base_Model_Path**: `D:/AAA_Petercjk/RL_Quant_Trading\docs\experiments\20260308_1910_base_experiment\checkpoints\ppo_agent_median_20260308_1928_seed70.zip`
- **Environment_Core**: `Top_K=5, Rebalance=5 days, Lot_Size=100`
- **Reward_Shaping**: `Risk_Penalty=0.1, Turnover_Penalty=0.01, Scaling=1.0`
- **Trading_Costs**: `Buy: 0.001, Sell: 0.001, Initial_Amount: 10000`
- **State_Features**: `[40 cols] macd, rsi_30, cci_30, dx_30, boll_ub, boll_lb, close_30_sma, close_60_sma, log_return, return_20, momentum_60, momentum_120, momentum_252, volatility_20, volatility_annual, sma_5, sma_10, sma_20, sma_30, sma_60, bias_20, bias_60, amplitude, intraday_return, amount_20_mean, volume_20_mean, volume_ratio, high_20, low_20, position_20, turbulence, open, high, low, close, pre_close, change, pct_chg, volume, amount`
- **Market_Features**: `market_return, market_volatility_20, market_turbulence`
- **Processed_File**: `D:/AAA_Petercjk/RL_Quant_Trading\data\processed\processed_40_pool.csv`
- **Roll_Schedule**: `[{'train_start': '2019-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31', 'window_name': 'Roll_1_Test_2022'}, {'train_start': '2020-01-01', 'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31', 'window_name': 'Roll_2_Test_2023'}, {'train_start': '2021-01-01', 'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31', 'window_name': 'Roll_3_Test_2024'}, {'train_start': '2022-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31', 'window_name': 'Roll_4_Test_2025'}]`

## 4. 十次滚动结果总览
| Seed | 期末总资产 | 年化收益率 | 夏普比率 | 最大回撤 | 年化波动率 | 累计换手率 | 交易成本/初始资金 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9442.43 | -1.48% | -0.0893 | 14.98% | 10.51% | 12249.56% | 12.25% |
| 1 | 8925.27 | -2.91% | -0.2256 | 21.77% | 10.61% | 10773.95% | 10.77% |
| 2 | 7220.48 | -8.12% | -0.7726 | 29.15% | 10.28% | 10082.85% | 10.08% |
| 3 | 9004.40 | -2.69% | -0.1981 | 25.41% | 10.81% | 10962.86% | 10.96% |
| 4 | 8710.52 | -3.53% | -0.2453 | 22.47% | 11.79% | 13672.52% | 13.67% |
| 5 | 7686.38 | -6.61% | -0.5456 | 26.30% | 11.35% | 12652.08% | 12.65% |
| 6 | 8566.51 | -3.94% | -0.3281 | 20.19% | 10.56% | 10261.44% | 10.26% |
| 7 | 10415.15 | 1.06% | 0.1470 | 23.48% | 12.53% | 13737.54% | 13.74% |
| 8 | 9330.70 | -1.79% | -0.1362 | 17.20% | 9.74% | 10759.74% | 10.76% |
| 9 | 9745.15 | -0.67% | -0.0025 | 24.44% | 11.32% | 12946.72% | 12.95% |

## 5. 十次结果统计指标
- **年化收益率均值**: -3.07%
- **年化收益率方差**: 6.6235
- **年化收益率中位数**: -2.80%
- **年化收益率标准差**: 2.57%
- **Seed 数量**: 10

## 6. Benchmark 对照指标
- **EqualWeight_Benchmark_期末总资产**: 11392.03
- **EqualWeight_Benchmark_年化收益率**: 3.45%
- **EqualWeight_Benchmark_夏普比率**: 0.2928
- **EqualWeight_Benchmark_最大回撤**: 17.96%
- **EqualWeight_Benchmark_年化波动率**: 15.92%
- **EqualWeight_Benchmark_累计换手率**: 598.11%
- **EqualWeight_Benchmark_累计交易成本/初始资金**: 0.59%
- **---**: ---
- **Top5Momentum_Benchmark_期末总资产**: 6833.54
- **Top5Momentum_Benchmark_年化收益率**: -9.43%
- **Top5Momentum_Benchmark_夏普比率**: -0.2895
- **Top5Momentum_Benchmark_最大回撤**: 45.64%
- **Top5Momentum_Benchmark_年化波动率**: 24.13%
- **Top5Momentum_Benchmark_累计换手率**: 17897.89%
- **Top5Momentum_Benchmark_累计交易成本/初始资金**: 12.96%

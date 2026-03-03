# Development Log

## 2026-01-29
- Rearranged the structure. 根据实际情况重新编排了文件夹结构。
- Ran MVP successfully, training results obtained. 成功跑通MVP流程，拿到了训练结果。
- Updated requirements.txt, README.md 更新了requirements.txt和README.md等文件。
- Checked PPO output in backtest. 检查了PPO在回测中的输出结果。

## 2026-03-03
- Fixed backtest execution path. Solved the issue where account value was reset to one day after evaluation. 修复了回测执行链路，解决了评估后净值只剩一天的问题。
- Added backtest diagnostics CSV outputs (actions / trades / holdings) for analysis. 新增回测诊断表（动作/成交/持仓）CSV输出，便于定位策略行为与成本影响。
- Updated benchmark to fair setting (equal-weight daily rebalance with buy/sell costs). 将Benchmark更新为更公平口径（等权日频再平衡+双边手续费）。
- Ran end-to-end experiment successfully; PPO agent outperformed benchmark in latest run. 完整流程再次跑通，最新实验中PPO策略跑赢Benchmark。
![Latest Performance Comparison](docs/images/performance_comparison_20260303_2329.png)


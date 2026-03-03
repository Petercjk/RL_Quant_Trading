import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.base_config import DATA_PATH, TIME_WINDOW
from configs.experiment.base_experiment import (
    TASK_CONTROL,
    EXP_PATHS,
    init_base_experiment,
    finalize_experiment,
    plot_comparison,
)
from src.data_processing.processor import DataProcessor
from src.training.train_agent import AgentTrainer


def build_fair_benchmark(
    trade_df: pd.DataFrame,
    initial_amount: float = 1_000_000,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
) -> pd.DataFrame:
    """Equal-weight benchmark with daily rebalance and transaction costs."""
    price_pivot = trade_df.pivot(index="date", columns="tic", values="close").sort_index()
    if price_pivot.empty:
        return pd.DataFrame(columns=["date", "account_value"])

    daily_ret = price_pivot.pct_change().fillna(0.0)
    n_assets = price_pivot.shape[1]
    target_w = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

    # Invest on the first day and pay one-way buy cost.
    nav = float(initial_amount) * (1.0 - buy_cost_pct)
    weights = target_w.copy()

    rows = [{"date": price_pivot.index[0], "account_value": nav}]

    for i in range(1, len(price_pivot.index)):
        r = daily_ret.iloc[i].to_numpy(dtype=np.float64)

        # Portfolio PnL with current weights.
        nav *= 1.0 + float(np.dot(weights, r))

        # Drift after return, then rebalance to equal weights and pay turnover costs.
        gross_weights = weights * (1.0 + r)
        gross_sum = float(np.sum(gross_weights))
        if gross_sum > 0:
            drift_w = gross_weights / gross_sum
            buy_turnover = float(np.maximum(target_w - drift_w, 0.0).sum())
            sell_turnover = float(np.maximum(drift_w - target_w, 0.0).sum())
            fee_ratio = buy_turnover * buy_cost_pct + sell_turnover * sell_cost_pct
            nav *= max(0.0, 1.0 - fee_ratio)

        weights = target_w.copy()
        rows.append({"date": price_pivot.index[i], "account_value": nav})

    return pd.DataFrame(rows)


def run_experiment_pipeline():
    try:
        init_base_experiment()
        print("SUCCESS: 实验目录初始化成功，输出路径:", EXP_PATHS["root"])
    except Exception as e:
        print(f"ERROR: 实验目录初始化失败: {e}")
        return

    if TASK_CONTROL["do_preprocessing"]:
        print("INFO: 执行数据预处理...")
        try:
            processor = DataProcessor(token="d00985e44d97b66607e1bb3209880a913e9e651e861477dbbdbfacaf")
            processor.run()
            if not os.path.exists(DATA_PATH["processed"]):
                print("ERROR: 预处理完成但未找到 processed CSV")
                return
            print("SUCCESS: 数据预处理完成")
        except Exception as e:
            print(f"ERROR: 数据预处理失败: {e}")
            return
    else:
        print("SKIP: 跳过数据预处理，使用现有 processed 数据")

    try:
        full_df = pd.read_csv(DATA_PATH["processed"])
        train_df = full_df[
            (full_df.date >= TIME_WINDOW["train_start"]) & (full_df.date <= TIME_WINDOW["train_end"])
        ]
        trade_df = full_df[
            (full_df.date >= TIME_WINDOW["trade_start"]) & (full_df.date <= TIME_WINDOW["trade_end"])
        ]

        print("========== 数据范围检查 ==========")
        print("FULL RANGE:", full_df["date"].min(), "->", full_df["date"].max())
        print("FULL UNIQUE DAYS:", full_df["date"].nunique())
        print("TRAIN UNIQUE DAYS:", train_df["date"].nunique())
        print("TRADE UNIQUE DAYS:", trade_df["date"].nunique())
        print("=================================")

        if train_df.empty or trade_df.empty:
            print("ERROR: 训练集或回测集为空，请检查时间窗口")
            return
        print(f"SUCCESS: 数据加载成功。训练样本 {len(train_df)} 行，回测样本 {len(trade_df)} 行")
    except Exception as e:
        print(f"ERROR: 加载 processed 数据失败: {e}")
        return

    trainer = AgentTrainer(train_df, trade_df, EXP_PATHS)
    model = None

    if TASK_CONTROL["do_training"]:
        print("INFO: 开始 PPO 训练...")
        try:
            model = trainer.run_training(total_timesteps=50000)
            print("SUCCESS: 模型训练完成")
        except Exception as e:
            print(f"ERROR: 训练失败: {e}")
            return
    else:
        print("SKIP: 跳过训练")

    if TASK_CONTROL["do_backtesting"]:
        if model is None:
            print("ERROR: 无法回测，模型为空")
            return

        print("INFO: 启动测试集回测 (2024-2025)...")
        try:
            df_account_value, _ = trainer.run_backtest(model)
            print("SUCCESS: 回测执行成功，已获取策略净值序列")

            # Fair benchmark: equal-weight + daily rebalance + same transaction costs.
            df_benchmark = build_fair_benchmark(
                trade_df,
                initial_amount=1_000_000,
                buy_cost_pct=0.001,
                sell_cost_pct=0.001,
            )

            table_dir = EXP_PATHS["table"]
            os.makedirs(table_dir, exist_ok=True)
            df_benchmark.to_csv(os.path.join(table_dir, "benchmark_account_value.csv"), index=False)
            print("SUCCESS: 公平基准净值已保存至 tables/benchmark_account_value.csv")

            if TASK_CONTROL["do_plotting"]:
                plot_comparison(df_account_value, df_benchmark)
                print("SUCCESS: 对比图已生成")

            final_nav = df_account_value["account_value"].iloc[-1]
            total_return = (final_nav / 1_000_000) - 1
            perf_stats = {
                "期末总资产": f"{final_nav:.2f}",
                "累计收益率": f"{total_return * 100:.2f}%",
                "测试交易日天数": len(df_account_value),
            }
            finalize_experiment(df_account_value, perf_stats)
            print(f"SUCCESS: 实验报告已更新: {EXP_PATHS['root']}")

        except Exception as e:
            print(f"ERROR: 回测/分析/绘图失败: {e}")
            return


if __name__ == "__main__":
    run_experiment_pipeline()

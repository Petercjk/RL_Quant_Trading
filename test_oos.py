import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from configs.base_config import DATA_PATH, DOCS_DIR, TECHNICAL_INDICATORS
from configs.agent.ppo import PPO_PARAMS
from src.envs.env_stocktrading import StockTradingEnv
from src.training.train_agent import AgentTrainer

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# ===== OOS SETTINGS =====
OOS_START = "2022-01-01"
OOS_END = "2025-12-31"
INITIAL_AMOUNT = 10_000
BUY_COST_PCT = 0.001
SELL_COST_PCT = 0.001
TOP_K = 5
REBALANCE_WINDOW = 5

DEFAULT_MODEL_PATH = os.path.join(
    DOCS_DIR,
    "experiments",
    "20260308_1910_base_experiment",
    "checkpoints",
    "ppo_agent_median_20260308_1928_seed70.zip",
)


def _prepare_price_pivot(trade_df: pd.DataFrame) -> pd.DataFrame:
    return trade_df.pivot(index="date", columns="tic", values="close").sort_index()


def build_equal_weight_benchmark(
    trade_df: pd.DataFrame,
    initial_amount: float,
    rebalance_window: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
):
    """Benchmark1：全股票等权 + 每5日再平衡 + 双边手续费。"""
    price_pivot = _prepare_price_pivot(trade_df)
    if price_pivot.empty:
        empty = pd.DataFrame(columns=["date", "account_value"])
        return empty, {"turnover_ratio": 0.0, "cost_ratio": 0.0}

    daily_ret = price_pivot.pct_change().fillna(0.0)
    n_assets = price_pivot.shape[1]
    target_w = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

    nav = float(initial_amount)
    weights = np.zeros(n_assets, dtype=np.float64)
    turnover_sum = 0.0
    total_fee_amount = 0.0
    rows = [{"date": price_pivot.index[0], "account_value": nav}]

    for i in range(1, len(price_pivot.index)):
        r = daily_ret.iloc[i].to_numpy(dtype=np.float64)
        nav *= 1.0 + float(np.dot(weights, r))

        gross_weights = weights * (1.0 + r)
        gross_sum = float(np.sum(gross_weights))
        drift_w = gross_weights / gross_sum if gross_sum > 0 else np.zeros_like(weights)
        weights = drift_w

        if i % rebalance_window == 0:
            buy_turnover = float(np.maximum(target_w - weights, 0.0).sum())
            sell_turnover = float(np.maximum(weights - target_w, 0.0).sum())
            fee_ratio = buy_turnover * buy_cost_pct + sell_turnover * sell_cost_pct
            fee_amount = nav * fee_ratio
            nav *= max(0.0, 1.0 - fee_ratio)

            turnover_sum += buy_turnover + sell_turnover
            total_fee_amount += fee_amount
            weights = target_w.copy()

        rows.append({"date": price_pivot.index[i], "account_value": nav})

    stats = {
        "turnover_ratio": turnover_sum,
        "cost_ratio": total_fee_amount / float(initial_amount),
    }
    return pd.DataFrame(rows), stats


def build_topk_momentum_benchmark(
    trade_df: pd.DataFrame,
    initial_amount: float,
    top_k: int,
    rebalance_window: int,
    momentum_window: int,
    buy_cost_pct: float,
    sell_cost_pct: float,
):
    """Benchmark2：Top-K 动量 + 每5日调仓 + 双边手续费。"""
    price_pivot = _prepare_price_pivot(trade_df)
    if price_pivot.empty:
        empty = pd.DataFrame(columns=["date", "account_value"])
        return empty, {"turnover_ratio": 0.0, "cost_ratio": 0.0}

    daily_ret = price_pivot.pct_change().fillna(0.0)
    close_np = price_pivot.to_numpy(dtype=np.float64)
    n_assets = close_np.shape[1]
    top_k = int(max(1, min(top_k, n_assets)))

    nav = float(initial_amount)
    weights = np.zeros(n_assets, dtype=np.float64)
    turnover_sum = 0.0
    total_fee_amount = 0.0
    rows = [{"date": price_pivot.index[0], "account_value": nav}]

    for i in range(1, len(price_pivot.index)):
        r = daily_ret.iloc[i].to_numpy(dtype=np.float64)
        nav *= 1.0 + float(np.dot(weights, r))

        gross_weights = weights * (1.0 + r)
        gross_sum = float(np.sum(gross_weights))
        drift_w = gross_weights / gross_sum if gross_sum > 0 else np.zeros_like(weights)
        weights = drift_w

        if i % rebalance_window == 0:
            lookback_end = i - 1
            lookback_start = max(0, lookback_end - momentum_window)
            momentum = close_np[lookback_end] / np.maximum(close_np[lookback_start], 1e-8) - 1.0

            select_idx = np.argsort(momentum)[::-1][:top_k]
            target_w = np.zeros(n_assets, dtype=np.float64)
            target_w[select_idx] = 1.0 / float(top_k)

            buy_turnover = float(np.maximum(target_w - weights, 0.0).sum())
            sell_turnover = float(np.maximum(weights - target_w, 0.0).sum())
            fee_ratio = buy_turnover * buy_cost_pct + sell_turnover * sell_cost_pct
            fee_amount = nav * fee_ratio
            nav *= max(0.0, 1.0 - fee_ratio)

            turnover_sum += buy_turnover + sell_turnover
            total_fee_amount += fee_amount
            weights = target_w

        rows.append({"date": price_pivot.index[i], "account_value": nav})

    stats = {
        "turnover_ratio": turnover_sum,
        "cost_ratio": total_fee_amount / float(initial_amount),
    }
    return pd.DataFrame(rows), stats


def compute_six_metrics(
    account_value_df: pd.DataFrame,
    initial_amount: float,
    turnover_ratio: float,
    cost_ratio: float,
) -> dict:
    """统一计算六项指标。"""
    nav_df = account_value_df.copy()
    nav_df = nav_df.sort_values("date").reset_index(drop=True)
    nav = nav_df["account_value"].astype(float)
    daily_ret = nav.pct_change().fillna(0.0)

    n = len(nav)
    final_nav = float(nav.iloc[-1]) if n > 0 else float(initial_amount)
    annual_return = (final_nav / float(initial_amount)) ** (252.0 / max(n, 1)) - 1.0

    vol_annual = float(daily_ret.std(ddof=0) * np.sqrt(252.0))
    sharpe = 0.0
    if vol_annual > 1e-12:
        sharpe = float((daily_ret.mean() * np.sqrt(252.0)) / (daily_ret.std(ddof=0) + 1e-12))

    cummax = nav.cummax()
    drawdown = (cummax - nav) / cummax.replace(0, np.nan)
    max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    return {
        "期末总资产": f"{final_nav:.2f}",
        "年化收益率": f"{annual_return * 100:.2f}%",
        "夏普比率": f"{sharpe:.4f}",
        "最大回撤": f"{max_drawdown * 100:.2f}%",
        "年化波动率": f"{vol_annual * 100:.2f}%",
        "累计换手率": f"{turnover_ratio * 100:.2f}%",
        "累计交易成本/初始资金": f"{cost_ratio * 100:.2f}%",
    }


def _init_oos_experiment(exp_paths: dict, exp_meta: dict, task_control: dict, hyperparams: dict):
    for p in exp_paths.values():
        os.makedirs(p, exist_ok=True)

    log_path = os.path.join(exp_paths["root"], "experiment_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment Log: {exp_meta['exp_name']}\n\n")
        f.write("## 1. 实验描述\n")
        for k in ["task_name", "description", "indicators", "benchmark"]:
            f.write(f"- **{k}**: {exp_meta[k]}\n")
        f.write("\n## 2. 任务流配置\n")
        for k, v in task_control.items():
            f.write(f"- {k}: {'ENABLED' if v else 'DISABLED'}\n")

        if hyperparams:
            f.write("\n## 3. 核心配置与超参数 (Hyperparameters)\n")
            for k, v in hyperparams.items():
                f.write(f"- **{k}**: `{v}`\n")


def _finalize_oos_experiment(exp_paths: dict, account_value_df: pd.DataFrame, stats: dict):
    log_path = os.path.join(exp_paths["root"], "experiment_log.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n## 4. 回测性能指标\n")
        for k, v in stats.items():
            f.write(f"- **{k}**: {v}\n")

    account_value_df.to_csv(os.path.join(exp_paths["table"], "account_value.csv"), index=False)


def _plot_comparison(exp_paths: dict, ai_results: pd.DataFrame, benchmark_results: dict):
    import matplotlib.pyplot as plt

    os.makedirs(exp_paths["plot"], exist_ok=True)
    plt.figure(figsize=(12, 6))

    ai_results = ai_results.copy()
    ai_results["date"] = pd.to_datetime(ai_results["date"])
    plt.plot(ai_results["date"], ai_results["account_value"], label="AI Agent (PPO)", color="red")

    for label, df_bm in benchmark_results.items():
        df_bm = df_bm.copy()
        df_bm["date"] = pd.to_datetime(df_bm["date"])
        plt.plot(df_bm["date"], df_bm["account_value"], label=label, linestyle="--")

    plt.title("Backtest Performance Comparison")
    plt.xlabel("Date")
    plt.ylabel("Account Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_paths["plot"], "performance_comparison.png"))
    plt.close()


def _resolve_processed_path() -> str:
    # Prefer 40-pool processed file if present.
    candidate = os.path.join(os.path.dirname(DATA_PATH["processed"]), "processed_40_pool.csv")
    if os.path.exists(candidate):
        return candidate
    return DATA_PATH["processed"]


def run_oos():
    exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_oos_experiment"
    exp_dir = os.path.join(DOCS_DIR, "experiments", exp_name)
    exp_paths = {
        "root": exp_dir,
        "model": os.path.join(exp_dir, "checkpoints"),
        "log": os.path.join(exp_dir, "logs"),
        "plot": os.path.join(exp_dir, "plots"),
        "table": os.path.join(exp_dir, "tables"),
    }

    exp_meta = {
        "exp_name": exp_name,
        "task_name": "A-Share Long OOS Backtest",
        "description": f"长期纯样本外测试：{OOS_START} ~ {OOS_END}",
        "indicators": "Standard FinRL Indicators (MACD, RSI, etc.)",
        "benchmark": "Equal Weight + Top5 Momentum (with transaction costs)",
    }

    task_control = {
        "do_preprocessing": False,
        "do_training": False,
        "do_backtesting": True,
        "do_plotting": True,
    }

    processed_path = _resolve_processed_path()
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    full_df = pd.read_csv(processed_path)
    trade_df = full_df[(full_df.date >= OOS_START) & (full_df.date <= OOS_END)]
    if trade_df.empty:
        raise ValueError("OOS trade_df is empty; check OOS date range or data file.")

    stock_dim = len(trade_df.tic.unique())
    tmp_env = StockTradingEnv(
        df=trade_df,
        stock_dim=stock_dim,
        hmax=1000,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=[BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[SELL_COST_PCT] * stock_dim,
        tech_indicator_list=TECHNICAL_INDICATORS,
    )

    hyperparams_log = {
        "OOS_Window": f"{OOS_START} to {OOS_END}",
        "Model_Path": DEFAULT_MODEL_PATH,
        "PPO_Params": str(PPO_PARAMS),
        "Total_Timesteps": 50000,
        "Environment_Core": f"Top_K={tmp_env.top_k}, Rebalance={tmp_env.rebalance_window} days, Lot_Size={tmp_env.lot_size}",
        "Reward_Shaping": f"Risk_Penalty={tmp_env.risk_penalty}, Turnover_Penalty={tmp_env.turnover_penalty}, Scaling={tmp_env.reward_scaling}",
        "Trading_Costs": f"Buy: {BUY_COST_PCT}, Sell: {SELL_COST_PCT}, Initial_Amount: {INITIAL_AMOUNT}",
        "State_Features": f"[{len(tmp_env.feature_cols)} cols] " + ", ".join(tmp_env.feature_cols),
        "Market_Features": ", ".join(list(tmp_env.market_df.columns)),
        "Processed_File": processed_path,
    }

    _init_oos_experiment(exp_paths, exp_meta, task_control, hyperparams_log)
    print(f"SUCCESS: OOS 实验目录初始化成功: {exp_paths['root']}")

    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PPO.load(model_path)
    trainer = AgentTrainer(trade_df, trade_df, exp_paths)

    df_account_value, _ = trainer.run_backtest(model)

    df_bm_equal, bm_equal_stats = build_equal_weight_benchmark(
        trade_df,
        initial_amount=INITIAL_AMOUNT,
        rebalance_window=REBALANCE_WINDOW,
        buy_cost_pct=BUY_COST_PCT,
        sell_cost_pct=SELL_COST_PCT,
    )
    df_bm_mom, bm_mom_stats = build_topk_momentum_benchmark(
        trade_df,
        initial_amount=INITIAL_AMOUNT,
        top_k=TOP_K,
        rebalance_window=REBALANCE_WINDOW,
        momentum_window=20,
        buy_cost_pct=BUY_COST_PCT,
        sell_cost_pct=SELL_COST_PCT,
    )

    table_dir = exp_paths["table"]
    os.makedirs(table_dir, exist_ok=True)
    df_bm_equal.to_csv(os.path.join(table_dir, "benchmark_equal_weight.csv"), index=False)
    df_bm_mom.to_csv(os.path.join(table_dir, "benchmark_top5_momentum.csv"), index=False)

    _plot_comparison(
        exp_paths,
        df_account_value,
        {
            "Benchmark: Equal-Weight (5-day)": df_bm_equal,
            "Benchmark: Top5 Momentum (5-day)": df_bm_mom,
        },
    )

    # Compute RL metrics
    trade_csv = os.path.join(table_dir, "backtest_trades.csv")
    if os.path.exists(trade_csv):
        df_trades = pd.read_csv(trade_csv)
        rl_turn_ratio = float(df_trades["trade_notional"].sum()) / float(INITIAL_AMOUNT)
        rl_cost_ratio = float(df_trades["trade_fee"].sum()) / float(INITIAL_AMOUNT)
    else:
        rl_turn_ratio, rl_cost_ratio = 0.0, 0.0

    rl_metrics = compute_six_metrics(df_account_value, INITIAL_AMOUNT, rl_turn_ratio, rl_cost_ratio)
    bm_equal_metrics = compute_six_metrics(
        df_bm_equal, INITIAL_AMOUNT, bm_equal_stats["turnover_ratio"], bm_equal_stats["cost_ratio"]
    )
    bm_mom_metrics = compute_six_metrics(
        df_bm_mom, INITIAL_AMOUNT, bm_mom_stats["turnover_ratio"], bm_mom_stats["cost_ratio"]
    )

    perf_stats = {"RL_OOS_" + k: v for k, v in rl_metrics.items()}
    perf_stats["---"] = "---"
    perf_stats.update({"EqualWeight_Benchmark_" + k: v for k, v in bm_equal_metrics.items()})
    perf_stats["---"] = "---"
    perf_stats.update({"Top5Momentum_Benchmark_" + k: v for k, v in bm_mom_metrics.items()})

    _finalize_oos_experiment(exp_paths, df_account_value, perf_stats)
    print(f"SUCCESS: OOS 实验完成，日志已写入: {exp_paths['root']}")


if __name__ == "__main__":
    run_oos()

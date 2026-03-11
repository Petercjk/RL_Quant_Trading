import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch as th

from configs.base_config import DATA_PATH, DOCS_DIR, TECHNICAL_INDICATORS
from configs.agent.ppo import PPO_PARAMS
from src.envs.env_stocktrading import StockTradingEnv
from src.training.train_agent import AgentTrainer

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# ===== ROLLING OOS SETTINGS =====
FULL_OOS_START = "2022-01-01"
FULL_OOS_END = "2025-12-31"
INITIAL_AMOUNT = 10_000
BUY_COST_PCT = 0.001
SELL_COST_PCT = 0.001
TOP_K = 5
REBALANCE_WINDOW = 5
ROLLING_TIMESTEPS = 30_000  # 每次微调的步数
ROLLING_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 基座模型路径 (Stage 1 中位数模型)
DEFAULT_MODEL_PATH = os.path.join(
    DOCS_DIR,
    "experiments",
    "20260308_1910_base_experiment",
    "checkpoints",
    "ppo_agent_median_20260308_1928_seed70.zip",
)

# 4年滑动窗口滚动计划
ROLL_SCHEDULE = [
    {
        "train_start": "2019-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
        "window_name": "Roll_1_Test_2022",
    },
    {
        "train_start": "2020-01-01",
        "train_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
        "window_name": "Roll_2_Test_2023",
    },
    {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
        "window_name": "Roll_3_Test_2024",
    },
    {
        "train_start": "2022-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
        "window_name": "Roll_4_Test_2025",
    },
]


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
    raw = compute_metrics_raw(account_value_df, initial_amount, turnover_ratio, cost_ratio)
    return format_metrics(raw)


def compute_metrics_raw(
    account_value_df: pd.DataFrame,
    initial_amount: float,
    turnover_ratio: float,
    cost_ratio: float,
) -> dict:
    """输出数值型指标，便于统计分析。"""
    nav_df = account_value_df.copy().sort_values("date").reset_index(drop=True)
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
        "final_nav": final_nav,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "vol_annual": vol_annual,
        "turnover_ratio": turnover_ratio,
        "cost_ratio": cost_ratio,
    }


def format_metrics(raw: dict) -> dict:
    return {
        "期末总资产": f"{raw['final_nav']:.2f}",
        "年化收益率": f"{raw['annual_return'] * 100:.2f}%",
        "夏普比率": f"{raw['sharpe']:.4f}",
        "最大回撤": f"{raw['max_drawdown'] * 100:.2f}%",
        "年化波动率": f"{raw['vol_annual'] * 100:.2f}%",
        "累计换手率": f"{raw['turnover_ratio'] * 100:.2f}%",
        "累计交易成本/初始资金": f"{raw['cost_ratio'] * 100:.2f}%",
    }


def _init_rolling_experiment(exp_paths: dict, exp_meta: dict, task_control: dict, hyperparams: dict):
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


def _append_section(log_path: str, title: str, stats: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n")
        for k, v in stats.items():
            f.write(f"- **{k}**: {v}\n")


def _plot_comparison(exp_paths: dict, ai_results: pd.DataFrame, benchmark_results: dict):
    import matplotlib.pyplot as plt

    os.makedirs(exp_paths["plot"], exist_ok=True)
    plt.figure(figsize=(12, 6))

    ai_results = ai_results.copy()
    ai_results["date"] = pd.to_datetime(ai_results["date"])
    plt.plot(ai_results["date"], ai_results["account_value"], label="AI Agent (Rolling PPO)", color="red")

    for label, df_bm in benchmark_results.items():
        df_bm = df_bm.copy()
        df_bm["date"] = pd.to_datetime(df_bm["date"])
        plt.plot(df_bm["date"], df_bm["account_value"], label=label, linestyle="--")

    plt.title("Rolling Retrain Backtest Performance Comparison (2022-2025)")
    plt.xlabel("Date")
    plt.ylabel("Account Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_paths["plot"], "performance_comparison.png"))
    plt.close()


def _resolve_processed_path() -> str:
    candidate = os.path.join(os.path.dirname(DATA_PATH["processed"]), "processed_40_pool.csv")
    return candidate if os.path.exists(candidate) else DATA_PATH["processed"]


def _collect_turnover_and_cost(trades_csv: str, initial_amount: float) -> tuple:
    if not os.path.exists(trades_csv):
        return 0.0, 0.0
    df_trades = pd.read_csv(trades_csv)
    turnover_ratio = float(df_trades["trade_notional"].sum()) / float(initial_amount)
    cost_ratio = float(df_trades["trade_fee"].sum()) / float(initial_amount)
    return turnover_ratio, cost_ratio


def _append_seed_table(log_path: str, seed_rows: list):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n## 4. 十次滚动结果总览\n")
        f.write("| Seed | 期末总资产 | 年化收益率 | 夏普比率 | 最大回撤 | 年化波动率 | 累计换手率 | 交易成本/初始资金 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in seed_rows:
            f.write(
                "| {seed} | {final_nav:.2f} | {annual_return:.2f}% | {sharpe:.4f} | {max_drawdown:.2f}% | {vol_annual:.2f}% | {turnover_ratio:.2f}% | {cost_ratio:.2f}% |\n".format(
                    **row
                )
            )


def _plot_seed_annual_returns(exp_paths: dict, seed_rows: list):
    import matplotlib.pyplot as plt

    os.makedirs(exp_paths["plot"], exist_ok=True)
    seeds = [r["seed"] for r in seed_rows]
    ann = [r["annual_return"] for r in seed_rows]

    plt.figure(figsize=(12, 6))
    plt.bar([str(s) for s in seeds], ann, color="#1f77b4")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title("Rolling OOS Annual Return Comparison (10 Seeds)")
    plt.xlabel("Seed")
    plt.ylabel("Annual Return (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_paths["plot"], "seed_annual_return_comparison.png"))
    plt.close()


def run_rolling():
    exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_rolling_experiment"
    exp_dir = os.path.join(DOCS_DIR, "experiments", exp_name)
    exp_paths = {
        "root": exp_dir,
        "model": os.path.join(exp_dir, "checkpoints"),
        "log": os.path.join(exp_dir, "logs"),
        "plot": os.path.join(exp_dir, "plots"),
        "table": os.path.join(exp_dir, "tables"),
        "rolls": os.path.join(exp_dir, "rolls"),
        "seeds": os.path.join(exp_dir, "seeds"),
    }

    exp_meta = {
        "exp_name": exp_name,
        "task_name": "A-Share Rolling Retrain Backtest",
        "description": f"滚动重训机制验证：4年 ({FULL_OOS_START} ~ {FULL_OOS_END})",
        "indicators": "Standard FinRL Indicators (MACD, RSI, etc.)",
        "benchmark": "Equal Weight + Top5 Momentum (with transaction costs)",
    }

    task_control = {
        "do_preprocessing": False,
        "do_training": True,
        "do_backtesting": True,
        "do_plotting": True,
    }

    processed_path = _resolve_processed_path()
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"找不到数据文件: {processed_path}")

    full_df = pd.read_csv(processed_path)
    full_oos_df = full_df[(full_df.date >= FULL_OOS_START) & (full_df.date <= FULL_OOS_END)]
    if full_oos_df.empty:
        raise ValueError("Rolling OOS 数据为空，请检查时间范围或数据文件。")

    stock_dim = len(full_oos_df.tic.unique())
    tmp_env = StockTradingEnv(
        df=full_oos_df,
        stock_dim=stock_dim,
        hmax=1000,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=[BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[SELL_COST_PCT] * stock_dim,
        tech_indicator_list=TECHNICAL_INDICATORS,
    )

    hyperparams_log = {
        "Rolling_OOS_Window": f"{FULL_OOS_START} to {FULL_OOS_END} (4 Years)",
        "Rolling_Timesteps": ROLLING_TIMESTEPS,
        "Rolling_Seeds": str(ROLLING_SEEDS),
        "Base_Model_Path": DEFAULT_MODEL_PATH,
        "Environment_Core": f"Top_K={tmp_env.top_k}, Rebalance={tmp_env.rebalance_window} days, Lot_Size={tmp_env.lot_size}",
        "Reward_Shaping": f"Risk_Penalty={tmp_env.risk_penalty}, Turnover_Penalty={tmp_env.turnover_penalty}, Scaling={tmp_env.reward_scaling}",
        "Trading_Costs": f"Buy: {BUY_COST_PCT}, Sell: {SELL_COST_PCT}, Initial_Amount: {INITIAL_AMOUNT}",
        "State_Features": f"[{len(tmp_env.feature_cols)} cols] " + ", ".join(tmp_env.feature_cols),
        "Market_Features": ", ".join(list(tmp_env.market_df.columns)),
        "Processed_File": processed_path,
        "Roll_Schedule": str(ROLL_SCHEDULE),
    }

    _init_rolling_experiment(exp_paths, exp_meta, task_control, hyperparams_log)
    print(f"SUCCESS: Rolling 实验目录初始化成功: {exp_paths['root']}")

    # ============== 1) 全周期 Benchmark ==============
    print("INFO: 正在生成四年完整 Benchmark 曲线...")
    df_bm_equal, bm_equal_stats = build_equal_weight_benchmark(
        full_oos_df,
        initial_amount=INITIAL_AMOUNT,
        rebalance_window=REBALANCE_WINDOW,
        buy_cost_pct=BUY_COST_PCT,
        sell_cost_pct=SELL_COST_PCT,
    )
    df_bm_mom, bm_mom_stats = build_topk_momentum_benchmark(
        full_oos_df,
        initial_amount=INITIAL_AMOUNT,
        top_k=TOP_K,
        rebalance_window=REBALANCE_WINDOW,
        momentum_window=20,
        buy_cost_pct=BUY_COST_PCT,
        sell_cost_pct=SELL_COST_PCT,
    )

    os.makedirs(exp_paths["table"], exist_ok=True)
    df_bm_equal.to_csv(os.path.join(exp_paths["table"], "benchmark_equal_weight.csv"), index=False)
    df_bm_mom.to_csv(os.path.join(exp_paths["table"], "benchmark_top5_momentum.csv"), index=False)

    # ============== 2) 多 Seed 滚动重训与分年推理 ==============
    seed_summary_rows = []
    seed_log_path = os.path.join(exp_paths["root"], "experiment_log.md")

    for seed in ROLLING_SEEDS:
        seed_dir = os.path.join(exp_paths["seeds"], f"seed_{seed}")
        seed_paths = {
            "root": seed_dir,
            "model": os.path.join(seed_dir, "checkpoints"),
            "log": os.path.join(seed_dir, "logs"),
            "plot": os.path.join(seed_dir, "plots"),
            "table": os.path.join(seed_dir, "tables"),
            "rolls": os.path.join(seed_dir, "rolls"),
        }

        seed_meta = exp_meta.copy()
        seed_meta["exp_name"] = f"{exp_name}_seed{seed}"
        seed_meta["description"] = f"{exp_meta['description']} | Seed={seed}"

        seed_hyper = hyperparams_log.copy()
        seed_hyper["Rolling_Seed"] = seed

        _init_rolling_experiment(seed_paths, seed_meta, task_control, seed_hyper)
        print(f"\n================ 开始 Seed={seed} 滚动流水线 ================")

        current_model_path = DEFAULT_MODEL_PATH
        rolling_rows = []
        rolling_nav = float(INITIAL_AMOUNT)
        roll_metrics = {}
        total_turnover_ratio = 0.0
        total_cost_ratio = 0.0

        for roll_idx, schedule in enumerate(ROLL_SCHEDULE, 1):
            window_name = schedule["window_name"]
            print(f"\n>>> Seed {seed} | 阶段 {roll_idx}/{len(ROLL_SCHEDULE)} : [{window_name}]")
            print(f"    - 微调窗口: {schedule['train_start']} ~ {schedule['train_end']}")
            print(f"    - 测试窗口: {schedule['test_start']} ~ {schedule['test_end']}")

            train_df = full_df[(full_df.date >= schedule["train_start"]) & (full_df.date <= schedule["train_end"])][:]
            test_df = full_df[(full_df.date >= schedule["test_start"]) & (full_df.date <= schedule["test_end"])][:]
            if train_df.empty or test_df.empty:
                raise ValueError(f"数据为空: {window_name}")

            # --- 微调 (Fine-tune) ---
            np.random.seed(seed)
            th.manual_seed(seed)
            train_env = StockTradingEnv(
                df=train_df,
                stock_dim=stock_dim,
                hmax=1000,
                initial_amount=INITIAL_AMOUNT,
                buy_cost_pct=[BUY_COST_PCT] * stock_dim,
                sell_cost_pct=[SELL_COST_PCT] * stock_dim,
                tech_indicator_list=TECHNICAL_INDICATORS,
            )
            sb3_train_env, _ = train_env.get_sb_env()
            try:
                sb3_train_env.seed(seed)
            except Exception:
                pass

            model = PPO.load(current_model_path, env=sb3_train_env)
            try:
                model.set_random_seed(seed)
            except Exception:
                pass
            model.learn(total_timesteps=ROLLING_TIMESTEPS)

            os.makedirs(seed_paths["model"], exist_ok=True)
            current_model_path = os.path.join(seed_paths["model"], f"ppo_{window_name}.zip")
            model.save(current_model_path)

            # --- 测试推理 (Inference) ---
            roll_dir = os.path.join(seed_paths["rolls"], window_name)
            roll_paths = {
                "root": roll_dir,
                "model": os.path.join(roll_dir, "checkpoints"),
                "log": os.path.join(roll_dir, "logs"),
                "plot": os.path.join(roll_dir, "plots"),
                "table": os.path.join(roll_dir, "tables"),
            }
            for p in roll_paths.values():
                os.makedirs(p, exist_ok=True)

            trainer = AgentTrainer(train_df, test_df, roll_paths)
            df_account_value, _ = trainer.run_backtest(model)

            # 保存该年的净值表
            df_account_value.to_csv(os.path.join(roll_paths["table"], "account_value.csv"), index=False)

            # 计算该年的绩效指标
            trade_csv = os.path.join(roll_paths["table"], "backtest_trades.csv")
            roll_turn, roll_cost = _collect_turnover_and_cost(trade_csv, INITIAL_AMOUNT)
            total_turnover_ratio += roll_turn
            total_cost_ratio += roll_cost

            roll_metrics[window_name] = compute_six_metrics(
                df_account_value, INITIAL_AMOUNT, roll_turn, roll_cost
            )

            # 记录滚动净值
            df_account_value = df_account_value.sort_values("date").reset_index(drop=True)
            daily_ret = df_account_value["account_value"].pct_change().fillna(0.0).to_numpy()
            for i, row in df_account_value.iterrows():
                if not rolling_rows:
                    rolling_nav = float(INITIAL_AMOUNT)
                    rolling_rows.append({"date": row["date"], "account_value": rolling_nav})
                    continue
                rolling_nav *= (1.0 + float(daily_ret[i]))
                rolling_rows.append({"date": row["date"], "account_value": rolling_nav})

        # Seed 总体评估
        df_rolling_full = pd.DataFrame(rolling_rows)
        raw_rl_metrics = compute_metrics_raw(
            df_rolling_full, INITIAL_AMOUNT, total_turnover_ratio, total_cost_ratio
        )
        rl_metrics = format_metrics(raw_rl_metrics)

        # 写入 seed 日志
        seed_log = os.path.join(seed_paths["root"], "experiment_log.md")
        per_roll_stats = {}
        for roll_name, metrics in roll_metrics.items():
            per_roll_stats[roll_name] = ""
            for k, v in metrics.items():
                per_roll_stats[f"{roll_name}_{k}"] = v
        _append_section(seed_log, "4. 分窗口回测指标", per_roll_stats)

        _append_section(seed_log, "5. 全周期回测性能指标", {"RL_Rolling_OOS_" + k: v for k, v in rl_metrics.items()})

        # 保存净值与图表
        os.makedirs(seed_paths["table"], exist_ok=True)
        df_rolling_full.to_csv(os.path.join(seed_paths["table"], "account_value.csv"), index=False)

        _plot_comparison(
            seed_paths,
            df_rolling_full,
            {
                "Benchmark: Equal-Weight (5-day)": df_bm_equal,
                "Benchmark: Top5 Momentum (5-day)": df_bm_mom,
            },
        )

        seed_summary_rows.append(
            {
                "seed": seed,
                "final_nav": raw_rl_metrics["final_nav"],
                "annual_return": raw_rl_metrics["annual_return"] * 100.0,
                "sharpe": raw_rl_metrics["sharpe"],
                "max_drawdown": raw_rl_metrics["max_drawdown"] * 100.0,
                "vol_annual": raw_rl_metrics["vol_annual"] * 100.0,
                "turnover_ratio": raw_rl_metrics["turnover_ratio"] * 100.0,
                "cost_ratio": raw_rl_metrics["cost_ratio"] * 100.0,
            }
        )

    # ============== 3) 汇总统计与全局报告 ==============
    bm_equal_metrics = compute_six_metrics(
        df_bm_equal, INITIAL_AMOUNT, bm_equal_stats["turnover_ratio"], bm_equal_stats["cost_ratio"]
    )
    bm_mom_metrics = compute_six_metrics(
        df_bm_mom, INITIAL_AMOUNT, bm_mom_stats["turnover_ratio"], bm_mom_stats["cost_ratio"]
    )

    seed_summary_rows = sorted(seed_summary_rows, key=lambda x: x["seed"])
    df_seed_summary = pd.DataFrame(seed_summary_rows)
    df_seed_summary.to_csv(os.path.join(exp_paths["table"], "seed_summary.csv"), index=False)

    _append_seed_table(seed_log_path, seed_summary_rows)

    ann_returns = df_seed_summary["annual_return"].to_numpy(dtype=float)
    stats_block = {
        "年化收益率均值": f"{ann_returns.mean():.2f}%",
        "年化收益率方差": f"{ann_returns.var(ddof=0):.4f}",
        "年化收益率中位数": f"{np.median(ann_returns):.2f}%",
        "年化收益率标准差": f"{ann_returns.std(ddof=0):.2f}%",
        "Seed 数量": f"{len(ann_returns)}",
    }
    _append_section(seed_log_path, "5. 十次结果统计指标", stats_block)

    final_stats = {"EqualWeight_Benchmark_" + k: v for k, v in bm_equal_metrics.items()}
    final_stats["---"] = "---"
    final_stats.update({"Top5Momentum_Benchmark_" + k: v for k, v in bm_mom_metrics.items()})
    _append_section(seed_log_path, "6. Benchmark 对照指标", final_stats)

    _plot_seed_annual_returns(exp_paths, seed_summary_rows)

    print(f"SUCCESS: Rolling OOS(10 seeds) 完成！报告与绘图已写入: {exp_paths['root']}")


if __name__ == "__main__":
    run_rolling()

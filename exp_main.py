import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import time
from datetime import datetime
from configs.agent.ppo import PPO_PARAMS


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
from src.training.train_agent import AgentTrainer


def _prepare_price_pivot(trade_df: pd.DataFrame) -> pd.DataFrame:
    """将长表行情转换为 date x ticker 的收盘价矩阵。"""
    pivot = trade_df.pivot(index="date", columns="tic", values="close").sort_index()
    return pivot


def build_equal_weight_benchmark(
    trade_df: pd.DataFrame,
    initial_amount: float = 10_000,
    rebalance_window: int = 5,  # 新增：对齐 5日调仓 频率
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
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

        # 每日股价变动导致权重漂移
        gross_weights = weights * (1.0 + r)
        gross_sum = float(np.sum(gross_weights))
        drift_w = gross_weights / gross_sum if gross_sum > 0 else np.zeros_like(weights)
        weights = drift_w

        # 修改核心：仅在 rebalance_window 的倍数日进行再平衡操作
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
    initial_amount: float = 10_000,
    top_k: int = 5,
    rebalance_window: int = 5,
    momentum_window: int = 20,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
):
    """Benchmark2：Top-K动量 + 每5日调仓 + 双边手续费。"""
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

        # 每 rebalance_window 天调仓一次，使用到 t-1 为止可见信息打分
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



def run_experiment_pipeline():
    initial_amount = 10_000
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    
    # 设置 10 个随机种子
    SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  
    
    hyperparams_log = {
        "Train_Window": f"{TIME_WINDOW['train_start']} to {TIME_WINDOW['train_end']}",
        "Test_Window": f"{TIME_WINDOW['trade_start']} to {TIME_WINDOW['trade_end']}",
        "Initial_Amount": initial_amount,
        "Transaction_Costs": f"Buy: {buy_cost_pct}, Sell: {sell_cost_pct}",
        "PPO_Params": str(PPO_PARAMS),
        "Total_Timesteps": 50000,
        "Rebalance_Window": 5,
        "Top_K": 5
    }

    try:
        # 将参数字典传入初始化函数
        init_base_experiment(hyperparams_dict=hyperparams_log)
        print(f"SUCCESS: 实验目录初始化成功，输出路径: {EXP_PATHS['root']}")
    except Exception as e:
        print(f"ERROR: 实验目录初始化失败: {e}")
        return

    # 数据加载与预处理
    try:
        full_df = pd.read_csv(DATA_PATH["processed"])
        train_df = full_df[(full_df.date >= TIME_WINDOW["train_start"]) & (full_df.date <= TIME_WINDOW["train_end"])]
        trade_df = full_df[(full_df.date >= TIME_WINDOW["trade_start"]) & (full_df.date <= TIME_WINDOW["trade_end"])]
        if train_df.empty or trade_df.empty:
            print("ERROR: 训练集或回测集为空，请检查时间窗口")
            return
    except Exception as e:
        print(f"ERROR: 加载 processed 数据失败: {e}")
        return

    trainer = AgentTrainer(train_df, trade_df, EXP_PATHS)

    print("\nINFO: 正在计算 Benchmarks...")
    df_bm_equal, bm_equal_stats = build_equal_weight_benchmark(
        trade_df, initial_amount=initial_amount, rebalance_window=5,
        buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct
    )
    df_bm_mom, bm_mom_stats = build_topk_momentum_benchmark(
        trade_df, initial_amount=initial_amount, top_k=5, rebalance_window=5,
        momentum_window=20, buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct
    )
    
    bm_mom_metrics = compute_six_metrics(df_bm_mom, initial_amount, bm_mom_stats["turnover_ratio"], bm_mom_stats["cost_ratio"])
    bm_mom_sharpe = float(bm_mom_metrics["夏普比率"])

    table_dir = EXP_PATHS["table"]
    os.makedirs(table_dir, exist_ok=True)
    df_bm_equal.to_csv(os.path.join(table_dir, "benchmark_equal_weight.csv"), index=False)
    df_bm_mom.to_csv(os.path.join(table_dir, "benchmark_top5_momentum.csv"), index=False)

    print(f"\n开始多随机种子实验，共计 {len(SEEDS)} 轮。")
    print("--------------------------------------------------")
    
    seed_results = []
    
    # 用一个字典临时在内存中保留所有模型和净值数据
    seed_artifacts = {}

    for idx, seed in enumerate(SEEDS, 1):
        print(f"\n[进度 {idx}/{len(SEEDS)}] 正在执行 Seed = {seed} ...")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        start_time = time.time()

        if TASK_CONTROL["do_training"]:
            print(f"   正在训练 PPO 模型...")
            try:
                model = trainer.run_training(total_timesteps=50000)
                train_time_sec = time.time() - start_time
                print(f"   Seed {seed} 训练完成！[耗时: {train_time_sec/60:.2f} 分钟]")
            except Exception as e:
                print(f"   训练失败: {e}")
                return
        else:
            model = None

        if TASK_CONTROL["do_backtesting"]:
            if model is None: return
            
            print(f"   正在执行回测 (2019-2021)...")
            try:
                df_account_value, _ = trainer.run_backtest(model)
                
                # 读取回测明细计算成本
                trade_csv = os.path.join(table_dir, "backtest_trades.csv")
                if os.path.exists(trade_csv):
                    df_trades = pd.read_csv(trade_csv)
                    rl_turn_ratio = float(df_trades["trade_notional"].sum()) / float(initial_amount)
                    rl_cost_ratio = float(df_trades["trade_fee"].sum()) / float(initial_amount)
                else:
                    rl_turn_ratio, rl_cost_ratio = 0.0, 0.0

                rl_metrics = compute_six_metrics(df_account_value, initial_amount, rl_turn_ratio, rl_cost_ratio)
                
                cur_sharpe = float(rl_metrics["夏普比率"])
                cur_dd = float(rl_metrics["最大回撤"].strip('%'))
                
                total_time_sec = time.time() - start_time
                seed_results.append({
                    "seed": seed, 
                    "sharpe": cur_sharpe, 
                    "drawdown": cur_dd,
                    "time_min": total_time_sec / 60.0
                })
                
                print(f"   回测完毕！当次夏普比率: {cur_sharpe:.4f} (基准为: {bm_mom_sharpe:.4f})")
                
                # 把模型和回测曲线先存在内存里
                seed_artifacts[seed] = {
                    "model": model,
                    "df_account_value": df_account_value
                }

            except Exception as e:
                print(f"   回测失败: {e}")
                return

    print("\n================ 阶段1 最终实验结论 ================")
    rl_sharpes = [res["sharpe"] for res in seed_results]
    rl_drawdowns = [res["drawdown"] for res in seed_results]
    rl_times = [res["time_min"] for res in seed_results]
    
    mean_sharpe = np.mean(rl_sharpes)
    std_sharpe = np.std(rl_sharpes)
    mean_dd = np.mean(rl_drawdowns)
    avg_time = np.mean(rl_times)
    
    print(f"RL 策略平均夏普比率: {mean_sharpe:.4f} ± {std_sharpe:.4f}")
    print(f"Top5 动量基准夏普: {bm_mom_sharpe:.4f}")
    print(f"RL 策略平均最大回撤: {mean_dd:.2f}%")
    print(f"平均单次训练+回测耗时: {avg_time:.2f} 分钟")
    
    pass_count = sum(1 for s in rl_sharpes if s > bm_mom_sharpe)
    print(f"跑赢基准的 Seed 数量: {pass_count} / {len(SEEDS)}")
    print("==========================================================")

    # 寻找最接近均值的模型，执行保存与画图
    if len(seed_results) > 0:
        # 计算谁最接近平均夏普
        closest_seed = min(seed_results, key=lambda x: abs(x["sharpe"] - mean_sharpe))["seed"]
        closest_sharpe = next(res["sharpe"] for res in seed_results if res["seed"] == closest_seed)
        
        # 提取模型和数据
        median_model = seed_artifacts[closest_seed]["model"]
        median_df = seed_artifacts[closest_seed]["df_account_value"]
        
        # 保存中位数模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        median_model_name = f"ppo_agent_median_{timestamp}_seed{closest_seed}.zip"
        median_model_path = os.path.join(EXP_PATHS["model"], median_model_name)
        median_model.save(median_model_path)
        print(f"\n已选取表现最贴近均值的模型 (Seed {closest_seed}, 夏普 {closest_sharpe:.4f}) 并保存为: {median_model_name}")

        # 仅使用中位数模型的数据进行绘图
        if TASK_CONTROL["do_plotting"]:
            print("INFO: 正在生成对比图表 (展示中位数模型的净值曲线)...")
            try:
                plot_comparison(
                    median_df,
                    {
                        "Benchmark: Equal-Weight (5-day)": df_bm_equal,
                        "Benchmark: Top5 Momentum (5-day)": df_bm_mom,
                    },
                )
                print("SUCCESS: 曲线对比图已保存。")
            except Exception as e:
                print(f"ERROR: 绘图失败: {e}")
            
            # 更新报告
            perf_stats = {}
            for res in seed_results:
                marker = "⭐ (本轮代表)" if res["seed"] == closest_seed else ""
                perf_stats[f"Seed_{res['seed']}_夏普比率"] = f"{res['sharpe']:.4f} (回撤: {res['drawdown']:.2f}%) {marker}"
            
            perf_stats["---"] = "---"
            perf_stats["汇总_RL_平均夏普"] = f"{mean_sharpe:.4f} ± {std_sharpe:.4f}"
            perf_stats["汇总_RL_平均最大回撤"] = f"{mean_dd:.2f}%"
            perf_stats["汇总_选取基准模型"] = f"Seed {closest_seed} (夏普 {closest_sharpe:.4f}，最接近均值)"
            perf_stats["---"] = "---"
            
            for k, v in bm_mom_metrics.items():
                perf_stats[f"Top5动量Benchmark_{k}"] = v
                
            finalize_experiment(median_df, perf_stats)
            print(f"SUCCESS: 实验报告已更新至: {EXP_PATHS['root']}")

if __name__ == "__main__":
    run_experiment_pipeline()
import json
import os
import sys
from datetime import datetime, time as dtime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tushare as ts
from stable_baselines3 import PPO

# 确保项目根目录可导入
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.base_config import DOCS_DIR, TECHNICAL_INDICATORS
from src.envs.env_stocktrading import StockTradingEnv
from src.data_processing.fetch_40_pool import (
    DEFAULT_TOKEN,
    build_universe_df,
    clean_and_engineer,
    fetch_daily_with_retry,
)


# ===== Live Demo 基本配置 =====
LIVE_START_DATE = "2026-01-01"  # 冷启动历史数据起点（用于技术指标）
DEFAULT_INITIAL_CASH = 200_000  # 模拟盘初始资金（默认20万）
BUY_COST_PCT = 0.001
SELL_COST_PCT = 0.001
REBALANCE_WINDOW = 5
LOT_SIZE = 100

# 默认模型路径（可通过环境变量覆盖）
DEFAULT_MODEL_PATH = os.path.join(
    DOCS_DIR,
    "experiments",
    "20260311_0254_rolling_experiment",
    "seeds",
    "seed_7",
    "checkpoints",
    "ppo_Roll_4_Test_2025.zip",
)
MODEL_PATH = os.getenv("LIVE_MODEL_PATH", DEFAULT_MODEL_PATH)

# Live Demo 输出目录
LIVE_DIR = os.path.join(DOCS_DIR, "live_demo")
INPUT_DIR = os.path.join(LIVE_DIR, "inputs")
TABLE_DIR = os.path.join(LIVE_DIR, "tables")
LOG_DIR = os.path.join(LIVE_DIR, "logs")
PLOT_DIR = os.path.join(LIVE_DIR, "plots")
STATE_PATH = os.path.join(LIVE_DIR, "state.json")

RAW_LIVE_PATH = os.path.join(INPUT_DIR, "raw_daily_40.csv")
PROCESSED_LIVE_PATH = os.path.join(INPUT_DIR, "processed_40_pool_live.csv")


def _ensure_dirs() -> None:
    for p in [LIVE_DIR, INPUT_DIR, TABLE_DIR, LOG_DIR, PLOT_DIR]:
        os.makedirs(p, exist_ok=True)


def _parse_date(date_str: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(date_str, format="%Y-%m-%d")
    except Exception as exc:
        raise ValueError("日期格式错误，请使用 YYYY-MM-DD") from exc


def _is_market_closed(input_date: pd.Timestamp) -> bool:
    now = datetime.now()
    today = pd.Timestamp(now.date())
    if input_date > today:
        return False
    if input_date < today:
        return True
    return now.time() >= dtime(15, 30)


def _normalize_ticker(code: str) -> str:
    code = code.strip().upper()
    if code.endswith(".SH") or code.endswith(".SZ"):
        return code
    if len(code) == 6 and code.isdigit():
        return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
    return code


def _load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _fetch_range(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取固定40股票池在指定区间的日线数据。"""
    token = os.getenv("TUSHARE_TOKEN", DEFAULT_TOKEN)
    if not token:
        raise RuntimeError("缺少 Tushare Token，请设置 TUSHARE_TOKEN 环境变量或更新默认值。")
    ts.set_token(token)
    pro = ts.pro_api()

    universe_df = build_universe_df()
    all_frames: List[pd.DataFrame] = []
    for _, row in universe_df.iterrows():
        ts_code = row["ts_code"]
        df = fetch_daily_with_retry(
            pro=pro,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            continue
        df = df.rename(columns={"ts_code": "tic", "trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])
        df["name"] = row["name"]
        df["industry"] = row["industry"]
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()
    raw_df = pd.concat(all_frames, ignore_index=True)
    raw_df = raw_df.sort_values(["date", "tic"]).reset_index(drop=True)
    return raw_df


def _update_live_data(target_date: str) -> None:
    """若本地数据不足，则自动拉取并更新 raw/processed。"""
    _ensure_dirs()
    target_ts = _parse_date(target_date)

    if os.path.exists(RAW_LIVE_PATH):
        raw_df = pd.read_csv(RAW_LIVE_PATH)
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
        max_date = raw_df["date"].max()
        if pd.isna(max_date) or max_date < target_ts:
            start_date = (
                (max_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                if pd.notna(max_date)
                else LIVE_START_DATE.replace("-", "")
            )
            end_date = target_ts.strftime("%Y%m%d")
            print(f"INFO: 本地数据不足，开始拉取 {start_date} ~ {end_date} ...")
            new_raw = _fetch_range(start_date, end_date)
            if not new_raw.empty:
                raw_df = pd.concat([raw_df, new_raw], ignore_index=True)
                raw_df = raw_df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"])
            raw_df.to_csv(RAW_LIVE_PATH, index=False, encoding="utf-8-sig")
        else:
            print("INFO: 本地数据已覆盖目标日期，无需更新。")
    else:
        start_date = LIVE_START_DATE.replace("-", "")
        end_date = target_ts.strftime("%Y%m%d")
        print(f"INFO: 初始化拉取 {start_date} ~ {end_date} ...")
        raw_df = _fetch_range(start_date, end_date)
        if raw_df.empty:
            raise RuntimeError("拉取数据失败，请检查网络或 Token。")
        raw_df.to_csv(RAW_LIVE_PATH, index=False, encoding="utf-8-sig")

    raw_df = pd.read_csv(RAW_LIVE_PATH)
    processed_df = clean_and_engineer(raw_df)
    processed_df.to_csv(PROCESSED_LIVE_PATH, index=False, encoding="utf-8-sig")
    print(f"SUCCESS: 已更新处理后数据: {PROCESSED_LIVE_PATH}")


def _load_processed_until(target_date: str) -> pd.DataFrame:
    if not os.path.exists(PROCESSED_LIVE_PATH):
        raise FileNotFoundError("未找到处理后数据，请先运行数据更新。")
    df = pd.read_csv(PROCESSED_LIVE_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    target_ts = _parse_date(target_date)
    df = df[df["date"] <= target_ts].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _prompt_holdings(tickers: List[str]) -> Tuple[Dict[str, int], List[dict], float]:
    """交互式输入持仓。返回持仓股数、输入记录、现金余额。"""
    input_count = input("请输入持仓数量(0-5，回车跳过): ").strip()
    if input_count == "":
        return {}, [], np.nan
    try:
        count = int(input_count)
    except Exception:
        raise ValueError("持仓数量必须为整数。")
    count = max(0, min(5, count))

    holdings: Dict[str, int] = {}
    input_rows: List[dict] = []
    for i in range(count):
        raw = input(f"请输入第{i + 1}只股票(代码,股数,金额): ").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            raise ValueError("格式错误，应为 代码,股数,金额(可留空)")
        tic = _normalize_ticker(parts[0])
        shares = int(float(parts[1]))
        amount = float(parts[2]) if len(parts) >= 3 and parts[2] else np.nan

        if tic not in tickers:
            print(f"WARN: 股票 {tic} 不在40股票池内，将被忽略。")
            continue
        holdings[tic] = int(shares)
        input_rows.append({"tic": tic, "shares": int(shares), "amount": amount})

    cash_raw = input("请输入现金余额(回车自动计算): ").strip()
    cash = float(cash_raw) if cash_raw else np.nan
    return holdings, input_rows, cash


def _get_next_trading_date(date_list: List[pd.Timestamp], target_ts: pd.Timestamp) -> str:
    for d in date_list:
        if d > target_ts:
            return pd.Timestamp(d).strftime("%Y-%m-%d")
    return ""


def _prompt_account_status() -> Tuple[float, float]:
    """可选输入总资产与可用现金，回车跳过。"""
    total_raw = input("请输入当前总资产/总市值(回车跳过): ").strip()
    cash_raw = input("请输入当前可用现金(回车跳过): ").strip()

    total_asset = float(total_raw) if total_raw else np.nan
    cash_asset = float(cash_raw) if cash_raw else np.nan
    return total_asset, cash_asset


def _append_csv(path: str, rows: List[dict], columns: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[columns]
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def _resolve_initial_cash(state: dict) -> float:
    if "initial_cash" in state:
        return float(state["initial_cash"])
    raw = input(f"请输入初始资金(回车默认 {DEFAULT_INITIAL_CASH}): ").strip()
    val = float(raw) if raw else float(DEFAULT_INITIAL_CASH)
    state["initial_cash"] = float(val)
    _save_state(state)
    return float(val)


def run_live_demo() -> None:
    _ensure_dirs()

    decision_date_str = input("请输入交易日期(YYYY-MM-DD): ").strip()
    decision_date = _parse_date(decision_date_str)

    preview_mode = not _is_market_closed(decision_date)
    data_end_date = decision_date
    if preview_mode:
        data_end_date = decision_date - pd.Timedelta(days=1)
        print("WARN: 当前日期未收盘，将使用最近收盘日数据生成策略。")

    _update_live_data(data_end_date.strftime("%Y-%m-%d"))
    processed_df = _load_processed_until(decision_date_str)

    if processed_df.empty:
        print("ERROR: 处理后数据为空，请检查数据范围。")
        return

    stock_dim = processed_df["tic"].nunique()
    if stock_dim <= 0:
        print("ERROR: 股票数量不足。")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: 找不到模型文件: {MODEL_PATH}")
        return

    state = _load_state()
    initial_cash = _resolve_initial_cash(state)

    env = StockTradingEnv(
        df=processed_df,
        stock_dim=stock_dim,
        hmax=1000,
        initial_amount=initial_cash,
        buy_cost_pct=[BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[SELL_COST_PCT] * stock_dim,
        tech_indicator_list=TECHNICAL_INDICATORS,
        rebalance_window=REBALANCE_WINDOW,
        lot_size=LOT_SIZE,
    )

    # 确定日期索引
    date_list = list(env.dates)
    target_ts = pd.Timestamp(decision_date)
    effective_ts = max([d for d in date_list if d <= target_ts], default=None)
    if effective_ts is None:
        print("ERROR: 无可用交易数据，请检查数据覆盖范围。")
        return
    if effective_ts != target_ts:
        preview_mode = True
        print(f"WARN: 使用最近收盘日 {pd.Timestamp(effective_ts).strftime('%Y-%m-%d')} 数据生成策略。")

    date_idx = date_list.index(effective_ts)
    env.day = int(date_idx)
    env._update_market_data()

    if "live_start_feature_date" not in state:
        state["live_start_feature_date"] = pd.Timestamp(effective_ts).strftime("%Y-%m-%d")
        state["live_start_decision_date"] = decision_date_str
        _save_state(state)
        print(f"INFO: 已设置冷启动起点为 {state['live_start_feature_date']}")

    start_ts = pd.Timestamp(_parse_date(state["live_start_feature_date"]))
    if start_ts not in date_list:
        print("ERROR: 冷启动起点不在数据中，请检查输入日期或数据更新。")
        return

    start_idx = date_list.index(start_ts)
    day_index = int(date_idx - start_idx)
    is_rebalance_day = (day_index % REBALANCE_WINDOW == 0)
    next_trade_date = _get_next_trading_date(date_list, pd.Timestamp(effective_ts))

    # 输入持仓（可选）
    manual_holdings, input_rows, cash_input = _prompt_holdings(env.tickers)
    holdings = np.zeros(stock_dim, dtype=np.int32)
    for tic, shares in manual_holdings.items():
        holdings[env.ticker_to_idx[tic]] = int(shares)

    if manual_holdings:
        env.holdings = holdings.astype(np.float32)
        holdings_value = float(np.sum(env.holdings * env.prices))
        if np.isnan(cash_input):
            cash = max(0.0, initial_cash - holdings_value)
        else:
            cash = float(cash_input)
        env.cash = cash
        state_source = "manual"
    else:
        prev_holdings = state.get("holdings", {})
        for tic, shares in prev_holdings.items():
            if tic in env.ticker_to_idx:
                holdings[env.ticker_to_idx[tic]] = int(shares)
        env.holdings = holdings.astype(np.float32)
        env.cash = float(state.get("cash", initial_cash))
        state_source = "auto"
        holdings_value = float(np.sum(env.holdings * env.prices))

    input_total_asset, input_cash = _prompt_account_status()
    if not np.isnan(input_cash):
        env.cash = float(input_cash)
    elif not np.isnan(input_total_asset):
        env.cash = max(0.0, float(input_total_asset) - holdings_value)

    if not np.isnan(input_total_asset) and not np.isnan(input_cash):
        if abs((input_cash + holdings_value) - input_total_asset) > 1e-6:
            print("INFO: 总资产与现金/持仓估算不一致，已优先使用现金输入。")

    total_asset = float(env.cash + holdings_value)

    model = PPO.load(MODEL_PATH)
    obs = env._get_state()
    action, _ = model.predict(obs, deterministic=True)

    trade_rows: List[dict] = []
    signal_rows: List[dict] = []
    holding_rows: List[dict] = []
    account_rows: List[dict] = []
    input_rows_to_log: List[dict] = []

    trade_fee_total = 0.0
    traded_notional = 0.0

    if is_rebalance_day:
        target_weights = env._scores_to_target_weights(action)
        target_shares = env._target_shares_from_weights(target_weights, total_asset)
        current_shares = env.holdings.copy().astype(np.int32)
        deltas = target_shares - current_shares

        selected_idx = np.argsort(action)[::-1][: env.top_k]
        selected_tics = [env.tickers[i] for i in selected_idx]
        selected_weights = [float(target_weights[i]) for i in selected_idx]
        selected_values = [float(target_weights[i] * total_asset) for i in selected_idx]
        selected_shares = [int(target_shares[i]) for i in selected_idx]
        selected_prices = [float(env.prices[i]) for i in selected_idx]

        print("INFO: 今日为调仓日，已生成 Top-5 策略。")
        print("INFO: 目标股票：", ", ".join(selected_tics))
        print("INFO: 建议目标持仓(股数为100股整数倍)：")
        for tic, price, shares, value in zip(selected_tics, selected_prices, selected_shares, selected_values):
            print(f"  - {tic} | 价格: {price:.2f} | 目标股数: {shares} | 目标金额: {value:.2f}")
        if next_trade_date:
            print(f"INFO: 下一交易日预计为 {next_trade_date}")

        apply_action = input("是否执行调仓建议？(Y/N): ").strip().lower()
        accepted = apply_action in {"y", "yes"}

        if accepted:
            # 先卖后买
            for i in range(stock_dim):
                if deltas[i] < 0:
                    sold = env._sell(i, abs(int(deltas[i])))
                    if sold > 0:
                        fee = sold * float(env.prices[i]) * float(env.sell_cost_pct[i])
                        notional = sold * float(env.prices[i])
                        trade_fee_total += fee
                        traded_notional += notional
                        trade_rows.append(
                            {
                                "date": decision_date_str,
                                "tic": env.tickers[i],
                                "action": "SELL",
                                "shares": int(sold),
                                "price": float(env.prices[i]),
                                "notional": float(notional),
                                "fee": float(fee),
                            }
                        )

            for i in range(stock_dim):
                if deltas[i] > 0:
                    bought = env._buy(i, int(deltas[i]))
                    if bought > 0:
                        fee = bought * float(env.prices[i]) * float(env.buy_cost_pct[i])
                        notional = bought * float(env.prices[i])
                        trade_fee_total += fee
                        traded_notional += notional
                        trade_rows.append(
                            {
                                "date": decision_date_str,
                                "tic": env.tickers[i],
                                "action": "BUY",
                                "shares": int(bought),
                                "price": float(env.prices[i]),
                                "notional": float(notional),
                                "fee": float(fee),
                            }
                        )
        else:
            print("INFO: 已选择不执行调仓，维持当前持仓。")

        signal_rows.append(
            {
                "date": decision_date_str,
                "feature_date": pd.Timestamp(effective_ts).strftime("%Y-%m-%d"),
                "preview_mode": bool(preview_mode),
                "is_rebalance_day": True,
                "accepted": accepted,
                "next_trading_date": next_trade_date,
                "selected_tickers": ";".join(selected_tics),
                "target_weights": ";".join([f"{w:.4f}" for w in selected_weights]),
                "target_values": ";".join([f"{v:.2f}" for v in selected_values]),
                "cash_before": total_asset - holdings_value,
                "total_asset_before": total_asset,
                "cash_after": float(env.cash),
                "total_asset_after": float(env._get_total_asset()),
                "trade_fee_total": float(trade_fee_total),
                "turnover_ratio": float(traded_notional / max(total_asset, 1e-8)),
            }
        )
    else:
        signal_rows.append(
            {
                "date": decision_date_str,
                "feature_date": pd.Timestamp(effective_ts).strftime("%Y-%m-%d"),
                "preview_mode": bool(preview_mode),
                "is_rebalance_day": False,
                "accepted": False,
                "next_trading_date": next_trade_date,
                "selected_tickers": "",
                "target_weights": "",
                "target_values": "",
                "cash_before": total_asset - holdings_value,
                "total_asset_before": total_asset,
                "cash_after": float(env.cash),
                "total_asset_after": float(env._get_total_asset()),
                "trade_fee_total": 0.0,
                "turnover_ratio": 0.0,
            }
        )
        print("INFO: 非调仓日，维持当前策略。")
        if next_trade_date:
            print(f"INFO: 下一交易日预计为 {next_trade_date}")

    # 记录持仓与账户
    for i, tic in enumerate(env.tickers):
        holding_rows.append(
            {
                "date": decision_date_str,
                "tic": tic,
                "shares": int(env.holdings[i]),
                "price": float(env.prices[i]),
                "market_value": float(env.holdings[i] * env.prices[i]),
            }
        )
    account_rows.append(
        {
            "date": decision_date_str,
            "feature_date": pd.Timestamp(effective_ts).strftime("%Y-%m-%d"),
            "preview_mode": bool(preview_mode),
            "input_total_asset": "" if np.isnan(input_total_asset) else float(input_total_asset),
            "input_cash": "" if np.isnan(input_cash) else float(input_cash),
            "cash": float(env.cash),
            "holdings_value": float(np.sum(env.holdings * env.prices)),
            "total_asset": float(env._get_total_asset()),
            "trade_fee_total": float(trade_fee_total),
            "turnover_ratio": float(traded_notional / max(total_asset, 1e-8)),
            "is_rebalance_day": bool(is_rebalance_day),
        }
    )

    # 记录输入
    if input_rows:
        for row in input_rows:
            row.update({"date": decision_date_str, "state_source": state_source})
            input_rows_to_log.append(row)

    _append_csv(
        os.path.join(TABLE_DIR, "daily_signals.csv"),
        signal_rows,
        [
            "date",
            "feature_date",
            "preview_mode",
            "is_rebalance_day",
            "accepted",
            "next_trading_date",
            "selected_tickers",
            "target_weights",
            "target_values",
            "cash_before",
            "total_asset_before",
            "cash_after",
            "total_asset_after",
            "trade_fee_total",
            "turnover_ratio",
        ],
    )
    _append_csv(
        os.path.join(TABLE_DIR, "daily_trades.csv"),
        trade_rows,
        ["date", "tic", "action", "shares", "price", "notional", "fee"],
    )
    _append_csv(
        os.path.join(TABLE_DIR, "daily_holdings.csv"),
        holding_rows,
        ["date", "tic", "shares", "price", "market_value"],
    )
    _append_csv(
        os.path.join(TABLE_DIR, "daily_account.csv"),
        account_rows,
        [
            "date",
            "feature_date",
            "preview_mode",
            "input_total_asset",
            "input_cash",
            "cash",
            "holdings_value",
            "total_asset",
            "trade_fee_total",
            "turnover_ratio",
            "is_rebalance_day",
        ],
    )
    _append_csv(
        os.path.join(TABLE_DIR, "user_inputs.csv"),
        input_rows_to_log,
        ["date", "tic", "shares", "amount", "state_source"],
    )

    # 更新状态
    state["last_date"] = decision_date_str
    state["last_feature_date"] = pd.Timestamp(effective_ts).strftime("%Y-%m-%d")
    state["cash"] = float(env.cash)
    state["holdings"] = {tic: int(env.holdings[i]) for i, tic in enumerate(env.tickers)}
    _save_state(state)
    print("SUCCESS: Live demo 执行完成，日志与CSV已写入 docs/live_demo。")

    # ===== 可选绘图：从持仓起始日到当前日期的收益曲线对比 =====
    # 说明：
    # 1) 使用 daily_account.csv 作为策略净值序列
    # 2) 使用等权基准（含手续费、5日调仓）作为对照
    # 3) 绘图输出到 docs/live_demo/plots/{日期}.png
    # 4) 仅在已有账户记录时绘图
    try:
        import matplotlib.pyplot as plt

        account_path = os.path.join(TABLE_DIR, "daily_account.csv")
        if os.path.exists(account_path):
            df_acc = pd.read_csv(account_path)
            df_acc["date"] = pd.to_datetime(df_acc["date"])
            df_acc = df_acc.sort_values("date").reset_index(drop=True)

            start_date = df_acc["date"].min()
            end_date = df_acc["date"].max()

            df_slice = processed_df.copy()
            df_slice["date"] = pd.to_datetime(df_slice["date"])
            df_slice = df_slice[(df_slice["date"] >= start_date) & (df_slice["date"] <= end_date)]

            price_pivot = (
                df_slice.pivot(index="date", columns="tic", values="close")
                .sort_index()
                .fillna(method="ffill")
            )

            if not price_pivot.empty:
                daily_ret = price_pivot.pct_change().fillna(0.0)
                n_assets = price_pivot.shape[1]
                target_w = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

                nav = float(initial_cash)
                weights = np.zeros(n_assets, dtype=np.float64)
                rows = [{"date": price_pivot.index[0], "benchmark": nav}]

                for i in range(1, len(price_pivot.index)):
                    r = daily_ret.iloc[i].to_numpy(dtype=np.float64)
                    nav *= 1.0 + float(np.dot(weights, r))

                    gross_weights = weights * (1.0 + r)
                    gross_sum = float(np.sum(gross_weights))
                    weights = gross_weights / gross_sum if gross_sum > 0 else np.zeros_like(weights)

                    if i % REBALANCE_WINDOW == 0:
                        buy_turnover = float(np.maximum(target_w - weights, 0.0).sum())
                        sell_turnover = float(np.maximum(weights - target_w, 0.0).sum())
                        fee_ratio = buy_turnover * BUY_COST_PCT + sell_turnover * SELL_COST_PCT
                        nav *= max(0.0, 1.0 - fee_ratio)
                        weights = target_w.copy()

                    rows.append({"date": price_pivot.index[i], "benchmark": nav})

                df_bm = pd.DataFrame(rows)
                df_plot = df_acc.merge(df_bm, on="date", how="left")

                plt.figure(figsize=(12, 6))
                plt.plot(df_plot["date"], df_plot["total_asset"], label="Agent")
                plt.plot(df_plot["date"], df_plot["benchmark"], label="Benchmark (Equal-Weight)")
                plt.title("Live Demo Performance Comparison")
                plt.xlabel("Date")
                plt.ylabel("Account Value")
                plt.grid(True)
                plt.legend()
                plot_path = os.path.join(PLOT_DIR, f"{decision_date_str}.png")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    run_live_demo()

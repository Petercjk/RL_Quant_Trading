import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import tushare as ts
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.base_config import TECHNICAL_INDICATORS


DEFAULT_TOKEN = "d00985e44d97b66607e1bb3209880a913e9e651e861477dbbdbfacaf"
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20251231"
DEFAULT_RAW_OUTPUT = "data/raw/40_pool.csv"
DEFAULT_PROCESSED_OUTPUT = "data/processed/processed_40_pool.csv"
DEFAULT_UNIVERSE_OUTPUT = "data/raw/40_pool_universe.csv"


# 固定40只股票池（Top-5轮动基础池），包含行业标签用于后续分析。
STOCK_POOL: List[Dict[str, str]] = [
    # 银行
    {"code": "601166", "name": "兴业银行", "industry": "银行"},
    {"code": "601398", "name": "工商银行", "industry": "银行"},
    {"code": "601939", "name": "建设银行", "industry": "银行"},
    {"code": "600016", "name": "民生银行", "industry": "银行"},
    {"code": "600036", "name": "招商银行", "industry": "银行"},
    {"code": "600000", "name": "浦发银行", "industry": "银行"},
    # 保险金融 / 证券
    {"code": "601601", "name": "中国太保", "industry": "保险金融"},
    {"code": "601628", "name": "中国人寿", "industry": "保险金融"},
    {"code": "600030", "name": "中信证券", "industry": "证券"},
    # 能源石油
    {"code": "601857", "name": "中国石油", "industry": "能源石油"},
    {"code": "600028", "name": "中国石化", "industry": "能源石油"},
    {"code": "601088", "name": "中国神华", "industry": "能源石油"},
    # 电力/公用事业
    {"code": "600900", "name": "长江电力", "industry": "电力/公用事业"},
    {"code": "600011", "name": "华能国际", "industry": "电力/公用事业"},
    {"code": "600642", "name": "申能股份", "industry": "电力/公用事业"},
    {"code": "601991", "name": "大唐发电", "industry": "电力/公用事业"},
    # 交通运输
    {"code": "601111", "name": "中国国航", "industry": "交通运输"},
    {"code": "600029", "name": "南方航空", "industry": "交通运输"},
    {"code": "600009", "name": "上海机场", "industry": "交通运输"},
    {"code": "600377", "name": "宁沪高速", "industry": "交通运输"},
    # 消费食品
    {"code": "600887", "name": "伊利股份", "industry": "消费食品"},
    {"code": "000895", "name": "双汇发展", "industry": "消费食品"},
    {"code": "000729", "name": "燕京啤酒", "industry": "消费食品"},
    {"code": "000848", "name": "承德露露", "industry": "消费食品"},
    {"code": "600597", "name": "光明乳业", "industry": "消费食品"},
    # 医药
    {"code": "600535", "name": "天士力", "industry": "医药"},
    {"code": "600085", "name": "同仁堂", "industry": "医药"},
    {"code": "000999", "name": "华润三九", "industry": "医药"},
    {"code": "600812", "name": "华北制药", "industry": "医药"},
    # 地产建筑
    {"code": "000002", "name": "万科A", "industry": "地产建筑"},
    {"code": "600048", "name": "保利发展", "industry": "地产建筑"},
    {"code": "601668", "name": "中国建筑", "industry": "地产建筑"},
    # 周期/材料
    {"code": "600019", "name": "宝钢股份", "industry": "周期/材料"},
    {"code": "601600", "name": "中国铝业", "industry": "周期/材料"},
    {"code": "601899", "name": "紫金矿业", "industry": "周期/材料"},
    {"code": "601919", "name": "中远海控", "industry": "周期/材料"},
    # 科技/制造
    {"code": "000725", "name": "京东方A", "industry": "科技/制造"},
    {"code": "000063", "name": "中兴通讯", "industry": "科技/制造"},
    {"code": "600522", "name": "中天科技", "industry": "科技/制造"},
    {"code": "000100", "name": "TCL科技", "industry": "科技/制造"},
]


def to_ts_code(code: str) -> str:
    """将6位股票代码转换为Tushare标准代码（.SH/.SZ）。"""
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"


def build_universe_df() -> pd.DataFrame:
    """把固定股票池展开成DataFrame，包含顺序、代码、名称和行业。"""
    rows = []
    for idx, item in enumerate(STOCK_POOL, start=1):
        rows.append(
            {
                "rank": idx,
                "code": item["code"],
                "ts_code": to_ts_code(item["code"]),
                "name": item["name"],
                "industry": item["industry"],
            }
        )
    return pd.DataFrame(rows)


def fetch_daily_with_retry(
    pro: ts.pro_api,
    ts_code: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """按单只股票抓取日线，失败时重试，避免偶发网络/频率异常。"""
    for attempt in range(1, max_retries + 1):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:
            print(f"WARN: {ts_code} fetch failed (attempt {attempt}/{max_retries}): {exc}")
            if attempt == max_retries:
                return pd.DataFrame()
            time.sleep(float(attempt))
    return pd.DataFrame()


def fetch_raw_data(
    token: str,
    start_date: str,
    end_date: str,
    sleep_sec: float,
    raw_output: str,
    universe_output: str,
) -> pd.DataFrame:
    """抓取40只股票的原始日线数据，并保存raw与universe文件。"""
    ts.set_token(token)
    pro = ts.pro_api()

    universe_df = build_universe_df()
    os.makedirs(os.path.dirname(universe_output), exist_ok=True)
    universe_df.to_csv(universe_output, index=False, encoding="utf-8-sig")
    print(f"SUCCESS: Universe saved to {universe_output}")

    all_frames: List[pd.DataFrame] = []
    failed_codes: List[str] = []

    for i, row in universe_df.iterrows():
        ts_code = row["ts_code"]
        df = fetch_daily_with_retry(
            pro=pro,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            failed_codes.append(ts_code)
        else:
            all_frames.append(df)

        print(f"INFO: Progress {i + 1}/{len(universe_df)} - {ts_code}")
        time.sleep(max(0.0, sleep_sec))

    if not all_frames:
        raise RuntimeError("No daily data fetched. Check token/permissions/network.")

    raw_df = pd.concat(all_frames, ignore_index=True)
    raw_df = raw_df.rename(
        columns={
            "ts_code": "tic",
            "trade_date": "date",
            "vol": "volume",
        }
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"], format="%Y%m%d", errors="coerce")
    raw_df = raw_df.dropna(subset=["date"]).copy()
    raw_df["date"] = raw_df["date"].dt.strftime("%Y-%m-%d")

    meta_map = universe_df.rename(columns={"ts_code": "tic"})
    raw_df = raw_df.merge(meta_map[["tic", "name", "industry"]], on="tic", how="left")
    raw_df = raw_df.sort_values(["date", "tic"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(raw_output), exist_ok=True)
    raw_df.to_csv(raw_output, index=False, encoding="utf-8-sig")
    print(f"SUCCESS: Raw data saved to {raw_output}")
    print(
        "INFO: Raw rows={rows}, tickers={tickers}, date range={dmin} -> {dmax}".format(
            rows=len(raw_df),
            tickers=raw_df["tic"].nunique(),
            dmin=raw_df["date"].min(),
            dmax=raw_df["date"].max(),
        )
    )
    if failed_codes:
        print(f"WARN: Failed tickers ({len(failed_codes)}): {failed_codes}")
    else:
        print("SUCCESS: All 40 tickers fetched.")
    return raw_df


def clean_and_engineer(raw_df: pd.DataFrame) -> pd.DataFrame:
    """执行数据对齐、特征工程和自定义指标计算，输出训练可用表。"""
    # 对齐交易日和股票维度，避免缺失导致训练维度变化。
    unique_dates = sorted(raw_df["date"].unique())
    tickers = sorted(raw_df["tic"].unique())

    valid_tickers = []
    dropped_tickers = []
    for tic in tickers:
        coverage = len(raw_df[raw_df["tic"] == tic])
        if coverage > len(unique_dates) * 0.8:
            valid_tickers.append(tic)
        else:
            dropped_tickers.append(tic)

    if not valid_tickers:
        raise RuntimeError("No ticker left after coverage filter.")
    if dropped_tickers:
        print(f"WARN: Dropped by coverage filter ({len(dropped_tickers)}): {dropped_tickers}")

    full_idx = pd.MultiIndex.from_product([unique_dates, valid_tickers], names=["date", "tic"])
    aligned = raw_df.set_index(["date", "tic"]).reindex(full_idx).reset_index()

    # 价格类字段前向填充，成交量/成交额缺失按0处理。
    for col in ["open", "high", "low", "close"]:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned[["open", "high", "low", "close"]] = aligned.groupby("tic")[
        ["open", "high", "low", "close"]
    ].ffill()
    aligned["volume"] = aligned["volume"].fillna(0) if "volume" in aligned.columns else 0
    aligned["amount"] = aligned["amount"].fillna(0) if "amount" in aligned.columns else 0
    aligned = aligned.groupby("tic", group_keys=False).apply(lambda x: x.bfill())

    # 行业与股票名称在原始数据中是常量，按股票维度向前向后补齐。
    if "name" in aligned.columns:
        aligned["name"] = aligned.groupby("tic")["name"].ffill().bfill()
    if "industry" in aligned.columns:
        aligned["industry"] = aligned.groupby("tic")["industry"].ffill().bfill()

    # A股日线在部分区间可能无法稳定计算 turbulence，这里加入降级处理
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECHNICAL_INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    try:
        processed = fe.preprocess_data(aligned)
    except Exception:
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=TECHNICAL_INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(aligned)
        if "turbulence" not in processed.columns:
            processed["turbulence"] = 0.0
    processed = processed.sort_values(["tic", "date"]).reset_index(drop=True)

    # 追加基础市场特征，均按单股票时间序列计算，避免截面混算。
    def _add_features(group: pd.DataFrame) -> pd.DataFrame:
        """对单只股票计算滚动特征，避免跨股票污染。"""
        g = group.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        open_ = g["open"]
        volume = g["volume"]
        amount = g["amount"]
        pre_close = g["pre_close"] if "pre_close" in g.columns else close.shift(1)

        # 收益类指标：日对数收益、20日累计收益
        g["log_return"] = np.log(close / pre_close.replace(0, np.nan))
        g["return_20"] = close / close.shift(20) - 1.0

        # 动量指标：3个月、6个月、12个月
        g["momentum_60"] = close / close.shift(60) - 1.0
        g["momentum_120"] = close / close.shift(120) - 1.0
        g["momentum_252"] = close / close.shift(252) - 1.0

        # 波动率指标：20日波动与年化波动
        g["volatility_20"] = g["log_return"].rolling(20).std()
        g["volatility_annual"] = g["volatility_20"] * np.sqrt(252.0)

        # 趋势指标：短中期均线
        g["sma_5"] = close.rolling(5).mean()
        g["sma_10"] = close.rolling(10).mean()
        g["sma_20"] = close.rolling(20).mean()
        g["sma_30"] = close.rolling(30).mean()
        g["sma_60"] = close.rolling(60).mean()

        # 均线偏离率
        g["bias_20"] = (close - g["sma_20"]) / g["sma_20"].replace(0, np.nan)
        g["bias_60"] = (close - g["sma_60"]) / g["sma_60"].replace(0, np.nan)

        # 价格波动：日振幅与日内收益
        g["amplitude"] = (high - low) / close.replace(0, np.nan)
        g["intraday_return"] = (close - open_) / open_.replace(0, np.nan)

        # 成交量/成交额指标
        g["amount_20_mean"] = amount.rolling(20).mean()
        g["volume_20_mean"] = volume.rolling(20).mean()
        g["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

        # 价格通道与区间位置
        g["high_20"] = high.rolling(20).max()
        g["low_20"] = low.rolling(20).min()
        g["position_20"] = (close - g["low_20"]) / (g["high_20"] - g["low_20"]).replace(0, np.nan)
        return g

    processed = processed.groupby("tic", group_keys=False).apply(_add_features)
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)

    # 保留滚动窗口初期的 NaN，避免将“无历史数据”误写成 0。
    # 训练前可按需要对关键特征列执行 dropna。
    key_feature_cols = [
        "return_20",
        "momentum_60",
        "momentum_120",
        "momentum_252",
        "volatility_20",
        "sma_20",
        "sma_60",
        "position_20",
    ]
    processed["feature_ready"] = (~processed[key_feature_cols].isna().any(axis=1)).astype(int)

    # 清洗结果说明：
    # 1) 每个交易日-股票都有记录；
    # 2) 基础行情缺失已按股票维度补齐；
    # 3) 滚动特征窗口初期保留 NaN，避免错误填0；
    # 4) 增加 feature_ready 标记供训练前筛样。
    return processed


def main() -> None:
    """脚本入口：若raw存在则直接处理，否则先抓取再处理。"""
    parser = argparse.ArgumentParser(description="Fetch and clean fixed 40-stock pool daily data.")
    parser.add_argument("--token", type=str, default=DEFAULT_TOKEN)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--raw-output", type=str, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--processed-output", type=str, default=DEFAULT_PROCESSED_OUTPUT)
    parser.add_argument("--universe-output", type=str, default=DEFAULT_UNIVERSE_OUTPUT)
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    args = parser.parse_args()

    if os.path.exists(args.raw_output):
        print(f"INFO: Found existing raw file, skip fetch: {args.raw_output}")
        raw_df = pd.read_csv(args.raw_output)
    else:
        raw_df = fetch_raw_data(
            token=args.token,
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_sec=args.sleep_sec,
            raw_output=args.raw_output,
            universe_output=args.universe_output,
        )

    processed_df = clean_and_engineer(raw_df)
    os.makedirs(os.path.dirname(args.processed_output), exist_ok=True)
    processed_df.to_csv(args.processed_output, index=False, encoding="utf-8-sig")
    print(f"SUCCESS: Processed data saved to {args.processed_output}")
    print(
        "INFO: Processed rows={rows}, tickers={tickers}, date range={dmin} -> {dmax}".format(
            rows=len(processed_df),
            tickers=processed_df["tic"].nunique(),
            dmin=processed_df["date"].min(),
            dmax=processed_df["date"].max(),
        )
    )


if __name__ == "__main__":
    main()

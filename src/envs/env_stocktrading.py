import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    """
    A股多资产交易环境：
    - 动作：对全部股票输出打分向量
    - 调仓：每 `rebalance_window` 个交易日执行一次
    - 组合：按打分选 Top-K 等权持仓
    - 奖励：收益 - 风险惩罚 - 换手惩罚
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: float,
        buy_cost_pct: list,
        sell_cost_pct: list,
        tech_indicator_list: list,
        reward_scaling: float = 1.0,
        top_k: int = 5,
        rebalance_window: int = 5,
        lot_size: int = 100,
        risk_penalty: float = 0.1,
        turnover_penalty: float = 0.01,
    ):
        """初始化环境、构建状态列、缓存按日行情切片。"""
        super().__init__()

        self.df = df.copy()
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = float(initial_amount)
        self.buy_cost_pct = np.array(buy_cost_pct, dtype=np.float32)
        self.sell_cost_pct = np.array(sell_cost_pct, dtype=np.float32)
        self.reward_scaling = float(reward_scaling)
        self.top_k = int(max(1, min(top_k, stock_dim)))
        self.rebalance_window = int(max(1, rebalance_window))
        self.lot_size = int(max(1, lot_size))
        self.risk_penalty = float(max(0.0, risk_penalty))
        self.turnover_penalty = float(max(0.0, turnover_penalty))

        # 固定股票顺序，避免跨日期错位。
        self.tickers = sorted(self.df["tic"].unique().tolist())
        if len(self.tickers) != self.stock_dim:
            raise ValueError(
                f"stock_dim={stock_dim} mismatch with df tickers={len(self.tickers)}"
            )
        self.ticker_to_idx = {tic: i for i, tic in enumerate(self.tickers)}

        # 特征列：配置指标 + 扩展指标 + 自动补充数值列。
        base_non_feature_cols = {"date", "tic", "name", "industry", "feature_ready"}
        preferred_extra_cols = [
            "log_return",
            "return_20",
            "momentum_60",
            "momentum_120",
            "momentum_252",
            "volatility_20",
            "volatility_annual",
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_30",
            "sma_60",
            "bias_20",
            "bias_60",
            "amplitude",
            "intraday_return",
            "amount_20_mean",
            "volume_20_mean",
            "volume_ratio",
            "high_20",
            "low_20",
            "position_20",
            "turbulence",
        ]

        tech_indicator_list = tech_indicator_list or []
        configured_cols = [
            col
            for col in tech_indicator_list
            if col in self.df.columns and col not in base_non_feature_cols
        ]
        extra_cols = [
            col
            for col in preferred_extra_cols
            if col in self.df.columns and col not in configured_cols
        ]
        auto_numeric_cols = [
            col
            for col in self.df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in configured_cols
            and col not in extra_cols
            and col not in {"feature_ready"}
        ]

        self.feature_cols = configured_cols + extra_cols + auto_numeric_cols
        if not self.feature_cols:
            raise ValueError("No valid feature columns found in processed data.")

        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values(["date", "tic"]).reset_index(drop=True)
        self.dates = self.df["date"].drop_duplicates().to_numpy()
        self.max_step = len(self.dates) - 1

        # 预缓存每日数据切片（按固定ticker顺序排列）。
        self.data_by_date = {}
        for d, group in self.df.groupby("date", sort=True):
            g = group.copy()
            g["tic_idx"] = g["tic"].map(self.ticker_to_idx)
            g = g.sort_values("tic_idx")
            if len(g) != self.stock_dim:
                raise ValueError(
                    f"Date {pd.Timestamp(d).date()} has {len(g)} tickers, expected {self.stock_dim}"
                )
            self.data_by_date[pd.Timestamp(d)] = g

        self.market_df = self._build_market_features()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.stock_dim,), dtype=np.float32
        )
        # 状态 = 现金占比 + 持仓权重 + 股票特征 + 市场特征 + 调仓标记
        state_dim = 1 + self.stock_dim + self.stock_dim * len(self.feature_cols) + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        """重置环境到第一个交易日并初始化账户。"""
        super().reset(seed=seed)

        self.day = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)

        # 回测诊断字段：保持与现有回测导出逻辑兼容。
        self.last_action_raw = np.zeros(self.stock_dim, dtype=np.float32)
        self.last_action_shares = np.zeros(self.stock_dim, dtype=np.int32)  # target delta shares
        self.last_trade_shares = np.zeros(self.stock_dim, dtype=np.int32)  # executed shares
        self.last_trade_prices = np.zeros(self.stock_dim, dtype=np.float32)
        self.last_trade_fees = np.zeros(self.stock_dim, dtype=np.float32)

        self.portfolio_return_memory = []
        self.turnover_memory = []

        self._update_market_data()
        init_asset = self._get_total_asset()
        self.asset_memory = [init_asset]
        self.date_memory = [self.current_date]

        return self._get_state(), {}

    def step(self, action):
        """执行一步：若为调仓日则交易，否则仅持仓推进到下一日。"""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        begin_asset = self._get_total_asset()
        trade_prices = self.prices.copy()
        is_rebalance_day = (self.day % self.rebalance_window == 0)

        action_deltas = np.zeros(self.stock_dim, dtype=np.int32)
        executed_shares = np.zeros(self.stock_dim, dtype=np.int32)
        trade_fees = np.zeros(self.stock_dim, dtype=np.float32)
        traded_notional = 0.0

        if is_rebalance_day:
            target_weights = self._scores_to_target_weights(action)
            current_shares = self.holdings.copy().astype(np.int32)
            target_shares = self._target_shares_from_weights(target_weights, begin_asset)
            deltas = target_shares - current_shares
            action_deltas = deltas.copy()

            # 先卖后买，减少现金约束下的交易失败。
            for i in range(self.stock_dim):
                if deltas[i] < 0:
                    sold = self._sell(i, abs(int(deltas[i])))
                    executed_shares[i] -= sold
                    fee = sold * trade_prices[i] * self.sell_cost_pct[i]
                    trade_fees[i] += fee
                    traded_notional += sold * trade_prices[i]

            for i in range(self.stock_dim):
                if deltas[i] > 0:
                    bought = self._buy(i, int(deltas[i]))
                    executed_shares[i] += bought
                    fee = bought * trade_prices[i] * self.buy_cost_pct[i]
                    trade_fees[i] += fee
                    traded_notional += bought * trade_prices[i]

        self.last_action_raw = action.copy()
        self.last_action_shares = action_deltas
        self.last_trade_shares = executed_shares
        self.last_trade_prices = trade_prices
        self.last_trade_fees = trade_fees

        self.day += 1
        terminated = self.day >= self.max_step
        self._update_market_data()

        end_asset = self._get_total_asset()
        portfolio_return = (end_asset - begin_asset) / max(begin_asset, 1e-8)
        turnover_ratio = traded_notional / max(begin_asset, 1e-8)

        self.portfolio_return_memory.append(portfolio_return)
        self.turnover_memory.append(turnover_ratio)

        risk_20 = 0.0
        if len(self.portfolio_return_memory) >= 20:
            risk_20 = float(np.std(self.portfolio_return_memory[-20:], ddof=0))

        reward = (
            portfolio_return
            - self.risk_penalty * risk_20
            - self.turnover_penalty * turnover_ratio
        ) * self.reward_scaling

        self.asset_memory.append(end_asset)
        self.date_memory.append(self.current_date)

        info = {
            "portfolio_return": float(portfolio_return),
            "risk_20": float(risk_20),
            "turnover": float(turnover_ratio),
            "is_rebalance_day": bool(is_rebalance_day),
        }
        return self._get_state(), float(reward), terminated, False, info

    def _build_market_features(self) -> pd.DataFrame:
        """构造市场级特征（横截面均值收益、20日波动、turbulence）。"""
        grouped = self.df.groupby("date", sort=True)

        if "log_return" in self.df.columns:
            market_ret = grouped["log_return"].mean()
        elif "pct_chg" in self.df.columns:
            market_ret = grouped["pct_chg"].mean() / 100.0
        else:
            market_ret = grouped["close"].mean().pct_change().fillna(0.0)

        market_vol_20 = market_ret.rolling(20).std().fillna(0.0)

        if "turbulence" in self.df.columns:
            market_turbulence = grouped["turbulence"].mean().fillna(0.0)
        else:
            market_turbulence = pd.Series(
                np.zeros(len(market_ret), dtype=np.float32), index=market_ret.index
            )

        return pd.DataFrame(
            {
                "market_return": market_ret.fillna(0.0).astype(np.float32),
                "market_volatility_20": market_vol_20.astype(np.float32),
                "market_turbulence": market_turbulence.astype(np.float32),
            }
        )

    def _update_market_data(self):
        """根据当前 day 加载当日价格、特征与市场特征。"""
        self.current_date = pd.Timestamp(self.dates[self.day])
        self.data = self.data_by_date[self.current_date]

        self.current_tics = self.data["tic"].tolist()
        self.prices = self.data["close"].to_numpy(dtype=np.float32)
        self.techs = (
            self.data[self.feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            .flatten()
        )
        self.market_features = (
            self.market_df.loc[
                self.current_date,
                ["market_return", "market_volatility_20", "market_turbulence"],
            ].to_numpy(dtype=np.float32)
        )

    def _scores_to_target_weights(self, scores: np.ndarray) -> np.ndarray:
        """将动作打分映射为 Top-K 等权目标权重。"""
        idx = np.argsort(scores)[::-1]
        selected = idx[: self.top_k]
        w = np.zeros(self.stock_dim, dtype=np.float32)
        w[selected] = 1.0 / float(self.top_k)
        return w

    def _target_shares_from_weights(
        self, target_weights: np.ndarray, total_asset: float
    ) -> np.ndarray:
        """由目标权重计算目标股数，并按A股整手向下取整。"""
        target_value = target_weights * float(total_asset)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_shares = np.floor(target_value / np.maximum(self.prices, 1e-8))
        raw_shares = np.nan_to_num(raw_shares, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)
        lot_shares = (raw_shares // self.lot_size) * self.lot_size
        return np.maximum(lot_shares, 0)

    def _get_holding_weights(self, total_asset: float) -> np.ndarray:
        """计算当前持仓在总资产中的权重。"""
        if total_asset <= 0:
            return np.zeros(self.stock_dim, dtype=np.float32)
        values = self.holdings * self.prices
        return (values / total_asset).astype(np.float32)

    def _get_state(self):
        """拼接并返回当前状态向量。"""
        total_asset = self._get_total_asset()
        cash_ratio = np.array([self.cash / max(total_asset, 1e-8)], dtype=np.float32)
        holding_weights = self._get_holding_weights(total_asset)
        rebalance_flag = np.array(
            [1.0 if (self.day % self.rebalance_window == 0) else 0.0], dtype=np.float32
        )
        return np.concatenate(
            [cash_ratio, holding_weights, self.techs, self.market_features, rebalance_flag]
        )

    def _get_total_asset(self):
        """计算总资产=现金+持仓市值。"""
        return float(self.cash + np.sum(self.prices * self.holdings))

    def _buy(self, index, amount):
        """买入指定股票，自动处理手续费和整手约束。"""
        price = float(self.prices[index])
        max_buy = self.cash // (price * (1.0 + self.buy_cost_pct[index]))
        buy_shares = min(int(max_buy), int(amount))
        buy_shares = (buy_shares // self.lot_size) * self.lot_size

        if buy_shares > 0:
            cost = buy_shares * price * (1.0 + self.buy_cost_pct[index])
            self.cash -= cost
            self.holdings[index] += buy_shares

        return int(buy_shares)

    def _sell(self, index, amount):
        """卖出指定股票，自动处理手续费和整手约束。"""
        sell_shares = min(int(self.holdings[index]), int(amount))
        sell_shares = (sell_shares // self.lot_size) * self.lot_size

        if sell_shares > 0:
            revenue = sell_shares * float(self.prices[index]) * (1.0 - self.sell_cost_pct[index])
            self.cash += revenue
            self.holdings[index] -= sell_shares

        return int(sell_shares)

    def get_sb_env(self):
        """返回 Stable-Baselines3 需要的 DummyVecEnv 包装。"""
        env = DummyVecEnv([lambda: self])
        return env, env.reset()

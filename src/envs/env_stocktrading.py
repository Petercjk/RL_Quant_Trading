import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    """
    自定义 A 股多股票交易环境（不依赖 FinRL）

    State:
        [cash,
         prices (N),
         holdings (N),
         technical indicators (N * K, flatten)]

    Action:
        action ∈ [-1, 1]^N
        表示对每只股票的买卖强度
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
        reward_scaling: float = 1e-4,
    ):
        super().__init__()

        self.df = df.copy()
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling

        # === 构建交易日索引 ===
        self.dates = self.df.date.unique()
        self.max_step = len(self.dates) - 1

        # === Action / Observation Space ===
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_dim,), dtype=np.float32)

        state_dim = 1 + 2 * stock_dim + stock_dim * len(tech_indicator_list)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.reset()

    # =========================
    # Core Functions
    # =========================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.day = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float32)

        self._update_market_data()
        self.asset_memory = [self._get_total_asset()]
        self.date_memory = [self.current_date]

        return self._get_state(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)

        begin_asset = self._get_total_asset()

        # === 执行交易（先卖后买）===
        actions = (action * self.hmax).astype(int)
        actions = (actions // 100) * 100  # A股一手

        # sell
        for i in range(self.stock_dim):
            if actions[i] < 0:
                self._sell(i, abs(actions[i]))

        # buy
        for i in range(self.stock_dim):
            if actions[i] > 0:
                self._buy(i, actions[i])

        # === 时间推进 ===
        self.day += 1
        terminated = self.day >= self.max_step

        self._update_market_data()
        end_asset = self._get_total_asset()

        reward = (end_asset - begin_asset) * self.reward_scaling

        self.asset_memory.append(end_asset)
        self.date_memory.append(self.current_date)

        return self._get_state(), reward, terminated, False, {}

    # =========================
    # Helper Functions
    # =========================

    def _update_market_data(self):
        self.current_date = self.dates[self.day]
        self.data = self.df[self.df.date == self.current_date]

        self.prices = self.data.close.values.astype(np.float32)
        self.techs = self.data[self.tech_indicator_list].values.flatten().astype(np.float32)

    def _get_state(self):
        state = np.concatenate(
            [[self.cash], self.prices, self.holdings, self.techs]
        )
        return state

    def _get_total_asset(self):
        return self.cash + np.sum(self.prices * self.holdings)

    def _buy(self, index, amount):
        price = self.prices[index]
        max_buy = self.cash // (price * (1 + self.buy_cost_pct[index]))
        buy_shares = min(max_buy, amount)
        buy_shares = (buy_shares // 100) * 100

        if buy_shares > 0:
            cost = buy_shares * price * (1 + self.buy_cost_pct[index])
            self.cash -= cost
            self.holdings[index] += buy_shares

    def _sell(self, index, amount):
        sell_shares = min(self.holdings[index], amount)
        sell_shares = (sell_shares // 100) * 100

        if sell_shares > 0:
            revenue = sell_shares * self.prices[index] * (1 - self.sell_cost_pct[index])
            self.cash += revenue
            self.holdings[index] -= sell_shares

    def get_sb_env(self):
        env = DummyVecEnv([lambda: self])
        return env, env.reset()

    
    
import os
import pandas as pd
from stable_baselines3 import PPO
from finrl.agents.stablebaselines3 import models as finrl_models
from src.envs.env_stocktrading import StockTradingEnv
from configs.base_config import TECHNICAL_INDICATORS
from configs.agent.ppo import PPO_PARAMS

finrl_models.pd = pd  # Ensure pandas is available in FinRL internals.


class AgentTrainer:
    def __init__(self, train_data, trade_data, paths):
        self.train_data = train_data
        self.trade_data = trade_data
        self.paths = paths
        self.stock_dim = len(train_data.tic.unique())

    def create_env(self, df):
        env = StockTradingEnv(
            df=df,
            stock_dim=self.stock_dim,
            hmax=1000,
            initial_amount=10000,
            buy_cost_pct=[0.001] * self.stock_dim,
            sell_cost_pct=[0.001] * self.stock_dim,
            tech_indicator_list=TECHNICAL_INDICATORS,
        )
        return env.get_sb_env()

    def run_training(self, total_timesteps=50000):
        env_train, _ = self.create_env(self.train_data)

        model = PPO(
            policy="MlpPolicy",
            env=env_train,
            tensorboard_log=None,
            **PPO_PARAMS,
        )

        model.learn(total_timesteps=total_timesteps)

        # 删除了原来的 model.save() 逻辑，将保存权移交给外层主程序
        return model
    
    def run_backtest(self, model):
        # Use raw env for evaluation to avoid DummyVecEnv auto-reset.
        env_trade = StockTradingEnv(
            df=self.trade_data,
            stock_dim=self.stock_dim,
            hmax=1000,
            initial_amount=10000,
            buy_cost_pct=[0.001] * self.stock_dim,
            sell_cost_pct=[0.001] * self.stock_dim,
            tech_indicator_list=TECHNICAL_INDICATORS,
        )
        obs, _ = env_trade.reset()

        action_records = []
        trade_records = []
        holding_records = []

        # Initial portfolio snapshot (day 0).
        init_date = env_trade.current_date
        init_total_asset = float(env_trade._get_total_asset())
        for i, tic in enumerate(env_trade.current_tics):
            holding = float(env_trade.holdings[i])
            price = float(env_trade.prices[i])
            holding_records.append(
                {
                    "date": init_date,
                    "tic": tic,
                    "holding_shares": holding,
                    "price": price,
                    "position_value": holding * price,
                    "cash": float(env_trade.cash),
                    "total_asset": init_total_asset,
                }
            )

        done = False
        while not done:
            trade_date = env_trade.current_date
            trade_tics = list(env_trade.current_tics)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env_trade.step(action)
            done = terminated or truncated

            for i, tic in enumerate(trade_tics):
                raw_action = float(env_trade.last_action_raw[i])
                target_shares = int(env_trade.last_action_shares[i])
                exec_shares = int(env_trade.last_trade_shares[i])
                trade_price = float(env_trade.last_trade_prices[i])
                trade_fee = float(env_trade.last_trade_fees[i])

                action_records.append(
                    {
                        "date": trade_date,
                        "tic": tic,
                        "action_raw": raw_action,
                        "target_shares": target_shares,
                    }
                )
                trade_records.append(
                    {
                        "date": trade_date,
                        "tic": tic,
                        "target_shares": target_shares,
                        "executed_shares": exec_shares,
                        "trade_price": trade_price,
                        "trade_notional": abs(exec_shares) * trade_price,
                        "trade_fee": trade_fee,
                    }
                )

            valuation_date = env_trade.current_date
            total_asset = float(env_trade._get_total_asset())
            for i, tic in enumerate(env_trade.current_tics):
                holding = float(env_trade.holdings[i])
                price = float(env_trade.prices[i])
                holding_records.append(
                    {
                        "date": valuation_date,
                        "tic": tic,
                        "holding_shares": holding,
                        "price": price,
                        "position_value": holding * price,
                        "cash": float(env_trade.cash),
                        "total_asset": total_asset,
                    }
                )

        account_value = env_trade.asset_memory
        dates = env_trade.date_memory

        table_dir = self.paths["table"]
        os.makedirs(table_dir, exist_ok=True)

        df_actions = pd.DataFrame(action_records)
        df_trades = pd.DataFrame(trade_records)
        df_holdings = pd.DataFrame(holding_records)

        df_actions.to_csv(os.path.join(table_dir, "backtest_actions.csv"), index=False)
        df_trades.to_csv(os.path.join(table_dir, "backtest_trades.csv"), index=False)
        df_holdings.to_csv(os.path.join(table_dir, "backtest_holdings.csv"), index=False)

        return (
            pd.DataFrame({"date": dates, "account_value": account_value}),
            df_actions,
        )

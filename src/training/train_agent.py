import os
import pandas as pd
from stable_baselines3 import PPO
from finrl.agents.stablebaselines3 import models as finrl_models
from finrl.agents.stablebaselines3.models import DRLAgent
from src.envs.env_stocktrading import StockTradingEnv
from configs.base_config import TECHNICAL_INDICATORS
from configs.agent.ppo import PPO_PARAMS
finrl_models.pd = pd  # 确保pandas在finrl模型中可调用


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
            initial_amount=1_000_000,
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

        print("开始训练 PPO（纯 SB3，自定义 A 股环境）...")
        model.learn(total_timesteps=total_timesteps)

        save_path = os.path.join(self.paths["model"], "ppo_agent_base")
        model.save(save_path)
        return model

    def run_backtest(self, model):
        env_trade, _ = self.create_env(self.trade_data)
        obs = env_trade.reset()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env_trade.step(action)

        env = env_trade.envs[0]
        account_value = env.asset_memory
        dates = env.date_memory

        return (
            pd.DataFrame({"date": dates, "account_value": account_value}),
            None,
        )

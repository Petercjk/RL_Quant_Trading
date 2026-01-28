# A股市场环境基础配置

ENV_KWARGS = {
    # 资金与头寸控制
    "initial_amount": 1000000,    # 初始账户资金100万
    
    # Hmax：单只股票单次最大交易股数。
    # A 股环境中，动作 a=1 通常被环境解析为买入 hmax 股。
    # 为符合“手”的概念，hmax 设为100的整数倍。
    "hmax": 1000,                 # 每次下单最大允许买入/卖出10手
    
    # A股交易成本
    # A股买入：佣金（约0.03%）；卖出：佣金（0.03%）+ 印花税（0.05%或0.1%）
    # 综合考量规费与滑点，基础流程中设定为0.001是合理的。
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    
    # 奖励函数
    "reward_scaling": 1e-4,       # 奖励缩放，协助模型收敛
    
    # 风控与状态参数/40
    "turbulence_threshold": 70,   # 湍流阈值
}

# 创新点：A 股交易单位约束配置 (Trading Unit Constraints)
# 虽然标准 StockTradingEnv 接受连续动作，但我们通过该参数提醒
# 后续在 src/envs/env_stocktrading.py 中需将 actions * hmax 后进行 100 股取整。
A_SHARE_RULES = {
    "min_share_per_trade": 100,   # 最小交易单位：1手（100股）
    "is_scr_trade": True,         # 是否启用“一手”交易限制逻辑
}


# # 创新点：风险敏感奖励函数参数 (Innovation: Risk Penalty)
# REWARD_PARAMS = {
#     "lambda_risk": 0.1,           # 风险惩罚系数（待跑通基础实验后启用）
#     "enable_max_drawdown_penalty": True
# }
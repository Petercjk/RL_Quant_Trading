import torch


# PPO 训练参数（面向当前 Top-K 组合任务）
PPO_PARAMS = {
    # 采样与批处理
    "n_steps": 2048,            # 每轮更新前采样步数
    "batch_size": 256,          # 你指定的推荐值

    # 优化器
    "learning_rate": 3e-4,      # 你指定的推荐值
    "n_epochs": 10,

    # 强化学习核心超参数
    "gamma": 0.99,              # 你指定的推荐值
    "gae_lambda": 0.95,         # 你指定的推荐值
    "clip_range": 0.2,          # 你指定的推荐值
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # 设备与日志
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose": 1,
}

# 默认训练步数（可在主流程按实验需要覆盖）
TOTAL_TIMESTEPS = 50000

import torch


PPO_PARAMS = {
    # 训练步数与批次大小
    "n_steps": 2048,           # 每次更新前收集的样本数
    "batch_size": 128,         # 每次梯度下降的批次大小
    
    # 优化
    "learning_rate": 0.00025,  # 学习率
    "n_epochs": 10,            # 每次更新时优化器跑多少遍数据
    
    # 强化学习核心参数
    "gamma": 0.99,             # 折扣因子
    "gae_lambda": 0.95,        # 优势估计参数
    "clip_range": 0.2,         # 策略裁剪范围
    "ent_coef": 0.01,          # 熵系数（鼓励探索）
    "vf_coef": 0.5,            # 价值函数损失权重
    "max_grad_norm": 0.5,      # 梯度裁剪阈值
    
    # 运行设备检测
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose": 1               # 日志输出级别
}

# 基础训练总步数（仅作为默认参考）
TOTAL_TIMESTEPS = 50000        # 参考FinRL中基础训练时长
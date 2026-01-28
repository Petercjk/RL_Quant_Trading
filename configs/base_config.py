import os
import pandas as pd

# 绝对路径配置
BASE_DIR = r"D:/AAA_Petercjk/RL_Quant_Trading"
# 数据路径
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DATA_PATH = {
    "raw": os.path.join(DATA_RAW_DIR, "China_A_share_Real_Data.csv"),
    "processed": os.path.join(DATA_PROCESSED_DIR, "processed_data.csv"),
}

# 输出路径 (全部存放于 docs 目录下，避免根目录混乱)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
OUTPUT_PATH = {
    "model": os.path.join(DOCS_DIR, "trained_models"),
    "log": os.path.join(DOCS_DIR, "tensorboard_log"),
    "result": os.path.join(DOCS_DIR, "experiments"),
    "plot": os.path.join(DOCS_DIR, "plots"),
}

# 自动创建必要的输出子文件夹
for path in OUTPUT_PATH.values():
    if not os.path.exists(path):
        os.makedirs(path)


# 特征工程因子配置 (Feature Engineering)
# 基础技术指标 (参考FinRL标准指标)
TECHNICAL_INDICATORS = [
    "macd", "rsi_30", "cci_30", "dx_30", 
    "boll_ub", "boll_lb", "close_30_sma", "close_60_sma"
]

# # 创新点：A 股特有的外部因子
# # 这些因子将在数据预处理阶段被合并进来
# EXTERNAL_FACTORS = [
#     "north_money_net",      # 北向资金净流入 (陆股通数据)
#     "shibor_1w",            # 宏观因子：1周银行间拆借利率 (反映流动性)
#     "market_sentiment",     # 情绪因子：融资融券余额或成交量变化率
#     "pe_ratio_rank"         # 估值因子：个股或行业的 PE 分位数
# ]

# 训练与交易时间窗口
TIME_WINDOW = {
    "train_start": "2015-01-01",
    "train_end": "2023-12-31",    # 训练集：2015-2023
    "trade_start": "2024-01-01",
    "trade_end": "2025-12-31",    # 测试集：2024-2025
}

# 滚动训练配置 (Innovation: Rolling Strategy)
# 关于滚动训练的说明：
# 具体的执行逻辑会写在项目根目录的 train/ 文件夹下的脚本中。
# 此处仅定义驱动滚动逻辑的参数。

# ROLLING_CONFIG = {
#     "rolling_window_months": 3,   # 创新点3：每3个月作为一个滚动窗口追加数据
#     "retrain_timesteps": 10000,   # 滚动追加训练时的迭代次数
#     "online_fine_tune": True      # 是否开启针对最新数据的在线微调
# }
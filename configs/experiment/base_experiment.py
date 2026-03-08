import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from configs.base_config import DOCS_DIR  # 引用全局基础路径

# 实验身份与元数据定义
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
EXP_NAME = f"{CURRENT_TIME}_base_experiment"

EXP_META = {
    "task_name": "A-Share Base Training & Backtest",
    "description": "基础流程：数据对齐、PPO训练、2019-2021回测及结果绘图对比",
    "indicators": "Standard FinRL Indicators (MACD, RSI, etc.)",
    "benchmark": "Equal Weight + Top5 Momentum (with transaction costs)"
}

# 基础实验任务流控制
TASK_CONTROL = {
    "do_preprocessing": False,    # 数据抓取已独立脚本化，这里默认关闭
    "do_training": True,          # 是否启动训练
    "do_backtesting": True,       # 是否执行回测
    "do_plotting": True,          # 是否生成对比图
}

# 动态路径管理
EXP_DIR = os.path.join(DOCS_DIR, "experiments", EXP_NAME)
EXP_PATHS = {
    "root": EXP_DIR,
    "model": os.path.join(EXP_DIR, "checkpoints"),
    "log": os.path.join(EXP_DIR, "logs"),
    "plot": os.path.join(EXP_DIR, "plots"),
    "table": os.path.join(EXP_DIR, "tables"),
}


def init_base_experiment(hyperparams_dict=None):
    """
    初始化基础实验环境：创建目录并生成实验记录 MD 文件
    增加了超参数字典的传入和记录
    """
    if not os.path.exists(EXP_PATHS["root"]):
        for p in EXP_PATHS.values():
            os.makedirs(p, exist_ok=True)
        print(f"SUCCESS: 已创建新实验文件夹: {EXP_DIR}")
        
        readme_path = os.path.join(EXP_DIR, "experiment_log.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Log: {EXP_NAME}\n\n")
            f.write(f"## 1. 实验描述\n")
            for k, v in EXP_META.items():
                f.write(f"- **{k}**: {v}\n")
            f.write(f"\n## 2. 任务流配置\n")
            for task, status in TASK_CONTROL.items():
                f.write(f"- {task}: {'ENABLED' if status else 'DISABLED'}\n")
            
            # 新增：将训练超参数等关键配置完整记录到 Markdown 中
            if hyperparams_dict:
                f.write(f"\n## 3. 核心配置与超参数 (Hyperparameters)\n")
                for k, v in hyperparams_dict.items():
                    f.write(f"- **{k}**: `{v}`\n")
    else:
        print(f"INFO: 检测到同名文件夹，将沿用当前目录进行调试: {EXP_DIR}")



def finalize_experiment(account_value_df, stats):
    # 将最终回测指标写入 Markdown
    log_path = os.path.join(EXP_DIR, "experiment_log.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n## 3. 回测性能指标\n")
        for metric, value in stats.items():
            f.write(f"- **{metric}**: {value}\n")
    
    # 保存净值表格
    account_value_df.to_csv(os.path.join(EXP_PATHS["table"], "account_value.csv"), index=False)

def plot_comparison(ai_results, benchmark_results_dict):
    # 修复 matplotlib 的日期解析问题，强制转换为 datetime
    os.makedirs(EXP_PATHS["plot"], exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # 强制转换日期格式
    ai_results["date"] = pd.to_datetime(ai_results["date"])
    plt.plot(ai_results["date"], ai_results["account_value"], label="AI Agent (PPO)", color="red")

    for label, df_bm in benchmark_results_dict.items():
        df_bm["date"] = pd.to_datetime(df_bm["date"])
        plt.plot(
            df_bm["date"],
            df_bm["account_value"],
            label=label,
            linestyle="--",
        )

    plt.title('Backtest Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(EXP_PATHS["plot"], "performance_comparison.png"))
    plt.close()

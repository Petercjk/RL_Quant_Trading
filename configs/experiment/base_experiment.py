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
    "description": "基础流程：数据对齐、PPO训练、2024年回测及结果绘图对比",
    "indicators": "Standard FinRL Indicators (MACD, RSI, etc.)",
    "benchmark": "Equal Weight Buy-and-Hold"
}

# 基础实验任务流控制
TASK_CONTROL = {
    "do_preprocessing": False,     # 是否重新处理数据
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


def init_base_experiment():
    """
    初始化基础实验环境：创建目录并生成实验记录 MD 文件
    """
    # 检查根实验目录是否存在，不存在则创建
    if not os.path.exists(EXP_PATHS["root"]):
        for p in EXP_PATHS.values():
            os.makedirs(p, exist_ok=True)
        print(f"SUCCESS: 已创建新实验文件夹: {EXP_DIR}")
        
        # 仅在新建文件夹时生成初始MD文件
        readme_path = os.path.join(EXP_DIR, "experiment_log.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Log: {EXP_NAME}\n\n")
            f.write(f"##  实验描述\n")
            for k, v in EXP_META.items():
                f.write(f"- **{k}**: {v}\n")
            f.write(f"\n##  任务流配置\n")
            for task, status in TASK_CONTROL.items():
                f.write(f"- {task}: {'ENABLED' if status else 'DISABLED'}\n")
    else:
        # 如果当前分钟内已经运行过，则保持静默并沿用路径
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

def plot_comparison(ai_results, benchmark_results):
    # 绘制并保存净值对比图
    plt.figure(figsize=(12, 6))
    plt.plot(ai_results['date'], ai_results['account_value'], label='AI Agent (PPO)', color='red')
    plt.plot(benchmark_results['date'], benchmark_results['account_value'], label='Benchmark', color='blue', linestyle='--')
    plt.title('Backtest Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(EXP_PATHS["plot"], "performance_comparison.png"))
    plt.close()
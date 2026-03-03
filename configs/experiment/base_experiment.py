import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from configs.base_config import DOCS_DIR  # 寮曠敤鍏ㄥ眬鍩虹璺緞

# 瀹為獙韬唤涓庡厓鏁版嵁瀹氫箟
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
EXP_NAME = f"{CURRENT_TIME}_base_experiment"

EXP_META = {
    "task_name": "A-Share Base Training & Backtest",
    "description": "鍩虹娴佺▼锛氭暟鎹榻愩€丳PO璁粌銆?024骞村洖娴嬪強缁撴灉缁樺浘瀵规瘮",
    "indicators": "Standard FinRL Indicators (MACD, RSI, etc.)",
    "benchmark": "Equal Weight Daily-Rebalanced with Buy/Sell Costs"
}

# 鍩虹瀹為獙浠诲姟娴佹帶鍒?
TASK_CONTROL = {
    "do_preprocessing": False,     # 鏄惁閲嶆柊澶勭悊鏁版嵁
    "do_training": True,          # 鏄惁鍚姩璁粌
    "do_backtesting": True,       # 鏄惁鎵ц鍥炴祴
    "do_plotting": True,          # 鏄惁鐢熸垚瀵规瘮鍥?
}

# 鍔ㄦ€佽矾寰勭鐞?
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
    鍒濆鍖栧熀纭€瀹為獙鐜锛氬垱寤虹洰褰曞苟鐢熸垚瀹為獙璁板綍 MD 鏂囦欢
    """
    # 妫€鏌ユ牴瀹為獙鐩綍鏄惁瀛樺湪锛屼笉瀛樺湪鍒欏垱寤?
    if not os.path.exists(EXP_PATHS["root"]):
        for p in EXP_PATHS.values():
            os.makedirs(p, exist_ok=True)
        print(f"SUCCESS: 宸插垱寤烘柊瀹為獙鏂囦欢澶? {EXP_DIR}")
        
        # 浠呭湪鏂板缓鏂囦欢澶规椂鐢熸垚鍒濆MD鏂囦欢
        readme_path = os.path.join(EXP_DIR, "experiment_log.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Log: {EXP_NAME}\n\n")
            f.write(f"##  瀹為獙鎻忚堪\n")
            for k, v in EXP_META.items():
                f.write(f"- **{k}**: {v}\n")
            f.write(f"\n##  浠诲姟娴侀厤缃甛n")
            for task, status in TASK_CONTROL.items():
                f.write(f"- {task}: {'ENABLED' if status else 'DISABLED'}\n")
    else:
        # 濡傛灉褰撳墠鍒嗛挓鍐呭凡缁忚繍琛岃繃锛屽垯淇濇寔闈欓粯骞舵部鐢ㄨ矾寰?
        print(f"INFO: 妫€娴嬪埌鍚屽悕鏂囦欢澶癸紝灏嗘部鐢ㄥ綋鍓嶇洰褰曡繘琛岃皟璇? {EXP_DIR}")



def finalize_experiment(account_value_df, stats):
    # 灏嗘渶缁堝洖娴嬫寚鏍囧啓鍏?Markdown
    log_path = os.path.join(EXP_DIR, "experiment_log.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n## 3. 鍥炴祴鎬ц兘鎸囨爣\n")
        for metric, value in stats.items():
            f.write(f"- **{metric}**: {value}\n")
    
    # 淇濆瓨鍑€鍊艰〃鏍?
    account_value_df.to_csv(os.path.join(EXP_PATHS["table"], "account_value.csv"), index=False)

def plot_comparison(ai_results, benchmark_results):
    # 缁樺埗骞朵繚瀛樺噣鍊煎姣斿浘
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

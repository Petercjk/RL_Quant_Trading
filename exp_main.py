import os
import sys
import pandas as pd

# 将项目根目录添加到系统路径，确保跨目录导入正常
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入配置与实验管理逻辑
from configs.base_config import DATA_PATH, TIME_WINDOW
from configs.experiment.base_experiment import (
    TASK_CONTROL, EXP_PATHS, init_base_experiment, 
    finalize_experiment, plot_comparison
)

# 导入src的核心功能模块
from src.data_processing.processor import DataProcessor
from src.training.train_agent import AgentTrainer

def run_experiment_pipeline():
    # 初始化实验环境与目录结构
    try:
        init_base_experiment()
        print("SUCCESS: 实验目录初始化成功，所有产出将存入:", EXP_PATHS["root"])
    except Exception as e:
        print(f"ERROR: 实验目录初始化失败, 详细信息: {e}")
        return

    # 数据预处理
    if TASK_CONTROL["do_preprocessing"]:
        print("INFO: 正在调用 src.data_processing 启动数据抓取与对齐...")
        try:
            # 请确保在此处或环境变量中设置了你的 Tushare Token
            processor = DataProcessor(token="d00985e44d97b66607e1bb3209880a913e9e651e861477dbbdbfacaf")
            processor.run()
            
            if os.path.exists(DATA_PATH["processed"]):
                print("SUCCESS: 数据预处理完成，数据已就绪")
            else:
                print("ERROR: 数据处理器运行结束，但未找到预期的CSV输出文件")
                return
        except Exception as e:
            print(f"ERROR: 数据预处理模块崩溃, 错误代码: {e}")
            return
    else:
        print("SKIP: 根据配置跳过数据预处理，将直接使用现有数据")

    # 数据加载与样本完整性检查
    try:
        full_df = pd.read_csv(DATA_PATH["processed"])
        train_df = full_df[(full_df.date >= TIME_WINDOW["train_start"]) & (full_df.date <= TIME_WINDOW["train_end"])]
        trade_df = full_df[(full_df.date >= TIME_WINDOW["trade_start"]) & (full_df.date <= TIME_WINDOW["trade_end"])]
        
        if train_df.empty or trade_df.empty:
            print("ERROR: 数据集切分异常，训练集或回测集数据量为0，请检查日期窗口")
            return
        print(f"SUCCESS: 数据加载成功。训练样本: {len(train_df)} 行, 回测样本: {len(trade_df)} 行")
    except Exception as e:
        print(f"ERROR: 加载处理后的数据失败, 请检查路径: {DATA_PATH['processed']}, 错误: {e}")
        return

    # 智能体训练
    trainer = AgentTrainer(train_df, trade_df, EXP_PATHS)
    model = None

    if TASK_CONTROL["do_training"]:
        print("INFO: 正在调用 src.training 启动 PPO 强化学习训练...")
        try:
            # 执行10只股票的训练任务
            model = trainer.run_training(total_timesteps=50000)
            print("SUCCESS: 模型训练流程完成，权重文件已保存至checkpoints目录")
        except Exception as e:
            print(f"ERROR: 训练环节失败, 错误原因: {e}")
            return
    else:
        print("SKIP: 根据配置跳过训练环节")

    # 回测预测与解析产出
    if TASK_CONTROL["do_backtesting"]:
        if model is None:
            print("ERROR: 无法执行回测，当前内存中没有已训练或已加载的模型")
            return
        
        print("INFO: 启动测试集回测预测 (2024-2025)...")
        try:
            # 获取AI策略的账户价值序列与交易动作详情
            df_account_value, df_actions = trainer.run_backtest(model)
            print("SUCCESS: 回测预测执行成功，已捕获净值曲线数据")

            # 计算等权重持有基准 (Benchmark) 用于对比
            price_pivot = trade_df.pivot(index='date', columns='tic', values='close')
            daily_returns = price_pivot.pct_change().fillna(0).mean(axis=1)
            df_benchmark = pd.DataFrame({
                "date": daily_returns.index,
                "account_value": (1 + daily_returns).cumprod() * 1000000
            })

            # 执行可视化
            if TASK_CONTROL["do_plotting"]:
                plot_comparison(df_account_value, df_benchmark)
                print("SUCCESS: 策略对比图已生成至 plots 目录")
            
            # 最终性能评估与报告生成
            final_nav = df_account_value['account_value'].iloc[-1]
            total_return = (final_nav / 1000000) - 1
            perf_stats = {
                "期末总资产": f"{final_nav:.2f}",
                "累计收益率": f"{total_return*100:.2f}%",
                "测试交易日天数": len(df_account_value)
            }
            finalize_experiment(df_account_value, perf_stats)
            print(f"SUCCESS: 实验报告已更新。全部结果请查看: {EXP_PATHS['root']}")

        except Exception as e:
            print(f"ERROR: 回测解析或绘图环节出现异常, 详细信息: {e}")
            return

if __name__ == "__main__":
    run_experiment_pipeline()
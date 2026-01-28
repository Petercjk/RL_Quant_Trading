import pandas as pd
import numpy as np
import os
import tushare as ts
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from configs.base_config import DATA_PATH, TECHNICAL_INDICATORS, TIME_WINDOW

class DataProcessor:
    def __init__(self, token="YOUR_TUSHARE_TOKEN"):
        """
        初始化 Tushare 接口和基础参数
        """
        ts.set_token(token)
        self.pro = ts.pro_api()
        # 目标股票列表 (10只核心蓝筹)

        # 招商银行 (600036.SH)：金融权重，数据极稳。
        # 中国平安 (601318.SH)：核心蓝筹，流动性极佳。
        # 贵州茅台 (600519.SH)：消费标杆，趋势性强，适合 RL 学习。
        # 长江电力 (600900.SH)：波动率低，防御性代表。
        # 伊利股份 (600887.SH)：消费类蓝筹。                                                                                                               
        # 中信证券 (600030.SH)：券商龙头，对市场情绪敏感。
        # 万华化学 (600309.SH)：化工龙头，周期性特征明显。
        # 格力电器 (000651.SZ)：深市核心蓝筹。
        # 比亚迪 (002594.SZ)：新能源代表，近期波动逻辑丰富。
        # 海康威视 (002415.SZ)：科技类核心标杆。

        self.ticker_list = [
            '600036.SH', '601318.SH', '600519.SH', '600900.SH', '600887.SH',
            '600030.SH', '600309.SH', '000651.SZ', '002594.SZ', '002415.SZ'
        ]
        self.raw_path = DATA_PATH["raw"]
        self.processed_path = DATA_PATH["processed"]
        self.indicators = TECHNICAL_INDICATORS

    def fetch_data(self):
        """
        从 Tushare 抓取 OHLCV 数据，并验证数据完整性
        """
        print(f" 开始抓取 10 只股票数据，范围: {TIME_WINDOW['train_start']} 至 {TIME_WINDOW['trade_end']}")
        
        df_list = []
        for ticker in self.ticker_list:
            # Tushare 接口返回：ts_code, trade_date, open, high, low, close, vol, amount
            df_temp = ts.pro_bar(
                ts_code=ticker, 
                adj='qfq', # 前复权，保证价格连续性
                start_date=TIME_WINDOW['train_start'].replace("-", ""),
                end_date=TIME_WINDOW['trade_end'].replace("-", "")
            )
            if df_temp is not None and not df_temp.empty:
                df_list.append(df_temp)
                print(f" 成功获取 {ticker}, 长度: {len(df_temp)}")
            else:
                print(f" 警告: 无法获取 {ticker} 数据")

        # 合并并重命名列以适配FinRL
        df = pd.concat(df_list, axis=0)
        df = df.rename(columns={
            'ts_code': 'tic', 
            'trade_date': 'date', 
            'vol': 'volume'
        })
        
        # 转换日期格式 YYYYMMDD -> YYYY-MM-DD
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        # 降序变升序
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # 保存原始备份
        df.to_csv(self.raw_path, index=False)
        return df

    def align_and_clean(self, df):
        """
        停牌数据对齐与技术指标填充
        """
        print(" 正在执行 A 股停牌逻辑对齐...")
        
        unique_dates = sorted(df['date'].unique())
        
        # 过滤掉数据量极少的股票 (可能中途退市)
        valid_tickers = []
        for tic in self.ticker_list:
            if len(df[df.tic == tic]) > len(unique_dates) * 0.8: # 保证至少有80%数据
                valid_tickers.append(tic)
        
        # MultiIndex 强制拉平网格
        full_idx = pd.MultiIndex.from_product([unique_dates, valid_tickers], names=['date', 'tic'])
        df_aligned = df.set_index(['date', 'tic']).reindex(full_idx).reset_index()
        
        # 价格填充 (停牌期间价格不变，成交量为0)
        df_aligned[['open', 'high', 'low', 'close']] = df_aligned.groupby('tic')[['open', 'high', 'low', 'close']].ffill()
        df_aligned['volume'] = df_aligned['volume'].fillna(0)
        # 处理初期缺失
        df_aligned = df_aligned.groupby('tic', group_keys=False).apply(lambda x: x.bfill())
        
        # 特征工程
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.indicators,
            use_vix=False,
            use_turbulence=True, # A股湍流指数
            user_defined_feature=False
        )
        
        processed = fe.preprocess_data(df_aligned)
        processed = processed.ffill().fillna(0)
        processed = processed.replace([np.inf, -np.inf], 0)
        
        return processed

    def run(self):
        """
        主运行逻辑
        """
        df_raw = self.fetch_data()
        df_final = self.align_and_clean(df_raw)
        
        df_final.to_csv(self.processed_path, index=False)
        print(f" 处理完成！已保存至 {self.processed_path}")
        print(f"最终股票池: {df_final.tic.unique()}")

if __name__ == "__main__":
    # 请在此处输入你的 Tushare Token
    processor = DataProcessor(token="d00985e44d97b66607e1bb3209880a913e9e651e861477dbbdbfacaf")
    processor.run()
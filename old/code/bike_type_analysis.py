import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import numpy as np

# 配置可视化样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 英文显示
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="colorblind")

class BikeStrategyAnalyzer:
    """针对业务问题的单车策略分析器"""
    def __init__(self, data_path, chart_dir, csv_dir):
        # 路径配置
        self.data_path = data_path
        self.chart_dir = chart_dir  # 图表保存目录
        self.csv_dir = csv_dir      # 数据保存目录
        
        # 初始化数据
        self.df = None
        self.bike_types = None  # 存储车辆类型（普通车/电车）
        self.user_types = None  # 存储用户类型（会员/非会员）
        
        # 创建目录
        os.makedirs(self.chart_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def load_and_preprocess(self, chunk_size=1_000_000):
        """加载数据并预处理（适配大规模数据）"""
        print("Loading and preprocessing data...")
        
        # 定义数据类型，减少内存占用
        dtypes = {
            'member_casual': 'category',
            'rideable_type': 'category',
            'start_station_name': 'category',
            'end_station_name': 'category',
            'duration_min': 'float32',
            'ride_id': 'category',
            'start_station_id': 'category',
            'end_station_id': 'category',
            'start_lat': 'float32',
            'start_lng': 'float32',
            'end_lat': 'float32',
            'end_lng': 'float32'
        }
        
        # 分块读取并预处理
        chunks = pd.read_csv(
            self.data_path,
            parse_dates=['started_at', 'ended_at'],
            dtype=dtypes,
            chunksize=chunk_size,
            low_memory=False
        )
        
        processed_chunks = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # 1. 提取时间特征
            chunk['hour'] = chunk['started_at'].dt.hour.astype('int8')
            chunk['is_holiday'] = chunk['started_at'].dt.weekday.isin([5,6]).astype('int8')
            chunk['is_peak'] = (chunk['hour'].between(7,9) | chunk['hour'].between(17,19)).astype('int8')
            
            # 2. 过滤无效数据
            valid_mask = (
                (chunk['duration_min'] > 0) &
                (chunk['start_station_name'].notna()) &
                (chunk['end_station_name'].notna())
            )
            chunk = chunk[valid_mask]
            
            # 3. 只保留必要列
            keep_columns = [
                'member_casual', 'rideable_type', 'duration_min',
                'start_station_name', 'end_station_name',
                'hour', 'is_holiday', 'is_peak'
            ]
            chunk = chunk[keep_columns]
            
            processed_chunks.append(chunk)
        
        # 合并分块
        self.df = pd.concat(processed_chunks, ignore_index=True)
        del processed_chunks  # 释放内存
        
        # 提取类型列表
        self.bike_types = self.df['rideable_type'].unique().tolist()
        self.user_types = self.df['member_casual'].unique().tolist()
        
        print(f"Data loaded: {len(self.df):,} records")
        print(f"Bike types: {self.bike_types}")
        print(f"User types: {self.user_types}")
        return self

    def analyze_member_frequency(self):
        """1. 会员与非会员使用普通车/电车的频率"""
        print("\nAnalyzing member vs non-member usage frequency...")
        save_chart = os.path.join(self.chart_dir, "member_frequency.png")
        save_csv = os.path.join(self.csv_dir, "member_frequency.csv")
        
        freq_data = self.df.groupby(
            ['member_casual', 'rideable_type'], 
            observed=True
        ).size().unstack(fill_value=0)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        freq_data.plot(kind='bar', width=0.7)
        plt.title("Ride Frequency by User Type and Bike Type", fontsize=14)
        plt.xlabel("User Type", fontsize=12)
        plt.ylabel("Total Rides", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Bike Type", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        freq_data.to_csv(save_csv)
        print(f"Frequency analysis saved to {save_chart} and {save_csv}")

    def analyze_avg_duration(self):
        """2. 普通车与电车的平均使用时长"""
        print("\nAnalyzing average ride duration...")
        save_chart = os.path.join(self.chart_dir, "avg_duration.png")
        save_csv = os.path.join(self.csv_dir, "avg_duration.csv")
        
        duration_data = self.df.groupby(
            ['rideable_type', 'member_casual'], 
            observed=True
        )['duration_min'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(10, 6))
        duration_data.plot(kind='bar', width=0.7)
        plt.title("Average Ride Duration (Minutes) by Bike Type", fontsize=14)
        plt.xlabel("Bike Type", fontsize=12)
        plt.ylabel("Average Duration (Minutes)", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="User Type", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        duration_data.to_csv(save_csv)
        print(f"Duration analysis saved to {save_chart} and {save_csv}")

    def analyze_peak_holiday(self):
        """3. 早晚高峰、节假日的车辆使用情况（修复类型错误）"""
        print("\nAnalyzing peak hour and holiday usage...")
        save_chart = os.path.join(self.chart_dir, "peak_holiday_usage.png")
        save_csv = os.path.join(self.csv_dir, "peak_holiday_usage.csv")
        
        # 修复：先生成字符串类型，再转换为category（避免np.select直接处理category）
        self.df['scenario'] = np.select(
            [
                (self.df['is_peak'] == 0) & (self.df['is_holiday'] == 0),
                (self.df['is_peak'] == 1) & (self.df['is_holiday'] == 0),
                (self.df['is_peak'] == 0) & (self.df['is_holiday'] == 1),
                (self.df['is_peak'] == 1) & (self.df['is_holiday'] == 1)
            ],
            [
                'Off-Peak Weekday',
                'Peak Weekday',
                'Off-Peak Holiday',
                'Peak Holiday'
            ],
            default='Unknown'  # 先输出为字符串
        )
        # 单独转换为category类型（在np.select之外处理）
        self.df['scenario'] = self.df['scenario'].astype('category')
        
        scenario_data = self.df.groupby(
            ['scenario', 'rideable_type'], 
            observed=True
        ).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 7))
        scenario_data.plot(kind='bar', width=0.7)
        plt.title("Ride Volume by Scenario (Peak/Holiday) and Bike Type", fontsize=14)
        plt.xlabel("Scenario", fontsize=12)
        plt.ylabel("Total Rides", fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.legend(title="Bike Type", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        scenario_data.to_csv(save_csv)
        print(f"Peak/holiday analysis saved to {save_chart} and {save_csv}")

    def analyze_popular_routes(self, top_n=10):
        """4. 普通车与电车的热门路线（优化性能）"""
        print(f"\nAnalyzing top {top_n} popular routes...")
        save_chart = os.path.join(self.chart_dir, f"top{top_n}_routes.png")
        save_csv = os.path.join(self.csv_dir, f"top{top_n}_routes.csv")
        
        # 1. 按起点、终点、车型分组统计
        route_counts = self.df.groupby(
            ['start_station_name', 'end_station_name', 'rideable_type'],
            observed=True
        ).size().reset_index(name='count')
        
        # 2. 按车型提取Top N路线
        top_routes = []
        for bike_type in self.bike_types:
            bike_data = route_counts[route_counts['rideable_type'] == bike_type]
            bike_top = bike_data.sort_values('count', ascending=False).head(top_n).copy()
            # 生成路线字符串
            bike_top['route'] = bike_top['start_station_name'].str.cat(
                bike_top['end_station_name'], sep=" → "
            )
            top_routes.append(bike_top[['route', 'count', 'rideable_type']])
        
        # 合并数据
        route_data = pd.concat(top_routes, ignore_index=True)
        
        # 3. 可视化
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=route_data.sort_values('count', ascending=False),
            x='count',
            y='route',
            hue='rideable_type',
            palette='viridis'
        )
        plt.title(f"Top {top_n} Popular Routes by Bike Type", fontsize=14)
        plt.xlabel("Number of Rides", fontsize=12)
        plt.ylabel("Route (Start → End)", fontsize=12)
        plt.legend(title="Bike Type", fontsize=10)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        # 保存数据
        route_data.to_csv(save_csv, index=False)
        print(f"Routes analysis saved to {save_chart} and {save_csv}")

    def run_all(self):
        """运行所有分析"""
        self.load_and_preprocess()
        self.analyze_member_frequency()
        self.analyze_avg_duration()
        self.analyze_peak_holiday()
        self.analyze_popular_routes()
        print("\nAll analyses completed!")


if __name__ == "__main__":
    # 配置路径（请根据实际情况修改）
    DATA_PATH = "../processed_data/2020_featured.csv"
    CHART_DIR = "../results/charts/2020 charts/strategy_charts"
    CSV_DIR = "../results/strategy_data"
    
    # 执行分析
    analyzer = BikeStrategyAnalyzer(DATA_PATH, CHART_DIR, CSV_DIR)
    analyzer.run_all()
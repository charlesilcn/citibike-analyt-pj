import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm
from geopy.distance import geodesic
from scipy import stats  # 新增：用于统计异常值检测

# 配置可视化样式（仅图表用英文显示，避免乱码）
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # 英文图表字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", palette="Set2")

class BikeUser时空Analyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.chart_dir = os.path.join(output_dir, "图表")  # 中文目录名
        self.data_dir = os.path.join(output_dir, "数据")      # 中文目录名
        
        # 初始化数据容器
        self.df = None
        self.bike_types = None  # 单车类型列表
        self.user_types = None  # 用户类型列表
        
        # 创建输出目录
        os.makedirs(self.chart_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def load_and_preprocess(self, chunk_size=1_000_000):
        """加载数据并预处理（新增异常值处理和季度特征）"""
        print("加载并预处理数据中...")
        
        # 定义数据类型以减少内存占用
        dtypes = {
            'rideable_type': 'category',
            'member_casual': 'category',
            'start_station_name': 'category',
            'end_station_name': 'category',
            'start_station_id': 'category',
            'end_station_id': 'category',
            'ride_id': 'category',
            'start_lat': 'float32',
            'start_lng': 'float32',
            'end_lat': 'float32',
            'end_lng': 'float32'
        }
        
        # 分块读取数据
        chunks = pd.read_csv(
            self.data_path,
            parse_dates=['started_at', 'ended_at'],  # 解析时间列
            dtype=dtypes,
            chunksize=chunk_size,
            low_memory=False
        )
        
        processed_chunks = []
        for chunk in tqdm(chunks, desc="处理数据块"):
            # 1. 计算骑行时长（分钟）和距离（公里）
            chunk['duration_min'] = (chunk['ended_at'] - chunk['started_at']).dt.total_seconds() / 60
            chunk['distance_km'] = chunk.apply(
                lambda row: geodesic(
                    (row['start_lat'], row['start_lng']),
                    (row['end_lat'], row['end_lng'])
                ).km if pd.notna(row['start_lat']) and pd.notna(row['end_lat']) else None,
                axis=1
            )
            
            # 2. 提取时间特征（新增季度特征）
            chunk['hour'] = chunk['started_at'].dt.hour.astype('int8')  # 小时（0-23）
            chunk['day_of_week'] = chunk['started_at'].dt.weekday.astype('int8')  # 周几（0=周一）
            chunk['is_weekend'] = chunk['day_of_week'].isin([5, 6]).astype('int8')  # 是否周末（5=周六，6=周日）
            chunk['month'] = chunk['started_at'].dt.month.astype('int8')  # 月份
            chunk['quarter'] = chunk['started_at'].dt.quarter.astype('int8')  # 季度（1-4）
            
            # 3. 过滤无效数据（新增统计法异常值处理）
            valid_mask = (
                (chunk['duration_min'] > 0) &  # 排除负时长
                (chunk['start_station_name'].notna()) &
                (chunk['end_station_name'].notna()) &
                (chunk['distance_km'].notna()) &  # 排除无效距离
                (chunk['distance_km'] > 0.1)  # 排除过短距离（<100米）
            )
            chunk = chunk[valid_mask].copy()  # 避免SettingWithCopyWarning
            
            # 用Z-score过滤极端异常值（时长和距离）
            if not chunk.empty:
                z_duration = np.abs(stats.zscore(chunk['duration_min']))
                z_distance = np.abs(stats.zscore(chunk['distance_km']))
                chunk = chunk[(z_duration < 3) & (z_distance < 3)]  # 保留99.7%置信区间内的数据
            
            # 4. 只保留必要列
            keep_cols = [
                'rideable_type', 'member_casual', 'duration_min', 'distance_km',
                'start_station_name', 'end_station_name', 'start_lat', 'start_lng',
                'hour', 'is_weekend', 'month', 'quarter', 'day_of_week'
            ]
            chunk = chunk[keep_cols]
            
            processed_chunks.append(chunk)
        
        # 合并所有数据块
        self.df = pd.concat(processed_chunks, ignore_index=True)
        del processed_chunks  # 释放内存
        
        # 提取类型列表
        self.bike_types = self.df['rideable_type'].unique().tolist()
        self.user_types = self.df['member_casual'].unique().tolist()
        
        print(f"数据加载完成：{len(self.df):,} 条记录")
        print(f"单车类型：{self.bike_types}")
        print(f"用户类型：{self.user_types}")
        return self

    # ------------------------------
    # 一、用户行为分析
    # ------------------------------
    def analyze_user_preference(self):
        """1. 用户类型对单车类型的偏好（频率+占比）"""
        print("\n分析用户类型对单车的偏好...")
        save_chart = os.path.join(self.chart_dir, "用户-单车偏好.png")
        
        # 统计频率（用户类型×单车类型的骑行次数）
        freq_data = self.df.groupby(
            ['member_casual', 'rideable_type'], observed=True
        ).size().unstack(fill_value=0)
        
        # 计算占比（各用户类型中，每种单车的使用占比）
        ratio_data = freq_data.div(freq_data.sum(axis=1), axis=0) * 100
        
        # 可视化（双轴图：左侧频率，右侧占比）
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # 左侧：频率柱状图
        freq_data.plot(kind='bar', ax=ax1, width=0.7, alpha=0.7)
        ax1.set_ylabel("Total Rides", fontsize=12)  # 英文标签
        ax1.set_xlabel("User Type", fontsize=12)
        ax1.tick_params(axis='x', rotation=0)
        
        # 右侧：占比折线图
        ax2 = ax1.twinx()
        for col in ratio_data.columns:
            ax2.plot(
                ratio_data.index, ratio_data[col], 
                marker='o', linewidth=2, label=col
            )
        ax2.set_ylabel("Proportion (%)", fontsize=12)  # 英文标签
        ax2.set_ylim(0, 100)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, title="Bike Type", loc='upper left')  # 英文图例
        
        plt.title("Ride Frequency and Proportion by User Type", fontsize=14)  # 英文标题
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        # 保存数据
        freq_data.to_csv(os.path.join(self.data_dir, "用户-单车偏好_频率.csv"))
        ratio_data.to_csv(os.path.join(self.data_dir, "用户-单车偏好_占比.csv"))
        print(f"用户偏好分析结果保存至 {save_chart}")

    def analyze_riding_behavior(self):
        """2. 不同单车类型的骑行行为（时长+距离+分布特征）"""
        print("\n分析不同单车的骑行行为...")
        save_chart = os.path.join(self.chart_dir, "单车-骑行行为.png")
        save_boxplot = os.path.join(self.chart_dir, "单车-时长距离分布.png")  # 新增箱线图
        
        # 按单车类型和用户类型统计：平均时长、平均距离
        behavior_data = self.df.groupby(
            ['rideable_type', 'member_casual'], observed=True
        ).agg({
            'duration_min': ['mean', 'median'],  # 新增中位数（抗异常值）
            'distance_km': ['mean', 'median']
        }).round(2)
        behavior_data.columns = ['_'.join(col) for col in behavior_data.columns]  # 合并列名
        
        # 可视化平均指标（双轴柱状图）
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # 左侧：平均时长
        behavior_data['duration_min_mean'].unstack().plot(
            kind='bar', ax=ax1, width=0.4, position=0, alpha=0.7, label='Avg Duration (min)'
        )
        ax1.set_ylabel("Average Duration (minutes)", fontsize=12)  # 英文标签
        ax1.set_xlabel("Bike Type", fontsize=12)
        ax1.tick_params(axis='x', rotation=0)
        
        # 右侧：平均距离
        ax2 = ax1.twinx()
        behavior_data['distance_km_mean'].unstack().plot(
            kind='bar', ax=ax2, width=0.4, position=1, alpha=0.7, color='orange', label='Avg Distance (km)'
        )
        ax2.set_ylabel("Average Distance (kilometers)", fontsize=12)  # 英文标签
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, title="User Type", loc='upper left')  # 英文图例
        
        plt.title("Average Ride Duration and Distance by Bike Type", fontsize=14)  # 英文标题
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        # 新增：时长和距离的分布箱线图（展示数据分布特征）
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df, x='rideable_type', y='duration_min', hue='member_casual')
        plt.title("Duration Distribution by Bike Type", fontsize=12)
        plt.xlabel("Bike Type")
        plt.ylabel("Duration (minutes)")
        plt.ylim(0, 60)  # 限制范围以便观察
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x='rideable_type', y='distance_km', hue='member_casual')
        plt.title("Distance Distribution by Bike Type", fontsize=12)
        plt.xlabel("Bike Type")
        plt.ylabel("Distance (km)")
        plt.ylim(0, 10)  # 限制范围以便观察
        
        plt.tight_layout()
        plt.savefig(save_boxplot, dpi=300)
        plt.close()
        
        # 保存数据
        behavior_data.to_csv(os.path.join(self.data_dir, "单车-骑行行为指标.csv"))
        print(f"骑行行为分析结果保存至 {save_chart} 和 {save_boxplot}")

    # ------------------------------
    # 二、时空分布分析（新增季度分析）
    # ------------------------------
    def analyze_time_distribution(self):
        """3. 时间分布：日内+周末/工作日+季度趋势"""
        print("\n分析不同单车的时间分布...")
        # 子图1：日内小时分布
        hourly_chart = os.path.join(self.chart_dir, "单车-日内分布.png")
        # 子图2：周末vs工作日分布
        weekday_chart = os.path.join(self.chart_dir, "单车-周内分布.png")
        # 新增：季度趋势图
        quarter_chart = os.path.join(self.chart_dir, "单车-季度趋势.png")
        
        # 1. 日内小时分布（按单车类型）
        hourly_data = self.df.groupby(
            ['hour', 'rideable_type'], observed=True
        ).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        hourly_data.plot(kind='line', marker='o', linewidth=2)
        plt.title("Hourly Ride Distribution by Bike Type", fontsize=14)  # 英文标题
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Number of Rides", fontsize=12)
        plt.xticks(range(0, 24, 2))  # 每2小时显示一次
        plt.legend(title="Bike Type")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(hourly_chart, dpi=300)
        plt.close()
        
        # 2. 周末vs工作日分布（按单车类型）
        weekday_data = self.df.groupby(
            ['is_weekend', 'rideable_type'], observed=True
        ).size().unstack(fill_value=0)
        weekday_data.index = ['Weekday', 'Weekend']  # 英文索引
        
        plt.figure(figsize=(10, 6))
        weekday_data.plot(kind='bar', width=0.6)
        plt.title("Rides: Weekdays vs Weekends", fontsize=14)  # 英文标题
        plt.xlabel("Day Type", fontsize=12)
        plt.ylabel("Total Rides", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Bike Type")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(weekday_chart, dpi=300)
        plt.close()
        
        # 3. 新增：季度趋势分析（按单车类型）
        quarter_data = self.df.groupby(
            ['quarter', 'rideable_type'], observed=True
        ).size().unstack(fill_value=0)
        quarter_data.index = [f'Q{q}' for q in quarter_data.index]  # 格式化季度标签（Q1-Q4）
        
        plt.figure(figsize=(10, 6))
        quarter_data.plot(kind='line', marker='s', linewidth=2)
        plt.title("Quarterly Ride Trend by Bike Type", fontsize=14)  # 英文标题
        plt.xlabel("Quarter", fontsize=12)
        plt.ylabel("Total Rides", fontsize=12)
        plt.legend(title="Bike Type")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(quarter_chart, dpi=300)
        plt.close()
        
        # 保存数据
        hourly_data.to_csv(os.path.join(self.data_dir, "单车-日内分布.csv"))
        weekday_data.to_csv(os.path.join(self.data_dir, "单车-周内分布.csv"))
        quarter_data.to_csv(os.path.join(self.data_dir, "单车-季度趋势.csv"))
        print(f"时间分布分析结果保存至 {hourly_chart}、{weekday_chart} 和 {quarter_chart}")

    def analyze_spatial_distribution(self, top_n=15):
        """4. 空间分布：热门起点站+区域聚集（新增区域聚合逻辑）"""
        print(f"\n分析不同单车的空间分布（Top {top_n} 站点）...")
        save_chart = os.path.join(self.chart_dir, "单车-空间分布.png")
        save_heatmap = os.path.join(self.chart_dir, "单车-热力图分布.png")  # 新增热力图
        
        # 1. 统计各单车类型的Top N热门起点站
        top_stations = {}
        for bike in self.bike_types:
            # 筛选该单车类型的起点站使用次数
            station_counts = self.df[self.df['rideable_type'] == bike]['start_station_name'].value_counts()
            top_stations[bike] = station_counts.head(top_n).index  # 保留Top N站点名称
        
        # 2. 合并所有Top站点，统计各单车类型的使用次数
        all_top_stations = set()
        for stations in top_stations.values():
            all_top_stations.update(stations)
        all_top_stations = list(all_top_stations)
        
        # 按站点和单车类型统计次数
        spatial_data = self.df[
            self.df['start_station_name'].isin(all_top_stations)
        ].groupby(
            ['start_station_name', 'rideable_type'], observed=True
        ).size().unstack(fill_value=0)
        
        # 3. 可视化热门站点分布（按总次数排序）
        spatial_data['total'] = spatial_data.sum(axis=1)  # 计算每个站点的总次数
        spatial_data = spatial_data.sort_values('total', ascending=False).drop(columns='total')  # 排序并删除总次数列
        
        plt.figure(figsize=(14, 8))
        spatial_data.plot(kind='bar', stacked=True, width=0.8)
        plt.title(f"Bike Type Distribution at Top {top_n} Stations", fontsize=14)  # 英文标题
        plt.xlabel("Start Station", fontsize=12)
        plt.ylabel("Number of Rides", fontsize=12)
        plt.xticks(rotation=45, ha='right')  # 旋转站点名称避免重叠
        plt.legend(title="Bike Type")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=300)
        plt.close()
        
        # 新增：骑行起点热力图（按单车类型聚合）
        plt.figure(figsize=(16, 10))
        for i, bike_type in enumerate(self.bike_types, 1):
            plt.subplot(1, len(self.bike_types), i)
            bike_data = self.df[self.df['rideable_type'] == bike_type]
            # 绘制核密度图（热力图效果）
            sns.kdeplot(
                data=bike_data, x='start_lng', y='start_lat', 
                fill=True, cmap="viridis", bw_adjust=0.5, alpha=0.8
            )
            plt.title(f"Start Location Density: {bike_type}", fontsize=12)  # 英文标题
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.xticks([])  # 隐藏具体坐标（如需显示可删除）
            plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(save_heatmap, dpi=300)
        plt.close()
        
        # 保存数据
        spatial_data.to_csv(os.path.join(self.data_dir, "单车-空间分布.csv"))
        print(f"空间分布分析结果保存至 {save_chart} 和 {save_heatmap}")

    # ------------------------------
    # 运行所有分析
    # ------------------------------
    def run_all(self):
        self.load_and_preprocess()
        # 用户行为分析
        self.analyze_user_preference()   # 用户偏好（频率+占比）
        self.analyze_riding_behavior()   # 骑行时长+距离（新增分布分析）
        # 时空分布分析
        self.analyze_time_distribution() # 日内+周末/工作日+季度趋势（新增季度）
        self.analyze_spatial_distribution()  # 热门站点+热力图（新增热力图）
        print("\n所有分析完成！结果保存至：", self.output_dir)


if __name__ == "__main__":
    # 配置路径（请修改为实际路径）
    DATA_PATH = "../processed_data/2020_featured.csv"  # 输入数据路径
    OUTPUT_DIR = "../results/单车类型分析"  # 输出目录
    
    # 执行分析
    analyzer = BikeUser时空Analyzer(DATA_PATH, OUTPUT_DIR)
    analyzer.run_all()
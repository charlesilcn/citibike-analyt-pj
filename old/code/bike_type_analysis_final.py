import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm
from haversine import haversine_vector, Unit

# 全局配置（确保图表正常显示）
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['axes.labelpad'] = 10
sns.set_theme(style="whitegrid", palette="Set3")


class BikeAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.chart_dir = os.path.join(output_dir, "图表")
        self.data_dir = os.path.join(output_dir, "数据")
        os.makedirs(self.chart_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.df = None
        self.bike_types = None
        self.user_types = None

    def load_and_preprocess(self, chunk_size=2_000_000):
        """加载并预处理数据（移除对end_station_name的依赖）"""
        print("加载并预处理数据中...")
        
        # 定义数据类型（仅包含数据中实际存在的字段）
        dtypes = {
            'rideable_type': 'category',
            'member_casual': 'category',
            'start_station_name': 'category',
            'start_lat': 'float32',
            'start_lng': 'float32',
            'end_lat': 'float32',
            'end_lng': 'float32'
        }
        # 只读取数据中存在的字段（避免KeyError）
        usecols = list(dtypes.keys()) + ['started_at', 'ended_at']
        
        try:
            chunks = pd.read_csv(
                self.data_path,
                parse_dates=['started_at', 'ended_at'],
                dtype=dtypes,
                usecols=usecols,
                chunksize=chunk_size,
                low_memory=False
            )
        except Exception as e:
            print(f"数据读取错误：{str(e)}")
            raise  # 抛出错误便于调试
        
        processed_chunks = []
        for chunk in tqdm(chunks, desc="处理数据块"):
            # 计算骑行时长（分钟）
            chunk['duration_min'] = (chunk['ended_at'] - chunk['started_at']).dt.total_seconds() / 60
            
            # 批量计算骑行距离（公里）
            valid_mask = chunk[['start_lat', 'end_lat', 'start_lng', 'end_lng']].notna().all(axis=1)
            if valid_mask.sum() > 0:
                start_coords = chunk.loc[valid_mask, ['start_lat', 'start_lng']].values
                end_coords = chunk.loc[valid_mask, ['end_lat', 'end_lng']].values
                chunk.loc[valid_mask, 'distance_km'] = haversine_vector(
                    start_coords, end_coords, unit=Unit.KILOMETERS
                ).astype('float32')
            else:
                chunk['distance_km'] = np.nan
            
            # 提取时间特征
            chunk['hour'] = chunk['started_at'].dt.hour.astype('int8')
            chunk['is_weekend'] = chunk['started_at'].dt.weekday.isin([5,6]).astype('int8')
            chunk['quarter'] = chunk['started_at'].dt.quarter.astype('int8')
            
            # 过滤无效数据（关键修复：移除对end_station_name的检查）
            valid_filter = (
                (chunk['duration_min'] > 0) & (chunk['duration_min'] <= 60) &  # 有效时长
                (chunk['distance_km'].fillna(0) > 0.1) & (chunk['distance_km'] <= 10) &  # 有效距离
                (chunk['start_station_name'].notna())  # 仅检查起点站非空
            )
            
            # 只保留必要字段
            keep_cols = [
                'rideable_type', 'member_casual', 'duration_min', 'distance_km',
                'start_station_name', 'start_lat', 'start_lng',
                'hour', 'is_weekend', 'quarter'
            ]
            processed_chunks.append(chunk[valid_filter][keep_cols])
        
        # 合并数据并释放内存
        self.df = pd.concat(processed_chunks, ignore_index=True)
        del processed_chunks
        
        # 提取类型列表（兼容空数据场景）
        if not self.df.empty:
            self.bike_types = self.df['rideable_type'].unique().tolist()
            self.user_types = self.df['member_casual'].unique().tolist()
            print(f"数据加载完成：{len(self.df):,} 条有效记录")
            print(f"单车类型：{self.bike_types} | 用户类型：{self.user_types}")
        else:
            print("警告：未加载到有效数据，请检查数据源")
        
        return self

    def analyze_user_preference(self):
        """分析用户对单车类型的偏好"""
        if self.df.empty:
            print("数据为空，跳过用户偏好分析")
            return
        
        print("\n分析用户类型对单车的偏好...")
        save_bar = os.path.join(self.chart_dir, "用户-单车偏好_柱状图.png")
        save_pie = os.path.join(self.chart_dir, "用户-单车偏好_饼图.png")
        
        # 计算频率和占比
        freq_data = pd.crosstab(self.df['member_casual'], self.df['rideable_type'])
        ratio_data = freq_data.div(freq_data.sum(axis=1), axis=0) * 100
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = freq_data.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', alpha=0.8)
        for container in bars.containers:
            ax.bar_label(container, fmt='%d', fontsize=9, padding=5)
        ax.set_title("Ride Distribution by User Type and Bike Type", fontsize=14, pad=20)
        ax.set_xlabel("User Type", fontsize=12)
        ax.set_ylabel("Total Rides", fontsize=12)
        ax.legend(title="Bike Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_bar)
        plt.close()
        
        # 绘制饼图
        fig, axes = plt.subplots(1, len(self.user_types), figsize=(12, 5))
        for i, user_type in enumerate(self.user_types):
            ax = axes[i] if len(self.user_types) > 1 else axes
            ratio_data.loc[user_type].plot(
                kind='pie', ax=ax, autopct='%.1f%%', startangle=90,
                wedgeprops=dict(width=0.3, edgecolor='white')
            )
            ax.set_title(f"User Type: {user_type}", fontsize=12)
            ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(save_pie)
        plt.close()
        
        # 保存数据
        freq_data.to_parquet(os.path.join(self.data_dir, "用户-单车偏好_频率.parquet"))
        ratio_data.to_parquet(os.path.join(self.data_dir, "用户-单车偏好_占比.parquet"))
        print(f"用户偏好分析完成：{save_bar}, {save_pie}")

    def analyze_riding_behavior(self):
        """分析不同单车的骑行行为特征"""
        if self.df.empty:
            print("数据为空，跳过骑行行为分析")
            return
        
        print("\n分析不同单车的骑行行为...")
        save_bar = os.path.join(self.chart_dir, "单车-骑行行为_柱状图.png")
        save_box = os.path.join(self.chart_dir, "单车-骑行行为_箱线图.png")
        
        # 计算行为指标
        behavior_data = self.df.groupby(
            ['rideable_type', 'member_casual'], observed=True
        ).agg({
            'duration_min': ['mean', 'median'],
            'distance_km': ['mean', 'median']
        }).round(2)
        behavior_data.columns = ['_'.join(col) for col in behavior_data.columns]
        
        # 绘制柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        behavior_data['duration_min_mean'].unstack().plot(
            kind='bar', ax=ax1, width=0.7, edgecolor='black', alpha=0.8
        )
        ax1.set_title("Average Duration (minutes)", fontsize=12)
        ax1.set_xlabel("Bike Type")
        ax1.legend(title="User Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, behavior_data['duration_min_mean'].max() * 1.2)
        
        behavior_data['distance_km_mean'].unstack().plot(
            kind='bar', ax=ax2, width=0.7, edgecolor='black', alpha=0.8
        )
        ax2.set_title("Average Distance (km)", fontsize=12)
        ax2.set_xlabel("Bike Type")
        ax2.legend(title="User Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, behavior_data['distance_km_mean'].max() * 1.2)
        
        plt.suptitle("Riding Behavior by Bike Type", fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(save_bar)
        plt.close()
        
        # 绘制箱线图
        sample_data = self.df.sample(frac=0.1, random_state=42)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(
            data=sample_data, x='rideable_type', y='duration_min', 
            hue='member_casual', ax=ax1, palette="Set2",
            boxprops=dict(edgecolor='black'),
            flierprops=dict(marker='o', markersize=3)
        )
        ax1.set_title("Duration Distribution", fontsize=12)
        ax1.set_ylim(0, 40)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        sns.boxplot(
            data=sample_data, x='rideable_type', y='distance_km', 
            hue='member_casual', ax=ax2, palette="Set2",
            boxprops=dict(edgecolor='black')
        )
        ax2.set_title("Distance Distribution", fontsize=12)
        ax2.set_ylim(0, 8)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle("Behavior Distribution by Bike Type", fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(save_box)
        plt.close()
        
        behavior_data.to_parquet(os.path.join(self.data_dir, "单车-骑行行为指标.parquet"))
        print(f"骑行行为分析完成：{save_bar}, {save_box}")

    def analyze_time_distribution(self):
        """分析单车使用的时间分布特征"""
        if self.df.empty:
            print("数据为空，跳过时间分布分析")
            return
        
        print("\n分析不同单车的时间分布...")
        save_hourly = os.path.join(self.chart_dir, "单车-日内分布.png")
        save_weekday = os.path.join(self.chart_dir, "单车-周内分布.png")
        
        # 计算时间分布数据
        time_data = {
            'hourly': self.df.groupby(['hour', 'rideable_type'], observed=True).size().unstack(fill_value=0),
            'weekday': self.df.groupby(['is_weekend', 'rideable_type'], observed=True).size().unstack(fill_value=0)
        }
        
        # 绘制日内分布
        plt.figure(figsize=(10, 6))
        for bike_type in time_data['hourly'].columns:
            plt.plot(
                time_data['hourly'].index,
                time_data['hourly'][bike_type],
                marker='o', linewidth=2, label=bike_type, alpha=0.8
            )
        plt.title("Hourly Ride Distribution by Bike Type", fontsize=14)
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Number of Rides", fontsize=12)
        plt.xticks(range(0, 24, 3))
        plt.legend(title="Bike Type", loc='upper right')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_hourly)
        plt.close()
        
        # 绘制周内分布
        plt.figure(figsize=(8, 6))
        weekday_data = time_data['weekday'].rename(index={0: 'Weekday', 1: 'Weekend'})
        weekday_data.plot(kind='bar', width=0.7, edgecolor='black', alpha=0.8)
        plt.title("Ride Distribution: Weekday vs Weekend", fontsize=14)
        plt.xlabel("Day Type", fontsize=12)
        plt.ylabel("Total Rides", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Bike Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_weekday)
        plt.close()
        
        time_data['hourly'].to_parquet(os.path.join(self.data_dir, "单车-日内分布.parquet"))
        weekday_data.to_parquet(os.path.join(self.data_dir, "单车-周内分布.parquet"))
        print(f"时间分布分析完成：{save_hourly}, {save_weekday}")

    def analyze_spatial_distribution(self, top_n=10):
        """分析单车使用的空间分布特征"""
        if self.df.empty:
            print("数据为空，跳过空间分布分析")
            return
        
        print(f"\n分析不同单车的空间分布（Top {top_n} 站点）...")
        save_bar = os.path.join(self.chart_dir, "单车-空间分布_柱状图.png")
        save_heat = os.path.join(self.chart_dir, "单车-空间分布_热力图.png")
        
        # 统计热门站点
        top_stations = {}
        for bike_type in self.bike_types:
            station_counts = self.df[
                self.df['rideable_type'] == bike_type
            ]['start_station_name'].value_counts()
            top_stations[bike_type] = station_counts.head(top_n).index
        
        all_top_stations = set().union(*top_stations.values())
        
        # 计算空间分布数据（修复变量引用）
        spatial_data = pd.crosstab(
            self.df[self.df['start_station_name'].isin(all_top_stations)]['start_station_name'],
            self.df['rideable_type']
        )
        spatial_data['total_rides'] = spatial_data.sum(axis=1)
        spatial_data = spatial_data.sort_values('total_rides', ascending=False).drop(columns='total_rides')
        
        # 绘制热门站点柱状图
        plt.figure(figsize=(12, 7))
        spatial_data.plot(kind='bar', stacked=True, width=0.8, edgecolor='black')
        plt.title(f"Top {top_n} Start Stations by Bike Type", fontsize=14)
        plt.xlabel("Start Station", fontsize=12)
        plt.ylabel("Number of Rides", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Bike Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_bar)
        plt.close()
        
        # 绘制热力图
        plt.figure(figsize=(10, 5))
        for i, bike_type in enumerate(self.bike_types, 1):
            plt.subplot(1, len(self.bike_types), i)
            sample_data = self.df[self.df['rideable_type'] == bike_type].sample(
                frac=0.01, random_state=42, replace=True
            )
            if not sample_data.empty:
                sns.kdeplot(
                    data=sample_data, x='start_lng', y='start_lat',
                    fill=True, cmap="viridis", bw_adjust=1.0
                )
            plt.title(f"{bike_type}", fontsize=12)
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(save_heat)
        plt.close()
        
        spatial_data.to_parquet(os.path.join(self.data_dir, "单车-空间分布.parquet"))
        print(f"空间分布分析完成：{save_bar}, {save_heat}")

    def run_all(self):
        """运行所有分析步骤"""
        self.load_and_preprocess()
        self.analyze_user_preference()
        self.analyze_riding_behavior()
        self.analyze_time_distribution()
        self.analyze_spatial_distribution()
        print(f"\n所有分析完成！结果保存至：{self.output_dir}")


if __name__ == "__main__":
    # 请修改为实际数据路径
    DATA_PATH = "../processed_data/2020_featured.csv"  # 输入数据路径
    OUTPUT_DIR = "../results/bike_tpye_op"            # 输出结果目录
    
    analyzer = BikeAnalyzer(DATA_PATH, OUTPUT_DIR)
    analyzer.run_all()
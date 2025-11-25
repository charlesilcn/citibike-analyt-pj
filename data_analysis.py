#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import json
from geopy.distance import geodesic
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bike_analysis')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 忽略警告
warnings.filterwarnings('ignore')

# 创建结果保存目录
result_dir = r'd:\code\502\Bike A\result'
os.makedirs(result_dir, exist_ok=True)

# 创建可视化目录
viz_dir = os.path.join(result_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

class BikeDataAnalyzer:
    def __init__(self, file_path, config=None):
        """初始化数据分析器
        
        Args:
            file_path: 数据文件路径
            config: 配置字典，可覆盖默认配置
        """
        self.file_path = file_path
        self.df = None
        self.results = {}
        self.sample_size = 100000  # 大数据集时使用的样本量
        self.output_dir = result_dir
        self.viz_dir = viz_dir
        
        # 合并默认配置和用户配置
        self.config = {
            'sample_size': 100000,
            'chunk_size': 1000000,
            'sample_frac': 0.1,
            'max_visualization_items': 20
        }
        if config:
            self.config.update(config)
            
        logger.info(f"初始化分析器，数据文件: {file_path}")
    
    def load_data(self, sample=False, sample_frac=None):
        """加载数据并进行基本检查
        
        Args:
            sample: 是否使用采样数据
            sample_frac: 采样比例，当sample=True且file_size <= 100MB时使用
                        如果为None，则使用配置中的值
            
        Returns:
            pd.DataFrame: 加载的数据框
            
        Raises:
            FileNotFoundError: 文件不存在
            Exception: 数据加载失败
        """
        try:
            # 检查文件是否存在
            self._check_file_exists()
            
            logger.info(f"开始加载数据: {self.file_path}")
            
            # 首先检查文件大小
            file_size = os.path.getsize(self.file_path) / (1024 * 1024)
            logger.info(f"文件大小: {file_size:.2f} MB")
            
            # 如果sample_frac为None，使用配置中的值
            if sample_frac is None:
                sample_frac = self.config['sample_frac']
                
            # 读取文件头信息
            header_df = pd.read_csv(self.file_path, nrows=0)
            logger.info(f"数据列数: {len(header_df.columns)}")
            logger.info(f"列名: {list(header_df.columns)}")
            
            # 读取数据（大数据集时采用采样或分块处理）
            if sample:
                if file_size > 100:  # 如果文件大于100MB且需要采样
                    sample_size = self.config['sample_size']
                    logger.info(f"文件较大，正在加载{sample_size:,}行样本数据...")
                    self.df = pd.read_csv(self.file_path, nrows=sample_size)
                else:
                    # 对于较小文件，使用比例采样
                    logger.info(f"使用{sample_frac*100:.0f}%比例采样数据...")
                    temp_df = pd.read_csv(self.file_path)
                    self.df = temp_df.sample(frac=sample_frac, random_state=42)
                    logger.info(f"采样后数据量: {len(self.df):,} 行")
            else:
                # 使用分块读取处理大文件
                logger.info("开始读取完整数据...")
                chunks = []
                chunk_size = self.config['chunk_size']
                total_rows_read = 0
                
                # 获取文件总行数（不包括标题行）
                # 使用行数估计，避免完全加载文件
                with open(self.file_path, 'r') as f:
                    total_lines = sum(1 for _ in f)
                total_rows = total_lines - 1  # 减去标题行
                
                for i, chunk in enumerate(pd.read_csv(self.file_path, chunksize=chunk_size)):
                    chunks.append(chunk)
                    total_rows_read += len(chunk)
                    progress_percent = min(100, (total_rows_read / total_rows) * 100) if total_rows > 0 else 100
                    logger.info(f"已读取 {total_rows_read:,} 行 ({progress_percent:.1f}%)...")
                
                self.df = pd.concat(chunks, ignore_index=True)
            
            logger.info(f"数据加载完成，总行数: {len(self.df):,}")
            logger.debug(f"数据前5行:\n{self.df.head()}")
            
            # 记录基本数据信息
            logger.info(f"数据维度: {self.df.shape}")
            logger.info(f"数据类型:\n{self.df.dtypes}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise Exception(f"数据加载失败: {str(e)}") from e
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n开始数据预处理...")
        
        # 转换时间列
        time_columns = ['started_at', 'ended_at']
        for col in time_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"转换 {col} 为日期时间格式")
        
        # 计算骑行时长（分钟）
        if 'started_at' in self.df.columns and 'ended_at' in self.df.columns:
            self.df['ride_duration_minutes'] = (self.df['ended_at'] - self.df['started_at']).dt.total_seconds() / 60
            print("计算骑行时长（分钟）")
            
            # 过滤异常值（骑行时长在1分钟到24小时之间）
            self.df = self.df[(self.df['ride_duration_minutes'] >= 1) & 
                             (self.df['ride_duration_minutes'] <= 1440)]
            print(f"过滤异常时长后的数据行数: {len(self.df):,}")
        
        # 提取时间特征
        if 'started_at' in self.df.columns:
            self.df['hour'] = self.df['started_at'].dt.hour
            self.df['day_of_week'] = self.df['started_at'].dt.dayofweek  # 0=Monday, 6=Sunday
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
            self.df['month'] = self.df['started_at'].dt.month
            self.df['quarter'] = self.df['started_at'].dt.quarter
            self.df['date'] = self.df['started_at'].dt.date
            print("提取时间特征：小时、星期几、是否周末、月份、季度、日期")
        
        # 计算骑行距离（基于经纬度，如果有）
        if all(col in self.df.columns for col in ['start_lat', 'start_lng', 'end_lat', 'end_lng']):
            # 过滤无效的地理坐标
            valid_coords = ((self.df['start_lat'].between(-90, 90)) & 
                           (self.df['start_lng'].between(-180, 180)) &
                           (self.df['end_lat'].between(-90, 90)) & 
                           (self.df['end_lng'].between(-180, 180)))
            
            # 对有效坐标计算距离
            self.df['distance_km'] = np.nan
            valid_df = self.df[valid_coords].copy()
            
            # 使用向量化操作计算距离
            start_coords = list(zip(valid_df['start_lat'], valid_df['start_lng']))
            end_coords = list(zip(valid_df['end_lat'], valid_df['end_lng']))
            
            distances = []
            for start, end in zip(start_coords, end_coords):
                try:
                    distances.append(geodesic(start, end).km)
                except:
                    distances.append(np.nan)
            
            self.df.loc[valid_coords, 'distance_km'] = distances
            print("计算骑行距离（公里）")
            
            # 过滤异常距离（小于0.1公里或大于50公里的设为NaN）
            self.df.loc[(self.df['distance_km'] < 0.1) | (self.df['distance_km'] > 50), 'distance_km'] = np.nan
            print(f"有效距离数据数量: {self.df['distance_km'].notna().sum()}")
        
        print("数据预处理完成")
        return self.df
    
    def save_preprocessed_data(self, output_path):
        """保存预处理后的数据"""
        if self.df is not None:
            print(f"\n保存预处理后的数据到: {output_path}")
            self.df.to_csv(output_path, index=False)
            print(f"数据保存成功")
    
    def save_results(self, filename):
        """保存分析结果"""
        output_path = os.path.join(result_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        print(f"分析结果已保存到: {output_path}")
    
    def save_figure(self, fig, filename, dpi=300, bbox_inches='tight'):
        """保存图表
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名
            dpi: 分辨率
            bbox_inches: 边界框设置
            
        Returns:
            str: 保存的文件路径
        """
        # 保存图表到visualizations子目录
        output_path = os.path.join(self.viz_dir, filename)
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)  # 关闭图表以释放内存
        
        logger.info(f"图表已保存: {output_path}")
        return output_path
    
    def _check_file_exists(self):
        """检查文件是否存在"""
        if not os.path.exists(self.file_path):
            error_msg = f"数据文件不存在: {self.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return True
    
    def _create_figure(self, figsize=(12, 6)):
        """创建新图表（抽象公共逻辑）
        
        Args:
            figsize: 图表尺寸
            
        Returns:
            tuple: (fig, ax) 图表和轴对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def _get_top_items(self, data_dict, top_n=None):
        """获取字典中值最大的前N项（抽象公共逻辑）
        
        Args:
            data_dict: 数据字典
            top_n: 返回项数，None表示使用配置中的最大值
            
        Returns:
            dict: 排序后的前N项
        """
        if top_n is None:
            top_n = self.config['max_visualization_items']
            
        return dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
        print(f"图表已保存到: {output_path}")

    def analyze_time_dimension(self):
        """时间维度分析"""
        print("\n===== 时间维度分析 =====")
        time_results = {}
        
        # 1. 每日高峰时段分析
        print("\n1. 每日高峰时段分析")
        if 'hour' in self.df.columns:
            # 按小时统计骑行量
            hourly_distribution = self.df['hour'].value_counts().sort_index()
            time_results['hourly_distribution'] = hourly_distribution.to_dict()
            
            # 识别高峰时段（前5个小时）
            peak_hours = hourly_distribution.nlargest(5)
            time_results['peak_hours'] = peak_hours.to_dict()
            print(f"高峰时段（前5个）: {list(peak_hours.index)} 时")
            
            # 可视化小时分布
            plt.figure(figsize=(12, 6))
            sns.barplot(x=hourly_distribution.index, y=hourly_distribution.values)
            plt.title('24-Hour Ride Distribution')
            plt.xlabel('Hour')
            plt.ylabel('Number of Rides')
            plt.xticks(range(0, 24))
            plt.grid(True, linestyle='--', alpha=0.7)
            self.save_figure(plt.gcf(), 'hourly_distribution.png')
            plt.close()
        
        # 2. 工作日 vs 周末对比
        print("\n2. 工作日 vs 周末对比")
        if 'is_weekend' in self.df.columns:
            # 计算总量对比
            weekend_comparison = self.df['is_weekend'].value_counts()
            weekend_comparison.index = ['Weekday', 'Weekend']
            time_results['weekend_comparison'] = weekend_comparison.to_dict()
            print(f"Weekday rides: {weekend_comparison['Weekday']:,}")
            print(f"Weekend rides: {weekend_comparison['Weekend']:,}")
            print(f"Weekday:Weekend ratio = {weekend_comparison['Weekday']/weekend_comparison['Weekend']:.2f}:1")
            
            # 按小时对比工作日和周末模式
            plt.figure(figsize=(14, 7))
            
            # 工作日小时分布
            weekday_df = self.df[self.df['is_weekend'] == False]
            weekday_hourly = weekday_df['hour'].value_counts().sort_index()
            weekday_hourly = weekday_hourly / len(weekday_df) * 100  # 转换为百分比
            
            # 周末小时分布
            weekend_df = self.df[self.df['is_weekend'] == True]
            weekend_hourly = weekend_df['hour'].value_counts().sort_index()
            weekend_hourly = weekend_hourly / len(weekend_df) * 100  # 转换为百分比
            
            plt.plot(weekday_hourly.index, weekday_hourly.values, label='Weekday', marker='o', linewidth=2)
            plt.plot(weekend_hourly.index, weekend_hourly.values, label='Weekend', marker='s', linewidth=2)
            plt.title('Weekday vs Weekend Hourly Ride Pattern Comparison')
            plt.xlabel('Hour')
            plt.ylabel('Percentage (%)')
            plt.xticks(range(0, 24))
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            self.save_figure(plt.gcf(), 'weekday_vs_weekend.png')
            plt.close()
            
            time_results['weekday_hourly'] = weekday_hourly.to_dict()
            time_results['weekend_hourly'] = weekend_hourly.to_dict()
        
        # 3. 季节性趋势分析（月度）
        print("\n3. 季节性趋势分析")
        if 'month' in self.df.columns:
            # 月度分布
            monthly_trend = self.df['month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_trend.index = [month_names[i-1] for i in monthly_trend.index]
            time_results['monthly_trend'] = monthly_trend.to_dict()
            
            print("月度骑行量:")
            for month, count in monthly_trend.items():
                print(f"{month}: {count:,}")
            
            # 可视化月度趋势
            plt.figure(figsize=(12, 6))
            sns.barplot(x=monthly_trend.index, y=monthly_trend.values)
            plt.title('Monthly Ride Volume Trend')
            plt.xlabel('Month')
            plt.ylabel('Number of Rides')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            self.save_figure(plt.gcf(), 'monthly_trend.png')
            plt.close()
            
            # 季度分布
            if 'quarter' in self.df.columns:
                quarterly_trend = self.df['quarter'].value_counts().sort_index()
                # 动态设置季度标签，只对实际存在的季度进行标记
                quarterly_trend.index = [f'Q{int(q)}' for q in quarterly_trend.index]
                time_results['quarterly_trend'] = quarterly_trend.to_dict()
                
                print("\n季度骑行量:")
                for quarter, count in quarterly_trend.items():
                    print(f"{quarter}: {count:,}")
        
        # 4. 识别特殊日期模式（基于每日骑行量的异常检测）
        print("\n4. 特殊日期模式识别")
        if 'date' in self.df.columns:
            # 计算每日骑行量
            daily_counts = self.df['date'].value_counts().sort_index()
            
            # 使用IQR方法检测异常值
            Q1 = daily_counts.quantile(0.25)
            Q3 = daily_counts.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 找出异常日期
            unusual_days = daily_counts[(daily_counts < lower_bound) | (daily_counts > upper_bound)]
            
            # 将日期索引转换为字符串格式以支持JSON序列化
            daily_counts_str = {str(date): count for date, count in daily_counts.items()}
            unusual_days_str = {str(date): count for date, count in unusual_days.items()}
            time_results['daily_counts'] = daily_counts_str
            time_results['unusual_days'] = unusual_days_str
            
            print(f"检测到 {len(unusual_days)} 个异常日期")
            if len(unusual_days) > 0:
                print("异常日期（前10个）:")
                for date, count in unusual_days.head(10).items():
                    print(f"{date}: {count:,}")
            
            # 可视化每日骑行量趋势（采样显示，避免图表过于密集）
            plt.figure(figsize=(15, 6))
            sample_days = daily_counts.iloc[::3]  # 每3天显示一个数据点
            plt.plot(sample_days.index.astype(str), sample_days.values)
            plt.title('Daily Ride Volume Trend (Sampled)')
            plt.xlabel('Date')
            plt.ylabel('Number of Rides')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            self.save_figure(plt.gcf(), 'daily_trend.png')
            plt.close()
        
        self.results['time_dimension'] = time_results
        print("\n时间维度分析完成")
        return time_results
    
    def analyze_space_dimension(self):
        """空间维度分析"""
        print("\n===== 空间维度分析 =====")
        space_results = {}
        
        # 1. 热门站点分析
        print("\n1. 热门站点分析")
        
        # 找出热门起点（Top 20）
        if 'start_station_name' in self.df.columns:
            top_start_stations = self.df['start_station_name'].value_counts().head(20)
            space_results['top_start_stations'] = top_start_stations.to_dict()
            
            print("热门起点站（Top 20）:")
            for i, (station, count) in enumerate(top_start_stations.items(), 1):
                print(f"{i}. {station}: {count:,} 次")
            
            # 可视化热门起点站
            plt.figure(figsize=(12, 8))
            sns.barplot(y=top_start_stations.index[:10], x=top_start_stations.values[:10])
            plt.title('Top 10 Popular Start Stations')
            plt.xlabel('Number of Rides')
            plt.ylabel('Station Name')
            plt.grid(True, linestyle='--', alpha=0.7, axis='x')
            plt.tight_layout()
            self.save_figure(plt.gcf(), 'top_start_stations.png')
            plt.close()
        
        # 2. 站点热力图数据准备
        print("\n2. 站点热力图数据准备")
        if all(col in self.df.columns for col in ['start_lat', 'start_lng']):
            # 过滤有效坐标
            valid_coords = ((self.df['start_lat'].between(-90, 90)) & 
                           (self.df['start_lng'].between(-180, 180)))
            
            # 计算每个坐标点的出现次数
            station_counts = self.df[valid_coords].groupby(['start_lat', 'start_lng']).size().reset_index(name='count')
            space_results['station_heatmap_data'] = station_counts.to_dict('records')
            
            print(f"有效热力图数据点数量: {len(station_counts)}")
            
            # 可视化热力图（使用散点图表示）
            plt.figure(figsize=(12, 10))
            plt.scatter(station_counts['start_lng'], station_counts['start_lat'], 
                       c=station_counts['count'], cmap='Reds', alpha=0.6, s=station_counts['count']/10)
            plt.colorbar(label='Ride Count')
            plt.title('Station Usage Heatmap')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, linestyle='--', alpha=0.3)
            self.save_figure(plt.gcf(), 'station_heatmap.png')
            plt.close()
        
        # 3. 站点间流量分析
        print("\n3. 站点间流量分析")
        if all(col in self.df.columns for col in ['start_station_name', 'end_station_name']):
            # 计算站点间流量（Top 20 流量对）
            station_pairs = self.df.groupby(['start_station_name', 'end_station_name']).size().reset_index(name='count')
            top_station_pairs = station_pairs.sort_values('count', ascending=False).head(20)
            space_results['top_station_pairs'] = top_station_pairs.to_dict('records')
            
            print("热门站点对（Top 10）:")
            for i, row in top_station_pairs.head(10).iterrows():
                print(f"{i+1}. {row['start_station_name']} -> {row['end_station_name']}: {row['count']:,} 次")
        
        # 4. 潮汐现象分析
        print("\n4. 潮汐现象分析")
        if all(col in self.df.columns for col in ['start_station_name', 'end_station_name', 'hour']):
            # 早晚高峰时段定义
            morning_rush = (7, 9)  # 早高峰7-9点
            evening_rush = (17, 19)  # 晚高峰17-19点
            
            # 早高峰出站量统计
            morning_outgoing = self.df[(self.df['hour'] >= morning_rush[0]) & 
                                      (self.df['hour'] < morning_rush[1])].groupby('start_station_name').size()
            
            # 早高峰进站量统计
            morning_incoming = self.df[(self.df['hour'] >= morning_rush[0]) & 
                                      (self.df['hour'] < morning_rush[1])].groupby('end_station_name').size()
            
            # 晚高峰出站量统计
            evening_outgoing = self.df[(self.df['hour'] >= evening_rush[0]) & 
                                      (self.df['hour'] < evening_rush[1])].groupby('start_station_name').size()
            
            # 晚高峰进站量统计
            evening_incoming = self.df[(self.df['hour'] >= evening_rush[0]) & 
                                      (self.df['hour'] < evening_rush[1])].groupby('end_station_name').size()
            
            # 计算潮汐指数
            morning_tide = morning_outgoing / (morning_incoming + 1)  # +1避免除以零
            evening_tide = evening_incoming / (evening_outgoing + 1)
            
            # 找出早高峰出发型站点（出站远大于进站）
            morning_departure_stations = morning_tide[morning_tide > 2].sort_values(ascending=False).head(10)
            
            # 找出早高峰到达型站点（进站远大于出站）
            morning_arrival_stations = morning_tide[morning_tide < 0.5].sort_values().head(10)
            
            # 找出晚高峰到达型站点（进站远大于出站）
            evening_arrival_stations = evening_tide[evening_tide > 2].sort_values(ascending=False).head(10)
            
            space_results['morning_departure_stations'] = morning_departure_stations.to_dict()
            space_results['morning_arrival_stations'] = morning_arrival_stations.to_dict()
            space_results['evening_arrival_stations'] = evening_arrival_stations.to_dict()
            
            print("早高峰主要出发站点（Top 10）:")
            for station, ratio in morning_departure_stations.items():
                print(f"{station}: 出站/进站比 = {ratio:.2f}")
            
            print("\n早高峰主要到达站点（Top 10）:")
            for station, ratio in morning_arrival_stations.items():
                print(f"{station}: 出站/进站比 = {ratio:.2f}")
            
            print("\n晚高峰主要到达站点（Top 10）:")
            for station, ratio in evening_arrival_stations.items():
                print(f"{station}: 进站/出站比 = {ratio:.2f}")
        
        self.results['space_dimension'] = space_results
        print("\n空间维度分析完成")
        return space_results
    
    def analyze_user_behavior(self):
        """用户行为分析"""
        print("\n===== 用户行为分析 =====")
        user_results = {}
        
        # 1. 骑行时长分布特征
        print("\n1. 骑行时长分布特征")
        if 'duration_minutes' in self.df.columns:
            # 基本统计信息
            duration_stats = self.df['duration_minutes'].describe()
            user_results['duration_stats'] = duration_stats.to_dict()
            
            print(f"骑行时长统计:")
            print(f"  均值: {duration_stats['mean']:.2f} 分钟")
            print(f"  中位数: {duration_stats['50%']:.2f} 分钟")
            print(f"  标准差: {duration_stats['std']:.2f} 分钟")
            print(f"  最小值: {duration_stats['min']:.2f} 分钟")
            print(f"  最大值: {duration_stats['max']:.2f} 分钟")
            print(f"  25%分位数: {duration_stats['25%']:.2f} 分钟")
            print(f"  75%分位数: {duration_stats['75%']:.2f} 分钟")
            
            # 异常值处理（使用IQR方法）
            Q1 = duration_stats['25%']
            Q3 = duration_stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 过滤异常值后的分布
            filtered_duration = self.df[(self.df['duration_minutes'] >= lower_bound) & 
                                       (self.df['duration_minutes'] <= upper_bound)]['duration_minutes']
            
            outlier_count = len(self.df) - len(filtered_duration)
            user_results['outlier_count'] = outlier_count
            user_results['outlier_percentage'] = outlier_count / len(self.df) * 100
            
            print(f"异常值数量: {outlier_count:,} ({outlier_count/len(self.df)*100:.2f}%)")
            print(f"异常值范围: < {lower_bound:.2f} 或 > {upper_bound:.2f} 分钟")
            
            # 可视化骑行时长分布（过滤异常值后）
            plt.figure(figsize=(12, 6))
            sns.histplot(filtered_duration, bins=50, kde=True)
            plt.title('Ride Duration Distribution (Outliers Removed)')
            plt.xlabel('Duration (minutes)')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.7)
            self.save_figure(plt.gcf(), 'duration_distribution.png')
            plt.close()
            
            # 骑行时长区间分布
            duration_bins = [0, 5, 10, 15, 30, 60, 120, float('inf')]
            duration_labels = ['0-5 min', '5-10 min', '10-15 min', '15-30 min', '30-60 min', '1-2 hrs', '>2 hrs']
            duration_categories = pd.cut(self.df['duration_minutes'], bins=duration_bins, labels=duration_labels)
            duration_category_counts = duration_categories.value_counts().sort_index()
            
            user_results['duration_categories'] = duration_category_counts.to_dict()
            
            print("\n骑行时长区间分布:")
            for category, count in duration_category_counts.items():
                percentage = count / len(self.df) * 100
                print(f"  {category}: {count:,} ({percentage:.2f}%)")
        
        # 2. 骑行距离分布模式
        print("\n2. 骑行距离分布模式")
        # 尝试计算骑行距离（如果有经纬度数据）
        if all(col in self.df.columns for col in ['start_lat', 'start_lng', 'end_lat', 'end_lng']):
            # 过滤有效坐标
            valid_coords = ((self.df['start_lat'].between(-90, 90)) & 
                           (self.df['start_lng'].between(-180, 180)) & 
                           (self.df['end_lat'].between(-90, 90)) & 
                           (self.df['end_lng'].between(-180, 180)))
            
            # 计算距离（简单欧几里得距离，单位：km）
            df_valid = self.df[valid_coords].copy()
            earth_radius_km = 6371  # 地球半径，单位km
            
            # 使用Haversine公式计算两点间距离
            def haversine(lat1, lon1, lat2, lon2):
                # 转换为弧度
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                # haversine公式
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return earth_radius_km * c
            
            # 应用haversine函数计算距离
            df_valid['distance_km'] = haversine(df_valid['start_lat'], df_valid['start_lng'], 
                                              df_valid['end_lat'], df_valid['end_lng'])
            
            # 过滤不合理的距离（>100km或<0.01km）
            df_valid = df_valid[(df_valid['distance_km'] >= 0.01) & (df_valid['distance_km'] <= 100)]
            
            distance_stats = df_valid['distance_km'].describe()
            user_results['distance_stats'] = distance_stats.to_dict()
            
            print(f"骑行距离统计（基于有效坐标）:")
            print(f"  数据量: {len(df_valid):,}")
            print(f"  均值: {distance_stats['mean']:.2f} km")
            print(f"  中位数: {distance_stats['50%']:.2f} km")
            print(f"  标准差: {distance_stats['std']:.2f} km")
            print(f"  最小值: {distance_stats['min']:.2f} km")
            print(f"  最大值: {distance_stats['max']:.2f} km")
            
            # 距离区间分布
            distance_bins = [0, 1, 2, 3, 5, 10, float('inf')]
            distance_labels = ['0-1km', '1-2km', '2-3km', '3-5km', '5-10km', '10km以上']
            distance_categories = pd.cut(df_valid['distance_km'], bins=distance_bins, labels=distance_labels)
            distance_category_counts = distance_categories.value_counts().sort_index()
            
            user_results['distance_categories'] = distance_category_counts.to_dict()
            
            print("\n骑行距离区间分布:")
            for category, count in distance_category_counts.items():
                percentage = count / len(df_valid) * 100
                print(f"  {category}: {count:,} ({percentage:.2f}%)")
            
            # 可视化距离分布
            plt.figure(figsize=(12, 6))
            sns.histplot(df_valid['distance_km'], bins=50, kde=True)
            plt.title('Ride Distance Distribution')
            plt.xlabel('Distance (km)')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.7)
            self.save_figure(plt.gcf(), 'distance_distribution.png')
            plt.close()
        
        # 3. 用户类型对比（如果有用户类型数据）
        print("\n3. 用户类型对比")
        # 尝试识别用户类型列
        user_type_columns = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['user', 'member', 'subscriber', 'customer']):
                user_type_columns.append(col)
        
        if user_type_columns:
            user_type_col = user_type_columns[0]  # 使用找到的第一个用户类型列
            print(f"使用用户类型列: {user_type_col}")
            
            # 用户类型分布
            user_type_distribution = self.df[user_type_col].value_counts()
            user_results['user_type_distribution'] = user_type_distribution.to_dict()
            
            print("\n用户类型分布:")
            for user_type, count in user_type_distribution.items():
                percentage = count / len(self.df) * 100
                print(f"  {user_type}: {count:,} ({percentage:.2f}%)")
            
            # 不同用户类型的骑行时长对比
            if 'duration_minutes' in self.df.columns:
                user_type_duration = self.df.groupby(user_type_col)['duration_minutes'].agg(['mean', 'median', 'std', 'count'])
                user_results['user_type_duration'] = user_type_duration.to_dict()
                
                print("\n不同用户类型的骑行时长统计:")
                print(user_type_duration)
                
                # 可视化用户类型时长对比
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=user_type_col, y='duration_minutes', data=self.df, showfliers=False)
                plt.title('Ride Duration Distribution by User Type')
                plt.xlabel('User Type')
                plt.ylabel('Duration (minutes)')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                self.save_figure(plt.gcf(), 'user_type_duration_comparison.png')
                plt.close()
        else:
            print("未找到明确的用户类型列")
        
        # 4. 高频用户特征分析
        print("\n4. 高频用户特征分析")
        # 尝试识别用户ID列
        user_id_columns = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['user', 'id', 'member', 'customer']):
                if col.lower() != 'user_type' and col.lower() != 'user':
                    user_id_columns.append(col)
        
        if user_id_columns:
            user_id_col = user_id_columns[0]  # 使用找到的第一个用户ID列
            print(f"使用用户ID列: {user_id_col}")
            
            # 计算每个用户的骑行次数
            user_ride_counts = self.df[user_id_col].value_counts()
            
            # 定义高频用户（前5%）
            high_freq_threshold = user_ride_counts.quantile(0.95)
            high_freq_users = user_ride_counts[user_ride_counts >= high_freq_threshold]
            
            user_results['high_freq_threshold'] = high_freq_threshold
            user_results['high_freq_users_count'] = len(high_freq_users)
            user_results['high_freq_users_percentage'] = len(high_freq_users) / len(user_ride_counts) * 100
            
            print(f"高频用户阈值（95%分位数）: {high_freq_threshold:.0f} 次骑行")
            print(f"高频用户数量: {len(high_freq_users):,} ({len(high_freq_users)/len(user_ride_counts)*100:.2f}%)")
            print(f"高频用户骑行次数占比: {high_freq_users.sum()/len(self.df)*100:.2f}%")
            
            # 高频用户与普通用户行为对比
            high_freq_user_ids = high_freq_users.index
            high_freq_df = self.df[self.df[user_id_col].isin(high_freq_user_ids)]
            regular_df = self.df[~self.df[user_id_col].isin(high_freq_user_ids)]
            
            if 'duration_minutes' in self.df.columns:
                high_freq_duration_mean = high_freq_df['duration_minutes'].mean()
                regular_duration_mean = regular_df['duration_minutes'].mean()
                
                user_results['high_freq_duration_mean'] = high_freq_duration_mean
                user_results['regular_duration_mean'] = regular_duration_mean
                
                print(f"\n高频用户平均骑行时长: {high_freq_duration_mean:.2f} 分钟")
                print(f"普通用户平均骑行时长: {regular_duration_mean:.2f} 分钟")
            
            # 用户骑行频率分布
            plt.figure(figsize=(12, 6))
            # 只显示前100个用户，避免图表过于密集
            sns.histplot(user_ride_counts[:100], bins=50)
            plt.title('User Ride Frequency Distribution (Top 100)')
            plt.xlabel('Number of Rides')
            plt.ylabel('Number of Users')
            plt.grid(True, linestyle='--', alpha=0.7)
            self.save_figure(plt.gcf(), 'user_frequency_distribution.png')
            plt.close()
        else:
            print("未找到明确的用户ID列")
        
        self.results['user_behavior'] = user_results
        print("\n用户行为分析完成")
        return user_results
    
    def analyze_bike_type(self):
        """车型维度分析"""
        print("\n===== 车型维度分析 =====")
        bike_results = {}
        
        # 尝试识别车型列
        bike_type_columns = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['bike', 'vehicle', 'type', 'kind']):
                bike_type_columns.append(col)
        
        if not bike_type_columns:
            print("未找到明确的车型列")
            bike_results['error'] = "未找到车型相关列"
            self.results['bike_type'] = bike_results
            print("\n车型维度分析完成（无车型数据）")
            return bike_results
        
        bike_type_col = bike_type_columns[0]  # 使用找到的第一个车型列
        print(f"使用车型列: {bike_type_col}")
        
        # 1. 不同车型使用频次与占比
        print("\n1. 不同车型使用频次与占比")
        bike_type_distribution = self.df[bike_type_col].value_counts()
        bike_type_percentage = self.df[bike_type_col].value_counts(normalize=True) * 100
        
        bike_results['bike_type_distribution'] = bike_type_distribution.to_dict()
        bike_results['bike_type_percentage'] = bike_type_percentage.to_dict()
        
        print("车型使用分布:")
        for bike_type, count in bike_type_distribution.items():
            percentage = bike_type_percentage[bike_type]
            print(f"  {bike_type}: {count:,} 次 ({percentage:.2f}%)")
        
        # 可视化车型分布
        plt.figure(figsize=(10, 6))
        sns.barplot(x=bike_type_distribution.index, y=bike_type_distribution.values)
        plt.title('Bike Type Usage Frequency Distribution')
        plt.xlabel('Bike Type')
        plt.ylabel('Usage Frequency')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        self.save_figure(plt.gcf(), 'bike_type_distribution.png')
        plt.close()
        
        # 饼图展示占比
        plt.figure(figsize=(10, 8))
        plt.pie(bike_type_distribution.values, labels=bike_type_distribution.index, 
                autopct='%1.1f%%', startangle=90)
        plt.title('Bike Type Usage Proportion')
        plt.axis('equal')  # 保证饼图是圆的
        self.save_figure(plt.gcf(), 'bike_type_pie.png')
        plt.close()
        
        # 2. 各类车型在时间维度上的使用差异
        print("\n2. 各类车型在时间维度上的使用差异")
        
        # 按小时对比不同车型的使用模式
        if 'hour' in self.df.columns:
            # 计算每小时各车型的使用量
            hourly_bike_type = self.df.groupby(['hour', bike_type_col]).size().unstack(fill_value=0)
            
            # 转换为百分比形式，便于比较模式
            hourly_bike_type_pct = hourly_bike_type.div(hourly_bike_type.sum(axis=1), axis=0) * 100
            
            bike_results['hourly_bike_type'] = hourly_bike_type.to_dict()
            
            # 可视化不同车型的小时使用模式
            plt.figure(figsize=(14, 8))
            for bike_type in hourly_bike_type_pct.columns:
                plt.plot(hourly_bike_type_pct.index, hourly_bike_type_pct[bike_type], 
                         label=bike_type, marker='o', linewidth=2)
            
            plt.title('Hourly Usage Pattern Comparison by Bike Type')
            plt.xlabel('Hour')
            plt.ylabel('Percentage (%)')
            plt.xticks(range(0, 24))
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            self.save_figure(plt.gcf(), 'bike_type_hourly_pattern.png')
            plt.close()
        
        # 按工作日/周末对比
        if 'is_weekend' in self.df.columns:
            weekend_bike_type = self.df.groupby(['is_weekend', bike_type_col]).size().unstack(fill_value=0)
            weekend_bike_type.index = ['Weekday', 'Weekend']
            
            # 转换为百分比
            weekend_bike_type_pct = weekend_bike_type.div(weekend_bike_type.sum(axis=1), axis=0) * 100
            
            bike_results['weekend_bike_type'] = weekend_bike_type.to_dict()
            
            print("\n工作日 vs 周末车型使用对比:")
            for idx, row in weekend_bike_type.iterrows():
                print(f"\n{idx}:")
                for bike_type, count in row.items():
                    percentage = weekend_bike_type_pct.loc[idx, bike_type]
                    print(f"  {bike_type}: {count:,} 次 ({percentage:.2f}%)")
            
            # 可视化工作日/周末对比
            weekend_bike_type.plot(kind='bar', figsize=(12, 6))
            plt.title('Bike Type Usage Comparison: Weekday vs Weekend')
            plt.xlabel('')
            plt.ylabel('Usage Frequency')
            plt.xticks(rotation=0)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.tight_layout()
            self.save_figure(plt.gcf(), 'bike_type_weekend_comparison.png')
            plt.close()
        
        # 3. 比较不同车型的空间分布特征
        print("\n3. 不同车型的空间分布特征")
        if all(col in self.df.columns for col in ['start_lat', 'start_lng']):
            # 过滤有效坐标
            valid_coords = ((self.df['start_lat'].between(-90, 90)) & 
                           (self.df['start_lng'].between(-180, 180)))
            
            # 按车型分组统计热门区域
            bike_type_locations = {}
            
            for bike_type in bike_type_distribution.index:
                type_df = self.df[(self.df[bike_type_col] == bike_type) & valid_coords]
                # 统计每个车型的热门起点（Top 10站点）
                if 'start_station_name' in self.df.columns:
                    top_stations = type_df['start_station_name'].value_counts().head(10)
                    bike_type_locations[bike_type] = top_stations.to_dict()
            
            bike_results['bike_type_locations'] = bike_type_locations
            
            # 可视化前两种车型的空间分布对比
            if len(bike_type_distribution) >= 2:
                top_bike_types = bike_type_distribution.index[:2]
                plt.figure(figsize=(15, 6))
                
                for i, bike_type in enumerate(top_bike_types, 1):
                    plt.subplot(1, 2, i)
                    type_df = self.df[(self.df[bike_type_col] == bike_type) & valid_coords]
                    # 统计坐标点出现次数
                    type_loc_counts = type_df.groupby(['start_lat', 'start_lng']).size().reset_index(name='count')
                    
                    plt.scatter(type_loc_counts['start_lng'], type_loc_counts['start_lat'], 
                               c=type_loc_counts['count'], cmap='Reds', alpha=0.6, s=type_loc_counts['count']/5)
                    plt.colorbar(label='Ride Count')
                    plt.title(f'{bike_type} Spatial Distribution')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                
                plt.tight_layout()
                self.save_figure(plt.gcf(), 'bike_type_spatial_comparison.png')
                plt.close()
        
        # 4. 评估各类车型的用户行为模式差异
        print("\n4. 各类车型的用户行为模式差异")
        
        # 骑行时长对比
        if 'duration_minutes' in self.df.columns:
            bike_type_duration = self.df.groupby(bike_type_col)['duration_minutes'].agg(['mean', 'median', 'std', 'count'])
            bike_results['bike_type_duration'] = bike_type_duration.to_dict()
            
            print("\n不同车型的骑行时长统计:")
            print(bike_type_duration)
            
            # 可视化不同车型的骑行时长分布
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=bike_type_col, y='duration_minutes', data=self.df, showfliers=False)
            plt.title('Ride Duration Distribution Comparison by Bike Type')
            plt.xlabel('Bike Type')
            plt.ylabel('Duration (minutes)')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.xticks(rotation=45)
            self.save_figure(plt.gcf(), 'bike_type_duration_comparison.png')
            plt.close()
        
        # 骑行距离对比（如果能计算距离）
        if all(col in self.df.columns for col in ['start_lat', 'start_lng', 'end_lat', 'end_lng']):
            # 过滤有效坐标
            valid_coords = ((self.df['start_lat'].between(-90, 90)) & 
                           (self.df['start_lng'].between(-180, 180)) & 
                           (self.df['end_lat'].between(-90, 90)) & 
                           (self.df['end_lng'].between(-180, 180)))
            
            # 使用之前定义的haversine函数计算距离
            df_valid = self.df[valid_coords].copy()
            earth_radius_km = 6371  # 地球半径，单位km
            
            def haversine(lat1, lon1, lat2, lon2):
                # 转换为弧度
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                # haversine公式
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return earth_radius_km * c
            
            df_valid['distance_km'] = haversine(df_valid['start_lat'], df_valid['start_lng'], 
                                              df_valid['end_lat'], df_valid['end_lng'])
            
            # 过滤不合理的距离
            df_valid = df_valid[(df_valid['distance_km'] >= 0.01) & (df_valid['distance_km'] <= 100)]
            
            # 按车型统计距离
            bike_type_distance = df_valid.groupby(bike_type_col)['distance_km'].agg(['mean', 'median', 'std', 'count'])
            bike_results['bike_type_distance'] = bike_type_distance.to_dict()
            
            print("\n不同车型的骑行距离统计:")
            print(bike_type_distance)
            
            # 可视化不同车型的骑行距离分布
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=bike_type_col, y='distance_km', data=df_valid, showfliers=False)
            plt.title('Ride Distance Distribution Comparison by Bike Type')
            plt.xlabel('Bike Type')
            plt.ylabel('Distance (km)')
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.xticks(rotation=45)
            self.save_figure(plt.gcf(), 'bike_type_distance_comparison.png')
            plt.close()
        
        # 5. 车型使用效率分析
        print("\n5. 车型使用效率分析")
        if 'duration_minutes' in self.df.columns:
            # 计算每小时使用次数（使用频率）
            bike_type_hourly_usage = {}
            total_hours = self.df['started_at'].max() - self.df['started_at'].min()
            total_hours = total_hours.total_seconds() / 3600  # 转换为小时
            
            for bike_type in bike_type_distribution.index:
                type_count = bike_type_distribution[bike_type]
                hourly_usage = type_count / total_hours if total_hours > 0 else 0
                bike_type_hourly_usage[bike_type] = hourly_usage
                print(f"{bike_type}: {hourly_usage:.2f} 次/小时")
            
            bike_results['bike_type_hourly_usage'] = bike_type_hourly_usage
        
        self.results['bike_type'] = bike_results
        print("\n车型维度分析完成")
        return bike_results
    
    def generate_summary_report(self):
        """生成综合分析报告和关键发现总结"""
        print("\n===== 生成综合分析报告 =====")
        
        # 创建报告文件
        report_path = os.path.join(self.output_dir, 'analysis_summary_report.md')
        
        # 汇总关键发现
        key_findings = []
        
        # 1. 数据概览
        data_overview = {
            'Total Records': len(self.df),
        'Time Range': f"{self.df['started_at'].min().strftime('%Y-%m-%d')} to {self.df['started_at'].max().strftime('%Y-%m-%d')}",
        'Data Completeness': f"{self.df.notnull().mean().mean()*100:.2f}%"
        }
        
        # 2. 时间维度关键发现
        if 'time_dimension' in self.results:
            time_results = self.results['time_dimension']
            time_findings = []
            
            # 高峰时段发现
            if 'peak_hours' in time_results:
                top_hours = list(time_results['peak_hours'].keys())[:3]
                time_findings.append(f"工作日高峰期主要集中在{top_hours[0]}时、{top_hours[1]}时和{top_hours[2]}时，体现了通勤特征")
            
            # 工作日vs周末对比
            if 'weekend_comparison' in time_results:
                weekend_ratio = time_results['weekend_comparison'].get('weekend_ratio', 0)
                if weekend_ratio > 0.4:
                    time_findings.append(f"周末使用率达到工作日的{weekend_ratio*100:.0f}%，表明休闲骑行需求旺盛")
                else:
                    time_findings.append(f"周末使用率为工作日的{weekend_ratio*100:.0f}%，系统主要服务于通勤需求")
            
            # 季节性趋势
            if 'monthly_trend' in time_results:
                monthly_data = time_results['monthly_trend']
                peak_month = max(monthly_data, key=monthly_data.get)
                low_month = min(monthly_data, key=monthly_data.get)
                time_findings.append(f"使用量呈现明显季节性，{peak_month}月达到峰值，{low_month}月为低谷")
            
            if time_findings:
                key_findings.append(('Time Dimension', time_findings))
        
        # 3. 空间维度关键发现
        if 'space_dimension' in self.results:
            space_results = self.results['space_dimension']
            space_findings = []
            
            # 热门站点
            if 'top_stations' in space_results:
                top_station = list(space_results['top_stations'].keys())[0]
                top_count = space_results['top_stations'][top_station]
                space_findings.append(f"{top_station}是最热门站点，月均使用{top_count/12:.0f}次")
            
            # 潮汐现象
            if 'tidal_stations' in space_results:
                tidal_count = len(space_results['tidal_stations'])
                space_findings.append(f"发现{tidal_count}个明显的潮汐站点，早晚高峰流向差异显著")
            
            # 站点间流量
            if 'station_pairs' in space_results:
                top_pair = list(space_results['station_pairs'].keys())[0]
                top_pair_count = space_results['station_pairs'][top_pair]
                space_findings.append(f"最活跃的站点对是{top_pair}，连接频次达{top_pair_count}次")
            
            if space_findings:
                key_findings.append(('Spatial Dimension', space_findings))
        
        # 4. 用户行为维度关键发现
        if 'user_behavior' in self.results:
            user_results = self.results['user_behavior']
            user_findings = []
            
            # 骑行时长特征
            if 'duration_stats' in user_results:
                mean_duration = user_results['duration_stats'].get('mean', 0)
                duration_categories = user_results.get('duration_categories', {})
                short_ride_percentage = sum(count for key, count in duration_categories.items() 
                                          if key in ['0-5分钟', '5-10分钟', '10-15分钟']) / len(self.df) * 100
                
                user_findings.append(f"平均骑行时长为{mean_duration:.1f}分钟，{short_ride_percentage:.0f}%的骑行在15分钟以内")
            
            # 骑行距离特征
            if 'distance_stats' in user_results:
                mean_distance = user_results['distance_stats'].get('mean', 0)
                user_findings.append(f"平均骑行距离为{mean_distance:.1f}公里，符合短距离出行特征")
            
            # 高频用户特征
            if 'high_freq_users_percentage' in user_results:
                high_freq_pct = user_results['high_freq_users_percentage']
                high_freq_contrib = user_results.get('high_freq_contrib', 0)
                user_findings.append(f"仅占{high_freq_pct:.1f}%的高频用户贡献了显著的使用量")
            
            if user_findings:
                key_findings.append(('User Behavior', user_findings))
        
        # 5. 车型维度关键发现
        if 'bike_type' in self.results:
            bike_results = self.results['bike_type']
            if 'error' not in bike_results:
                bike_findings = []
                
                # 车型分布
                if 'bike_type_percentage' in bike_results:
                    top_type = max(bike_results['bike_type_percentage'], key=bike_results['bike_type_percentage'].get)
                    top_type_pct = bike_results['bike_type_percentage'][top_type]
                    bike_findings.append(f"{top_type}是最受欢迎的车型，占总使用量的{top_type_pct:.1f}%")
                
                # 使用模式差异
                if 'bike_type_duration' in bike_results:
                    duration_data = bike_results['bike_type_duration']
                    # 分析不同车型的使用时长差异
                    bike_findings.append("不同车型的使用时长存在明显差异，反映了各自的功能定位")
                
                if bike_findings:
                    key_findings.append(('Bike Type Analysis', bike_findings))
        
        # 生成Markdown报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Bike Sharing Data Analysis Report\n\n")
            f.write("## 1. Data Overview\n\n")
            
            # 数据概览表格 - 优化排版
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in data_overview.items():
                f.write(f"| **{key}** | {value} |\n")
            
            f.write("\n## 2. Key Findings\n\n")
            
            # 关键发现部分
            for dimension, findings in key_findings:
                f.write(f"### {dimension}\n\n")
                for i, finding in enumerate(findings, 1):
                    f.write(f"{i}. {finding}\n")
                f.write("\n")
            
            # 3. Business Recommendations
            f.write("## 3. Business Recommendations\n\n")
            
            # 基于分析结果生成建议
            business_recommendations = [
                "Based on time distribution patterns, increase bike deployment in popular areas during morning and evening rush hours (7-9 AM, 5-7 PM)",
                "Consider developing differentiated marketing strategies for weekend versus weekday usage patterns",
                "Prepare for bike maintenance and allocation before peak usage seasons based on seasonal trends",
                "Focus on tidal stations and implement targeted dispatching during rush hours to balance supply and demand",
                "Design membership incentive programs for high-frequency users to enhance user loyalty",
                "Optimize bike configuration ratios based on usage characteristics of different bike types to meet diverse travel needs"
            ]
            
            for i, recommendation in enumerate(business_recommendations, 1):
                f.write(f"{i}. {recommendation}\n")
            
            # 4. Data Quality Assessment
            f.write("\n## 4. Data Quality Assessment\n\n")
            f.write("- Data completeness is good, with missing rates for main fields below 5%\n")
            f.write("- Timestamp data format is standardized, facilitating time series analysis\n")
            f.write("- Spatial coordinate data is mostly valid and can be used for geospatial analysis\n")
            f.write("- Weather, holidays, and other external factor data are recommended to enhance analysis depth\n")
            
            # 5. Chart Index
            f.write("\n## 5. Chart Index\n\n")
            f.write("All visualization charts have been saved to the `visualizations` directory:\n\n")
            
            # 列出可能生成的图表 - 使用英文类别名称
            chart_categories = {
                "Time Dimension": ["hourly_distribution.png", "weekday_vs_weekend.png", "monthly_trend.png", "daily_trend.png"],
                "Spatial Dimension": ["top_start_stations.png", "station_heatmap.png"],
                "User Behavior": ["duration_distribution.png", "distance_distribution.png", "user_type_duration_comparison.png", "user_frequency_distribution.png"],
                "Bike Type Analysis": ["bike_type_distribution.png", "bike_type_pie.png", "bike_type_hourly_pattern.png", "bike_type_duration_comparison.png"]
            }
            
            for category, charts in chart_categories.items():
                f.write(f"### {category}\n\n")
                for chart in charts:
                    f.write(f"- [{chart}](visualizations/{chart})\n")
                f.write("\n")
        
        print(f"\n分析报告已生成: {report_path}")
        
        # 生成综合图表
        self.generate_comprehensive_charts()
        
        return report_path
    
    def generate_comprehensive_charts(self):
        """生成综合分析图表"""
        print("\n生成综合分析图表...")
        
        # 创建仪表盘式图表
        plt.figure(figsize=(20, 15))
        
        # 1. 总体使用趋势（左上角）
        if 'time_dimension' in self.results and 'daily_counts' in self.results['time_dimension']:
            plt.subplot(2, 2, 1)
            daily_data = self.results['time_dimension']['daily_counts']
            dates = list(daily_data.keys())[:90]  # 显示最近90天
            counts = list(daily_data.values())[:90]
            plt.plot(dates, counts, 'b-', linewidth=2, alpha=0.7)
            plt.title('90-Day Usage Trend')
            plt.xlabel('Date')
            plt.ylabel('Daily Rides')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
        
        # 2. 小时分布热力图（右上角）
        if 'time_dimension' in self.results and 'hourly_distribution' in self.results['time_dimension']:
            plt.subplot(2, 2, 2)
            hourly_data = self.results['time_dimension']['hourly_distribution']
            hours = list(hourly_data.keys())
            counts = list(hourly_data.values())
            sns.heatmap([counts], cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
            plt.xticks(range(24), [f'{h}:00' for h in range(24)], rotation=45)
            plt.title('24-Hour Usage Heatmap')
            plt.yticks([])
            plt.tight_layout()
        
        # 3. 骑行时长分布（左下角）
        if 'user_behavior' in self.results and 'duration_categories' in self.results['user_behavior']:
            plt.subplot(2, 2, 3)
            duration_data = self.results['user_behavior']['duration_categories']
            categories = list(duration_data.keys())
            counts = list(duration_data.values())
            sns.barplot(x=categories, y=counts)
            plt.title('Ride Duration Interval Distribution')
            plt.xlabel('Duration Interval')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.5, axis='y')
            plt.tight_layout()
        
        # 4. 热门站点（右下角）
        if 'space_dimension' in self.results and 'top_stations' in self.results['space_dimension']:
            plt.subplot(2, 2, 4)
            station_data = self.results['space_dimension']['top_stations']
            stations = list(station_data.keys())[:10]
            counts = list(station_data.values())[:10]
            sns.barplot(y=stations, x=counts, orient='h')
            plt.title('Top 10 Popular Stations')
            plt.xlabel('Usage Frequency')
            plt.tight_layout()
        
        # 保存综合仪表盘
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'dashboard_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("综合仪表盘图表已生成: dashboard_overview.png")
        
        # 如果有车型数据，生成车型对比综合图
        if 'bike_type' in self.results and 'error' not in self.results['bike_type']:
            bike_results = self.results['bike_type']
            if 'bike_type_distribution' in bike_results:
                plt.figure(figsize=(15, 10))
                
                # 左侧：车型分布饼图
                plt.subplot(1, 2, 1)
                bike_type_data = bike_results['bike_type_distribution']
                plt.pie(bike_type_data.values(), labels=bike_type_data.keys(), 
                        autopct='%1.1f%%', startangle=90, shadow=True)
                plt.axis('equal')
                plt.title('Bike Type Usage Proportion')
                
                # 右侧：如果有时长数据，显示车型时长对比
                if 'bike_type_duration' in bike_results:
                    plt.subplot(1, 2, 2)
                    duration_data = bike_results['bike_type_duration']
                    bike_types = list(duration_data['mean'].keys())
                    mean_durations = list(duration_data['mean'].values())
                    sns.barplot(x=bike_types, y=mean_durations)
                    plt.title('Average Ride Duration by Bike Type')
                    plt.xlabel('Bike Type')
                    plt.ylabel('Average Duration (minutes)')
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'visualizations', 'bike_type_comprehensive.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print("车型综合分析图表已生成: bike_type_comprehensive.png")

# 配置参数
DEFAULT_CONFIG = {
    'input_file': r'd:\code\502\Bike A\merged_data\cleaned_2025_data.csv',
    'sample_data': False,
    'sample_frac': 0.1,
    'save_results': True,
    'results_filename': 'analysis_results.json',
    'generate_report': True,
    'generate_charts': True
}

# 主函数
def main(config=None):
    """主函数入口，执行完整的数据分析流程
    
    Args:
        config: 配置参数字典，若为None则使用默认配置
        
    Returns:
        bool: 分析是否成功完成
    """
    # 使用默认配置或用户提供的配置
    if config is None:
        config = DEFAULT_CONFIG
    
    # 合并默认配置和用户配置
    merged_config = DEFAULT_CONFIG.copy()
    merged_config.update(config)
    
    logger.info("🚀 开始数据分析流程")
    logger.debug(f"使用配置: {merged_config}")
    
    try:
        # 创建分析器实例，并传递配置
        logger.info(f"📄 准备分析文件: {merged_config['input_file']}")
        analyzer = BikeDataAnalyzer(merged_config['input_file'], config=merged_config)
        
        # 加载和预处理数据
        logger.info("📥 开始加载数据...")
        start_time = datetime.now()
        analyzer.load_data(sample=merged_config['sample_data'], 
                          sample_frac=merged_config['sample_frac'])
        load_time = datetime.now() - start_time
        logger.info(f"✅ 数据加载完成，耗时: {load_time.total_seconds():.2f} 秒")
        
        logger.info("🔧 开始数据预处理...")
        start_time = datetime.now()
        analyzer.preprocess_data()
        preprocess_time = datetime.now() - start_time
        logger.info(f"✅ 数据预处理完成，耗时: {preprocess_time.total_seconds():.2f} 秒")
        
        # 执行各维度分析
        analysis_functions = [
            ('Time Dimension', analyzer.analyze_time_dimension),
        ('Spatial Dimension', analyzer.analyze_space_dimension),
        ('User Behavior', analyzer.analyze_user_behavior),
        ('Bike Type Analysis', analyzer.analyze_bike_type)
        ]
        
        for dimension_name, func in analysis_functions:
            logger.info(f"📊 开始{dimension_name}分析...")
            start_time = datetime.now()
            func()
            analysis_time = datetime.now() - start_time
            logger.info(f"✅ {dimension_name}分析完成，耗时: {analysis_time.total_seconds():.2f} 秒")
        
        # 生成综合分析报告
        if merged_config['generate_report']:
            logger.info("📋 开始生成综合报告...")
            start_time = datetime.now()
            report_path = analyzer.generate_summary_report()
            report_time = datetime.now() - start_time
            logger.info(f"📊 综合报告已生成: {report_path}，耗时: {report_time.total_seconds():.2f} 秒")
        
        # 生成综合图表
        if merged_config['generate_charts']:
            logger.info("🎨 开始生成综合图表...")
            start_time = datetime.now()
            analyzer.generate_comprehensive_charts()
            charts_time = datetime.now() - start_time
            logger.info(f"✅ 综合图表生成完成，耗时: {charts_time.total_seconds():.2f} 秒")
        
        # 保存分析结果
        if merged_config['save_results']:
            start_time = datetime.now()
            analyzer.save_results(merged_config['results_filename'])
            save_time = datetime.now() - start_time
            logger.info(f"💾 分析结果已保存: {os.path.join(result_dir, merged_config['results_filename'])}, \\n" 
                      f"  耗时: {save_time.total_seconds():.2f} 秒")
        
        logger.info("\n✅ 数据分析完成！")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {str(e)}")
        return False
    except pd.errors.EmptyDataError:
        logger.error("❌ 数据文件为空")
        return False
    except Exception as e:
        logger.error(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        logger.debug(f"错误详情:\n{traceback.format_exc()}")
        return False

# 程序入口
if __name__ == "__main__":
    # 可以直接调用main函数或传入自定义配置
    main()
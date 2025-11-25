import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

class BikeUserTypeAnalyzer:
    def __init__(self, file_path, sample_size=100000):
        """初始化分析器，加载数据"""
        self.file_path = file_path
        self.sample_size = sample_size
        self.df = None
        self.user_type_stats = None
        
    def load_data(self):
        """加载数据并进行基本预处理"""
        print("正在加载数据...")
        
        # 由于文件很大，使用采样
        self.df = pd.read_csv(
            self.file_path,
            parse_dates=['started_at', 'ended_at'],
            low_memory=False
        )
        
        print(f"原始数据行数: {len(self.df):,}")
        
        # 如果数据量超过采样大小，进行随机采样
        if len(self.df) > self.sample_size:
            self.df = self.df.sample(n=self.sample_size, random_state=42)
            print(f"采样后数据行数: {len(self.df):,}")
        
        # 确保骑行时长为正数
        self.df = self.df[self.df['ride_duration_minutes'] > 0]
        
        # 添加时间相关特征
        self.df['start_hour'] = self.df['started_at'].dt.hour
        self.df['start_day'] = self.df['started_at'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['start_month'] = self.df['started_at'].dt.month
        
        print("数据加载完成")
        print(f"用户类型分布: {self.df['member_casual'].value_counts().to_dict()}")
    
    def analyze_user_type_stats(self):
        """分析不同用户类型的基本统计信息"""
        print("\n=== 用户类型基本统计分析 ===")
        
        # 计算基本统计指标
        self.user_type_stats = self.df.groupby('member_casual').agg({
            'ride_id': 'count',
            'ride_duration_minutes': ['mean', 'median', 'std', 'min', 'max'],
            'start_station_id': 'nunique',
            'end_station_id': 'nunique'
        }).round(2)
        
        # 重命名列
        self.user_type_stats.columns = [
            '总骑行次数', '平均骑行时长(分钟)', '中位数骑行时长(分钟)',
            '骑行时长标准差', '最短骑行时长(分钟)', '最长骑行时长(分钟)',
            '使用起始站点数', '使用结束站点数'
        ]
        
        print("\n用户类型统计:")
        print(self.user_type_stats)
        
        # 计算额外指标
        self.user_type_stats['平均每站使用次数'] = (
            self.user_type_stats['总骑行次数'] / 
            ((self.user_type_stats['使用起始站点数'] + self.user_type_stats['使用结束站点数']) / 2)
        ).round(2)
        
        return self.user_type_stats
    
    def analyze_usage_patterns(self):
        """分析不同用户类型的使用模式"""
        print("\n=== 用户使用模式分析 ===")
        
        # 1. 小时使用分布
        print("\n小时使用分布:")
        hourly_dist = self.df.groupby(['member_casual', 'start_hour']).size().unstack()
        
        # 2. 周使用分布
        print("\n周使用分布 (0=周一, 6=周日):")
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_dist = self.df.groupby(['member_casual', 'start_day']).size().unstack()
        weekday_dist.columns = weekday_names
        print(weekday_dist)
        
        # 3. 月份使用分布
        print("\n月份使用分布:")
        month_dist = self.df.groupby(['member_casual', 'start_month']).size().unstack()
        print(month_dist)
        
        return {
            'hourly_dist': hourly_dist,
            'weekday_dist': weekday_dist,
            'month_dist': month_dist
        }
    
    def analyze_ride_duration(self):
        """分析不同用户类型的骑行时长分布"""
        print("\n=== 骑行时长分布分析 ===")
        
        # 计算不同用户类型的骑行时长分布
        duration_stats = {}
        
        for user_type in ['member', 'casual']:
            durations = self.df[self.df['member_casual'] == user_type]['ride_duration_minutes']
            
            # 过滤异常值（超过3小时的骑行）
            filtered_durations = durations[durations <= 180]
            
            duration_stats[user_type] = {
                '总样本数': len(durations),
                '过滤后样本数': len(filtered_durations),
                '平均时长': durations.mean(),
                '中位数时长': durations.median(),
                '75百分位': durations.quantile(0.75),
                '90百分位': durations.quantile(0.90),
                '95百分位': durations.quantile(0.95),
                '99百分位': durations.quantile(0.99)
            }
            
            print(f"\n{user_type.upper()} 用户骑行时长统计:")
            for key, value in duration_stats[user_type].items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        
        return duration_stats
    
    def analyze_station_usage(self):
        """分析不同用户类型的站点使用情况"""
        print("\n=== 站点使用分析 ===")
        
        # 计算每个站点的使用频率
        start_station_counts = self.df.groupby(['member_casual', 'start_station_name']).size().reset_index(name='使用次数')
        
        # 获取每个用户类型最常用的前10个站点
        top_stations = {}
        for user_type in ['member', 'casual']:
            user_stations = start_station_counts[start_station_counts['member_casual'] == user_type]
            top_stations[user_type] = user_stations.nlargest(10, '使用次数')
            
            print(f"\n{user_type.upper()} 用户最常用的前10个起始站点:")
            print(top_stations[user_type][['start_station_name', '使用次数']].to_string(index=False))
        
        return top_stations
    
    def analyze_vehicle_preference(self):
        """分析不同用户类型的车辆偏好"""
        print("\n=== 车辆类型偏好分析 ===")
        
        vehicle_preference = self.df.groupby(['member_casual', 'rideable_type']).size().unstack()
        vehicle_preference_percent = vehicle_preference.div(vehicle_preference.sum(axis=1), axis=0) * 100
        
        print("\n车辆类型使用次数:")
        print(vehicle_preference)
        
        print("\n车辆类型使用百分比:")
        print(vehicle_preference_percent.round(2))
        
        return {
            'counts': vehicle_preference,
            'percentages': vehicle_preference_percent
        }
    
    def create_visualizations(self, output_dir='.'):
        """Create visualization charts"""
        print("\n=== Creating Visualizations ===")
        
        # 1. User Type Distribution Pie Chart
        plt.figure(figsize=(10, 6))
        user_counts = self.df['member_casual'].value_counts()
        plt.pie(user_counts, labels=['Member', 'Casual'], 
                autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05), 
                shadow=True, colors=['#4CAF50', '#FF9800'])
        plt.title('User Type Distribution', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/user_type_distribution.png', dpi=300, bbox_inches='tight')
        
        # 2. Ride Duration Distribution Box Plot
        plt.figure(figsize=(12, 6))
        # Filter outliers
        filtered_df = self.df[self.df['ride_duration_minutes'] <= 120]
        sns.boxplot(x='member_casual', y='ride_duration_minutes', data=filtered_df)
        plt.title('Ride Duration Distribution by User Type (≤120 minutes)', fontsize=16)
        plt.xlabel('User Type')
        plt.ylabel('Ride Duration (minutes)')
        plt.xticks([0, 1], ['Member', 'Casual'])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ride_duration_boxplot.png', dpi=300, bbox_inches='tight')
        
        # 3. Hourly Usage Distribution
        plt.figure(figsize=(14, 7))
        hourly_dist = self.df.groupby(['member_casual', 'start_hour']).size().unstack()
        hourly_dist.plot(kind='line', marker='o')
        plt.title('Hourly Usage Distribution by User Type', fontsize=16)
        plt.xlabel('Hour')
        plt.ylabel('Number of Rides')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.legend(['Member', 'Casual'])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hourly_usage_distribution.png', dpi=300, bbox_inches='tight')
        
        # 4. Weekly Usage Distribution
        plt.figure(figsize=(14, 7))
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_dist = self.df.groupby(['member_casual', 'start_day']).size().unstack()
        weekday_dist.columns = weekday_names
        weekday_dist.plot(kind='bar')
        plt.title('Weekly Usage Distribution by User Type', fontsize=16)
        plt.xlabel('User Type')
        plt.ylabel('Number of Rides')
        plt.xticks([0, 1], ['Member', 'Casual'], rotation=0)
        plt.legend(title='Day of Week')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/weekly_usage_distribution.png', dpi=300, bbox_inches='tight')
        
        # 5. Vehicle Type Preference
        plt.figure(figsize=(12, 6))
        vehicle_counts = self.df.groupby(['member_casual', 'rideable_type']).size().unstack()
        vehicle_counts.plot(kind='bar', stacked=True)
        plt.title('Vehicle Type Preference by User Type', fontsize=16)
        plt.xlabel('User Type')
        plt.ylabel('Number of Rides')
        plt.xticks([0, 1], ['Member', 'Casual'], rotation=0)
        plt.legend(title='Vehicle Type')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/vehicle_preference.png', dpi=300, bbox_inches='tight')
        
        # 6. Monthly Usage Trend
        plt.figure(figsize=(14, 7))
        month_dist = self.df.groupby(['member_casual', 'start_month']).size().unstack()
        month_dist.plot(kind='line', marker='o')
        plt.title('Monthly Usage Trend by User Type', fontsize=16)
        plt.xlabel('Month')
        plt.ylabel('Number of Rides')
        plt.xticks(range(1, 13))
        plt.grid(True, alpha=0.3)
        plt.legend(['Member', 'Casual'])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monthly_trend.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
        print(f"Charts saved to {output_dir} directory")
    
    def generate_report(self, output_file='user_type_analysis_report.txt'):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 共享单车用户类型分析报告\n")
            f.write("\n## 1. 项目概述\n")
            f.write(f"- 分析文件: {self.file_path}\n")
            f.write(f"- 分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 数据量: {len(self.df):,} 条骑行记录\n")
            f.write("\n## 2. 数据概览\n")
            f.write("\n### 2.1 用户类型分布\n")
            user_counts = self.df['member_casual'].value_counts()
            for user_type, count in user_counts.items():
                percentage = count / len(self.df) * 100
                f.write(f"- {user_type} (用户类型): {count:,} 次骑行 ({percentage:.1f}%)\n")
            
            f.write("\n### 2.2 数据字段说明\n")
            f.write("- ride_id: 骑行记录ID\n")
            f.write("- rideable_type: 车辆类型 (classic_bike, electric_bike)\n")
            f.write("- started_at: 开始骑行时间\n")
            f.write("- ended_at: 结束骑行时间\n")
            f.write("- start_station_name/id: 起始站点名称/ID\n")
            f.write("- end_station_name/id: 结束站点名称/ID\n")
            f.write("- start_lat/lng: 起始位置经纬度\n")
            f.write("- end_lat/lng: 结束位置经纬度\n")
            f.write("- member_casual: 用户类型 (member=注册用户, casual=临时用户)\n")
            f.write("- ride_duration_minutes: 骑行时长(分钟)\n")
            
            f.write("\n## 3. 用户类型分析\n")
            f.write("\n### 3.1 基本统计对比\n")
            f.write("\n")
            f.write(self.user_type_stats.to_string())
            
            f.write("\n\n### 3.2 骑行行为特征\n")
            
            # 骑行时长对比
            duration_by_type = self.df.groupby('member_casual')['ride_duration_minutes'].agg(['mean', 'median']).round(2)
            f.write("\n#### 3.2.1 骑行时长对比\n")
            for user_type, row in duration_by_type.iterrows():
                f.write(f"- {user_type} 用户:\n")
                f.write(f"  * 平均骑行时长: {row['mean']} 分钟\n")
                f.write(f"  * 中位数骑行时长: {row['median']} 分钟\n")
            
            # 站点使用情况
            f.write("\n#### 3.2.2 站点使用情况\n")
            start_station_counts = self.df.groupby(['member_casual', 'start_station_name']).size().reset_index(name='使用次数')
            for user_type in ['member', 'casual']:
                user_stations = start_station_counts[start_station_counts['member_casual'] == user_type]
                top_stations = user_stations.nlargest(5, '使用次数')
                f.write(f"\n{user_type.upper()} 用户最常用的5个起始站点:\n")
                for _, row in top_stations.iterrows():
                    f.write(f"  * {row['start_station_name']}: {row['使用次数']} 次\n")
            
            # 车辆类型偏好
            f.write("\n#### 3.2.3 车辆类型偏好\n")
            vehicle_counts = self.df.groupby(['member_casual', 'rideable_type']).size().unstack()
            vehicle_percent = vehicle_counts.div(vehicle_counts.sum(axis=1), axis=0) * 100
            
            for user_type in ['member', 'casual']:
                f.write(f"\n{user_type.upper()} 用户车辆类型使用情况:\n")
                for vehicle_type in vehicle_counts.columns:
                    if vehicle_type in vehicle_counts.loc[user_type]:
                        count = vehicle_counts.loc[user_type, vehicle_type]
                        percent = vehicle_percent.loc[user_type, vehicle_type]
                        f.write(f"  * {vehicle_type}: {count:,} 次 ({percent:.1f}%)\n")
            
            f.write("\n### 3.3 使用时间模式\n")
            
            # 小时分布
            f.write("\n#### 3.3.1 小时使用分布\n")
            hourly_dist = self.df.groupby(['member_casual', 'start_hour']).size().unstack()
            for user_type in ['member', 'casual']:
                peak_hour = hourly_dist.loc[user_type].idxmax()
                f.write(f"- {user_type} 用户高峰使用时段: {peak_hour}:00 点\n")
            
            # 周分布
            f.write("\n#### 3.3.2 周使用分布\n")
            weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            weekday_dist = self.df.groupby(['member_casual', 'start_day']).size().unstack()
            weekday_dist.columns = weekday_names
            
            for user_type in ['member', 'casual']:
                f.write(f"\n{user_type.upper()} 用户周使用分布:\n")
                for day, count in weekday_dist.loc[user_type].items():
                    f.write(f"  * {day}: {count:,} 次\n")
            
            f.write("\n## 4. 关键发现\n")
            
            # 基于分析结果生成关键发现
            member_durations = self.df[self.df['member_casual'] == 'member']['ride_duration_minutes']
            casual_durations = self.df[self.df['member_casual'] == 'casual']['ride_duration_minutes']
            
            f.write("\n### 4.1 用户行为差异\n")
            if casual_durations.mean() > member_durations.mean():
                f.write("- 临时用户的平均骑行时长比注册用户长\n")
            else:
                f.write("- 注册用户的平均骑行时长比临时用户长\n")
            
            # 站点使用差异
            member_stations = self.df[self.df['member_casual'] == 'member']['start_station_id'].nunique()
            casual_stations = self.df[self.df['member_casual'] == 'casual']['start_station_id'].nunique()
            
            if member_stations > casual_stations:
                f.write("- 注册用户使用的站点范围更广\n")
            else:
                f.write("- 临时用户使用的站点范围更广\n")
            
            f.write("\n### 4.2 使用模式洞察\n")
            # 分析工作日vs周末使用模式
            member_weekday = self.df[(self.df['member_casual'] == 'member') & (self.df['start_day'] < 5)].shape[0]
            member_weekend = self.df[(self.df['member_casual'] == 'member') & (self.df['start_day'] >= 5)].shape[0]
            casual_weekday = self.df[(self.df['member_casual'] == 'casual') & (self.df['start_day'] < 5)].shape[0]
            casual_weekend = self.df[(self.df['member_casual'] == 'casual') & (self.df['start_day'] >= 5)].shape[0]
            
            member_weekday_ratio = member_weekday / (member_weekday + member_weekend) * 100
            casual_weekday_ratio = casual_weekday / (casual_weekday + casual_weekend) * 100
            
            f.write(f"- 注册用户工作日使用率: {member_weekday_ratio:.1f}%\n")
            f.write(f"- 临时用户工作日使用率: {casual_weekday_ratio:.1f}%\n")
            
            if member_weekday_ratio > casual_weekday_ratio:
                f.write("- 注册用户更倾向于工作日使用，可能用于通勤\n")
                f.write("- 临时用户更倾向于周末使用，可能用于休闲骑行\n")
            else:
                f.write("- 临时用户更倾向于工作日使用\n")
                f.write("- 注册用户在工作日和周末的使用相对均衡\n")
            
            f.write("\n## 5. 业务建议\n")
            f.write("\n### 5.1 用户获取策略\n")
            f.write("- 针对临时用户，推出灵活的会员套餐，降低注册门槛\n")
            f.write("- 在临时用户集中的站点加强会员推广\n")
            f.write("\n### 5.2 运营优化建议\n")
            f.write("- 根据不同用户类型的使用高峰，优化车辆调度\n")
            f.write("- 在工作日高峰期，增加注册用户集中区域的车辆投放\n")
            f.write("- 在周末，加强临时用户集中区域的车辆供应\n")
            f.write("\n### 5.3 产品改进建议\n")
            f.write("- 针对注册用户，开发通勤相关功能，如定点预约、常用路线保存\n")
            f.write("- 针对临时用户，提供更便捷的租车流程和景点周边推荐\n")
            f.write("- 根据车辆类型偏好，调整不同区域的车辆投放比例\n")
            
            f.write("\n## 6. 结论\n")
            f.write("\n本次分析通过对共享单车用户类型的深入研究，揭示了注册用户和临时用户在使用行为、时间模式和偏好上的显著差异。这些发现为运营策略制定、资源配置优化和产品改进提供了数据支持。建议进一步结合地理空间分析和用户生命周期价值评估，以实现更精准的用户运营和业务增长。")
        
        print(f"分析报告已保存到 {output_file}")

# 主函数
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = BikeUserTypeAnalyzer(
        file_path='d:\\code\\502\\Bike A\\merged_data\\cleaned_2025_data.csv',
        sample_size=500000  # 使用较大的采样量以获得更准确的结果
    )
    
    # 执行分析流程
    analyzer.load_data()
    analyzer.analyze_user_type_stats()
    analyzer.analyze_usage_patterns()
    analyzer.analyze_ride_duration()
    analyzer.analyze_station_usage()
    analyzer.analyze_vehicle_preference()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 生成报告
    analyzer.generate_report()
    
    print("\n=== 分析完成 ===")
    print("请查看生成的报告文件和图表以获取详细分析结果。")
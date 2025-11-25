import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_featured_data(featured_path):
    """加载特征工程后的数据，筛选早高峰时段数据"""
    df = pd.read_csv(featured_path, low_memory=False)
    # 还原时间字段
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    
    # 筛选早高峰数据：工作日 + 7:00-9:00
    df["is_weekday"] = df["started_at"].dt.weekday.apply(lambda x: 1 if x < 5 else 0)  # 1=工作日
    df["hour"] = df["started_at"].dt.hour
    rush_hour_df = df[(df["is_weekday"] == 1) & (df["hour"].between(7, 9))]  # 早高峰7-9点
    
    print(f"早高峰数据加载完成：共 {len(rush_hour_df)} 条骑行记录")
    return rush_hour_df

def calculate_supply_demand(rush_hour_df):
    """计算每个站点的供需指标：出发量（需求）、到达量（供给）、供需缺口"""
    # 1. 统计各站点的出发量（需求：用户从该站骑车，消耗车辆）
    demand = rush_hour_df.groupby("start_station_name").size().reset_index(name="demand_count")
    # 2. 统计各站点的到达量（供给：用户骑车到该站，补充车辆）
    supply = rush_hour_df.groupby("end_station_name").size().reset_index(name="supply_count")
    
    # 3. 合并供需数据（确保所有站点都被包含，缺失值用0填充）
    supply_demand = pd.merge(
        demand, supply,
        left_on="start_station_name", right_on="end_station_name",
        how="outer", suffixes=("_demand", "_supply")
    )
    # 统一站点名称列
    supply_demand["station_name"] = supply_demand["start_station_name"].fillna(supply_demand["end_station_name"])
    # 填充缺失的供需量（没有出发/到达记录的站点，供需量为0）
    supply_demand["demand_count"] = supply_demand["demand_count"].fillna(0).astype(int)
    supply_demand["supply_count"] = supply_demand["supply_count"].fillna(0).astype(int)
    
    # 4. 计算核心指标
    supply_demand["net_demand"] = supply_demand["demand_count"] - supply_demand["supply_count"]  # 净需求（正数=缺口，负数=过剩）
    supply_demand["supply_demand_ratio"] = (supply_demand["demand_count"] / supply_demand["supply_count"]).round(2)  # 供需比（>1=缺口）
    
    # 处理分母为0的情况（无到达量的站点，供需比设为"极高需求"）
    supply_demand["supply_demand_ratio"] = supply_demand["supply_demand_ratio"].replace([float('inf'), -float('inf')], "极高需求")
    
    # 5. 标记缺口类型
    def classify_gap(row):
        if row["net_demand"] > 50:  # 净需求>50：严重缺口（可根据数据调整阈值）
            return "严重缺口"
        elif 0 < row["net_demand"] <= 50:  # 净需求1-50：轻微缺口
            return "轻微缺口"
        elif -50 <= row["net_demand"] <= 0:  # 净需求-50-0：轻微过剩
            return "轻微过剩"
        else:  # 净需求<-50：严重过剩
            return "严重过剩"
    supply_demand["gap_type"] = supply_demand.apply(classify_gap, axis=1)
    
    # 只保留有意义的列，按净需求降序排序（缺口最大的在前）
    result = supply_demand[["station_name", "demand_count", "supply_count", "net_demand", "supply_demand_ratio", "gap_type"]]
    result = result.sort_values("net_demand", ascending=False).reset_index(drop=True)
    
    return result

def visualize_gap_top_stations(supply_demand_result, save_dir):
    """可视化早高峰热门站点的供需缺口（Top 10缺口站点 + Top 10过剩站点）"""
    # 筛选Top 10严重缺口站点和Top 10严重过剩站点
    top_gap_stations = supply_demand_result[supply_demand_result["gap_type"] == "严重缺口"].head(10)
    top_surplus_stations = supply_demand_result[supply_demand_result["gap_type"] == "严重过剩"].tail(10)  # tail取最后10个（净需求最小）
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Top 10严重缺口站点（需补车）
    sns.barplot(
        y="station_name", x="net_demand", 
        data=top_gap_stations, 
        palette="Reds",  # 红色系表示缺口
        ax=ax1
    )
    ax1.set_title("早高峰严重缺口站点 Top 10（需重点补车）", fontsize=14)
    ax1.set_xlabel("净需求（出发量-到达量）", fontsize=12)
    ax1.set_ylabel("站点名称", fontsize=12)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    # 在柱状图上添加数值标签
    for i, v in enumerate(top_gap_stations["net_demand"]):
        ax1.text(v + 1, i, str(v), va="center", fontsize=10)
    
    # 2. Top 10严重过剩站点（需挪车）
    sns.barplot(
        y="station_name", x="net_demand", 
        data=top_surplus_stations, 
        palette="Greens",  # 绿色系表示过剩
        ax=ax2
    )
    ax2.set_title("早高峰严重过剩站点 Top 10（需重点挪车）", fontsize=14)
    ax2.set_xlabel("净需求（出发量-到达量）", fontsize=12)
    ax2.set_ylabel("站点名称", fontsize=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    # 在柱状图上添加数值标签
    for i, v in enumerate(top_surplus_stations["net_demand"]):
        ax2.text(v - 10, i, str(v), va="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rush_hour_supply_demand_gap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 早高峰供需缺口可视化图表已保存")

def visualize_gap_distribution(supply_demand_result, save_dir):
    """可视化所有站点的缺口类型分布（饼图）"""
    gap_type_count = supply_demand_result["gap_type"].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = ["#ff6b6b", "#ffd93d", "#6fdd6f", "#4ecdc4"]  # 红（严重缺口）、黄（轻微缺口）、绿（轻微过剩）、青（严重过剩）
    plt.pie(
        gap_type_count, 
        labels=gap_type_count.index, 
        autopct="%1.1f%%",  # 显示百分比
        startangle=90,
        colors=colors,
        textprops={"fontsize": 12}
    )
    plt.title("早高峰共享单车站点供需缺口类型分布", fontsize=14)
    plt.axis("equal")  # 保证饼图为正圆形
    plt.savefig(os.path.join(save_dir, "rush_hour_gap_type_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 早高峰缺口类型分布饼图已保存")

def save_supply_demand_result(supply_demand_result, save_dir):
    """保存供需缺口分析结果到CSV文件（便于后续查看详细数据）"""
    save_path = os.path.join(save_dir, "rush_hour_supply_demand_result.csv")
    supply_demand_result.to_csv(save_path, index=False, encoding="utf-8-sig")  # utf-8-sig支持中文显示
    print(f"✅ 供需缺口分析结果已保存至：{save_path}")

def main():
    # 1. 路径配置（适配你的文件夹结构）
    featured_data_path = "../processed_data/2024_featured.csv"  # 特征工程后的数据
    result_save_dir = "../results/2024 charts/supply_demand_analysis"  # 供需分析结果保存目录
    charts_save_dir = "../results/charts/2024 charts"  # 图表保存目录
    
    # 2. 创建保存目录
    os.makedirs(result_save_dir, exist_ok=True)
    os.makedirs(charts_save_dir, exist_ok=True)
    
    # 3. 执行供需缺口分析流程
    print("===== 开始早高峰热门站点供需缺口分析 =====")
    rush_hour_df = load_featured_data(featured_data_path)
    supply_demand_result = calculate_supply_demand(rush_hour_df)
    
    # 4. 输出关键统计信息
    print(f"\n【早高峰供需缺口关键统计】")
    print(f"总站点数：{len(supply_demand_result)}")
    print(f"严重缺口站点数：{len(supply_demand_result[supply_demand_result['gap_type'] == '严重缺口'])}")
    print(f"严重过剩站点数：{len(supply_demand_result[supply_demand_result['gap_type'] == '严重过剩'])}")
    print(f"缺口最大的站点：{supply_demand_result.iloc[0]['station_name']}（净需求：{supply_demand_result.iloc[0]['net_demand']}）")
    
    # 5. 可视化与结果保存
    visualize_gap_top_stations(supply_demand_result, charts_save_dir)
    visualize_gap_distribution(supply_demand_result, charts_save_dir)
    save_supply_demand_result(supply_demand_result, result_save_dir)
    
    print("\n===== 早高峰供需缺口分析全部完成！ =====")

if __name__ == "__main__":
    main()
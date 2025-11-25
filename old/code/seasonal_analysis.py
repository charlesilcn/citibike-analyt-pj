import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(featured_path):
    """加载特征工程后的数据，添加季节标签"""
    df = pd.read_csv(featured_path, low_memory=False, parse_dates=['started_at'])
    
    # 定义季节划分函数（3-5春，6-8夏，9-11秋，12-2冬）
    def get_season(month):
        if 3 <= month <= 5:
            return "春季"
        elif 6 <= month <= 8:
            return "夏季"
        elif 9 <= month <= 11:
            return "秋季"
        else:
            return "冬季"
    
    df["month"] = df["started_at"].dt.month
    df["season"] = df["month"].apply(get_season)
    
    # 过滤无效路线（起点≠终点）
    valid_df = df[df["start_station_name"] != df["end_station_name"]].copy()
    print(f"数据加载完成：共 {len(valid_df):,} 条有效记录，包含 {valid_df['season'].nunique()} 个季节")
    return valid_df

def analyze_seasonal_rides(df, save_dir):
    """1. 不同季节的骑行量对比"""
    # 按季节统计总骑行量
    season_rides = df.groupby("season").size().reindex(["春季", "夏季", "秋季", "冬季"]).reset_index(name="ride_count")
    
    # 可视化（修复palette警告：显式指定hue并关闭legend）
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="season", 
        y="ride_count", 
        data=season_rides, 
        hue="season",  # 显式关联x和hue
        palette="viridis", 
        legend=False  # 关闭图例
    )
    plt.title("不同季节的骑行总量对比", fontsize=14)
    plt.xlabel("季节", fontsize=12)
    plt.ylabel("骑行次数", fontsize=12)
    # 添加数值标签
    for i, v in enumerate(season_rides["ride_count"]):
        plt.text(i, v + 10000, f"{v:,}", ha="center", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "seasonal_ride_count.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 季节骑行量对比图已保存")
    return season_rides

def analyze_seasonal_routes(df, save_dir, top_n=5):
    """2. 不同季节的热门线路对比（Top5路线）"""
    # 按季节分组，统计各路线的骑行次数
    season_routes = {}
    for season in ["春季", "夏季", "秋季", "冬季"]:
        season_df = df[df["season"] == season]
        top_routes = season_df["route"].value_counts().head(top_n).reset_index()
        top_routes.columns = ["route", "count"]
        season_routes[season] = top_routes
    
    # 可视化（4个子图对比）
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle(f"不同季节的热门骑行路线 Top {top_n}", fontsize=18)
    seasons = ["春季", "夏季", "秋季", "冬季"]
    # 颜色列表（修正：用列表传递颜色，而非单个颜色码）
    colors = [
        ["#4CAF50"] * top_n,  # 春季：绿色系
        ["#FF9800"] * top_n,  # 夏季：橙色系
        ["#2196F3"] * top_n,  # 秋季：蓝色系
        ["#FF5722"] * top_n   # 冬季：红色系
    ]
    
    for i, (season, ax) in enumerate(zip(seasons, axes.flat)):
        sns.barplot(
            y="route", 
            x="count", 
            data=season_routes[season], 
            hue="route",  # 显式关联y和hue
            palette=colors[i],  # 传递颜色列表
            ax=ax,
            legend=False  # 关闭图例
        )
        ax.set_title(f"{season}热门路线", fontsize=14)
        ax.set_xlabel("骑行次数", fontsize=12)
        ax.set_ylabel("路线（起点→终点）", fontsize=12)
        # 添加数值标签
        for j, v in enumerate(season_routes[season]["count"]):
            ax.text(v + 5, j, f"{v:,}", va="center", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 预留标题空间
    plt.savefig(os.path.join(save_dir, f"seasonal_top{top_n}_routes.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 季节热门路线对比图已保存")
    return season_routes

def analyze_seasonal_rush_hour(df, save_dir):
    """3. 不同季节的早晚高峰对比（早7-9点，晚17-19点）"""
    # 标记早晚高峰
    df["hour"] = df["started_at"].dt.hour
    df["rush_hour"] = df["hour"].apply(
        lambda x: "早高峰" if 7 <= x <= 9 else ("晚高峰" if 17 <= x <= 19 else "非高峰")
    )
    
    # 按季节和高峰类型统计骑行量
    season_rush = df[df["rush_hour"] != "非高峰"].groupby(["season", "rush_hour"]).size().unstack()
    season_rush = season_rush.reindex(["春季", "夏季", "秋季", "冬季"])  # 按季节排序
    
    # 可视化（修复palette警告）
    plt.figure(figsize=(12, 7))
    season_rush.plot(
        kind="bar", 
        width=0.7, 
        color=["#FF5722", "#2196F3"],  # 早高峰红色，晚高峰蓝色
        ax=plt.gca()
    )
    plt.title("不同季节的早晚高峰骑行量对比", fontsize=14)
    plt.xlabel("季节", fontsize=12)
    plt.ylabel("骑行次数", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="高峰类型")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "seasonal_rush_hour.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 季节早晚高峰对比图已保存")
    return season_rush

def save_results(season_rides, season_routes, season_rush, save_dir):
    """保存分析结果到CSV"""
    # 季节骑行量
    season_rides.to_csv(os.path.join(save_dir, "季节骑行量统计.csv"), index=False, encoding="utf-8-sig")
    # 季节热门路线
    for season, routes in season_routes.items():
        routes.to_csv(os.path.join(save_dir, f"{season}_热门路线.csv"), index=False, encoding="utf-8-sig")
    # 季节高峰统计
    season_rush.to_csv(os.path.join(save_dir, "季节早晚高峰统计.csv"), encoding="utf-8-sig")
    print(f"✅ 所有分析结果已保存至：{save_dir}")

def main():
    # 路径配置（适配你的项目结构）
    featured_path = "../processed_data/2024_featured.csv"  # 特征工程后的数据
    charts_dir = "../results/charts/2024 charts/seasonal_analysis"      # 图表保存目录
    results_dir = "../results/2024 charts/seasonal_analysis"            # 结果数据目录
    
    # 创建目录
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 执行分析
    print("===== 开始季节（季度）骑行数据分析 =====")
    df = load_data(featured_path)
    season_rides = analyze_seasonal_rides(df, charts_dir)
    season_routes = analyze_seasonal_routes(df, charts_dir)
    season_rush = analyze_seasonal_rush_hour(df, charts_dir)
    save_results(season_rides, season_routes, season_rush, results_dir)
    print("===== 季节分析完成！ =====")

if __name__ == "__main__":
    main()
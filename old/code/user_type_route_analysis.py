import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def haversine_distance(lat1, lon1, lat2, lon2):
    """纯numpy实现地理距离计算"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)** 2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # 地球半径（公里）
    return c * r

def load_data(featured_path):
    use_cols = [
        'started_at', 'member_casual', 'route', 'duration_min',
        'start_station_name', 'end_station_name',
        'start_lat', 'start_lng', 'end_lat', 'end_lng'
    ]
    df = pd.read_csv(
        featured_path, 
        usecols=use_cols,
        parse_dates=['started_at'],
        low_memory=False
    )
    
    valid_mask = (
        df["start_station_name"].notnull() &
        df["end_station_name"].notnull() &
        (df["start_station_name"] != df["end_station_name"]) &
        (df["duration_min"] <= 60) &
        df[['start_lat', 'start_lng', 'end_lat', 'end_lng']].notnull().all(axis=1)
    )
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) > 10_000_000:
        valid_df = valid_df.sample(frac=0.5, random_state=42)
    
    print(f"数据加载完成：{len(valid_df):,} 条有效记录")
    return valid_df

def calculate_route_distance(df):
    df["route_distance_km"] = haversine_distance(
        df["start_lat"].values,
        df["start_lng"].values,
        df["end_lat"].values,
        df["end_lng"].values
    ).round(2)
    
    df = df[df["route_distance_km"] <= 20]
    print(f"路线距离计算完成：{len(df):,} 条记录")
    return df

def analyze_popular_routes(df, save_dir, top_n=5):
    member_routes = df[df["member_casual"] == "member"]["route"].value_counts().head(top_n).reset_index()
    casual_routes = df[df["member_casual"] == "casual"]["route"].value_counts().head(top_n).reset_index()
    
    member_routes.columns = ["route", "count"]
    casual_routes.columns = ["route", "count"]
    
    # 解决palette警告：显式指定hue并关闭legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(f"会员与非会员热门骑行路线 Top {top_n}", fontsize=16)
    
    sns.barplot(
        y="route", x="count", 
        data=member_routes, 
        hue="route",  # 显式指定hue
        palette="Blues", 
        ax=ax1,
        legend=False  # 关闭图例
    )
    ax1.set_title("会员热门路线")
    ax1.set_xlabel("骑行次数")
    ax1.set_ylabel("路线（起点→终点）")
    [ax1.text(v+5, i, f"{v:,}", va="center") for i, v in enumerate(member_routes["count"])]
    
    sns.barplot(
        y="route", x="count", 
        data=casual_routes, 
        hue="route",  # 显式指定hue
        palette="Oranges", 
        ax=ax2,
        legend=False  # 关闭图例
    )
    ax2.set_title("非会员热门路线")
    ax2.set_xlabel("骑行次数")
    ax2.set_ylabel("路线（起点→终点）")
    [ax2.text(v+5, i, f"{v:,}", va="center") for i, v in enumerate(casual_routes["count"])]
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"top{top_n}_routes.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return member_routes, casual_routes

def analyze_distance_duration(df, save_dir):
    sample_df = df.sample(frac=0.1, random_state=42)
    
    # 解决palette警告：显式指定hue并关闭legend
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="member_casual", 
        y="route_distance_km", 
        data=sample_df, 
        hue="member_casual",  # 显式指定hue
        palette=["#66b3ff", "#ff9999"],
        legend=False  # 关闭图例
    )
    plt.title("会员与非会员骑行路线距离对比")
    plt.xlabel("用户类型")
    plt.ylabel("距离（公里）")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "distance_comparison.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="member_casual", 
        y="duration_min", 
        data=sample_df, 
        hue="member_casual",  # 显式指定hue
        palette=["#66b3ff", "#ff9999"],
        legend=False  # 关闭图例
    )
    plt.title("会员与非会员骑行时长对比")
    plt.xlabel("用户类型")
    plt.ylabel("时长（分钟）")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "duration_comparison.png"), dpi=300)
    plt.close()
    
    distance_stats = df.groupby("member_casual")["route_distance_km"].agg(["mean", "median"]).round(2)
    duration_stats = df.groupby("member_casual")["duration_min"].agg(["mean", "median"]).round(2)
    return distance_stats, duration_stats

def analyze_time_pattern(df, save_dir):
    # 1. 工作日/周末分布（移除keepdims参数，兼容旧pandas）
    df["is_weekday"] = (df["started_at"].dt.weekday < 5).astype(int)
    weekday_dist = df.groupby(["member_casual", "is_weekday"]).size().unstack()
    
    # 计算占比（不使用keepdims，改用reshape实现相同效果）
    row_sums = weekday_dist.sum(axis=1)  # 行总和
    weekday_dist = (weekday_dist.T / row_sums).T * 100  # 转置后计算占比，再转置回来
    weekday_dist = weekday_dist.round(1)
    
    plt.figure(figsize=(10, 6))
    weekday_dist.plot(kind="bar", color=["#ffcc99", "#99ccff"], width=0.6)
    plt.title("工作日/周末骑行占比")
    plt.xlabel("用户类型")
    plt.ylabel("占比（%）")
    plt.xticks(rotation=0)
    plt.legend(["周末", "工作日"])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "weekday_weekend.png"), dpi=300)
    plt.close()
    
    # 2. 时段分布（同样移除keepdims）
    hours = df["started_at"].dt.hour
    df["time_period"] = np.select(
        [hours.between(6, 11), hours.between(12, 17), hours.between(18, 23)],
        ["早上", "下午", "晚上"],
        default="凌晨"
    )
    period_dist = df.groupby(["member_casual", "time_period"]).size().unstack()
    
    # 计算占比（兼容旧pandas）
    row_sums_period = period_dist.sum(axis=1)
    period_dist = (period_dist.T / row_sums_period).T * 100
    period_dist = period_dist.round(1)
    
    plt.figure(figsize=(12, 6))
    period_dist.plot(kind="bar", color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"], width=0.7)
    plt.title("不同时段骑行占比")
    plt.xlabel("用户类型")
    plt.ylabel("占比（%）")
    plt.xticks(rotation=0)
    plt.legend(title="时段")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "time_period.png"), dpi=300)
    plt.close()

def save_results(member_routes, casual_routes, distance_stats, duration_stats, save_dir):
    member_routes.to_csv(os.path.join(save_dir, "会员热门路线.csv"), index=False, encoding="utf-8-sig")
    casual_routes.to_csv(os.path.join(save_dir, "非会员热门路线.csv"), index=False, encoding="utf-8-sig")
    distance_stats.to_csv(os.path.join(save_dir, "距离统计.csv"), encoding="utf-8-sig")
    duration_stats.to_csv(os.path.join(save_dir, "时长统计.csv"), encoding="utf-8-sig")
    print(f"分析结果已保存至：{save_dir}")

def main():
    featured_path = "../processed_data/2024_featured.csv"
    charts_dir = "../results/charts/2024 charts"
    results_dir = "../results/2024 charts/user_analysis"
    
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("===== 开始会员与非会员路线行为分析 =====")
    df = load_data(featured_path)
    df = calculate_route_distance(df)
    member_routes, casual_routes = analyze_popular_routes(df, charts_dir)
    distance_stats, duration_stats = analyze_distance_duration(df, charts_dir)
    analyze_time_pattern(df, charts_dir)
    save_results(member_routes, casual_routes, distance_stats, duration_stats, results_dir)
    
    print("===== 分析完成！ =====")

if __name__ == "__main__":
    main()
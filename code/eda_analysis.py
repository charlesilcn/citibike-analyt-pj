import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_featured_data(featured_path):
    df = pd.read_csv(featured_path, low_memory=False)
    df["started_at"] = pd.to_datetime(df["started_at"])
    print(f"加载EDA数据：共 {df.shape[0]} 行，{df.shape[1]} 列")
    return df

def analyze_time_trends(df, save_dir):
    # 1. 月度骑行量（更换为官方支持的调色板 'Blues'）
    monthly_trips = df.groupby("month").size().reset_index(name="trip_count")
    plt.figure(figsize=(12, 6))
    sns.barplot(x="month", y="trip_count", data=monthly_trips, palette="Blues")  # 修改此处
    plt.title("2020年各月骑行量趋势", fontsize=14)
    plt.xlabel("月份", fontsize=12)
    plt.ylabel("骑行次数（万次）", fontsize=12)
    plt.xticks(range(12), [f"{i}月" for i in range(1, 13)])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "2020_monthly_trips.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 月度骑行量图表已保存")

    # 2. 工作日vs周末的小时骑行量（使用官方支持的颜色）
    weekday_hour = df[df["is_weekday"] == 1].groupby("hour").size().reset_index(name="trip_count")
    weekend_hour = df[df["is_weekday"] == 0].groupby("hour").size().reset_index(name="trip_count")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="hour", y="trip_count", data=weekday_hour, label="工作日", linewidth=2, color="blue")  # 修改此处
    sns.lineplot(x="hour", y="trip_count", data=weekend_hour, label="周末", linewidth=2, color="orange")  # 修改此处
    plt.title("2020年工作日vs周末每小时骑行量对比", fontsize=14)
    plt.xlabel("小时（0-23）", fontsize=12)
    plt.ylabel("骑行次数", fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.legend(fontsize=11)
    plt.grid(linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "2020_weekday_weekend_hour.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 工作日vs周末小时骑行量图表已保存")

def analyze_user_behavior(df, save_dir):
    # 1. 用户类型占比（更换为官方支持的调色板）
    user_count = df["member_casual"].value_counts()
    plt.figure(figsize=(8, 8))
    colors = ["green", "red"]  # 修改为基础颜色
    plt.pie(user_count, labels=user_count.index, autopct="%1.1f%%",
            startangle=90, colors=colors, textprops={"fontsize": 12})
    plt.title("2020年用户类型占比（会员vs非会员）", fontsize=14)
    plt.axis("equal")
    plt.savefig(os.path.join(save_dir, "2020_user_type_pie.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 用户类型占比饼图已保存")

    # 2. 会员vs非会员的骑行时长对比（更换为官方支持的调色板）
    df_filtered = df[df["duration_min"] <= 60]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="member_casual", y="duration_min", data=df_filtered, palette="Pastel1")  # 修改此处
    plt.title("2020年会员vs非会员骑行时长对比（≤60分钟）", fontsize=14)
    plt.xlabel("用户类型", fontsize=12)
    plt.ylabel("骑行时长（分钟）", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "2020_user_duration_box.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 用户骑行时长对比箱线图已保存")

def analyze_station_popularity(df, save_dir):
    # 1. 热门出发站点（更换为官方支持的调色板）
    top_start_stations = df["start_station_name"].value_counts().head(10).reset_index()
    top_start_stations.columns = ["station_name", "trip_count"]
    plt.figure(figsize=(12, 8))
    sns.barplot(y="station_name", x="trip_count", data=top_start_stations, palette="Greens")  # 修改此处
    plt.title("2020年热门出发站点 Top 10", fontsize=14)
    plt.xlabel("骑行出发次数", fontsize=12)
    plt.ylabel("站点名称", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, "2020_top_start_stations.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 热门出发站点图表已保存")

def main():
    featured_data_path = "../processed_data/2020_featured.csv"
    charts_save_dir = "../results/charts"
    os.makedirs(charts_save_dir, exist_ok=True)

    df = load_featured_data(featured_data_path)
    analyze_time_trends(df, charts_save_dir)
    analyze_user_behavior(df, charts_save_dir)
    analyze_station_popularity(df, charts_save_dir)

    print("\n🎉 所有EDA分析完成！图表已保存至：bike/results/charts")

if __name__ == "__main__":
    main()
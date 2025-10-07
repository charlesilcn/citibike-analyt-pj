import pandas as pd
import os

def load_clean_data(cleaned_path):
    """加载清洗后的数据，适配你的字段名"""
    # 禁用低内存模式，避免混合类型警告
    df = pd.read_csv(cleaned_path, low_memory=False)
    # 转换时间字段（你的字段是started_at和ended_at）
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    print(f"加载清洗后数据：共 {df.shape[0]} 行，{df.shape[1]} 列")
    return df

def add_time_features(df):
    """添加时间相关特征（基于started_at）"""
    # 提取小时、星期、月份、季度
    df["hour"] = df["started_at"].dt.hour  # 0-23小时
    df["weekday"] = df["started_at"].dt.weekday  # 0=周一，6=周日
    df["month"] = df["started_at"].dt.month  # 1-12月
    df["quarter"] = df["started_at"].dt.quarter  # 1-4季度

    # 标记工作日/周末（1=工作日，0=周末）
    df["is_weekday"] = df["weekday"].apply(lambda x: 1 if x < 5 else 0)

    # 标记高峰时段（早7-9点，晚17-19点）
    df["is_rush_hour"] = df["hour"].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

    # 标记时段（早/中/晚/凌晨）
    def get_time_period(hour):
        if 6 <= hour < 12:
            return "早上"
        elif 12 <= hour < 18:
            return "下午"
        elif 18 <= hour < 24:
            return "晚上"
        else:
            return "凌晨"
    df["time_period"] = df["hour"].apply(get_time_period)
    return df

def add_duration_features(df):
    """添加骑行时长特征（根据started_at和ended_at计算）"""
    # 计算骑行时长（秒）
    df["tripduration"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()
    # 转换为分钟（保留1位小数）
    df["duration_min"] = df["tripduration"] / 60
    df["duration_min"] = df["duration_min"].round(1)

    # 骑行时长分类
    def get_duration_category(minutes):
        if minutes <= 15:
            return "短时间（≤15分钟）"
        elif 15 < minutes <= 60:
            return "中时间（15-60分钟）"
        else:
            return "长时间（>60分钟）"
    df["duration_category"] = df["duration_min"].apply(get_duration_category)
    return df

def add_route_features(df):
    """添加骑行路线特征（基于你的站点字段）"""
    # 构造路线名称（出发→到达）
    df["route"] = df["start_station_name"] + " → " + df["end_station_name"]
    # 构造路线ID（出发ID-到达ID）
    df["route_id"] = df["start_station_id"].astype(str) + "-" + df["end_station_id"].astype(str)
    return df

def save_featured_data(df, save_path):
    """保存特征工程后的数据"""
    df.to_csv(save_path, index=False)
    print(f"特征工程完成！数据保存至：{save_path}")
    print(f"最终特征字段：{df.columns.tolist()}")

def main():
    # 路径配置（和你的文件夹结构匹配）
    cleaned_data_path = "../processed_data/2020_cleaned.csv"
    featured_data_path = "../processed_data/2020_featured.csv"

    # 执行特征工程流程
    df = load_clean_data(cleaned_data_path)
    df = add_time_features(df)
    df = add_duration_features(df)
    df = add_route_features(df)
    save_featured_data(df, featured_data_path)

if __name__ == "__main__":
    main()
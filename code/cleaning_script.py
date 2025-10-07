import pandas as pd
import os
import logging
from datetime import datetime

def load_data(raw_path):
    try:
        df = pd.read_csv(raw_path, low_memory=False)
        logging.info(f"成功加载数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
        return df
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        raise

def clean_missing_values(df):
    initial_missing = df.isnull().sum().sum()
    logging.info(f"清洗前总缺失值: {initial_missing}")

    station_cols = ['start station name', 'end station name']
    for col in station_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            missing = df[col].isnull().sum()
            logging.info(f"填充 {col} 缺失值 {missing} 个")

    coord_cols = [
        ('start station name', 'start station latitude', 'start station longitude'),
        ('end station name', 'end station latitude', 'end station longitude')
    ]
    for name_col, lat_col, lon_col in coord_cols:
        if all(col in df.columns for col in [name_col, lat_col, lon_col]):
            lat_mean = df.groupby(name_col)[lat_col].transform('mean')
            lon_mean = df.groupby(name_col)[lon_col].transform('mean')
            df[lat_col] = df[lat_col].fillna(lat_mean)
            df[lon_col] = df[lon_col].fillna(lon_mean)
            logging.info(f"用站点均值填充 {lat_col} 和 {lon_col} 缺失值")

    df = df.dropna()
    final_missing = df.isnull().sum().sum()
    logging.info(f"清洗后总缺失值: {final_missing}，删除 {initial_missing - final_missing} 行")
    return df

def clean_abnormal_values(df):
    initial_rows = df.shape[0]

    if 'tripduration' in df.columns:
        df = df[(df['tripduration'] >= 60) & (df['tripduration'] <= 86400)]
        logging.info(f"骑行时长过滤后剩余 {df.shape[0]} 行")

    id_cols = ['bikeid', 'start station id', 'end station id']
    for col in id_cols:
        if col in df.columns:
            df = df[df[col].notnull() & (df[col] > 0)]
            logging.info(f"过滤 {col} 异常值后剩余 {df.shape[0]} 行")

    lat_lon_ranges = {
        'start station latitude': (40.4, 40.9),
        'end station latitude': (40.4, 40.9),
        'start station longitude': (-74.3, -73.7),
        'end station longitude': (-74.3, -73.7)
    }
    for col, (min_val, max_val) in lat_lon_ranges.items():
        if col in df.columns:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
            logging.info(f"过滤 {col} 异常值后剩余 {df.shape[0]} 行")

    logging.info(f"异常值处理共删除 {initial_rows - df.shape[0]} 行")
    return df

def format_conversion(df):
    time_cols = ['starttime', 'stoptime']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df[df[col].notnull()]
            logging.info(f"转换 {col} 为datetime格式")

    if 'usertype' in df.columns:
        df['usertype'] = df['usertype'].str.strip().str.title()
        df['usertype'] = df['usertype'].replace({
            'Subscriber': 'Member',
            'Customer': 'Non-Member'
        })
        logging.info(f"用户类型标准化后分类: {df['usertype'].unique().tolist()}")

    return df

def data_consistency_check(df):
    if all(col in df.columns for col in ['starttime', 'stoptime']):
        invalid_time = df[df['starttime'] > df['stoptime']].shape[0]
        df = df[df['starttime'] <= df['stoptime']]
        logging.info(f"删除开始时间晚于结束时间的记录 {invalid_time} 行")

    if all(col in df.columns for col in ['start station id', 'end station id']):
        same_station = df[df['start station id'] == df['end station id']].shape[0]
        df = df[df['start station id'] != df['end station id']]
        logging.info(f"删除出发/到达站点相同的记录 {same_station} 行")

    return df

def main():
    # 修正路径：根据实际文件夹结构设置（脚本放在 bike/code 下）
    raw_data_path = "../processed_data/2020_combined.csv"  # 相对路径：code文件夹 -> 上级 -> processed_data
    cleaned_data_path = "../processed_data/2020_cleaned.csv"
    log_path = "../processed_data/cleaning_log.txt"  # 日志文件保存路径

    # 确保输出目录存在
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)

    # 配置日志（使用修正后的路径）
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        logging.info("===== 开始数据清洗 =====")
        df = load_data(raw_data_path)
        df = clean_missing_values(df)
        df = format_conversion(df)
        df = clean_abnormal_values(df)
        df = data_consistency_check(df)

        df.to_csv(cleaned_data_path, index=False)
        logging.info(f"清洗完成！保存至 {cleaned_data_path}，最终数据量: {df.shape[0]} 行")
        logging.info("===== 清洗结束 =====")
        print(f"清洗成功！数据已保存至 {cleaned_data_path}")

    except Exception as e:
        logging.error(f"清洗过程出错: {str(e)}")
        print(f"清洗失败: {str(e)}")

if __name__ == "__main__":
    main()
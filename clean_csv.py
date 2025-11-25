import pandas as pd
import numpy as np
import datetime
import os

# 设置文件路径
input_file = r'd:\code\502\Bike A\merged_data\merged_2025_data.csv'
output_file = r'd:\code\502\Bike A\merged_data\cleaned_2025_data.csv'
log_file = r'd:\code\502\Bike A\25cleaning_log.txt'

# 定义分块大小（根据内存情况调整）
chunksize = 1000000

def log_message(message):
    """记录清洗过程中的消息"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def clean_chunk(chunk, chunk_num):
    """清洗单个数据块"""
    original_rows = len(chunk)
    log_message(f"处理数据块 {chunk_num}：原始行数 {original_rows}")
    
    # 1. 删除重复行
    chunk = chunk.drop_duplicates()
    duplicate_count = original_rows - len(chunk)
    if duplicate_count > 0:
        log_message(f"删除了 {duplicate_count} 行重复数据")
    
    # 2. 转换日期时间列
    chunk['started_at'] = pd.to_datetime(chunk['started_at'], errors='coerce')
    chunk['ended_at'] = pd.to_datetime(chunk['ended_at'], errors='coerce')
    
    # 3. 删除日期时间无效的行
    datetime_invalid = chunk['started_at'].isnull() | chunk['ended_at'].isnull()
    if datetime_invalid.any():
        log_message(f"删除了 {datetime_invalid.sum()} 行日期时间无效的数据")
        chunk = chunk[~datetime_invalid]
    
    # 4. 计算骑行时长（分钟）并筛选合理的骑行时长（1分钟到24小时）
    chunk['ride_duration_minutes'] = (chunk['ended_at'] - chunk['started_at']).dt.total_seconds() / 60
    duration_invalid = (chunk['ride_duration_minutes'] < 1) | (chunk['ride_duration_minutes'] > 1440)
    if duration_invalid.any():
        log_message(f"删除了 {duration_invalid.sum()} 行骑行时长异常的数据")
        chunk = chunk[~duration_invalid]
    
    # 5. 处理经纬度异常值（纽约大致范围：纬度 40.4 到 41.0，经度 -74.3 到 -73.5）
    lat_invalid = (chunk['start_lat'] < 40.4) | (chunk['start_lat'] > 41.0) | \
                 (chunk['end_lat'] < 40.4) | (chunk['end_lat'] > 41.0)
    lng_invalid = (chunk['start_lng'] < -74.3) | (chunk['start_lng'] > -73.5) | \
                 (chunk['end_lng'] < -74.3) | (chunk['end_lng'] > -73.5)
    location_invalid = lat_invalid | lng_invalid
    if location_invalid.any():
        log_message(f"删除了 {location_invalid.sum()} 行地理位置异常的数据")
        chunk = chunk[~location_invalid]
    
    # 6. 处理缺失值
    # 删除关键信息缺失的行（如果起点或终点信息完全缺失）
    station_info_missing = (chunk['start_station_name'].isnull() & chunk['start_station_id'].isnull()) | \
                          (chunk['end_station_name'].isnull() & chunk['end_station_id'].isnull())
    if station_info_missing.any():
        log_message(f"删除了 {station_info_missing.sum()} 行站点信息完全缺失的数据")
        chunk = chunk[~station_info_missing]
    
    # 补充站点ID为浮点数的问题（转换为字符串类型）
    if 'start_station_id' in chunk.columns:
        chunk['start_station_id'] = chunk['start_station_id'].astype('object')
    if 'end_station_id' in chunk.columns:
        chunk['end_station_id'] = chunk['end_station_id'].astype('object')
    
    # 7. 验证用户类型
    valid_user_types = ['member', 'casual']
    user_type_invalid = ~chunk['member_casual'].isin(valid_user_types)
    if user_type_invalid.any():
        log_message(f"删除了 {user_type_invalid.sum()} 行用户类型无效的数据")
        chunk = chunk[~user_type_invalid]
    
    # 8. 验证车辆类型
    valid_rideable_types = ['classic_bike', 'electric_bike', 'docked_bike']
    rideable_type_invalid = ~chunk['rideable_type'].isin(valid_rideable_types)
    if rideable_type_invalid.any():
        log_message(f"删除了 {rideable_type_invalid.sum()} 行车辆类型无效的数据")
        chunk = chunk[~rideable_type_invalid]
    
    cleaned_rows = len(chunk)
    log_message(f"数据块 {chunk_num} 清洗完成：保留 {cleaned_rows} 行（清洗率：{((original_rows - cleaned_rows) / original_rows * 100):.2f}%）")
    
    return chunk

def main():
    """主函数：分块读取、清洗并保存数据"""
    # 初始化日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== 数据清洗日志 ===\n")
        f.write(f"开始时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    log_message(f"开始清洗数据：{input_file}")
    log_message(f"输出文件：{output_file}")
    log_message(f"分块大小：{chunksize} 行")
    
    # 记录总体统计信息
    total_original_rows = 0
    total_cleaned_rows = 0
    first_chunk = True
    
    try:
        # 分块处理数据
        for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize), 1):
            total_original_rows += len(chunk)
            
            # 清洗当前块
            cleaned_chunk = clean_chunk(chunk, i)
            total_cleaned_rows += len(cleaned_chunk)
            
            # 保存清洗后的数据（第一块保留表头，后续块不保留）
            cleaned_chunk.to_csv(output_file, mode='w' if first_chunk else 'a', 
                                index=False, header=first_chunk)
            first_chunk = False
            
            # 记录进度
            log_message(f"总体进度：处理了 {total_original_rows} 行，保留了 {total_cleaned_rows} 行")
        
        # 计算总体清洗统计
        cleaning_rate = ((total_original_rows - total_cleaned_rows) / total_original_rows * 100) if total_original_rows > 0 else 0
        log_message("\n=== 清洗完成 ===")
        log_message(f"原始总行数：{total_original_rows}")
        log_message(f"清洗后总行数：{total_cleaned_rows}")
        log_message(f"总体清洗率：{cleaning_rate:.2f}%")
        log_message(f"清洗后的文件大小：{os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
        
    except Exception as e:
        error_message = f"清洗过程中发生错误：{str(e)}"
        log_message(error_message)
        raise

if __name__ == "__main__":
    print("=== 数据清洗脚本开始执行 ===")
    start_time = datetime.datetime.now()
    try:
        main()
        end_time = datetime.datetime.now()
        print(f"\n=== 数据清洗完成 ===")
        print(f"总耗时：{(end_time - start_time).total_seconds():.2f} 秒")
        print(f"日志文件：{log_file}")
    except Exception as e:
        print(f"\n=== 数据清洗失败 ===")
        print(f"错误信息：{str(e)}")
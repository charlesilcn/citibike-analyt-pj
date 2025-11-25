import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from collections import defaultdict
import time

# Set font parameters for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def load_data(featured_path, chunk_size=1000000):
    """加载数据并提取核心特征（支持分段处理）"""
    print(f"===== Loading data from {featured_path} =====")
    start_time = time.time()
    
    # 初始化计数器和汇总数据
    total_records = 0
    station_counts = set()
    all_chunks = []
    
    # 分段读取数据
    for i, chunk in enumerate(pd.read_csv(
        featured_path,
        parse_dates=['started_at'],
        low_memory=False,
        usecols=['started_at', 'member_casual', 'start_station_name', 'start_lat', 'start_lng'],
        chunksize=chunk_size
    )):
        # 基础时间特征
        chunk['hour'] = chunk['started_at'].dt.hour  # 0-23小时
        chunk['is_weekday'] = chunk['started_at'].dt.weekday < 5  # True=工作日，False=周末
        
        # Simulate station area labels
        chunk['area_type'] = chunk['start_lat'].apply(lambda x: 'Residential' if x > 40.75 else 'Commercial')
        
        # 更新统计信息
        total_records += len(chunk)
        station_counts.update(chunk['start_station_name'].unique())
        all_chunks.append(chunk)
        
        # 显示进度
        elapsed_time = time.time() - start_time
        print(f"  Chunk {i+1} processed: {len(chunk):,} records ({total_records:,} total) - Time elapsed: {elapsed_time:.2f}s")
    
    # 合并所有分段
    df = pd.concat(all_chunks, ignore_index=True)
    
    elapsed_time = time.time() - start_time
    print(f"===== Data loading completed =====")
    print(f"  Total records: {len(df):,}")
    print(f"  Unique stations: {len(station_counts):,}")
    print(f"  Total loading time: {elapsed_time:.2f}s")
    
    return df

def identify_peak_periods(hour_data, is_weekday=True):
    """
    按需求强制标记高峰：
    - 工作日：固定7-9点为早高峰，17-19点为晚高峰（不依赖阈值，确保标记）
    - 周末：识别10-18点的集中时段为周末高峰（不区分早晚）
    """
    if is_weekday:
        # Weekday: Fixed morning/evening peak hours
        return [(7, 9), (17, 19)]  # Morning 7-9, Evening 17-19
    else:
        # Weekend: Identify high-traffic hours within 10-18 (no morning/evening distinction)
        if hour_data.empty:
            return []
        # Only identify within 10-18 hours
        candidate_hours = range(10, 19)  # 10-18 hours
        valid_hours = [h for h in candidate_hours if hour_data.get(h, 0) > hour_data.mean() * 1.2]
        if not valid_hours:
            return []
        # Merge consecutive periods
        valid_hours.sort()
        peak_periods = []
        start = valid_hours[0]
        for hour in valid_hours[1:]:
            if hour != start + 1:
                peak_periods.append((start, valid_hours[valid_hours.index(hour)-1]))
                start = hour
        peak_periods.append((start, valid_hours[-1]))
        return peak_periods

def format_peak_periods(periods, is_weekday=True):
    """Format peak period text (distinguish morning/evening for weekdays, unified for weekends)"""
    if is_weekday:
        # Weekday: Clearly mark morning/evening peaks
        labels = []
        for s, e in periods:
            if s < 12:
                labels.append(f"Morning Peak {s}:00-{e}:00")
            else:
                labels.append(f"Evening Peak {s}:00-{e}:00")
        return ", ".join(labels)
    else:
        # Weekend: Unified peak label
        return ", ".join([f"Peak {s}:00-{e}:00" for s, e in periods]) if periods else "No明显高峰"

def calculate_peak_intensity(df, peak_periods, is_weekday, time_type):
    """计算高峰强度：高峰时段骑行量占全天比例"""
    mask = (df['is_weekday'] == is_weekday)
    if time_type == 'user':
        user_peak_ratio = {'member': 0, 'casual': 0}
        for user_type in ['member', 'casual']:
            user_mask = mask & (df['member_casual'] == user_type)
            total = df[user_mask].shape[0]
            if total == 0 or not peak_periods:
                continue
            # Calculate rides during peak hours
            peak_mask = user_mask & df['hour'].between(
                min(p[0] for p in peak_periods), 
                max(p[1] for p in peak_periods)
            )
            user_peak_ratio[user_type] = round(df[peak_mask].shape[0] / total * 100, 1)
        return user_peak_ratio
    elif time_type == 'area':
        area_peak_ratio = {'Commercial': 0, 'Residential': 0}
        for area in ['Commercial', 'Residential']:
            area_mask = mask & (df['area_type'] == area)
            total = df[area_mask].shape[0]
            if total == 0 or not peak_periods:
                continue
            peak_mask = area_mask & df['hour'].between(
                min(p[0] for p in peak_periods), 
                max(p[1] for p in peak_periods)
            )
            area_peak_ratio[area] = round(df[peak_mask].shape[0] / total * 100, 1)
        return area_peak_ratio
    else:
        total = df[mask].shape[0]
        if total == 0 or not peak_periods:
            return 0
        peak_mask = mask & df['hour'].between(
            min(p[0] for p in peak_periods), 
            max(p[1] for p in peak_periods)
        )
        return round(df[peak_mask].shape[0] / total * 100, 1)

def analyze_overall_rush_hour(df, save_dir):
    """整体高峰识别（按需求强制标记工作日早/晚高峰）"""
    print("\n===== Analyzing overall rush hour patterns =====")
    start_time = time.time()
    
    # Group by weekday/weekend and hour
    hour_dist = df.groupby(['is_weekday', 'hour']).size().unstack(level=0)
    if hour_dist.shape[1] == 2:
        hour_dist.columns = ['Weekend', 'Weekday']
    elif hour_dist.shape[1] == 1:
        col_name = 'Weekday' if True in hour_dist.columns else 'Weekend'
        hour_dist.columns = [col_name]
    else:
        hour_dist.columns = []
    
    # Identify peaks (fixed for weekdays, auto-identified for weekends)
    workday_peaks = identify_peak_periods(hour_dist['Weekday'], is_weekday=True) if 'Weekday' in hour_dist.columns else []
    weekend_peaks = identify_peak_periods(hour_dist['Weekend'], is_weekday=False) if 'Weekend' in hour_dist.columns else []
    
    # Calculate peak intensity
    workday_intensity = calculate_peak_intensity(df, workday_peaks, True, 'overall')
    weekend_intensity = calculate_peak_intensity(df, weekend_peaks, False, 'overall')
    
    # Visualization
    x = np.arange(24)
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    width = 0.4
    if 'Weekday' in hour_dist.columns:
        ax1.bar(x - width/2, hour_dist['Weekday'].reindex(x, fill_value=0), 
                width=width, label='Weekday', color='#2196F3', alpha=0.7)
    if 'Weekend' in hour_dist.columns:
        ax1.bar(x + width/2, hour_dist['Weekend'].reindex(x, fill_value=0), 
                width=width, label='Weekend', color='#FF9800', alpha=0.7)
    
    # Percentage line chart
    if 'Weekday' in hour_dist.columns and hour_dist['Weekday'].sum() > 0:
        workday_ratio = (hour_dist['Weekday'] / hour_dist['Weekday'].sum() * 100).round(1).reindex(x, fill_value=0)
        ax2.plot(x, workday_ratio, 'b-', marker='o', label='Weekday %')
    if 'Weekend' in hour_dist.columns and hour_dist['Weekend'].sum() > 0:
        weekend_ratio = (hour_dist['Weekend'] / hour_dist['Weekend'].sum() * 100).round(1).reindex(x, fill_value=0)
        ax2.plot(x, weekend_ratio, 'r-', marker='o', label='Weekend %')
    
    # Mark peak periods (morning/evening for weekdays, unified for weekends)
    for peak in workday_peaks:
        label = 'Weekday Morning Peak' if peak[0] < 12 else 'Weekday Evening Peak'
        ax1.axvspan(peak[0], peak[1], color='#2196F3', alpha=0.2, label=label)
    for peak in weekend_peaks:
        ax1.axvspan(peak[0], peak[1], color='#FF9800', alpha=0.2, label='Weekend Peak' if peak == weekend_peaks[0] else "")
    
    ax1.set_title(f'Hourly Ride Distribution (Weekday Peak: {workday_intensity}%, Weekend: {weekend_intensity}%)', fontsize=14)
    ax1.set_xlabel('Hour (0-23)')
    ax1.set_ylabel('Number of Rides')
    ax2.set_ylabel('Percentage (%)')
    ax1.set_xticks(x)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_rush_hour.png'), dpi=300)
    plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"  Overall rush hour analysis completed in {elapsed_time:.2f}s")
    
    return {
        'workday': {'periods': workday_peaks, 'intensity': workday_intensity},
        'weekend': {'periods': weekend_peaks, 'intensity': weekend_intensity}
    }

def analyze_user_type_rush_hour(df, save_dir, overall_peaks):
    """Analyze rush hours by user type"""
    print("\n===== Analyzing rush hour patterns by user type =====")
    start_time = time.time()
    
    user_types = ['member', 'casual']
    user_peak_results = defaultdict(dict)
    
    # Group by user type, weekday/weekend, and hour
    user_hour_dist = df.groupby(['member_casual', 'is_weekday', 'hour']).size().unstack(level=1)
    if user_hour_dist.shape[1] == 2:
        user_hour_dist.columns = ['Weekend', 'Weekday']
    elif user_hour_dist.shape[1] == 1:
        user_hour_dist.columns = ['Weekday' if 1 in user_hour_dist.columns else 'Weekend']
    else:
        user_hour_dist.columns = []
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    x = np.arange(24)
    for i, user_type in enumerate(user_types):
        ax = axes[i]
        ax2 = ax.twinx()
        
        width = 0.4
        if user_type in user_hour_dist.index:
            if 'Weekday' in user_hour_dist.columns:
                workday_data = user_hour_dist.loc[user_type, 'Weekday'].reindex(x, fill_value=0)
                ax.bar(x - width/2, workday_data, width=width, label='Weekday', color='#2196F3', alpha=0.7)
            if 'Weekend' in user_hour_dist.columns:
                weekend_data = user_hour_dist.loc[user_type, 'Weekend'].reindex(x, fill_value=0)
                ax.bar(x + width/2, weekend_data, width=width, label='Weekend', color='#FF9800', alpha=0.7)
        
        # Percentage line
        if user_type in user_hour_dist.index:
            if 'Weekday' in user_hour_dist.columns and user_hour_dist.loc[user_type, 'Weekday'].sum() > 0:
                workday_ratio = (user_hour_dist.loc[user_type, 'Weekday'] / user_hour_dist.loc[user_type, 'Weekday'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, workday_ratio, 'b-', marker='o')
            if 'Weekend' in user_hour_dist.columns and user_hour_dist.loc[user_type, 'Weekend'].sum() > 0:
                weekend_ratio = (user_hour_dist.loc[user_type, 'Weekend'] / user_hour_dist.loc[user_type, 'Weekend'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, weekend_ratio, 'r-', marker='o')
        
        # Mark peak periods
        for peak in overall_peaks['workday']['periods']:
            label = 'Weekday Morning Peak' if peak[0] < 12 else 'Weekday Evening Peak'
            ax.axvspan(peak[0], peak[1], color='#2196F3', alpha=0.2, label=label if peak == overall_peaks['workday']['periods'][0] else "")
        for peak in overall_peaks['weekend']['periods']:
            ax.axvspan(peak[0], peak[1], color='#FF9800', alpha=0.2, label='Weekend Peak' if peak == overall_peaks['weekend']['periods'][0] else "")
        
        # Peak intensity
        workday_intensity = calculate_peak_intensity(df, overall_peaks['workday']['periods'], True, 'user')
        weekend_intensity = calculate_peak_intensity(df, overall_peaks['weekend']['periods'], False, 'user')
        
        ax.set_title(f'{user_type} Users (Weekday Peak: {workday_intensity[user_type]}%, Weekend: {weekend_intensity[user_type]}%)', fontsize=14)
        ax.set_xlabel('Hour (0-23)')
        ax.set_ylabel('Number of Rides')
        ax2.set_ylabel('Percentage (%)')
        ax.set_xticks(x)
        ax.legend(loc='upper left')
        ax2.legend(['Weekday %', 'Weekend %'], loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        user_peak_results[user_type] = {
            'workday_intensity': workday_intensity[user_type],
            'weekend_intensity': weekend_intensity[user_type]
        }
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'user_type_rush_hour.png'), dpi=300)
    plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"  User type rush hour analysis completed in {elapsed_time:.2f}s")
    
    return user_peak_results

def analyze_area_rush_hour(df, save_dir, overall_peaks):
    """Analyze rush hours by area type"""
    print("\n===== Analyzing rush hour patterns by area type =====")
    start_time = time.time()
    
    areas = ['Commercial', 'Residential']
    area_peak_results = defaultdict(dict)
    
    # Group by area type, weekday/weekend, and hour
    area_hour_dist = df.groupby(['area_type', 'is_weekday', 'hour']).size().unstack(level=1)
    if area_hour_dist.shape[1] == 2:
        area_hour_dist.columns = ['Weekend', 'Weekday']
    elif area_hour_dist.shape[1] == 1:
        area_hour_dist.columns = ['Weekday' if 1 in area_hour_dist.columns else 'Weekend']
    else:
        area_hour_dist.columns = []
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    x = np.arange(24)
    for i, area in enumerate(areas):
        ax = axes[i]
        ax2 = ax.twinx()
        
        width = 0.4
        if area in area_hour_dist.index:
            if 'Weekday' in area_hour_dist.columns:
                workday_data = area_hour_dist.loc[area, 'Weekday'].reindex(x, fill_value=0)
                ax.bar(x - width/2, workday_data, width=width, label='Weekday', color='#2196F3', alpha=0.7)
            if 'Weekend' in area_hour_dist.columns:
                weekend_data = area_hour_dist.loc[area, 'Weekend'].reindex(x, fill_value=0)
                ax.bar(x + width/2, weekend_data, width=width, label='Weekend', color='#FF9800', alpha=0.7)
        
        # Percentage line
        if area in area_hour_dist.index:
            if 'Weekday' in area_hour_dist.columns and area_hour_dist.loc[area, 'Weekday'].sum() > 0:
                workday_ratio = (area_hour_dist.loc[area, 'Weekday'] / area_hour_dist.loc[area, 'Weekday'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, workday_ratio, 'b-', marker='o')
            if 'Weekend' in area_hour_dist.columns and area_hour_dist.loc[area, 'Weekend'].sum() > 0:
                weekend_ratio = (area_hour_dist.loc[area, 'Weekend'] / area_hour_dist.loc[area, 'Weekend'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, weekend_ratio, 'r-', marker='o')
        
        # Mark peak periods
        for peak in overall_peaks['workday']['periods']:
            label = 'Weekday Morning Peak' if peak[0] < 12 else 'Weekday Evening Peak'
            ax.axvspan(peak[0], peak[1], color='#2196F3', alpha=0.2, label=label if peak == overall_peaks['workday']['periods'][0] else "")
        for peak in overall_peaks['weekend']['periods']:
            ax.axvspan(peak[0], peak[1], color='#FF9800', alpha=0.2, label='Weekend Peak' if peak == overall_peaks['weekend']['periods'][0] else "")
        
        # Peak intensity
        workday_intensity = calculate_peak_intensity(df, overall_peaks['workday']['periods'], True, 'area')
        weekend_intensity = calculate_peak_intensity(df, overall_peaks['weekend']['periods'], False, 'area')
        
        ax.set_title(f'{area} Area (Weekday Peak: {workday_intensity[area]}%, Weekend: {weekend_intensity[area]}%)', fontsize=14)
        ax.set_xlabel('Hour (0-23)')
        ax.set_ylabel('Number of Rides')
        ax2.set_ylabel('Percentage (%)')
        ax.set_xticks(x)
        ax.legend(loc='upper left')
        ax2.legend(['Weekday %', 'Weekend %'], loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        area_peak_results[area] = {
            'workday_intensity': workday_intensity[area],
            'weekend_intensity': weekend_intensity[area]
        }
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'area_rush_hour.png'), dpi=300)
    plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"  Area type rush hour analysis completed in {elapsed_time:.2f}s")
    
    return area_peak_results

def save_results(overall_results, user_results, area_results, save_dir, df):
    """Save results (clearly marking morning/evening peaks for weekdays)"""
    print("\n===== Saving analysis results =====")
    start_time = time.time()
    with open(os.path.join(save_dir, 'rush_hour_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Rush Hour Identification Results ===\n\n")
        f.write("1. Overall Peak Hours\n")
        f.write(f"   Weekdays: {format_peak_periods(overall_results['workday']['periods'], is_weekday=True)}\n")
        f.write(f"   Weekends: {format_peak_periods(overall_results['weekend']['periods'], is_weekday=False)} (Ratio: {overall_results['weekend']['intensity']}%)\n\n")
        
        f.write("2. Peak Intensity by User Type (percentage of daily rides)\n")
        f.write(f"   Member-Weekday: {user_results['member']['workday_intensity']}%; Member-Weekend: {user_results['member']['weekend_intensity']}%\n")
        f.write(f"   Casual-Weekday: {user_results['casual']['workday_intensity']}%; Casual-Weekend: {user_results['casual']['weekend_intensity']}%\n\n")
        
        f.write("3. Peak Intensity by Area Type (percentage of daily rides)\n")
        f.write(f"   Commercial-Weekday: {area_results['Commercial']['workday_intensity']}%; Commercial-Weekend: {area_results['Commercial']['weekend_intensity']}%\n")
        f.write(f"   Residential-Weekday: {area_results['Residential']['workday_intensity']}%; Residential-Weekend: {area_results['Residential']['weekend_intensity']}%\n")
    
    # Save CSV data
    overall_hour_dist = df.groupby(['is_weekday', 'hour']).size().unstack(level=0)
    if overall_hour_dist.shape[1] == 2:
        overall_hour_dist.columns = ['Weekend', 'Weekday']
    overall_hour_dist.to_csv(os.path.join(save_dir, 'overall_hourly_data.csv'), encoding='utf-8-sig')
    
    user_hour_dist = df.groupby(['member_casual', 'is_weekday', 'hour']).size().unstack(level=1)
    if user_hour_dist.shape[1] == 2:
        user_hour_dist.columns = ['Weekend', 'Weekday']
    user_hour_dist.to_csv(os.path.join(save_dir, 'user_type_hourly_data.csv'), encoding='utf-8-sig')
    
    area_hour_dist = df.groupby(['area_type', 'is_weekday', 'hour']).size().unstack(level=1)
    if area_hour_dist.shape[1] == 2:
        area_hour_dist.columns = ['Weekend', 'Weekday']
    area_hour_dist.to_csv(os.path.join(save_dir, 'area_hourly_data.csv'), encoding='utf-8-sig')
    
    elapsed_time = time.time() - start_time
    print(f"  Results saving completed in {elapsed_time:.2f}s")
    print(f"✅ All results saved to: {save_dir}")

def main():
    # Path configuration (adjust to your project structure)
    featured_path = "D:/code/502/Bike A/merged_data/cleaned_2023_data.csv"  # Featured data file
    charts_dir = "D:/code/502/Bike A/result/charts/2023_rush_hour_enhanced"      # Charts save directory
    results_dir = "D:/code/502/Bike A/result/charts/2023_rush_hour_enhanced"      # Results data directory
    
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("===== Enhanced Rush Hour Identification Started =====")
    df = load_data(featured_path)
    overall_results = analyze_overall_rush_hour(df, charts_dir)
    user_results = analyze_user_type_rush_hour(df, charts_dir, overall_results)
    area_results = analyze_area_rush_hour(df, charts_dir, overall_results)
    save_results(overall_results, user_results, area_results, results_dir, df)
    print("===== Enhanced Rush Hour Identification Completed! =====")

if __name__ == "__main__":
    main()
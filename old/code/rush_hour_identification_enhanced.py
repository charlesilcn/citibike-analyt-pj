import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from collections import defaultdict

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(featured_path):
    """加载数据并提取核心特征"""
    df = pd.read_csv(
        featured_path,
        parse_dates=['started_at'],
        low_memory=False,
        usecols=['started_at', 'member_casual', 'start_station_name', 'start_lat', 'start_lng']
    )
    
    # 基础时间特征
    df['hour'] = df['started_at'].dt.hour  # 0-23小时
    df['is_weekday'] = df['started_at'].dt.weekday < 5  # True=工作日，False=周末
    
    # 模拟站点区域标签
    df['area_type'] = df['start_lat'].apply(lambda x: '住宅区' if x > 40.75 else '商业区')
    
    print(f"数据加载完成：共 {len(df):,} 条记录，包含 {df['start_station_name'].nunique()} 个站点")
    return df

def identify_peak_periods(hour_data, is_weekday=True):
    """
    按需求强制标记高峰：
    - 工作日：固定7-9点为早高峰，17-19点为晚高峰（不依赖阈值，确保标记）
    - 周末：识别10-18点的集中时段为周末高峰（不区分早晚）
    """
    if is_weekday:
        # 工作日：强制固定早/晚高峰时段
        return [(7, 9), (17, 19)]  # 早7-9，晚17-19
    else:
        # 周末：识别10-18点内的高流量时段（不区分早晚）
        if hour_data.empty:
            return []
        # 仅在10-18点范围内识别
        candidate_hours = range(10, 19)  # 10-18点
        valid_hours = [h for h in candidate_hours if hour_data.get(h, 0) > hour_data.mean() * 1.2]
        if not valid_hours:
            return []
        # 合并连续时段
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
    """格式化高峰时段文本（工作日区分早晚，周末统一为高峰）"""
    if is_weekday:
        # 工作日明确标注早/晚高峰
        labels = []
        for s, e in periods:
            if s < 12:
                labels.append(f"早高峰 {s}:00-{e}:00")
            else:
                labels.append(f"晚高峰 {s}:00-{e}:00")
        return ", ".join(labels)
    else:
        # 周末统一标注为高峰
        return ", ".join([f"高峰 {s}:00-{e}:00" for s, e in periods]) if periods else "无明显高峰"

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
            # 计算高峰时段内的骑行量
            peak_mask = user_mask & df['hour'].between(
                min(p[0] for p in peak_periods), 
                max(p[1] for p in peak_periods)
            )
            user_peak_ratio[user_type] = round(df[peak_mask].shape[0] / total * 100, 1)
        return user_peak_ratio
    elif time_type == 'area':
        area_peak_ratio = {'商业区': 0, '住宅区': 0}
        for area in ['商业区', '住宅区']:
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
    # 按工作日/周末和小时统计
    hour_dist = df.groupby(['is_weekday', 'hour']).size().unstack(level=0)
    if hour_dist.shape[1] == 2:
        hour_dist.columns = ['周末', '工作日']
    elif hour_dist.shape[1] == 1:
        col_name = '工作日' if True in hour_dist.columns else '周末'
        hour_dist.columns = [col_name]
    else:
        hour_dist.columns = []
    
    # 识别高峰（工作日强制标记，周末自动识别）
    workday_peaks = identify_peak_periods(hour_dist['工作日'], is_weekday=True) if '工作日' in hour_dist.columns else []
    weekend_peaks = identify_peak_periods(hour_dist['周末'], is_weekday=False) if '周末' in hour_dist.columns else []
    
    # 计算高峰强度
    workday_intensity = calculate_peak_intensity(df, workday_peaks, True, 'overall')
    weekend_intensity = calculate_peak_intensity(df, weekend_peaks, False, 'overall')
    
    # 可视化
    x = np.arange(24)
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    width = 0.4
    if '工作日' in hour_dist.columns:
        ax1.bar(x - width/2, hour_dist['工作日'].reindex(x, fill_value=0), 
                width=width, label='工作日', color='#2196F3', alpha=0.7)
    if '周末' in hour_dist.columns:
        ax1.bar(x + width/2, hour_dist['周末'].reindex(x, fill_value=0), 
                width=width, label='周末', color='#FF9800', alpha=0.7)
    
    # 占比折线图
    if '工作日' in hour_dist.columns and hour_dist['工作日'].sum() > 0:
        workday_ratio = (hour_dist['工作日'] / hour_dist['工作日'].sum() * 100).round(1).reindex(x, fill_value=0)
        ax2.plot(x, workday_ratio, 'b-', marker='o', label='工作日占比')
    if '周末' in hour_dist.columns and hour_dist['周末'].sum() > 0:
        weekend_ratio = (hour_dist['周末'] / hour_dist['周末'].sum() * 100).round(1).reindex(x, fill_value=0)
        ax2.plot(x, weekend_ratio, 'r-', marker='o', label='周末占比')
    
    # 标注高峰时段（工作日明确早/晚，周末统一为高峰）
    for peak in workday_peaks:
        label = '工作日早高峰' if peak[0] < 12 else '工作日晚高峰'
        ax1.axvspan(peak[0], peak[1], color='#2196F3', alpha=0.2, label=label)
    for peak in weekend_peaks:
        ax1.axvspan(peak[0], peak[1], color='#FF9800', alpha=0.2, label='周末高峰' if peak == weekend_peaks[0] else "")
    
    ax1.set_title(f'整体小时骑行量分布（工作日高峰占比{workday_intensity}%，周末{weekend_intensity}%）', fontsize=14)
    ax1.set_xlabel('小时（0-23）')
    ax1.set_ylabel('骑行次数')
    ax2.set_ylabel('占比（%）')
    ax1.set_xticks(x)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_rush_hour.png'), dpi=300)
    plt.close()
    
    return {
        'workday': {'periods': workday_peaks, 'intensity': workday_intensity},
        'weekend': {'periods': weekend_peaks, 'intensity': weekend_intensity}
    }

def analyze_user_type_rush_hour(df, save_dir, overall_peaks):
    """按用户类型拆分高峰"""
    user_types = ['member', 'casual']
    user_peak_results = defaultdict(dict)
    
    # 分组统计
    user_hour_dist = df.groupby(['member_casual', 'is_weekday', 'hour']).size().unstack(level=1)
    if user_hour_dist.shape[1] == 2:
        user_hour_dist.columns = ['周末', '工作日']
    elif user_hour_dist.shape[1] == 1:
        user_hour_dist.columns = ['工作日' if 1 in user_hour_dist.columns else '周末']
    else:
        user_hour_dist.columns = []
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    x = np.arange(24)
    for i, user_type in enumerate(user_types):
        ax = axes[i]
        ax2 = ax.twinx()
        
        width = 0.4
        if user_type in user_hour_dist.index:
            if '工作日' in user_hour_dist.columns:
                workday_data = user_hour_dist.loc[user_type, '工作日'].reindex(x, fill_value=0)
                ax.bar(x - width/2, workday_data, width=width, label='工作日', color='#2196F3', alpha=0.7)
            if '周末' in user_hour_dist.columns:
                weekend_data = user_hour_dist.loc[user_type, '周末'].reindex(x, fill_value=0)
                ax.bar(x + width/2, weekend_data, width=width, label='周末', color='#FF9800', alpha=0.7)
        
        # 占比折线
        if user_type in user_hour_dist.index:
            if '工作日' in user_hour_dist.columns and user_hour_dist.loc[user_type, '工作日'].sum() > 0:
                workday_ratio = (user_hour_dist.loc[user_type, '工作日'] / user_hour_dist.loc[user_type, '工作日'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, workday_ratio, 'b-', marker='o')
            if '周末' in user_hour_dist.columns and user_hour_dist.loc[user_type, '周末'].sum() > 0:
                weekend_ratio = (user_hour_dist.loc[user_type, '周末'] / user_hour_dist.loc[user_type, '周末'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, weekend_ratio, 'r-', marker='o')
        
        # 标注高峰时段
        for peak in overall_peaks['workday']['periods']:
            label = '工作日早高峰' if peak[0] < 12 else '工作日晚高峰'
            ax.axvspan(peak[0], peak[1], color='#2196F3', alpha=0.2, label=label if peak == overall_peaks['workday']['periods'][0] else "")
        for peak in overall_peaks['weekend']['periods']:
            ax.axvspan(peak[0], peak[1], color='#FF9800', alpha=0.2, label='周末高峰' if peak == overall_peaks['weekend']['periods'][0] else "")
        
        # 高峰强度
        workday_intensity = calculate_peak_intensity(df, overall_peaks['workday']['periods'], True, 'user')
        weekend_intensity = calculate_peak_intensity(df, overall_peaks['weekend']['periods'], False, 'user')
        
        ax.set_title(f'{user_type}用户（工作日高峰占比{workday_intensity[user_type]}%，周末{weekend_intensity[user_type]}%）', fontsize=14)
        ax.set_xlabel('小时（0-23）')
        ax.set_ylabel('骑行次数')
        ax2.set_ylabel('占比（%）')
        ax.set_xticks(x)
        ax.legend(loc='upper left')
        ax2.legend(['工作日占比', '周末占比'], loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        user_peak_results[user_type] = {
            'workday_intensity': workday_intensity[user_type],
            'weekend_intensity': weekend_intensity[user_type]
        }
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'user_type_rush_hour.png'), dpi=300)
    plt.close()
    return user_peak_results

def analyze_area_rush_hour(df, save_dir, overall_peaks):
    """按区域拆分高峰"""
    areas = ['商业区', '住宅区']
    area_peak_results = defaultdict(dict)
    
    # 分组统计
    area_hour_dist = df.groupby(['area_type', 'is_weekday', 'hour']).size().unstack(level=1)
    if area_hour_dist.shape[1] == 2:
        area_hour_dist.columns = ['周末', '工作日']
    elif area_hour_dist.shape[1] == 1:
        area_hour_dist.columns = ['工作日' if 1 in area_hour_dist.columns else '周末']
    else:
        area_hour_dist.columns = []
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    x = np.arange(24)
    for i, area in enumerate(areas):
        ax = axes[i]
        ax2 = ax.twinx()
        
        width = 0.4
        if area in area_hour_dist.index:
            if '工作日' in area_hour_dist.columns:
                workday_data = area_hour_dist.loc[area, '工作日'].reindex(x, fill_value=0)
                ax.bar(x - width/2, workday_data, width=width, label='工作日', color='#2196F3', alpha=0.7)
            if '周末' in area_hour_dist.columns:
                weekend_data = area_hour_dist.loc[area, '周末'].reindex(x, fill_value=0)
                ax.bar(x + width/2, weekend_data, width=width, label='周末', color='#FF9800', alpha=0.7)
        
        # 占比折线
        if area in area_hour_dist.index:
            if '工作日' in area_hour_dist.columns and area_hour_dist.loc[area, '工作日'].sum() > 0:
                workday_ratio = (area_hour_dist.loc[area, '工作日'] / area_hour_dist.loc[area, '工作日'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, workday_ratio, 'b-', marker='o')
            if '周末' in area_hour_dist.columns and area_hour_dist.loc[area, '周末'].sum() > 0:
                weekend_ratio = (area_hour_dist.loc[area, '周末'] / area_hour_dist.loc[area, '周末'].sum() * 100).round(1).reindex(x, fill_value=0)
                ax2.plot(x, weekend_ratio, 'r-', marker='o')
        
        # 标注高峰时段
        for peak in overall_peaks['workday']['periods']:
            label = '工作日早高峰' if peak[0] < 12 else '工作日晚高峰'
            ax.axvspan(peak[0], peak[1], color='#2196F3', alpha=0.2, label=label if peak == overall_peaks['workday']['periods'][0] else "")
        for peak in overall_peaks['weekend']['periods']:
            ax.axvspan(peak[0], peak[1], color='#FF9800', alpha=0.2, label='周末高峰' if peak == overall_peaks['weekend']['periods'][0] else "")
        
        # 高峰强度
        workday_intensity = calculate_peak_intensity(df, overall_peaks['workday']['periods'], True, 'area')
        weekend_intensity = calculate_peak_intensity(df, overall_peaks['weekend']['periods'], False, 'area')
        
        ax.set_title(f'{area}（工作日高峰占比{workday_intensity[area]}%，周末{weekend_intensity[area]}%）', fontsize=14)
        ax.set_xlabel('小时（0-23）')
        ax.set_ylabel('骑行次数')
        ax2.set_ylabel('占比（%）')
        ax.set_xticks(x)
        ax.legend(loc='upper left')
        ax2.legend(['工作日占比', '周末占比'], loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        area_peak_results[area] = {
            'workday_intensity': workday_intensity[area],
            'weekend_intensity': weekend_intensity[area]
        }
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'area_rush_hour.png'), dpi=300)
    plt.close()
    return area_peak_results

def save_results(overall_results, user_results, area_results, save_dir, df):
    """保存结果（工作日明确早/晚高峰）"""
    with open(os.path.join(save_dir, 'rush_hour_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 早晚高峰时间段识别结果汇总 ===\n\n")
        f.write("1. 整体高峰时段\n")
        f.write(f"   工作日：{format_peak_periods(overall_results['workday']['periods'], is_weekday=True)}\n")
        f.write(f"   周末：{format_peak_periods(overall_results['weekend']['periods'], is_weekday=False)}（占比{overall_results['weekend']['intensity']}%）\n\n")
        
        f.write("2. 用户类型高峰强度（占该类用户全天骑行量比例）\n")
        f.write(f"   会员-工作日：{user_results['member']['workday_intensity']}%；会员-周末：{user_results['member']['weekend_intensity']}%\n")
        f.write(f"   非会员-工作日：{user_results['casual']['workday_intensity']}%；非会员-周末：{user_results['casual']['weekend_intensity']}%\n\n")
        
        f.write("3. 区域高峰强度（占该区域全天骑行量比例）\n")
        f.write(f"   商业区-工作日：{area_results['商业区']['workday_intensity']}%；商业区-周末：{area_results['商业区']['weekend_intensity']}%\n")
        f.write(f"   住宅区-工作日：{area_results['住宅区']['workday_intensity']}%；住宅区-周末：{area_results['住宅区']['weekend_intensity']}%\n")
    
    # 保存CSV数据
    overall_hour_dist = df.groupby(['is_weekday', 'hour']).size().unstack(level=0)
    if overall_hour_dist.shape[1] == 2:
        overall_hour_dist.columns = ['周末', '工作日']
    overall_hour_dist.to_csv(os.path.join(save_dir, 'overall_hourly_data.csv'), encoding='utf-8-sig')
    
    user_hour_dist = df.groupby(['member_casual', 'is_weekday', 'hour']).size().unstack(level=1)
    if user_hour_dist.shape[1] == 2:
        user_hour_dist.columns = ['周末', '工作日']
    user_hour_dist.to_csv(os.path.join(save_dir, 'user_type_hourly_data.csv'), encoding='utf-8-sig')
    
    area_hour_dist = df.groupby(['area_type', 'is_weekday', 'hour']).size().unstack(level=1)
    if area_hour_dist.shape[1] == 2:
        area_hour_dist.columns = ['周末', '工作日']
    area_hour_dist.to_csv(os.path.join(save_dir, 'area_hourly_data.csv'), encoding='utf-8-sig')
    
    print(f"✅ 所有结果已保存至：{save_dir}")

def main():
    featured_path = "../processed_data/2024_featured.csv"
    charts_dir = "../results/charts/2024 charts/rush_hour_enhanced"
    results_dir = "../results/2024 charts/rush_hour_enhanced"
    
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("===== 开始增强版早晚高峰时间段识别 =====")
    df = load_data(featured_path)
    overall_results = analyze_overall_rush_hour(df, charts_dir)
    user_results = analyze_user_type_rush_hour(df, charts_dir, overall_results)
    area_results = analyze_area_rush_hour(df, charts_dir, overall_results)
    save_results(overall_results, user_results, area_results, results_dir, df)
    print("===== 增强版高峰时段识别完成！ =====")

if __name__ == "__main__":
    main()
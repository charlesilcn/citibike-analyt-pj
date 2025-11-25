import pandas as pd
import numpy as np

# 设置文件路径
file_path = r'd:\code\502\Bike A\merged_data\merged_2023_data.csv'

def analyze_csv_structure():
    print("开始分析CSV文件结构...")
    
    # 读取前100行数据进行初步分析
    print("读取前100行数据...")
    df_sample = pd.read_csv(file_path, nrows=100)
    
    print("\n1. 文件基本信息：")
    print(f"列数: {df_sample.shape[1]}")
    print(f"示例行数: {df_sample.shape[0]}")
    
    print("\n2. 列名和数据类型：")
    print(df_sample.dtypes)
    
    print("\n3. 前5行数据：")
    print(df_sample.head())
    
    print("\n4. 数据统计信息：")
    print(df_sample.describe(include='all'))
    
    # 计算缺失值
    print("\n5. 每列缺失值统计：")
    missing_values = df_sample.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # 检查是否有混合类型的列
    print("\n6. 检查混合类型的列：")
    mixed_type_columns = []
    for col in df_sample.columns:
        if df_sample[col].apply(type).nunique() > 1:
            mixed_type_columns.append(col)
    print(f"混合类型的列: {mixed_type_columns}")
    
    # 尝试估算总行数（不加载整个文件）
    print("\n7. 估算总行数...")
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    print(f"估算总行数: {line_count - 1}（减去表头）")
    
    print("\n分析完成！")

if __name__ == "__main__":
    try:
        analyze_csv_structure()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
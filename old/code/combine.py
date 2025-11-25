import pandas as pd
import os

# 原始数据路径,合并年份文件位置*******************************************************************
raw_data_path = r"D:\code\MYSQL\Programes\bike\raw_data\2024"
#***********************************************************************************************
# 合并后的数据保存路径
processed_data_path = r"D:\code\MYSQL\Programes\bike\processed_data"
os.makedirs(processed_data_path, exist_ok=True)  

all_dfs = []
# 遍历目录下的所有CSV文件
for filename in os.listdir(raw_data_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(raw_data_path, filename)
        print(f"正在读取: {filename}")
        # 读取CSV文件（注意：若文件过大，可加low_memory=False避免类型推断警告）
        df = pd.read_csv(file_path, low_memory=False)
        all_dfs.append(df)

# 合并所有DataFrame
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # 保存合并后的数据
    #**********************************************************************************************
    combined_csv_path = os.path.join(processed_data_path, "2024_combined.csv") #保存文件的名称
    #**********************************************************************************************
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"合并完成！共 {len(combined_df)} 行数据，保存至: {combined_csv_path}")
else:
    print("未找到CSV文件")
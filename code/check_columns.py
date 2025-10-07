import pandas as pd

# 读取清洗后的CSV文件（路径和你之前的脚本一致）
df = pd.read_csv("../processed_data/2020_cleaned.csv")

# 打印所有字段名
print("你的数据里的所有字段名：")
for column in df.columns:
    print(column)
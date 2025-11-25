import os
import pandas as pd
import glob

# 设置源目录和输出文件路径
source_dir = r'd:\code\502\Bike A\Rdata\24'
output_file = r'd:\code\502\Bike A\merged_2024_data.csv'

def merge_csv_files():
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    
    if not csv_files:
        print(f"在目录 {source_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件需要合并")
    
    # 创建一个空的DataFrame来存储合并的数据
    merged_data = pd.DataFrame()
    
    # 记录是否已经写入表头
    header_written = False
    
    # 遍历所有CSV文件
    for i, file in enumerate(csv_files, 1):
        print(f"正在处理文件 {i}/{len(csv_files)}: {os.path.basename(file)}")
        
        try:
            # 读取当前CSV文件
            df = pd.read_csv(file)
            
            # 如果是第一个文件，保留表头；否则不写表头
            if not header_written:
                merged_data = pd.concat([merged_data, df], ignore_index=True)
                header_written = True
            else:
                # 只合并数据，不合并表头
                merged_data = pd.concat([merged_data, df], ignore_index=True)
                
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 保存合并后的数据
    if not merged_data.empty:
        merged_data.to_csv(output_file, index=False)
        print(f"合并完成！结果已保存至 {output_file}")
        print(f"合并后的文件包含 {len(merged_data)} 行数据")
    else:
        print("合并失败，没有成功读取任何数据")

if __name__ == "__main__":
    print("开始合并CSV文件...")
    merge_csv_files()
    print("合并过程已完成")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV文件转换为Parquet格式工具

此脚本用于将CSV文件转换为Parquet格式，支持单个文件转换和批量转换。
Parquet格式具有更好的压缩率和查询性能，适合大数据分析。
"""

import os
import sys
import argparse
import pandas as pd
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")


def convert_csv_to_parquet(input_file, output_file, compression='snappy'):
    """
    将单个CSV文件转换为Parquet格式
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出Parquet文件路径
        compression (str): 压缩格式，可选 'snappy', 'gzip', 'brotli', 'lz4', 默认为'snappy'
    
    Returns:
        bool: 转换是否成功
    """
    try:
        # 读取CSV文件，设置low_memory=False以避免混合类型警告
        logger.info(f"开始读取CSV文件: {input_file}")
        df = pd.read_csv(input_file, low_memory=False)
        logger.info(f"成功读取CSV文件，共 {len(df)} 行数据")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        ensure_directory_exists(output_dir)
        
        # 处理可能的混合类型列
        logger.info("处理混合数据类型...")
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # 尝试将列转换为分类类型以优化存储
                    df[col] = df[col].astype('category')
                    logger.info(f"  列 {col} 转换为分类类型")
                except Exception:
                    # 如果失败，保持原样
                    logger.debug(f"  列 {col} 保持为对象类型")
        
        # 转换为Parquet格式，使用fastparquet引擎处理复杂类型
        logger.info(f"开始转换为Parquet格式，输出文件: {output_file}")
        df.to_parquet(output_file, compression=compression, index=False, engine='pyarrow')
        
        # 获取文件大小进行比较
        csv_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        parquet_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0
        
        logger.info(f"✅ 转换成功！")
        logger.info(f"   CSV文件大小: {csv_size:.2f} MB")
        logger.info(f"   Parquet文件大小: {parquet_size:.2f} MB")
        logger.info(f"   压缩率: {compression_ratio:.2f}x")
        
        return True
    except Exception as e:
        logger.error(f"❌ 转换失败: {str(e)}")
        return False


def batch_convert_csv_to_parquet(input_dir, output_dir, pattern='*.csv', compression='snappy'):
    """
    批量将CSV文件转换为Parquet格式
    
    Args:
        input_dir (str): 输入CSV文件目录
        output_dir (str): 输出Parquet文件目录
        pattern (str): 文件匹配模式
        compression (str): 压缩格式
    
    Returns:
        tuple: (成功数量, 失败数量)
    """
    import glob
    
    # 获取所有匹配的CSV文件
    search_pattern = os.path.join(input_dir, pattern)
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        logger.warning(f"在目录 {input_dir} 中未找到匹配 {pattern} 的CSV文件")
        return 0, 0
    
    logger.info(f"找到 {len(csv_files)} 个CSV文件待转换")
    ensure_directory_exists(output_dir)
    
    success_count = 0
    fail_count = 0
    
    # 批量转换
    for csv_file in tqdm(csv_files, desc="转换进度", unit="文件"):
        # 生成输出文件名
        filename = os.path.basename(csv_file)
        parquet_filename = os.path.splitext(filename)[0] + '.parquet'
        parquet_file = os.path.join(output_dir, parquet_filename)
        
        # 转换文件
        if convert_csv_to_parquet(csv_file, parquet_file, compression):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"批量转换完成！")
    logger.info(f"✅ 成功: {success_count} 个文件")
    logger.info(f"❌ 失败: {fail_count} 个文件")
    
    return success_count, fail_count


def validate_parquet_file(parquet_file):
    """
    验证Parquet文件是否可以正确读取
    
    Args:
        parquet_file (str): Parquet文件路径
    
    Returns:
        bool: 是否验证成功
    """
    try:
        logger.info(f"验证Parquet文件: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        logger.info(f"✅ Parquet文件验证成功，共 {len(df)} 行数据")
        return True
    except Exception as e:
        logger.error(f"❌ Parquet文件验证失败: {str(e)}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSV文件转换为Parquet格式工具')
    
    # 互斥的参数组：单个文件转换或批量转换
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input', help='输入CSV文件路径')
    group.add_argument('-d', '--dir', help='输入CSV文件目录')
    
    parser.add_argument('-o', '--output', help='输出Parquet文件或目录路径')
    parser.add_argument('-c', '--compression', default='snappy', 
                        choices=['snappy', 'gzip', 'brotli', 'lz4', 'uncompressed'],
                        help='压缩格式，默认: snappy')
    parser.add_argument('-p', '--pattern', default='*.csv', help='批量转换时的文件匹配模式，默认: *.csv')
    parser.add_argument('--validate', action='store_true', help='验证生成的Parquet文件')
    
    args = parser.parse_args()
    
    # 处理单个文件转换
    if args.input:
        if not args.output:
            # 如果没有指定输出文件，则在输入文件同目录下生成
            base_name = os.path.splitext(args.input)[0]
            args.output = base_name + '.parquet'
        
        success = convert_csv_to_parquet(args.input, args.output, args.compression)
        
        # 验证生成的Parquet文件
        if success and args.validate:
            validate_parquet_file(args.output)
    
    # 处理批量转换
    elif args.dir:
        if not args.output:
            # 如果没有指定输出目录，则在输入目录同级创建parquet_output目录
            parent_dir = os.path.dirname(os.path.abspath(args.dir))
            args.output = os.path.join(parent_dir, 'parquet_output')
        
        success_count, fail_count = batch_convert_csv_to_parquet(
            args.dir, args.output, args.pattern, args.compression
        )
        
        # 验证生成的Parquet文件（只验证成功的文件）
        if args.validate and success_count > 0:
            import glob
            parquet_files = glob.glob(os.path.join(args.output, '*.parquet'))
            logger.info(f"开始验证 {len(parquet_files)} 个Parquet文件")
            
            valid_count = 0
            invalid_count = 0
            for parquet_file in tqdm(parquet_files, desc="验证进度", unit="文件"):
                if validate_parquet_file(parquet_file):
                    valid_count += 1
                else:
                    invalid_count += 1
            
            logger.info(f"验证完成！")
            logger.info(f"✅ 有效: {valid_count} 个文件")
            logger.info(f"❌ 无效: {invalid_count} 个文件")


def convert_specific_files():
    """直接转换用户指定的三个CSV文件"""
    logger.info("开始转换用户指定的CSV文件...")
    
    # 指定的三个CSV文件路径
    files_to_convert = [
        r'd:\code\502\Bike A\merged_data\cleaned_2023_data.csv',
        r'd:\code\502\Bike A\merged_data\cleaned_2024_data.csv',
        r'd:\code\502\Bike A\merged_data\cleaned_2025_data.csv'
    ]
    
    # 输出目录
    output_dir = r'd:\code\502\Bike A\merged_data\parquet_files'
    ensure_directory_exists(output_dir)
    
    success_count = 0
    fail_count = 0
    
    for csv_file in files_to_convert:
        if not os.path.exists(csv_file):
            logger.error(f"❌ 文件不存在: {csv_file}")
            fail_count += 1
            continue
        
        # 生成输出文件名
        filename = os.path.basename(csv_file)
        parquet_filename = os.path.splitext(filename)[0] + '.parquet'
        parquet_file = os.path.join(output_dir, parquet_filename)
        
        # 转换文件
        if convert_csv_to_parquet(csv_file, parquet_file, 'snappy'):
            success_count += 1
            # 验证生成的文件
            validate_parquet_file(parquet_file)
        else:
            fail_count += 1
    
    logger.info(f"指定文件转换完成！")
    logger.info(f"✅ 成功: {success_count} 个文件")
    logger.info(f"❌ 失败: {fail_count} 个文件")


if __name__ == "__main__":
    # 检查命令行参数，如果没有参数则直接调用转换函数
    if len(sys.argv) == 1:
        logger.info("未提供命令行参数，将直接转换指定的CSV文件")
        convert_specific_files()
    else:
        main()
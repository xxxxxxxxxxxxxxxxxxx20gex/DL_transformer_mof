#!/usr/bin/env python3
"""
合并cif_id.npy和mofid_string.npy为id_prop.npy格式
"""

import numpy as np
import os

def merge_id_prop(cif_id_file, mofid_string_file, output_file):
    """
    合并cif_id.npy和mofid_string.npy为id_prop.npy格式
    
    Args:
        cif_id_file: cif_id.npy文件路径
        mofid_string_file: mofid_string.npy文件路径  
        output_file: 输出的id_prop.npy文件路径
    """
    print("=" * 60)
    print("合并CIF ID和MOFid字符串文件")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(cif_id_file):
        print(f"错误: 文件不存在 - {cif_id_file}")
        return False
        
    if not os.path.exists(mofid_string_file):
        print(f"错误: 文件不存在 - {mofid_string_file}")
        return False
    
    # 加载数据
    print(f"加载 {cif_id_file}...")
    cif_ids = np.load(cif_id_file, allow_pickle=True)
    print(f"  - 形状: {cif_ids.shape}")
    print(f"  - 数据类型: {cif_ids.dtype}")
    
    print(f"加载 {mofid_string_file}...")
    mofid_strings = np.load(mofid_string_file, allow_pickle=True)
    print(f"  - 形状: {mofid_strings.shape}")
    print(f"  - 数据类型: {mofid_strings.dtype}")
    
    # 检查数据长度是否一致
    if len(cif_ids) != len(mofid_strings):
        print(f"错误: 数据长度不一致!")
        print(f"  - cif_ids: {len(cif_ids)}")
        print(f"  - mofid_strings: {len(mofid_strings)}")
        return False
    
    print(f"数据长度一致: {len(cif_ids)} 个条目")
    
    # 合并数据
    print("合并数据...")
    merged_data = []
    for i in range(len(cif_ids)):
        # 移除.cif扩展名（如果存在）
        cif_id = cif_ids[i]
        if cif_id.endswith('.cif'):
            cif_id = cif_id[:-4]  # 移除.cif扩展名
        
        merged_data.append([cif_id, mofid_strings[i]])
    
    # 转换为numpy数组
    merged_array = np.array(merged_data, dtype=object)
    print(f"合并后形状: {merged_array.shape}")
    
    # 保存结果
    print(f"保存到 {output_file}...")
    np.save(output_file, merged_array)
    
    # 验证保存结果
    print("验证保存结果...")
    loaded_data = np.load(output_file, allow_pickle=True)
    print(f"验证 - 形状: {loaded_data.shape}")
    print(f"验证 - 数据类型: {loaded_data.dtype}")
    
    # 显示前5个条目
    print("\n前5个条目:")
    for i, item in enumerate(loaded_data[:5]):
        print(f"{i}: {item}")
    
    print(f"\n✓ 成功合并并保存到: {output_file}")
    print(f"✓ 总共处理了 {len(merged_data)} 个条目")
    
    return True

if __name__ == "__main__":
    # 文件路径
    cif_id_file = "/root/autodl-tmp/MOFormer/cif_all/cif_id.npy"
    mofid_string_file = "/root/autodl-tmp/MOFormer/cif_all/mofid_string.npy"
    output_file = "/root/autodl-tmp/MOFormer/cif_all/id_prop.npy"
    
    # 执行合并
    success = merge_id_prop(cif_id_file, mofid_string_file, output_file)
    
    if success:
        print("\n" + "=" * 60)
        print("合并完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("合并失败！")
        print("=" * 60)

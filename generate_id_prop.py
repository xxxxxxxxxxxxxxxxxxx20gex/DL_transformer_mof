import os
import sys
import numpy as np
import glob
import multiprocessing as mp
from tqdm import tqdm
import time
import pickle

# 添加mofid目录到Python路径，确保优先使用本地版本
mofid_path = os.path.join(os.path.dirname(__file__), 'mofid')
sys.path.insert(0, mofid_path)
from Python.run_mofid import cif2mofid

def process_single_cif(cif_file):
    """
    处理单个CIF文件

    Args:
        cif_file: CIF文件路径
        
    Returns:
        tuple: (cif_id, mofid_string) 或 (cif_id, None) 如果处理失败
    """
    try:
        # 获取CIF文件名（不含扩展名）
        cif_id = os.path.splitext(os.path.basename(cif_file))[0]
        
        # 使用MOFid生成MOFid字符串
        mofid_result = cif2mofid(cif_file)
        mofid_string = mofid_result['mofid']
        
        return (cif_id, mofid_string)
        
    except Exception as e:
        error_msg = str(e)
        # 特殊处理"More than one fragment found"错误
        if "More than one fragment found" in error_msg:
            print(f"警告: {cif_file} - 发现多个片段，跳过此文件")
        else:
            print(f"错误: {cif_file} - {error_msg}")
        return (os.path.splitext(os.path.basename(cif_file))[0], None)

def save_checkpoint(data, processed_files, output_file, checkpoint_file):
    """
    保存检查点数据
    
    Args:
        data: 已处理的数据
        processed_files: 已处理的文件列表
        output_file: 输出文件路径
        checkpoint_file: 检查点文件路径
    """
    # 保存npy文件
    if data:
        id_prop_array = np.array(data, dtype=object)
        np.save(output_file, id_prop_array)
    
    # 保存检查点信息
    checkpoint_info = {
        'processed_files': processed_files,
        'data_count': len(data),
        'timestamp': time.time()
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_info, f)

def load_checkpoint(checkpoint_file:str):
    """
    加载检查点数据
    
    Args:
        checkpoint_file: 检查点文件路径
        
    Returns:
        tuple: (已处理的数据, 已处理的文件列表) 或 (None, None) 如果检查点不存在
    """
    if not os.path.exists(checkpoint_file):
        return None, None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_info = pickle.load(f)
        
        # 尝试加载npy文件
        output_file = checkpoint_file.replace('.checkpoint', '.npy')
        if os.path.exists(output_file):
            data = np.load(output_file, allow_pickle=True).tolist()
            return data, checkpoint_info['processed_files']
        else:
            return [], checkpoint_info['processed_files']
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return None, None

def generate_id_prop_npy_serial(cif_dir, output_file='id_prop.npy', save_interval=1000):
    """
    为CIF文件夹生成id_prop.npy文件（串行版本，支持增量保存）
    
    Args:
        cif_dir: CIF文件目录
        output_file: 输出的npy文件名
        save_interval: 每处理多少个文件保存一次
    """
    cif_files = glob.glob(os.path.join(cif_dir, '*.cif'))
    print(f"找到 {len(cif_files)} 个CIF文件")
    
    if not cif_files:
        print("未找到CIF文件")
        return np.array([], dtype=object)
    
    # 检查点文件路径
    checkpoint_file = output_file.replace('.npy', '.checkpoint')
    
    # 尝试加载检查点
    id_prop_data, processed_files = load_checkpoint(checkpoint_file)
    if id_prop_data is not None:
        print(f"发现检查点，已处理 {len(processed_files)} 个文件")
        print(f"从断点继续处理...")
        start_index = len(processed_files)
    else:
        print("开始新的处理...")
        id_prop_data = []
        processed_files = []
        start_index = 0
    
    print("开始串行处理...")
    start_time = time.time()
    
    successful_count = len([x for x in id_prop_data if x[1] is not None])
    failed_count = len([x for x in id_prop_data if x[1] is None])
    fragment_error_count = 0
    
    # 使用tqdm显示进度，从断点开始
    remaining_files = cif_files[start_index:]
    for i, cif_file in enumerate(tqdm(remaining_files, desc="处理CIF文件", initial=start_index, total=len(cif_files))):
        result = process_single_cif(cif_file)
        
        if result[1] is not None:
            id_prop_data.append(result)
            successful_count += 1
        else:
            id_prop_data.append(result)
            failed_count += 1
            # 检查是否是片段错误
            try:
                cif2mofid(cif_file)
            except Exception as e:
                if "More than one fragment found" in str(e):
                    fragment_error_count += 1
            except:
                pass
        
        processed_files.append(cif_file)
        
        # 定期保存检查点
        if (i + 1) % save_interval == 0:
            save_checkpoint(id_prop_data, processed_files, output_file, checkpoint_file)
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / (i + 1)
            remaining_files_count = len(cif_files) - (i + 1)
            estimated_remaining_time = remaining_files_count * avg_time_per_file
            
            print(f"\n检查点保存 (已处理 {i + 1}/{len(cif_files)}):")
            print(f"  成功: {successful_count}, 失败: {failed_count}")
            print(f"  片段错误: {fragment_error_count}")
            print(f"  平均每文件耗时: {avg_time_per_file:.2f}秒")
            print(f"  预计剩余时间: {estimated_remaining_time/3600:.1f}小时")
            print(f"  完成度: {(i + 1)/len(cif_files)*100:.1f}%")
            print(f"  内存使用: {len(id_prop_data)} 个条目")
    
    # 最终保存
    save_checkpoint(id_prop_data, processed_files, output_file, checkpoint_file)
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n处理完成！")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
    print(f"成功处理: {successful_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"片段错误: {fragment_error_count} 个文件")
    print(f"成功率: {successful_count/len(cif_files)*100:.1f}%")
    
    # 清理检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"已清理检查点文件: {checkpoint_file}")
    
    if id_prop_data:
        print(f"成功生成 {output_file}，包含 {len(id_prop_data)} 个条目")
        return np.array(id_prop_data, dtype=object)
    else:
        print("没有成功处理任何文件")
        return np.array([], dtype=object)

if __name__ == "__main__":
    # 为cif_all/cif文件夹生成id_prop.npy
    cif_directory = "/root/autodl-tmp/MOFormer/cif_toy"  # CIF文件夹路径
    output_file = "./cif_all/npy/id_prop.npy"  # 输出文件路径

    print(f"开始处理目录: {cif_directory}")
    print(f"输出文件: {output_file}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 生成id_prop.npy文件（串行版本，每1000个文件保存一次）
    result = generate_id_prop_npy_serial(
        cif_directory, 
        output_file,
        save_interval=1000  # 每1000个文件保存一次
    )

    print(f"处理完成！共处理了 {len(result)} 个CIF文件")
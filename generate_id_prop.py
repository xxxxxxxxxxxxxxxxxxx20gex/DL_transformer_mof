import os
import sys
import numpy as np
import glob
import multiprocessing as mp
from multiprocessing import Pool, Manager
from tqdm import tqdm
import time
import pickle
import tempfile
import shutil

# 添加mofid目录到Python路径，确保优先使用本地版本
mofid_path = os.path.join(os.path.dirname(__file__), 'mofid')
sys.path.insert(0, mofid_path)
from Python.run_mofid import cif2mofid

def process_single_cif(cif_file):
    try:
        # 获取CIF文件名（包含.cif扩展名）
        cif_filename = os.path.basename(cif_file)
        
        # 使用MOFid生成MOFid字符串
        mofid_result = cif2mofid(cif_file)
        mofid_string = mofid_result['mofid']
        return [cif_filename, mofid_string]
        
    except Exception as e:
        error_msg = str(e)
        # 特殊处理"More than one fragment found"错误
        if "More than one fragment found" in error_msg:
            # 静默处理片段错误，避免过多输出
            pass
        return [os.path.basename(cif_file), None]

def process_batch_cif(args):
    """
    并行处理一批CIF文件
    
    Args:
        args: tuple (cif_files_batch, batch_id, temp_dir, progress_counter, progress_lock)
        
    Returns:
        tuple: (batch_id, temp_file_path, successful_count, failed_count, fragment_error_count)
    """
    cif_files_batch, batch_id, temp_dir, progress_counter, progress_lock = args
    
    # 批次开始通知
    print(f"🚀 开始处理批次 {batch_id}: {len(cif_files_batch)} 个文件")
    
    batch_results = []
    successful_count = 0
    failed_count = 0
    fragment_error_count = 0
    
    # 处理当前批次的所有文件
    for cif_file in cif_files_batch:
        result = process_single_cif(cif_file)
        batch_results.append(result)
        print(batch_results)
        if result[1] is not None:
            successful_count += 1
        else:
            failed_count += 1
            # 检查是否是片段错误
            try:
                cif2mofid(cif_file)
            except Exception as e:
                if "More than one fragment found" in str(e):
                    fragment_error_count += 1
            except:
                pass
        
        # 更新全局进度
        if progress_counter is not None and progress_lock is not None:
            with progress_lock:
                progress_counter.value += 1
    
    # 保存批次结果到临时文件
    temp_file = os.path.join(temp_dir, f'batch_{batch_id}.npy')
    if batch_results:
        batch_array = np.array(batch_results, dtype=object)
        np.save(temp_file, batch_array)
    
    # 批次完成通知
    print(f"✓ 批次 {batch_id} 完成: {len(cif_files_batch)} 个文件 "
          f"(成功: {successful_count}, 失败: {failed_count}, 片段错误: {fragment_error_count})")
    
    return (batch_id, temp_file, successful_count, failed_count, fragment_error_count)

def generate_id_prop_npy_serial(cif_dir, output_file='id_prop.npy', save_interval=1000):
    print("注意: 串行版本已弃用，自动切换到并行版本以提升性能")
    return generate_id_prop_npy_parallel(cif_dir, output_file, batch_size=save_interval)

def generate_id_prop_npy_parallel(cif_dir, output_file='id_prop.npy', batch_size=1000, num_processes=None):
    """
    为CIF文件夹生成id_prop.npy文件（并行版本，内存优化）
    
    Args:
        cif_dir: CIF文件目录
        output_file: 输出的npy文件名
        batch_size: 每批处理的文件数量
        num_processes: 进程数量，默认为CPU核心数
    """
    cif_files = glob.glob(os.path.join(cif_dir, '*.cif'))
    print(f"找到 {len(cif_files)} 个CIF文件")
    
    if not cif_files:
        print("未找到CIF文件")
        return np.array([], dtype=object)
    
    # 设置进程数量
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # 最多8个进程，避免过度占用资源
    
    print(f"使用 {num_processes} 个进程进行并行处理")
    print(f"批次大小: {batch_size}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='mof_conversion_')
    print(f"临时目录: {temp_dir}")
    
    try:
        # 将文件分批
        batches = []
        for i in range(0, len(cif_files), batch_size):
            batch_files = cif_files[i:i+batch_size]
            batches.append(batch_files)
        
        print(f"总共分为 {len(batches)} 个批次")
        
        # 创建共享进度计数器和锁
        manager = Manager()
        progress_counter = manager.Value('i', 0)
        progress_lock = manager.Lock()
        
        # 准备并行处理参数
        process_args = []
        for batch_id, batch_files in enumerate(batches):
            process_args.append((batch_files, batch_id, temp_dir, progress_counter, progress_lock))
        
        start_time = time.time()
        
        # 并行处理所有批次
        with Pool(processes=num_processes) as pool:
            # 启动进度监控
            print("开始并行处理...")
            
            # 异步执行所有批次
            async_results = pool.map_async(process_batch_cif, process_args)
            
            # 监控进度
            total_files = len(cif_files)
            with tqdm(total=total_files, desc="处理CIF文件") as pbar:
                last_progress = 0
                while not async_results.ready():
                    current_progress = progress_counter.value
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                    time.sleep(1)
                
                # 确保进度条完成
                final_progress = progress_counter.value
                pbar.update(final_progress - last_progress)
            
            # 获取所有批次结果
            batch_results = async_results.get()
        
        # 统计处理结果
        total_successful = 0
        total_failed = 0
        total_fragment_errors = 0
        temp_files = []
        
        print(f"\n批次处理结果汇总:")
        print("-" * 50)
        for batch_id, temp_file, successful, failed, fragment_errors in batch_results:
            total_successful += successful
            total_failed += failed
            total_fragment_errors += fragment_errors
            if os.path.exists(temp_file):
                temp_files.append(temp_file)
            
            # 显示每个批次的详细结果
            batch_total = successful + failed
            success_rate = (successful / batch_total * 100) if batch_total > 0 else 0
            print(f"批次 {batch_id:3d}: {batch_total:4d} 文件, "
                  f"成功 {successful:4d} ({success_rate:5.1f}%), "
                  f"失败 {failed:3d}, 片段错误 {fragment_errors:3d}")
        
        print(f"\n所有批次处理完成！")
        print(f"成功: {total_successful}, 失败: {total_failed}, 片段错误: {total_fragment_errors}")
        
        # 合并所有临时文件
        print("开始合并临时文件...")
        merge_temp_files(temp_files, output_file)
        
        # 最终统计
        total_time = time.time() - start_time
        print(f"\n处理完成！")
        print(f"总耗时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        print(f"成功处理: {total_successful} 个文件")
        print(f"处理失败: {total_failed} 个文件")
        print(f"片段错误: {total_fragment_errors} 个文件")
        print(f"成功率: {total_successful/len(cif_files)*100:.1f}%")
        print(f"加速比: 预计比串行处理快 {num_processes:.1f}x")
        
        # 验证最终文件
        if os.path.exists(output_file):
            final_data = np.load(output_file, allow_pickle=True)
            print(f"最终文件包含 {len(final_data)} 个条目")
            return final_data
        else:
            print("最终文件生成失败")
            return np.array([], dtype=object)
            
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")

def merge_temp_files(temp_files, output_file):
    """
    合并临时npy文件为最终文件
    
    Args:
        temp_files: 临时文件列表
        output_file: 输出文件路径
    """
    all_data = []
    
    print(f"合并 {len(temp_files)} 个临时文件...")
    for temp_file in tqdm(temp_files, desc="合并文件"):
        if os.path.exists(temp_file):
            try:
                batch_data = np.load(temp_file, allow_pickle=True)
                all_data.extend(batch_data.tolist())
            except Exception as e:
                print(f"加载临时文件失败: {temp_file} - {e}")
    
    if all_data:
        # 保存最终结果
        final_array = np.array(all_data, dtype=object)
        np.save(output_file, final_array)
        print(f"成功合并保存到: {output_file}")
    else:
        print("没有数据可以合并")

if __name__ == "__main__":
    # 为cif_all/cif文件夹生成id_prop.npy
    cif_directory = "/root/autodl-tmp/MOFormer/cif_toy"  # CIF文件夹路径
    output_file = "/root/autodl-tmp/MOFormer/cif_toy/id_prop1.npy"  # 输出文件路径

    print("=" * 60)
    print("MOF数据集并行转换工具")
    print("=" * 60)
    print(f"输入目录: {cif_directory}")
    print(f"输出文件: {output_file}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 检查系统资源
    cpu_count = mp.cpu_count()
    print(f"系统CPU核心数: {cpu_count}")
    
    # 生成id_prop.npy文件（并行版本，内存优化）
    result = generate_id_prop_npy_parallel(
        cif_directory, 
        output_file,
        batch_size=500,  # 每批处理500个文件，平衡内存和效率
        num_processes=min(cpu_count, 8)  # 最多使用8个进程
    )

    print("=" * 60)
    print(f"处理完成！共处理了 {len(result)} 个CIF文件")
    print("=" * 60)
    
    # 显示前5个条目作为验证
    if len(result) > 0:
        print("\n前5个条目:")
        for i, item in enumerate(result[:5]):
            print(f"{i}: {item}")
    else:
        print("没有成功处理任何文件！")
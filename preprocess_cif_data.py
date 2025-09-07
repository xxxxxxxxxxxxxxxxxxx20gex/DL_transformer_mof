#!/usr/bin/env python3
"""
CIF数据预处理脚本
将CIF文件预处理为更快的.npz格式，大幅减少训练时的数据加载时间
"""

import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import argparse
from pymatgen.core.structure import Structure
from tokenizer.mof_tokenizer import MOFTokenizer
from dataset.dataset_multiview import AtomCustomJSONInitializer, GaussianDistance

def preprocess_cif_data(root_dir, output_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, 
                       vocab_path='tokenizer/vocab_full.txt', batch_size=100):
    """
    预处理CIF数据为.npz格式
    
    Args:
        root_dir: CIF文件目录
        output_dir: 输出目录
        max_num_nbr: 最大邻居数
        radius: 搜索半径
        dmin: 最小距离
        step: 步长
        vocab_path: 词汇表路径
        batch_size: 批处理大小
    """
    
    print(f"🚀 开始预处理CIF数据...")
    print(f"   输入目录: {root_dir}")
    print(f"   输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    id_prop_file = os.path.join(root_dir, 'id_prop.npy')
    if not os.path.exists(id_prop_file):
        raise FileNotFoundError(f"找不到 {id_prop_file}")
    
    id_prop_data = np.load(id_prop_file, allow_pickle=True)
    print(f"   数据量: {len(id_prop_data)} 个样本")
    
    # 初始化组件
    tokenizer = MOFTokenizer(vocab_path, model_max_length=512, padding_side='right')
    atom_init_file = os.path.join('benchmark_datasets/atom_init.json')
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    
    # 预处理数据
    processed_data = []
    
    for i, (cif_id, mofid) in enumerate(tqdm(id_prop_data, desc="预处理进度")):
        try:
            # 读取CIF文件
            fname = cif_id if cif_id.endswith('.cif') else cif_id + '.cif'
            cif_path = os.path.join(root_dir, fname)
            
            if not os.path.exists(cif_path):
                print(f"⚠️  跳过不存在的文件: {cif_path}")
                continue
                
            crys = Structure.from_file(cif_path)
            
            # 处理token
            tokens = tokenizer.encode(mofid, max_length=512, truncation=True, padding='max_length')
            
            # 处理原子特征
            atom_fea = np.vstack([ari.get_atom_fea(crys[i].specie.number) 
                                 for i in range(len(crys))])
            
            # 处理邻居特征
            all_nbrs = crys.get_all_neighbors(radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < max_num_nbr:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                     [0] * (max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                 [radius + 1.] * (max_num_nbr - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
            
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = gdf.expand(nbr_fea)
            
            # 保存处理后的数据
            processed_data.append({
                'cif_id': cif_id,
                'mofid': mofid,
                'tokens': tokens,
                'atom_fea': atom_fea,
                'nbr_fea': nbr_fea,
                'nbr_fea_idx': nbr_fea_idx
            })
            
            # 批量保存
            if len(processed_data) >= batch_size:
                save_batch(processed_data, output_dir, i // batch_size)
                processed_data = []
                
        except Exception as e:
            print(f"❌ 处理 {cif_id} 时出错: {e}")
            continue
    
    # 保存剩余数据
    if processed_data:
        save_batch(processed_data, output_dir, len(id_prop_data) // batch_size)
    
    # 保存元数据
    metadata = {
        'total_samples': len(id_prop_data),
        'max_num_nbr': max_num_nbr,
        'radius': radius,
        'dmin': dmin,
        'step': step,
        'vocab_path': vocab_path
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ 预处理完成！数据保存在: {output_dir}")

def save_batch(data_batch, output_dir, batch_idx):
    """保存一批数据"""
    batch_file = os.path.join(output_dir, f'batch_{batch_idx:04d}.npz')
    
    # 提取所有字段
    cif_ids = [item['cif_id'] for item in data_batch]
    mofids = [item['mofid'] for item in data_batch]
    tokens = np.array([item['tokens'] for item in data_batch])
    
    # 由于atom_fea和nbr_fea的尺寸可能不同，我们需要特殊处理
    # 这里我们保存为列表，在加载时再处理
    atom_feas = [item['atom_fea'] for item in data_batch]
    nbr_feas = [item['nbr_fea'] for item in data_batch]
    nbr_fea_idxs = [item['nbr_fea_idx'] for item in data_batch]
    
    np.savez_compressed(
        batch_file,
        cif_ids=cif_ids,
        mofids=mofids,
        tokens=tokens,
        atom_feas=atom_feas,
        nbr_feas=nbr_feas,
        nbr_fea_idxs=nbr_fea_idxs
    )

def main():
    parser = argparse.ArgumentParser(description='预处理CIF数据')
    parser.add_argument('--input_dir', type=str, required=True, help='CIF文件输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='预处理数据输出目录')
    parser.add_argument('--max_num_nbr', type=int, default=12, help='最大邻居数')
    parser.add_argument('--radius', type=float, default=8, help='搜索半径')
    parser.add_argument('--dmin', type=float, default=0, help='最小距离')
    parser.add_argument('--step', type=float, default=0.2, help='步长')
    parser.add_argument('--vocab_path', type=str, default='tokenizer/vocab_full.txt', help='词汇表路径')
    parser.add_argument('--batch_size', type=int, default=100, help='批处理大小')
    
    args = parser.parse_args()
    
    preprocess_cif_data(
        root_dir=args.input_dir,
        output_dir=args.output_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()

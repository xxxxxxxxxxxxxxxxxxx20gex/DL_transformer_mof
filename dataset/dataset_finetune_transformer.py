"""
Dataset for Transformer Fine-tuning

该数据集用于Transformer模型的微调训练，处理MOFid文本序列和对应的回归标签。
"""

from __future__ import print_function, division

import functools
import numpy as np
import torch
from torch.utils.data import Dataset


class MOF_ID_Dataset(Dataset):
    """
    用于Transformer微调的数据集
    
    数据格式：
    - 输入：CSV的两列 => [MOFid文本, 数值标签]
    - 输出：token id张量（长度512，已padding）与float标签
    
    Args:
        data: numpy数组，shape=(N, 2)，第一列为MOFid文本，第二列为标签
        tokenizer: MOFTokenizer实例，用于将文本编码为token序列
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.mofid = self.data[:, 0].astype(str)
        # 编码：将MOFid文本encode为定长token序列（超长截断，右侧补齐）
        self.tokens = np.array([
            tokenizer.encode(i, max_length=512, truncation=True, padding='max_length') 
            for i in self.mofid
        ])
        self.label = self.data[:, 1].astype(float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)
            
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        """
        获取第index条样本
        
        Returns:
            X: token id张量, shape=(512,), dtype=long
            y: 标签张量, shape=(1,), dtype=float32
        """
        X = torch.tensor(self.tokens[index], dtype=torch.long)
        y = torch.tensor(self.label[index], dtype=torch.float32).view(-1, 1)
        return X, y


from __future__ import print_function, division

import csv
import functools
import  json
#import  you
import  random
import warnings
import math
import  numpy  as  np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler



class CORE_Dataset(Dataset):
    """
    用于 CORE 数据的示例数据集（保留原始实现，仅补充注释）。

    约定：
    - data: numpy 数组，含多列，第二列为 MOFid 文本，其余列为不同标签
    - tokenizer: 将 MOFid 文本编码为 token id 序列
    - which_label: 选择使用哪一列作为回归目标
    """
    def __init__(self, data, tokenizer, use_ratio = 1, which_label = 'void_fraction'):
            label_dict = {
                'void_fraction':2,
                'pld':3,
                'lcd':4
            }
            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 1].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, label_dict[which_label]].astype(float)
            # self.label = self.label/np.max(self.label)
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
            # 将第 index 条样本转换为 PyTorch 张量
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()

class MOF_ID_Dataset(Dataset):
    """
    用于微调的主数据集：
    - 输入：CSV 的两列 => [MOFid 文本, 数值标签]
    - 输出：token id 张量（长度 512，已 padding）与 float 标签
    """
    def __init__(self, data, tokenizer):
            self.data = data
        #     self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data[:, 0].astype(str)
            # 编码：将 MOFid 文本 encode 为定长 token 序列（超长截断，右侧补齐）
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
            self.label = self.data[:, 1].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
            # 读取第 index 条样本并转为张量
            X = torch.from_numpy(np.asarray(self.tokens[index]))  # 建议在训练前确保 dtype 为 long
            y = torch.from_numpy(np.asarray(self.label[index])).view(-1,1)

            return X, y.float()


class MOF_pretrain_Dataset(Dataset):
    """
    仅用于预训练阶段的无标签序列数据集：
    - 输入：MOFid 文本数组
    - 输出：token id 长整型张量
    """
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.mofid = self.data.astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.mofid)
            
    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
            # 将第 index 条样本编码为 LongTensor 以兼容 nn.Embedding
            X = torch.from_numpy(np.asarray(self.tokens[index]))

            return X.type(torch.LongTensor)


class MOF_tsne_Dataset(Dataset):
    """用于可视化/分析的 t-SNE 数据集（保留原始实现，仅补充注释）。"""
    def __init__(self, data, tokenizer):
            self.data = data
            self.mofid = self.data[:, 0].astype(str)
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 1].astype(float)

            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = self.label[index]
            topo = self.mofid[index].split('&&')[-1].split('.')[0]
            return X, y, topo


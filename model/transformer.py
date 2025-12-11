"""
Transformer 与回归头（微调版本）

本文件实现了用于微调的Transformer模型：
- PositionalEncoding: 位置编码
- Transformer: Transformer主干网络
- regressoionHead: 回归头
- TransformerRegressor: 完整的微调模型（Transformer + 回归头）

注意：当前 `PositionalEncoding` 的实现默认输入张量形状为 [seq_len, batch, d_model]，
而 `TransformerEncoderLayer` 使用了 `batch_first=True`（期望 [batch, seq_len, d_model]）。
"""

import pandas as pd
import logging
import numpy as np
import torch
import math
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    标准位置编码（正弦/余弦）
    
    使用固定的三角函数进行位置编码
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch, d_model] 或 [batch, seq_len, d_model]
        """
        # 支持 batch_first
        if x.dim() == 3 and x.size(1) > x.size(0):  # [batch, seq_len, d_model]
            x = x + self.pe[:x.size(1), 0, :]
        else:  # [seq_len, batch, d_model]
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    
    让模型自己学习最优的位置表示，而非使用固定的三角函数
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 可学习的位置嵌入
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    
    不仅考虑绝对位置，还考虑token之间的相对距离
    这对于化学序列特别重要，因为相邻原子的关系比远距离原子更重要
    
    参考: "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # 相对位置的范围: [-max_len+1, max_len-1]
        # 总共 2*max_len-1 个相对位置
        self.relative_positions = 2 * max_len - 1
        
        # 为每个相对位置学习一个嵌入
        self.relative_pos_embed = nn.Embedding(self.relative_positions, d_model)
        
        # 初始化
        nn.init.xavier_uniform_(self.relative_pos_embed.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch, seq_len, d_model]
        
        Returns:
            x: Tensor with relative position information, same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # 计算相对位置矩阵
        # positions[i, j] = i - j + max_len - 1
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        rel_positions = positions.T - positions + self.max_len - 1
        
        # Clamp to valid range
        rel_positions = torch.clamp(rel_positions, 0, self.relative_positions - 1)
        
        # 获取相对位置嵌入
        rel_pos_embed = self.relative_pos_embed(rel_positions)  # [seq_len, seq_len, d_model]
        
        # 将相对位置信息加到输入上
        # 简化版本：对每个token，加上所有其他token的相对位置嵌入的平均
        rel_pos_avg = rel_pos_embed.mean(dim=1)  # [seq_len, d_model]
        
        x = x + rel_pos_avg.unsqueeze(0)  # [1, seq_len, d_model]
        
        return self.dropout(x)


class regressoionHead(nn.Module):

    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, d_embedding//4)
        self.layer3 = nn.Linear(d_embedding//4, d_embedding//8)
        self.layer4 = nn.Linear(d_embedding//8, 1)
        self.relu=nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        return self.layer4(x)


class Transformer(nn.Module):
    """
    Transformer编码器
    
    支持三种位置编码方式：
    - 'sinusoidal': 标准的正弦/余弦位置编码（默认）
    - 'learnable': 可学习的位置编码
    - 'relative': 相对位置编码
    """
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, 
                 dropout: float = 0.1, pos_encoding_type: str = 'sinusoidal'):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        
        # 根据配置选择位置编码类型
        if pos_encoding_type == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
        elif pos_encoding_type == 'relative':
            self.pos_encoder = RelativePositionalEncoding(d_model, dropout)
        else:  # 'sinusoidal' (default)
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            src: Tensor, shape [batch, seq_len] - token id序列
            
        Returns:
            output: Tensor, shape [batch, seq_len, d_model] - 编码后的特征
        """
        # Token嵌入 + 缩放
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        return output

class TransformerRegressor(nn.Module):
    """
    Transformer微调模型：结合Transformer主干和回归头
    
    输入：token id序列
    输出：回归预测值（标量）
    """
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = regressoionHead(d_model)

    def forward(self, src: Tensor = None, input_ids: Tensor = None, **kwargs) -> Tensor:
        """
        前向传播：提取CLS token特征并通过回归头预测
        
        Args:
            src: Tensor, shape [batch, seq_len] - token id序列（位置参数）
            input_ids: Tensor, shape [batch, seq_len] - token id序列（peft兼容）
            **kwargs: 其他可能的参数（忽略，用于 peft 兼容性）
            
        Returns:
            output: Tensor, shape [batch, 1] - 回归预测值
        """
        # 兼容 peft：优先使用 input_ids，否则使用 src
        x = input_ids if input_ids is not None else src
        if x is None:
            raise ValueError("Either src or input_ids must be provided")
        
        output = self.transformer(x)
        cls_token = output[:, 0:1, :]  # [batch, 1, d_model]
        output = self.regressionHead(cls_token)
        return output
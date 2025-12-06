"""
Transformer 与回归头（微调版本）

本文件实现了用于微调的Transformer模型：
- PositionalEncoding: 位置编码
- Transformer: Transformer主干网络
- regressoionHead: 回归头
- TransformerRegressor: 完整的微调模型（Transformer + 回归头）
- BDC_Representation: BDC二阶统计表示模块
- TransformerRegressorWithBDC: 带BDC模块的Transformer回归模型

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
        """将位置编码加到输入张量上。

        重要：此版本假设输入形状为 [seq_len, batch, d_model]。
        若你的编码器层使用了 batch_first=True，请改用教学版 `PositionalEncodingBatchFirst`。
        """
        x = x + self.pe[:x.size(0)]
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

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        # self.out = nn.Sequential(
        #     nn.LayerNorm(d_model),
        #     nn.Identity(),
        #     nn.Linear(d_model, ntoken) 
        # )
        self.init_weights()

    def init_weights(self) -> None:
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        """前向传播。

        参数
        - src: token id 张量，形状约定与 `PositionalEncoding` 保持一致。
        返回
        - output: Transformer 编码后的序列特征
        """
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
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

    def forward(self, src: Tensor) -> Tensor:
        """
        前向传播：提取CLS token特征并通过回归头预测
        
        Args:
            src: Tensor, shape [batch, seq_len] - token id序列
            
        Returns:
            output: Tensor, shape [batch, 1] - 回归预测值
        """
        output = self.transformer(src)
        cls_token = output[:, 0:1, :]  # [batch, 1, d_model]
        output = self.regressionHead(cls_token)
        return output


class BDC_Representation(nn.Module):
    """
    BDC (Bilateral Divergence Covariance) 表示模块
    
    通过计算特征的双边散度协方差矩阵来提取二阶统计信息，
    增强特征表示能力。
    
    参考：BDC-CLIP论文
    """
    def __init__(self):
        super().__init__()

    def bdc_pooling(self, x: Tensor) -> Tensor:
        """
        BDC池化：计算双边散度协方差矩阵
        
        Args:
            x: Tensor, shape [batch, dim, M] - 输入特征
            
        Returns:
            bdc: Tensor, shape [batch, dim, dim] - BDC协方差矩阵
        """
        batchSize, dim, M = x.shape
        
        # 单位矩阵和全1矩阵
        I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        
        # 计算二次项
        x_pow2 = x.bmm(x.transpose(1, 2)) / (2 * M)
        
        # 计算散度协方差
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.sqrt(dcov + 1e-5)
        
        # 中心化
        d1 = dcov.bmm(I_M / dim)
        d2 = (I_M / dim).bmm(dcov)
        d3 = (I_M / dim).bmm(dcov).bmm(I_M / dim)
        bdc = dcov - d1 - d2 + d3
        
        return bdc
    
    def triuvec(self, x: Tensor) -> Tensor:
        """
        提取上三角矩阵的向量表示
        
        Args:
            x: Tensor, shape [batch, dim, dim] - 输入矩阵
            
        Returns:
            y: Tensor, shape [batch, dim*(dim+1)/2] - 上三角向量
        """
        batchSize, dim, _ = x.shape
        r = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero(as_tuple=False)
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
        y = r[:, index].squeeze()
        return y
    
    def compute_weighted_features(self, global_token: Tensor, features: Tensor) -> Tensor:
        """
        使用全局token（如CLS token）计算注意力加权特征
        
        Args:
            global_token: Tensor, shape [batch, d] - 全局token特征
            features: Tensor, shape [batch, seq_len, d] - 序列特征
            
        Returns:
            weighted_features: Tensor, shape [batch, seq_len, d] - 加权后的特征
        """
        _, _, d = features.shape
        q = global_token.unsqueeze(1)  # [batch, 1, d]
        k, v = features, features
        
        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权特征
        weighted_features = attn_weights.transpose(1, 2) * v
        
        return weighted_features

    def forward(self, x_g: Tensor, x: Tensor) -> Tensor:
        """
        前向传播：计算BDC表示
        
        Args:
            x_g: Tensor, shape [batch, d] - 全局token（如CLS token）
            x: Tensor, shape [batch, seq_len, d] - 序列特征
            
        Returns:
            x_bdc: Tensor, shape [batch, d*(d+1)/2] - BDC表示向量
        """
        # 注意力加权
        x_weighted = self.compute_weighted_features(x_g, x)
        
        # BDC池化
        x_bdc = self.bdc_pooling(x_weighted.transpose(1, 2))
        
        # 提取上三角向量
        x_bdc = self.triuvec(x_bdc)
        
        return x_bdc


class BDCRegressionHead(nn.Module):
    """
    BDC回归头：将BDC表示映射到回归预测值
    
    输入维度：d_model * (d_model + 1) / 2
    输出维度：1
    """
    def __init__(self, d_model: int, reduce_dim: int = None):
        super().__init__()
        bdc_dim = int(d_model * (d_model + 1) / 2)
        
        # 如果BDC维度太大，先降维
        if reduce_dim is not None and reduce_dim < d_model:
            self.dim_reduction = nn.Linear(d_model, reduce_dim)
            bdc_dim = int(reduce_dim * (reduce_dim + 1) / 2)
        else:
            self.dim_reduction = None
            reduce_dim = d_model
        
        # 回归头
        self.layer1 = nn.Linear(bdc_dim, reduce_dim)
        self.layer2 = nn.Linear(reduce_dim, reduce_dim // 2)
        self.layer3 = nn.Linear(reduce_dim // 2, reduce_dim // 4)
        self.layer4 = nn.Linear(reduce_dim // 4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch, bdc_dim] - BDC表示
            
        Returns:
            output: Tensor, shape [batch, 1] - 回归预测值
        """
        x = self.dropout(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)


class TransformerRegressorWithBDC(nn.Module):
    """
    带BDC模块的Transformer回归模型
    
    使用BDC表示来增强特征表示能力，捕获二阶统计信息。
    
    输入：token id序列
    输出：回归预测值（标量）
    """
    def __init__(self, transformer, d_model: int, reduce_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.bdc_layer = BDC_Representation()
        
        # 可选的特征降维（如果d_model太大，BDC维度会爆炸）
        if reduce_dim is not None and reduce_dim < d_model:
            self.feature_reduce = nn.Linear(d_model, reduce_dim)
            self.use_reduction = True
            effective_dim = reduce_dim
        else:
            self.feature_reduce = None
            self.use_reduction = False
            effective_dim = d_model
        
        self.regressionHead = BDCRegressionHead(effective_dim)

    def forward(self, src: Tensor) -> Tensor:
        """
        前向传播：使用BDC表示进行回归预测
        
        Args:
            src: Tensor, shape [batch, seq_len] - token id序列
            
        Returns:
            output: Tensor, shape [batch, 1] - 回归预测值
        """
        # Transformer编码
        features = self.transformer(src)  # [batch, seq_len, d_model]
        
        # 可选的特征降维
        if self.use_reduction:
            features = self.feature_reduce(features)  # [batch, seq_len, reduce_dim]
        
        # 提取CLS token作为全局特征
        cls_token = features[:, 0, :]  # [batch, d_model or reduce_dim]
        
        # 计算BDC表示
        bdc_features = self.bdc_layer(cls_token, features)  # [batch, bdc_dim]
        
        # 回归预测
        output = self.regressionHead(bdc_features)  # [batch, 1]
        
        return output
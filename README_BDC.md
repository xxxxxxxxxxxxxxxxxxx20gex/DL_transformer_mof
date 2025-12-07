# BDC模块集成说明

## 概述

本项目集成了BDC (Bilateral Divergence Covariance) 模块，用于增强Transformer模型的特征表示能力。BDC模块通过计算二阶统计信息（协方差矩阵）来捕获特征之间的关系，从而提升模型的表现。

## BDC模块原理

### 核心思想

BDC模块的核心是计算特征的**双边散度协方差矩阵**，这是一种二阶统计表示方法，相比传统的一阶统计（如平均池化）能够捕获更丰富的特征信息。

### 处理流程

```
Transformer编码器输出 [batch, seq_len, d_model]
    ↓
1. 提取CLS token作为全局特征
    ↓
2. 使用CLS token对序列特征进行注意力加权
    ↓
3. 计算加权特征的BDC协方差矩阵 [batch, d_model, d_model]
    ↓
4. 提取上三角向量 [batch, d_model*(d_model+1)/2]
    ↓
5. 通过回归头预测 [batch, 1]
```

### 关键组件

1. **注意力加权**：使用CLS token作为query，对所有token特征进行加权，突出重要特征
2. **BDC池化**：计算双边散度协方差矩阵，捕获特征间的二阶关系
3. **降维选项**：由于BDC维度为 `d*(d+1)/2`，提供降维选项以控制计算复杂度

## 模型架构

### 标准模型（TransformerRegressor）

```python
Input (MOFid tokens) [batch, seq_len]
    ↓
Transformer Encoder
    ↓ [batch, seq_len, d_model]
Extract CLS token [batch, 1, d_model]
    ↓
Regression Head (4-layer MLP)
    ↓
Output [batch, 1]
```

### BDC增强模型（TransformerRegressorWithBDC）

```python
Input (MOFid tokens) [batch, seq_len]
    ↓
Transformer Encoder
    ↓ [batch, seq_len, d_model]
Optional: Feature Reduction [batch, seq_len, reduce_dim]
    ↓
BDC Module (Attention + Covariance Pooling)
    ↓ [batch, bdc_dim]
BDC Regression Head
    ↓
Output [batch, 1]
```

## 配置参数说明
### 预期效果

根据BDC方法在其他任务上的表现，预期BDC模块能够：

- ✅ 提升模型的表征能力
- ✅ 捕获特征间的二阶统计关系
- ✅ 在小样本场景下表现更好
- ✅ 提高模型泛化能力

## 代码结构

### 新增文件

- `model/transformer.py`：添加了以下类
  - `BDC_Representation`：BDC表示模块
  - `BDCRegressionHead`：BDC回归头
  - `TransformerRegressorWithBDC`：带BDC的完整模型

## 技术细节

### BDC协方差矩阵计算

```python
def bdc_pooling(self, x):
    """
    输入: x [batch, dim, M]  # M为序列长度
    输出: bdc [batch, dim, dim]
    """
    # 1. 计算二次项
    x_pow2 = x @ x.T / (2*M)
    
    # 2. 计算散度协方差
    dcov = I_M @ (x_pow2 * I) + (x_pow2 * I) @ I_M - 2 * x_pow2
    dcov = sqrt(clamp(dcov, min=0) + eps)
    
    # 3. 中心化
    d1 = dcov @ I_M / dim
    d2 = I_M / dim @ dcov
    d3 = I_M / dim @ dcov @ I_M / dim
    bdc = dcov - d1 - d2 + d3
    
    return bdc
```

### 注意力加权

```python
def compute_weighted_features(self, global_token, features):
    """
    使用全局token（CLS）对序列特征进行加权
    """
    q = global_token.unsqueeze(1)  # [batch, 1, d]
    k, v = features, features       # [batch, seq_len, d]
    
    attn_scores = (q @ k.T) / sqrt(d)
    attn_weights = softmax(attn_scores, dim=-1)
    weighted_features = attn_weights.T * v
    
    return weighted_features
```

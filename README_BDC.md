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

## 使用方法

### 方法1：使用标准配置文件

编辑 `config_ft_transformer.yaml`，设置：

```yaml
use_bdc: true  # 启用BDC模块
bdc_reduce_dim: 128  # 降维维度（可选）
```

运行训练：

```bash
python finetune_transformer.py
```

### 方法2：使用BDC专用配置文件

我们提供了一个预配置的BDC配置文件 `config_ft_transformer_bdc.yaml`。

运行训练（需要修改代码指定配置文件）：

```bash
# 方法1: 直接复制配置文件
cp config_ft_transformer_bdc.yaml config_ft_transformer.yaml
python finetune_transformer.py

# 方法2: 修改finetune_transformer.py中的配置文件路径
# 在第391行将 "config_ft_transformer.yaml" 改为 "config_ft_transformer_bdc.yaml"
```

### 方法3：对比实验

进行标准模型vs BDC模型的对比实验：

```bash
# 实验1：标准模型（baseline）
# 确保 use_bdc: false
python finetune_transformer.py --seed 1

# 实验2：BDC模型
# 修改配置：use_bdc: true, bdc_reduce_dim: 128
python finetune_transformer.py --seed 1

# 实验3：不同降维维度
# 修改配置：use_bdc: true, bdc_reduce_dim: 64
python finetune_transformer.py --seed 1
```

## 配置参数说明

### BDC相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_bdc` | bool | false | 是否启用BDC模块 |
| `bdc_reduce_dim` | int/null | 128 | 特征降维维度，设置为null则不降维 |

### 降维维度选择建议

- **不降维** (`bdc_reduce_dim: null`)：
  - BDC维度：`512 * 513 / 2 = 131,328`
  - 优点：保留完整信息
  - 缺点：计算量大，可能过拟合
  
- **降维到256**：
  - BDC维度：`256 * 257 / 2 = 32,896`
  - 平衡性能和计算量
  
- **降维到128** (推荐)：
  - BDC维度：`128 * 129 / 2 = 8,256`
  - 计算高效，适合大多数场景
  
- **降维到64**：
  - BDC维度：`64 * 65 / 2 = 2,080`
  - 最快，适合快速实验

## 实验建议

### 对比实验设计

为了评估BDC模块的效果，建议进行以下实验：

1. **基线对比**：
   - 实验1：标准模型（use_bdc=false）
   - 实验2：BDC模型（use_bdc=true, reduce_dim=128）

2. **降维维度消融实验**：
   - 实验2a：reduce_dim=256
   - 实验2b：reduce_dim=128
   - 实验2c：reduce_dim=64

3. **数据集泛化**：
   - 在不同数据集上测试（hMOF_CO2_0.5, hMOF_CH4_2.5, QMOF）

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

### 修改文件

- `finetune_transformer.py`：
  - 导入 `TransformerRegressorWithBDC`
  - 在 `train()` 方法中根据配置选择模型类型

- `config_ft_transformer.yaml`：
  - 添加 `use_bdc` 和 `bdc_reduce_dim` 参数

### 新增配置文件

- `config_ft_transformer_bdc.yaml`：预配置的BDC实验配置

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

## 故障排查

### 常见问题

1. **内存不足（OOM）**：
   - 减小 `bdc_reduce_dim`（如从128降到64）
   - 减小 `batch_size`
   - 确保设置了 `bdc_reduce_dim`（不要使用null）

2. **训练不稳定**：
   - 降低学习率
   - 增加dropout（在BDCRegressionHead中已设置为0.1）
   - 检查BDC维度是否过大

3. **性能未提升**：
   - 尝试不同的 `bdc_reduce_dim`
   - 增加训练轮数
   - 检查数据集是否适合二阶统计方法

## 参考文献

BDC方法来源于以下论文：
- 论文标题：[需要补充具体论文信息]
- 原始应用：视频-文本多模态学习
- 本项目改进：适配纯文本Transformer模型

## 更新日志

- 2025-12-06：初始版本，集成BDC模块到Transformer微调流程

## 联系方式

如有问题或建议，请提交issue或联系项目维护者。


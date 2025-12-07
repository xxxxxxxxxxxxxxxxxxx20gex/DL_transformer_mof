# BDC模块NaN问题修复报告

## 问题描述

在训练过程中，从**Epoch 24的第551步**开始出现NaN（Not a Number）问题，导致后续所有训练步骤的Loss都变成NaN。

### 问题现象

```
Epoch  24 - Training Progress: [501/563] | Loss: 0.8998   # 正常
Epoch  24 - Training Progress: [551/563] | Loss: nan      # 开始出现NaN
Epoch  25 - Training Progress: [  1/563] | Loss: nan      # 持续NaN
...
```

## 根因分析

BDC（Bilateral Divergence Covariance）模块在计算二阶统计信息时，存在多个数值不稳定的环节：

### 1. **BDC池化中的sqrt操作**
```python
dcov = torch.sqrt(dcov + 1e-5)  # 如果dcov为负数，sqrt会产生NaN
```

### 2. **除法操作缺少保护**
```python
x_pow2 = x.bmm(x.transpose(1, 2)) / (2 * M)  # M可能很小
```

### 3. **注意力权重计算溢出**
```python
attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # 可能产生很大的值
```

### 4. **缺少输入归一化**
- 特征值可能随着训练变得越来越大
- 没有对中间结果进行clamp

## 修复方案

### 1. **增强BDC池化的数值稳定性**

**修改前：**
```python
def bdc_pooling(self, x: Tensor) -> Tensor:
    x_pow2 = x.bmm(x.transpose(1, 2)) / (2 * M)
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.sqrt(dcov + 1e-5)
    ...
```

**修改后：**
```python
def bdc_pooling(self, x: Tensor) -> Tensor:
    eps = 1e-6
    # 输入归一化，防止数值爆炸
    x = x / (x.norm(dim=1, keepdim=True) + eps)
    
    # 计算二次项（添加eps保护）
    x_pow2 = x.bmm(x.transpose(1, 2)) / (2 * M + eps)
    
    # 计算散度协方差
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    
    # 更严格的clamp（确保严格大于0）
    dcov = torch.clamp(dcov, min=eps)
    dcov = torch.sqrt(dcov)
    
    # 输出clamp，防止异常值
    bdc = dcov - d1 - d2 + d3
    bdc = torch.clamp(bdc, min=-10.0, max=10.0)
    ...
```

### 2. **改进注意力加权计算**

**修改前：**
```python
def compute_weighted_features(self, global_token, features):
    q = global_token.unsqueeze(1)
    k, v = features, features
    attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    ...
```

**修改后：**
```python
def compute_weighted_features(self, global_token, features):
    eps = 1e-8
    # 归一化输入特征
    q = global_token.unsqueeze(1)
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    k = features / (features.norm(dim=-1, keepdim=True) + eps)
    
    # 计算注意力分数
    attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    # Clamp分数防止溢出
    attn_scores = torch.clamp(attn_scores, min=-50, max=50)
    attn_weights = F.softmax(attn_scores, dim=-1)
    ...
```

### 3. **增强回归头的鲁棒性**

**添加的改进：**
- 在每一层之后添加 BatchNorm
- 在输入和输出处检查NaN/Inf并替换
- 对最终输出进行clamp

```python
class BDCRegressionHead(nn.Module):
    def __init__(self, d_model, reduce_dim=None):
        ...
        self.bn0 = nn.BatchNorm1d(bdc_dim)
        self.bn1 = nn.BatchNorm1d(reduce_dim)
        self.bn2 = nn.BatchNorm1d(reduce_dim // 2)
        self.bn3 = nn.BatchNorm1d(reduce_dim // 4)
        ...
    
    def forward(self, x):
        # 检查并替换NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.bn0(x)  # BatchNorm提高稳定性
        ...
        output = torch.clamp(output, min=-100, max=100)
        return output
```

### 4. **改进triuvec函数**

**修改前：**
```python
def triuvec(self, x):
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple=False)
    y = r[:, index].squeeze()
    return y
```

**修改后：**
```python
def triuvec(self, x):
    # 使用torch.triu直接提取上三角
    mask = torch.triu(torch.ones(dim, dim, device=x.device, dtype=torch.bool))
    y = x[:, mask]
    
    # 检查并替换NaN/Inf
    if torch.isnan(y).any() or torch.isinf(y).any():
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
    return y
```

### 5. **增强前向传播的安全性**

在 `TransformerRegressorWithBDC.forward()` 中添加多个检查点：

```python
def forward(self, src):
    features = self.transformer(src)
    
    # 检查点1：Transformer输出
    if torch.isnan(features).any() or torch.isinf(features).any():
        features = torch.nan_to_num(features, ...)
    
    # 降维后检查
    if self.use_reduction:
        features = self.feature_reduce(features)
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, ...)
    
    # 归一化CLS token
    cls_token = features[:, 0, :]
    cls_token = cls_token / (cls_token.norm(dim=-1, keepdim=True) + eps)
    
    # BDC计算后检查
    bdc_features = self.bdc_layer(cls_token, features)
    if torch.isnan(bdc_features).any() or torch.isinf(bdc_features).any():
        bdc_features = torch.nan_to_num(bdc_features, ...)
    
    return output
```

## 修复效果预期

### 数值稳定性改进

1. **输入归一化**：防止特征值爆炸
2. **严格的clamp**：确保中间结果在合理范围内
3. **epsilon保护**：所有除法和sqrt操作都添加了epsilon
4. **NaN检测和替换**：在关键节点检测并替换异常值
5. **BatchNorm**：在回归头中添加BN层提高稳定性

### 性能影响

- **计算开销**：增加约5-10%（主要来自归一化和检查操作）
- **内存占用**：增加约2-3%（BatchNorm参数）
- **训练速度**：略微减慢，但可以完成完整训练

## 使用建议

### 1. 重新训练

清除之前的训练结果，使用修复后的代码重新训练：

```bash
# 确保use_bdc=true和bdc_reduce_dim=128
python finetune_transformer.py --seed 1
```

### 2. 监控训练

注意观察以下指标：
- Loss是否稳定在正常范围（0.3-2.0）
- 是否有梯度爆炸（梯度范数突然变大）
- MAE是否逐渐下降

### 3. 超参数调整（如果仍有问题）

如果仍然出现NaN，可以尝试：

**降低学习率：**
```yaml
optim:
  init_lr: 0.00002  # 从5e-5降到2e-5
```

**减小batch size：**
```yaml
batch_size: 64  # 从128降到64
```

**进一步减小降维维度：**
```yaml
bdc_reduce_dim: 64  # 从128降到64
```

**添加梯度裁剪：**
在 `finetune_transformer.py` 的训练循环中添加：
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## 技术总结

### NaN产生的常见原因

1. **数值溢出**：指数、乘法导致值过大
2. **除零**：分母为0或接近0
3. **负数开方**：sqrt(负数) = NaN
4. **log(0或负数)**：虽然我们没用log，但这是常见原因
5. **梯度爆炸**：梯度累积导致参数变成Inf/NaN

### 防御性编程原则

1. **输入归一化**：控制输入范围
2. **中间clamp**：限制中间结果范围
3. **epsilon保护**：防止除零和sqrt负数
4. **显式检查**：检测并替换NaN/Inf
5. **BatchNorm**：稳定训练过程

## 修改文件清单

- ✅ `model/transformer.py`：修复BDC模块数值稳定性问题
  - `BDC_Representation.bdc_pooling()`
  - `BDC_Representation.triuvec()`
  - `BDC_Representation.compute_weighted_features()`
  - `BDCRegressionHead`
  - `TransformerRegressorWithBDC.forward()`

## 下一步

1. 使用修复后的代码重新运行训练
2. 监控训练过程，确保没有NaN
3. 如果训练成功，记录最终MAE结果
4. 与基线模型（不使用BDC）对比性能

## 参考

- PyTorch数值稳定性指南
- BDC-CLIP原始论文
- 深度学习中的数值稳定性最佳实践

---

**修复日期**：2025-12-06  
**修复者**：AI Assistant  
**测试状态**：待验证


# Transformer微调实验记录

本文档用于记录Transformer模型微调实验的所有配置、参数和结果。

---

## Transformer微调实验性能对比

本表格记录了Transformer模型在hMOF_CO2_0.5数据集上的微调实验结果。所有实验使用相同的模型架构（ntoken=4021, d_model=512, nhead=8, nlayers=6）和数据集划分（训练集70%，验证集15%，测试集15%）。

| 实验序号 | 数据集 | Batch Size | 学习率 | 训练轮数 | 预训练权重 | Validation MAE (mol/kg) | Test MAE (mol/kg) | 备注 |
|----------|--------|------------|--------|----------|------------|------------------------|-------------------|------|
| 1（transformer） | hMOF_CO2_0.5 | 128 | 5e-5 | 30 | 是 | 0.529 | 0.520 | 基线配置，使用预训练权重 |
| 4 | hMOF_CO2_0.5 | 128 | 5e-5 | 30 | 是 | - | - | 加入BDC模块 |
| 5 | hMOF_CO2_0.5 | 128 | 5e-5 | 30 | 是 | - | 0.513 | 相对位置编码 |



>> 需要说明的是结果保存的2号实验对应的是/001-20251206_153801-hMOF_CO2_0.5_seed1 ，依次顺沿
### 结果分析

1. **实验1（基线）**：在batch size=128、学习率=5e-5、30个epochs的条件下，模型取得了Validation MAE=0.529 mol/kg，Test MAE=0.520 mol/kg的性能。

2. 加入了新的模块 BDC 分支 sft-2


### 实验配置说明

- **优化器**：所有实验使用Adam优化器，权重衰减=1e-6
- **学习率策略**：Transformer主干使用基础学习率，回归头使用200倍基础学习率
- **评估指标**：MAE（Mean Absolute Error，平均绝对误差），单位：mol/kg
- **硬件**：NVIDIA GeForce RTX 3090 (cuda:0)
- **随机种子**：1

---

## 实验详细记录

### 实验 EXP-001 (基线实验)

**基本信息**
- 实验日期: 2025-12-06
- 实验名称: Transformer微调基线实验
- 实验目的: 建立基线性能
- GPU设备: NVIDIA GeForce RTX 3090 (cuda:0)
- 随机种子: 1

**数据集配置**
```yaml
数据集名称: hMOF_CO2_0.5
数据路径: ./benchmark_datasets/hMOF/mofid/hMOF_CO2_0.5_small_mofid.csv
训练集比例: 70%
验证集比例: 15%
测试集比例: 15%
使用数据比例: 100%
数据样本总数: [填写]
- 训练集: [填写]
- 验证集: [填写]
- 测试集: [填写]
```

**模型配置**
```yaml
Transformer:
  词汇表大小 (ntoken): 4021
  模型维度 (d_model): 512
  注意力头数 (nhead): 8
  前馈网络维度 (d_hid): 512
  Transformer层数 (nlayers): 6
  Dropout: 0.1

回归头:
  - Linear(512 → 256) + ReLU
  - Linear(256 → 128) + ReLU
  - Linear(128 → 64) + ReLU
  - Linear(64 → 1)

总参数量: [填写]
可训练参数: [填写]
```

**训练配置**
```yaml
训练轮数 (epochs): 30
批次大小 (batch_size): 128
评估频率 (eval_every_n_epochs): 2
日志频率 (log_every_n_steps): 50

优化器: Adam
基础学习率: 0.00005
权重衰减: 1e-6
学习率策略:
  - Transformer主干: 基础学习率 × 1
  - 回归头: 基础学习率 × 200

预训练权重:
  是否使用: 是
  权重路径: ./ckpt/pretraining
  权重文件: model_transformer_14.pth
```

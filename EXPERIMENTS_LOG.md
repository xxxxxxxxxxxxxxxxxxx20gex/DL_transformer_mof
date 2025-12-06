# Transformer微调实验记录

本文档用于记录Transformer模型微调实验的所有配置、参数和结果。

---

## 实验汇总表

| 实验ID | 日期 | 数据集 | 预训练 | Test MAE | 备注 |
|--------|------|--------|--------|----------|------|
| EXP-001 | 2025-12-06 | hMOF_CO2_0.5 | ✓ | - | 基线实验 |
| EXP-002 | - | - | - | - | - |
| EXP-003 | - | - | - | - | - |

---

## 实验详细记录

### 实验 EXP-001 (模板)

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

| 实验ID | Batch Size | 学习率 | Validation MAE | Test MAE | 备注 |
|--------|------------|--------|----------------|----------|------|
| EXP-BS-02 | 128 | 5e-5 | - | - | - |

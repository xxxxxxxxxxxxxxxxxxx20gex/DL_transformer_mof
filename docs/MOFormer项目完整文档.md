<!-- pip install torch torchvision torchaudio
pip install pymatgen  # 用于处理晶体结构
pip install transformers  # Transformer模型
pip install tensorboard  # 训练监控
pip install ase  # 原子模拟环境 -->

# MOFormer项目完整文档

## 项目概述

**MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction**

这是一个基于Transformer的金属有机框架(MOF)性质预测项目，采用自监督学习框架，结合结构无关的序列表示和基于结构的图表示。

## 项目结构

```
MOFormer/
├── pretrain_SSL.py              # 自监督预训练脚本
├── pretrain.py                  # 传统预训练脚本（MLM）
├── finetune_transformer.py      # Transformer微调脚本
├── finetune_cgcnn.py           # CGCNN微调脚本
├── config_multiview.yaml        # SSL预训练配置文件
├── config_ft_transformer.yaml   # Transformer微调配置
├── config_ft_cgcnn.yaml        # CGCNN微调配置
├── cif_toy/                     # 测试数据集（100个CIF文件）
├── cif/                         # 完整数据集（CIF文件）
├── benchmark_datasets/          # 微调数据集
├── ckpt/                        # 模型权重
│   ├── pretraining/            # 预训练模型
│   └── finetuning/             # 微调模型
├── dataset/                     # 数据集处理模块
├── model/                       # 模型定义
├── tokenizer/                   # 分词器
└── loss/                        # 损失函数
```

## 训练流程

### 1. 预训练阶段（Pre-training）

#### 1.1 自监督预训练（SSL）
**脚本**: `pretrain_SSL.py`
**配置文件**: `config_multiview.yaml`

**训练目标**：
- 多视图自监督学习
- 对齐Transformer序列表示和CGCNN图表示
- 使用Barlow Twins Loss

**数据要求**：
```
cif/
├── *.cif                    # CIF结构文件
├── id_prop.npy             # [cif_id, mofid]对应关系
└── atom_init.json          # 原子特征向量（从benchmark_datasets复制）
```

**模型架构**：
- **Transformer**: 处理MOFid字符串序列
- **CGCNN**: 处理晶体结构图
- **损失函数**: Barlow Twins Loss（对比学习）

**运行命令**：
```bash
python pretrain_SSL.py
```

#### 1.2 传统预训练（MLM）
**脚本**: `pretrain.py`

**训练目标**：
- 掩码语言模型（Masked Language Modeling）
- 预测被掩盖的MOFid token

**数据要求**：
- `data/large_pretrain_512.npy`（MOFid序列数据）

### 2. 微调阶段（Fine-tuning）

#### 2.1 Transformer微调
**脚本**: `finetune_transformer.py`
**配置文件**: `config_ft_transformer.yaml`

**训练目标**：
- 在预训练Transformer基础上进行监督学习
- 预测MOF的特定属性（如CO2吸附量）

**数据要求**：
```
benchmark_datasets/
└── QMOF/
    └── mofid/
        └── QMOF_small_mofid.csv  # [mofid, property_value]
```

**运行命令**：
```bash
python finetune_transformer.py
```

#### 2.2 CGCNN微调
**脚本**: `finetune_cgcnn.py`
**配置文件**: `config_ft_cgcnn.yaml`

**训练目标**：
- 在预训练CGCNN基础上进行监督学习
- 预测MOF的特定属性

**数据要求**：
```
benchmark_datasets/
└── QMOF/
    ├── graph/
    │   └── large/
    │       └── QMOF_large_graph.csv  # 预处理的结构图数据
    └── QMOF_cg/                      # CIF文件目录
```

**运行命令**：
```bash
python finetune_cgcnn.py
```

## 文件作用详解

### 核心脚本文件

#### 1. `pretrain_SSL.py`
- **作用**: 自监督多视图预训练
- **输入**: CIF文件 + id_prop.npy
- **输出**: 预训练模型权重
- **特点**: 同时训练Transformer和CGCNN，使用对比学习

#### 2. `pretrain.py`
- **作用**: 传统MLM预训练
- **输入**: MOFid序列数据
- **输出**: 预训练Transformer权重
- **特点**: 只训练Transformer，使用掩码语言模型

#### 3. `finetune_transformer.py`
- **作用**: Transformer微调
- **输入**: 预训练权重 + 属性数据
- **输出**: 微调后的Transformer模型
- **特点**: 监督学习，预测特定属性

#### 4. `finetune_cgcnn.py`
- **作用**: CGCNN微调
- **输入**: 预训练权重 + 结构图数据
- **输出**: 微调后的CGCNN模型
- **特点**: 监督学习，预测特定属性

### 配置文件

#### 1. `config_multiview.yaml`
```yaml
# SSL预训练配置
batch_size: 32
epochs: 1
graph_dataset:
  root_dir: cif_toy          # CIF文件目录
  max_num_nbr: 12
  radius: 8
barlow_loss:
  embed_size: 512
  lambd: 0.0051
```

#### 2. `config_ft_transformer.yaml`
```yaml
# Transformer微调配置
batch_size: 64
epochs: 200
fine_tune_from: ./ckpt/pretraining
dataset:
  dataPath: './benchmark_datasets/QMOF/mofid/QMOF_small_mofid.csv'
```

#### 3. `config_ft_cgcnn.yaml`
```yaml
# CGCNN微调配置
batch_size: 128
epochs: 60
dataset:
  root_dir: ./QMOF_cg
  label_dir: ./benchmark_datasets/QMOF/graph/large/QMOF_large_graph.csv
```

### 数据文件

#### 1. CIF文件（`.cif`）
- **作用**: 存储MOF的晶体结构信息
- **格式**: 标准CIF格式
- **用途**: 预训练阶段的结构数据源

#### 2. `id_prop.npy`
- **作用**: 存储CIF文件ID和对应MOFid的映射关系
- **格式**: numpy数组，形状为(N, 2)
- **内容**: `[[cif_id1, mofid1], [cif_id2, mofid2], ...]`
- **用途**: 预训练阶段的数据索引

#### 3. `atom_init.json`
- **作用**: 存储每个化学元素的特征向量
- **格式**: JSON文件，键为原子序数，值为92维特征向量
- **内容**: `{"1": [0,1,0,...], "2": [0,0,0,...], ...}`
- **用途**: CGCNN模型的原子特征初始化

#### 4. 预处理数据文件
- **graph/**: 已构建的晶体图数据（CSV格式）
- **mofid/**: MOFid字符串和属性值（CSV格式）

### 模型权重文件

#### 1. 预训练模型（`ckpt/pretraining/`）
- `model_transformer_14.pth`: Transformer预训练权重
- `model_graph_14.pth`: CGCNN预训练权重
- **用途**: 微调阶段的初始化权重

#### 2. 微调模型（`ckpt/finetuning/`）
- `Trans_CGCNN_hMOF_CO2_0.5_1/model.pth`: 特定任务的微调模型
- `Trans_CGCNN_QMOF_1/model.pth`: 特定任务的微调模型
- **用途**: 最终可部署的模型

## 数据关系详解

### 数据流程

```
原始数据 → 预训练 → 微调 → 部署
   ↓         ↓       ↓      ↓
CIF文件   SSL/MLM   监督学习  预测模型
   ↓         ↓       ↓      ↓
id_prop.npy  预训练权重  微调权重  最终模型
   ↓         ↓       ↓      ↓
atom_init.json  benchmark_datasets  特定任务
```

### 文件夹关系

#### 1. `cif/` vs `benchmark_datasets/`

| 特征 | cif文件夹 | benchmark_datasets |
|------|-----------|-------------------|
| **数据格式** | 原始CIF文件 | 预处理数据 |
| **用途** | 预训练 | 微调 |
| **处理方式** | 实时解析 | 直接加载 |
| **数据量** | 完整数据集 | 特定任务子集 |
| **标签** | 无标签 | 有属性标签 |

#### 2. 数据转换关系

```
cif/ → 预训练 → benchmark_datasets/
├── *.cif     ├── 解析结构     ├── graph/
├── id_prop.npy ├── 构建图       └── mofid/
└── atom_init.json ├── 生成MOFid
```

## 运行指南

### 完整训练流程

#### 1. 环境准备
```bash
conda create -n moformer python=3.9
conda activate moformer
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install --channel conda-forge pymatgen
pip install transformers
conda install -c conda-forge tensorboard
```

#### 2. 数据准备
```bash
# 复制原子特征文件
cp benchmark_datasets/atom_init.json cif/

# 生成id_prop.npy（如果使用自己的CIF文件）
python generate_id_prop.py
```

#### 3. 预训练
```bash
# SSL预训练
python pretrain_SSL.py

要进行SSL（自监督学习）预训练，您需要准备以下几个方面的内容：
数据文件准备
1. CIF结构文件数据集
位置: 创建一个目录（如cif/或使用现有的cif_toy/）
内容: 包含您要训练的所有MOF的CIF格式结构文件
格式: *.cif 文件，每个文件包含一个MOF的晶体结构信息
2. id_prop.npy映射文件 ⭐ 关键文件
作用: 建立CIF文件名到MOFid字符串的映射关系
格式: numpy数组，每行包含 [CIF文件名, MOFid字符串]
生成方法:
使用项目中的generate_id_prop.py脚本
需要先安装和配置MOFid工具
示例格式：[['hMOF-1000268', 'Zn.O2C&&O4C2&&cds'], ...]
3. atom_init.json原子特征文件
位置: 直接复制benchmark_datasets/atom_init.json到数据目录
作用: 提供每种元素的初始特征向量
内容: JSON格式，映射原子序数到特征向量
```

#### 4. 微调
```bash
# Transformer微调
python finetune_transformer.py

# CGCNN微调
python finetune_cgcnn.py
```

### 快速测试

使用`cif_toy`数据集进行快速测试：
```bash
# 修改config_multiview.yaml
graph_dataset:
  root_dir: cif_toy

# 运行SSL预训练
python pretrain_SSL.py
```

## 关键概念

### 1. 自监督学习（SSL）
- **目标**: 学习通用的MOF表示
- **方法**: 多视图对比学习
- **优势**: 无需标签，利用结构信息

### 2. 多视图学习
- **视图1**: MOFid字符串序列（Transformer）
- **视图2**: 晶体结构图（CGCNN）
- **对齐**: 使用Barlow Twins Loss

### 3. 预训练-微调范式
- **预训练**: 在大规模无标签数据上学习通用表示
- **微调**: 在特定任务数据上适应到具体应用

## 注意事项

1. **数据路径**: 确保配置文件中的路径正确
2. **GPU内存**: 根据GPU内存调整batch_size
3. **数据格式**: 确保CIF文件格式正确
4. **依赖安装**: 确保所有依赖包正确安装
5. **文件权限**: 确保有读写权限

## 常见问题

### Q1: 如何生成id_prop.npy文件？
A: 使用`generate_id_prop.py`脚本，需要安装mofid工具。

### Q2: atom_init.json文件从哪里来？
A: 直接使用`benchmark_datasets/atom_init.json`，这是通用的原子特征文件。

### Q3: 预训练和微调的区别？
A: 预训练是无监督学习，微调是监督学习。预训练学习通用表示，微调适应特定任务。

### Q4: 如何选择数据集？
A: 预训练使用cif文件夹，微调使用benchmark_datasets中的特定任务数据。

---

**文档版本**: 1.0  
**最后更新**: 2024年  
**项目地址**: https://github.com/zcao0420/MOFormer 
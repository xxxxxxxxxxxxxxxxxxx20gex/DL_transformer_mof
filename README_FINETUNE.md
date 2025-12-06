# MOFormer - Transformer微调分支

本分支专门用于Transformer模型的微调（Fine-tuning），已移除所有预训练和CGCNN相关代码，仅保留微调所需的核心功能。

## 项目结构

```
DL_transformer_mof/
├── finetune_transformer.py          # Transformer微调主脚本
├── config_ft_transformer.yaml       # 微调配置文件
├── requirements.txt                 # Python依赖
├── README.md                        # 原始项目README
├── README_FINETUNE.md              # 本文件
├── model/
│   ├── transformer.py              # Transformer模型定义（简化版）
│   └── utils.py                    # 工具函数（Normalizer, AverageMeter等）
├── dataset/
│   └── dataset_finetune_transformer.py  # 微调数据集（仅MOF_ID_Dataset）
├── tokenizer/
│   ├── mof_tokenizer.py           # MOFid分词器
│   ├── vocab_full.txt             # 词汇表
│   └── RCSR_topologies.npy        # 拓扑数据
├── loss/                           # 损失函数（如需要）
├── ckpt/                           # 预训练权重目录
│   └── pretraining/               # 用于加载预训练权重
└── benchmark_datasets/             # 微调数据集
    ├── hMOF/mofid/                # hMOF数据
    └── QMOF/mofid/                # QMOF数据
```

## 重构说明

### 已删除的内容

1. **预训练脚本**：
   - `pretrain_SSL.py` - 自监督预训练
   - `pretrain.py` - MLM预训练

2. **CGCNN相关**：
   - `finetune_cgcnn.py`
   - `config_ft_cgcnn.yaml`
   - `model/cgcnn_finetune.py`
   - `model/cgcnn_pretrain.py`
   - `dataset/dataset_finetune_cgcnn.py`

3. **其他模型和模块**：
   - `model/GSOP.py` - GSoP注意力模块
   - `model/MPNCOV.py` - MPNCOV模块
   - `model/soap.py` - SOAP模型
   - `model/mlm_pytorch.py` 和 `model/mlm_pytorch_new.py` - MLM模型

4. **不相关数据集**：
   - `dataset/dataset_multiview.py` - 多视图数据集
   - `dataset/dataset_mlm.py` - MLM数据集
   - `dataset/dataset_finetune_soap.py` - SOAP数据集
   - `dataset/augmentation.py` - 数据增强

5. **多余配置文件**：
   - `config_multiview.yaml`
   - `config_ft_cgcnn.yaml`

### 保留的核心功能

- ✅ Transformer模型（简化版，移除预训练类）
- ✅ TransformerRegressor（微调模型）
- ✅ 预训练权重加载功能
- ✅ MOF_ID_Dataset数据集
- ✅ 完整的训练、验证、测试流程
- ✅ 差异化学习率设置（新层 vs 预训练层）

## 使用方法

### 1. 环境配置

```bash
conda create -n moformer python=3.9
conda activate moformer
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers tensorboard pymatgen
pip install --ignore-installed ruamel.yaml
```

### 2. 准备数据

数据格式为CSV文件，包含两列：
- 第一列：MOFid文本（SMILES&&拓扑）
- 第二列：数值标签（回归目标）

示例：
```
SMILES&&topology,property_value
C1=CC=CC=C1&&pcu,2.45
...
```

### 3. 配置文件

编辑 `config_ft_transformer.yaml`：

```yaml
batch_size: 128
epochs: 30
fine_tune_from: ./ckpt/pretraining  # 预训练权重路径，设为'scratch'则从头训练
gpu: cuda:0                          # 使用的GPU设备

dataset:
  data_name: 'hMOF_CO2_0.5'
  dataPath: './benchmark_datasets/hMOF/mofid/hMOF_CO2_0.5_small_mofid.csv'

Transformer:
  ntoken: 4021    # 词汇表大小
  d_model: 512    # 模型维度
  nhead: 8        # 注意力头数
  nlayers: 6      # Transformer层数
```

### 4. 运行微调

```bash
# 使用默认随机种子（seed=1）
python finetune_transformer.py

# 指定随机种子
python finetune_transformer.py --seed 42
```

### 5. 查看训练日志

```bash
tensorboard --logdir training_results/finetuning/Transformer
```

## 模型架构

```
Input (MOFid text)
    ↓
Tokenizer (转换为token ids)
    ↓
Transformer Encoder (6层)
    ├── Token Embedding
    ├── Positional Encoding
    └── Multi-Head Self-Attention × 6
    ↓
CLS Token Extraction (取首位token特征)
    ↓
Regression Head (4层全连接网络)
    ├── Linear(512 → 256) + ReLU
    ├── Linear(256 → 128) + ReLU
    ├── Linear(128 → 64) + ReLU
    └── Linear(64 → 1)
    ↓
Output (回归预测值)
```

## 训练特性

1. **预训练权重加载**：支持从预训练模型初始化
2. **差异化学习率**：
   - 预训练层：较小学习率（base_lr × 1）
   - 新增回归头：较大学习率（base_lr × 200）
3. **标签标准化**：训练时标准化，评估时反标准化
4. **自动保存最佳模型**：基于验证集MAE
5. **完整日志记录**：TensorBoard + 文件日志

## 输出结果

训练完成后，结果保存在 `training_results/finetuning/Transformer/` 目录：

```
Trans_{method}_{task}_{seed}_{timestamp}/
├── checkpoints/
│   ├── model.pth                    # 最佳模型权重
│   └── config_ft_transformer.yaml   # 配置文件备份
├── test_results.csv                 # 测试集预测结果
├── training.log                     # 训练日志
└── events.out.tfevents.*           # TensorBoard日志
```

## 常见问题

### Q: 如何从头训练（不使用预训练权重）？
A: 在 `config_ft_transformer.yaml` 中设置 `fine_tune_from: 'scratch'`

### Q: 如何修改模型大小？
A: 编辑配置文件中的 `Transformer` 部分：
- `d_model`: 特征维度
- `nhead`: 注意力头数（必须整除 d_model）
- `nlayers`: Transformer层数

### Q: 训练速度太慢怎么办？
A: 
- 增加 `batch_size`
- 减少 `num_workers`（如果IO不是瓶颈）
- 使用更快的GPU

### Q: 内存不足？
A: 
- 减小 `batch_size`
- 减少模型参数（`d_model`, `nlayers`）

## 引用

如果使用本代码，请引用原始论文：

```bibtex
@article{cao2023moformer,
  title={MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction},
  author={Cao, Zhonglin and Magar, Rishikesh and Wang, Yuyang and Barati Farimani, Amir},
  journal={Journal of the American Chemical Society},
  year={2023},
  publisher={ACS Publications}
}
```

## 技术支持

如有问题，请参考：
- 原始项目：https://github.com/zcao0420/MOFormer
- 论文：https://pubs.acs.org/doi/10.1021/jacs.2c11420


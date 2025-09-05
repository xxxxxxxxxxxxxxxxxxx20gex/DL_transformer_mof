### Transformer 微调（教学友好版）快速上手

本指南配套以下文件：
- `transformer_tutorial.py`：batch_first 版本的 Transformer 与回归头，中文注释
- `finetune_transformer_tutorial.py`：训练/验证/测试脚本（细分步骤+中文注释）
 
#### 一、环境准备
- 需要安装 PyTorch、transformers（为分词器）、tensorboard

#### 二、数据与配置
- 配置文件示例：`config_ft_transformer.yaml`
  - `dataset.dataPath` 为 CSV，第一列是 MOFid 文本（形如 `SMILES&&拓扑`），第二列是标签（浮点数）
  - `vocab_path` 指向 `tokenizer/vocab_full.txt`

#### 三、运行训练（
```bash
python finetune_transformer.py 
```
输出目录：`training_results/finetuning/TransformerTutorial/...`

#### 四、代码整体结构
- 数据管线：读取 CSV -> 使用 `MOFTokenizer` 编码 -> 构造 `MOF_ID_Dataset` -> `DataLoader`
- 模型：`TransformerTutorial`（Embedding + PositionalEncodingBatchFirst + TransformerEncoder）
- 下游：`TransformerRegressorTutorial` 取序列首位特征（类似 [CLS]）进入回归头
- 训练：MSELoss，评估 MAE，保存最佳权重

#### 五、常见问题
- 若出现位置维度不匹配，请确认 batch_first 版本位置编码是否生效
- 若显存不足，适当调小 `batch_size` 或 `nlayers`/`d_model`


# 使用 TensorBoard
tensorboard --logdir training_results/finetuning/Transformer

# 或查看文本日志
python check_logs.py



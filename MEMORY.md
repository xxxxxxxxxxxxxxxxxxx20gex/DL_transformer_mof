# 项目备忘

## 基准

- **任务**：hMOF，CO₂ 吸附 **0.5 bar**，指标为测试集 **MAE（mol/kg）**。
- **论文表**：MOFormer pretrain 约 **0.545**；本仓库曾跑通约 **0.493**（30 epoch 微调、multiview 预训练权重，具体见 `docs/record.md`）。

## 预训练（SSL）

- **结构**：MOFid → Transformer，CIF → CGCNN，两路对齐同一批 MOF。
- **损失**：已从 **Barlow Twins** 换成 **CLIP 式对称 InfoNCE**（`loss/clip_loss.py`，温度在 `config_multiview.yaml` 的 `clip_loss.temperature`）。老实现仍在 `loss/barlow_twins.py`。
- **注意**：换损失后需 **重新跑 SSL** 再微调；旧 checkpoint 与新版目标不一致。
- **实验输出**：`exp/SSL/`（目录一般在 `.gitignore` 里）。

## 实验记录

- **id1**：切换 SSL 损失为 **CLIP 式对称 InfoNCE**，预训练 checkpoint 使用 `exp/SSL/runs_multiview/Apr19_17-11-52/checkpoints/best_transformer_model.pth`。
- **下游任务**：`hMOF_CO2_0.5`，Transformer 微调 30 epoch，`seed=42` 测试集 **MAE = 0.4927**。
- **对照观察**：与旧的 Barlow 预训练结果基本持平，当前结论是 **InfoNCE 至少不劣于 Barlow**，后续应以多 seed 均值继续比较。

## 常用路径
- 预训练：`pretrain_SSL.py`、`config_multiview.yaml`
- 微调：`finetune_transformer.py` 与对应 yaml
- 论文数值摘录：`docs/MOF_experiments_reproduction.md`

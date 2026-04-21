# 项目备忘

## 基准

- **任务**：hMOF，CO₂ 吸附 **0.5 bar**，指标为测试集 **MAE（mol/kg）**。
- **论文表**：MOFormer pretrain 约 **0.545**；本仓库当前已跑到 **0.485**（Barlow baseline，60 epoch）与 **0.491**（InfoNCE，60 epoch），具体见 `RECORD.md`。

## 预训练（SSL）

- **结构**：MOFid → Transformer，CIF → CGCNN，两路对齐同一批 MOF。
- **损失**：已从 **Barlow Twins** 换成 **CLIP 式对称 InfoNCE**（`loss/clip_loss.py`，温度在 `config_multiview.yaml` 的 `clip_loss.temperature`）。老实现仍在 `loss/barlow_twins.py`。
- **注意**：换损失后需 **重新跑 SSL** 再微调；旧 checkpoint 与新版目标不一致。
- **实验输出**：`exp/SSL/`（目录一般在 `.gitignore` 里）。

## 实验记录

- **总表路径**：论文基准与本仓库实验台账统一维护在 `RECORD.md`。
- **维护约定**：以后只要有新的 SSL、微调或对比实验结果，都应同步更新 `RECORD.md`，包括任务、初始化方式、epochs、best valid MAE、test MAE 和当前状态（done/stopped/running）。
- **id1**：切换 SSL 损失为 **CLIP 式对称 InfoNCE**，预训练 checkpoint 使用 `exp/SSL/runs_multiview/Apr19_17-11-52/checkpoints/best_transformer_model.pth`。
- **下游任务**：`hMOF_CO2_0.5` 当前主比较口径已统一为 **60 epoch**。已完成结果为：Barlow baseline **0.485**，InfoNCE **0.491**。
- **对照观察**：30 epoch 下两者几乎持平（**0.4929 vs 0.4927**）；60 epoch、同 seed 的单次对比里，**Barlow 略优于 InfoNCE**（**0.485 vs 0.491**）。
- **旧设置说明**：`200 epoch` 的 InfoNCE 微调在 `epoch 59` 出现 best valid **0.477**，但后续未形成稳定收益，日志停在 `epoch 137`。由于后续实验协议统一改为 **60 epoch**，该设置不再继续，也不再补 test。

## 常用路径
- 预训练：`pretrain_SSL.py`、`config_multiview.yaml`
- 微调：`finetune_transformer.py` 与对应 yaml
- 实验记录总表：`/home/DL_transformer_mof/RECORD.md`
- 论文解析：`/home/DL_transformer_mof/docs/论文解析.md`

## 环境
- 虚拟环境：`mof`

## 待完成事项
- 若推进创新线：在 SSL 侧先做 **Q-Former 桥接原型**（最小实现：query + cross-attn + 对称 InfoNCE）。
- 每完成一次实验后，立即更新 `RECORD.md`（任务、初始化、epoch、best valid、test、状态）。

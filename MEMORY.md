# 项目备忘

## 文件分工

- `MEMORY.md`：记录**当前状态、最新实验进展、运行中的任务、近期结论与下一步计划**。
- `AGENTS.md`：记录**长期稳定的协作规则、实验规范、关键入口与操作约定**。
- `RECORD.md`：记录**实验结果总表与对比结论**。

## 基准

- **任务**：hMOF，CO₂ 吸附 **0.5 bar**，指标为测试集 **MAE（mol/kg）**。
- **论文表对比**：在 `hMOF CO2 0.5 bar` 上，论文给出的 **MOFormer scratch = 0.558 ± 0.001**、**MOFormer pretrain = 0.545 ± 0.008**；本仓库当前最好结果已到 **0.454**（Q-Former + InfoNCE，60 epoch）。相较论文 **MOFormer pretrain** 绝对下降 **0.091**、相对下降约 **16.7%**；相较论文 **MOFormer scratch** 绝对下降 **0.104**、相对下降约 **18.6%**。具体实验台账见 `RECORD.md`。

## 预训练（SSL）

- **结构**：MOFid → Transformer，CIF → CGCNN，两路对齐同一批 MOF。
- **损失**：已从 **Barlow Twins** 换成 **CLIP 式对称 InfoNCE**（`loss/clip_loss.py`，温度在 `config_multiview.yaml` 的 `clip_loss.temperature`）。老实现仍在 `loss/barlow_twins.py`。
- **注意**：换损失后需 **重新跑 SSL** 再微调；旧 checkpoint 与新版目标不一致。
- **创新线当前进展**：已在 `infonce-qformer` 分支接入 **Q-Former 桥接原型**。当前实现为：Transformer 返回整段序列特征，`model/qformer.py` 中的 learnable queries 对文本 hidden states 做 cross-attn，再聚合为文本侧对比向量，与图侧继续做对称 InfoNCE。
- **实现状态**：已修正 `model/transformer.py` 中 batch-first 位置编码；已完成 `Transformer sequence -> QFormerBridge -> contrastive embedding` 的导入和 shape 验证；`pretrain_SSL.py` 与 `config_multiview.yaml` 已接入 `qformer.enabled` 开关；`dataset/dataset_multiview.py` 已修复 `num_workers=0` 时 `prefetch_factor` 的兼容性问题。
- **当前运行状态**：Q-Former 版 SSL sanity run 已于 **2026-04-21 16:48** 重新在后台启动；PID=`3163844`；输出目录为 `exp/SSL/runs_multiview/Apr21_16-48-06`；外层启动日志为 `exp/SSL/launch_logs/qformer_ssl_20260421_164804.log`。当前配置为 `qformer.enabled=true`、`fine_tune_from=None`、`log_every_n_steps=10`，即从头开始预训练并提高日志频率以便观察。
- **输出清理**：已删除无价值且未跑完的调试/残留目录 `Apr21_16-39-04`、`Apr21_16-44-44`、`Apr21_16-47-35`，以及旧启动日志 `qformer_ssl_20260421_163902.log`；当前仅保留有效历史 run 与正在运行的 `Apr21_16-48-06`。
- **运行经验**：当前 Q-Former 训练在 `tmux attach -t mof` 中前台运行更稳定；此前通过 `nohup`/后台排查配合宽匹配 `pkill -f 'python pretrain_SSL.py'` 的方式，存在误杀正在运行训练进程的风险。后续长训练优先使用 `tmux`，停止任务时必须按 **PID** 精确处理，不再使用宽匹配 `pkill`。
- **实验输出**：`exp/SSL/`（目录一般在 `.gitignore` 里）。

## 实验记录

- **总表路径**：论文基准与本仓库实验台账统一维护在 `RECORD.md`。
- **维护约定**：以后只要有新的 SSL、微调或对比实验结果，都应同步更新 `RECORD.md`，包括任务、初始化方式、epochs、best valid MAE、test MAE 和当前状态（done/stopped/running）。
- **id1**：切换 SSL 损失为 **CLIP 式对称 InfoNCE**，预训练 checkpoint 使用 `exp/SSL/runs_multiview/Apr19_17-11-52/checkpoints/best_transformer_model.pth`。
- **下游任务**：`hMOF_CO2_0.5` 当前主比较口径已统一为 **60 epoch**。已完成结果为：Barlow baseline **0.485**，InfoNCE **0.491**，Q-Former + InfoNCE **0.454**。
- **对照观察**：30 epoch 下两者几乎持平（**0.4929 vs 0.4927**）；60 epoch、同 seed 的单次对比里，排序为 **Q-Former + InfoNCE (0.454) < Barlow baseline (0.485) < InfoNCE (0.491)**。
- **与论文对比**：Q-Former + InfoNCE 当前测试 **0.454**，已经明显优于论文中 `hMOF CO2 0.5 bar` 的 **MOFormer pretrain 0.545 ± 0.008** 与 **MOFormer scratch 0.558 ± 0.001**。
- **旧设置说明**：`200 epoch` 的 InfoNCE 微调在 `epoch 59` 出现 best valid **0.477**，但后续未形成稳定收益，日志停在 `epoch 137`。由于后续实验协议统一改为 **60 epoch**，该设置不再继续，也不再补 test。

## 常用路径
- 预训练：`pretrain_SSL.py`、`config_multiview.yaml`
- 协作说明：`/home/DL_transformer_mof/AGENTS.md`
- 微调：`finetune_transformer.py` 与对应 yaml
- 实验记录总表：`/home/DL_transformer_mof/RECORD.md`
- 论文解析：`/home/DL_transformer_mof/docs/论文解析.md`

## 环境
- 虚拟环境：`mof`

## 待完成事项
- q-former跑完实验后更新实验记录

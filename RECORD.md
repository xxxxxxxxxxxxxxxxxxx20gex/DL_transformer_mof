# MOFormer 论文基准与本仓库实验记录

## 论文基准任务与协议

- 指标统一为 **MAE**。
- QMOF 任务单位为 **eV**。
- hMOF 气体吸附任务单位为 **mol kg⁻¹**。
- 论文中下游数据划分为 **train / valid / test = 0.7 / 0.15 / 0.15**。
- 论文中 SSL 预训练完成后，下游微调使用预训练权重并训练 **200 epochs**。
- MOFormer 可用样本数少于结构模型，因为只使用有 MOFid 的子集。

## 论文基准（MAE）

### QMOF 带隙（eV）

| 模型 | MAE |
|--|--|
| CGCNN scratch | 0.275 ± 0.015 |
| CGCNN pretrain | 0.256 ± 0.006 |
| SOAP | 0.424 ± 0.007 |
| MOFormer scratch | 0.387 ± 0.001 |
| MOFormer pretrain | 0.367 ± 0.005 |
| Stoichiometric-120 | 0.466 ± 0.011 |
| RACs | 0.441 ± 0.008 |

### hMOF 气体吸附（mol kg⁻¹）

| 模型 | CO₂ 0.05 bar | CO₂ 0.5 bar | CO₂ 2.5 bar | CH₄ 0.05 bar | CH₄ 0.5 bar | CH₄ 2.5 bar |
|--|--|--|--|--|--|--|
| CGCNN scratch | 0.126 ± 0.005 | 0.391 ± 0.017 | 0.818 ± 0.050 | 0.028 ± 0.001 | 0.121 ± 0.006 | 0.333 ± 0.017 |
| CGCNN pretrain | 0.110 ± 0.001 | 0.330 ± 0.002 | 0.645 ± 0.003 | 0.025 ± 0.001 | 0.099 ± 0.001 | 0.258 ± 0.008 |
| SOAP | 0.115 ± 0.002 | 0.339 ± 0.004 | 0.666 ± 0.003 | 0.022 ± 0.001 | 0.106 ± 0.001 | 0.239 ± 0.002 |
| MOFormer scratch | 0.178 ± 0.002 | 0.558 ± 0.001 | 1.000 ± 0.013 | 0.034 ± 0.000 | 0.174 ± 0.002 | 0.385 ± 0.003 |
| MOFormer pretrain | 0.158 ± 0.001 | 0.545 ± 0.008 | 0.982 ± 0.011 | 0.033 ± 0.000 | 0.161 ± 0.011 | 0.384 ± 0.003 |
| Stoichiometric-120 | 0.282 ± 0.002 | 0.983 ± 0.005 | 1.895 ± 0.003 | 0.050 ± 0.001 | 0.269 ± 0.001 | 0.631 ± 0.002 |
| RACs | 0.248 ± 0.002 | 0.842 ± 0.004 | 1.681 ± 0.004 | 0.044 ± 0.001 | 0.236 ± 0.002 | 0.570 ± 0.004 |

## 论文补充结论

- QMOF 上，MOFormer pretrain 相比 MOFormer scratch 从 **0.387** 降到 **0.367**，相对下降约 **5.34%**。
- hMOF 六个吸附任务上，论文给出的结论是 MOFormer 预训练平均提升约 **4.3%**，CGCNN 预训练平均提升约 **16.5%**。
- 对当前仓库重点关注的 `hMOF CO₂ 0.5 bar`，论文中 **MOFormer pretrain = 0.545 ± 0.008**，**MOFormer scratch = 0.558 ± 0.001**。

## 本仓库近期实验

### 已结束

| ID | 状态 | 任务 | SSL/初始化 | 下游 epochs | seed | best valid MAE | test MAE | 备注 |
|--|--|--|--|--|--|--|--|--|
| exp-20260419-1 | done | hMOF CO₂ 0.5 bar | Barlow baseline, `Apr17_17-39-05` | 30 | 1 | 0.500 | 0.4929 | 载入 74 个预训练参数 |
| exp-20260420-1 | done | hMOF CO₂ 0.5 bar | InfoNCE, `Apr19_17-11-52` | 30 | 42 | 0.504 | 0.4927 | 载入 74 个预训练参数 |
| exp-20260420-2 | stopped | hMOF CO₂ 0.5 bar | InfoNCE, `Apr19_17-11-52` | 200 | 42 | 0.477 | 不补跑 | 输出 `exp/finetune/Trans_multiview_hMOF_CO2_0.5_42_2026-04-20_16-26-16`；best valid @ ep59；训练日志停在 epoch 137；由于后续统一改为 60 epoch，该 200 epoch 设置不再继续，也不再补 test |
| exp-20260421-1 | done | hMOF CO₂ 0.5 bar | Barlow baseline, `Apr17_17-39-05_baseline` | 60 | 42 | 0.492 | 0.485 | 输出 `exp/finetune/Trans_multiview_hMOF_CO2_0.5_42_2026-04-20_22-50-07`；best valid @ ep55；载入 74 个预训练参数 |
| exp-20260421-2 | done | hMOF CO₂ 0.5 bar | InfoNCE, `Apr19_17-11-52` | 60 | 42 | 0.497 | 0.491 | 输出 `exp/finetune/Trans_multiview_hMOF_CO2_0.5_42_2026-04-21_10-58-48`；best valid @ ep55；已跑完并完成 test；载入 74 个预训练参数 |

### 进行中

- 当前无进行中的 `hMOF CO₂ 0.5 bar` 微调实验。


## 当前观察

- **Barlow baseline、60 epoch、seed 42**（`exp-20260421-1`）验证最优 **0.492**、测试 **0.485**；**InfoNCE、60 epoch、seed 42**（`exp-20260421-2`）验证最优 **0.497**、测试 **0.491**。在当前这组单次对比里，Barlow 略优于 InfoNCE。
- 两组 **30 epoch** 微调结果几乎一致：**0.4929 vs 0.4927**，说明在较短训练下两种初始化差距很小。
- 当前更有参考价值的是 **同 seed、同 epochs 的 60 epoch 对比**；若要进一步下结论，仍建议看多个重复实验的均值与方差。
- **200 epoch InfoNCE** 在 `epoch 59` 达到当前记录中的最优 valid **0.477**，但之后继续训练没有形成稳定收益，且日志最终停在 `epoch 137`。由于后续实验协议已统一为 **60 epoch**，该设置不再继续，也不再补 test。

# AGENTS Guide

## Document Role

- 本文件用于记录 **长期稳定的协作规则、实验规范、关键入口与操作约定**。
- 会频繁变化的内容，例如当前运行中的 PID、具体 run 目录、最新实验进度、近期结论与下一步计划，统一写入 `MEMORY.md`。
- 实验结果汇总与对比表统一写入 `RECORD.md`。

## Project Scope

- 项目名称：`DL_transformer_mof`
- 主要任务：MOF 自监督预训练与下游性质预测，当前重点任务为 `hMOF CO2 0.5 bar`
- 当前主线：在 InfoNCE 版 SSL 基础上推进 Q-Former 桥接创新线

## Key Files

- 预训练入口：`pretrain_SSL.py`
- 预训练配置：`config_multiview.yaml`
- Transformer 文本塔：`model/transformer.py`
- Q-Former 原型：`model/qformer.py`
- 图塔：`model/cgcnn_pretrain.py`
- 对比损失：`loss/clip_loss.py`
- 实验总表：`RECORD.md`
- 项目记忆：`MEMORY.md`

## Environment
- conda 环境：`mof`
- 默认 GPU：`cuda:0`
- SSL 输出根目录：`exp/SSL/`
- 默认工作分支建议：`infonce-qformer`

## Experiment Conventions

- 当前下游比较口径统一按 `60 epochs` 记录和对比。
- `200 epoch` 的旧 InfoNCE 微调只作为历史记录，不再继续补跑。
- 做创新实验时，优先保持图塔、损失、数据和 batch size 不变，只改目标模块，便于控制变量。
- 若需要切换到 `num_workers=0` 做调试，当前 `dataset/dataset_multiview.py` 已兼容该模式，不再传递无效的 `prefetch_factor`。

## Memory Rules

- 每次有新的实验启动、停止、完成或结论变化，必须同步更新 `MEMORY.md`。
- 每次实验产出可记录结果后，必须同步更新 `RECORD.md`。
- 若路径、默认配置、主分支策略或当前主线发生变化，也必须更新 `MEMORY.md`。
- 若新增了新的协作约定、实验规范或关键入口文件，应同步更新本文件 `AGENTS.md`。

## Working Rules

- 修改训练逻辑前，先确认当前分支和后台任务状态，避免与正在运行的实验冲突。
- 新实验尽量写清楚：
  - 初始化方式
  - epochs
  - seed
  - 输出目录
  - 是否从头训练或加载旧 checkpoint
- 启动后台训练后，记录 PID、日志路径和 run 目录。
- 如果实验只是 sanity run，也要在 `MEMORY.md` 中注明它的目的和当前状态。
- 长训练优先在 `tmux` 会话中运行，例如 `tmux attach -t mof`，避免 `nohup`/脱离终端排查时造成状态不透明。
- 停止训练进程时必须按 **PID** 精确处理；不要使用宽匹配的 `pkill -f 'python pretrain_SSL.py'`，以免误杀正在运行的正式实验。

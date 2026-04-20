#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIP 式对称跨模态对比损失（InfoNCE）：同一样本两路 embedding 对齐，batch 内其它样本为负。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_a: [N, D]，如 MOFid / Transformer 向量
            z_b: [N, D]，如 CGCNN 图向量
        """
        z_a = F.normalize(z_a.float(), dim=-1)
        z_b = F.normalize(z_b.float(), dim=-1)
        logits = z_a @ z_b.T / self.temperature
        n = logits.size(0)
        labels = torch.arange(n, device=logits.device, dtype=torch.long)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) * 0.5

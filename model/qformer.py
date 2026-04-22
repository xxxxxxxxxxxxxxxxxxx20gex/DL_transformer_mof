"""Lightweight Q-Former style bridge for SSL pretraining.

The goal of this module is intentionally modest:
- start from a small set of learnable query tokens
- let queries cross-attend to text sequence features
- aggregate query outputs into a single contrastive embedding

This keeps the rest of the SSL training loop unchanged while making the
text-side representation extractor more expressive than a plain CLS readout.
"""

import torch
from torch import nn


class QFormerBlock(nn.Module):
    """A minimal query block: self-attn on queries, cross-attn to text, then MLP."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)
        self.norm_q3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        q_self, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm_q1(queries + self.dropout(q_self))

        q_cross, _ = self.cross_attn(
            queries,
            encoder_hidden_states,
            encoder_hidden_states,
            need_weights=False,
        )
        queries = self.norm_q2(queries + self.dropout(q_cross))

        queries = self.norm_q3(queries + self.dropout(self.mlp(queries)))
        return queries


class QFormerBridge(nn.Module):
    """Pool text sequence features with learnable queries and project to contrastive space."""

    def __init__(
        self,
        d_model: int,
        num_queries: int = 8,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                QFormerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_hidden_states.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_hidden_states)
        pooled = self.query_norm(queries).mean(dim=1)
        return self.proj_out(pooled)

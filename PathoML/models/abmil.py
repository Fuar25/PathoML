"""Shared ABMIL building blocks used by multiple pathology subsystems."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import Aggregator, Classifier


class FeatureEncoder(nn.Module):
    """将实例级别特征进行线性投影和激活的编码器。

    通过线性层将输入特征投影到指定的嵌入维度，
    再经过 GELU 激活和 Dropout 正则化。
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        """初始化特征编码器。

        Args:
            input_dim: 输入特征的维度（由基础模型决定，如 1536）。
            embed_dim: 投影后的嵌入维度（hidden_dim）。
            dropout: Dropout 比率。
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """对输入特征执行线性投影、激活和 Dropout。

        Args:
            features: 实例特征张量，形状为 (B, N, input_dim)，
                      B 为批次大小，N 为 patch 数量。

        Returns:
            编码后的特征张量，形状为 (B, N, embed_dim)。
        """
        # features: (B, N, C)
        return self.dropout(self.act(self.proj(features)))


class GatedAttention(Aggregator):
    """标准门控注意力（Gated Attention）MIL 聚合器。

    参考 Ilse et al. (2018) 的门控注意力机制，
    使用两个并行分支（tanh 和 sigmoid）的逐元素乘积计算注意力权重，
    对所有实例特征进行加权求和得到 Bag 级别嵌入。
    """

    def __init__(self, embed_dim: int, attn_dim: Optional[int], gated: bool, dropout: float) -> None:
        """初始化门控注意力聚合器。

        Args:
            embed_dim: 输入实例嵌入的维度。
            attn_dim: 注意力隐藏层维度。若为 None，默认取 embed_dim // 2（最小为 1）。
            gated: 若为 True，启用门控机制（sigmoid 分支）；否则使用标准注意力。
            dropout: 应用于注意力权重计算前的 Dropout 比率。
        """
        super().__init__()
        attn_dim = attn_dim or embed_dim // 2 or 1
        self.gated = gated
        self.attention_a = nn.Linear(embed_dim, attn_dim)
        self.attention_b = nn.Linear(embed_dim, attn_dim) if gated else None
        self.attention_c = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, instances: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对实例序列计算注意力权重并聚合为 Bag 嵌入。

        Args:
            instances: 实例嵌入张量，形状为 (B, N, embed_dim)。
            mask: 可选布尔掩码，形状为 (B, N)，False 表示填充位置，
                  其注意力 logit 将被设为 -inf。

        Returns:
            tuple:
                - bag_embeddings (Tensor): Bag 级别嵌入，形状为 (B, embed_dim)。
                - attention (Tensor): 归一化注意力权重，形状为 (B, N)。
        """
        # instances: (B, N, D)
        attn_a = torch.tanh(self.attention_a(instances))
        if self.gated and self.attention_b is not None:
            attn_b = torch.sigmoid(self.attention_b(instances))
            attn_a = attn_a * attn_b
        logits = self.attention_c(self.dropout(attn_a)).squeeze(-1)  # (B, N)
        self.last_logits = logits                                   # mask 前，供蒸馏取用
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        attention = torch.softmax(logits, dim=1)
        bag_embeddings = torch.bmm(attention.unsqueeze(1), instances).squeeze(1)
        return bag_embeddings, attention


class LinearClassifier(Classifier):
    """带可选 Dropout 的单层线性分类头。"""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float) -> None:
        """初始化线性分类头。

        Args:
            embed_dim: 输入 Bag 嵌入的维度。
            num_classes: 输出类别数（二分类时为 1，多分类时为实际类别数）。
            dropout: 应用于输入嵌入的 Dropout 比率。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, bag_embeddings: torch.Tensor) -> torch.Tensor:
        """对 Bag 嵌入应用 Dropout 并线性映射到 logits。

        Args:
            bag_embeddings: Bag 级别嵌入，形状为 (B, embed_dim)。

        Returns:
            分类 logits，形状为 (B, num_classes)。
        """
        return self.linear(self.dropout(bag_embeddings))


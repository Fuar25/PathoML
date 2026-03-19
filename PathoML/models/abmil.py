"""Attention-Based MIL (ABMIL) 模型实现及注册表注入。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import Aggregator, BaseMIL, Classifier, DataDict
from ..registry import register_model


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


@register_model('abmil')
class ABMIL(BaseMIL):
    """Attention-Based MIL 模型，由编码器、聚合器和分类头三阶段组成。

    实现 Ilse et al. (2018) 的 Attention-Based Deep Multiple Instance Learning。
    Pipeline: FeatureEncoder → GatedAttention → LinearClassifier

    支持通过 ``attach_external_impl`` 插入自定义的论文实现，
    在保持接口统一的前提下灵活替换内部逻辑。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        attention_dim: Optional[int] = None,
        gated: bool = True,
        encoder_dropout: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        external_impl: Optional[nn.Module] = None,
    ) -> None:
        """初始化 ABMIL 模型。

        Args:
            input_dim: 输入实例特征的维度（与基础模型一致，如 1536）。
            hidden_dim: 编码器输出和聚合器内部的嵌入维度。
            num_classes: 输出类别数（二分类时为 1）。
            dropout: 全局 Dropout 比率，同时应用于聚合器。
            attention_dim: 注意力隐藏层维度。若为 None，默认取 hidden_dim // 2。
            gated: 是否启用门控注意力机制。
            encoder_dropout: 编码器的 Dropout 比率。若为 None，使用全局 dropout。
            classifier_dropout: 分类头的 Dropout 比率。若为 None，使用全局 dropout。
            external_impl: 可选的外部实现模块。若提供，则 forward 将完全委托给它。
        """
        super().__init__()
        self.external_impl = external_impl
        enc_dropout = encoder_dropout if encoder_dropout is not None else dropout
        clf_dropout = classifier_dropout if classifier_dropout is not None else dropout

        self.encoder = FeatureEncoder(input_dim, hidden_dim, enc_dropout)
        self.aggregator = GatedAttention(hidden_dim, attention_dim, gated, dropout)
        self.classifier = LinearClassifier(hidden_dim, num_classes, clf_dropout)

    def attach_external_impl(self, module: nn.Module) -> None:
        """插入自定义的外部实现模块，用于替换默认的三阶段 Pipeline。

        Args:
            module: 自定义的 nn.Module 实现，其 forward 方法需与 BaseMIL 接口兼容。
        """
        self.external_impl = module

    def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
        """执行 ABMIL 前向传播，返回 logits、Bag 嵌入和注意力权重。

        若已通过 ``attach_external_impl`` 设置了外部实现，则直接委托调用。

        Args:
            data_dict: 输入字典，需包含：
                - 'features' (Tensor): 实例特征，形状为 (B, N, input_dim)。
                - 'mask' (Tensor, optional): 布尔掩码，形状为 (B, N)。

        Returns:
            包含以下键的输出字典：
                - 'logits' (Tensor): 分类 logits，形状为 (B, num_classes)。
                - 'bag_embeddings' (Tensor): Bag 嵌入，形状为 (B, hidden_dim)。
                - 'attention' (Tensor): 注意力权重，形状为 (B, N)。
        """
        if self.external_impl is not None:
            return self.external_impl(data_dict)

        features = data_dict['features']  # (B, N, C)
        mask = data_dict.get('mask')  # Optional (B, N) boolean mask
        encoded = self.encoder(features)
        bag_embeddings, attention = self.aggregator(encoded, mask)
        logits = self.classifier(bag_embeddings)

        return {
            'logits': logits,
            'bag_embeddings': bag_embeddings,
            'attention': attention,
        }

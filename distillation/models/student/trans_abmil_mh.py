"""StudentTransABMIL_MH: 多头 Cross-Attention Pooling 替代单头 GatedAttention。

架构:
  HE patches (B, N, patch_dim)
    → ResidualEncoder → (B, N, hidden_dim)
    → TransformerEncoder → (B, N, hidden_dim)
    → CrossAttentionPooling: learnable query + MultiheadAttention → (B, hidden_dim)
    → LinearClassifier → logits (B, 1)

与 StudentTransABMIL 唯一区别: 聚合层从单头 GatedAttention 换为多头 Cross-Attention。
多头自然缓解梯度稀释 — patch 只需被任一 head 关注即可获得梯度。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from PathoML.models.abmil import LinearClassifier
from .trans_abmil import ResidualEncoder


class CrossAttentionPooling(nn.Module):
  """可学习 query token + 多头交叉注意力聚合。

  query (1, D) 作为 Q，patch embeddings 作为 K/V。
  K 个 head 各自关注不同 pattern，concat 后投影回 D 维。
  """

  def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
    super().__init__()
    self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
    self.pool = nn.MultiheadAttention(
      hidden_dim, num_heads, dropout=dropout, batch_first=True,
    )
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(
    self, instances: torch.Tensor, mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      instances: (B, N, D) patch embeddings
      mask: (B, N) bool, True=valid, False=padding

    Returns:
      bag_embeddings: (B, D)
      attention: (B, N) — 多头平均 attention weights
    """
    B = instances.size(0)
    query = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
    # MultiheadAttention: key_padding_mask True=忽略，需取反
    pad_mask = ~mask if mask is not None else None
    bag, attn_weights = self.pool(
      query, instances, instances,
      key_padding_mask=pad_mask,
    )                                                      # (B, 1, D), (B, 1, N)
    bag = self.norm(bag.squeeze(1))                        # (B, D)
    attention = attn_weights.squeeze(1)                    # (B, N)
    return bag, attention


class StudentTransABMIL_MH(nn.Module):
  """多头聚合学生: ResidualEncoder → TransformerEncoder → CrossAttentionPooling → Classifier。"""

  def __init__(
    self,
    patch_dim: int = 1536,
    hidden_dim: int = 256,
    attention_dim: int = 128,
    dropout: float = 0.2,
    n_transformer_layers: int = 2,
    nhead: int = 4,
    pool_heads: int = 4,
    proj_dim: int | None = None,
  ) -> None:
    """
    Args:
      nhead: Transformer self-attention 头数。
      pool_heads: Cross-Attention Pooling 头数。
      attention_dim: 未使用（保持与 StudentTransABMIL 接口一致）。
    """
    super().__init__()
    self.encoder = ResidualEncoder(patch_dim, hidden_dim, dropout)
    # (1) Transformer: patch 间信息交互
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=hidden_dim,
      nhead=nhead,
      dim_feedforward=hidden_dim * 4,
      dropout=dropout,
      activation='gelu',
      batch_first=True,
      norm_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
    # (2) 多头聚合 + 分类
    self.aggregator = CrossAttentionPooling(hidden_dim, pool_heads, dropout)
    self.classifier = LinearClassifier(hidden_dim, 1, dropout)
    # (3) Projection head
    self.proj_head = nn.Linear(hidden_dim, proj_dim) if proj_dim else None

  def forward(self, data: dict) -> dict:
    patches = data['he_patches']                          # (B, N, patch_dim)
    mask = data.get('mask')                               # (B, N) or None
    encoded = self.encoder(patches)                       # (B, N, hidden_dim)
    pad_mask = ~mask if mask is not None else None
    encoded = self.transformer(encoded, src_key_padding_mask=pad_mask)
    bag_embeddings, attention = self.aggregator(encoded, mask=mask)
    logits = self.classifier(bag_embeddings)              # (B, 1)
    out = {'hidden': bag_embeddings, 'logits': logits, 'attention': attention,
           'encoded': encoded}
    if mask is not None:
      out['mask'] = mask
    if self.proj_head is not None:
      out['proj'] = self.proj_head(bag_embeddings)
      out['encoded_proj'] = self.proj_head(encoded)
    return out

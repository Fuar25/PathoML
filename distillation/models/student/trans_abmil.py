"""StudentTransABMIL: 增强型学生模型 — 残差 Encoder + Transformer + GatedAttention。

架构:
  HE patches (B, N, patch_dim)
    → ResidualEncoder: 2层残差MLP → (B, N, hidden_dim)
    → TransformerEncoder: L层 self-attention，patch间信息交互 → (B, N, hidden_dim)
    → GatedAttention(gated=True) → bag_embeddings (B, hidden_dim)
    → ProjectionHead: Linear(hidden_dim→proj_dim) → proj (B, proj_dim)   ← 对齐teacher
    → LinearClassifier: Dropout + Linear(hidden_dim→1) → logits (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from PathoML.models.abmil import GatedAttention, LinearClassifier


# ---------------------------------------------------------------------------
# (1) 残差 Encoder — 替代单层 FeatureEncoder
# ---------------------------------------------------------------------------

class ResidualEncoder(nn.Module):
  """2层残差MLP: proj→GELU→Dropout→Linear→GELU→Dropout + skip。

  第一层 Linear(patch_dim→hidden_dim) 完成维度变换，
  第二层 Linear(hidden_dim→hidden_dim) 在残差连接下加深特征变换。
  """

  def __init__(self, patch_dim: int, hidden_dim: int, dropout: float) -> None:
    super().__init__()
    # (1) 维度投影层
    self.proj = nn.Linear(patch_dim, hidden_dim)
    self.act1 = nn.GELU()
    self.drop1 = nn.Dropout(dropout)
    # (2) 残差变换层
    self.fc = nn.Linear(hidden_dim, hidden_dim)
    self.act2 = nn.GELU()
    self.drop2 = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: (B, N, patch_dim) → (B, N, hidden_dim)"""
    h = self.drop1(self.act1(self.proj(x)))     # (B, N, hidden_dim)
    h = h + self.drop2(self.act2(self.fc(h)))   # residual
    return self.norm(h)


# ---------------------------------------------------------------------------
# (2) StudentTransABMIL
# ---------------------------------------------------------------------------

class StudentTransABMIL(nn.Module):
  """增强型学生: ResidualEncoder → TransformerEncoder → GatedAttention → Classifier。"""

  def __init__(
    self,
    patch_dim: int = 1536,
    hidden_dim: int = 256,
    attention_dim: int = 128,
    dropout: float = 0.2,
    n_transformer_layers: int = 2,
    nhead: int = 4,
    proj_dim: int | None = None,
  ) -> None:
    """
    Args:
      n_transformer_layers: Transformer self-attention 层数。
      nhead: 多头注意力头数（需整除 hidden_dim）。
      proj_dim: 蒸馏 projection head 输出维度（对齐 teacher hidden_dim）。
                None 时不创建 projection head，'hidden' 直接用于 L_feat。
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
    # (2) 聚合 + 分类
    self.aggregator = GatedAttention(hidden_dim, attention_dim, gated=True, dropout=dropout)
    self.classifier = LinearClassifier(hidden_dim, 1, dropout)
    # (3) Projection head（蒸馏特征对齐用）
    self.proj_head = nn.Linear(hidden_dim, proj_dim) if proj_dim else None

  def forward(self, data: dict) -> dict:
    patches = data['he_patches']                          # (B, N, patch_dim)
    encoded = self.encoder(patches)                       # (B, N, hidden_dim)
    encoded = self.transformer(encoded)                   # (B, N, hidden_dim)
    bag_embeddings, attention = self.aggregator(encoded)  # (B, hidden_dim)
    logits = self.classifier(bag_embeddings)              # (B, 1)
    out = {'hidden': bag_embeddings, 'logits': logits, 'attention': attention}
    if self.proj_head is not None:
      out['proj'] = self.proj_head(bag_embeddings)
    return out

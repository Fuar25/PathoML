"""StudentBasicABMIL: 与PathoML ABMIL架构完全一致的基础学生模型。

架构:
  HE patches (B, N, patch_dim)
    → FeatureEncoder: Linear(patch_dim→hidden_dim) + GELU + Dropout → (B, N, hidden_dim)
    → GatedAttention(gated=True) → bag_embeddings (B, hidden_dim)  ← 'hidden'，对齐teacher
    → LinearClassifier: Dropout + Linear(hidden_dim→1) → logits (B, 1)

hidden_dim 与 teacher hidden_dim 保持一致（默认128）。
forward 接收 dict（与 PathoML TrainingMixin._model_inputs 约定一致），取 'he_patches'。
"""

from __future__ import annotations

import torch.nn as nn

from PathoML.models.abmil import FeatureEncoder, GatedAttention, LinearClassifier


class StudentBasicABMIL(nn.Module):
  """与PathoML ABMIL架构完全一致的学生模型，输入HE patch特征，输出hidden和BCE logit。"""

  def __init__(
    self,
    patch_dim: int = 1536,
    hidden_dim: int = 128,
    attention_dim: int = 128,
    dropout: float = 0.2,
  ) -> None:
    super().__init__()
    self.encoder    = FeatureEncoder(patch_dim, hidden_dim, dropout)
    self.aggregator = GatedAttention(hidden_dim, attention_dim, gated=True, dropout=dropout)
    self.classifier = LinearClassifier(hidden_dim, 1, dropout)

  def forward(self, data: dict) -> dict:
    patches = data['he_patches']                          # (B, N, patch_dim)
    mask = data.get('mask')                               # (B, N) or None; True=valid
    encoded = self.encoder(patches)                       # (B, N, hidden_dim)
    bag_embeddings, attention = self.aggregator(encoded, mask=mask)
    logits = self.classifier(bag_embeddings)              # (B, 1)
    out = {'hidden': bag_embeddings, 'logits': logits, 'attention': attention,
           'attn_logits': self.aggregator.last_logits,    # (B, N) softmax 前
           'encoded': encoded}                            # (B, N, hidden_dim)
    if mask is not None:
      out['mask'] = mask                                  # (B, N)
    return out

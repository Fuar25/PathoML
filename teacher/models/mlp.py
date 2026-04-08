"""Teacher model: single-hidden-layer MLP."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from PathoML.interfaces import BaseModel, DataDict
from PathoML.registry import register_model


@register_model('mlp')
class MLP(BaseModel):
  """Teacher concrete MLP model."""

  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    num_classes: int = 1,
    dropout: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
      nn.Linear(hidden_dim, num_classes),
    )

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    features = data_dict['features']
    if features.dim() == 3 and features.size(1) == 1:
      features = features.squeeze(1)
    logits = self.net(features)
    return {'logits': logits}

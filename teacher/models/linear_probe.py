"""Teacher model: linear probe baseline."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from PathoML.interfaces import BaseModel, DataDict
from PathoML.registry import register_model


@register_model('linear_probe')
class LinearProbe(BaseModel):
  """Teacher concrete linear probe model."""

  def __init__(
    self,
    input_dim: int,
    num_classes: int = 1,
    dropout: float = 0,
    **kwargs,
  ) -> None:
    super().__init__()
    self.linear = nn.Linear(input_dim, num_classes)
    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    features = data_dict['features']
    if features.dim() == 3 and features.size(1) == 1:
      features = features.squeeze(1)
    features = self.dropout(features)
    logits = self.linear(features)
    return {'logits': logits}

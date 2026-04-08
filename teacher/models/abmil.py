"""Teacher model: Attention-Based MIL."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from PathoML.interfaces import BaseMIL, DataDict
from PathoML.models.abmil import FeatureEncoder, GatedAttention, LinearClassifier
from PathoML.registry import register_model


@register_model('abmil')
class ABMIL(BaseMIL):
  """Teacher concrete ABMIL model."""

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
    super().__init__()
    self.external_impl = external_impl
    enc_dropout = encoder_dropout if encoder_dropout is not None else dropout
    clf_dropout = classifier_dropout if classifier_dropout is not None else dropout
    self.encoder = FeatureEncoder(input_dim, hidden_dim, enc_dropout)
    self.aggregator = GatedAttention(hidden_dim, attention_dim, gated, dropout)
    self.classifier = LinearClassifier(hidden_dim, num_classes, clf_dropout)

  def attach_external_impl(self, module: nn.Module) -> None:
    self.external_impl = module

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    if self.external_impl is not None:
      return self.external_impl(data_dict)

    features = data_dict['features']
    mask = data_dict.get('mask')
    encoded = self.encoder(features)
    bag_embeddings, attention = self.aggregator(encoded, mask)
    logits = self.classifier(bag_embeddings)
    return {
      'logits': logits,
      'bag_embeddings': bag_embeddings,
      'attention': attention,
    }

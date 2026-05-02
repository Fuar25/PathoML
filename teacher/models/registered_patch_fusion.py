"""Teacher model: registered multimodal patch fusion MIL."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from PathoML.interfaces import BaseMIL, DataDict
from PathoML.models.abmil import GatedAttention, LinearClassifier
from PathoML.registry import register_model


@register_model('registered_patch_fusion_mil')
class RegisteredPatchFusionMIL(BaseMIL):
  """Stain-aware MIL for registered patch-concat multimodal features."""

  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    num_modalities: int = 3,
    modality_hidden_dim: int = 128,
    dropout: float = 0.25,
    attention_dim: Optional[int] = None,
    gated: bool = True,
    modality_dropout: float = 0.1,
    use_modality_mask_features: bool = True,
  ) -> None:
    super().__init__()
    if num_modalities <= 0:
      raise ValueError("num_modalities must be positive")
    if input_dim % num_modalities != 0:
      raise ValueError(
        f"input_dim={input_dim} must be divisible by num_modalities={num_modalities}"
      )

    self.num_modalities = int(num_modalities)
    self.modality_feature_dim = input_dim // self.num_modalities
    self.modality_dropout = float(modality_dropout)
    self.use_modality_mask_features = bool(use_modality_mask_features)

    self.modality_encoders = nn.ModuleList([
      nn.Sequential(
        nn.Linear(self.modality_feature_dim, modality_hidden_dim),
        nn.LayerNorm(modality_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
      )
      for _ in range(self.num_modalities)
    ])
    fusion_input_dim = self.num_modalities * modality_hidden_dim
    if self.use_modality_mask_features:
      fusion_input_dim += self.num_modalities
    self.fusion = nn.Sequential(
      nn.Linear(fusion_input_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.aggregator = GatedAttention(hidden_dim, attention_dim, gated, dropout)
    self.classifier = LinearClassifier(hidden_dim, num_classes, dropout)

  def _encode_modalities(self, features: torch.Tensor) -> torch.Tensor:
    chunks = features.split(self.modality_feature_dim, dim=-1)
    encoded = [
      encoder(chunk)
      for encoder, chunk in zip(self.modality_encoders, chunks)
    ]
    return torch.stack(encoded, dim=2)

  def _apply_modality_mask(
    self,
    encoded: torch.Tensor,
    modality_mask: Optional[torch.Tensor],
  ) -> torch.Tensor:
    if modality_mask is None:
      return encoded
    return encoded * modality_mask.unsqueeze(-1).to(encoded.dtype)

  def _apply_modality_dropout(self, encoded: torch.Tensor) -> torch.Tensor:
    if not self.training or self.modality_dropout <= 0:
      return encoded

    keep_prob = 1.0 - self.modality_dropout
    keep = (
      torch.rand(
        encoded.size(0),
        1,
        encoded.size(2),
        1,
        device=encoded.device,
      ) < keep_prob
    ).to(encoded.dtype)

    all_dropped = keep.sum(dim=2).eq(0).view(encoded.size(0))
    if all_dropped.any():
      keep[all_dropped, :, 0, :] = 1.0

    return encoded * keep / keep_prob

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    features = data_dict['features']
    mask = data_dict.get('mask')
    modality_mask = data_dict.get('modality_mask')

    encoded_modalities = self._encode_modalities(features)
    encoded_modalities = self._apply_modality_mask(encoded_modalities, modality_mask)
    encoded_modalities = self._apply_modality_dropout(encoded_modalities)
    fusion_input = encoded_modalities.flatten(start_dim=2)
    if self.use_modality_mask_features:
      if modality_mask is None:
        modality_mask = torch.ones(
          *features.shape[:2],
          self.num_modalities,
          dtype=features.dtype,
          device=features.device,
        )
      fusion_input = torch.cat([fusion_input, modality_mask.to(features.dtype)], dim=-1)
    fused_instances = self.fusion(fusion_input)
    bag_embeddings, attention = self.aggregator(fused_instances, mask)
    logits = self.classifier(bag_embeddings)
    return {
      'logits': logits,
      'bag_embeddings': bag_embeddings,
      'attention': attention,
    }

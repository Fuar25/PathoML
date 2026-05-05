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
    self.hidden_dim = int(hidden_dim)

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

  def _apply_encoded_modality_transform(
    self,
    encoded: torch.Tensor,
    data_dict: Optional[DataDict] = None,
  ) -> torch.Tensor:
    return encoded

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

  def _fuse_instances(
    self,
    data_dict: DataDict,
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    features = data_dict['features']
    mask = data_dict.get('mask')
    modality_mask = data_dict.get('modality_mask')

    encoded_modalities = self._encode_modalities(features)
    encoded_modalities = self._apply_modality_mask(encoded_modalities, modality_mask)
    encoded_modalities = self._apply_encoded_modality_transform(
      encoded_modalities,
      data_dict,
    )
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
    return fused_instances, mask

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    fused_instances, mask = self._fuse_instances(data_dict)
    bag_embeddings, attention = self.aggregator(fused_instances, mask)
    logits = self.classifier(bag_embeddings)
    return {
      'logits': logits,
      'bag_embeddings': bag_embeddings,
      'attention': attention,
    }


@register_model('registered_patch_coord_fusion_mil')
class RegisteredPatchCoordFusionMIL(RegisteredPatchFusionMIL):
  """Registered patch fusion MIL with per-bag normalized coordinate features."""

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
    coord_hidden_dim: int = 16,
  ) -> None:
    if coord_hidden_dim <= 0:
      raise ValueError("coord_hidden_dim must be positive")
    super().__init__(
      input_dim=input_dim,
      hidden_dim=hidden_dim,
      num_classes=num_classes,
      num_modalities=num_modalities,
      modality_hidden_dim=modality_hidden_dim,
      dropout=dropout,
      attention_dim=attention_dim,
      gated=gated,
      modality_dropout=modality_dropout,
      use_modality_mask_features=use_modality_mask_features,
    )
    self.coord_hidden_dim = int(coord_hidden_dim)
    self.coord_encoder = nn.Sequential(
      nn.Linear(2, self.coord_hidden_dim),
      nn.LayerNorm(self.coord_hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    mil_input_dim = self.hidden_dim + self.coord_hidden_dim
    self.aggregator = GatedAttention(mil_input_dim, attention_dim, gated, dropout)
    self.classifier = LinearClassifier(mil_input_dim, num_classes, dropout)

  @staticmethod
  def _normalized_coords(
    coords: torch.Tensor,
    mask: Optional[torch.Tensor],
    dtype: torch.dtype,
  ) -> torch.Tensor:
    if coords.dim() != 3 or coords.shape[-1] != 2:
      raise ValueError("coords must have shape (batch, patches, 2)")

    coords = coords.to(dtype=dtype)
    if mask is None:
      valid = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    else:
      valid = mask.bool().to(coords.device)
      if valid.shape != coords.shape[:2]:
        raise ValueError("mask must have shape matching coords batch and patch dims")

    valid_f = valid.unsqueeze(-1).to(dtype)
    counts = valid_f.sum(dim=1).clamp_min(1.0)
    center = (coords * valid_f).sum(dim=1) / counts
    centered = coords - center.unsqueeze(1)

    valid_centered = centered.masked_fill(~valid.unsqueeze(-1), 0.0)
    scale = valid_centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    normalized = centered / scale
    return normalized.masked_fill(~valid.unsqueeze(-1), 0.0)

  def _coord_encoder_input(self, normalized_coords: torch.Tensor) -> torch.Tensor:
    return normalized_coords

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    features = data_dict['features']
    fused_instances, mask = self._fuse_instances(data_dict)
    coords = data_dict.get('coords')
    if coords is None:
      coords = torch.zeros(
        *features.shape[:2],
        2,
        dtype=features.dtype,
        device=features.device,
      )
    coords = coords.to(device=features.device)
    coord_features = self.coord_encoder(
      self._coord_encoder_input(
        self._normalized_coords(coords, mask, fused_instances.dtype)
      )
    )
    mil_instances = torch.cat([fused_instances, coord_features], dim=-1)
    bag_embeddings, attention = self.aggregator(mil_instances, mask)
    logits = self.classifier(bag_embeddings)
    return {
      'logits': logits,
      'bag_embeddings': bag_embeddings,
      'attention': attention,
    }


@register_model('registered_patch_polycoord_fusion_mil')
class RegisteredPatchPolyCoordFusionMIL(RegisteredPatchCoordFusionMIL):
  """Registered patch fusion MIL with polynomial coordinate features."""

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
    coord_hidden_dim: int = 16,
  ) -> None:
    super().__init__(
      input_dim=input_dim,
      hidden_dim=hidden_dim,
      num_classes=num_classes,
      num_modalities=num_modalities,
      modality_hidden_dim=modality_hidden_dim,
      dropout=dropout,
      attention_dim=attention_dim,
      gated=gated,
      modality_dropout=modality_dropout,
      use_modality_mask_features=use_modality_mask_features,
      coord_hidden_dim=coord_hidden_dim,
    )
    self.coord_encoder = nn.Sequential(
      nn.Linear(5, self.coord_hidden_dim),
      nn.LayerNorm(self.coord_hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
    )

  def _coord_encoder_input(self, normalized_coords: torch.Tensor) -> torch.Tensor:
    x = normalized_coords[..., 0:1]
    y = normalized_coords[..., 1:2]
    return torch.cat([x, y, x * y, x.square(), y.square()], dim=-1)


@register_model('registered_patch_polycoord_stain_affine_gate_fusion_mil')
class RegisteredPatchPolyCoordStainAffineGateFusionMIL(RegisteredPatchPolyCoordFusionMIL):
  """Polycoord registered patch fusion MIL with per-stain scalar affine gates."""

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
    coord_hidden_dim: int = 16,
  ) -> None:
    super().__init__(
      input_dim=input_dim,
      hidden_dim=hidden_dim,
      num_classes=num_classes,
      num_modalities=num_modalities,
      modality_hidden_dim=modality_hidden_dim,
      dropout=dropout,
      attention_dim=attention_dim,
      gated=gated,
      modality_dropout=modality_dropout,
      use_modality_mask_features=use_modality_mask_features,
      coord_hidden_dim=coord_hidden_dim,
    )
    self.stain_affine_scale = nn.Parameter(torch.ones(self.num_modalities))
    self.stain_affine_bias = nn.Parameter(torch.zeros(self.num_modalities))

  def _apply_encoded_modality_transform(
    self,
    encoded: torch.Tensor,
    data_dict: Optional[DataDict] = None,
  ) -> torch.Tensor:
    scale = self.stain_affine_scale.to(dtype=encoded.dtype).view(
      1,
      1,
      self.num_modalities,
      1,
    )
    bias = self.stain_affine_bias.to(dtype=encoded.dtype).view(
      1,
      1,
      self.num_modalities,
      1,
    )
    return encoded * scale + bias


@register_model('registered_patch_polycoord_stain_bias_coord_gate_fusion_mil')
class RegisteredPatchPolyCoordStainBiasCoordGateFusionMIL(RegisteredPatchPolyCoordFusionMIL):
  """Polycoord fusion MIL with per-stain bias and coordinate-conditioned gates."""

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
    coord_hidden_dim: int = 16,
    coord_gate_scale: float = 0.1,
  ) -> None:
    if coord_gate_scale < 0:
      raise ValueError("coord_gate_scale must be non-negative")
    super().__init__(
      input_dim=input_dim,
      hidden_dim=hidden_dim,
      num_classes=num_classes,
      num_modalities=num_modalities,
      modality_hidden_dim=modality_hidden_dim,
      dropout=dropout,
      attention_dim=attention_dim,
      gated=gated,
      modality_dropout=modality_dropout,
      use_modality_mask_features=use_modality_mask_features,
      coord_hidden_dim=coord_hidden_dim,
    )
    self.coord_gate_scale = float(coord_gate_scale)
    self.stain_bias = nn.Parameter(torch.zeros(self.num_modalities))
    self.coord_modality_gate = nn.Linear(5, self.num_modalities)
    nn.init.zeros_(self.coord_modality_gate.weight)
    nn.init.zeros_(self.coord_modality_gate.bias)

  def _coord_gate_input(
    self,
    encoded: torch.Tensor,
    data_dict: Optional[DataDict],
  ) -> torch.Tensor:
    if data_dict is None:
      return encoded.new_zeros(*encoded.shape[:2], 5)

    coords = data_dict.get('coords')
    if coords is None:
      coords = encoded.new_zeros(*encoded.shape[:2], 2)
    coords = coords.to(device=encoded.device)
    normalized = self._normalized_coords(
      coords,
      data_dict.get('mask'),
      encoded.dtype,
    )
    return self._coord_encoder_input(normalized)

  def _apply_encoded_modality_transform(
    self,
    encoded: torch.Tensor,
    data_dict: Optional[DataDict] = None,
  ) -> torch.Tensor:
    coord_input = self._coord_gate_input(encoded, data_dict)
    gate_logits = self.coord_modality_gate(coord_input)
    gate_residual = 2.0 * torch.sigmoid(gate_logits) - 1.0
    gate = 1.0 + self.coord_gate_scale * gate_residual
    bias = self.stain_bias.to(dtype=encoded.dtype).view(
      1,
      1,
      self.num_modalities,
      1,
    )
    return encoded * gate.to(dtype=encoded.dtype).unsqueeze(-1) + bias

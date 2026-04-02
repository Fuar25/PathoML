"""Slide-level multimodal fusion (weighted mean) dataset."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ...registry import register_dataset
from ...config.defaults import PATIENT_ID_PATTERN
from ._base import _MultimodalSlideBase


@register_dataset('MultimodalFusionSlideDataset')
class MultimodalFusionSlideDataset(_MultimodalSlideBase):
  """Slide-level multimodal fusion dataset via weighted mean.

  Fuses modality feature vectors by weighted mean, preserving feature dim D:
      (1, D), (1, D) → (1, D)

  Usage:
      dataset = MultimodalFusionSlideDataset(
          modality_paths={'HE': '/data/HE', 'CD20': '/data/CD20'},
          modality_names=['HE', 'CD20'],
          fusion_weights={'HE': 0.6, 'CD20': 0.4},
      )
  """

  def __init__(
    self,
    modality_paths: Dict[str, str],
    modality_names: List[str],
    fusion_weights: Dict[str, float],
    labels_csv: str,
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allow_missing_modalities: bool = True,
    verbose: bool = True,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    # Normalize weights before calling super().__init__ so verbose print can use them
    total = sum(fusion_weights.values())
    self.fusion_weights = {k: v / total for k, v in fusion_weights.items()}

    super().__init__(
      modality_paths=modality_paths,
      modality_names=modality_names,
      labels_csv=labels_csv,
      patient_id_pattern=patient_id_pattern,
      allow_missing_modalities=allow_missing_modalities,
      verbose=False,  # suppress base print; we print below with extra info
      allowed_sample_keys=allowed_sample_keys,
    )
    if verbose:
      print(
        f"MultimodalFusionSlideDataset: {len(self.samples)} samples, "
        f"modalities={self.modality_names}, fusion_weights={self.fusion_weights}"
      )

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    item = self.samples[idx]
    sample_key = item['sample_key']
    loaded_features, loaded_coords = self._load_modality_features(sample_key)

    if not loaded_features:
      raise RuntimeError(f"No modality features loaded for sample {sample_key}")

    # Weighted mean over available modalities (re-normalize if some are missing)
    available = [m for m in self.modality_names if m in loaded_features]
    avail_weights = {m: self.fusion_weights.get(m, 1.0) for m in available}
    total_w = sum(avail_weights.values())
    normalized = {m: w / total_w for m, w in avail_weights.items()}

    fused: Optional[torch.Tensor] = None
    for m in available:
      weighted = normalized[m] * loaded_features[m]
      fused = weighted if fused is None else fused + weighted

    features: torch.Tensor = fused  # type: ignore[assignment]

    coords = loaded_coords.get(available[0]) if available else None
    if coords is None:
      coords = torch.zeros(features.shape[0], 2, dtype=torch.float32)

    label_tensor = (
      torch.tensor(item['label']).float()
      if self.binary_mode
      else torch.tensor(item['label']).long()
    )
    return {
      'features':   features,
      'coords':     coords,
      'label':      label_tensor,
      'slide_id':   f"{sample_key[0]}{sample_key[1]}",
      'patient_id': item['patient_id'],
      'tissue_id':  item['tissue_id'],
      'modalities': available,
    }

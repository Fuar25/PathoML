"""Slide-level bimodal concat + interaction dataset."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from ...registry import register_dataset
from ._base import _MultimodalSlideBase


@register_dataset('BimodalConcatInteractSlideDataset')
class BimodalConcatInteractSlideDataset(_MultimodalSlideBase):
  """Slide-level bimodal concat + element-wise interaction dataset.

  For exactly 2 modalities A and B (each dim D):
      [feat_A, feat_B, feat_A * feat_B]  →  (1, 3D)

  Usage:
      dataset = BimodalConcatInteractSlideDataset(
          modality_paths={'HE': '/data/HE', 'CD20': '/data/CD20'},
          modality_names=['HE', 'CD20'],
      )
  """

  def __init__(self, *args, **kwargs) -> None:
    modality_names = kwargs.get('modality_names') or (args[1] if len(args) > 1 else None)
    if modality_names is not None and len(modality_names) != 2:
      raise ValueError(
        f"BimodalConcatInteractSlideDataset requires exactly 2 modality names, "
        f"got {len(modality_names)}: {modality_names}"
      )
    super().__init__(*args, **kwargs)

    # Validate equal feature dims
    self._modality_feature_dims = self._infer_feature_dims()
    self._validate_equal_dims()

    if self.verbose:
      d = self._modality_feature_dims.get(self.modality_names[0], 0)
      print(
        f"BimodalConcatInteractSlideDataset: {len(self.samples)} samples, "
        f"modalities={self.modality_names}, concat_interact_dim={3 * d}"
      )

  def _infer_feature_dims(self) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for modality_name in self.modality_names:
      modality_dir = self._get_modality_dir(modality_name)
      if modality_dir is None:
        continue
      found = False
      for root, _, files in os.walk(modality_dir):
        for filename in files:
          if not filename.endswith('.h5'):
            continue
          try:
            with h5py.File(os.path.join(root, filename), 'r') as f:
              dims[modality_name] = int(torch.from_numpy(np.array(f['features'])).float().shape[-1])
              found = True
              break
          except Exception:
            continue
        if found:
          break
      if not found and self.verbose:
        print(f"Warning: could not infer feature dim for modality '{modality_name}'")
    return dims

  def _validate_equal_dims(self) -> None:
    name_a, name_b = self.modality_names
    dim_a = self._modality_feature_dims.get(name_a)
    dim_b = self._modality_feature_dims.get(name_b)
    if dim_a is None or dim_b is None:
      return  # will surface at __getitem__ time if dims are wrong
    if dim_a != dim_b:
      raise ValueError(
        f"Both modalities must have equal feature dimensions for element-wise product. "
        f"Got '{name_a}'={dim_a}, '{name_b}'={dim_b}."
      )

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    item = self.samples[idx]
    sample_key = item['sample_key']
    loaded_features, loaded_coords = self._load_modality_features(sample_key)

    if not loaded_features:
      raise RuntimeError(f"No modality features loaded for sample {sample_key}")

    name_a, name_b = self.modality_names
    available_modalities: List[str] = []

    def _get_or_zeros(name: str) -> torch.Tensor:
      if name in loaded_features:
        available_modalities.append(name)
        return loaded_features[name]
      if not self.allow_missing_modalities:
        raise RuntimeError(f"Missing modality '{name}' for sample {sample_key}")
      dim = self._modality_feature_dims.get(name, 0)
      if dim <= 0:
        raise RuntimeError(
          f"Cannot zero-pad missing modality '{name}': unknown feature dim"
        )
      n = next(iter(loaded_features.values())).shape[0]
      return torch.zeros(n, dim, dtype=torch.float32)

    feat_a = _get_or_zeros(name_a)
    feat_b = _get_or_zeros(name_b)
    interaction = feat_a * feat_b         # zeros when either modality is absent

    features = torch.cat([feat_a, feat_b, interaction], dim=-1)  # (..., 3D)

    coords = None
    for m in self.modality_names:
      if m in loaded_coords:
        coords = loaded_coords[m]
        break
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
      'modalities': available_modalities,
    }

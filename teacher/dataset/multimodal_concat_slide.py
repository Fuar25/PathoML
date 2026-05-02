"""Teacher dataset: multimodal concat over slide-level feature bags."""

from __future__ import annotations

from typing import Any, Dict, List

import h5py
import numpy as np
import torch

from PathoML.dataset.base import MultimodalSlideDatasetBase
from PathoML.dataset.utils import _walk_h5_files
from PathoML.registry import register_dataset


@register_dataset('MultimodalConcatSlideDataset')
class MultimodalConcatSlideDataset(MultimodalSlideDatasetBase):
  """Slide-level multimodal concat dataset for teacher experiments."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._modality_feature_dims = self._infer_feature_dims()
    if self.verbose:
      total_dim = sum(self._modality_feature_dims.get(m, 0) for m in self.modality_names)
      print(
        f"MultimodalConcatSlideDataset: {len(self.samples)} samples, "
        f"modalities={self.modality_names}, concat_dim={total_dim}"
      )

  def _infer_feature_dims(self) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for modality_name in self.modality_names:
      for _, filepath in _walk_h5_files(self.data_root, stain=modality_name):
        try:
          with h5py.File(filepath, 'r') as f:
            dims[modality_name] = int(torch.from_numpy(np.array(f['features'])).float().shape[-1])
          break
        except Exception:
          continue
      if modality_name not in dims and self.verbose:
        print(f"Warning: could not infer feature dim for modality '{modality_name}'")
    return dims

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    item = self.samples[idx]
    sample_key = item['sample_key']
    loaded_features, loaded_coords = self._load_modality_features(sample_key)
    if not loaded_features:
      raise RuntimeError(f"No modality features loaded for sample {sample_key}")

    concat_parts: List[torch.Tensor] = []
    available_modalities: List[str] = []
    for modality_name in self.modality_names:
      if modality_name in loaded_features:
        concat_parts.append(loaded_features[modality_name])
        available_modalities.append(modality_name)
        continue
      if not self.allow_missing_modalities:
        raise RuntimeError(f"Missing modality '{modality_name}' for sample {sample_key}")
      dim = self._modality_feature_dims.get(modality_name, 0)
      if dim <= 0:
        raise RuntimeError(
          f"Cannot zero-pad missing modality '{modality_name}': unknown feature dim"
        )
      n = next(iter(loaded_features.values())).shape[0]
      concat_parts.append(torch.zeros(n, dim, dtype=torch.float32))

    features = torch.cat(concat_parts, dim=-1)
    coords = None
    for modality_name in self.modality_names:
      if modality_name in loaded_coords:
        coords = loaded_coords[modality_name]
        break
    if coords is None:
      coords = torch.zeros(features.shape[0], 2, dtype=torch.float32)

    label_tensor = (
      torch.tensor(item['label']).float()
      if self.binary_mode
      else torch.tensor(item['label']).long()
    )
    return {
      'features': features,
      'coords': coords,
      'label': label_tensor,
      'slide_id': f"{sample_key[0]}{sample_key[1]}",
      'patient_id': item['patient_id'],
      'tissue_id': item['tissue_id'],
      'modalities': available_modalities,
    }

"""Multi-modal feature concatenation dataset.

Unlike multimodal_dataset_add.py (weighted sum), this concatenates features
from different modalities along the channel dimension, letting the model learn
cross-modal mappings: features shape becomes (N, sum(C_i)).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from ..config.defaults import PATIENT_ID_PATTERN
from ..optimization.interfaces import BaseDataset
from ..optimization.registry import register_dataset


def _extract_patient_tissue_id(filename: str, pattern: str) -> Optional[tuple]:
  """Extract (patient_id, tissue_id) from a filename."""
  patient_match = re.search(pattern, filename)
  if not patient_match:
    return None

  patient_id = patient_match.group(1)
  remaining = filename[patient_match.end():]
  tissue_match = re.match(r"([A-Za-z0-9])-", remaining)
  if tissue_match:
    tissue_id = tissue_match.group(1)
    return (patient_id, tissue_id)
  return None


def _normalize_feature_array(feature_array: np.ndarray) -> np.ndarray:
  """Normalize feature array to shape (N, C).

  Rules:
    1D  → (1, C)  — single-instance feature
    2D  → returned as-is
    >=3D → first dim kept as N, rest flattened to C
  """
  if feature_array.ndim == 1:
    return feature_array.reshape(1, -1)
  if feature_array.ndim == 2:
    return feature_array
  return feature_array.reshape(feature_array.shape[0], -1)


@register_dataset("multimodal_concat")
class MultimodalConcatDataset(BaseDataset):
  """WSI dataset that concatenates multi-modal patch features along channel dim.

  Design:
  - One sample = one (patient_id, tissue_id) pair.
  - Features from each modality are concatenated → (N, sum(C_i)).
  - Patch counts across modalities are aligned to the minimum N.
  - Missing modalities are zero-padded when allow_missing_modalities=True.

  Usage:
      dataset = MultimodalConcatDataset(
          modality_paths={'HE': '/data/HE', 'CD20': '/data/CD20'},
          modality_names=['HE', 'CD20'],
      )
  """

  def __init__(
    self,
    modality_paths: Dict[str, str],
    modality_names: List[str],
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allow_missing_modalities: bool = True,
    binary_mode: Optional[bool] = None,
    verbose: bool = True,
  ) -> None:
    self.modality_paths = modality_paths
    self.modality_names = modality_names
    self.patient_id_pattern = patient_id_pattern
    self.allow_missing_modalities = allow_missing_modalities
    self.verbose = verbose

    self.classes = self._detect_classes()
    self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    self.modality_feature_dims = self._infer_modality_feature_dims()
    self.samples: List[Dict[str, Any]] = []
    self.modality_index: Dict[tuple, Dict[str, str]] = {}
    self._build_samples()
    self.data = self.samples

    if self.verbose:
      total_dim = sum(self.modality_feature_dims.get(m, 0) for m in self.modality_names)
      print(
        f"MultimodalConcatDataset: {len(self.samples)} samples, "
        f"modalities={self.modality_names}, concat_dim={total_dim}"
      )

  def _detect_classes(self) -> List[str]:
    classes = set()
    if not self.modality_paths:
      return []
    first_modality_path = next(iter(self.modality_paths.values()))
    if os.path.isdir(first_modality_path):
      for item in os.listdir(first_modality_path):
        if os.path.isdir(os.path.join(first_modality_path, item)):
          classes.add(item)
    return sorted(list(classes))

  def _infer_modality_feature_dims(self) -> Dict[str, int]:
    """Infer feature dimension for each modality (needed for zero-padding missing ones)."""
    dims: Dict[str, int] = {}
    for modality_name in self.modality_names:
      modality_dir = self._get_modality_dir(modality_name)
      if modality_dir is None:
        continue
      found = False
      for root, _, files in os.walk(modality_dir):
        for filename in files:
          if not filename.endswith(".h5"):
            continue
          try:
            with h5py.File(os.path.join(root, filename), "r") as f:
              feats = _normalize_feature_array(np.array(f['features']))
              dims[modality_name] = int(feats.shape[1])
              found = True
              break
          except Exception:
            continue
        if found:
          break
      if not found and self.verbose:
        print(f"Warning: could not infer feature dim for modality '{modality_name}'")
    return dims

  def _build_samples(self) -> None:
    if not self.modality_paths:
      return
    first_modality_path = next(iter(self.modality_paths.values()))
    seen_keys = set()

    for cls_name in self.classes:
      class_dir = os.path.join(first_modality_path, cls_name)
      if not os.path.isdir(class_dir):
        continue
      label = self.class_to_idx[cls_name]

      for root, _, files in os.walk(class_dir):
        for filename in files:
          if not filename.endswith(".h5"):
            continue
          key_info = _extract_patient_tissue_id(filename, self.patient_id_pattern)
          if key_info is None:
            continue

          patient_id, tissue_id = key_info
          sample_key = (patient_id, tissue_id)
          if sample_key in seen_keys:
            continue
          seen_keys.add(sample_key)

          modality_filepaths = {}
          for modality_name in self.modality_names:
            fp = self._find_modality_file_in_class(filename, modality_name, cls_name)
            if fp is not None:
              modality_filepaths[modality_name] = fp

          if not modality_filepaths:
            continue
          if not self.allow_missing_modalities and len(modality_filepaths) < len(self.modality_names):
            continue

          self.modality_index[sample_key] = modality_filepaths
          self.samples.append({
            "sample_key": sample_key,
            "patient_id": patient_id,
            "tissue_id": tissue_id,
            "label": label,
            "class_name": cls_name,
            "modalities": list(modality_filepaths.keys()),
          })

  def _get_modality_dir(self, modality_name: str) -> Optional[str]:
    modality_lower = modality_name.lower()
    for key, modality_dir in self.modality_paths.items():
      if key.lower() == modality_lower:
        return modality_dir
    return None

  def _find_modality_file_in_class(
    self, reference_filename: str, modality_name: str, class_name: str
  ) -> Optional[str]:
    key_info = _extract_patient_tissue_id(reference_filename, self.patient_id_pattern)
    if not key_info:
      return None
    patient_id, tissue_id = key_info
    prefix = f"{patient_id}{tissue_id}-"
    modality_dir = self._get_modality_dir(modality_name)
    if modality_dir is None:
      return None
    class_dir = os.path.join(modality_dir, class_name)
    if not os.path.isdir(class_dir):
      return None
    for filename in os.listdir(class_dir):
      if filename.endswith(".h5") and filename.startswith(prefix):
        return os.path.join(class_dir, filename)
    return None

  def get_patient_ids(self) -> List[str]:
    return [item["patient_id"] for item in self.samples]

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    item = self.samples[idx]
    sample_key = item["sample_key"]
    modality_filepaths = self.modality_index[sample_key]

    loaded_features: Dict[str, torch.Tensor] = {}
    loaded_coords: Dict[str, torch.Tensor] = {}

    for modality_name, file_path in modality_filepaths.items():
      with h5py.File(file_path, "r") as f:
        feats_np = _normalize_feature_array(np.array(f['features']))
        feats = torch.from_numpy(feats_np).float()
        loaded_features[modality_name] = feats
        if 'coords' in f:
          loaded_coords[modality_name] = torch.from_numpy(np.array(f['coords'])).float()

    if not loaded_features:
      raise RuntimeError(f"No modality features loaded for sample {sample_key}")

    # (1) Align patch counts: use minimum N across available modalities
    target_n = min(feats.shape[0] for feats in loaded_features.values())

    # (2) Concatenate along channel dim; zero-pad missing modalities
    concat_parts: List[torch.Tensor] = []
    available_modalities: List[str] = []
    for modality_name in self.modality_names:
      if modality_name in loaded_features:
        concat_parts.append(loaded_features[modality_name][:target_n])
        available_modalities.append(modality_name)
        continue

      if not self.allow_missing_modalities:
        raise RuntimeError(f"Missing modality '{modality_name}' for sample {sample_key}")

      dim = self.modality_feature_dims.get(modality_name, 0)
      if dim <= 0:
        raise RuntimeError(
          f"Cannot zero-pad missing modality '{modality_name}': unknown feature dim"
        )
      concat_parts.append(torch.zeros(target_n, dim, dtype=torch.float32))

    features = torch.cat(concat_parts, dim=1)  # (N, sum(C_i))

    # (3) Use first available modality's coords
    coords = None
    for modality_name in self.modality_names:
      if modality_name in loaded_coords:
        coords = loaded_coords[modality_name][:target_n]
        break
    if coords is None:
      coords = torch.zeros(target_n, 2, dtype=torch.float32)

    label_tensor = (
      torch.tensor(item["label"]).float()
      if self.binary_mode
      else torch.tensor(item["label"]).long()
    )

    return {
      "features":   features,   # (N, sum(C_i))
      "coords":     coords,     # (N, 2)
      "label":      label_tensor,
      "sample_id":  f"{item['sample_key'][0]}{item['sample_key'][1]}",
      "patient_id": item["patient_id"],
      "tissue_id":  item["tissue_id"],
      "modalities": available_modalities,
    }

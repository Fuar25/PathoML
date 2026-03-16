"""Multi-modal feature fusion dataset.

Unlike multimodal_dataset_concat.py (channel concatenation), this fuses features
from different modalities via weighted mean, preserving the original feature dimension D.
Output shape: (N, D) where N is the minimum patch count across modalities.
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


@register_dataset("multimodal_fusion")
class MultimodalFusionDataset(BaseDataset):
  """WSI dataset that fuses multi-modal features via weighted mean.

  Design:
  - One sample = one (patient_id, tissue_id) pair.
  - Features from each modality are combined via weighted mean → (N, D).
  - Patch counts are aligned to the minimum N across modalities.
  - Missing modalities are excluded from fusion when allow_missing_modalities=True.

  Usage:
      dataset = MultimodalFusionDataset(
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
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allow_missing_modalities: bool = True,
    fusion_strategy: str = "weighted_mean",
    binary_mode: Optional[bool] = None,
    verbose: bool = True,
  ) -> None:
    self.modality_paths = modality_paths
    self.modality_names = modality_names
    self.patient_id_pattern = patient_id_pattern
    self.allow_missing_modalities = allow_missing_modalities
    self.fusion_strategy = fusion_strategy
    self.verbose = verbose

    self.classes = self._detect_classes()
    self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    # Normalize fusion weights so they sum to 1
    total_weight = sum(fusion_weights.values())
    self.fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}

    self.samples: List[Dict[str, Any]] = []
    self.modality_index: Dict[tuple, Dict[str, str]] = {}
    self._build_samples()
    self.data = self.samples

    if self.verbose:
      print(
        f"MultimodalFusionDataset: {len(self.samples)} samples, "
        f"modalities={self.modality_names}, fusion_weights={self.fusion_weights}"
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
    first_coords: Optional[torch.Tensor] = None

    for modality_name, file_path in modality_filepaths.items():
      with h5py.File(file_path, "r") as f:
        feats = torch.from_numpy(np.array(f['features'])).float()
        # Normalize to 2D: (N, D)
        if feats.ndim == 1:
          feats = feats.unsqueeze(0)
        loaded_features[modality_name] = feats
        if first_coords is None and 'coords' in f:
          first_coords = torch.from_numpy(np.array(f['coords'])).float()

    if not loaded_features:
      raise RuntimeError(f"No modality features loaded for sample {sample_key}")

    # (1) Align patch counts to minimum N
    target_n = min(feats.shape[0] for feats in loaded_features.values())

    # (2) Weighted mean fusion over available modalities
    available_modalities = [m for m in self.modality_names if m in loaded_features]
    available_weights = {m: self.fusion_weights.get(m, 1.0) for m in available_modalities}
    total_w = sum(available_weights.values())
    normalized = {m: w / total_w for m, w in available_weights.items()}

    fused: Optional[torch.Tensor] = None
    for modality_name in available_modalities:
      weighted = normalized[modality_name] * loaded_features[modality_name][:target_n]
      fused = weighted if fused is None else fused + weighted

    features: torch.Tensor = fused  # type: ignore[assignment]

    # (3) Coords from first available modality
    coords = (
      first_coords[:target_n]
      if first_coords is not None
      else torch.zeros(target_n, 2, dtype=torch.float32)
    )

    label_tensor = (
      torch.tensor(item["label"]).float()
      if self.binary_mode
      else torch.tensor(item["label"]).long()
    )

    return {
      "features":   features,            # (N, D)
      "coords":     coords,              # (N, 2)
      "label":      label_tensor,
      "sample_id":  f"{item['sample_key'][0]}{item['sample_key'][1]}",
      "patient_id": item["patient_id"],
      "tissue_id":  item["tissue_id"],
      "modalities": available_modalities,
    }

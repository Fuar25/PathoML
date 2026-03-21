"""Shared base class for multimodal slide-level datasets."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py

from ...interfaces import BaseDataset
from ...config.defaults import PATIENT_ID_PATTERN
from ..utils import _extract_patient_tissue_id


class _MultimodalSlideBase(BaseDataset):
  """Base for multimodal slide-level datasets.

  Handles class detection, symmetric modality scanning, and sample building.
  Subclasses implement __getitem__ for their specific fusion strategy.

  Directory layout per modality path:
      modality_path/
        <class_name_1>/*.h5
        <class_name_2>/*.h5
  Class names are detected as the union of subdirectory names across all modalities.
  """

  def __init__(
    self,
    modality_paths: Dict[str, str],
    modality_names: List[str],
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allow_missing_modalities: bool = True,
    binary_mode: Optional[bool] = None,
    verbose: bool = True,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    self.modality_paths = modality_paths
    self.modality_names = modality_names
    self.patient_id_pattern = patient_id_pattern
    self.allow_missing_modalities = allow_missing_modalities
    self.verbose = verbose
    self.allowed_sample_keys = allowed_sample_keys

    self.classes = self._detect_classes()
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    self.samples: List[Dict[str, Any]] = []
    self.modality_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    self._build_samples()

  def _detect_classes(self) -> List[str]:
    """Union of class subdirectory names across all modality paths."""
    classes: Set[str] = set()
    for modality_dir in self.modality_paths.values():
      if os.path.isdir(modality_dir):
        for item in os.listdir(modality_dir):
          if os.path.isdir(os.path.join(modality_dir, item)):
            classes.add(item)
    return sorted(classes)

  def _get_modality_dir(self, modality_name: str) -> Optional[str]:
    modality_lower = modality_name.lower()
    for key, modality_dir in self.modality_paths.items():
      if key.lower() == modality_lower:
        return modality_dir
    return None

  def _collect_modality_map(
    self, modality_name: str, modality_dir: str
  ) -> Dict[Tuple[str, str, str], str]:
    """Return {(patient_id, tissue_id, class_name): filepath} for one modality."""
    result: Dict[Tuple[str, str, str], str] = {}
    for cls_name in self.classes:
      class_dir = os.path.join(modality_dir, cls_name)
      if not os.path.isdir(class_dir):
        continue
      for root, _, files in os.walk(class_dir):
        for filename in files:
          if not filename.endswith('.h5'):
            continue
          key_info = _extract_patient_tissue_id(filename, self.patient_id_pattern)
          if key_info is None:
            continue
          patient_id, tissue_id = key_info
          full_key = (patient_id, tissue_id, cls_name)
          if full_key not in result:
            result[full_key] = os.path.join(root, filename)
    return result

  def _build_samples(self) -> None:
    """Build sample list treating all modalities symmetrically (no anchor)."""
    # (1) Collect per-modality maps
    modality_maps: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for modality_name in self.modality_names:
      modality_dir = self._get_modality_dir(modality_name)
      modality_maps[modality_name] = (
        self._collect_modality_map(modality_name, modality_dir)
        if modality_dir is not None else {}
      )

    # (2) Union of all (patient_id, tissue_id, class_name) keys
    all_full_keys: Set[Tuple[str, str, str]] = set()
    for m in self.modality_names:
      all_full_keys |= set(modality_maps[m].keys())

    # (3) Apply allowed_sample_keys filter on (patient_id, tissue_id)
    if self.allowed_sample_keys is not None:
      all_full_keys = {k for k in all_full_keys if (k[0], k[1]) in self.allowed_sample_keys}

    # (4) Build samples; guard against duplicate (patient_id, tissue_id)
    seen: Set[Tuple[str, str]] = set()
    for patient_id, tissue_id, cls_name in sorted(all_full_keys):
      sample_key = (patient_id, tissue_id)
      if sample_key in seen:
        continue
      seen.add(sample_key)

      modality_filepaths: Dict[str, str] = {
        modality_name: fp
        for modality_name in self.modality_names
        if (fp := modality_maps[modality_name].get((patient_id, tissue_id, cls_name))) is not None
      }

      if not modality_filepaths:
        continue
      if not self.allow_missing_modalities and len(modality_filepaths) < len(self.modality_names):
        continue

      self.modality_index[sample_key] = modality_filepaths
      self.samples.append({
        'sample_key': sample_key,
        'patient_id': patient_id,
        'tissue_id':  tissue_id,
        'label':      self.class_to_idx[cls_name],
        'class_name': cls_name,
        'modalities': list(modality_filepaths.keys()),
      })

  def _load_modality_features(
    self, sample_key: Tuple[str, str]
  ) -> Dict[str, Any]:
    """Load raw H5 features and coords for each available modality."""
    import numpy as np
    import torch
    modality_filepaths = self.modality_index[sample_key]
    loaded_features: Dict[str, 'torch.Tensor'] = {}
    loaded_coords: Dict[str, 'torch.Tensor'] = {}
    for modality_name, file_path in modality_filepaths.items():
      with h5py.File(file_path, 'r') as f:
        loaded_features[modality_name] = torch.from_numpy(np.array(f['features'])).float()
        if 'coords' in f:
          loaded_coords[modality_name] = torch.from_numpy(np.array(f['coords'])).float()
    return loaded_features, loaded_coords

  def get_patient_ids(self) -> List[str]:
    return [item['patient_id'] for item in self.samples]

  def get_labels(self) -> List[int]:
    return [item['label'] for item in self.samples]

  def __len__(self) -> int:
    return len(self.samples)

"""Shared dataset base classes for pathology feature loading."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch

from ..config.defaults import PATIENT_ID_PATTERN
from ..interfaces import BaseDataset
from .utils import _extract_patient_tissue_id, _walk_h5_files, load_labels_csv


class UnimodalFeatureDatasetBase(BaseDataset):
  """Shared base for single-modality feature datasets."""

  def __init__(
    self,
    data_root: str,
    labels_csv: str,
    stain: Optional[str] = None,
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    binary_mode: Optional[bool] = None,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    self.data_root = data_root
    self.stain = stain
    self.patient_id_pattern = patient_id_pattern
    self.allowed_sample_keys = allowed_sample_keys
    self._label_map = load_labels_csv(labels_csv)
    self.classes = sorted(set(self._label_map.values()), reverse=True)
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    self.samples: List[Dict[str, Any]] = []
    self._scan_files()
    self.samples.sort(key=lambda x: (x['patient_id'], x['tissue_id']))
    print(f"{self.__class__.__name__} loaded: {len(self.samples)} samples, classes={self.classes}")

  def _scan_files(self) -> None:
    for filename, filepath in _walk_h5_files(self.data_root, stain=self.stain):
      key = _extract_patient_tissue_id(filename, self.patient_id_pattern)
      if key is None:
        continue
      if self.allowed_sample_keys is not None and key not in self.allowed_sample_keys:
        continue
      patient_id, tissue_id = key
      cls_name = self._label_map.get(patient_id)
      if cls_name is None:
        continue
      self.samples.append({
        'slide_id': filename.replace('.h5', ''),
        'patient_id': patient_id,
        'tissue_id': tissue_id,
        '_feature_path': filepath,
        'label': self.class_to_idx[cls_name],
        'class_name': cls_name,
      })

  def get_patient_ids(self) -> List[str]:
    return [item['patient_id'] for item in self.samples]

  def get_labels(self) -> List[int]:
    return [item['label'] for item in self.samples]

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    item = self.samples[idx]
    with h5py.File(item['_feature_path'], 'r') as f:
      features = torch.from_numpy(np.array(f['features'])).float()
      coords = (
        torch.from_numpy(np.array(f['coords'])).float()
        if 'coords' in f
        else torch.zeros(features.shape[0], 2)
      )
    label_tensor = (
      torch.tensor(item['label']).float()
      if self.binary_mode
      else torch.tensor(item['label']).long()
    )
    return {
      'features': features,
      'coords': coords,
      'label': label_tensor,
      'slide_id': item['slide_id'],
      'patient_id': item['patient_id'],
      'tissue_id': item['tissue_id'],
    }


class MultimodalSlideDatasetBase(BaseDataset):
  """Shared base for multimodal slide-level datasets."""

  def __init__(
    self,
    data_root: str,
    modality_names: List[str],
    labels_csv: str,
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allow_missing_modalities: bool = True,
    binary_mode: Optional[bool] = None,
    verbose: bool = True,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    self.data_root = data_root
    self.modality_names = modality_names
    self.patient_id_pattern = patient_id_pattern
    self.allow_missing_modalities = allow_missing_modalities
    self.verbose = verbose
    self.allowed_sample_keys = allowed_sample_keys
    self._label_map = load_labels_csv(labels_csv)

    self.classes = sorted(set(self._label_map.values()), reverse=True)
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    self.samples: List[Dict[str, Any]] = []
    self.modality_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    self._build_samples()

  def _collect_modality_map(self, stain: str) -> Dict[Tuple[str, str, str], str]:
    result: Dict[Tuple[str, str, str], str] = {}
    for filename, filepath in _walk_h5_files(self.data_root, stain=stain):
      key_info = _extract_patient_tissue_id(filename, self.patient_id_pattern)
      if key_info is None:
        continue
      patient_id, tissue_id = key_info
      cls_name = self._label_map.get(patient_id)
      if cls_name is None:
        continue
      full_key = (patient_id, tissue_id, cls_name)
      if full_key not in result:
        result[full_key] = filepath
    return result

  def _build_samples(self) -> None:
    modality_maps: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for modality_name in self.modality_names:
      modality_maps[modality_name] = self._collect_modality_map(modality_name)

    all_full_keys: Set[Tuple[str, str, str]] = set()
    for modality_name in self.modality_names:
      all_full_keys |= set(modality_maps[modality_name].keys())

    if self.allowed_sample_keys is not None:
      all_full_keys = {k for k in all_full_keys if (k[0], k[1]) in self.allowed_sample_keys}

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
        'tissue_id': tissue_id,
        'label': self.class_to_idx[cls_name],
        'class_name': cls_name,
        'modalities': list(modality_filepaths.keys()),
      })

  def _load_modality_features(
    self,
    sample_key: Tuple[str, str],
  ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    loaded_features: Dict[str, torch.Tensor] = {}
    loaded_coords: Dict[str, torch.Tensor] = {}
    modality_filepaths = self.modality_index[sample_key]
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

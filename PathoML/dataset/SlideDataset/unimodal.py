"""Slide-level unimodal WSI dataset."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch

from ...interfaces import BaseDataset
from ...registry import register_dataset
from ...config.defaults import PATIENT_ID_PATTERN
from ..utils import _extract_patient_tissue_id


class _UnimodalSlideBase(BaseDataset):
  """Base for single-modality slide-level datasets.

  Directory layout:
      data_path/
        <class_name_1>/*.h5
        <class_name_2>/*.h5
  """

  def __init__(
    self,
    data_path: str,
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    binary_mode: Optional[bool] = None,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    """
    Args:
        data_path: Root directory with one subdirectory per class.
        patient_id_pattern: Regex with group 1 matching the patient ID.
        binary_mode: Float labels for binary, long for multi-class.
            Auto-detected from number of classes if None.
        allowed_sample_keys: Optional (patient_id, tissue_id) whitelist.
    """
    self.data_path = data_path
    self.patient_id_pattern = patient_id_pattern
    self.allowed_sample_keys = allowed_sample_keys
    self.classes = self._detect_classes()
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    self.samples: List[Dict[str, Any]] = []
    self._scan_files()

    print(f"{self.__class__.__name__} loaded: {len(self.samples)} samples, classes={self.classes}")

  def _detect_classes(self) -> List[str]:
    if not os.path.isdir(self.data_path):
      return []
    return sorted(
      item for item in os.listdir(self.data_path)
      if os.path.isdir(os.path.join(self.data_path, item))
    )

  def _scan_files(self) -> None:
    for cls_name in self.classes:
      class_dir = os.path.join(self.data_path, cls_name)
      label = self.class_to_idx[cls_name]
      for root, _, files in os.walk(class_dir):
        for filename in files:
          if not filename.endswith('.h5'):
            continue
          key = _extract_patient_tissue_id(filename, self.patient_id_pattern)
          if key is None:
            print(f"Warning: Could not extract patient/tissue ID from '{filename}'. Skipping.")
            continue
          if self.allowed_sample_keys is not None and key not in self.allowed_sample_keys:
            continue
          patient_id, tissue_id = key
          self.samples.append({
            'slide_id':      filename.replace('.h5', ''),
            'patient_id':    patient_id,
            'tissue_id':     tissue_id,
            '_feature_path': os.path.join(root, filename),
            'label':         label,
            'class_name':    cls_name,
          })

  def get_patient_ids(self) -> List[str]:
    return [item['patient_id'] for item in self.samples]

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
      'features':   features,
      'coords':     coords,
      'label':      label_tensor,
      'slide_id':   item['slide_id'],
      'patient_id': item['patient_id'],
      'tissue_id':  item['tissue_id'],
    }


@register_dataset('UnimodalSlideDataset')
class UnimodalSlideDataset(_UnimodalSlideBase):
  """Slide-level unimodal dataset.

  H5 'features' shape: (1, D). No MIL needed downstream.

  Usage:
      dataset = UnimodalSlideDataset(data_path='/data/root')
  """
  pass

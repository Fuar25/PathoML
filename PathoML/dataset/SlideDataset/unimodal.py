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
from ..utils import _extract_patient_tissue_id, _walk_h5_files, load_labels_csv


class _UnimodalSlideBase(BaseDataset):
  """Base for single-modality slide-level datasets.

  Directory layout (patient-based):
      data_root/<patient_id>/<tissue_id>/<patient_id><tissue_id>-<stain>.h5
      labels_csv  ← patient_id,label
  """

  def __init__(
    self,
    data_root: str,
    labels_csv: str,
    stain: Optional[str] = None,
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    binary_mode: Optional[bool] = None,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    """
    Args:
        data_root: Feature root directory (scanned recursively).
        labels_csv: Path to CSV with 'patient_id' and 'label' columns.
        stain: If provided, only include H5 files matching this stain.
        patient_id_pattern: Regex with group 1 matching the patient ID.
        binary_mode: Float labels for binary, long for multi-class.
            Auto-detected from number of classes if None.
        allowed_sample_keys: Optional (patient_id, tissue_id) whitelist.
    """
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
        'slide_id':      filename.replace('.h5', ''),
        'patient_id':    patient_id,
        'tissue_id':     tissue_id,
        '_feature_path': filepath,
        'label':         self.class_to_idx[cls_name],
        'class_name':    cls_name,
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
      dataset = UnimodalSlideDataset(data_root='/data/Slide', stain='HE', labels_csv='labels.csv')
  """
  pass

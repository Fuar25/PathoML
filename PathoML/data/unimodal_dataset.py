"""WSI dataset: load single-modality WSI features from H5 files."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from ..optimization.interfaces import BaseDataset
from ..optimization.registry import register_dataset
from ..config.defaults import PATIENT_ID_PATTERN


def _extract_patient_id(filename: str, pattern: str) -> Optional[str]:
  """Extract patient ID (pathology number) from a filename.

  Args:
      filename: H5 filename, e.g. "B2022-01475B-cd20.h5".
      pattern: Regex with capture group 1 for the patient ID.

  Returns:
      Matched patient ID string, or None if no match.
  """
  match = re.search(pattern, filename)
  if match:
    return match.group(1)
  return None


@register_dataset('wsi_h5')
class UnimodalDataset(BaseDataset):
  """WSI feature dataset that loads single-modality features from H5 files.

  Each H5 file must contain 'features' (N, C) and optionally 'coords' (N, 2).
  Data directories are organized by class name under data_paths.

  Usage:
      dataset = UnimodalDataset(
          data_paths={'positive': '/path/MALT', 'negative': '/path/Reactive'}
      )
  """

  def __init__(
    self,
    data_paths: Dict[str, str],
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    binary_mode: Optional[bool] = None,
  ) -> None:
    """
    Args:
        data_paths: Mapping of class name to directory path.
            e.g. {'positive': '/path/MALT', 'negative': '/path/Reactive'}
            Each directory is scanned recursively for .h5 files.
        patient_id_pattern: Regex to extract patient ID from filenames.
        binary_mode: If True, labels are float tensors (binary classification).
            Auto-detected from number of classes if None.
    """
    self.data_paths = data_paths
    self.patient_id_pattern = patient_id_pattern
    self.classes = sorted(data_paths.keys())
    self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    self.binary_mode = binary_mode if binary_mode is not None else len(self.classes) == 2

    self.samples: List[Dict[str, Any]] = []
    self._scan_files()
    self.data = self.samples  # alias for dataset.data access pattern

    print(f"Dataset loaded: {len(self.samples)} samples, {len(self.classes)} classes")
    print(f"Class mapping: {self.class_to_idx}")
    print(f"Mode: {'Binary' if self.binary_mode else 'Multi-class'}")

  def _scan_files(self) -> None:
    """Recursively scan data_paths directories and populate self.samples."""
    for cls_name, dir_path in self.data_paths.items():
      if not os.path.exists(dir_path):
        print(f"Warning: Directory {dir_path} for class '{cls_name}' does not exist.")
        continue

      label = self.class_to_idx[cls_name]
      for root, _, files in os.walk(dir_path):
        for filename in files:
          if not filename.endswith('.h5'):
            continue
          patient_id = _extract_patient_id(filename, self.patient_id_pattern)
          if patient_id is None:
            print(f"Warning: Could not extract patient ID from '{filename}'. Skipping.")
            continue
          self.samples.append({
            'sample_id': filename.replace('.h5', ''),
            'patient_id': patient_id,
            'filename': filename,
            'feature_path': os.path.join(root, filename),
            'label': label,
            'class_name': cls_name,
          })

  def get_patient_ids(self) -> List[str]:
    """Return patient IDs in the same order as self.samples."""
    return [item['patient_id'] for item in self.samples]

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    """Lazily load and return one WSI sample from its H5 file.

    Returns:
        {
          'features':     (N, C) float tensor — instance features,
          'coords':       (N, 2) float tensor — patch top-left coordinates,
          'label':        scalar tensor (float for binary, long for multi-class),
          'sample_id':    str — filename without .h5 extension,
          'patient_id':   str — patient pathology number,
          'feature_path': str — full path to the H5 file,
        }
    """
    item = self.samples[idx]
    with h5py.File(item['feature_path'], 'r') as f:
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
      'features':     features,       # (N, C)
      'coords':       coords,         # (N, 2)
      'label':        label_tensor,
      'sample_id':    item['sample_id'],
      'patient_id':   item['patient_id'],
      'feature_path': item['feature_path'],
    }

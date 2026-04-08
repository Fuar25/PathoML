"""Dataset assembly for distillation training."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from PathoML.config.defaults import PATIENT_ID_PATTERN
from PathoML.dataset.utils import (
  _extract_patient_tissue_id,
  _walk_h5_files,
  find_common_sample_keys,
  load_labels_csv,
)
from PathoML.interfaces import BaseDataset


def _load_h5_features(path: str) -> torch.Tensor:
  """Load the `features` array from an H5 file."""
  with h5py.File(path, 'r') as f:
    if 'features' not in f:
      raise KeyError(f"'features' key not found in H5 file: {path}")
    return torch.from_numpy(np.array(f['features'])).float()


def _build_key_map(
  root: str,
  pattern: str,
  label_map: Dict[str, str],
  stain: Optional[str] = None,
  allowed_keys: Optional[set] = None,
) -> Dict[Tuple[str, str], Tuple[str, str]]:
  """Build `(patient_id, tissue_id) -> (class_name, abs_path)` for one root."""
  key_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
  if not os.path.isdir(root):
    raise FileNotFoundError(f"Data root does not exist: {root}")
  for fname, filepath in _walk_h5_files(root, stain=stain):
    key = _extract_patient_tissue_id(fname, pattern)
    if key is None:
      continue
    if allowed_keys is not None and key not in allowed_keys:
      continue
    patient_id, _ = key
    class_name = label_map.get(patient_id)
    if class_name is None:
      continue
    key_map[key] = (class_name, filepath)
  return key_map


class DistillationDataset(BaseDataset):
  """Load HE patch features plus ordered slide embeddings for distillation."""

  def __init__(
    self,
    patch_root: str,
    slide_root: str,
    slide_stains: List[str],
    labels_csv: str,
    patch_stain: str = 'HE',
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allowed_sample_keys: Optional[set] = None,
  ) -> None:
    super().__init__()
    if not slide_stains:
      raise ValueError("slide_stains cannot be empty.")
    self.patient_id_pattern = patient_id_pattern
    self.slide_stains = slide_stains
    self.patch_stain = patch_stain
    label_map = load_labels_csv(labels_csv)

    if allowed_sample_keys is not None:
      common_keys = allowed_sample_keys
    else:
      patch_keys = find_common_sample_keys(patch_root, [patch_stain], patient_id_pattern)
      slide_keys = find_common_sample_keys(slide_root, slide_stains, patient_id_pattern)
      common_keys = patch_keys & slide_keys
      if not common_keys:
        raise FileNotFoundError(
          "No common samples were found across patch and slide roots. "
          "Check roots and naming conventions."
        )

    patch_map = _build_key_map(
      patch_root,
      patient_id_pattern,
      label_map,
      stain=patch_stain,
      allowed_keys=common_keys,
    )
    slide_maps: Dict[str, Dict] = {
      stain: _build_key_map(
        slide_root,
        patient_id_pattern,
        label_map,
        stain=stain,
        allowed_keys=common_keys,
      )
      for stain in slide_stains
    }

    all_classes = sorted(set(label_map.values()), reverse=True)
    self.classes = all_classes
    self.class_to_label = {cls: i for i, cls in enumerate(all_classes)}

    self.samples: List[dict] = []
    for key in sorted(common_keys):
      if key not in patch_map:
        continue
      patient_id, tissue_id = key
      class_name, patch_path = patch_map[key]
      if any(key not in slide_maps[stain] for stain in self.slide_stains):
        continue
      slide_paths = {stain: slide_maps[stain][key][1] for stain in self.slide_stains}
      slide_id = os.path.splitext(os.path.basename(patch_path))[0]
      self.samples.append({
        'patient_id': patient_id,
        'tissue_id': tissue_id,
        'slide_id': slide_id,
        'label': self.class_to_label[class_name],
        'patch_path': patch_path,
        'slide_paths': slide_paths,
      })

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> dict:
    sample = self.samples[idx]
    he_patches = _load_h5_features(sample['patch_path'])
    slide_tensors = [
      _load_h5_features(sample['slide_paths'][stain]).view(-1)
      for stain in self.slide_stains
    ]
    slide_concat = torch.cat(slide_tensors, dim=0)
    return {
      'he_patches': he_patches,
      'slide_concat': slide_concat,
      'label': torch.tensor(sample['label'], dtype=torch.float32),
      'patient_id': sample['patient_id'],
      'tissue_id': sample['tissue_id'],
      'slide_id': sample['slide_id'],
    }

  def get_patient_ids(self) -> List[str]:
    return [sample['patient_id'] for sample in self.samples]

  def get_labels(self) -> List[int]:
    return [sample['label'] for sample in self.samples]

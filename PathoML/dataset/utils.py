"""Dataset-level utility functions."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

import torch

from ..config.defaults import PATIENT_ID_PATTERN


# ---------------------------------------------------------------------------
# (0) Collate helper — handles variable-length tensors (e.g. coords)
# ---------------------------------------------------------------------------

def _variable_size_collate(batch):
  """Collate that pads variable-length dim-0 tensors and creates a shared mask.

  Convention: mask True = valid, False = padding.
  Assumption: all variable-length tensors in a sample share the same N.
  """
  # (1) Transpose: List[Dict] → Dict[List], then collate each key
  grouped = {key: [d[key] for d in batch] for key in batch[0]}

  result = {}
  lengths = None
  for key, values in grouped.items():
    # (1.1) Non-tensor (e.g. slide_id strings) → keep as list
    if not torch.is_tensor(values[0]):
      result[key] = values
    # (1.2) Uniform shape → stack
    elif all(v.shape == values[0].shape for v in values):
      result[key] = torch.stack(values)
    # (1.3) Variable dim-0, same rest → pad
    elif (values[0].dim() >= 1
          and all(v.shape[1:] == values[0].shape[1:] for v in values)):
      sizes = [v.shape[0] for v in values]
      max_n = max(sizes)
      rest_shape = values[0].shape[1:]
      padded = torch.zeros(len(values), max_n, *rest_shape, dtype=values[0].dtype)
      for i, v in enumerate(values):
        padded[i, :v.shape[0]] = v
      result[key] = padded
      if lengths is None:
        lengths = sizes
    # (1.4) Fallback → keep as list
    else:
      result[key] = values

  # (2) Shared mask
  if lengths is not None:
    max_n = max(lengths)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    for i, n in enumerate(lengths):
      mask[i, :n] = True
    result['mask'] = mask
  return result


def _extract_patient_tissue_id(
  filename: str, pattern: str
) -> Optional[Tuple[str, str]]:
  """Extract (patient_id, tissue_id) from an H5 filename.

  Filename format: <patient_id><tissue_id>-<anything>.h5
  Example: "B2022-01475B-he.h5" → ("B2022-01475", "B")
  """
  patient_match = re.search(pattern, filename)
  if not patient_match:
    return None
  patient_id = patient_match.group(1)
  remaining = filename[patient_match.end():]
  tissue_match = re.match(r"([A-Za-z0-9])-", remaining)
  if tissue_match:
    return (patient_id, tissue_match.group(1))
  return None



def load_labels_csv(csv_path: str) -> Dict[str, str]:
  """Load patient_id → class_name mapping from a CSV file.

  CSV format:
      patient_id,label
      B2018-06208,MALT
      B2022-16580,Reactive

  Args:
      csv_path: Path to CSV file with 'patient_id' and 'label' columns.
  """
  import csv as csv_mod
  labels: Dict[str, str] = {}
  with open(csv_path, 'r', encoding='utf-8') as f:
    for row in csv_mod.DictReader(f):
      labels[row['patient_id']] = row['label']
  return labels


def find_common_sample_keys(
  dirs: List[str],
  patient_id_pattern: str = PATIENT_ID_PATTERN,
) -> Set[Tuple[str, str]]:
  """Return the intersection of (patient_id, tissue_id) pairs across all directories.

  Each dir is expected to be a modality root containing class subdirectories
  (e.g. /data/HE with MALT/ and Reactive/ inside). All .h5 files are scanned
  recursively, and (patient_id, tissue_id) pairs are extracted from filenames
  following the convention <patient_id><tissue_id>-<anything>.h5.

  Usage:
      common = find_common_sample_keys(['/data/HE', '/data/CD20'])
      # Returns only keys present in both directories.

  Args:
      dirs: List of modality root directories to intersect.
      patient_id_pattern: Regex with group 1 matching the patient ID.

  Returns:
      Set of (patient_id, tissue_id) tuples present in every directory.
  """
  if not dirs:
    return set()

  per_dir_keys: List[Set[Tuple[str, str]]] = []
  for d in dirs:
    keys: Set[Tuple[str, str]] = set()
    for root, _, files in os.walk(d):
      for filename in files:
        if not filename.endswith(".h5"):
          continue
        key = _extract_patient_tissue_id(filename, patient_id_pattern)
        if key is not None:
          keys.add(key)
    per_dir_keys.append(keys)

  result = per_dir_keys[0]
  for keys in per_dir_keys[1:]:
    result = result & keys
  return result

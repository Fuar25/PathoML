"""Dataset-level utility functions."""

from __future__ import annotations

import os
import re
import hashlib
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import torch
from torch.utils.data import Sampler

from ..config.defaults import PATIENT_ID_PATTERN


# ---------------------------------------------------------------------------
# (0) Stain helpers — normalize stain names for matching
# ---------------------------------------------------------------------------

def _normalize_stain(name: str) -> str:
  """Normalize stain name for matching: lowercase + strip hyphens.

  Examples: "Ki-67" → "ki67", "CK-pan" → "ckpan", "CD20" → "cd20"
  """
  return name.lower().replace('-', '')


def _extract_stain(filename: str) -> Optional[str]:
  """Extract normalized stain from H5 filename.

  Example: "B2022-01475B-HE.h5" → "he", "B2022-01475B-cd20.h5" → "cd20"
  """
  base = filename.replace('.h5', '')
  parts = base.rsplit('-', 1)
  if len(parts) == 2:
    return parts[1].lower()
  return None


def _walk_h5_files(
  root: str,
  stain: Optional[str] = None,
) -> List[Tuple[str, str]]:
  """Recursively scan root for H5 files, optionally filtering by stain.

  Args:
      root: Directory to scan recursively.
      stain: If provided, only return files whose stain matches (case/hyphen insensitive).

  Returns:
      List of (filename, absolute_filepath) tuples.
  """
  target = _normalize_stain(stain) if stain else None
  results = []
  for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
      if not fname.endswith('.h5'):
        continue
      if target is not None and _extract_stain(fname) != target:
        continue
      results.append((fname, os.path.join(dirpath, fname)))
  return results


# ---------------------------------------------------------------------------
# (1) Collate helper — handles variable-length tensors (e.g. coords)
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


class LengthBucketBatchSampler(Sampler[List[int]]):
  """Batch sampler that groups similarly sized variable-length samples.

  The sampler yields indices relative to the dataset passed to DataLoader. For
  shuffled training, it shuffles bucket and batch order while preserving
  length-adjacent samples inside each batch to reduce padding waste.
  """

  def __init__(
    self,
    lengths: Iterable[int],
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool = False,
    bucket_size_multiplier: int = 16,
    generator: Optional[torch.Generator] = None,
  ) -> None:
    self.lengths = [int(v) for v in lengths]
    self.batch_size = int(batch_size)
    self.shuffle = bool(shuffle)
    self.drop_last = bool(drop_last)
    self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))
    self.generator = generator
    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")

  def __iter__(self) -> Iterator[List[int]]:
    indices = list(range(len(self.lengths)))
    indices.sort(key=lambda i: self.lengths[i])
    buckets = [
      indices[i:i + self.bucket_size]
      for i in range(0, len(indices), self.bucket_size)
    ]

    if self.shuffle:
      bucket_order = torch.randperm(len(buckets), generator=self.generator).tolist()
    else:
      bucket_order = list(range(len(buckets)))

    for bucket_idx in bucket_order:
      bucket = list(buckets[bucket_idx])
      batches = [
        bucket[start:start + self.batch_size]
        for start in range(0, len(bucket), self.batch_size)
      ]
      if self.shuffle:
        batch_order = torch.randperm(len(batches), generator=self.generator).tolist()
      else:
        batch_order = list(range(len(batches)))
      for batch_idx in batch_order:
        batch = batches[batch_idx]
        if len(batch) == self.batch_size or not self.drop_last:
          yield batch

  def __len__(self) -> int:
    if self.drop_last:
      return len(self.lengths) // self.batch_size
    return (len(self.lengths) + self.batch_size - 1) // self.batch_size


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
  data_root: str,
  stains: List[str],
  patient_id_pattern: str = PATIENT_ID_PATTERN,
) -> Set[Tuple[str, str]]:
  """Return the intersection of (patient_id, tissue_id) pairs across stains.

  Scans data_root recursively, groups files by stain, then intersects.

  Usage:
      common = find_common_sample_keys('/data/Slide', ['HE', 'CD20', 'CD3'])

  Args:
      data_root: Feature root directory (patient/tissue structure or flat).
      stains: List of stain names to intersect.
      patient_id_pattern: Regex with group 1 matching the patient ID.

  Returns:
      Set of (patient_id, tissue_id) tuples present for every stain.
  """
  if not stains:
    return set()

  per_stain_keys: List[Set[Tuple[str, str]]] = []
  for stain in stains:
    keys: Set[Tuple[str, str]] = set()
    for fname, _ in _walk_h5_files(data_root, stain=stain):
      key = _extract_patient_tissue_id(fname, patient_id_pattern)
      if key is not None:
        keys.add(key)
    per_stain_keys.append(keys)

  result = per_stain_keys[0]
  for keys in per_stain_keys[1:]:
    result = result & keys
  return result


def fingerprint_sample_keys(sample_keys: Set[Tuple[str, str]]) -> str:
  """Return a stable fingerprint for a sample-key set."""
  payload = '\n'.join(f'{patient_id}|{tissue_id}' for patient_id, tissue_id in sorted(sample_keys))
  return hashlib.sha256(payload.encode('utf-8')).hexdigest()

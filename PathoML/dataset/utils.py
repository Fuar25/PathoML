"""Dataset-level utility functions."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

from ..config.defaults import PATIENT_ID_PATTERN


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

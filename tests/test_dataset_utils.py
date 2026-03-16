"""Tests for pure utility functions in data modules (no file I/O required)."""

import numpy as np

from PathoML.data.unimodal_dataset import _extract_patient_id
from PathoML.data.multimodal_dataset_concat import _normalize_feature_array
from PathoML.config.defaults import PATIENT_ID_PATTERN


# ---------------------------------------------------------------------------
# _extract_patient_id
# ---------------------------------------------------------------------------

def test_extract_patient_id_standard():
  result = _extract_patient_id("B2022-42849A-cd20.h5", PATIENT_ID_PATTERN)
  assert result == "B2022-42849"


def test_extract_patient_id_xs_prefix():
  result = _extract_patient_id("xsB2021-24069B-he.h5", PATIENT_ID_PATTERN)
  assert result == "xsB2021-24069"


def test_extract_patient_id_no_match():
  result = _extract_patient_id("random_file_no_id.h5", PATIENT_ID_PATTERN)
  assert result is None


def test_extract_patient_id_from_full_path():
  # (1) Function uses re.search so works on full paths too
  result = _extract_patient_id("/mnt/data/MALT/B2023-00001/A/B2023-00001A-he.h5", PATIENT_ID_PATTERN)
  assert result == "B2023-00001"


# ---------------------------------------------------------------------------
# _normalize_feature_array
# ---------------------------------------------------------------------------

def test_normalize_1d_array():
  # (1) 1D → (1, C): single-instance feature
  arr = np.ones(512)
  result = _normalize_feature_array(arr)
  assert result.shape == (1, 512)


def test_normalize_2d_array():
  # (2) 2D → unchanged
  arr = np.ones((10, 512))
  result = _normalize_feature_array(arr)
  assert result.shape == (10, 512)


def test_normalize_3d_array():
  # (3) >=3D → first dim kept as N, remaining dims flattened
  arr = np.ones((2, 5, 512))
  result = _normalize_feature_array(arr)
  assert result.shape == (2, 5 * 512)


def test_normalize_preserves_values():
  arr = np.array([1.0, 2.0, 3.0])
  result = _normalize_feature_array(arr)
  assert result.tolist() == [[1.0, 2.0, 3.0]]

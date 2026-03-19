"""Tests for pure utility functions in dataset module (no file I/O required)."""

from PathoML.dataset.utils import _extract_patient_tissue_id
from PathoML.config.defaults import PATIENT_ID_PATTERN


# ---------------------------------------------------------------------------
# _extract_patient_tissue_id
# ---------------------------------------------------------------------------

def test_extract_standard():
  result = _extract_patient_tissue_id("B2022-42849A-cd20.h5", PATIENT_ID_PATTERN)
  assert result == ("B2022-42849", "A")


def test_extract_xs_prefix():
  result = _extract_patient_tissue_id("xsB2021-24069B-he.h5", PATIENT_ID_PATTERN)
  assert result == ("xsB2021-24069", "B")


def test_extract_no_match():
  result = _extract_patient_tissue_id("random_file_no_id.h5", PATIENT_ID_PATTERN)
  assert result is None


def test_extract_from_filename_only():
  # (1) Function works on bare filenames (the expected use case)
  result = _extract_patient_tissue_id("B2023-00001A-he.h5", PATIENT_ID_PATTERN)
  assert result == ("B2023-00001", "A")

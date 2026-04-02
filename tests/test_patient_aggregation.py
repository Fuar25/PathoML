"""Tests for optimization.patient_aggregation: binary and multi-class patient-level MIL logic."""

import numpy as np
import pytest

from PathoML.optimization.patient_aggregation import aggregate_patient_predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary(slide_ids, patient_ids, probs, labels, threshold=0.5):
  return aggregate_patient_predictions(
    slide_ids, patient_ids,
    np.array(probs, dtype=float),
    np.array(labels, dtype=int),
    num_classes=1,
    threshold=threshold,
  )


# ---------------------------------------------------------------------------
# Binary aggregation
# ---------------------------------------------------------------------------

def test_binary_single_patient_positive():
  # (1) One tissue negative (0.3), one positive (0.8) → patient is positive
  sample_df, patient_df = _binary(
    ["P1-A", "P1-B"], ["P1", "P1"], [0.3, 0.8], [1, 1]
  )
  assert patient_df.loc[0, 'patient_pred'] == 1


def test_binary_single_patient_negative():
  sample_df, patient_df = _binary(
    ["P1-A", "P1-B"], ["P1", "P1"], [0.2, 0.3], [0, 0]
  )
  assert patient_df.loc[0, 'patient_pred'] == 0


def test_binary_multiple_patients():
  # (1) P1 → all negative; P2 → one tissue positive
  sample_df, patient_df = _binary(
    ["P1-A", "P2-A", "P2-B"],
    ["P1",   "P2",   "P2"],
    [0.2, 0.3, 0.8],
    [0, 1, 1],
  )
  by_patient = patient_df.set_index('patient_id')
  assert by_patient.loc['P1', 'patient_pred'] == 0
  assert by_patient.loc['P2', 'patient_pred'] == 1


def test_binary_patient_prob_is_max():
  # patient_prob = max over all tissues
  sample_df, patient_df = _binary(
    ["P1-A", "P1-B", "P1-C"],
    ["P1",   "P1",   "P1"],
    [0.2, 0.6, 0.4],
    [1, 1, 1],
  )
  assert patient_df.loc[0, 'patient_prob'] == pytest.approx(0.6)


def test_threshold_boundary():
  # prob == threshold → prediction 0 (rule: > threshold, strictly greater)
  sample_df, patient_df = _binary(
    ["P1-A"], ["P1"], [0.5], [1], threshold=0.5
  )
  assert patient_df.loc[0, 'patient_pred'] == 0


def test_sample_results_columns():
  sample_df, _ = _binary(["P1-A"], ["P1"], [0.7], [1])
  for col in ['slide_id', 'patient_id', 'slide_label', 'slide_prob', 'slide_pred']:
    assert col in sample_df.columns


def test_patient_results_columns():
  _, patient_df = _binary(["P1-A"], ["P1"], [0.7], [1])
  for col in ['patient_id', 'patient_prob', 'patient_pred', 'patient_label']:
    assert col in patient_df.columns


# ---------------------------------------------------------------------------
# Multi-class aggregation
# ---------------------------------------------------------------------------

def test_multiclass_aggregation():
  # (1) 3 tissues, 3 classes; per-class max then argmax
  # class maxima: [max(0.1,0.3,0.1), max(0.2,0.4,0.1), max(0.7,0.3,0.8)] = [0.3, 0.4, 0.8]
  # → predicted class 2
  probs = np.array([
    [0.1, 0.2, 0.7],
    [0.3, 0.4, 0.3],
    [0.1, 0.1, 0.8],
  ])
  labels = np.array([2, 2, 2])
  sample_df, patient_df = aggregate_patient_predictions(
    ["P1-A", "P1-B", "P1-C"],
    ["P1",   "P1",   "P1"],
    probs, labels,
    num_classes=3,
  )
  assert patient_df.loc[0, 'patient_pred'] == 2


def test_multiclass_patient_results_columns():
  probs = np.array([[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]])
  labels = np.array([1, 1])
  _, patient_df = aggregate_patient_predictions(
    ["P1-A", "P1-B"], ["P1", "P1"],
    probs, labels, num_classes=3,
  )
  for col in ['patient_id', 'patient_pred', 'patient_label',
              'patient_prob_class_0', 'patient_prob_class_1', 'patient_prob_class_2']:
    assert col in patient_df.columns

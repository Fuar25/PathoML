"""Patient-level aggregation: standard MIL logic — positive if any tissue is positive.

Standard practice in computational pathology: if any tissue slice from a patient
is classified as positive (probability > threshold), the patient is positive.
This matches clinical practice: "one finding is sufficient for diagnosis".
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def aggregate_patient_predictions(
  slide_ids: List[str],
  patient_ids: List[str],
  probs: np.ndarray,
  labels: np.ndarray,
  num_classes: int,
  threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate tissue-level predictions to patient-level (standard MIL logic).

  Binary aggregation rule:
    patient_prob = max(all tissue probs for that patient)
    patient_pred = 1 if patient_prob > threshold else 0

  Multi-class aggregation rule:
    for each class c: patient_prob[c] = max(tissue probs for class c)
    patient_pred = argmax(patient_probs)

  Args:
      slide_ids: Tissue-level sample ID list (length N).
      patient_ids: Corresponding patient ID list (length N).
      probs: Tissue-level predicted probabilities, shape (N,) for binary or (N,C) for multi-class.
      labels: Tissue-level ground-truth labels, shape (N,).
      num_classes: Number of classes (1 = binary, >1 = multi-class).
      threshold: Binary classification threshold (default 0.5).

  Returns:
      (sample_results, patient_results):
          - sample_results: Tissue-level DataFrame with slide_id, patient_id,
                            prediction, prob_positive (or prob_class_*), label.
          - patient_results: Patient-level aggregated DataFrame.
  """
  # (1) Build tissue-level results
  sample_results = _build_sample_results(
    slide_ids, patient_ids, probs, labels, num_classes, threshold
  )

  # (2) Aggregate to patient level
  if num_classes == 1:
    patient_results = _aggregate_binary(sample_results, threshold)
  else:
    patient_results = _aggregate_multiclass(sample_results)

  return sample_results, patient_results


def _build_sample_results(
  slide_ids: List[str],
  patient_ids: List[str],
  probs: np.ndarray,
  labels: np.ndarray,
  num_classes: int,
  threshold: float,
) -> pd.DataFrame:
  """Build tissue-level prediction results DataFrame."""
  results = {
    'slide_id': slide_ids,
    'patient_id': patient_ids,
    'label': labels.astype(int),
  }

  if num_classes == 1:
    results['prob_positive'] = probs
    results['prediction'] = (probs > threshold).astype(int)
  else:
    for i in range(probs.shape[1]):
      results[f'prob_class_{i}'] = probs[:, i]
    results['prediction'] = probs.argmax(axis=1)

  return pd.DataFrame(results)


def _aggregate_binary(sample_results: pd.DataFrame, threshold: float) -> pd.DataFrame:
  """Binary patient-level aggregation: positive if any tissue is positive."""
  patient_results = []

  for patient_id, group in sample_results.groupby('patient_id'):
    patient_prob = group['prob_positive'].max()  # standard MIL: take max
    patient_pred = int(patient_prob > threshold)
    patient_label = int(group['label'].max())    # MIL label: positive if any tissue positive

    patient_results.append({
      'patient_id': patient_id,
      'prob_positive': patient_prob,
      'prediction': patient_pred,
      'label': patient_label,
    })

  return pd.DataFrame(patient_results)


def _aggregate_multiclass(sample_results: pd.DataFrame) -> pd.DataFrame:
  """Multi-class patient-level aggregation: per-class max then argmax."""
  patient_results = []
  prob_cols = [c for c in sample_results.columns if c.startswith('prob_class_')]

  for patient_id, group in sample_results.groupby('patient_id'):
    patient_probs = group[prob_cols].max().to_dict()
    patient_pred = max(
      range(len(prob_cols)),
      key=lambda i: patient_probs[f'prob_class_{i}']
    )
    patient_label = int(group['label'].iloc[0])

    result = {'patient_id': patient_id, 'prediction': patient_pred, 'label': patient_label}
    result.update(patient_probs)
    patient_results.append(result)

  return pd.DataFrame(patient_results)

"""Training utilities: result containers and utility classes (non-fold-specific)."""

import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold


# ---------------------------------------------------------------------------
# (1) Result containers
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
  """Results container for an entire training run."""
  strategy_name: str
  fold_results: list
  result_dir: str


# ---------------------------------------------------------------------------
# (2) EarlyStopping — tracks validation AUC, saves checkpoint on improvement
# ---------------------------------------------------------------------------

class EarlyStopping:
  """Patience-based early stopping with integrated checkpoint management.

  Tracks val_auc (higher is better). Saves checkpoint only when a new global
  best is reached. Counter increments on non-improvement; resets on new best.
  """

  def __init__(self, patience: int, model: nn.Module, ckpt_path: str) -> None:
    self.patience = patience
    self.model = model
    self.ckpt_path = ckpt_path
    self.best_val_auc = float('-inf')
    self.patience_counter = 0
    self.best_epoch = 0

  def step(self, val_auc: float, current_epoch: int) -> bool:
    """Update state. Returns True if training should stop.

    Saves checkpoint when val_auc exceeds the global best (not just previous epoch).
    """
    if val_auc > self.best_val_auc:
      self.best_val_auc = val_auc
      self.best_epoch = current_epoch
      self.patience_counter = 0
      torch.save(self.model.state_dict(), self.ckpt_path)
      return False
    self.patience_counter += 1
    return self.patience_counter >= self.patience

  def load_best(self) -> None:
    """Restore best checkpoint weights into model."""
    self.model.load_state_dict(torch.load(self.ckpt_path, weights_only=True))

  def reset(self) -> None:
    self.best_val_auc = float('-inf')
    self.patience_counter = 0
    self.best_epoch = 0


# ---------------------------------------------------------------------------
# (3) stratified_patient_split — patient-level stratified split
# ---------------------------------------------------------------------------

def stratified_patient_split(
  indices: np.ndarray,
  patient_ids: np.ndarray,
  labels: np.ndarray,
  n_splits: int,
  seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
  """Patient-level stratified split, mapped back to slide indices.

  Assumption: all slides of the same patient share the same label (patient-level
  diagnosis). The label is taken from the first occurrence of each patient.

  (1) Deduplicate to unique patients with labels.
  (2) StratifiedKFold on patients (binary & multi-class).
      Fallback to KFold if any class has fewer patients than n_splits.
  (3) Map patient splits back to slide indices.

  Args:
    indices:     Slide indices (e.g. np.arange(len(dataset)) or a subset).
    patient_ids: Patient ID for each slide (same length as indices).
    labels:      Label for each slide (same length as indices).
    n_splits:    Number of folds to split patients into.
    seed:        Random seed for reproducibility.

  Returns:
    List of (group_a_indices, group_b_indices) tuples, length = n_splits.
  """
  # (1) Deduplicate: unique patients and their labels
  unique_patients, first_idx = np.unique(patient_ids, return_index=True)
  patient_labels = labels[first_idx].astype(int)
  n_patients = len(unique_patients)

  # (1.1) Build patient_id -> [slide indices] mapping
  patient_to_slides = {}
  for i, pid in enumerate(patient_ids):
    patient_to_slides.setdefault(pid, []).append(indices[i])

  # (2) Stratified K-fold on patients
  #     Cap n_splits so each fold has ≥1 patient per class
  min_class_count = min(Counter(patient_labels).values())
  effective_splits = min(n_splits, n_patients, min_class_count)
  if effective_splits < 2:
    return [(indices, np.array([], dtype=int))]
  if effective_splits < n_splits:
    warnings.warn(
      f"n_splits={n_splits} capped to {effective_splits} "
      f"(min class has only {min_class_count} patients, total {n_patients})")

  splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
  fold_iter = splitter.split(unique_patients, patient_labels)

  # (3) Map patient-level folds back to slide indices
  splits = []
  for p_group_a, p_group_b in fold_iter:
    slide_a = np.concatenate([patient_to_slides[unique_patients[p]] for p in p_group_a])
    slide_b = np.concatenate([patient_to_slides[unique_patients[p]] for p in p_group_b])
    splits.append((slide_a, slide_b))

  return splits

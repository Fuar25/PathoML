"""K-fold cross-validation strategy."""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from PathoML.config.config import RunTimeConfig, TrainingConfig
from PathoML.interfaces import BaseDataset
from ..training_utils import TrainingResult, EarlyStopping
from ..patient_aggregation import aggregate_patient_predictions
from .training_base import TrainingMixin, Strategy


# ---------------------------------------------------------------------------
# (1) Fold-specific data containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
  """Results container for a single fold."""
  fold: int
  best_epoch: int
  val_loss: float
  test_loss: float
  test_acc: float
  test_auc: float
  patient_acc: Optional[float] = None
  patient_auc: Optional[float] = None
  checkpoint_name: Optional[str] = None


# ---------------------------------------------------------------------------
# (2) CrossValidator — K-fold cross-validation scheduler
# ---------------------------------------------------------------------------

class CrossValidator(Strategy, TrainingMixin):
  """K-fold cross-validation scheduler.

  Inherits TrainingMixin for all training loop and dataset utilities.

  Usage:
      strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
      Trainer(strategy).fit()
  """

  def __init__(
    self,
    model_builder: Callable,
    dataset: BaseDataset,
    config: RunTimeConfig,
    k_folds: int = 5,
  ) -> None:
    # (1) Store all dependencies so execute() needs no arguments
    self.model_builder = model_builder
    self.dataset = dataset
    self.model_cfg = config.model
    self.training_cfg = config.training
    self.logging_cfg = config.logging
    self.device = torch.device(config.training.device)
    self.k_folds = k_folds
    n_classes = len(dataset.classes)
    self.num_classes = 1 if n_classes == 2 else n_classes

  @property
  def name(self) -> str:
    return "KFold CrossValidation"

  # -- Public entry point --

  def execute(self) -> TrainingResult:
    """Run K-fold CV. Each fold: split → train → evaluate → aggregate."""
    os.makedirs(self.logging_cfg.save_dir, exist_ok=True)
    split_iter, patient_ids = self._prepare_splits()

    fold_results = []
    all_test_details = []   # accumulate across folds for combined CSV
    cv_start = time.time()

    for fold, (train_val_ids, test_ids) in enumerate(split_iter):
      print(f"\n{'='*70}\nFold {fold+1} / {self.k_folds}\n{'='*70}")

      result, test_details = self._train_single_fold(
        fold=fold + 1,
        train_val_ids=train_val_ids,
        test_ids=test_ids,
        patient_ids=patient_ids,
      )
      fold_results.append(result)
      all_test_details.append(test_details)

    self._print_cv_summary(fold_results, time.time() - cv_start)
    self._save_cv_predictions(all_test_details)

    return TrainingResult(
      strategy_name=self.name,
      fold_results=fold_results,
      result_dir=self.logging_cfg.save_dir,
    )

  # -- Fold orchestration --

  def _prepare_splits(self) -> Tuple:
    """Build the K-fold split iterator. Returns (split_iter, patient_ids_array)."""
    all_indices = np.arange(len(self.dataset))
    patient_ids = np.array(self.dataset.get_patient_ids())
    labels = np.array([float(self.dataset[i]['label']) for i in range(len(self.dataset))])

    # (1) Binary: use stratified group split to balance class ratio across folds
    if self.num_classes == 1:
      splitter = StratifiedGroupKFold(
        n_splits=self.k_folds, shuffle=True, random_state=self.training_cfg.seed
      )
      split_iter = splitter.split(all_indices, labels, groups=patient_ids)
    else:
      splitter = GroupKFold(n_splits=self.k_folds)
      split_iter = splitter.split(all_indices, groups=patient_ids)

    return split_iter, patient_ids

  def _train_single_fold(
    self,
    fold: int,
    train_val_ids: np.ndarray,
    test_ids: np.ndarray,
    patient_ids: np.ndarray,
  ) -> Tuple['FoldResult', Dict[str, Any]]:
    """Full pipeline for one fold: split → build loaders → train → evaluate.

    Returns (FoldResult, test_details_dict).
    """
    # (1) Patient-aware train/val split
    train_ids, val_ids = self._split_train_val(
      train_val_ids, patient_ids[train_val_ids], seed=self.training_cfg.seed + fold
    )

    # (2) Build DataLoaders
    train_loader, val_loader, test_loader = self._build_loaders(train_ids, val_ids, test_ids)

    # (3) Initialize model, criterion, optimizer
    model = self.model_builder().to(self.device)
    criterion = self._build_criterion(self.num_classes)
    optimizer = self._build_optimizer(model)

    # (4) EarlyStopping manages checkpoint internally
    ckpt_path = os.path.join(self.logging_cfg.save_dir, f'model_fold_{fold}_best.pth')
    early_stopping = EarlyStopping(self.training_cfg.patience, model, ckpt_path)

    # (5) Run training loop
    self._run_train_val_loop(
      model, criterion, optimizer, early_stopping,
      train_loader, val_loader, label=f"fold_{fold}",
    )

    # (6) Restore best weights and evaluate on test set
    early_stopping.load_best()
    test_loss, test_acc, test_auc, test_details = self._evaluate_with_auc(
      model, test_loader, criterion
    )

    # (7) Compute patient-level metrics (runtime, no disk I/O)
    patient_acc, patient_auc = self._compute_patient_metrics(test_details)

    result = FoldResult(
      fold=fold,
      best_epoch=early_stopping.best_epoch,
      val_loss=early_stopping.best_val_loss,
      test_loss=test_loss,
      test_acc=test_acc,
      test_auc=test_auc,
      patient_acc=patient_acc,
      patient_auc=patient_auc,
      checkpoint_name=os.path.basename(ckpt_path),
    )
    return result, test_details

  # -- CV predictions CSV --

  def _save_cv_predictions(self, all_test_details: List[Dict[str, Any]]) -> None:
    """Save combined test-set predictions from all folds to a single CSV.

    Concatenates tissue-level details across all folds (each sample appears in
    exactly one test fold), runs patient-level aggregation on the combined data,
    then merges slide- and patient-level columns into one CSV file.

    Columns (binary):
        slide_id, patient_id, slide_label, slide_prob, slide_pred,
        patient_label, patient_prob, patient_pred
    Multi-class expands _prob columns to slide_prob_class_*, patient_prob_class_*.
    """
    # (1) Concatenate all fold details
    all_sample_ids, all_patient_ids = [], []
    all_probs_list, all_labels_list = [], []
    for d in all_test_details:
      all_sample_ids.extend(d['sample_ids'])
      all_patient_ids.extend(d['patient_ids'])
      all_probs_list.append(d['probs'])
      all_labels_list.append(d['labels'])

    combined_probs = np.concatenate(all_probs_list, axis=0)
    combined_labels = np.concatenate(all_labels_list, axis=0)

    # (2) Aggregate to patient level
    sample_results, patient_results = aggregate_patient_predictions(
      sample_ids=all_sample_ids,
      patient_ids=all_patient_ids,
      probs=combined_probs,
      labels=combined_labels,
      num_classes=self.num_classes,
      threshold=self.training_cfg.patient_threshold,
    )

    # (3) Rename columns and merge slide + patient level
    if self.num_classes == 1:
      slide_df = sample_results.rename(columns={
        'sample_id': 'slide_id',
        'label': 'slide_label',
        'prob_positive': 'slide_prob',
        'prediction': 'slide_pred',
      })
      patient_df = patient_results.rename(columns={
        'label': 'patient_label',
        'prob_positive': 'patient_prob',
        'prediction': 'patient_pred',
      })[['patient_id', 'patient_label', 'patient_prob', 'patient_pred']]
    else:
      prob_cols = [c for c in sample_results.columns if c.startswith('prob_class_')]
      slide_rename = {'sample_id': 'slide_id', 'label': 'slide_label', 'prediction': 'slide_pred'}
      slide_rename.update({c: f'slide_{c}' for c in prob_cols})
      slide_df = sample_results.rename(columns=slide_rename)

      pat_prob_cols = [c for c in patient_results.columns if c.startswith('prob_class_')]
      pat_rename = {'label': 'patient_label', 'prediction': 'patient_pred'}
      pat_rename.update({c: f'patient_{c}' for c in pat_prob_cols})
      patient_df = patient_results.rename(columns=pat_rename)
      patient_df = patient_df[
        ['patient_id', 'patient_label', 'patient_pred']
        + [f'patient_{c}' for c in pat_prob_cols]
      ]

    df = slide_df.merge(patient_df, on='patient_id', how='left')

    # (4) Save
    save_path = os.path.join(self.logging_cfg.save_dir, 'cv_predictions.csv')
    df.to_csv(save_path, index=False)
    print(f"  CV predictions saved to {save_path}")

  # -- Summary printing --

  def _print_cv_summary(self, fold_results: List['FoldResult'], total_time: float) -> None:
    """Print cross-validation summary statistics."""
    test_aucs    = [r.test_auc    for r in fold_results]
    test_accs    = [r.test_acc    for r in fold_results]
    best_epochs  = [r.best_epoch  for r in fold_results]
    patient_aucs = [r.patient_auc for r in fold_results]
    patient_accs = [r.patient_acc for r in fold_results]

    print(f"\n{'='*70}")
    print("Cross-Validation Summary:")
    print(f"{'='*70}")
    print(f"  Avg Test AUC:    {np.nanmean(test_aucs):.4f} ± {np.nanstd(test_aucs):.4f}")
    print(f"  Avg Test Acc:    {np.nanmean(test_accs):.4f} ± {np.nanstd(test_accs):.4f}")
    print(f"  Avg Patient AUC: {np.nanmean(patient_aucs):.4f} ± {np.nanstd(patient_aucs):.4f}")
    print(f"  Avg Patient Acc: {np.nanmean(patient_accs):.4f} ± {np.nanstd(patient_accs):.4f}")
    print(f"  Avg Best Epoch:  {np.nanmean(best_epochs):.1f} ± {np.nanstd(best_epochs):.1f}")
    print(f"  Total Time:      {total_time/60:.2f} min")
    print(f"{'='*70}")

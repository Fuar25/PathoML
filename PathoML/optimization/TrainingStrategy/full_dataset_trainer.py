"""Full-dataset training strategy for deployment."""

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from PathoML.config.config import RunTimeConfig
from PathoML.interfaces import BaseDataset
from ..training_utils import (
  TrainingResult, EarlyStopping,
  set_seed, build_criterion, build_optimizer, build_loaders, split_train_val,
)
from .training_base import Strategy, TrainingMixin


# ---------------------------------------------------------------------------
# (1) FullTrainingResult — result container for full-dataset training
# ---------------------------------------------------------------------------

@dataclass
class FullTrainingResult:
  """Results container for full-dataset training."""
  best_epoch: int
  val_loss: float
  val_acc: float
  val_auc: float
  patient_acc: float
  patient_auc: float
  patient_f1: float


# ---------------------------------------------------------------------------
# (2) FullDatasetTrainer — full-dataset training strategy
# ---------------------------------------------------------------------------

class FullDatasetTrainer(Strategy, TrainingMixin):
  """Full-dataset trainer for deployment.

  Uses a 9:1 patient-aware val split for early stopping to identify the best
  checkpoint. The final deployment model is the best-val-loss checkpoint.

  Usage:
      strategy = FullDatasetTrainer(model_builder, dataset, config)
      Trainer(strategy).fit()
  """

  def __init__(
    self,
    model_builder: Callable,
    dataset: BaseDataset,
    config: RunTimeConfig,
  ) -> None:
    self.model_builder = model_builder
    self.dataset = dataset
    self.model_cfg = config.model
    self.training_cfg = config.training
    self.logging_cfg = config.logging
    self.device = torch.device(config.training.device)
    n_classes = len(dataset.classes)
    self.num_classes = 1 if n_classes == 2 else n_classes

  @property
  def name(self) -> str:
    return "FullDataset Training"

  def execute(self) -> TrainingResult:
    """Train on full dataset with early stopping and save best checkpoint."""
    os.makedirs(self.logging_cfg.save_dir, exist_ok=True)

    # (1) Fix RNG + patient-aware 9:1 train/val split
    set_seed(self.training_cfg.seed)
    all_ids = np.arange(len(self.dataset))
    patient_ids = np.array(self.dataset.get_patient_ids())
    train_ids, val_ids = split_train_val(self.dataset, all_ids, patient_ids, self.training_cfg.seed)

    train_loader, val_loader = build_loaders(
      self.dataset, train_ids, val_ids, training_cfg=self.training_cfg,
    )

    # (2) Build model, criterion, optimizer
    model = self.model_builder().to(self.device)
    criterion = build_criterion(self.num_classes)
    optimizer = build_optimizer(model, self.training_cfg)

    # (3) EarlyStopping manages checkpoint
    ckpt_path = os.path.join(self.logging_cfg.save_dir, 'model_training_best.pth')
    early_stopping = EarlyStopping(
      self.training_cfg.patience,
      model,
      ckpt_path,
      monitor_name=self.training_cfg.early_stopping_metric,
      min_delta=self.training_cfg.min_delta,
    )

    # (4) Train with early stopping
    self._run_train_val_loop(
      model, criterion, optimizer, early_stopping,
      train_loader, val_loader, label='deployment',
    )

    # (5) Restore best weights, evaluate val set, compute patient metrics
    early_stopping.load_best()
    val_loss, val_acc, val_auc, val_details = self._evaluate_with_auc(model, val_loader, criterion)
    patient_acc, patient_auc, patient_f1 = self._compute_patient_metrics(val_details)

    # (6) Save deployment model and print summary
    deploy_path = os.path.join(self.logging_cfg.save_dir, 'model_deployment.pth')
    torch.save(model.state_dict(), deploy_path)

    result = FullTrainingResult(
      best_epoch=early_stopping.best_epoch,
      val_loss=val_loss,
      val_acc=val_acc,
      val_auc=val_auc,
      patient_acc=patient_acc,
      patient_auc=patient_auc,
      patient_f1=patient_f1,
    )
    self._print_training_summary(result, deploy_path)
    return TrainingResult(
      strategy_name=self.name,
      fold_results=[],
      result_dir=self.logging_cfg.save_dir,
    )

  def _print_training_summary(self, result: FullTrainingResult, deploy_path: str) -> None:
    print(f"\n{'='*70}")
    print("Full Dataset Training Summary:")
    print(f"{'='*70}")
    print(f"  Best Epoch:    {result.best_epoch}")
    print(f"  Val Loss:      {result.val_loss:.4f}")
    print(f"  Val Acc:       {result.val_acc:.4f}")
    print(f"  Val AUC:       {result.val_auc:.4f}")
    print(f"  Patient Acc:   {result.patient_acc:.4f}")
    print(f"  Patient AUC:   {result.patient_auc:.4f}")
    print(f"  Patient F1:    {result.patient_f1:.4f}")
    print(f"  Model saved -> {deploy_path}")
    print(f"{'='*70}")

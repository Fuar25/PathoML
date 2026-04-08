"""Distillation runtime adapter built on top of PathoML cross-validation."""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from PathoML.config.config import RunTimeConfig
from PathoML.interfaces import BaseDataset
from PathoML.optimization.TrainingStrategy import CrossValidator
from PathoML.optimization.training_utils import TrainingResult, move_to_device, model_inputs
from distillation.losses import DistillationLoss
from distillation.models.teacher import TeacherMLP


class DistillCrossValidator(CrossValidator):
  """Cross-validator with teacher checkpoint loading and distillation loss injection."""

  def __init__(
    self,
    student_builder: Callable[[], nn.Module],
    dataset: BaseDataset,
    config: RunTimeConfig,
    distill_loss: DistillationLoss,
    teacher_ckpt_tmpl: str,
    k_folds: int = 5,
  ) -> None:
    super().__init__(student_builder, dataset, config, k_folds)
    self.teacher_ckpt_tmpl = teacher_ckpt_tmpl
    self.distill_loss = distill_loss
    self.teacher = None

  @property
  def name(self) -> str:
    return "KFold CrossValidation (Distillation)"

  def execute(self) -> TrainingResult:
    """Run K-fold CV while validating the teacher checkpoint for each fold."""
    os.makedirs(self.logging_cfg.save_dir, exist_ok=True)
    split_iter, patient_ids = self._prepare_splits()

    fold_results, all_test_details = [], []
    cv_start = time.time()

    for fold, (train_val_ids, test_ids) in enumerate(split_iter):
      print(f"\n{'='*70}\nFold {fold+1} / {self.k_folds}\n{'='*70}")
      self.teacher = TeacherMLP.from_checkpoint(
        self.teacher_ckpt_tmpl.format(fold=fold + 1)
      ).to(self.device)

      if self.teacher.test_fold is not None:
        s_train = sorted(set(patient_ids[train_val_ids].tolist()))
        s_test = sorted(set(patient_ids[test_ids].tolist()))
        assert s_train == self.teacher.train_fold, (
          f"Fold {fold+1}: train_fold mismatch between teacher and distillation dataset."
        )
        assert s_test == self.teacher.test_fold, (
          f"Fold {fold+1}: test_fold mismatch between teacher and distillation dataset."
        )
      else:
        raise ValueError(
          f"Fold {fold+1}: teacher checkpoint lacks train_fold/test_fold and cannot be verified."
        )
      print(f"Fold {fold+1}: Teacher checkpoint loaded, fold splits verified.")

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

  def _train_epoch(
    self,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
  ):
    """Train one distillation epoch with frozen teacher forward passes."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc="  Training", unit="batch", leave=False) as pbar:
      for raw_batch in pbar:
        batch = move_to_device(raw_batch, self.device)
        labels = batch['label']

        with torch.no_grad():
          t_out = self.teacher(batch['slide_concat'])

        inputs = model_inputs(batch)
        s_out = model(inputs)

        loss = self.distill_loss(s_out, t_out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        probs = torch.sigmoid(s_out['logits'].view(-1))
        preds = (probs > self.training_cfg.patient_threshold).float()
        correct += (preds == labels.view(-1).float()).sum().item()
        total += batch_size
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total

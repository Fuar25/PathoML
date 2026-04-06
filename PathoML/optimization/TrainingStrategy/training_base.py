"""Base classes for training strategies: TrainingMixin and Strategy ABC.

TrainingMixin   — training loop and evaluation logic inherited by all strategies.
Strategy        — abstract base requiring name + execute().

Dependencies that concrete classes must set in __init__:
  self.device          — torch.device
  self.dataset         — BaseDataset
  self.num_classes     — int  (1 for binary, >1 for multi-class)
  self.training_cfg    — TrainingConfig
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..training_utils import (
  move_to_device, model_inputs, forward_and_decode, compute_auc,
)
from ..patient_aggregation import aggregate_patient_predictions


# ---------------------------------------------------------------------------
# (1) TrainingMixin — training loop and evaluation
# ---------------------------------------------------------------------------

class TrainingMixin:
  """Training loop and evaluation logic.

  Subclasses must set self.device, self.dataset, self.num_classes, self.training_cfg.
  Build helpers (set_seed, build_criterion, build_optimizer, build_loaders,
  split_train_val) are standalone functions in training_utils.py.
  """

  def _run_train_val_loop(
    self,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label: str = '',
    log_dir: Optional[str] = None,
  ) -> None:
    """Epoch loop until early stopping. EarlyStopping saves checkpoint on improvement.

    Args:
      log_dir: if provided, writes TensorBoard events to this directory.
    """
    writer = SummaryWriter(log_dir) if log_dir else None

    if self.training_cfg.scheduler == 'cosine':
      scheduler = CosineAnnealingLR(optimizer, T_max=self.training_cfg.epochs)
    else:
      scheduler = None

    start = time.time()
    with tqdm(range(self.training_cfg.epochs), desc=f"  {label}", leave=True) as pbar:
      for epoch in pbar:
        train_loss, _ = self._train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, _ = self._evaluate_with_auc(model, val_loader, criterion)
        if scheduler:
          scheduler.step()

        # TensorBoard logging
        if writer:
          writer.add_scalar('Loss/train', train_loss, epoch)
          writer.add_scalar('Loss/val', val_loss, epoch)
          writer.add_scalar('AUC/val', val_auc, epoch)

        should_stop = early_stopping.step(val_auc, epoch + 1)

        elapsed = time.time() - start
        eta = elapsed / (epoch + 1) * (self.training_cfg.epochs - epoch - 1)
        pbar.set_postfix(
          train_loss=f"{train_loss:.4f}",
          val_loss=f"{val_loss:.4f}",
          val_auc=f"{val_auc:.4f}",
          ETA=f"{int(eta//60)}:{int(eta%60):02d}",
        )
        if should_stop:
          tqdm.write(f"  Early stop @ epoch {epoch+1}, best={early_stopping.best_epoch}")
          break

    if writer:
      writer.close()

  def _train_epoch(
    self,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
  ) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc="  Training", unit="batch", leave=False) as pbar:
      for raw_batch in pbar:
        batch = move_to_device(raw_batch, self.device)
        inputs = model_inputs(batch)
        labels = batch['label']

        optimizer.zero_grad()
        logits = model(inputs)['logits']
        loss, probs, preds = forward_and_decode(
          logits, labels, criterion, self.num_classes, self.training_cfg.patient_threshold,
        )
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total

  def _evaluate_with_auc(
    self,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
  ) -> Tuple[float, float, float, Dict[str, Any]]:
    """Evaluate model on a loader. Returns (loss, acc, auc, details).

    details dict contains slide_ids, patient_ids, probs, labels — used for
    patient-level aggregation.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    all_slide_ids, all_patient_ids = [], []

    with torch.no_grad():
      for raw_batch in loader:
        batch = move_to_device(raw_batch, self.device)
        inputs = model_inputs(batch)
        labels = batch['label']
        slide_ids = batch.get('slide_id', ['unknown'] * len(labels))
        patient_ids_b = batch.get('patient_id', slide_ids)

        logits = model(inputs)['logits']
        loss, probs, preds = forward_and_decode(
          logits, labels, criterion, self.num_classes, self.training_cfg.patient_threshold,
        )

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_slide_ids.extend(slide_ids)
        all_patient_ids.extend(patient_ids_b)

    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)
    auc = compute_auc(labels_np, probs_np, self.num_classes)

    details = {
      'slide_ids': all_slide_ids,
      'patient_ids': all_patient_ids,
      'probs': probs_np,
      'labels': labels_np,
    }
    return total_loss / total, correct / total, auc, details

  def _compute_patient_metrics(
    self, eval_details: Dict[str, Any]
  ) -> Tuple[float, float, float]:
    """Compute patient-level accuracy, AUC, and F1 from in-memory eval predictions."""
    from sklearn.metrics import f1_score

    _, patient_results = aggregate_patient_predictions(
      slide_ids=eval_details['slide_ids'],
      patient_ids=eval_details['patient_ids'],
      probs=eval_details['probs'],
      labels=eval_details['labels'],
      num_classes=self.num_classes,
      threshold=self.training_cfg.patient_threshold,
    )

    p_labels = patient_results['patient_label'].values
    p_preds  = patient_results['patient_pred'].values
    patient_acc = float((p_labels == p_preds).mean())

    if self.num_classes == 1:
      p_probs = patient_results['patient_prob'].values
    else:
      prob_cols = [c for c in patient_results.columns if c.startswith('patient_prob_class_')]
      p_probs = patient_results[prob_cols].values
    patient_auc = compute_auc(p_labels, p_probs, self.num_classes)

    # (1) Patient-level F1
    try:
      avg = 'binary' if self.num_classes == 1 else 'macro'
      patient_f1 = float(f1_score(p_labels.astype(int), p_preds.astype(int), average=avg))
    except Exception as e:
      print(f"Warning: F1 computation failed: {e}")
      patient_f1 = float('nan')

    return patient_acc, patient_auc, patient_f1


# ---------------------------------------------------------------------------
# (2) Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
  """Abstract training strategy. Concrete classes own all dependencies."""

  @property
  @abstractmethod
  def name(self) -> str: ...

  @abstractmethod
  def execute(self): ...

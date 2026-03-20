"""Base classes for training strategies: TrainingMixin and Strategy ABC.

TrainingMixin   — shared epoch/eval logic inherited by all strategies.
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..training_utils import EarlyStopping, TrainingResult
from ..patient_aggregation import aggregate_patient_predictions


# ---------------------------------------------------------------------------
# (1) TrainingMixin — shared epoch/eval logic
# ---------------------------------------------------------------------------

class TrainingMixin:
  """Shared training loop, evaluation, and dataset utilities.

  Subclasses must set self.device, self.dataset, self.num_classes, self.training_cfg.
  Override _move_to_device or _model_inputs for non-standard batch layouts.
  """

  # -- Device / input helpers --

  def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Move all tensor values in a batch dict to self.device."""
    return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

  @staticmethod
  def _model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-model keys from batch dict (labels, IDs, paths)."""
    _exclude = {'label', 'sample_id', 'patient_id', 'feature_path', 'tissue_id', 'modalities'}
    return {k: v for k, v in batch.items() if k not in _exclude}

  # -- Build helpers --

  @staticmethod
  def _set_seed(seed: int) -> None:
    """Fix torch and CUDA RNG for reproducible weight init and dropout."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  @staticmethod
  def _build_criterion(num_classes: int) -> nn.Module:
    return nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

  def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
    """Build Adam optimizer from self.training_cfg hyperparameters."""
    return torch.optim.Adam(
      model.parameters(),
      lr=self.training_cfg.learning_rate,
      weight_decay=self.training_cfg.weight_decay,
    )

  def _build_loaders(
    self,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: Optional[np.ndarray] = None,
  ):
    """Build DataLoaders from index arrays.

    Returns (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    depending on whether test_ids is provided.
    """
    bs = self.training_cfg.batch_size
    train_loader = DataLoader(Subset(self.dataset, train_ids), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(self.dataset, val_ids),   batch_size=bs, shuffle=False)
    if test_ids is not None:
      test_loader = DataLoader(Subset(self.dataset, test_ids), batch_size=bs, shuffle=False)
      return train_loader, val_loader, test_loader
    return train_loader, val_loader

  def _split_train_val(
    self, indices: np.ndarray, patient_ids: np.ndarray, seed: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Patient-aware 9:1 train/val split.

    Shuffles unique patients with the given seed, reserves the first 10% as val,
    then maps back to sample indices. Returns (train_ids, val_ids).
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    rng.shuffle(unique_patients)
    n_val = max(1, int(len(unique_patients) * 0.1))
    val_patients = set(unique_patients[:n_val])
    val_mask = np.isin(patient_ids, list(val_patients))
    return indices[~val_mask], indices[val_mask]

  # -- Training loop --

  def _run_train_val_loop(
    self,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label: str = '',
  ) -> None:
    """Epoch loop until early stopping. EarlyStopping saves checkpoint on improvement."""
    start = time.time()
    with tqdm(range(self.training_cfg.epochs), desc=f"  {label}", leave=True) as pbar:
      for epoch in pbar:
        train_loss, _ = self._train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, _ = self._evaluate_with_auc(model, val_loader, criterion)

        should_stop = early_stopping.step(val_loss, epoch + 1)

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
        batch = self._move_to_device(raw_batch)
        inputs = self._model_inputs(batch)
        labels = batch['label']

        optimizer.zero_grad()
        logits = model(inputs)['logits']
        loss, probs, preds = self._forward_and_decode(logits, labels, criterion)
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

    details dict contains sample_ids, patient_ids, probs, labels — used for
    patient-level aggregation.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    all_sample_ids, all_patient_ids = [], []

    with torch.no_grad():
      for raw_batch in loader:
        batch = self._move_to_device(raw_batch)
        inputs = self._model_inputs(batch)
        labels = batch['label']
        sample_ids = batch.get('sample_id', ['unknown'] * len(labels))
        patient_ids_b = batch.get('patient_id', sample_ids)

        logits = model(inputs)['logits']
        loss, probs, preds = self._forward_and_decode(logits, labels, criterion)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_sample_ids.extend(sample_ids)
        all_patient_ids.extend(patient_ids_b)

    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)
    auc = self._compute_auc(labels_np, probs_np)

    details = {
      'sample_ids': all_sample_ids,
      'patient_ids': all_patient_ids,
      'probs': probs_np,
      'labels': labels_np,
    }
    return total_loss / total, correct / total, auc, details

  def _forward_and_decode(
    self,
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unify binary vs multi-class loss + prob + pred computation.

    Returns (loss, probs, preds). Binary threshold is read from
    self.training_cfg.patient_threshold.
    """
    if self.num_classes == 1:
      logits = logits.view(-1)
      labels = labels.view(-1).float()
      loss = criterion(logits, labels)
      probs = torch.sigmoid(logits)
      preds = (probs > self.training_cfg.patient_threshold).float()
    else:
      loss = criterion(logits, labels.long())
      probs = torch.softmax(logits, dim=1)
      preds = torch.argmax(probs, dim=1)
    return loss, probs, preds

  def _compute_auc(self, labels_np: np.ndarray, probs_np: np.ndarray) -> float:
    """Compute AUC, returning nan on failure (e.g. single-class batch)."""
    try:
      if self.num_classes == 1:
        return roc_auc_score(labels_np.astype(int), probs_np)
      return roc_auc_score(
        labels_np.astype(int), probs_np,
        multi_class='ovr', average='macro',
      )
    except Exception as e:
      print(f"Warning: AUC computation failed: {e}")
      return float('nan')

  def _compute_patient_metrics(
    self, eval_details: Dict[str, Any]
  ) -> Tuple[float, float]:
    """Compute patient-level accuracy and AUC from in-memory eval predictions."""
    _, patient_results = aggregate_patient_predictions(
      sample_ids=eval_details['sample_ids'],
      patient_ids=eval_details['patient_ids'],
      probs=eval_details['probs'],
      labels=eval_details['labels'],
      num_classes=self.num_classes,
      threshold=self.training_cfg.patient_threshold,
    )

    p_labels = patient_results['label'].values
    p_preds  = patient_results['prediction'].values
    patient_acc = float((p_labels == p_preds).mean())

    if self.num_classes == 1:
      p_probs = patient_results['prob_positive'].values
    else:
      prob_cols = [c for c in patient_results.columns if c.startswith('prob_class_')]
      p_probs = patient_results[prob_cols].values
    patient_auc = self._compute_auc(p_labels, p_probs)

    return patient_acc, patient_auc


# ---------------------------------------------------------------------------
# (2) Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
  """Abstract training strategy. Concrete classes own all dependencies."""

  @property
  @abstractmethod
  def name(self) -> str: ...

  @abstractmethod
  def execute(self) -> TrainingResult: ...

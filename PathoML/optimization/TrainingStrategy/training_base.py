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
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..training_utils import EarlyStopping, TrainingResult, stratified_patient_split
from ..patient_aggregation import aggregate_patient_predictions


# ---------------------------------------------------------------------------
# (0) Collate helper — handles variable-length tensors (e.g. coords)
# ---------------------------------------------------------------------------

def _variable_size_collate(batch):
  """Collate that pads variable-length tensors and creates a shared mask.

  When tensors differ only along dim 0 (e.g. features (N, D) with varying N),
  pads to max length and stores a boolean mask under 'mask'.
  Convention: True = valid position, False = padding.
  """
  elem = batch[0]
  result = {}

  # (1) First pass: detect variable-length dim-0 tensors, compute max_len
  #     All variable-length tensors in a WSI sample share the same N,
  #     so one mask suffices.
  max_len = None
  lengths = None
  for key in elem:
    values = [d[key] for d in batch]
    if torch.is_tensor(values[0]) and values[0].dim() >= 1:
      sizes_0 = [v.shape[0] for v in values]
      if not all(s == sizes_0[0] for s in sizes_0):
        rest_shape = values[0].shape[1:]
        if all(v.shape[1:] == rest_shape for v in values):
          max_len = max(max_len or 0, max(sizes_0))
          if lengths is None:
            lengths = sizes_0

  # (2) Second pass: collate each key
  for key in elem:
    values = [d[key] for d in batch]
    if torch.is_tensor(values[0]):
      if all(v.shape == values[0].shape for v in values):
        result[key] = torch.stack(values, 0)
      elif max_len is not None and values[0].dim() >= 1:
        rest_shape = values[0].shape[1:]
        if all(v.shape[1:] == rest_shape for v in values):
          padded = torch.zeros(len(values), max_len, *rest_shape,
                               dtype=values[0].dtype)
          for i, v in enumerate(values):
            padded[i, :v.shape[0]] = v
          result[key] = padded
        else:
          result[key] = values
      else:
        result[key] = values
    elif isinstance(values[0], (int, float)):
      result[key] = torch.tensor(values)
    else:
      result[key] = values

  # (3) Create shared mask if padding occurred
  if max_len is not None and lengths is not None:
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
      mask[i, :length] = True
    result['mask'] = mask

  return result


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
    out = {}
    for k, v in batch.items():
      if torch.is_tensor(v):
        out[k] = v.to(self.device)
      elif isinstance(v, list) and v and torch.is_tensor(v[0]):
        out[k] = [t.to(self.device) for t in v]
      else:
        out[k] = v
    return out

  @staticmethod
  def _model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-model keys from batch dict (labels, IDs, paths)."""
    _exclude = {'label', 'slide_id', 'patient_id', 'feature_path', 'tissue_id', 'modalities'}
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
    collate = _variable_size_collate if bs > 1 else None
    train_loader = DataLoader(Subset(self.dataset, train_ids), batch_size=bs, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(Subset(self.dataset, val_ids),   batch_size=bs, shuffle=False, collate_fn=collate)
    if test_ids is not None:
      test_loader = DataLoader(Subset(self.dataset, test_ids), batch_size=bs, shuffle=False, collate_fn=collate)
      return train_loader, val_loader, test_loader
    return train_loader, val_loader

  def _split_train_val(
    self, indices: np.ndarray, patient_ids: np.ndarray, seed: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Patient-level stratified 9:1 train/val split.

    Stratifies on patient labels so val contains both classes.
    Uses stratified_patient_split with n_splits=10, takes first fold as val.
    """
    all_labels = np.array(self.dataset.get_labels())
    labels = all_labels[indices]
    splits = stratified_patient_split(indices, patient_ids, labels, n_splits=10, seed=seed)
    return splits[0]

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

    details dict contains slide_ids, patient_ids, probs, labels — used for
    patient-level aggregation.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    all_slide_ids, all_patient_ids = [], []

    with torch.no_grad():
      for raw_batch in loader:
        batch = self._move_to_device(raw_batch)
        inputs = self._model_inputs(batch)
        labels = batch['label']
        slide_ids = batch.get('slide_id', ['unknown'] * len(labels))
        patient_ids_b = batch.get('patient_id', slide_ids)

        logits = model(inputs)['logits']
        loss, probs, preds = self._forward_and_decode(logits, labels, criterion)

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
    auc = self._compute_auc(labels_np, probs_np)

    details = {
      'slide_ids': all_slide_ids,
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
  ) -> Tuple[float, float, float]:
    """Compute patient-level accuracy, AUC, and F1 from in-memory eval predictions."""
    _, patient_results = aggregate_patient_predictions(
      slide_ids=eval_details['slide_ids'],
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
  def execute(self) -> TrainingResult: ...

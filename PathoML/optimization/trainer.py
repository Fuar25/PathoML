"""Training strategies for PathoML: cross-validation and full-dataset training.

Design overview (each class answers one question):
  EarlyStopping      — when should training stop?
  CheckpointManager  — how to save/load model checkpoints?
  TrainingMixin      — how to run one epoch and evaluate a loader?
  FoldTrainer        — how to train a single fold to convergence?
  CrossValidator     — how to run K-fold cross-validation?
  FullDatasetTrainer — how to train on the full dataset for deployment?
  Trainer            — which strategy should be used?
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .patient_aggregation import aggregate_patient_predictions


# ---------------------------------------------------------------------------
# (1) Data containers
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


@dataclass
class TrainingResult:
  """Results container for an entire training run."""
  strategy_name: str
  fold_results: List[FoldResult]
  result_dir: str


# ---------------------------------------------------------------------------
# (2) Utility classes (tool pieces, used via composition)
# ---------------------------------------------------------------------------

class EarlyStopping:
  """Tracks validation loss and signals when training should stop."""

  def __init__(self, patience: int = 5) -> None:
    self.patience = patience
    self.best_val_loss = float('inf')
    self.patience_counter = 0
    self.best_epoch = 0

  def step(self, val_loss: float, current_epoch: int) -> bool:
    """Returns True if training should stop."""
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      self.best_epoch = current_epoch
      self.patience_counter = 0
      return False
    self.patience_counter += 1
    return self.patience_counter >= self.patience

  def reset(self) -> None:
    self.best_val_loss = float('inf')
    self.patience_counter = 0
    self.best_epoch = 0


class CheckpointManager:
  """Handles model checkpoint save/load."""

  def __init__(self, save_dir: str) -> None:
    self.save_dir = save_dir
    os.makedirs(self.save_dir, exist_ok=True)

  def save(self, model: nn.Module, name: str) -> str:
    path = os.path.join(self.save_dir, name)
    torch.save(model.state_dict(), path)
    return path

  def load(self, model: nn.Module, name: str) -> nn.Module:
    path = os.path.join(self.save_dir, name)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


# ---------------------------------------------------------------------------
# (3) TrainingMixin — knows how to run one epoch and evaluate a loader
#
# Dependencies (provided by the concrete class that inherits this):
#   self.device          — torch.device
#   self.model_cfg       — ModelConfig with .num_classes
#   self._move_to_device — method that moves a batch dict to device
#   self._model_inputs   — method that filters batch to model-accepted keys
# ---------------------------------------------------------------------------

class TrainingMixin:
  """Training loop and evaluation capabilities."""

  BINARY_THRESHOLD: float = 0.5

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
    return_details: bool = False,
  ):
    """Evaluate model on a loader. Returns (loss, acc, auc) or with details dict.

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

    if return_details:
      return total_loss / total, correct / total, auc, {
        'sample_ids': all_sample_ids,
        'patient_ids': all_patient_ids,
        'probs': probs_np,
        'labels': labels_np,
      }
    return total_loss / total, correct / total, auc

  def _forward_and_decode(
    self,
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unify binary vs multi-class loss + prob + pred computation.

    Returns (loss, probs, preds).
    """
    if self.model_cfg.num_classes == 1:
      logits = logits.view(-1)
      labels = labels.view(-1).float()
      loss = criterion(logits, labels)
      probs = torch.sigmoid(logits)
      preds = (probs > self.BINARY_THRESHOLD).float()
    else:
      loss = criterion(logits, labels.long())
      probs = torch.softmax(logits, dim=1)
      preds = torch.argmax(probs, dim=1)
    return loss, probs, preds

  def _compute_auc(self, labels_np: np.ndarray, probs_np: np.ndarray) -> float:
    """Compute AUC, returning 0.0 on failure (e.g. single-class batch)."""
    try:
      if self.model_cfg.num_classes == 1:
        return roc_auc_score(labels_np.astype(int), probs_np)
      return roc_auc_score(
        labels_np.astype(int), probs_np,
        multi_class='ovr', average='macro',
      )
    except Exception as e:
      print(f"Warning: AUC computation failed: {e}")
      return 0.0

  @staticmethod
  def _build_criterion(num_classes: int) -> nn.Module:
    return nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# (4) Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
  """Abstract training strategy. Concrete classes own all dependencies."""

  @property
  @abstractmethod
  def name(self) -> str: ...

  @abstractmethod
  def execute(self) -> TrainingResult: ...


# ---------------------------------------------------------------------------
# (5) FoldTrainer — runs training loop for a single fold
#
# Design: FoldTrainer uses composition (not inheritance) for training logic.
# It receives the epoch/eval functions as callable arguments from CrossValidator,
# keeping FoldTrainer fully decoupled from the training loop implementation.
# ---------------------------------------------------------------------------

class FoldTrainer:
  """Single-fold training engine: connects early stopping, checkpointing, and the loop."""

  def __init__(
    self,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStopping,
    checkpoint_manager: CheckpointManager,
    config,  # TrainingConfig
  ) -> None:
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.early_stopping = early_stopping
    self.checkpoint_manager = checkpoint_manager
    self.config = config

  def fit(
    self,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold: int,
    train_epoch_fn: Callable,   # from TrainingMixin._train_epoch
    evaluate_fn: Callable,       # from TrainingMixin._evaluate_with_auc
  ) -> str:
    """Train until early stopping or max epochs. Returns best checkpoint filename."""
    best_ckpt = f'model_fold_{fold}_best.pth'
    start = time.time()

    with tqdm(range(self.config.epochs), desc=f"  Fold {fold}", leave=True) as pbar:
      for epoch in pbar:
        train_loss, train_acc = train_epoch_fn(
          self.model, train_loader, self.criterion, self.optimizer
        )
        val_loss, val_acc, val_auc = evaluate_fn(
          self.model, val_loader, self.criterion
        )

        # ETA estimation
        elapsed = time.time() - start
        eta = elapsed / (epoch + 1) * (self.config.epochs - epoch - 1)
        pbar.set_postfix(
          train_loss=f"{train_loss:.4f}",
          val_loss=f"{val_loss:.4f}",
          val_auc=f"{val_auc:.4f}",
          ETA=f"{int(eta//60)}:{int(eta%60):02d}",
        )

        should_stop = self.early_stopping.step(val_loss, epoch + 1)
        if not should_stop:
          self.checkpoint_manager.save(self.model, best_ckpt)
        else:
          tqdm.write(
            f"  Early stop @ epoch {epoch+1}, best epoch={self.early_stopping.best_epoch}"
          )
          break

    return best_ckpt


# ---------------------------------------------------------------------------
# (6) CrossValidator — K-fold cross-validation scheduler
# ---------------------------------------------------------------------------

class CrossValidator(Strategy, TrainingMixin):
  """K-fold cross-validation scheduler.

  Inherits TrainingMixin for _train_epoch / _evaluate_with_auc.
  Owns _move_to_device, _model_inputs, _split_train_val as direct methods.

  Usage:
      strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
      Trainer(strategy).fit()
  """

  def __init__(self, model_builder: Callable, dataset, config, k_folds: int = 5) -> None:
    # (1) Store all dependencies so execute() needs no arguments
    self.model_builder = model_builder
    self.dataset = dataset
    self.model_cfg = config.model
    self.training_cfg = config.training
    self.logging_cfg = config.logging
    self.device = torch.device(config.training.device)
    self.k_folds = k_folds

  @property
  def name(self) -> str:
    return "KFold CrossValidation"

  # -- Public entry point --

  def execute(self) -> TrainingResult:
    """Run K-fold CV. Each fold: split → train → evaluate → aggregate."""
    checkpoint_manager = CheckpointManager(self.logging_cfg.save_dir)
    split_iter, patient_ids = self._prepare_splits()

    fold_results = []
    cv_start = time.time()

    for fold, (train_val_ids, test_ids) in enumerate(split_iter):
      print(f"\n{'='*70}\nFold {fold+1} / {self.k_folds}\n{'='*70}")

      result = self._train_single_fold(
        fold=fold + 1,
        train_val_ids=train_val_ids,
        test_ids=test_ids,
        patient_ids=patient_ids,
        checkpoint_manager=checkpoint_manager,
      )
      fold_results.append(result)
      self._print_fold_result(result)

    self._print_cv_summary(fold_results, time.time() - cv_start)

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
    labels = np.array([item['label'] for item in self.dataset.data])

    # (1) Binary: use stratified group split to balance class ratio across folds
    if self.model_cfg.num_classes == 1:
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
    checkpoint_manager: CheckpointManager,
  ) -> FoldResult:
    """Full pipeline for one fold: split → build loaders → train → evaluate."""
    # (1) Patient-aware train/val split
    train_ids, val_ids = self._split_train_val(train_val_ids, patient_ids, fold)

    # (2) Build DataLoaders
    train_loader, val_loader, test_loader = self._build_loaders(train_ids, val_ids, test_ids)

    # (3) Initialize model, optimizer, loss, early stopping
    model = self.model_builder().to(self.device)
    criterion = self._build_criterion(self.model_cfg.num_classes)
    optimizer = optim.Adam(
      model.parameters(),
      lr=self.training_cfg.learning_rate,
      weight_decay=self.training_cfg.weight_decay,
    )
    early_stopping = EarlyStopping(patience=self.training_cfg.patience)

    # (4) Assemble FoldTrainer and run training
    fold_trainer = FoldTrainer(
      model=model,
      criterion=criterion,
      optimizer=optimizer,
      early_stopping=early_stopping,
      checkpoint_manager=checkpoint_manager,
      config=self.training_cfg,
    )
    best_ckpt = fold_trainer.fit(
      train_loader=train_loader,
      val_loader=val_loader,
      fold=fold,
      train_epoch_fn=self._train_epoch,
      evaluate_fn=self._evaluate_with_auc,
    )

    # (5) Load best checkpoint and evaluate on test set
    checkpoint_manager.load(model, best_ckpt)
    test_loss, test_acc, test_auc, test_details = self._evaluate_with_auc(
      model, test_loader, criterion, return_details=True
    )

    # (6) Compute patient-level metrics (runtime, no disk I/O)
    patient_acc, patient_auc = self._compute_patient_metrics(test_details)

    return FoldResult(
      fold=fold,
      best_epoch=early_stopping.best_epoch,
      val_loss=early_stopping.best_val_loss,
      test_loss=test_loss,
      test_acc=test_acc,
      test_auc=test_auc,
      patient_acc=patient_acc,
      patient_auc=patient_auc,
      checkpoint_name=best_ckpt,
    )

  # -- Batch/loader helpers --

  def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Move all tensor values in a batch dict to self.device."""
    return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

  @staticmethod
  def _model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-model keys from batch dict (labels, IDs, paths)."""
    _exclude = {'label', 'sample_id', 'patient_id', 'feature_path', 'tissue_id', 'modalities'}
    return {k: v for k, v in batch.items() if k not in _exclude}

  def _build_loaders(
    self,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from index arrays."""
    bs = self.training_cfg.batch_size
    train_loader = DataLoader(Subset(self.dataset, train_ids), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(Subset(self.dataset, val_ids),   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(Subset(self.dataset, test_ids),  batch_size=bs, shuffle=False)
    return train_loader, val_loader, test_loader

  def _split_train_val(
    self, train_val_ids: np.ndarray, patient_ids: np.ndarray, fold: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Patient-aware random train/val split within a fold's train+val pool."""
    rng = np.random.default_rng(self.training_cfg.seed + fold)
    tv_patients = patient_ids[train_val_ids]
    unique_patients = np.unique(tv_patients)
    rng.shuffle(unique_patients)

    n_val = max(1, int(len(unique_patients) * 0.2))
    val_patients = set(unique_patients[:n_val])

    val_mask = np.array([pid in val_patients for pid in tv_patients])
    return train_val_ids[~val_mask], train_val_ids[val_mask]

  # -- Patient-level metrics (runtime computation, no CSV) --

  def _compute_patient_metrics(
    self, test_details: Dict[str, Any]
  ) -> Tuple[float, float]:
    """Compute patient-level accuracy and AUC from in-memory test predictions.

    No disk I/O: uses aggregate_patient_predictions() on the in-memory
    test_details dict returned by _evaluate_with_auc(return_details=True).
    CSV export is deferred to the Interpretability module.
    """
    _, patient_results = aggregate_patient_predictions(
      sample_ids=test_details['sample_ids'],
      patient_ids=test_details['patient_ids'],
      probs=test_details['probs'],
      labels=test_details['labels'],
      num_classes=self.model_cfg.num_classes,
    )

    p_labels = patient_results['label'].values
    p_preds  = patient_results['prediction'].values
    patient_acc = float((p_labels == p_preds).mean())

    try:
      if self.model_cfg.num_classes == 1:
        patient_auc = roc_auc_score(p_labels, patient_results['prob_positive'].values)
      else:
        prob_cols = [c for c in patient_results.columns if c.startswith('prob_class_')]
        p_probs = patient_results[prob_cols].values
        patient_auc = roc_auc_score(
          p_labels, p_probs, multi_class='ovr', average='macro'
        )
    except Exception as e:
      print(f"Warning: Patient AUC computation failed: {e}")
      patient_auc = float('nan')

    return patient_acc, patient_auc

  # -- Summary printing --

  @staticmethod
  def _print_fold_result(result: FoldResult) -> None:
    print(
      f"  Fold {result.fold}: "
      f"Best Epoch={result.best_epoch}, "
      f"Test AUC={result.test_auc:.4f}, Test Acc={result.test_acc:.4f}, "
      f"Patient AUC={result.patient_auc:.4f}, Patient Acc={result.patient_acc:.4f}"
    )

  def _print_cv_summary(self, fold_results: List[FoldResult], total_time: float) -> None:
    """Print cross-validation summary statistics."""
    test_aucs  = [r.test_auc  for r in fold_results]
    test_accs  = [r.test_acc  for r in fold_results]
    best_epochs = [r.best_epoch for r in fold_results]
    patient_aucs = [
      r.patient_auc for r in fold_results
      if isinstance(r.patient_auc, float) and np.isfinite(r.patient_auc)
    ]
    patient_accs = [
      r.patient_acc for r in fold_results
      if isinstance(r.patient_acc, float) and np.isfinite(r.patient_acc)
    ]

    print(f"\n{'='*70}")
    print("Cross-Validation Results:")
    print(f"{'='*70}")
    for r in fold_results:
      print(
        f"  Fold {r.fold}: Best Epoch={r.best_epoch:3d}, "
        f"Test AUC={r.test_auc:.4f}, Test Acc={r.test_acc:.4f}, "
        f"Patient AUC={r.patient_auc:.4f}, Patient Acc={r.patient_acc:.4f}"
      )
    print(f"{'-'*70}")
    print(f"  Avg Test AUC:  {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"  Avg Test Acc:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    if patient_aucs:
      print(f"  Avg Patient AUC: {np.mean(patient_aucs):.4f} ± {np.std(patient_aucs):.4f}")
      print(f"  Avg Patient Acc: {np.mean(patient_accs):.4f} ± {np.std(patient_accs):.4f}")
    print(f"  Avg Best Epoch:  {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")
    print(f"  Total Time:      {total_time/60:.2f} min")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# (7) FullDatasetTrainer — trains on full dataset for deployment
# ---------------------------------------------------------------------------

class FullDatasetTrainer(Strategy, TrainingMixin):
  """Full-dataset trainer for deployment.

  No validation set, no test evaluation. Trains for config.training.epochs
  epochs and saves the final model.

  Usage:
      strategy = FullDatasetTrainer(model_builder, dataset, config)
      Trainer(strategy).fit()
  """

  def __init__(self, model_builder: Callable, dataset, config) -> None:
    self.model_builder = model_builder
    self.dataset = dataset
    self.model_cfg = config.model
    self.training_cfg = config.training
    self.logging_cfg = config.logging
    self.device = torch.device(config.training.device)

  @property
  def name(self) -> str:
    return "FullDataset Training"

  def execute(self) -> TrainingResult:
    """Train on full dataset for config.training.epochs epochs and save checkpoint."""
    loader = DataLoader(self.dataset, batch_size=self.training_cfg.batch_size, shuffle=True)
    model  = self.model_builder().to(self.device)
    criterion = self._build_criterion(self.model_cfg.num_classes)
    optimizer = optim.Adam(
      model.parameters(),
      lr=self.training_cfg.learning_rate,
      weight_decay=self.training_cfg.weight_decay,
    )
    checkpoint_manager = CheckpointManager(self.logging_cfg.save_dir)

    start = time.time()
    for epoch in tqdm(range(self.training_cfg.epochs), desc="Full Training"):
      train_loss, train_acc = self._train_epoch(model, loader, criterion, optimizer)

    ckpt_path = checkpoint_manager.save(model, 'model_deployment.pth')
    print(f"Saved deployment model to {ckpt_path} in {(time.time()-start)/60:.2f} min")

    return TrainingResult(
      strategy_name=self.name,
      fold_results=[],
      result_dir=self.logging_cfg.save_dir,
    )

  def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

  @staticmethod
  def _model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    _exclude = {'label', 'sample_id', 'patient_id', 'feature_path', 'tissue_id', 'modalities'}
    return {k: v for k, v in batch.items() if k not in _exclude}


# ---------------------------------------------------------------------------
# (8) Trainer — thin dispatcher, the only public entry point
# ---------------------------------------------------------------------------

class Trainer:
  """Dispatches training to the selected strategy.

  Usage:
      strategy = CrossValidator(model_builder, dataset, config, k_folds=5)
      result = Trainer(strategy).fit()
  """

  def __init__(self, strategy: Strategy) -> None:
    self.strategy = strategy

  def fit(self) -> TrainingResult:
    return self.strategy.execute()

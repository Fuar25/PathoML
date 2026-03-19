"""Training utilities: result containers and utility classes (non-fold-specific)."""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn


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
# (2) EarlyStopping — tracks validation loss, saves checkpoint on improvement
# ---------------------------------------------------------------------------

class EarlyStopping:
  """Patience-based early stopping with integrated checkpoint management.

  Saves the model to ckpt_path whenever val_loss improves.
  Call load_best() after training to restore the best weights.
  """

  def __init__(self, patience: int, model: nn.Module, ckpt_path: str) -> None:
    self.patience = patience
    self.model = model
    self.ckpt_path = ckpt_path
    self.best_val_loss = float('inf')
    self.patience_counter = 0
    self.best_epoch = 0

  def step(self, val_loss: float, current_epoch: int) -> bool:
    """Update state. Returns True if training should stop.

    Saves checkpoint automatically on improvement.
    """
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
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
    self.best_val_loss = float('inf')
    self.patience_counter = 0
    self.best_epoch = 0

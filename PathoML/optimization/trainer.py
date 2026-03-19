"""Training entry point: Trainer dispatcher and backward-compatible re-exports.

Import map:
  training_utils.py         — TrainingResult, EarlyStopping
  TrainingStrategy/
    training_base.py        — TrainingMixin, Strategy
    cross_validator.py      — FoldResult, CrossValidator
    full_dataset_trainer.py — FullTrainingResult, FullDatasetTrainer
"""

# Re-exports for backward-compatible direct imports from this module
from .training_utils import TrainingResult, EarlyStopping
from .TrainingStrategy import CrossValidator, FullDatasetTrainer, TrainingMixin, Strategy
from .TrainingStrategy.cross_validator import FoldResult
from .TrainingStrategy.full_dataset_trainer import FullTrainingResult


# ---------------------------------------------------------------------------
# Trainer — thin dispatcher, the only public entry point
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

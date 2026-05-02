from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .defaults import (
  DEFAULT_DATASET_NAME,
  DEFAULT_MODEL_NAME,
  PATIENT_ID_PATTERN,
)


@dataclass
class DatasetConfig:
  """Dataset configuration parameters."""

  dataset_name: str = DEFAULT_DATASET_NAME
  dataset_module_paths: List[str] = field(default_factory=list)
  dataset_kwargs: Dict[str, object] = field(default_factory=dict)
  patient_id_pattern: str = PATIENT_ID_PATTERN
  # binary_mode: auto-inferred by dataset (len(classes) == 2); not a config concern


@dataclass
class ModelConfig:
  """Model configuration parameters.

  model_kwargs: all model-specific parameters (e.g. hidden_dim, dropout, gated).
  input_dim and num_classes are inferred at runtime from the dataset.
  """

  model_name: str = DEFAULT_MODEL_NAME
  model_module_paths: List[str] = field(default_factory=list)
  model_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrainingConfig:
  """Training process configuration parameters."""

  epochs: int = 30
  batch_size: int = 1
  learning_rate: float = 0.0005
  weight_decay: float = 1e-5
  seed: int = 42
  device: str = 'cuda'
  patience: int = 5
  min_delta: float = 0.002
  early_stopping_metric: str = 'val_auc'  # 'val_auc' | 'patient_f1'
  patient_threshold: float = 0.5
  scheduler: str = 'none'  # 'none' | 'cosine'
  num_workers: int = 0
  pin_memory: bool = False
  persistent_workers: bool = False
  prefetch_factor: Optional[int] = None
  non_blocking_device_transfer: bool = False
  bucket_by_length: bool = False


@dataclass
class LoggingConfig:
  """Logging and checkpoint configuration parameters."""

  save_dir: str = './experiments'
  save_best_only: bool = True


@dataclass
class RunTimeConfig:
  """Top-level runtime configuration; instantiate directly (no singleton)."""

  dataset: DatasetConfig = field(default_factory=DatasetConfig)
  model: ModelConfig = field(default_factory=ModelConfig)
  training: TrainingConfig = field(default_factory=TrainingConfig)
  logging: LoggingConfig = field(default_factory=LoggingConfig)

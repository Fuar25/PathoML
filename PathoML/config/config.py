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
  binary_mode: Optional[bool] = None


@dataclass
class ModelConfig:
  """Model architecture configuration parameters.

  model_kwargs: model-specific parameters (e.g. gated, n_heads, attention_dim).
  These are passed directly to the model constructor and filtered by its signature.
  """

  model_name: str = DEFAULT_MODEL_NAME
  model_module_paths: List[str] = field(default_factory=list)
  model_kwargs: Dict[str, object] = field(default_factory=dict)
  input_dim: int = 1536
  hidden_dim: int = 512
  num_classes: int = 1
  dropout: float = 0.2


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
  patient_threshold: float = 0.5


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

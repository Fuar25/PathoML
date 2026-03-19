"""PathoML main entry point — configure and run training strategies directly."""

from PathoML.config.config import RunTimeConfig
from PathoML.registry import (
  create_dataset_from_config,
  model_builder_from_config,
  load_all_module,
)
from PathoML.optimization.trainer import CrossValidator, FullDatasetTrainer, Trainer


def build_and_run(config: RunTimeConfig, strategy: str = "cv", k_folds: int = 5) -> None:
  """Load dataset, build model factory, and run the specified training strategy.

  Args:
      config: Fully configured RunTimeConfig instance.
      strategy: 'cv' for cross-validation, 'train' for full-dataset training.
      k_folds: Number of folds (only used when strategy='cv').
  """
  # (1) Load plugins, build dataset and model factory
  load_all_module(config)
  dataset = create_dataset_from_config(config.dataset)

  if len(dataset) == 0:
    print("Error: No data found. Check your data paths.")
    return

  model_builder = model_builder_from_config(config.model, dataset)

  # (2) Instantiate strategy and run
  if strategy == "cv":
    runner = CrossValidator(model_builder, dataset, config, k_folds=k_folds)
  elif strategy == "train":
    runner = FullDatasetTrainer(model_builder, dataset, config)
  else:
    raise ValueError(f"Unknown strategy: '{strategy}'. Use 'cv' or 'train'.")

  trainer = Trainer(runner)
  trainer.fit()


if __name__ == "__main__":
  # =========================================================================
  # Experiment configuration — edit here and run directly
  # =========================================================================
  config = RunTimeConfig()

  # Training hyperparameters
  config.training.device = "cuda:0"
  config.training.epochs = 50
  config.training.learning_rate = 5e-4
  config.training.patience = 5

  # Logging / checkpoints
  config.logging.save_dir = "./experiments/cv_template"

  # --- Unimodal dataset ---
  config.dataset.dataset_name = "UnimodalPatchDataset"
  config.dataset.dataset_kwargs["data_path"] = "/path/to/data_root"

  # --- Model ---
  config.model.model_name = "abmil"           # 'abmil' or 'linear_probe'
  config.model.model_kwargs = {               # model-specific params (input_dim/num_classes auto-inferred)
    "gated": True,
    "attention_dim": None,
  }

  # --- Multimodal (concat) example — uncomment to use ---
  # config.dataset.dataset_name = "MultimodalConcatSlideDataset"
  # config.dataset.dataset_kwargs = {
  #   "modality_paths": {"HE": "/path/to/HE", "CD20": "/path/to/CD20"},
  #   "modality_names": ["HE", "CD20"],
  #   "allow_missing_modalities": True,
  # }

  build_and_run(config, strategy="cv", k_folds=5)

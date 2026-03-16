"""PathoML main entry point — configure and run training strategies directly."""

from dataclasses import asdict

from PathoML.config.config import RunTimeConfig
from PathoML.optimization.registry import create_dataset, create_model, load_runtime_plugins
from PathoML.optimization.trainer import CrossValidator, FullDatasetTrainer, Trainer


def build_and_run(config: RunTimeConfig, strategy: str = "cv", k_folds: int = 5) -> None:
  """Load dataset, build model factory, and run the specified training strategy.

  Args:
      config: Fully configured RunTimeConfig instance.
      strategy: 'cv' for cross-validation, 'train' for full-dataset training.
      k_folds: Number of folds (only used when strategy='cv').
  """
  dataset_cfg = config.dataset
  model_cfg = config.model

  # (1) Load dataset and model plugins
  load_runtime_plugins(config)

  # (2) Build dataset — config fields as defaults; dataset_kwargs override
  dataset_kwargs = dict(dataset_cfg.dataset_kwargs)
  dataset_kwargs.setdefault("patient_id_pattern", dataset_cfg.patient_id_pattern)
  dataset_kwargs.setdefault("binary_mode", dataset_cfg.binary_mode)

  dataset = create_dataset(dataset_cfg.dataset_name, **dataset_kwargs)

  if len(dataset) == 0:
    print("Error: No data found. Check your data paths.")
    return

  # (3) Auto-set num_classes from dataset
  n_classes = len(dataset.classes)
  config.model.num_classes = 1 if n_classes == 2 else n_classes
  print(f"Auto-configured num_classes={config.model.num_classes} ({n_classes} label classes).")

  # (4) Auto-align input_dim from first sample
  sample = dataset[0]
  if "features" in sample and sample["features"].ndim >= 2:
    inferred_dim = int(sample["features"].shape[-1])
    if "input_dim" not in model_cfg.model_kwargs and model_cfg.input_dim != inferred_dim:
      print(f"Info: Auto-adjust input_dim {model_cfg.input_dim} → {inferred_dim}.")
      config.model.input_dim = inferred_dim

  # (5) Model factory — each call returns a freshly initialised model
  def model_builder():
    all_kwargs = {
      k: v for k, v in asdict(model_cfg).items()
      if k not in ("model_name", "model_module_paths", "model_kwargs")
    }
    all_kwargs.update(model_cfg.model_kwargs)
    return create_model(model_cfg.model_name, **all_kwargs)

  # (6) Instantiate strategy and run
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
  config.dataset.dataset_name = "wsi_h5"
  config.dataset.dataset_kwargs["data_paths"] = {
    "positive": "/path/to/positive_h5",
    "negative": "/path/to/negative_h5",
  }

  # --- Model ---
  config.model.model_name = "abmil"           # 'abmil' or 'linear_probe'
  config.model.input_dim = 1536
  config.model.model_kwargs = {               # model-specific params
    "gated": True,
    "attention_dim": None,
  }

  # --- Multimodal (concat) example — uncomment to use ---
  # config.dataset.dataset_name = "multimodal_concat"
  # config.dataset.dataset_module_paths = ["data.multimodal_dataset_concat"]
  # config.dataset.dataset_kwargs = {
  #   "modality_paths": {
  #     "HE":   "/path/to/HE",
  #     "CD20": "/path/to/CD20",
  #   },
  #   "modality_names": ["HE", "CD20"],
  #   "allow_missing_modalities": True,
  # }

  build_and_run(config, strategy="cv", k_folds=5)

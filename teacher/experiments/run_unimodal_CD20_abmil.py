import os
from teacher.experiments.common import (
  run_condition, log_results, find_common_sample_keys,
  RunTimeConfig,
  PATCH_FEAT_ROOT, SLIDE_FEAT_ROOT, LABELS_CSV,
  N_RUNS, K_FOLDS, DEVICE, EPOCHS, PATIENCE, LR, WD, MLP_HIDDEN_DIM, DROPOUT_RATE, BATCH_SIZE,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

INTERSECTION_STAINS = ["HE", "CD20", "CD3"]
CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "UnimodalPatchDataset"
  config.dataset.dataset_kwargs["data_root"] = PATCH_FEAT_ROOT
  config.dataset.dataset_kwargs["stain"] = "CD20"
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.dataset.dataset_kwargs["labels_csv"] = LABELS_CSV
  config.model.model_name = "abmil"
  config.model.model_kwargs = {"hidden_dim": MLP_HIDDEN_DIM, "dropout": DROPOUT_RATE}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = LR
  config.training.weight_decay = WD
  config.training.batch_size = BATCH_SIZE
  return config


def main():
  common_keys = find_common_sample_keys(SLIDE_FEAT_ROOT, INTERSECTION_STAINS)
  print(f"公共样本数（{' ∩ '.join(INTERSECTION_STAINS)}）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results({CONDITION_NAME: results}, SHARED_LOG_FILE, config=config,
              stains=INTERSECTION_STAINS)


if __name__ == "__main__":
  main()

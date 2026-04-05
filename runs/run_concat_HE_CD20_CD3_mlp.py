# HE+CD20+CD3 多模态 Concat MLP 实验（GigaPath-Slide-Feature）。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys,
  RunTimeConfig,
  SLIDE_FEAT_ROOT, LABELS_CSV,
  N_RUNS, K_FOLDS, DEVICE, EPOCHS, WD, DROPOUT_RATE, BATCH_SIZE, SLIDE_LR, PATIENCE,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

MLP_HIDDEN_DIM = 128

STAINS = ["HE", "CD20", "CD3"]
CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "MultimodalConcatSlideDataset"
  config.dataset.dataset_kwargs = {
    "data_root": SLIDE_FEAT_ROOT,
    "modality_names": STAINS,
    "allowed_sample_keys": common_keys,
    "labels_csv": LABELS_CSV,
  }
  config.model.model_name = "mlp"
  config.model.model_kwargs = {"hidden_dim": MLP_HIDDEN_DIM, "dropout": DROPOUT_RATE}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = SLIDE_LR
  config.training.weight_decay = WD
  config.training.batch_size = BATCH_SIZE
  return config


def main():
  common_keys = find_common_sample_keys(SLIDE_FEAT_ROOT, STAINS)
  print(f"公共样本数（{' ∩ '.join(STAINS)}）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results({CONDITION_NAME: results}, SHARED_LOG_FILE, config=config, stains=STAINS)


if __name__ == "__main__":
  main()

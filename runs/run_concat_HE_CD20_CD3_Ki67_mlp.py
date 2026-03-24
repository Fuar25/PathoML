# HE+CD20+CD3+Ki67 多模态 Concat MLP 实验（GigaPath-Slide-Feature）。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys, modality_names,
  RunTimeConfig,
  HE_BASE, CD20_BASE, CD3_BASE, Ki67_BASE,
  N_RUNS, K_FOLDS, DEVICE, EPOCHS, PATIENCE, LR, WD, DROPOUT_RATE,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

MLP_HIDDEN_DIM = 512
DROPOUT_RATE = 0.3

CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "MultimodalConcatSlideDataset"
  config.dataset.dataset_kwargs["modality_paths"] = {
    "HE":   HE_BASE,
    "CD20": CD20_BASE,
    "CD3":  CD3_BASE,
    "Ki67": Ki67_BASE,
  }
  config.dataset.dataset_kwargs["modality_names"] = ["HE", "CD20", "CD3", "Ki67"]
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.model.model_name = "mlp"
  config.model.model_kwargs = {"hidden_dim": MLP_HIDDEN_DIM, "dropout": DROPOUT_RATE}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = LR
  config.training.weight_decay = WD
  return config


def main():
  intersection_bases = [HE_BASE, CD20_BASE, CD3_BASE, Ki67_BASE]
  common_keys = find_common_sample_keys(intersection_bases)
  print(f"公共样本数（HE ∩ CD20 ∩ CD3 ∩ Ki67）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results({CONDITION_NAME: results}, SHARED_LOG_FILE, config=config,
              sample_intersection=modality_names(intersection_bases))


if __name__ == "__main__":
  main()

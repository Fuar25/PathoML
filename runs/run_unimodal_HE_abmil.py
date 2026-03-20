# HE 单模态 ABMIL 实验（GigaPath-Patch-Feature）。

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys,
  RunTimeConfig,
  CD20_BASE, CD3_BASE,
  N_RUNS, K_FOLDS, DEVICE, EPOCHS, PATIENCE, LR, WD, MLP_HIDDEN_DIM, DROPOUT_RATE,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

_PATCH_FEAT_ROOT = "/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Patch-Feature"
HE_PATCH_BASE = f"{_PATCH_FEAT_ROOT}/HE"

CONDITION_NAME = "unimodal_HE_abmil"


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "UnimodalSlideDataset"
  config.dataset.dataset_kwargs["data_path"] = HE_PATCH_BASE
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.model.model_name = "abmil"
  config.model.model_kwargs = {"hidden_dim": MLP_HIDDEN_DIM, "dropout": DROPOUT_RATE}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = LR
  config.training.weight_decay = WD
  return config


def main():
  # 与其他实验保持一致：取 HE patch 与 CD20, CD3 slide 的交集样本
  common_keys = find_common_sample_keys([HE_PATCH_BASE, CD20_BASE, CD3_BASE])
  print(f"公共样本数（HE_patch ∩ CD20 ∩ CD3）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results({CONDITION_NAME: results}, SHARED_LOG_FILE, config=config)


if __name__ == "__main__":
  main()

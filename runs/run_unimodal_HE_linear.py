# HE 单模态 LinearProbe 实验。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys,
  RunTimeConfig,
  HE_BASE, CD20_BASE, CD3_BASE,
  N_RUNS, K_FOLDS, DEVICE, EPOCHS, PATIENCE, LR, WD,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

CONDITION_NAME = "unimodal_HE_linear"


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "UnimodalSlideDataset"
  config.dataset.dataset_kwargs["data_path"] = HE_BASE
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.model.model_name = "linear_probe"
  config.model.model_kwargs = {}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = LR
  config.training.weight_decay = WD
  return config


def main():
  # 仅保留 HE, CD20, CD3 均存在的样本，与多模态实验保持一致
  common_keys = find_common_sample_keys([HE_BASE, CD20_BASE, CD3_BASE])
  print(f"公共样本数（HE ∩ CD20 ∩ CD3）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results({CONDITION_NAME: results}, SHARED_LOG_FILE, config=config)


if __name__ == "__main__":
  main()

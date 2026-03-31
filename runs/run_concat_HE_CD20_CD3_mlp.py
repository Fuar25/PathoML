# HE+CD20+CD3 多模态 Concat MLP 实验（GigaPath-Slide-Feature）。
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys, modality_names,
  RunTimeConfig,
  HE_SLIDE_BASE, CD20_SLIDE_BASE, CD3_SLIDE_BASE, LABELS_CSV,
  N_RUNS, K_FOLDS, EPOCHS, WD, DROPOUT_RATE, BATCH_SIZE, SLIDE_LR, PATIENCE,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

DEVICE = "cuda:1"
MLP_HIDDEN_DIM = 128
TUNING_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_log.txt")

CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def make_config(common_keys, *, hidden_dim=MLP_HIDDEN_DIM, dropout=DROPOUT_RATE,
                lr=SLIDE_LR, wd=WD, batch_size=BATCH_SIZE, num_layers=1) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "MultimodalConcatSlideDataset"
  config.dataset.dataset_kwargs["modality_paths"] = {
    "HE":   HE_SLIDE_BASE,
    "CD20": CD20_SLIDE_BASE,
    "CD3":  CD3_SLIDE_BASE,
  }
  config.dataset.dataset_kwargs["modality_names"] = ["HE", "CD20", "CD3"]
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.dataset.dataset_kwargs["labels_csv"] = LABELS_CSV
  config.model.model_name = "mlp"
  config.model.model_kwargs = {"hidden_dim": hidden_dim, "dropout": dropout, "num_layers": num_layers}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = lr
  config.training.weight_decay = wd
  config.training.batch_size = batch_size
  return config


# ─── 超参实验组 ──────────────────────────────────────────────────────────────
EXPERIMENTS = [
  {"tag": "dropout0.4",                     "dropout": 0.4},
  {"tag": "dropout0.4_wd1e-3",              "dropout": 0.4, "wd": 1e-3},
  {"tag": "hidden64_dropout0.3",            "hidden_dim": 64,  "dropout": 0.3},
  {"tag": "hidden256_dropout0.3",           "hidden_dim": 256, "dropout": 0.3},
  {"tag": "bs32",                           "batch_size": 32},
  {"tag": "bs64",                           "batch_size": 64},
  {"tag": "lr0.001",                        "lr": 0.001},
  {"tag": "lr0.0001",                       "lr": 0.0001},
  {"tag": "hidden128_dropout0.4_wd1e-3_bs32",
   "hidden_dim": 128, "dropout": 0.4, "wd": 1e-3, "batch_size": 32},
  {"tag": "2layer", "num_layers": 2},
]


def main():
  intersection_bases = [HE_SLIDE_BASE, CD20_SLIDE_BASE, CD3_SLIDE_BASE]
  common_keys = find_common_sample_keys(intersection_bases)
  print(f"公共样本数（HE ∩ CD20 ∩ CD3）: {len(common_keys)}")

  for i, exp in enumerate(EXPERIMENTS):
    tag = exp["tag"]
    kwargs = {k: v for k, v in exp.items() if k != "tag"}
    condition = f"{CONDITION_NAME}__{tag}"

    print(f"\n{'='*80}")
    print(f"实验 {i+1}/{len(EXPERIMENTS)}: {condition}")
    print(f"参数: {kwargs}")
    print(f"{'='*80}")

    config = make_config(common_keys, **kwargs)
    results = run_condition(condition, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
    log_results({condition: results}, TUNING_LOG_FILE, config=config,
                sample_intersection=modality_names(intersection_bases))


if __name__ == "__main__":
  main()

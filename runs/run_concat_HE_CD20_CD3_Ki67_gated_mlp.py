# HE+CD20+CD3+Ki67 多模态 GatedFusionMLP 超参扫描实验（GigaPath-Slide-Feature）。
# 基线: hidden256, dropout0.2, gate_temperature=1.0
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys, modality_names,
  RunTimeConfig,
  HE_SLIDE_BASE, CD20_SLIDE_BASE, CD3_SLIDE_BASE, Ki67_SLIDE_BASE,
  N_RUNS, K_FOLDS, EPOCHS, WD, DROPOUT_RATE, BATCH_SIZE, SLIDE_LR, PATIENCE,
  OUTPUTS_DIR,
)

DEVICE = "cuda:1"
HIDDEN_DIM = 256
N_MODALITIES = 4
TUNING_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_log.txt")

CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def make_config(common_keys, *, hidden_dim=HIDDEN_DIM, dropout=DROPOUT_RATE,
                lr=SLIDE_LR, wd=WD, batch_size=BATCH_SIZE,
                gate_temperature=1.0, gate_hidden_dim=None,
                num_post_layers=1) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "MultimodalConcatSlideDataset"
  config.dataset.dataset_kwargs["modality_paths"] = {
    "HE":   HE_SLIDE_BASE,
    "CD20": CD20_SLIDE_BASE,
    "CD3":  CD3_SLIDE_BASE,
    "Ki67": Ki67_SLIDE_BASE,
  }
  config.dataset.dataset_kwargs["modality_names"] = ["HE", "CD20", "CD3", "Ki67"]
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.model.model_name = "gated_fusion_mlp"
  model_kwargs = {
    "n_modalities": N_MODALITIES,
    "hidden_dim": hidden_dim,
    "dropout": dropout,
    "gate_temperature": gate_temperature,
    "num_post_layers": num_post_layers,
  }
  if gate_hidden_dim is not None:
    model_kwargs["gate_hidden_dim"] = gate_hidden_dim
  config.model.model_kwargs = model_kwargs
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = lr
  config.training.weight_decay = wd
  config.training.batch_size = batch_size
  return config


# ─── 超参实验组 ──────────────────────────────────────────────────────────────
EXPERIMENTS = [
  # (1) 基线
  {"tag": "baseline"},
  # (2) Gate temperature 扫描
  {"tag": "temp0.5",        "gate_temperature": 0.5},
  {"tag": "temp2.0",        "gate_temperature": 2.0},
  # (3) Hidden dim 扫描
  {"tag": "hidden128",      "hidden_dim": 128},
  {"tag": "hidden512",      "hidden_dim": 512},
  # (4) Gate hidden dim 扫描
  {"tag": "gate_hidden64",  "gate_hidden_dim": 64},
  {"tag": "gate_hidden512", "gate_hidden_dim": 512},
  # (5) Post-fusion 深度
  {"tag": "post2layer",     "num_post_layers": 2},
  # (6) Dropout
  {"tag": "drop0.3",        "dropout": 0.3},
]


def main():
  intersection_bases = [HE_SLIDE_BASE, CD20_SLIDE_BASE, CD3_SLIDE_BASE, Ki67_SLIDE_BASE]
  common_keys = find_common_sample_keys(intersection_bases)
  print(f"公共样本数（HE ∩ CD20 ∩ CD3 ∩ Ki67）: {len(common_keys)}")

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

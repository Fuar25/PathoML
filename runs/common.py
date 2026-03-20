# runs/common.py — 公共基础设施，供所有实验脚本共用。
#
# 包含：PathoML 导入、数据路径常量、超参数默认值、工具函数。
# 每个 run_*.py 只需 from common import ... 即可使用。

import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PathoML.config.config import RunTimeConfig
from PathoML.dataset.utils import find_common_sample_keys
from PathoML.registry import (
  create_dataset_from_config,
  model_builder_from_config,
  load_all_module,
)
from PathoML.optimization.trainer import CrossValidator, Trainer


# ─── 数据路径 ────────────────────────────────────────────────────────────────

_FEAT_ROOT = "/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Slide-Feature"
HE_BASE    = f"{_FEAT_ROOT}/HE"
CD20_BASE  = f"{_FEAT_ROOT}/CD20"
CD21_BASE  = f"{_FEAT_ROOT}/CD21"
Ki67_BASE  = f"{_FEAT_ROOT}/Ki-67"
CKpan_BASE = f"{_FEAT_ROOT}/CK-pan"
CD3_BASE   = f"{_FEAT_ROOT}/CD3"


# ─── 超参数默认值 ─────────────────────────────────────────────────────────────

N_RUNS         = 10
K_FOLDS        = 5
DEVICE         = "cuda:0"
EPOCHS         = 100
PATIENCE       = 10
LR             = 1e-4
WD             = 1e-5
BASE_SEED      = 42
MLP_HIDDEN_DIM = 256
DROPOUT_RATE   = 0.2


# ─── 路径配置 ─────────────────────────────────────────────────────────────────

# 所有实验的 checkpoint 输出根目录（各条件在其下创建子目录）
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 统一日志文件（所有实验追加写入同一个文件）
SHARED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_log.txt")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def run_cv(config: RunTimeConfig, k_folds: int) -> list[float]:
  """运行一次 k 折 CV，返回每折的 patient-level AUC 列表。"""
  load_all_module(config)
  dataset = create_dataset_from_config(config.dataset)
  if len(dataset) == 0:
    raise RuntimeError("No data found. Check your data paths.")
  model_builder = model_builder_from_config(config.model, dataset)
  result = Trainer(CrossValidator(model_builder, dataset, config, k_folds)).fit()
  return [f.patient_auc for f in result.fold_results]


def run_condition(
  condition_name: str,
  base_config: RunTimeConfig,
  n_runs: int,
  k_folds: int,
  output_dir: str,
  base_seed: int = BASE_SEED,
) -> dict:
  """对一个条件运行 n_runs 次 CV，收集 AUC 数据。

  Returns:
      dict with keys:
        run_means     — 每次实验的折均 AUC（长度 n_runs）
        all_fold_aucs — 所有折的 AUC（长度 n_runs × k_folds）
  """
  run_means = []
  all_fold_aucs = []

  for i in range(n_runs):
    seed = base_seed + i
    run_dir = os.path.join(output_dir, condition_name, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    base_config.logging.save_dir = run_dir
    base_config.training.seed = seed

    print(f"\n[{condition_name}] Run {i+1}/{n_runs}  (seed={seed})")
    fold_aucs = run_cv(base_config, k_folds)

    run_mean = float(np.mean(fold_aucs))
    run_means.append(run_mean)
    all_fold_aucs.extend(fold_aucs)

    fold_str = "  ".join(f"fold{j+1}={v:.4f}" for j, v in enumerate(fold_aucs))
    print(f"  {fold_str}  →  mean={run_mean:.4f}")

  return {"run_means": run_means, "all_fold_aucs": all_fold_aucs}


def log_results(
  results: dict[str, dict],
  log_path: str = SHARED_LOG_FILE,
  *,
  config: "RunTimeConfig | None" = None,
  n_runs: int = N_RUNS,
  k_folds: int = K_FOLDS,
  base_seed: int = BASE_SEED,
) -> None:
  """将各条件 AUC 对比表以时间戳追加方式写入日志文件。"""
  # (1) 格式化表格
  sep  = "=" * 76
  hsep = "─" * 76
  lines = [
    sep,
    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  "
    f"条件: {', '.join(results.keys())}",
    hsep,
    f"{'条件':<28}  {'run-level AUC (mean±std)':<28}  fold-level AUC (mean±std)",
    hsep,
  ]
  for name, data in results.items():
    run_means = np.array(data["run_means"])
    fold_aucs = np.array(data["all_fold_aucs"])
    lines.append(
      f"{name:<28}  "
      f"{run_means.mean():.4f} ± {run_means.std():.4f}              "
      f"{fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}"
    )

  # (2) 配置摘要（排除 path/name 类字段）
  lines.append(hsep)
  lines.append(f"N_RUNS={n_runs}  K_FOLDS={k_folds}  BASE_SEED={base_seed}")
  if config is not None:
    t = config.training
    lines.append(
      f"epochs={t.epochs}  patience={t.patience}  "
      f"lr={t.learning_rate}  wd={t.weight_decay}  device={t.device}"
    )
    m = config.model
    if m.model_kwargs:
      kw_str = "  ".join(f"{k}={v}" for k, v in m.model_kwargs.items())
      lines.append(f"model={m.model_name}  {kw_str}")
    else:
      lines.append(f"model={m.model_name}")
  lines.append(sep)
  lines.append("")

  # (3) 打印到终端 + 追加写入日志文件
  print("\n" + "\n".join(lines))
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"结果已追加记录至: {log_path}")

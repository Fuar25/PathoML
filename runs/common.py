# runs/common.py — 公共基础设施，供所有实验脚本共用。
#
# 包含：PathoML 导入、数据路径常量、超参数默认值、工具函数。
# 每个 run_*.py 只需 from common import ... 即可使用。

import json
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

# 优先读环境变量，否则按候选路径自动探测
_FEAT_ROOT_CANDIDATES = [
  "/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Flat",  # 主服务器
  "/home/sbh/Features",                                       # 备用机器
]
_FEAT_ROOT = os.environ.get("PATHOML_FEAT_ROOT") or next(
  (p for p in _FEAT_ROOT_CANDIDATES if os.path.isdir(p)), _FEAT_ROOT_CANDIDATES[0]
)
LABELS_CSV = f"{_FEAT_ROOT}/labels.csv"

_SLIDE_FEAT_ROOT = f"{_FEAT_ROOT}/GigaPath-Slide-Feature"
HE_SLIDE_BASE    = f"{_SLIDE_FEAT_ROOT}/HE"
CD20_SLIDE_BASE  = f"{_SLIDE_FEAT_ROOT}/CD20"
CD21_SLIDE_BASE  = f"{_SLIDE_FEAT_ROOT}/CD21"
Ki67_SLIDE_BASE  = f"{_SLIDE_FEAT_ROOT}/Ki-67"
CKpan_SLIDE_BASE = f"{_SLIDE_FEAT_ROOT}/CK-pan"
CD3_SLIDE_BASE   = f"{_SLIDE_FEAT_ROOT}/CD3"

_PATCH_FEAT_ROOT = f"{_FEAT_ROOT}/GigaPath-Patch-Feature"
HE_PATCH_BASE    = f"{_PATCH_FEAT_ROOT}/HE"
CD20_PATCH_BASE  = f"{_PATCH_FEAT_ROOT}/CD20"
CD21_PATCH_BASE  = f"{_PATCH_FEAT_ROOT}/CD21"
Ki67_PATCH_BASE  = f"{_PATCH_FEAT_ROOT}/Ki-67"
CKpan_PATCH_BASE = f"{_PATCH_FEAT_ROOT}/CK-pan"
CD3_PATCH_BASE   = f"{_PATCH_FEAT_ROOT}/CD3"


# ─── 超参数默认值 ─────────────────────────────────────────────────────────────

N_RUNS         = 5
K_FOLDS        = 5
DEVICE         = "cuda:0"
EPOCHS         = 100
PATIENCE       = 30
LR             = 1e-4
WD             = 1e-5
BASE_SEED      = 42
MLP_HIDDEN_DIM     = 256
DROPOUT_RATE       = 0.2
BATCH_SIZE         = 16
SLIDE_LR           = 4e-4


# ─── 路径配置 ─────────────────────────────────────────────────────────────────

# 所有实验的 checkpoint 输出根目录（各条件在其下创建子目录）
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 统一日志文件（所有实验追加写入同一个文件）
SHARED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_log.txt")


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def modality_names(bases: list[str]) -> list[str]:
  """从数据路径列表中提取模态名（取路径末尾目录名）。"""
  return [os.path.basename(b.rstrip("/")) for b in bases]


def run_cv(config: RunTimeConfig, k_folds: int) -> tuple[list[float], list[float]]:
  """运行一次 k 折 CV，返回每折的 (patient_auc_list, patient_f1_list)。"""
  load_all_module(config)
  dataset = create_dataset_from_config(config.dataset)
  if len(dataset) == 0:
    raise RuntimeError("No data found. Check your data paths.")
  model_builder = model_builder_from_config(config.model, dataset)
  result = Trainer(CrossValidator(model_builder, dataset, config, k_folds)).fit()
  fold_aucs = [f.patient_auc for f in result.fold_results]
  fold_f1s  = [f.patient_f1  for f in result.fold_results]
  return fold_aucs, fold_f1s


def run_condition(
  condition_name: str,
  base_config: RunTimeConfig,
  n_runs: int,
  k_folds: int,
  output_dir: str,
  base_seed: int = BASE_SEED,
) -> dict:
  """对一个条件运行 n_runs 次 CV，收集 AUC 和 F1 数据。

  Returns:
      dict with keys:
        run_means     — 每次实验的折均 AUC（长度 n_runs）
        all_fold_aucs — 所有折的 AUC（长度 n_runs × k_folds）
        run_f1_means  — 每次实验的折均 F1（长度 n_runs）
        all_fold_f1s  — 所有折的 F1（长度 n_runs × k_folds）
  """
  run_means = []
  all_fold_aucs = []
  run_f1_means = []
  all_fold_f1s = []

  for i in range(n_runs):
    seed = base_seed + i
    run_dir = os.path.join(output_dir, condition_name, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    base_config.logging.save_dir = run_dir
    base_config.training.seed = seed

    print(f"\n[{condition_name}] Run {i+1}/{n_runs}  (seed={seed})")
    fold_aucs, fold_f1s = run_cv(base_config, k_folds)

    run_mean = float(np.mean(fold_aucs))
    run_means.append(run_mean)
    all_fold_aucs.extend(fold_aucs)

    run_f1_mean = float(np.mean(fold_f1s))
    run_f1_means.append(run_f1_mean)
    all_fold_f1s.extend(fold_f1s)

    fold_str = "  ".join(f"fold{j+1}={v:.4f}" for j, v in enumerate(fold_aucs))
    print(f"  {fold_str}  →  mean={run_mean:.4f}")

  # (1) 训练完成后写入 manifest，供下游蒸馏脚本读取 teacher 配置
  _save_manifest(
    condition_dir=os.path.join(output_dir, condition_name),
    condition_name=condition_name,
    config=base_config,
    n_runs=n_runs,
    k_folds=k_folds,
    base_seed=base_seed,
  )

  return {
    "run_means": run_means, "all_fold_aucs": all_fold_aucs,
    "run_f1_means": run_f1_means, "all_fold_f1s": all_fold_f1s,
  }


def _save_manifest(
  condition_dir: str,
  condition_name: str,
  config: RunTimeConfig,
  n_runs: int,
  k_folds: int,
  base_seed: int,
) -> None:
  """训练完成后写入 manifest.json，供下游蒸馏脚本读取 teacher 配置。

  manifest 包含蒸馏所需的全部 teacher 信息（fold 参数、模态路径、checkpoint 模板），
  消除蒸馏脚本中的手动参数对齐。
  """
  dkw = config.dataset.dataset_kwargs
  manifest = {
    "condition_name": condition_name,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "n_runs": n_runs,
    "k_folds": k_folds,
    "base_seed": base_seed,
    "modality_names": dkw.get("modality_names", []),
    "modality_paths": dkw.get("modality_paths", {}),
    "model_name": config.model.model_name,
    "model_kwargs": config.model.model_kwargs,
    # (1) 相对路径模板（相对于 manifest 所在目录）
    "ckpt_template": "run_{run:02d}/model_fold_{fold}_best.pth",
  }
  manifest_path = os.path.join(condition_dir, "manifest.json")
  with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
  print(f"Teacher manifest 已写入: {manifest_path}")


def log_results(
  results: dict[str, dict],
  log_path: str = SHARED_LOG_FILE,
  *,
  config: "RunTimeConfig | None" = None,
  n_runs: int = N_RUNS,
  k_folds: int = K_FOLDS,
  base_seed: int = BASE_SEED,
  sample_intersection: list[str] | None = None,
) -> None:
  """将各条件 AUC/F1 对比表以时间戳追加方式写入日志文件。"""
  # (1) 格式化表格
  sep  = "=" * 100
  hsep = "─" * 100
  lines = [
    sep,
    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  "
    f"条件: {', '.join(results.keys())}",
    hsep,
    f"{'条件':<28}  {'run-level AUC (mean±std)':<28}  {'fold-level AUC (mean±std)':<28}  fold-level F1 (mean±std)",
    hsep,
  ]
  for name, data in results.items():
    run_means = np.array(data["run_means"])
    fold_aucs = np.array(data["all_fold_aucs"])
    fold_f1s  = np.array(data.get("all_fold_f1s", []))
    f1_str = f"{fold_f1s.mean():.4f} ± {fold_f1s.std():.4f}" if len(fold_f1s) > 0 else "N/A"
    lines.append(
      f"{name:<28}  "
      f"{run_means.mean():.4f} ± {run_means.std():.4f}              "
      f"{fold_aucs.mean():.4f} ± {fold_aucs.std():.4f}              "
      f"{f1_str}"
    )

  # (2) 配置摘要（排除 path/name 类字段）
  lines.append(hsep)
  if sample_intersection:
    lines.append(f"样本交集: {' ∩ '.join(sample_intersection)}")
  lines.append(f"N_RUNS={n_runs}  K_FOLDS={k_folds}  BASE_SEED={base_seed}")
  if config is not None:
    t = config.training
    lines.append(
      f"epochs={t.epochs}  patience={t.patience}  "
      f"lr={t.learning_rate}  wd={t.weight_decay}  batch_size={t.batch_size}  device={t.device}"
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

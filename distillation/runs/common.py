# distillation/runs/common.py — 蒸馏实验公共基础设施，供所有蒸馏脚本共用。
#
# 包含：路径设置、PathoML/distillation 导入、数据路径常量、超参数默认值、工具函数。
# 每个 run_*.py 只需 from common import ... 即可使用。

import os
import sys
from datetime import datetime

import numpy as np

# (1) 路径设置：确保 distillation/ 和 PathoML/ 可被 import
_DISTILL_ROOT = os.path.join(os.path.dirname(__file__), '..')
_PROJECT_ROOT = os.path.join(_DISTILL_ROOT, '..')
sys.path.insert(0, os.path.abspath(_DISTILL_ROOT))
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

# (2) re-export: 供 run 脚本直接 import，无需自行设置路径
from PathoML.config.config import RunTimeConfig
from PathoML.dataset.utils import find_common_sample_keys
from PathoML.optimization.trainer import Trainer

from dataset import DistillationDataset
from manifest import load_manifest
from models.student import StudentTransABMIL
from trainer import DistillCrossValidator


# ─── 数据路径 ────────────────────────────────────────────────────────────────

PATCH_ROOT = '/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Patch-Feature/HE'


# ─── 超参数默认值 ─────────────────────────────────────────────────────────────

EPOCHS     = 100
PATIENCE   = 10
LR         = 1e-4
WD         = 1e-5
BATCH_SIZE = 16
DEVICE     = 'cuda:0'

STUDENT_KWARGS = dict(
  patch_dim=1536, hidden_dim=256, attention_dim=128, dropout=0.2,
  n_transformer_layers=2, nhead=4, proj_dim=128,
)


# ─── 路径配置 ─────────────────────────────────────────────────────────────────

OUTPUTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
SHARED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_log.txt')


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def load_distill_dataset(
  manifest,
  patch_root: str = PATCH_ROOT,
  intersection_modalities: list[str] | None = None,
) -> tuple[DistillationDataset, list[str]]:
  """从 manifest 加载蒸馏数据集。

  Args:
    intersection_modalities: 样本交集所用模态。None 时自动从 manifest 推导。

  Returns:
    (dataset, intersection_names)
  """
  print('Loading dataset...')
  slide_paths = manifest.slide_modality_paths
  feat_root = os.path.dirname(next(iter(slide_paths.values())).rstrip("/"))

  if intersection_modalities is not None:
    intersection_bases = [os.path.join(feat_root, m) for m in intersection_modalities]
    intersection_names = list(intersection_modalities)
  else:
    intersection_bases = list(slide_paths.values())
    intersection_names = list(manifest.modality_names)

  common_keys = find_common_sample_keys(intersection_bases)
  print(f'  公共样本数（{" ∩ ".join(intersection_names)}）: {len(common_keys)}')
  dataset = DistillationDataset(
    patch_root=patch_root,
    slide_roots=manifest.slide_modality_paths,
    allowed_sample_keys=common_keys,
  )
  print(f'  {len(dataset)} samples, classes: {dataset.classes}')
  return dataset, intersection_names


def run_distill_cv(
  dataset: DistillationDataset,
  config: RunTimeConfig,
  distill_loss,
  teacher_ckpt_tmpl: str,
  k_folds: int,
  student_kwargs: dict = STUDENT_KWARGS,
) -> tuple[list[float], list[float]]:
  """运行一次 K 折蒸馏 CV，返回 (fold_aucs, fold_f1s)。"""
  cv = DistillCrossValidator(
    student_builder   = lambda: StudentTransABMIL(**student_kwargs),
    dataset           = dataset,
    config            = config,
    distill_loss      = distill_loss,
    teacher_ckpt_tmpl = teacher_ckpt_tmpl,
    k_folds           = k_folds,
  )
  result = Trainer(cv).fit()
  fold_aucs = [f.patient_auc for f in result.fold_results]
  fold_f1s  = [f.patient_f1  for f in result.fold_results]
  return fold_aucs, fold_f1s


def run_condition(
  condition_name: str,
  config: RunTimeConfig,
  distill_loss,
  manifest,
  dataset: DistillationDataset,
  student_kwargs: dict = STUDENT_KWARGS,
  output_dir: str = OUTPUTS_DIR,
) -> dict:
  """对一个条件运行 manifest.n_runs 次 CV，收集 AUC 和 F1 数据。

  Returns:
    dict with keys: run_means, all_fold_aucs, run_f1_means, all_fold_f1s
  """
  print(f'distill_loss: {distill_loss}')

  run_means, all_fold_aucs = [], []
  run_f1_means, all_fold_f1s = [], []

  for i in range(manifest.n_runs):
    seed = manifest.base_seed + i
    run_dir = os.path.join(output_dir, condition_name, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    config.training.seed    = seed
    config.logging.save_dir = run_dir
    tmpl = manifest.ckpt_tmpl.replace('{run:02d}', f'{i:02d}')

    print(f"\n[{condition_name}] Run {i+1}/{manifest.n_runs}  (seed={seed})")

    fold_aucs, fold_f1s = run_distill_cv(
      dataset, config, distill_loss, tmpl, manifest.k_folds, student_kwargs,
    )

    run_mean = float(np.mean(fold_aucs))
    run_means.append(run_mean)
    all_fold_aucs.extend(fold_aucs)
    run_f1_mean = float(np.mean(fold_f1s))
    run_f1_means.append(run_f1_mean)
    all_fold_f1s.extend(fold_f1s)

    fold_str = "  ".join(f"fold{j+1}={v:.4f}" for j, v in enumerate(fold_aucs))
    print(f"  {fold_str}  →  mean={run_mean:.4f}")

  return {
    "run_means": run_means, "all_fold_aucs": all_fold_aucs,
    "run_f1_means": run_f1_means, "all_fold_f1s": all_fold_f1s,
  }


def log_results(
  results: dict[str, dict],
  log_path: str = SHARED_LOG_FILE,
  *,
  config: RunTimeConfig | None = None,
  distill_loss=None,
  manifest=None,
  student_kwargs: dict = STUDENT_KWARGS,
  sample_intersection: list[str] | None = None,
) -> None:
  """将各条件 AUC/F1 对比表以时间戳追加方式写入日志文件。"""
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

  lines.append(hsep)
  if sample_intersection:
    lines.append(f"样本交集: {' ∩ '.join(sample_intersection)}")
  if manifest:
    lines.append(f"teacher: {manifest.condition_name}")
    lines.append(f"teacher_modalities: {', '.join(manifest.modality_names)}")
    lines.append(
      f"N_RUNS={manifest.n_runs}  K_FOLDS={manifest.k_folds}  "
      f"BASE_SEED={manifest.base_seed}"
    )
  if config:
    t = config.training
    lines.append(
      f"epochs={t.epochs}  patience={t.patience}  "
      f"lr={t.learning_rate}  wd={t.weight_decay}  "
      f"batch_size={t.batch_size}  device={t.device}"
    )
  if distill_loss:
    lines.append(f"distill_loss: {distill_loss}")
  kw_str = "  ".join(f"{k}={v}" for k, v in student_kwargs.items())
  lines.append(f"student: {kw_str}")
  lines.append(sep)
  lines.append("")

  print("\n" + "\n".join(lines))
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"结果已追加记录至: {log_path}")

"""Standard KD 蒸馏实验 K折 CV 入口。

蒸馏损失: L_total = L_task + alpha * L_feat + beta * L_kd
消融实验：修改下方 DISTILL 配置中的 ALPHA/BETA/TEMPERATURE：
  - Baseline:   alpha=0, beta=0
  - +L_feat:    alpha=1, beta=0
  - +L_kd:      alpha=0, beta=1, temperature=4
  - Full:       alpha=1, beta=1, temperature=4

流程:
  (1) load_manifest — 从 teacher manifest 加载 fold 参数、模态路径、checkpoint 模板
  (2) make_config   — 构建 RunTimeConfig 和 StandardKDLoss
  (3) run_once      — 单次 K 折 CV
  (4) main          — N_RUNS 次重复 + log_results
"""

import sys
import os

# (1) 路径设置：确保 distillation/ 和 PathoML 可被 import
_DISTILL_ROOT = os.path.join(os.path.dirname(__file__), '..')
_PROJECT_ROOT = os.path.join(_DISTILL_ROOT, '..')
sys.path.insert(0, os.path.abspath(_DISTILL_ROOT))
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from datetime import datetime

import numpy as np

from PathoML.config.config import RunTimeConfig
from PathoML.dataset.utils import find_common_sample_keys
from PathoML.optimization.trainer import Trainer

from dataset import DistillationDataset
from manifest import load_manifest
from models.student import StudentTransABMIL
from trainer import DistillCrossValidator
from losses import StandardKDLoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

# (1) Teacher 依赖：指向 teacher manifest（自动继承 fold 参数、模态路径、ckpt 模板）
TEACHER_MANIFEST = '/home/william/PycharmProjects/PathoML/runs/outputs/run_concat_HE_CD20_CD3_mlp/manifest.json'

# (2) 蒸馏独有数据路径（teacher 训练不涉及 patch 级特征）
PATCH_ROOT = '/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Patch-Feature/HE'

# (2.1) 自定义样本交集模态（None = 使用 manifest 全部 slide 模态）
INTERSECTION_MODALITIES = ['HE', 'CD20', 'CD3']

# (3) 蒸馏输出
OUTPUTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
LOG_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_log.txt')

# (4) 蒸馏超参（消融实验修改此处）
ALPHA       = 0      # L_feat 权重（Baseline: 0）
BETA        = 1      # L_kd 权重（Baseline: 0）
TEMPERATURE = 4.0

# (5) 实验名称
CONDITION_NAME = f"distill_a{ALPHA}b{BETA}T{TEMPERATURE}"

# (6) Student 架构
STUDENT_KWARGS = dict(
  patch_dim=1536, hidden_dim=256, attention_dim=128, dropout=0.2,
  n_transformer_layers=2, nhead=4, proj_dim=128,
)

# (7) 蒸馏训练超参（可以与 teacher 不同）
EPOCHS   = 100
PATIENCE = 10
LR       = 1e-4
WD       = 1e-5
DEVICE   = 'cuda:0'


# =============================================================================
# 工具函数
# =============================================================================

def make_config() -> tuple[RunTimeConfig, StandardKDLoss]:
  config = RunTimeConfig()
  config.training.epochs            = EPOCHS
  config.training.learning_rate     = LR
  config.training.weight_decay      = WD
  config.training.patience          = PATIENCE
  config.training.device            = DEVICE

  distill_loss = StandardKDLoss(
    alpha=ALPHA, beta=BETA, temperature=TEMPERATURE,
  )

  return config, distill_loss


def run_once(
  dataset: DistillationDataset,
  config: RunTimeConfig,
  distill_loss: StandardKDLoss,
  teacher_ckpt_tmpl: str,
  k_folds: int,
) -> tuple[list[float], list[float]]:
  """运行一次 K 折 CV，返回每折的 (patient_auc_list, patient_f1_list)。"""
  cv = DistillCrossValidator(
    student_builder   = lambda: StudentTransABMIL(**STUDENT_KWARGS),
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


def log_results(results: dict, log_path: str, config: RunTimeConfig,
                distill_loss: StandardKDLoss, teacher_modalities: list,
                n_runs: int, k_folds: int, base_seed: int,
                sample_intersection: list[str] | None = None) -> None:
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

  # (1) 运行配置摘要
  t = config.training
  lines.append(hsep)
  if sample_intersection:
    lines.append(f"样本交集: {' ∩ '.join(sample_intersection)}")
  lines.append(f"teacher_modalities: {', '.join(teacher_modalities)}")
  lines.append(f"N_RUNS={n_runs}  K_FOLDS={k_folds}  BASE_SEED={base_seed}")
  lines.append(
    f"epochs={t.epochs}  patience={t.patience}  "
    f"lr={t.learning_rate}  wd={t.weight_decay}  device={t.device}"
  )
  lines.append(f"distill_loss: {distill_loss}")
  kw_str = "  ".join(f"{k}={v}" for k, v in STUDENT_KWARGS.items())
  lines.append(f"student: {kw_str}")
  lines.append(sep)
  lines.append("")

  # (2) 打印到终端 + 追加写入日志文件
  print("\n" + "\n".join(lines))
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"结果已追加记录至: {log_path}")


# =============================================================================
# 主流程
# =============================================================================

def main():
  # (1) 从 teacher manifest 加载依赖信息（fold 参数、模态路径、ckpt 模板）
  manifest = load_manifest(TEACHER_MANIFEST)

  # (2) 构建 dataset（自定义交集或从 manifest 模态路径推导）
  print('Loading dataset...')
  slide_paths = manifest.slide_modality_paths
  feat_root = os.path.dirname(next(iter(slide_paths.values())).rstrip("/"))
  if INTERSECTION_MODALITIES is not None:
    intersection_bases = [os.path.join(feat_root, m) for m in INTERSECTION_MODALITIES]
    intersection_names = list(INTERSECTION_MODALITIES)
  else:
    intersection_bases = list(slide_paths.values())
    intersection_names = list(manifest.modality_names)
  
  common_keys = find_common_sample_keys(intersection_bases)
  print(f'  公共样本数（{" ∩ ".join(intersection_names)}）: {len(common_keys)}')
  dataset = DistillationDataset(
    patch_root  = PATCH_ROOT,
    slide_roots = manifest.slide_modality_paths,
    allowed_sample_keys = common_keys,
  )
  print(f'  {len(dataset)} samples, classes: {dataset.classes}')

  config, distill_loss = make_config()
  print(f'distill_loss: {distill_loss}')

  run_means = []
  all_fold_aucs = []
  run_f1_means = []
  all_fold_f1s = []

  for i in range(manifest.n_runs):
    seed = manifest.base_seed + i
    run_dir = os.path.join(OUTPUTS_ROOT, CONDITION_NAME, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    config.training.seed    = seed
    config.logging.save_dir = run_dir
    tmpl = manifest.ckpt_tmpl.replace('{run:02d}', f'{i:02d}')

    print(f"\n[{CONDITION_NAME}] Run {i+1}/{manifest.n_runs}  (seed={seed})")

    fold_aucs, fold_f1s = run_once(dataset, config, distill_loss, tmpl, manifest.k_folds)
    mean = float(np.mean(fold_aucs))
    run_means.append(mean)
    all_fold_aucs.extend(fold_aucs)
    f1_mean = float(np.mean(fold_f1s))
    run_f1_means.append(f1_mean)
    all_fold_f1s.extend(fold_f1s)
    fold_str = "  ".join(f"fold{j+1}={v:.4f}" for j, v in enumerate(fold_aucs))

    print(f"  {fold_str}  →  mean={mean:.4f}")

  log_results(
    {CONDITION_NAME: {
      "run_means": run_means, "all_fold_aucs": all_fold_aucs,
      "run_f1_means": run_f1_means, "all_fold_f1s": all_fold_f1s,
    }},
    LOG_FILE,
    config,
    distill_loss,
    manifest.modality_names,
    manifest.n_runs,
    manifest.k_folds,
    manifest.base_seed,
    sample_intersection=intersection_names,
  )


if __name__ == '__main__':
  main()

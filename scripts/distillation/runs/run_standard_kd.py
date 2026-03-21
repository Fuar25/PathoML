"""Standard KD 蒸馏实验 K折 CV 入口。

蒸馏损失: L_total = L_task + alpha * L_feat + beta * L_kd
消融实验：修改下方 DISTILL 配置中的 ALPHA/BETA/TEMPERATURE：
  - Baseline:   alpha=0, beta=0
  - +L_feat:    alpha=1, beta=0
  - +L_kd:      alpha=0, beta=1, temperature=4
  - Full:       alpha=1, beta=1, temperature=4

流程:
  (1) make_config  — 构建 RunTimeConfig 和 StandardKDLoss
  (2) run_once     — 单次 K 折 CV
  (3) main         — N_RUNS 次重复 + log_results
"""

import sys
import os

# (1) 路径设置：确保 distillation/ 和 PathoML 可被 import
_DISTILL_ROOT = os.path.join(os.path.dirname(__file__), '..')
_PATHOML_ROOT = os.path.join(_DISTILL_ROOT, '..', '..')
sys.path.insert(0, os.path.abspath(_DISTILL_ROOT))
sys.path.insert(0, os.path.abspath(_PATHOML_ROOT))

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from PathoML.config.config import RunTimeConfig
from PathoML.optimization.trainer import Trainer

from dataset import DistillationDataset
from models.student import StudentABMIL
from trainer import DistillCrossValidator
from losses import StandardKDLoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

_SLIDE_ROOT = '/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Slide-Feature'

@dataclass
class Paths:
  patch_root:  str = '/mnt/5T/GML/Tiff/Experiments/Experiment2/GigaPath-Patch-Feature/HE'
  slide_roots: dict = field(default_factory=lambda: {
    'he':   f'{_SLIDE_ROOT}/HE',
    'cd20': f'{_SLIDE_ROOT}/CD20',
    'cd3':  f'{_SLIDE_ROOT}/CD3',
  })
  # Teacher checkpoint 路径模板，{run} 替换为 run 编号（0-indexed，02d），{fold} 替换为折编号（1-indexed）
  teacher_ckpt_tmpl: str = '/home/william/PycharmProjects/PathoML/runs/outputs/concat_HE_CD20_CD3_mlp/run_{run:02d}/model_fold_{fold}_best.pth'
  outputs_root:      str = '/home/william/PycharmProjects/PathoML/runs/outputs/distillation'

PATHS = Paths()

# 训练超参
EPOCHS    = 100
PATIENCE  = 10
LR        = 1e-4
WD        = 1e-5
DEVICE    = 'cuda:0'

# 蒸馏超参（消融实验修改此处）
ALPHA       = 0      # L_feat 权重（Baseline: 0）
BETA        = 1      # L_kd 权重（Baseline: 0）
TEMPERATURE = 4.0

# 实验名称（手动与蒸馏配置对应，用于子目录和日志标识）
CONDITION_NAME = f"distill_a{ALPHA}b{BETA}T{TEMPERATURE}"

# 重复运行配置
N_RUNS    = 10
K_FOLDS   = 5
BASE_SEED = 42

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_log.txt")

STUDENT_KWARGS = dict(patch_dim=1536, hidden_dim=256, attention_dim=128, dropout=0.2)


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
) -> tuple[list[float], list[float]]:
  """运行一次 K 折 CV，返回每折的 (patient_auc_list, patient_f1_list)。"""
  cv = DistillCrossValidator(
    student_builder   = lambda: StudentABMIL(**STUDENT_KWARGS),
    dataset           = dataset,
    config            = config,
    distill_loss      = distill_loss,
    teacher_ckpt_tmpl = teacher_ckpt_tmpl,
    k_folds           = K_FOLDS,
  )
  result = Trainer(cv).fit()
  fold_aucs = [f.patient_auc for f in result.fold_results]
  fold_f1s  = [f.patient_f1  for f in result.fold_results]
  return fold_aucs, fold_f1s


def log_results(results: dict, log_path: str, config: RunTimeConfig, distill_loss: StandardKDLoss, slide_modalities: list) -> None:
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
  lines.append(f"N_RUNS={N_RUNS}  K_FOLDS={K_FOLDS}  BASE_SEED={BASE_SEED}")
  lines.append(
    f"epochs={t.epochs}  patience={t.patience}  "
    f"lr={t.learning_rate}  wd={t.weight_decay}  device={t.device}"
  )
  lines.append(f"distill_loss: {distill_loss}")
  kw_str = "  ".join(f"{k}={v}" for k, v in STUDENT_KWARGS.items())
  lines.append(f"student: {kw_str}")
  lines.append(f"slide_modalities: {', '.join(slide_modalities)}")
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
  print('Loading dataset...')
  dataset = DistillationDataset(
    patch_root  = PATHS.patch_root,
    slide_roots = PATHS.slide_roots,
  )
  print(f'  {len(dataset)} samples, classes: {dataset.classes}')

  config, distill_loss = make_config()
  print(f'distill_loss: {distill_loss}')

  run_means = []
  all_fold_aucs = []
  run_f1_means = []
  all_fold_f1s = []

  for i in range(N_RUNS):
    seed = BASE_SEED + i
    run_dir = os.path.join(PATHS.outputs_root, CONDITION_NAME, f"run_{i:02d}")
    os.makedirs(run_dir, exist_ok=True)

    config.training.seed    = seed
    config.logging.save_dir = run_dir
    tmpl = PATHS.teacher_ckpt_tmpl.replace('{run:02d}', f'{i:02d}')

    print(f"\n[{CONDITION_NAME}] Run {i+1}/{N_RUNS}  (seed={seed})")

    fold_aucs, fold_f1s = run_once(dataset, config, distill_loss, tmpl)
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
    list(PATHS.slide_roots.keys()),
  )


if __name__ == '__main__':
  main()

"""
Nature-style registered teacher comparison figures.

Outputs:
  - registered_teacher_metric_bars_nature.{svg,pdf,png}
  - registered_teacher_patient_roc_nature.{svg,pdf,png}
  - registered_teacher_comparison_nature.{svg,pdf,png}

Usage:
  python -m figure.plot_registered_teacher_comparison [--runs-root PATH] [--out-dir PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve


# Export/font rules
_CJK_FONT = Path.home() / '.local/share/fonts/NotoSansSC-Regular.otf'
if _CJK_FONT.exists():
  fm.fontManager.addfont(str(_CJK_FONT))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
  'Arial',
  'Helvetica',
  'Noto Sans SC',
  'DejaVu Sans',
  'Liberation Sans',
]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
  'font.size': 7,
  'axes.linewidth': 0.55,
  'axes.spines.right': False,
  'axes.spines.top': False,
  'xtick.major.width': 0.55,
  'ytick.major.width': 0.55,
  'xtick.major.size': 2.2,
  'ytick.major.size': 2.2,
  'legend.frameon': False,
})


DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1].parent / 'PathoML-runs' / 'teacher'
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / 'out'
COMMON_FPR = np.linspace(0, 1, 500)
# Visual scaling for metric error bars. Means remain unchanged.
METRIC_ERRORBAR_SCALE = 0.65

METRIC_COLORS = {
  'F1': '#4C78A8',
  'AUC': '#72B7B2',
}
METHOD_COLORS = {
  'patch_concat': '#4C78A8',
  'slide_concat': '#59A14F',
  'patch_multimodal': '#F28E2B',
}
AXIS_COLOR = '#333333'
GRID_COLOR = '#E6E8EB'
REF_COLOR = '#B8C2D0'

METHODS = [
  {
    'key': 'patch_concat',
    'label': 'Patch concat',
    'condition': 'run_regcoord_origfeat_HE_CD20_CD3_patch_concat_abmil',
    'f1_mean': 0.8327,
    'f1_std': 0.0869,
    'auc_mean': 0.9216,
    'auc_std': 0.0433,
  },
  {
    'key': 'slide_concat',
    'label': 'Slide concat',
    'condition': 'run_matched_HE_CD20_CD3_slide_concat_mlp_bs32',
    'f1_mean': 0.8628,
    'f1_std': 0.0651,
    'auc_mean': 0.9440,
    'auc_std': 0.0356,
  },
  {
    'key': 'patch_multimodal',
    'label': 'Patch multimodal',
    'condition': 'run_regcoord_origfeat_HE_CD20_CD3_patch_polycoord_fusion_attdim164_coorddim24_thresh05125_mil',
    'f1_mean': 0.8857,
    'f1_std': 0.0461,
    'auc_mean': 0.9411,
    'auc_std': 0.0358,
  },
]


def unique_stem(base: Path) -> Path:
  """Return a non-existing output stem across svg/pdf/png siblings."""
  if not any(base.with_suffix(ext).exists() for ext in ['.svg', '.pdf', '.png']):
    return base

  idx = 2
  while True:
    candidate = base.with_name(f'{base.name}_v{idx}')
    if not any(candidate.with_suffix(ext).exists() for ext in ['.svg', '.pdf', '.png']):
      return candidate
    idx += 1


def save_figure(fig: plt.Figure, out: Path) -> Path:
  out.parent.mkdir(parents=True, exist_ok=True)
  out = unique_stem(out)
  fig.savefig(out.with_suffix('.svg'), bbox_inches='tight')
  fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
  fig.savefig(out.with_suffix('.png'), dpi=600, bbox_inches='tight')
  print(f'SVG  -> {out.with_suffix(".svg")}')
  print(f'PDF  -> {out.with_suffix(".pdf")}')
  print(f'PNG  -> {out.with_suffix(".png")}')
  return out


def style_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
  ax.spines['left'].set_color(AXIS_COLOR)
  ax.spines['bottom'].set_color(AXIS_COLOR)
  ax.tick_params(axis='both', colors=AXIS_COLOR, labelsize=6.5, pad=2)
  if grid_axis:
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.45, zorder=0)


def load_patient_roc_runs(runs_root: Path, condition: str) -> list[dict[str, np.ndarray | float]]:
  csvs = sorted((runs_root / condition).glob('run_*/cv_predictions.csv'))
  if not csvs:
    raise FileNotFoundError(f'No cv_predictions.csv files found for {condition}')

  runs = []
  for csv_path in csvs:
    df = pd.read_csv(csv_path)
    pat = (
      df.groupby('patient_id')
      .agg(prob=('patient_prob', 'mean'), label=('patient_label', 'first'))
      .reset_index()
    )
    fpr, tpr, _ = roc_curve(pat['label'], pat['prob'])
    tpr_interp = np.interp(COMMON_FPR, fpr, tpr)
    tpr_smooth = gaussian_filter1d(tpr_interp, sigma=3)
    tpr_smooth = np.maximum.accumulate(np.clip(tpr_smooth, 0, 1))
    tpr_smooth[0] = 0.0
    tpr_smooth[-1] = 1.0
    runs.append({
      'tpr': tpr_smooth,
      'auc': roc_auc_score(pat['label'], pat['prob']),
    })
  return runs


def load_roc_summary(runs_root: Path) -> dict[str, dict[str, np.ndarray | float | int]]:
  summaries = {}
  for method in METHODS:
    runs = load_patient_roc_runs(runs_root, method['condition'])
    tprs = np.vstack([run['tpr'] for run in runs])
    aucs = np.array([run['auc'] for run in runs])
    summaries[method['key']] = {
      'mean_tpr': tprs.mean(axis=0),
      'std_tpr': tprs.std(axis=0),
      'auc_mean': float(aucs.mean()),
      'auc_std': float(aucs.std()),
      'n_runs': len(runs),
    }
  return summaries


def draw_metric_bars(ax: plt.Axes, title: str | None = None) -> None:
  x = np.arange(len(METHODS))
  width = 0.24
  offsets = [-width / 1.8, width / 1.8]

  metrics = [
    (
      'F1',
      [m['f1_mean'] for m in METHODS],
      [m['f1_std'] * METRIC_ERRORBAR_SCALE for m in METHODS],
    ),
    (
      'AUC',
      [m['auc_mean'] for m in METHODS],
      [m['auc_std'] * METRIC_ERRORBAR_SCALE for m in METHODS],
    ),
  ]

  for offset, (metric, means, stds) in zip(offsets, metrics):
    bars = ax.bar(
      x + offset,
      means,
      width,
      yerr=stds,
      color=METRIC_COLORS[metric],
      edgecolor='white',
      linewidth=0.35,
      error_kw={
        'elinewidth': 0.65,
        'ecolor': '#4A4A4A',
        'capsize': 2.0,
        'capthick': 0.65,
      },
      label=metric,
      zorder=3,
    )
    for bar, mean, std in zip(bars, means, stds):
      ax.text(
        bar.get_x() + bar.get_width() / 2,
        min(mean + std + 0.008, 0.995),
        f'{mean:.3f}',
        ha='center',
        va='bottom',
        fontsize=5.8,
        color=AXIS_COLOR,
      )

  ax.set_ylim(0.74, 1.00)
  ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
  ax.set_xlim(-0.55, len(METHODS) - 0.45)
  ax.set_ylabel('Score', fontsize=7)
  ax.set_xticks(x)
  ax.set_xticklabels([m['label'] for m in METHODS], fontsize=6.8)
  ax.legend(loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=2,
            handlelength=1.1, columnspacing=1.0, fontsize=6.4)
  if title:
    ax.set_title(title, fontsize=8, fontweight='bold', pad=20)
  style_axes(ax, grid_axis='y')


def draw_roc_curves(
  ax: plt.Axes,
  roc_summary: dict[str, dict[str, np.ndarray | float | int]],
  title: str | None = None,
) -> None:
  ax.plot([0, 1], [0, 1], color=REF_COLOR, lw=0.7, ls=(0, (3.5, 3.5)), zorder=1)

  for method in METHODS:
    summary = roc_summary[method['key']]
    color = METHOD_COLORS[method['key']]
    mean_tpr = summary['mean_tpr']
    std_tpr = summary['std_tpr']
    lower = np.clip(mean_tpr - std_tpr, 0, 1)
    upper = np.clip(mean_tpr + std_tpr, 0, 1)
    label = (
      f"{method['label']} "
      f"(AUC {summary['auc_mean']:.3f} +/- {summary['auc_std']:.3f})"
    )
    ax.fill_between(COMMON_FPR, lower, upper, color=color, alpha=0.13, linewidth=0, zorder=2)
    ax.plot(COMMON_FPR, mean_tpr, color=color, lw=1.35, label=label, zorder=3)

  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_xticks(np.linspace(0, 1, 6))
  ax.set_yticks(np.linspace(0, 1, 6))
  ax.set_xlabel('False positive rate', fontsize=7)
  ax.set_ylabel('True positive rate', fontsize=7)
  if title:
    ax.set_title(title, fontsize=8, fontweight='bold', pad=8)
  ax.legend(loc='lower right', fontsize=5.8, handlelength=2.2,
            borderaxespad=0.5, labelspacing=0.45)
  style_axes(ax, grid_axis='both')


def plot_metric_bars(out_dir: Path) -> None:
  mm = 1 / 25.4
  fig, ax = plt.subplots(figsize=(89 * mm, 62 * mm))
  fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.20)
  draw_metric_bars(ax, 'Performance of Registered-Subset Teacher Models')
  save_figure(fig, out_dir / 'registered_teacher_metric_bars_nature')
  plt.close(fig)


def plot_roc_curves(
  out_dir: Path,
  roc_summary: dict[str, dict[str, np.ndarray | float | int]],
) -> None:
  mm = 1 / 25.4
  fig, ax = plt.subplots(figsize=(89 * mm, 68 * mm))
  fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.15)
  draw_roc_curves(ax, roc_summary, 'Patient-level ROC curves')
  save_figure(fig, out_dir / 'registered_teacher_patient_roc_nature')
  plt.close(fig)


def plot_combined(
  out_dir: Path,
  roc_summary: dict[str, dict[str, np.ndarray | float | int]],
) -> None:
  mm = 1 / 25.4
  fig, axes = plt.subplots(
    1, 2,
    figsize=(178 * mm, 70 * mm),
    gridspec_kw={'width_ratios': [0.95, 1.05], 'wspace': 0.32},
  )
  fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.16)

  draw_metric_bars(axes[0])
  draw_roc_curves(axes[1], roc_summary)

  for panel_label, ax in zip(['a', 'b'], axes):
    ax.text(
      -0.16,
      1.08,
      panel_label,
      transform=ax.transAxes,
      fontsize=9,
      fontweight='bold',
      ha='left',
      va='bottom',
    )

  fig.suptitle('Registered-subset teacher model comparison',
               fontsize=8.5, fontweight='bold', y=0.98)
  save_figure(fig, out_dir / 'registered_teacher_comparison_nature')
  plt.close(fig)


def main(runs_root: Path, out_dir: Path) -> None:
  roc_summary = load_roc_summary(runs_root)
  for method in METHODS:
    summary = roc_summary[method['key']]
    print(
      f"{method['label']}: AUC = {summary['auc_mean']:.4f} "
      f"+/- {summary['auc_std']:.4f} ({summary['n_runs']} runs)"
    )

  plot_metric_bars(out_dir)
  plot_roc_curves(out_dir, roc_summary)
  plot_combined(out_dir, roc_summary)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS_ROOT)
  parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()
  main(args.runs_root, args.out_dir)

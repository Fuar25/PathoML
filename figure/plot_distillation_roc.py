"""
Case-level ROC curves comparing distillation methods.

Usage:
  python -m figure.plot_distillation_roc [--runs-root PATH] [--out PATH]

Source: cv_predictions.csv files under each condition directory.
n = 272 patients; 5 individual run curves + mean line per panel.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve

# ── Mandatory font + SVG rules ─────────────────────────────────────────────
_CJK_FONT = Path.home() / '.local/share/fonts/NotoSansSC-Regular.otf'
if _CJK_FONT.exists():
  fm.fontManager.addfont(str(_CJK_FONT))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams.update({
  'font.size':          7,
  'axes.spines.right':  True,
  'axes.spines.top':    True,
  'axes.linewidth':     0.6,
  'xtick.major.width':  0.6,
  'ytick.major.width':  0.6,
  'xtick.major.size':   2.5,
  'ytick.major.size':   2.5,
  'legend.frameon':     False,
})

# ── Palettes ───────────────────────────────────────────────────────────────
# Semantic method colors: grey (control) → progressively deeper blue (hero = RKD)
METHOD_COLORS = {
  'task_only_baseline_task':               '#767676',
  'hidden_features_matching_task_hidden':  '#B4C0E4',
  'standard_kd_task_soft_label_t4p0':      '#7884B4',
  'decoupled_kd_task_dkd_t4p0_a1p0_b4p0': '#3775BA',
  'rkd_task_rkd_distance_rkd_angle_2p0':   '#0F4D92',
}
# Vivid ColorBrewer Set1 — used for individual run lines within every panel
RUN_COLORS = ['#E41A1C', '#FF7F00', '#4DAF4A', '#984EA3', '#00BCD4']

_NEUTRAL_REF = '#BBBBBB'
_COMMON_FPR  = np.linspace(0, 1, 400)
_TICKS       = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
_TICK_LABELS = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']

# ── Paths ──────────────────────────────────────────────────────────────────
DEFAULT_RUNS_ROOT = (
  Path(__file__).resolve().parents[1].parent
  / 'PathoML-runs' / 'distillation-prev-teacher'
)
DEFAULT_OUT = Path(__file__).resolve().parent / 'out' / 'distillation_roc'

CONDITIONS = [
  ('task_only_baseline_task',               'Baseline'),
  ('hidden_features_matching_task_hidden',  'Hidden Matching'),
  ('standard_kd_task_soft_label_t4p0',      'Standard KD'),
  ('decoupled_kd_task_dkd_t4p0_a1p0_b4p0', 'Decoupled KD'),
  ('rkd_task_rkd_distance_rkd_angle_2p0',   'Relational KD'),
]
PANEL_LABELS = ['a', 'b', 'c', 'd', 'e']


# ── Data loading ───────────────────────────────────────────────────────────

def load_condition(runs_root: Path, condition: str):
  """Return list of (fpr_interp, tpr_interp, auc) per run, or None."""
  cond_dir = runs_root / condition
  if not cond_dir.exists():
    return None
  csvs = sorted(cond_dir.glob('run_*/cv_predictions.csv'))
  if not csvs:
    return None

  runs = []
  for p in csvs:
    df = pd.read_csv(p)
    pat = (
      df.groupby('patient_id')
      .agg(prob=('patient_prob', 'mean'), label=('patient_label', 'first'))
      .reset_index()
    )
    fpr, tpr, _ = roc_curve(pat['label'], pat['prob'])
    tpr_interp = np.interp(_COMMON_FPR, fpr, tpr)
    # Gaussian smoothing for visual clarity (AUC computed from raw data)
    tpr_smooth = gaussian_filter1d(tpr_interp, sigma=4)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)
    tpr_smooth[0], tpr_smooth[-1] = 0.0, 1.0
    auc = roc_auc_score(pat['label'], pat['prob'])
    runs.append((tpr_smooth, auc))
  return runs


# ── Panel rendering ────────────────────────────────────────────────────────

def plot_roc_panel(
  ax: plt.Axes,
  runs: list,           # list of (tpr_interp, auc)
  title: str,
  panel_label: str,
  method_color: str,    # semantic mean-line color
  show_xlabel: bool,
  show_ylabel: bool,
) -> None:
  # Reference diagonal
  ax.plot([0, 1], [0, 1], color=_NEUTRAL_REF, lw=0.6, ls='--', zorder=1)

  # Individual run curves — vivid colors, thin
  tpr_stack = []
  for i, (tpr_interp, _) in enumerate(runs):
    ax.plot(_COMMON_FPR, tpr_interp,
            color=RUN_COLORS[i % len(RUN_COLORS)],
            lw=0.65, alpha=0.70, zorder=2)
    tpr_stack.append(tpr_interp)

  # Mean curve — method's semantic color, bold
  mean_tpr = np.mean(tpr_stack, axis=0)
  ax.plot(_COMMON_FPR, mean_tpr, color=method_color, lw=1.6, zorder=3)

  # AUC stats
  aucs = [a for _, a in runs]
  auc_mean, auc_std = np.mean(aucs), np.std(aucs)

  # Box frame
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.6)

  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_xticks(_TICKS)
  ax.set_yticks(_TICKS)

  if show_ylabel:
    ax.set_yticklabels(_TICK_LABELS, fontsize=5.5)
    ax.set_ylabel('True-positive rate', fontsize=6.5, labelpad=3)
  else:
    ax.set_yticklabels([])
    ax.set_ylabel('')

  # All panels show x tick numbers; only bottom row shows axis label text
  ax.set_xticklabels(_TICK_LABELS, fontsize=5.5)
  if show_xlabel:
    ax.set_xlabel('False-positive rate', fontsize=6.5, labelpad=3)
  else:
    ax.set_xlabel('')

  # AUC in title (two lines): method name + AUC ± std
  ax.set_title(
    f'{title}\n(AUC = {auc_mean:.4f} ± {auc_std:.4f})',
    fontsize=6.5, pad=4, color='black',
    linespacing=1.4,
  )

  # Nature-style panel label
  ax.text(
    -0.08, 1.04, panel_label,
    transform=ax.transAxes,
    fontsize=8, fontweight='bold', ha='left', va='bottom', color='black',
  )


# ── Main ───────────────────────────────────────────────────────────────────

def main(runs_root: Path, out: Path) -> None:
  mm = 1 / 25.4
  fig = plt.figure(figsize=(183 * mm, 120 * mm))

  from matplotlib.gridspec import GridSpec

  # 6-column grid: top panels span cols [0:2],[2:4],[4:6];
  # bottom panels span cols [1:3],[3:5] — centred on the a/b and b/c gaps
  gs = GridSpec(
    2, 6, figure=fig,
    hspace=0.70, wspace=0.35,
    top=0.91, bottom=0.11, left=0.09, right=0.97,
  )
  top_axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),
  ]
  bot_axes = [
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
  ]
  axes = top_axes + bot_axes

  # show_ylabel: leftmost panel in each row; show_xlabel: bottom row only
  ylabel_panels = {'a', 'd'}
  xlabel_panels = {'d', 'e'}

  missing = []
  for (condition, label), ax, plabel in zip(CONDITIONS, axes, PANEL_LABELS):
    color = METHOD_COLORS[condition]
    runs = load_condition(runs_root, condition)

    if runs is None:
      ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
              ha='center', va='center', fontsize=6)
      ax.set_title(label, fontsize=7, pad=3)
      missing.append(condition)
      continue

    plot_roc_panel(
      ax, runs, label, plabel, color,
      show_xlabel=(plabel in xlabel_panels),
      show_ylabel=(plabel in ylabel_panels),
    )
    aucs = [a for _, a in runs]
    print(f'  {plabel}. {label}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')

  fig.suptitle('Case-level ROC Curves of Distillation Methods',
               fontsize=8, fontweight='bold', y=0.98)

  if missing:
    print(f'\nWarning: missing data for {missing}', file=sys.stderr)

  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(str(out) + '.svg', bbox_inches='tight')
  fig.savefig(str(out) + '.pdf', bbox_inches='tight')
  fig.savefig(str(out) + '.png', dpi=600, bbox_inches='tight')
  plt.close(fig)
  print(f'\nSVG → {out}.svg')
  print(f'PDF → {out}.pdf')
  print(f'PNG → {out}.png')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS_ROOT)
  parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()
  main(args.runs_root, args.out)

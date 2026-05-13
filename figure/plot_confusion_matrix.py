"""
Case-level confusion matrix comparison: Baseline vs. cand048.

Usage:
  python -m figure.plot_confusion_matrix [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ── Mandatory font + SVG rules ─────────────────────────────────────────────
_CJK_FONT = Path.home() / '.local/share/fonts/NotoSansSC-Regular.otf'
if _CJK_FONT.exists():
    fm.fontManager.addfont(str(_CJK_FONT))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'font.size': 7,
    'axes.linewidth': 0.6,
    'legend.frameon': False,
})

# ── Paths ──────────────────────────────────────────────────────────────────
_RUNS_ROOT = Path(__file__).resolve().parents[1].parent / 'PathoML-runs'

BASELINE_DIR = _RUNS_ROOT / 'distillation-prev-teacher' / 'task_only_baseline_task'
CAND048_DIR  = (
    _RUNS_ROOT
    / 'distillation-autosearch/distill-f1-20260504/outputs'
    / 'cand048_lstd005_uang_tga_rkd'
    / 'lstd005_uang_tga_rkd_task_wrkd_uang_tga_lstd005'
)

DEFAULT_OUT = Path(__file__).resolve().parent / 'out' / 'confusion_matrix'

# ── Custom blue colormap: white → deep blue ────────────────────────────────
_BLUE_CMAP = LinearSegmentedColormap.from_list(
    'nature_blue', ['#FFFFFF', '#D6E4F7', '#6FA8D9', '#0F4D92']
)

# ── Cell labels ────────────────────────────────────────────────────────────
_CELL_NAMES = [['TN', 'FP'], ['FN', 'TP']]


# ── Data loading ───────────────────────────────────────────────────────────

def load_cm(cond_dir: Path):
    csvs = sorted(cond_dir.glob('run_*/cv_predictions.csv'))
    df = pd.concat([pd.read_csv(p) for p in csvs])
    pat = (
        df.groupby('patient_id')
        .agg(prob=('patient_prob', 'mean'), label=('patient_label', 'first'))
        .reset_index()
    )
    pat['pred'] = (pat['prob'] >= 0.5).astype(int)
    cm = confusion_matrix(pat['label'], pat['pred'])
    return cm, len(pat)


# ── Panel rendering ────────────────────────────────────────────────────────

def plot_cm_panel(ax, cm, title, panel_label, vmax=1.0):
    # Row-normalised matrix for color (each true-class row sums to 1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap=_BLUE_CMAP, vmin=0, vmax=vmax, aspect='equal')

    # Cell annotations: name + count + percentage
    for r in range(2):
        for c in range(2):
            val_norm = cm_norm[r, c]
            count    = cm[r, c]
            name     = _CELL_NAMES[r][c]
            luminance = 0.299 * 0.06 + 0.587 * 0.30 + 0.114 * 0.57  # rough mid
            text_color = 'white' if val_norm > 0.55 else '#1A1A1A'

            ax.text(c, r - 0.12, name,
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=text_color)
            ax.text(c, r + 0.18, f'{count}  ({val_norm:.1%})',
                    ha='center', va='center', fontsize=6,
                    color=text_color)

    # Axes styling
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'], fontsize=6.5)
    ax.set_yticklabels(['Negative', 'Positive'], fontsize=6.5,
                       rotation=90, va='center')
    ax.set_xlabel('Predicted label', fontsize=7, labelpad=4)
    ax.set_ylabel('True label', fontsize=7, labelpad=4)
    ax.set_title(title, fontsize=7.5, fontweight='bold', pad=6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(length=0)

    # Panel label
    ax.text(-0.18, 1.06, panel_label,
            transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')

    return im


# ── Main ───────────────────────────────────────────────────────────────────

def main(out: Path) -> None:
    cm_base, n_base = load_cm(BASELINE_DIR)
    cm_cand, n_cand = load_cm(CAND048_DIR)

    tn, fp, fn, tp = cm_base.ravel()
    print(f'Baseline  (n={n_base}): TN={tn}  FP={fp}  FN={fn}  TP={tp}'
          f'  Acc={(tn+tp)/n_base:.3f}')
    tn, fp, fn, tp = cm_cand.ravel()
    print(f'cand048   (n={n_cand}): TN={tn}  FP={fp}  FN={fn}  TP={tp}'
          f'  Acc={(tn+tp)/n_cand:.3f}')

    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1, 2, figsize=(130 * mm, 72 * mm),
        gridspec_kw={'wspace': 0.52},
    )
    fig.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.18)

    # Shared vmax = 1.0 so both panels use identical color scale
    im = plot_cm_panel(axes[0], cm_base, 'Baseline', 'a')
    plot_cm_panel(axes[1], cm_cand, 'cand048 (lstd·TGA·RKD)', 'b')

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.72, pad=0.03,
                        fraction=0.025)
    cbar.set_label('Proportion per true class', fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_visible(False)

    fig.suptitle('Case-level Confusion Matrix Comparison',
                 fontsize=8.5, fontweight='bold', y=0.97)

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
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out)

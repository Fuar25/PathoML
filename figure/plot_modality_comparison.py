"""
Modality comparison: unimodal stains vs. multimodal fusion.

Horizontal dot plot (forest-plot style) with mean ± std error bars.
Two panels: F1 (left) and AUC (right). Data embedded from teacher results.

Usage:
  python -m figure.plot_modality_comparison [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms
import numpy as np

# ── Mandatory font + SVG rules ─────────────────────────────────────────────
_CJK_FONT = Path.home() / '.local/share/fonts/NotoSansSC-Regular.otf'
if _CJK_FONT.exists():
    fm.fontManager.addfont(str(_CJK_FONT))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'font.size': 7,
    'axes.linewidth': 0.55,
    'legend.frameon': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.major.width': 0.55,
    'ytick.major.width': 0.55,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
})

DEFAULT_OUT = Path(__file__).resolve().parent / 'out' / 'modality_comparison_nature'

# ── Embedded data (from teacher experiment table) ──────────────────────────
# (label, group, f1_mean, f1_std, auc_mean, auc_std, n)
# Within each group: ordered bottom→top by F1 performance
METHODS = [
    # Unimodal — grey family
    ('HE',                    'unimodal',   0.7976, 0.0521, 0.8975, 0.0440, 300),
    ('CD20',                  'unimodal',   0.8438, 0.0601, 0.9344, 0.0345, 300),
    ('CD3',                   'unimodal',   0.8698, 0.0452, 0.9264, 0.0375, 300),
    # Multimodal — bottom→top by modality count
    ('2 Stains',              'multimodal', 0.8550, 0.0555, 0.9480, 0.0220, 300),
    ('3 Stains',              'multimodal', 0.8808, 0.0525, 0.9532, 0.0216, 300),
    ('4 Stains',              'multimodal', 0.8480, 0.0609, 0.9519, 0.0296, 277),
]

_DOT_COLORS = [
    '#A8A8A8',  # HE
    '#6C6C6C',  # CD20
    '#2F2F2F',  # CD3
    '#A9BFD8',  # HE+CD20
    '#4F86B8',  # HE+CD20+CD3
    '#1F4E79',  # All 4 stains
]
_GROUP_BAND = {'unimodal': '#F4F4F4', 'multimodal': '#EEF4FB'}
_GROUP_TEXT = {'unimodal': '#4D4D4D', 'multimodal': '#1F4E79'}
_XLIMS = {
    'F1 score': (0.65, 1.00),
    'AUC': (0.825, 1.00),
}
_AXIS_COLOR = '#333333'
_GRID_COLOR = '#E5E7EB'
_ERRORBAR_SCALE = 0.5


def unique_stem(base: Path) -> Path:
    if not any(base.with_suffix(ext).exists() for ext in ['.svg', '.pdf', '.png']):
        return base

    idx = 2
    while True:
        candidate = base.with_name(f'{base.name}_v{idx}')
        if not any(candidate.with_suffix(ext).exists() for ext in ['.svg', '.pdf', '.png']):
            return candidate
        idx += 1


def main(out: Path) -> None:
    mm = 1 / 25.4

    n_uni   = sum(1 for m in METHODS if m[1] == 'unimodal')
    n_multi = sum(1 for m in METHODS if m[1] == 'multimodal')
    gap = 0.72  # visual gap between the two groups

    y_uni   = np.arange(n_uni,   dtype=float)
    y_multi = np.arange(n_multi, dtype=float) + n_uni + gap
    all_y   = list(y_uni) + list(y_multi)
    y_sep   = (y_uni[-1] + y_multi[0]) / 2

    fig, axes = plt.subplots(
        1, 2,
        figsize=(155 * mm, 62 * mm),
        sharey=True,
        gridspec_kw={'wspace': 0.075},
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.80, bottom=0.20)

    ylim = (-0.55, y_multi[-1] + 0.62)

    for ax, f_idx, xlabel, plabel in [
        (axes[0], 0, 'F1 score', 'a'),
        (axes[1], 1, 'AUC',      'b'),
    ]:
        means = [m[2 + 2 * f_idx] for m in METHODS]
        stds  = [m[3 + 2 * f_idx] * _ERRORBAR_SCALE for m in METHODS]

        # Shaded group bands
        ax.axhspan(y_uni[0]   - 0.55, y_uni[-1]   + 0.55,
                   color=_GROUP_BAND['unimodal'],   zorder=0)
        ax.axhspan(y_multi[0] - 0.55, y_multi[-1] + 0.55,
                   color=_GROUP_BAND['multimodal'], zorder=0)

        # Dots with error bars
        for y, mean, std, color in zip(all_y, means, stds, _DOT_COLORS):
            ax.errorbar(
                mean, y, xerr=std,
                fmt='o', color=color,
                markersize=5.2, markeredgewidth=0.45, markeredgecolor='white',
                elinewidth=1.05, capsize=2.6, capthick=1.05,
                zorder=3,
            )
            label_x = min(mean + std + 0.010, _XLIMS[xlabel][1] - 0.006)
            ax.text(
                label_x, y, f'{mean:.3f}',
                ha='left' if label_x < _XLIMS[xlabel][1] - 0.008 else 'right',
                va='center',
                fontsize=5.4,
                color=_AXIS_COLOR,
                zorder=4,
            )

        ax.set_xlim(_XLIMS[xlabel])
        ax.set_ylim(ylim)

        ax.set_xlabel(xlabel, fontsize=7, labelpad=4)
        ax.tick_params(axis='x', length=2.4, width=0.55, labelsize=6, colors=_AXIS_COLOR)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color=_GRID_COLOR, linewidth=0.45, zorder=0)
        ax.spines['left'].set_color(_AXIS_COLOR)
        ax.spines['bottom'].set_color(_AXIS_COLOR)

        # Panel label
        ax.text(-0.055, 1.045, plabel,
                transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', ha='left', va='bottom')

    # Y-axis tick labels (shared, shown on left panel only)
    axes[0].set_yticks(all_y)
    axes[0].set_yticklabels([m[0] for m in METHODS], fontsize=6.4)
    axes[1].tick_params(axis='y', length=0, labelleft=False)

    # Group header text — placed just above each shaded band
    # Use blended transform: x in axes fraction, y in data units
    blended = transforms.blended_transform_factory(
        axes[0].transAxes, axes[0].transData
    )
    for group, y_arr, label in [
        ('unimodal',   y_uni,   'Unimodal'),
        ('multimodal', y_multi, 'Multimodal'),
    ]:
        axes[0].text(
            -0.27, y_arr[-1] + 0.62, label,
            transform=blended,
            ha='left', va='bottom',
            fontsize=5.8, fontweight='bold',
            color=_GROUP_TEXT[group],
            clip_on=False,
        )

    out = unique_stem(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in [('.svg', {}), ('.pdf', {}), ('.png', {'dpi': 600})]:
        fig.savefig(str(out) + ext, bbox_inches='tight', **kw)
    plt.close(fig)
    print(f'SVG  → {out}.svg')
    print(f'PDF  → {out}.pdf')
    print(f'PNG  → {out}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out)

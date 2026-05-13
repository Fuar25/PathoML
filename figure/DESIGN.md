# Figure Design Principles

Conventions used in this repository's publication figures.
Target venue: Nature-family journals (NMI / Nature Medicine style).

---

## 1. Mandatory rcParams

These three lines must appear before any figure is created:

```python
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'   # keeps text editable in Illustrator/Inkscape
plt.rcParams['pdf.fonttype'] = 42       # embeds TrueType, not outlines
```

For CJK text, load the local Noto Sans SC font before creating figures:

```python
from pathlib import Path
import matplotlib.font_manager as fm
cjk = Path.home() / '.local/share/fonts/NotoSansSC-Regular.otf'
if cjk.exists():
    fm.fontManager.addfont(str(cjk))
```

---

## 2. Export policy

Always save SVG as primary output (lossless, editable text):

```python
fig.savefig(f'{out}.svg', bbox_inches='tight')
fig.savefig(f'{out}.pdf', bbox_inches='tight')
fig.savefig(f'{out}.png', dpi=600, bbox_inches='tight')
plt.close(fig)
```

Outputs go under `figure/out/` to separate scripts from artifacts.

---

## 3. Figure size

Two-column Nature width: **183 mm × 120 mm** for a 3+2 panel grid.

```python
mm = 1 / 25.4
fig = plt.figure(figsize=(183 * mm, 120 * mm))
```

---

## 4. Global style

```python
plt.rcParams.update({
  'font.size':         7,      # journal-final dense composite
  'axes.linewidth':    0.6,
  'xtick.major.width': 0.6,
  'ytick.major.width': 0.6,
  'xtick.major.size':  2.5,
  'ytick.major.size':  2.5,
  'axes.spines.right': True,   # full box frame for ROC / comparison panels
  'axes.spines.top':   True,
  'legend.frameon':    False,
})
```

- Use a **full box frame** (all 4 spines) for quantitative comparison panels.
- Remove top/right spines only for bar charts and trend plots where the frame adds no information.

---

## 5. Color policy

**Semantic hierarchy** — one neutral family for controls, one blue family for methods:

```python
METHOD_COLORS = {
  'baseline':      '#767676',  # neutral grey  — control / no distillation
  'hidden_match':  '#B4C0E4',  # light blue
  'standard_kd':   '#7884B4',  # mid blue
  'decoupled_kd':  '#3775BA',  # blue
  'relational_kd': '#0F4D92',  # deep blue — hero / best method
}
```

**Vivid run palette** (ColorBrewer Set1) — for individual repeated runs within a panel:

```python
RUN_COLORS = ['#E41A1C', '#FF7F00', '#4DAF4A', '#984EA3', '#00BCD4']
```

Rules:
- Baseline / control: grey (`#767676`), visually quieter than method lines.
- Best method: `#0F4D92` (deep blue), the visually dominant line.
- Individual runs: vivid distinct hues, thin (`lw=0.65`, `alpha=0.70`).
- Mean/summary line: method's semantic color, bold (`lw=1.6`), drawn on top.
- Reserve green/red for directional cues (gains, drops), not for primary method lines.

---

## 6. ROC curve rendering

```python
_COMMON_FPR = np.linspace(0, 1, 400)

# Per run: interpolate then smooth (visual only; AUC computed from raw data)
tpr_interp = np.interp(_COMMON_FPR, fpr, tpr)
tpr_smooth = gaussian_filter1d(tpr_interp, sigma=4)
tpr_smooth = np.clip(tpr_smooth, 0, 1)
tpr_smooth[0], tpr_smooth[-1] = 0.0, 1.0
```

- Interpolate all runs onto a shared 400-point FPR grid before smoothing.
- `sigma=4` removes the staircase artifact from small cohorts without distorting shape.
- AUC is always computed from the original (unsmoothed) predictions.
- Reference diagonal: `color='#BBBBBB'`, `lw=0.6`, `ls='--'`.

---

## 7. Tick and axis label policy

- **Ticks**: `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` on both axes for ROC panels.
- **Tick labels**: shown on all panels (including non-edge panels).
- **Axis label text**: shown only on the outermost panels of each row/column.
  - Y label ("True-positive rate"): leftmost column only.
  - X label ("False-positive rate"): bottom row only.
- Axis labels in **English** for international venues.

---

## 8. Panel labels and titles

- Panel labels (**a**, **b**, …): bold, 8 pt, positioned at `(-0.08, 1.04)` in axes coordinates.
- Titles: two-line — method name on line 1, `(AUC = mean ± std)` on line 2, 6.5 pt.

```python
ax.set_title(f'{title}\n(AUC = {auc_mean:.4f} ± {auc_std:.4f})',
             fontsize=6.5, pad=4, linespacing=1.4)
```

---

## 9. Multi-panel layout

**3 + 2 staggered grid** using a 6-column `GridSpec`:

```
Row 0:  [  a  ][  b  ][  c  ]       → cols 0:2, 2:4, 4:6
Row 1:      [  d  ][  e  ]           → cols 1:3, 3:5
```

Panel d is centred under the a/b gap; panel e is centred under the b/c gap.
This avoids the "dashboard" look of equal-sized grids and follows Nature's
asymmetric / staggered composition conventions.

```python
gs = GridSpec(2, 6, figure=fig, hspace=0.70, wspace=0.35,
              top=0.91, bottom=0.11, left=0.09, right=0.97)
top_axes = [fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6])]
bot_axes = [fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5])]
```

---

## 10. Uncertainty visualization

For repeated experiments (multiple runs), prefer **individual run lines** over
confidence bands when the run count is small (≤ 5):

- Individual runs: thin vivid lines expose run-to-run variance directly.
- Mean line: semantic color, bold, drawn last (highest z-order).
- Report `AUC = mean ± std` in the panel title.

Confidence bands (`fill_between`) are suitable when runs ≥ 10 or when individual
lines would create visual clutter.

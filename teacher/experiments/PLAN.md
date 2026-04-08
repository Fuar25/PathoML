# teacher/experiments/PLAN

## Current Goal
Find the strongest teacher candidate for downstream distillation.
Current best: `run_concat_HE_CD20_CD3_mlp` (AUC 0.9531, hidden_dim=128)

## Results Summary

| Condition | Stains | Model | AUC (run-level) | Date |
|-----------|--------|-------|-----------------|------|
| unimodal_HE_abmil | HE | ABMIL | 0.9100 +/- 0.0080 | 03-31 15:14 |
| unimodal_CD20_abmil | CD20 | ABMIL | 0.9294 +/- 0.0041 | 03-31 15:23 |
| unimodal_HE_linear | HE | LinearProbe | 0.8975 +/- 0.0097 | 03-30 18:32 |
| unimodal_CD20_linear | CD20 | LinearProbe | 0.9344 +/- 0.0112 | 03-30 18:51 |
| concat_HE_CD20_mlp | HE+CD20 | MLP | 0.9480 +/- 0.0020 | 03-30 19:13 |
| concat_HE_CD20_CD3_mlp (dim=256) | HE+CD20+CD3 | MLP | 0.9512 +/- 0.0093 | 03-30 19:46 |
| **concat_HE_CD20_CD3_mlp (dim=128)** | HE+CD20+CD3 | MLP | **0.9531 +/- 0.0022** | **03-31 11:03** |

## Key Findings
- Three-stain concatenation outperforms the unimodal baselines.
- CD20-only teacher experiments outperform HE-only teacher experiments.
- `hidden_dim=128` is slightly better and more stable than `hidden_dim=256`.
- `patience=30` and `batch_size=16` are the current defaults.

## Next Steps
- [ ] Try four-stain concatenation: `HE + CD20 + CD3 + Ki67`
- [ ] Try attention-based fusion instead of plain concatenation
- [ ] Compare different foundation-model feature sources

## Decisions
- 2026-03-31: `concat_HE_CD20_CD3_mlp` with `hidden_dim=128` is the current default teacher
- 2026-03-31: `patience=30` and `batch_size=16` are the current teacher defaults

# distillation/experiments/PLAN

## Current Goal
- Report distillation baselines across two fixed teacher platforms.
- Keep the previous slide-level teacher platform as historical reference.
- Use the current `c094` RegCoord patch teacher platform as the active baseline block.
- Drop the experimental TGA / RKD+TGA exploration from the final narrative.

## Previous Teacher Platform
### Platform
- Teacher: `run_concat_HE_CD20_CD3_mlp_bs32`
- Teacher performance (fold-level): AUC `0.9532 +/- 0.0216`, F1Score `0.8808 +/- 0.0525` (teacher log 04-11)
- Student: `StudentBasicABMIL(hidden_dim=128, attention_dim=128)`
- Protocol: fold-level AUC and fold-level F1Score, `N_RUNS=5`, `K_FOLDS=5`
- Scope: fix teacher and student; vary distillation losses only

### Results
| Method | Loss Design | Fold-level AUC | Fold-level F1Score | Date |
|--------|-------------|----------------|--------------------|------|
| ABMIL baseline | L_task | 0.9110 +/- 0.0367 | 0.8343 +/- 0.0339 | 04-11 |
| Hidden matching | L_task + L_hidden | 0.9168 +/- 0.0351 | 0.8311 +/- 0.0426 | 04-11 |
| SPKD | L_task + L_similarity_preserving | 0.9104 +/- 0.0360 | 0.8331 +/- 0.0429 | 04-12 |
| StandardKD | L_task + L_soft_label(T=4) | 0.9175 +/- 0.0346 | 0.8453 +/- 0.0464 | 04-11 |
| DKD (strict) | L_task + L_dkd(T=4, alpha=1, beta=4) | 0.9175 +/- 0.0346 | 0.8453 +/- 0.0464 | 04-12 |
| RKD | L_task + L_rkd_distance + 2*L_rkd_angle | 0.9194 +/- 0.0331 | 0.8556 +/- 0.0318 | 04-11 |

### Read
- RKD is the best formal baseline on the previous platform by both AUC and F1Score.
- StandardKD and DKD (strict) match at the logged canonical parameters.
- Hidden matching gives a small AUC gain but does not improve F1Score over ABMIL.
- SPKD is weak on this platform and stays below ABMIL / StandardKD / RKD.
- Historical TGA and RKD+TGA tracks are excluded from the final narrative.

## Current Teacher Platform
### Platform
- Teacher: `run_regcoord_origfeat_HE_CD20_CD3_patch_c094_polycoord_stain_bias_coord_gate_scale020_thresh05125_mil`
- Teacher performance (fold-level): AUC `0.9418 +/- 0.0358`, F1Score `0.8922 +/- 0.0404` (teacher autosearch 3-run screen)
- Teacher artifact: `../PathoML-runs/teacher-winners/manifest.json`
- Teacher sample set: RegCoord patch HE/CD20/CD3, 264 samples, `N_RUNS=3`, `K_FOLDS=5`
- Student: `StudentBasicABMIL(hidden_dim=128, attention_dim=128)`
- Protocol: fold-level AUC and fold-level F1Score
- Scope: fix teacher and student; compare bare student, StandardKD, and RKD

### Results
| Method | Loss Design | Fold-level AUC | Fold-level F1Score | Date |
|--------|-------------|----------------|--------------------|------|
| ABMIL baseline | L_task | 0.8892 +/- 0.0517 | 0.8003 +/- 0.0525 | 05-07 |
| StandardKD | L_task + L_soft_label(T=4) | 0.8903 +/- 0.0525 | 0.8059 +/- 0.0457 | 05-07 |
| RKD | L_task + L_rkd_distance + 2*L_rkd_angle | 0.9078 +/- 0.0597 | 0.8461 +/- 0.0603 | 05-07 |

### Read
- RKD is the strongest current-platform baseline among the rerun set.
- StandardKD gives only a small gain over the bare student on the current platform.
- RKD improves over the bare student by `+0.0186` AUC and `+0.0458` F1Score.
- RKD improves over StandardKD by `+0.0175` AUC and `+0.0402` F1Score.
- Current-platform results are the active comparison block; previous-platform results are reference only.

## Next Steps
- [x] Recover previous platform official baseline summary from git history
- [x] Remove experimental TGA / RKD+TGA tracks from the formal `PLAN.md` narrative
- [x] Rerun current-platform ABMIL task-only baseline
- [x] Rerun current-platform StandardKD baseline
- [x] Rerun current-platform RKD baseline
- [ ] Decide whether the current-platform baseline block needs confirmation seeds before final reporting

## Active Decisions
- 2026-04-09: Freeze the previous multi-stain teacher + `StudentBasicABMIL` platform for controlled loss comparisons.
- 2026-04-11: Keep RKD as the previous-platform formal baseline winner.
- 2026-04-12: Keep SPKD and DKD (strict) as previous-platform formal baseline rows.
- 2026-04-26: Do not move the TGA / RKD+TGA exploration into the final baseline narrative.
- 2026-05-05: Use `c094` RegCoord patch PolyCoord stain-bias coordinate-gate MIL as the current fixed teacher winner.
- 2026-05-07: Split `PLAN.md` into previous and current teacher platform sections with parallel structure.
- 2026-05-07: Use bare student, StandardKD, and RKD as the current-platform baseline set.

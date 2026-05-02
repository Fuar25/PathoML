# distillation/experiments/PLAN

## Current Goal
- Fix platform: three-stain teacher + HE-only `StudentBasicABMIL`.
- Close the ABMIL-line loss search and explain the RKD gain mechanism.

## Active Platform
- Teacher: `run_concat_HE_CD20_CD3_mlp_bs32`
- Teacher performance (fold-level): AUC `0.9532 +/- 0.0216`, F1Score `0.8808 +/- 0.0525` (teacher log 04-11)
- Student: `StudentBasicABMIL(hidden_dim=128, attention_dim=128)`
- Protocol: fold-level AUC and fold-level F1Score, `N_RUNS=5`, `K_FOLDS=5`
- Scope: fix teacher and student; vary distillation losses only

## Active Results
| Method | Loss Design | Fold-level AUC | Fold-level F1Score | Date |
|--------|-------------|----------------|--------------------|------|
| ABMIL baseline | L_task | 0.9110 +/- 0.0367 | 0.8343 +/- 0.0339 | 04-11 |
| Hidden matching | L_task + L_hidden | 0.9168 +/- 0.0351 | 0.8311 +/- 0.0426 | 04-11 |
| SPKD | L_task + L_similarity_preserving | 0.9104 +/- 0.0360 | 0.8331 +/- 0.0429 | 04-12 |
| StandardKD | L_task + L_soft_label(T=4) | 0.9175 +/- 0.0346 | 0.8453 +/- 0.0464 | 04-11 |
| DKD (strict) | L_task + L_dkd(T=4, alpha=1, beta=4) | 0.9175 +/- 0.0346 | 0.8453 +/- 0.0464 | 04-12 |
| RKD | L_task + L_rkd_distance + 2*L_rkd_angle | 0.9194 +/- 0.0331 | 0.8556 +/- 0.0318 | 04-11 |

## TGA Track (Experimental)
| TGA Condition | Loss Design | Fold-level AUC | Fold-level F1Score | Status |
|--------------|-------------|----------------|--------------------|--------|
| Old TGA + no-detach | L_task + L_attn_cosine | 0.9175 +/- 0.0313 | 0.8426 +/- 0.0375 | done (04-11) |
| Old TGA + detach | L_task + L_attn_cosine_detach | 0.9162 +/- 0.0335 | 0.8406 +/- 0.0345 | done (04-11) |
| Soft TGA + detach | L_task + L_attn_soft_distribution_detach | 0.9112 +/- 0.0385 | 0.8413 +/- 0.0362 | done (04-11) |
| Soft TGA + no-detach | L_task + L_attn_soft_distribution_no_detach | 0.8973 +/- 0.0615 | 0.8122 +/- 0.0616 | done (04-11) |
| Batch-Contrastive TGA + detach | L_task + L_attn_batch_contrastive_detach | 0.9167 +/- 0.0373 | 0.8369 +/- 0.0352 | done (04-11) |
| Batch-Contrastive TGA + no-detach | L_task + L_attn_batch_contrastive_no_detach | 0.9072 +/- 0.0398 | 0.8329 +/- 0.0433 | done (04-11) |
| Confidence-Gated TGA + no-detach | L_task + L_attn_cosine_confidence_gated_no_detach | 0.9119 +/- 0.0401 | 0.8387 +/- 0.0420 | done (04-24) |
| Normalized Confidence-Gated TGA + no-detach | L_task + L_attn_cosine_confidence_gated_normalized_no_detach | 0.9114 +/- 0.0406 | 0.8400 +/- 0.0444 | done (04-26) |
| Class-Aware TGA + no-detach | L_task + L_attn_class_aware_cosine_no_detach(h=0.5, c=0.5) | 0.9123 +/- 0.0395 | 0.8402 +/- 0.0402 | done (04-26) |
| Class-Aware Rank-Margin TGA + detach | L_task + L_attn_class_aware_rank_margin_detach(h=0.5, c=0.5, r=0.25, m=1) | 0.9132 +/- 0.0380 | 0.8324 +/- 0.0480 | done (04-26) |

## RKD + TGA Track (Experimental)
| RKD + TGA Condition | Loss Design | Fold-level AUC | Fold-level F1Score | Status |
|---------------------|-------------|----------------|--------------------|--------|
| RKD + TGA weight 0.1 | L_task + L_rkd_distance + 2*L_rkd_angle + 0.1*L_attn_cosine_no_detach | 0.9188 +/- 0.0349 | 0.8488 +/- 0.0319 | done (04-26) |
| RKD + TGA weight 0.25 | L_task + L_rkd_distance + 2*L_rkd_angle + 0.25*L_attn_cosine_no_detach | 0.9187 +/- 0.0354 | 0.8563 +/- 0.0325 | done (04-26) |
| RKD + TGA weight 0.5 | L_task + L_rkd_distance + 2*L_rkd_angle + 0.5*L_attn_cosine_no_detach | 0.9202 +/- 0.0323 | 0.8586 +/- 0.0338 | done (04-26) |
| RKD + TGA weight 0.75 | L_task + L_rkd_distance + 2*L_rkd_angle + 0.75*L_attn_cosine_no_detach | 0.9190 +/- 0.0343 | 0.8592 +/- 0.0344 | done (04-26) |
| RKD + TGA weight 1.0 | L_task + L_rkd_distance + 2*L_rkd_angle + L_attn_cosine_no_detach | 0.9187 +/- 0.0346 | 0.8571 +/- 0.0320 | done (04-26) |

## Confirmation Track (Experimental)
| Confirmation Condition | Loss Design | Fold-level AUC | Fold-level F1Score | Status |
|------------------------|-------------|----------------|--------------------|--------|
| RKD confirm student seed 142 | L_task + L_rkd_distance + 2*L_rkd_angle | 0.9162 +/- 0.0320 | 0.8364 +/- 0.0456 | done (04-26) |
| RKD + TGA weight 0.5 confirm student seed 142 | L_task + L_rkd_distance + 2*L_rkd_angle + 0.5*L_attn_cosine_no_detach | 0.9151 +/- 0.0347 | 0.8469 +/- 0.0394 | done (04-26) |

## Current Read
- Keep RKD as the winner to beat on both AUC and F1.
- Treat `Hidden matching` as a weak feature baseline: AUC > ABMIL, F1 < ABMIL, below `StandardKD` and `RKD`.
- Treat `SPKD` as a weak similarity baseline on this platform: both AUC and F1 are slightly below ABMIL baseline, and clearly below `StandardKD`/`RKD`.
- Treat `DKD (strict)` as equivalent to `StandardKD` on this binary setup at canonical params; both are below `RKD`.
- Active table: `baseline`, `Hidden matching`, `SPKD`, `StandardKD`, `DKD (strict)`, `RKD`.
- Compare TGA targets: old logit vs soft distribution, each with detach vs no-detach.
- Treat old cosine-logit `detach` as slightly worse than `no-detach`.
- Treat soft-distribution TGA as negative in both modes.
- For batch-contrastive TGA, `detach` is clearly better than `no-detach`.
- Best batch-contrastive TGA (`detach`) is still below Old TGA no-detach, `StandardKD`, and `RKD`.
- Confidence-Gated TGA underperforms Old TGA no-detach, `StandardKD`, and `RKD`; treat the unnormalized continuous confidence gate as negative.
- Normalized Confidence-Gated TGA recovers only a small F1 delta over unnormalized gating, but AUC remains near baseline and below Old TGA no-detach.
- Class-Aware TGA improves slightly over confidence-gated variants but remains near baseline and below Old TGA no-detach; mixing signed classifier direction into cosine targets is not sufficient.
- Class-Aware Rank-Margin TGA improves AUC slightly over class-aware cosine but hurts F1 below baseline; hard top-vs-bottom attention ordering is negative.
- Test TGA as a weak auxiliary term on top of RKD rather than as the main distillation signal.
- In the experimental RKD + TGA track, weight 0.5 is the current best result on both AUC and F1.
- RKD + TGA shows a weight-sensitive trend: 0.1 underfits the attention auxiliary signal, 0.25 improves F1, and 0.5 improves both AUC and F1 over RKD.
- RKD + TGA at weight 0.75 gives the best F1 so far but gives up AUC versus weight 0.5.
- RKD + TGA at weight 1.0 declines versus 0.5/0.75; the useful range appears near 0.5-0.75.
- RKD vs RKD+TGA paired diagnostics show near-cancelled error flips: weight 0.5 fixes 33 run-patient errors and regresses 32; weight 0.75 also fixes 33 and regresses 32. Treat TGA gains as noise-level unless a stronger diagnostic target emerges.
- Final confirmation with fixed teacher split seeds and new student seeds (`STUDENT_BASE_SEED=142`) does not support moving RKD + TGA into the main baseline table: RKD + TGA weight 0.5 improves F1 over confirmation RKD (`0.8469` vs `0.8364`) but gives up AUC (`0.9151` vs `0.9162`).
- Confirmation paired flips favor RKD + TGA weight 0.5 (`fixed=36`, `regressed=22`, `net_fixed=14`), but the effect remains task-metric-specific and experimental rather than a clean AUC/F1 win.
- Distillation mechanism diagnostic: RKD's clearest contribution is fixed-threshold F1/recall behavior, reducing mean FN versus ABMIL (`19.6 -> 14.6`) while leaving mean FP nearly unchanged (`28.6 -> 28.4`).
- RKD has stronger paired error improvement over ABMIL than StandardKD (`net_fixed=26` vs `15`) and still improves over StandardKD directly (`net_fixed=11`).
- RKD is not a clean calibration win: Brier/ECE are worse than ABMIL and StandardKD; interpret the useful effect as relation-driven operating-point behavior rather than probability calibration.
- Pooled saved-prediction diagnostics do not replace logged fold-level AUC/F1; use them to explain mechanisms and error movement only.

## Next Steps
- [x] Freeze teacher and `StudentBasicABMIL` platform
- [x] Add `hidden_features_matching` feature baseline
- [x] Run `similarity_preserving_kd` similarity baseline
- [x] Run `decoupled_kd` strict logits baseline (single canonical setting)
- [x] Define TGA 2x2 matrix: `{old logit, soft distribution} x {detach, no-detach}`
- [x] Run Old TGA + detach
- [x] Run Soft TGA + detach
- [x] Run Soft TGA + no-detach
- [x] Compare best TGA result against `StandardKD` and `RKD`
- [x] Define Batch-Contrastive TGA target with explicit `detach/no-detach` variants
- [x] Run Batch-Contrastive TGA + detach
- [x] Run Batch-Contrastive TGA + no-detach
- [x] Compare Batch-Contrastive TGA (best) against Old TGA no-detach, `StandardKD`, and `RKD`
- [x] Run Confidence-Gated TGA + no-detach
- [x] Compare Confidence-Gated TGA against Old TGA no-detach, `StandardKD`, and `RKD`
- [x] Run Normalized Confidence-Gated TGA + no-detach
- [x] Compare Normalized Confidence-Gated TGA against Confidence-Gated TGA and Old TGA no-detach
- [x] Run Class-Aware TGA + no-detach
- [x] Compare Class-Aware TGA against Old TGA no-detach, `StandardKD`, and `RKD`
- [x] Run Class-Aware Rank-Margin TGA + detach
- [x] Compare Class-Aware Rank-Margin TGA against Old TGA no-detach, `StandardKD`, and `RKD`
- [x] Run RKD + TGA with TGA weight 0.25
- [x] Compare RKD + TGA against RKD
- [x] Run RKD + TGA with TGA weight 0.1
- [x] Run RKD + TGA with TGA weight 0.5
- [x] Compare RKD + TGA lambda sweep against RKD
- [x] Run RKD + TGA with TGA weight 0.75
- [x] Run RKD + TGA with TGA weight 1.0
- [x] Compare high-weight RKD + TGA sweep against weight 0.5
- [x] Run RKD vs RKD+TGA paired error diagnostic
- [x] Run final confirmation: RKD with confirmation student seeds
- [x] Run final confirmation: RKD + TGA weight 0.5 with confirmation student seeds
- [x] Generate distillation mechanism diagnostic report
- [ ] Close TGA large-exploration track in writing
- [ ] Define next RKD-focused diagnostic or ablation, if needed

## Active Decisions
- 2026-04-09: Freeze multi-stain teacher for this phase
- 2026-04-09: Use `StudentBasicABMIL` as the fixed student for controlled comparisons
- 2026-04-09: Focus this phase on MIL-aware distillation losses
- 2026-04-11: Extend the active table to `baseline`, `Hidden matching`, `StandardKD`, and `RKD`
- 2026-04-11: Keep RKD as the winner to beat
- 2026-04-11: Keep TGA in the experimental track until the 2x2 ablation is resolved
- 2026-04-11: Close the TGA 2x2 ablation; soft-distribution variants do not beat old cosine-logit TGA
- 2026-04-11: Batch-contrastive TGA `detach` > `no-detach`, but best batch-contrastive still does not beat Old TGA no-detach / `StandardKD` / `RKD`
- 2026-04-12: Add `SPKD` baseline; on the active platform it is below ABMIL baseline and clearly below `StandardKD`/`RKD`
- 2026-04-12: Add single-setting `DKD (strict)` baseline; result matches `StandardKD` at canonical params and remains below `RKD`
- 2026-04-24: Add confidence-gated cosine-logit TGA as the next TGA repair experiment
- 2026-04-24: Confidence-Gated TGA no-detach is negative versus Old TGA no-detach, `StandardKD`, and `RKD`
- 2026-04-26: Add normalized confidence-gated TGA to separate confidence weighting from loss-scale reduction
- 2026-04-26: Normalized confidence-gated TGA remains negative versus Old TGA no-detach, `StandardKD`, and `RKD`
- 2026-04-26: Add class-aware TGA using teacher hidden and signed classifier direction
- 2026-04-26: Class-Aware TGA remains negative versus Old TGA no-detach, `StandardKD`, and `RKD`
- 2026-04-26: Add class-aware rank-margin TGA to test bag-level top-vs-bottom attention ordering
- 2026-04-26: Class-Aware Rank-Margin TGA is negative; F1 falls below ABMIL baseline
- 2026-04-26: Add RKD + weak TGA auxiliary experiment
- 2026-04-26: RKD + TGA improves F1 slightly over RKD but gives up a small amount of AUC
- 2026-04-26: Start RKD + TGA lambda sweep at 0.1 and 0.5
- 2026-04-26: RKD + TGA at weight 0.5 becomes the best current method
- 2026-04-26: Start RKD + TGA high-weight sweep at 0.75 and 1.0
- 2026-04-26: RKD + TGA weight 0.75 gives best F1; weight 0.5 remains best balanced AUC/F1 point
- 2026-04-26: Paired error diagnostic does not support further large TGA-loss investment
- 2026-04-26: Final confirmation keeps RKD + TGA experimental; it improves F1 and paired flips but does not beat RKD on AUC.
- 2026-04-26: Distillation mechanism diagnostic supports RKD as the main BasicABMIL distillation baseline; its clearest gain is FN reduction / recall behavior, not calibration.
- 2026-05-02: New distillation runs consume the fixed teacher winner manifest at `../PathoML-runs/teacher-winners/manifest.json`; restore or regenerate the canonical `run_concat_HE_CD20_CD3_mlp_bs32` artifact before launching new runs.

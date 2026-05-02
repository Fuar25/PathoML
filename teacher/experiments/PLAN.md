# teacher/experiments/PLAN

## Current Goal
- Goal: diagnose why registration causes a teacher performance drop before returning to downstream distillation.
- Current best: `concat_HE_CD20_CD3_mlp_bs32` (fold-level AUC 0.9532 +/- 0.0216, fold-level F1 0.8808 +/- 0.0525, hidden_dim=128, batch_size=32)

## Results Summary

| Condition | Stains | Model | Fold-level AUC | Fold-level F1 | Date |
|-----------|--------|-------|----------------|---------------|------|
| unimodal_HE_abmil | HE | ABMIL | 0.9100 +/- 0.0342 | 0.8147 +/- 0.0570 | 03-31 15:14 |
| unimodal_CD20_abmil | CD20 | ABMIL | 0.9294 +/- 0.0291 | 0.8482 +/- 0.0353 | 03-31 15:23 |
| unimodal_HE_linear | HE | LinearProbe | 0.8975 +/- 0.0440 | 0.7976 +/- 0.0521 | 03-30 18:32 |
| unimodal_CD20_linear | CD20 | LinearProbe | 0.9344 +/- 0.0345 | 0.8438 +/- 0.0601 | 03-30 18:51 |
| concat_HE_CD20_mlp | HE+CD20 | MLP | 0.9480 +/- 0.0220 | 0.8550 +/- 0.0555 | 03-30 19:13 |
| concat_HE_CD20_CD3_mlp (dim=256) | HE+CD20+CD3 | MLP | 0.9512 +/- 0.0311 | 0.8499 +/- 0.0587 | 03-30 19:46 |
| concat_HE_CD20_CD3_mlp (dim=128) | HE+CD20+CD3 | MLP | 0.9531 +/- 0.0233 | 0.8641 +/- 0.0435 | 03-31 11:03 |
| **concat_HE_CD20_CD3_mlp_bs32** | HE+CD20+CD3 | MLP | **0.9532 +/- 0.0216** | **0.8808 +/- 0.0525** | **04-09 11:00** |

## Registered Patch Track (Experimental)
| Condition | Stains | Model | Fold-level AUC | Fold-level F1 | Status |
|-----------|--------|-------|----------------|---------------|--------|
| registered_HE_CD20_CD3_patch_concat_abmil | HE+CD20+CD3 | ABMIL | 0.9227 +/- 0.0431 | 0.8408 +/- 0.0672 | done, refreshed data |
| registered_HE_CD20_CD3_patch_fusion_mil | HE+CD20+CD3 | RegisteredPatchFusionMIL | 0.9276 +/- 0.0412 | 0.8573 +/- 0.0528 | done, refreshed data, union coords |

## RegCoord Original-Feature Patch Track (Experimental)
| Condition | Stains | Model | Fold-level AUC | Fold-level F1 | Status |
|-----------|--------|-------|----------------|---------------|--------|
| regcoord_origfeat_HE_CD20_CD3_patch_concat_abmil | HE+CD20+CD3 | ABMIL | 0.9216 +/- 0.0433 | 0.8327 +/- 0.0869 | done, refreshed 264 samples, union coords |
| regcoord_origfeat_HE_CD20_CD3_patch_fusion_mil | HE+CD20+CD3 | RegisteredPatchFusionMIL | 0.9332 +/- 0.0446 | 0.8648 +/- 0.0636 | done, refreshed 264 samples, union coords |

## Matched Slide Track (Experimental)
| Condition | Stains | Model | Fold-level AUC | Fold-level F1 | Status |
|-----------|--------|-------|----------------|---------------|--------|
| matched_HE_CD20_CD3_slide_concat_mlp_bs32 | HE+CD20+CD3 | MLP | 0.9440 +/- 0.0356 | 0.8628 +/- 0.0651 | done |
| registered_HE_CD20_CD3_slide_concat_mlp_bs32 | HE+CD20+CD3 | MLP | 0.9220 +/- 0.0389 | 0.8269 +/- 0.0724 | done |

## Matched Unimodal Slide Track (Experimental)
| Condition | Stain | Model | Fold-level AUC | Fold-level F1 | Status |
|-----------|-------|-------|----------------|---------------|--------|
| matched_HE_slide_linear | HE | LinearProbe | 0.8925 +/- 0.0532 | 0.8022 +/- 0.0631 | done |
| registered_HE_slide_linear | HE | LinearProbe | 0.8914 +/- 0.0385 | 0.7899 +/- 0.0410 | done |
| matched_CD20_slide_linear | CD20 | LinearProbe | 0.9132 +/- 0.0394 | 0.8292 +/- 0.0584 | done |
| registered_CD20_slide_linear | CD20 | LinearProbe | 0.9073 +/- 0.0459 | 0.8064 +/- 0.0687 | done |
| matched_CD3_slide_linear | CD3 | LinearProbe | 0.9113 +/- 0.0460 | 0.8317 +/- 0.0599 | done |
| registered_CD3_slide_linear | CD3 | LinearProbe | 0.8772 +/- 0.0473 | 0.7819 +/- 0.0734 | done |

## Key Findings
- Three-stain concatenation beats unimodal baselines.
- CD20-only experiments beat HE-only experiments.
- `hidden_dim=128` is better and more stable than `hidden_dim=256`.
- `batch_size=32` gives the best fold-level AUC in current three-stain MLP teachers.
- `patience=30` remains the default patience.
- Registered slide concat underperforms matched original slide concat on the same 264 samples.
- Unimodal diagnostics show the largest registered slide drop on CD3.
- Registered patch fusion improves over patch concat but remains below matched original slide concat.
- RegCoord original-feature patch concat matches registered patch concat, while RegCoord original-feature patch fusion improves over registered patch fusion.
- RegCoord original-feature patch fusion remains below matched original slide concat in AUC, but slightly exceeds it in F1.
- Feature-shift QC alone does not explain the registered drop: CD3 has high mean feature cosine (0.9908) despite the largest unimodal performance drop.
- Prediction-shift QC shows the registered drop is concentrated in bad flips: CD3 slide linear has 15 bad flips vs 5 good flips; three-stain slide concat has 13 bad flips vs 2 good flips.

## Next Steps
- [ ] Run four-stain concatenation: `HE + CD20 + CD3 + Ki67`
- [ ] Run attention-based fusion instead of plain concatenation
- [ ] Compare different foundation-model feature sources
- [x] Rerun registered patch-concat ABMIL teacher on refreshed `GigaPath-Patch-Feature-Reg`
- [x] Rerun stain-aware registered patch fusion MIL on refreshed `GigaPath-Patch-Feature-Reg`
- [x] Run matched original slide concat on the registered 264-sample subset
- [x] Run registered slide concat on the same 264-sample subset
- [x] Run matched/registered unimodal slide diagnostics for HE, CD20, and CD3
- [x] Add feature-shift and prediction-shift QC for original-vs-registered slide features
- [x] Rerun RegCoord original-feature patch concat ABMIL through `RegisteredMultimodalPatchDataset`
- [x] Rerun RegCoord original-feature patch fusion MIL through `RegisteredMultimodalPatchDataset`
- [ ] Inspect top CD3 and three-stain bad-flip samples for registration/image-level failure modes
- [ ] Test whether CD3 bad flips are driven by localized diagnostic signal loss rather than global slide embedding drift

## Decisions
- 2026-03-31: `concat_HE_CD20_CD3_mlp` with `hidden_dim=128` became the baseline three-stain teacher.
- 2026-04-09: `concat_HE_CD20_CD3_mlp_bs32` became the canonical condition name for the batch-size-32 teacher.
- 2026-04-09: Keep canonical teacher artifacts reproducible from standard scripts under `teacher/experiments/`; move one-off analysis to `teacher/script/`.
- 2026-04-10: `PLAN.md` tracks teacher results by fold-level AUC and fold-level F1; all rows are backfilled from `teacher/experiments/results_log.txt`.
- 2026-04-29: Start registered HE/CD20/CD3 patch-concat ABMIL teacher as the first spatially aligned multimodal pilot.
- 2026-04-29: Patch-concat ABMIL finished below the slide-level teacher; test stain-aware patch fusion before extracting registered slide features.
- 2026-04-29: Use union registered coordinates plus modality masks for patch fusion; reserve inner join as a clean baseline.
- 2026-04-30: Discard stale registered patch results after refreshing `GigaPath-Patch-Feature-Reg`; rerun patch-concat and patch-fusion teachers.
- 2026-04-30: Compare registered and original slide-level concat on the same 264 registered HE/CD20/CD3 samples.
- 2026-04-30: Run matched unimodal slide diagnostics to isolate stain-specific registration effects.
- 2026-04-30: Start registration-drop investigation with sample-level prediction-aware QC before changing model design.
- 2026-05-01: Keep RegCoord original features as a patch-level `RegisteredMultimodalPatchDataset` diagnostic; do not use direct slide-feature extraction for this comparison.
- 2026-05-02: Keep RegCoord original-feature patch fusion as the stronger patch diagnostic, but do not replace the canonical slide-level teacher.
- 2026-05-02: Move heavy teacher outputs outside the repository under `../PathoML-runs/teacher/`; keep only the current canonical teacher under `../PathoML-runs/teacher-winners/`.
- 2026-05-02: Current canonical teacher remains `run_concat_HE_CD20_CD3_mlp_bs32`; its checkpoint artifact is not present in the cleaned local outputs and must be regenerated or restored before new distillation runs.

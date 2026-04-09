# distillation/experiments/PLAN

## Current Goal
Find the strongest distillation strategy for transferring the three-stain teacher into an HE-only student.
Current best: `L_task + L_rkd_distance + 2*L_rkd_angle` (AUC 0.9322, F1Score 0.8411)

## Teacher
`run_concat_HE_CD20_CD3_mlp` (AUC 0.9531, F1Score 0.8641, hidden_dim=128, stains: HE+CD20+CD3)

## Results Summary

### StandardKD (three-stain teacher, `StudentTransABMIL`)
| Method | Loss Design | AUC | F1Score | Date |
|--------|-------------|-----|---------|------|
| StandardKD | L_task | 0.9226 +/- 0.0099 | 0.8169 +/- 0.0406 | 03-31 12:33 |
| StandardKD | L_task + L_hidden | 0.9222 +/- 0.0053 | 0.8284 +/- 0.0348 | 03-31 13:03 |
| StandardKD | L_task + L_soft_label(T=4) | 0.9268 +/- 0.0068 | 0.8352 +/- 0.0514 | 03-31 13:33 |
| StandardKD | L_task + L_hidden + L_soft_label(T=4) | 0.9276 +/- 0.0095 | 0.8329 +/- 0.0573 | 03-31 12:06 |

### RKD (three-stain teacher, `StudentTransABMIL`)
Note:
- The rows below are all RKD-family ablations.
- The earlier rows use distance-only RKD as the relational term, then test whether adding `L_hidden` or `L_soft_label(T=...)` improves it.
- The final row is the current mainline RKD formulation with both distance and angle terms.

| Method | Loss Design | AUC | F1Score | Date |
|--------|-------------|-----|---------|------|
| RKD | L_task | 0.9231 +/- 0.0086 | 0.8315 +/- 0.0592 | 03-31 16:07 |
| RKD | L_task + L_rkd_distance | 0.9325 +/- 0.0049 | 0.8381 +/- 0.0438 | 03-31 16:33 |
| RKD | L_task + L_hidden | 0.9256 +/- 0.0064 | 0.8225 +/- 0.0668 | 03-31 17:00 |
| RKD | L_task + L_hidden + L_rkd_distance | 0.9282 +/- 0.0072 | 0.8301 +/- 0.0538 | 03-31 17:28 |
| RKD | L_task + L_rkd_distance + L_soft_label(T=4) | 0.9272 +/- 0.0051 | 0.8288 +/- 0.0653 | 03-31 18:19 |
| RKD | L_task + L_rkd_distance + L_soft_label(T=1) | 0.9286 +/- 0.0042 | 0.8299 +/- 0.0528 | 03-31 18:47 |
| RKD | L_task + L_rkd_distance + L_soft_label(T=2) | 0.9273 +/- 0.0045 | 0.8230 +/- 0.0653 | 03-31 19:16 |
| **RKD** | **L_task + L_rkd_distance + 2*L_rkd_angle** | **0.9322 +/- 0.0044** | **0.8411 +/- 0.0507** | **03-31 22:02** |

### TGA / Relational TGA (three-stain teacher, `StudentTransABMIL`)
Key idea:
- TGA now refers to logits-space cosine guidance from teacher hidden states.
- RTGA now refers to logits-space relational discrimination across teachers.
- `L_contrast` trains the encoder directly through patch-to-teacher identification.

Historical note:
- Pre-logits TGA/RTGA runs are treated as erroneous experiments and are not kept in the active comparison set.
- The current `teacher_guided_attention` family refers only to the repaired logits-space implementation.
- Historical TGA `tau` sweeps are also excluded because `tau` did not actually affect the computation in that setup.

| Method | Loss Design | AUC | F1Score | Date |
|--------|-------------|-----|---------|------|
| TGA | L_task + L_attn_cosine | 0.9270 +/- 0.0082 | 0.8395 +/- 0.0660 | 04-07 10:38 |
| RTGA | L_task + L_attn_discrimination | 0.9273 +/- 0.0069 | 0.8399 +/- 0.0677 | 04-07 11:14 |
| RTGA | L_task + L_contrast | 0.9290 +/- 0.0067 | 0.8563 +/- 0.0374 | 04-07 10:12 |
| RTGA | L_task + L_attn_discrimination + L_contrast | 0.9291 +/- 0.0093 | 0.8469 +/- 0.0636 | 04-07 11:32 |
| RTGA | L_task + L_attn_discrimination + 0.1*L_contrast | 0.9270 +/- 0.0112 | 0.8395 +/- 0.0644 | 04-07 11:50 |

## Key Findings
- Relational knowledge distillation remains the strongest distillation direction so far.
- In the older distance-only RKD ablations, plain `L_task + L_rkd_distance` outperformed adding `L_hidden` or `L_soft_label`, and `T=1` was the strongest soft-label setting in that sweep.
- `L_task + L_contrast` currently has the best F1Score among the repaired TGA/RTGA variants, although `L_task + L_rkd_distance + 2*L_rkd_angle` still has the best AUC.
- Hidden matching is more effective than soft-label KD in the current setup.
- `StudentTransABMIL` outperforms the plain ABMIL student.
- Attention supervision is currently worse than the best RKD configuration.
- Pure contrastive supervision is better than the current attention-supervision variants, but still below RKD angle.

## Next Steps
- [ ] Compare `L_task + L_rkd_distance + 2*L_rkd_angle` vs. `L_task + L_contrast` under the AUC/F1Score tradeoff
- [ ] Tune student architecture parameters such as `hidden_dim`, `nhead`, and transformer depth

## Decisions
- 2026-03-31: `StudentTransABMIL` is the default student architecture
- 2026-03-31: RKD angle became the main distillation direction
- 2026-04-07: `L_attn` switched to logit-space MSE

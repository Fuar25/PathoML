# distillation/experiments/PLAN

## Current Goal
Find the strongest distillation strategy for transferring the three-stain teacher into an HE-only student.
Current best: `relational_knowledge_distillation_gamma_distance1_gamma_angle2` (AUC 0.9322, RKD distance+angle)

## Teacher
`run_concat_HE_CD20_CD3_mlp` (AUC 0.9531, hidden_dim=128, stains: HE+CD20+CD3)

## Results Summary

### StandardKD (three-stain teacher, `StudentTransABMIL`)
| Condition | alpha | beta | T | AUC | Date |
|-----------|-------|------|---|-----|------|
| standard_knowledge_distillation_alpha0_beta0_temperature4p0 | 0 | 0 | 4 | 0.9226 +/- 0.0099 | 03-31 12:33 |
| standard_knowledge_distillation_alpha1_beta0_temperature4p0 | 1 | 0 | 4 | 0.9222 +/- 0.0053 | 03-31 13:03 |
| standard_knowledge_distillation_alpha0_beta1_temperature4p0 | 0 | 1 | 4 | 0.9268 +/- 0.0068 | 03-31 13:33 |
| standard_knowledge_distillation_alpha1_beta1_temperature4p0 | 1 | 1 | 4 | 0.9276 +/- 0.0095 | 03-31 12:06 |

### RKD (three-stain teacher, `StudentTransABMIL`)
| Condition | Loss params | AUC | Date |
|-----------|-------------|-----|------|
| legacy_rkd_alpha0_gamma0 | alpha=0, gamma=0 | 0.9231 +/- 0.0086 | 03-31 16:07 |
| legacy_rkd_alpha0_gamma1 | alpha=0, gamma=1 | 0.9325 +/- 0.0049 | 03-31 16:33 |
| legacy_rkd_alpha1_gamma0 | alpha=1, gamma=0 | 0.9256 +/- 0.0064 | 03-31 17:00 |
| legacy_rkd_alpha1_gamma1 | alpha=1, gamma=1 | 0.9282 +/- 0.0072 | 03-31 17:28 |
| **relational_knowledge_distillation_gamma_distance1_gamma_angle2** | **gamma_d=1, gamma_a=2** | **0.9322 +/- 0.0044** | **03-31 22:02** |

### TGA / Relational TGA (three-stain teacher, `StudentTransABMIL`)
Key idea:
- TGA uses absolute cosine similarity as the attention target.
- RTGA uses relational discrimination across teachers.
- `L_contrast` trains the encoder directly through patch-to-teacher identification.

Bug note:
- 2026-04-07: `L_attn` was fixed from post-softmax MSE to logit-space MSE.
- Older `L_attn` results from 04-06 and early 04-07 should be treated as invalid.

| Condition | Loss | AUC | Date |
|-----------|------|-----|------|
| teacher_guided_attention_logit_space_alpha0_beta0_gamma1_delta0_tau1p0 | L_task + L_attn(cosine), tau=1.0 | 0.9270 +/- 0.0082 | 04-07 10:38 |
| teacher_guided_attention_logit_space_alpha0_beta0_gamma1_delta0_tau0p5 | L_task + L_attn(cosine), tau=0.5 | 0.9270 +/- 0.0082 | 04-07 10:56 |
| relational_teacher_guided_attention_logit_space_gamma1_lambda0_tau1p0 | L_task + L_attn(discrimination), tau=1.0 | 0.9273 +/- 0.0069 | 04-07 11:14 |
| relational_teacher_guided_attention_gamma0_lambda1_tau1p0 | L_task + L_contrast (gamma=0, lambda=1) | 0.9290 +/- 0.0067 | 04-07 10:12 |
| relational_teacher_guided_attention_logit_space_gamma1_lambda1_tau1p0 | L_task + L_attn + L_contrast | 0.9291 +/- 0.0093 | 04-07 11:32 |
| relational_teacher_guided_attention_logit_space_gamma1_lambda0p1_tau1p0 | L_task + L_attn + 0.1*L_contrast | 0.9270 +/- 0.0112 | 04-07 11:50 |

### Multi-head Cross-Attention Pooling (`StudentTransABMIL_MH`)
| Condition | Loss | AUC | Date |
|-----------|------|-----|------|
| multihead_contrastive_distillation_pool_heads4_baseline | L_task only (`pool_heads=4`) | 0.9304 +/- 0.0049 | 04-07 13:07 |
| multihead_contrastive_distillation_gamma0_lambda1_tau1p0_pool_heads4 | L_task + L_contrast | pending | |
| multihead_contrastive_distillation_gamma0_lambda0p1_tau1p0_pool_heads4 | L_task + 0.1*L_contrast | pending | |

## Key Findings
- Relational knowledge distillation remains the strongest distillation direction so far.
- Hidden matching is more effective than soft-label KD in the current setup.
- `StudentTransABMIL` outperforms the plain ABMIL student.
- Attention supervision is currently worse than the best RKD configuration.
- Pure contrastive supervision is better than the current attention-supervision variants, but still below RKD angle.

## Next Steps
- [ ] Run `multihead_contrastive_distillation_gamma0_lambda1_tau1p0_pool_heads4`
- [ ] Run `multihead_contrastive_distillation_gamma0_lambda0p1_tau1p0_pool_heads4`
- [ ] Try combining multi-head pooling with RKD angle
- [ ] Tune student architecture parameters such as `hidden_dim`, `nhead`, and transformer depth

## Decisions
- 2026-03-31: `StudentTransABMIL` is the default student architecture
- 2026-03-31: RKD angle became the main distillation direction
- 2026-04-07: `L_attn` switched to logit-space MSE

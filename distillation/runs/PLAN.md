# distillation/runs/PLAN — Knowledge Distillation

## Current Goal
找到最优蒸馏策略，将 3-stain teacher 知识迁移到 HE-only student。
当前最佳: `rkd_a0g1` (AUC 0.9325, RKD angle-only)

## Teacher
`run_concat_HE_CD20_CD3_mlp` (AUC 0.9531, hidden_dim=128, stains: HE+CD20+CD3)

## Results Summary

### StandardKD (3-stain teacher, TransABMIL student)
| Condition | alpha | beta | T | AUC | Date |
|-----------|-------|------|---|-----|------|
| distill_a0b0T4.0 | 0 | 0 | 4 | 0.9226 +/- 0.0099 | 03-31 |
| distill_a1b0T4.0 | 1 | 0 | 4 | 0.9222 +/- 0.0053 | 03-31 |
| distill_a0b1T4.0 | 0 | 1 | 4 | 0.9268 +/- 0.0068 | 03-31 |
| distill_a1b1T4.0 | 1 | 1 | 4 | 0.9276 +/- 0.0095 | 03-31 |

### RKD (3-stain teacher, TransABMIL student)
| Condition | Loss params | AUC | Date |
|-----------|-------------|-----|------|
| rkd_a0g0 | alpha=0, gamma=0 | 0.9231 +/- 0.0086 | 03-31 |
| **rkd_a0g1** | **alpha=0, gamma=1** | **0.9325 +/- 0.0049** | 03-31 |
| rkd_a1g0 | alpha=1, gamma=0 | 0.9256 +/- 0.0064 | 03-31 |
| rkd_a1g1 | alpha=1, gamma=1 | 0.9282 +/- 0.0072 | 03-31 |
| rkd_d1a2 | gamma_d=1, gamma_a=2 | 0.9322 +/- 0.0044 | 03-31 |

## Key Findings
- RKD angle loss (rkd_a0g1, 0.9325) 是当前最优蒸馏方法
- hidden matching (beta) 比 soft label (alpha) 更有效
- TransABMIL student 比 plain ABMIL 更好
- RKD+KD hybrid 没有比纯 RKD angle 好

## Next Steps
- [ ] 调整 student 架构参数 (hidden_dim, nhead, n_transformer_layers)
- [ ] 尝试更大的 angle weight (gamma_a > 2)
- [ ] 尝试其他蒸馏方法 (CRD, FitNet 等)

## Decisions
- 2026-03-31: TransABMIL 确认为默认 student 架构
- 2026-03-31: RKD angle loss 优于 StandardKD, 作为蒸馏主方向

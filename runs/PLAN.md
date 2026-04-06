# runs/PLAN — Teacher Selection

## Current Goal
找到最优多模态 teacher 用于蒸馏。当前最佳: `run_concat_HE_CD20_CD3_mlp` (AUC 0.9531, hidden_dim=128)

## Results Summary

| Condition | Stains | Model | AUC (run-level) | Date |
|-----------|--------|-------|------------------|------|
| unimodal_HE_abmil | HE | ABMIL | 0.9100 +/- 0.0080 | 03-31 |
| unimodal_CD20_abmil | CD20 | ABMIL | 0.9294 +/- 0.0041 | 03-31 |
| unimodal_HE_linear | HE | LinearProbe | 0.8975 +/- 0.0097 | 03-30 |
| unimodal_CD20_linear | CD20 | LinearProbe | 0.9344 +/- 0.0112 | 03-30 |
| concat_HE_CD20_mlp | HE+CD20 | MLP | 0.9480 +/- 0.0020 | 03-30 |
| concat_HE_CD20_CD3_mlp (dim=256) | HE+CD20+CD3 | MLP | 0.9512 +/- 0.0093 | 03-30 |
| **concat_HE_CD20_CD3_mlp (dim=128)** | HE+CD20+CD3 | MLP | **0.9531 +/- 0.0022** | 03-31 |

## Key Findings
- 多模态 > 单模态: 3-stain concat (0.953) >> HE-only (0.910)
- CD20 单模态 (0.929) > HE 单模态 (0.910)
- hidden_dim=128 比 256 略优且更稳定
- patience=30 vs 10 对最终结果影响不大

## Next Steps
- [ ] 尝试 4-stain (HE+CD20+CD3+Ki67) concat — 验证更多模态是否继续提升
- [ ] 尝试 attention-based fusion 替代简单 concat
- [ ] 比较不同 foundation model 特征 (GigaPath vs UNI2)

## Decisions
- 2026-03-31: 确认 3-stain concat MLP (hidden=128) 为当前最优 teacher, 已用于蒸馏
- 2026-03-31: 统一 patience=30, batch_size=16

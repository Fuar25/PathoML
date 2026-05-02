# RKD vs RKD+TGA Diagnostic

## Scope
- Uses saved patient-level `cv_predictions.csv`; no retraining.
- Compares RKD against RKD+TGA weights 0.5 and 0.75.
- Treats each run-patient pair as a diagnostic sample; this is not an independence claim.
- Reports pooled patient-level metrics from saved predictions; these are diagnostic and do not replace logged fold-level metrics.
- Caveat: historical RKD output uses the old teacher alias in logs, while RKD+TGA uses `run_concat_HE_CD20_CD3_mlp_bs32_lr4em4`.

## Per-Run Summary
| method       | auc_mean | auc_std | f1_mean | f1_std | acc_mean | fn_mean | fp_mean |
| ------------ | -------- | ------- | ------- | ------ | -------- | ------- | ------- |
| RKD          | 0.9004   | 0.0072  | 0.8546  | 0.0025 | 0.8419   | 14.6000 | 28.4000 |
| RKD+TGA_0.5  | 0.9112   | 0.0113  | 0.8576  | 0.0140 | 0.8426   | 12.2000 | 30.6000 |
| RKD+TGA_0.75 | 0.9108   | 0.0116  | 0.8582  | 0.0131 | 0.8426   | 11.6000 | 31.2000 |

## Paired Error Flips
| comparison          | fixed | regressed | net_fixed | both_wrong | both_correct | positive_fixed | positive_regressed | negative_fixed | negative_regressed |
| ------------------- | ----- | --------- | --------- | ---------- | ------------ | -------------- | ------------------ | -------------- | ------------------ |
| RKD vs RKD+TGA_0.5  | 33    | 32        | 1         | 182        | 1113         | 19             | 7                  | 14             | 25                 |
| RKD vs RKD+TGA_0.75 | 33    | 32        | 1         | 182        | 1113         | 21             | 6                  | 12             | 26                 |

## Teacher Confidence Groups
| comparison          | teacher_conf_bin | teacher_correct | n   | base_acc | other_acc | delta_acc | fixed | regressed |
| ------------------- | ---------------- | --------------- | --- | -------- | --------- | --------- | ----- | --------- |
| RKD vs RKD+TGA_0.5  | low<=0.5         | False           | 124 | 0.4758   | 0.4194    | -0.0565   | 3     | 10        |
| RKD vs RKD+TGA_0.5  | low<=0.5         | True            | 183 | 0.6776   | 0.7104    | 0.0328    | 14    | 8         |
| RKD vs RKD+TGA_0.5  | mid(0.5,0.8]     | False           | 38  | 0.1316   | 0.0789    | -0.0526   | 0     | 2         |
| RKD vs RKD+TGA_0.5  | mid(0.5,0.8]     | True            | 250 | 0.9000   | 0.9040    | 0.0040    | 10    | 9         |
| RKD vs RKD+TGA_0.5  | high>0.8         | False           | 22  | 0.0000   | 0.0455    | 0.0455    | 1     | 0         |
| RKD vs RKD+TGA_0.5  | high>0.8         | True            | 743 | 0.9852   | 0.9879    | 0.0027    | 5     | 3         |
| RKD vs RKD+TGA_0.75 | low<=0.5         | False           | 124 | 0.4758   | 0.4194    | -0.0565   | 4     | 11        |
| RKD vs RKD+TGA_0.75 | low<=0.5         | True            | 183 | 0.6776   | 0.6995    | 0.0219    | 14    | 10        |
| RKD vs RKD+TGA_0.75 | mid(0.5,0.8]     | False           | 38  | 0.1316   | 0.0789    | -0.0526   | 0     | 2         |
| RKD vs RKD+TGA_0.75 | mid(0.5,0.8]     | True            | 250 | 0.9000   | 0.9160    | 0.0160    | 10    | 6         |
| RKD vs RKD+TGA_0.75 | high>0.8         | False           | 22  | 0.0000   | 0.0455    | 0.0455    | 1     | 0         |
| RKD vs RKD+TGA_0.75 | high>0.8         | True            | 743 | 0.9852   | 0.9865    | 0.0013    | 4     | 3         |

## Read
- A useful TGA signal should show more fixed cases than regressed cases, especially in teacher-correct/high-confidence groups.
- If fixed and regressed counts are close, the aggregate F1 gain is likely noise-level rather than a reliable mechanism.

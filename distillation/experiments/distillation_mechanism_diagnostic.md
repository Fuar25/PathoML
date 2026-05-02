# Distillation Mechanism Diagnostic

## Scope
- Uses saved patient-level `cv_predictions.csv`; no retraining.
- Uses the fixed `StudentBasicABMIL` platform only.
- Reports pooled patient-level diagnostics from saved predictions.
- Logged fold-level metrics in `PLAN.md` remain the primary experiment results.
- Threshold sweeps are diagnostic upper bounds on saved test predictions; they are not deployable validation-tuned thresholds.

## Method Summary
| group        | method              | auc_mean | auc_std | pr_auc_mean | f1_mean | f1_std | acc_mean | recall_mean | specificity_mean | fp_mean | fn_mean | brier_mean | ece10_mean |
| ------------ | ------------------- | -------- | ------- | ----------- | ------- | ------ | -------- | ----------- | ---------------- | ------- | ------- | ---------- | ---------- |
| confirmation | RKD_TGA_0.5_confirm | 0.9030   | 0.0128  | 0.9047      | 0.8452  | 0.0138 | 0.8250   | 0.9191      | 0.7237           | 36.2000 | 11.4000 | 0.1459     | 0.1245     |
| confirmation | RKD_confirm         | 0.9000   | 0.0168  | 0.9008      | 0.8349  | 0.0174 | 0.8147   | 0.9007      | 0.7221           | 36.4000 | 14.0000 | 0.1446     | 0.1065     |
| experimental | RKD_TGA_0.5         | 0.9112   | 0.0113  | 0.9137      | 0.8576  | 0.0140 | 0.8426   | 0.9135      | 0.7664           | 30.6000 | 12.2000 | 0.1367     | 0.1222     |
| main         | ABMIL               | 0.9010   | 0.0142  | 0.9045      | 0.8344  | 0.0105 | 0.8228   | 0.8610      | 0.7817           | 28.6000 | 19.6000 | 0.1267     | 0.0672     |
| main         | DKD_strict          | 0.9113   | 0.0043  | 0.9195      | 0.8460  | 0.0062 | 0.8338   | 0.8809      | 0.7832           | 28.4000 | 16.8000 | 0.1224     | 0.0641     |
| main         | Hidden              | 0.9022   | 0.0166  | 0.9077      | 0.8293  | 0.0201 | 0.8132   | 0.8723      | 0.7496           | 32.8000 | 18.0000 | 0.1303     | 0.0682     |
| main         | RKD                 | 0.9004   | 0.0072  | 0.9027      | 0.8546  | 0.0025 | 0.8419   | 0.8965      | 0.7832           | 28.4000 | 14.6000 | 0.1343     | 0.0953     |
| main         | SPKD                | 0.9007   | 0.0168  | 0.9083      | 0.8330  | 0.0142 | 0.8191   | 0.8695      | 0.7649           | 30.8000 | 18.4000 | 0.1286     | 0.0571     |
| main         | StandardKD          | 0.9113   | 0.0043  | 0.9195      | 0.8460  | 0.0062 | 0.8338   | 0.8809      | 0.7832           | 28.4000 | 16.8000 | 0.1224     | 0.0641     |

## Paired Error Flips
| comparison                         | fixed | regressed | net_fixed | both_wrong | both_correct | positive_fixed | positive_regressed | negative_fixed | negative_regressed |
| ---------------------------------- | ----- | --------- | --------- | ---------- | ------------ | -------------- | ------------------ | -------------- | ------------------ |
| ABMIL vs StandardKD                | 55    | 40        | 15        | 186        | 1079         | 31             | 17                 | 24             | 23                 |
| ABMIL vs RKD                       | 70    | 44        | 26        | 171        | 1075         | 38             | 13                 | 32             | 31                 |
| ABMIL vs RKD_TGA_0.5               | 68    | 41        | 27        | 173        | 1078         | 45             | 8                  | 23             | 33                 |
| StandardKD vs RKD                  | 47    | 36        | 11        | 179        | 1098         | 22             | 11                 | 25             | 25                 |
| RKD vs RKD_TGA_0.5                 | 33    | 32        | 1         | 182        | 1113         | 19             | 7                  | 14             | 25                 |
| RKD_confirm vs RKD_TGA_0.5_confirm | 36    | 22        | 14        | 216        | 1086         | 21             | 8                  | 15             | 14                 |

## Teacher Confidence Groups
| comparison                         | teacher_conf_bin | teacher_correct | n   | base_acc | other_acc | delta_acc | fixed | regressed |
| ---------------------------------- | ---------------- | --------------- | --- | -------- | --------- | --------- | ----- | --------- |
| ABMIL vs StandardKD                | low<=0.5         | False           | 124 | 0.4355   | 0.4355    | 0.0000    | 11    | 11        |
| ABMIL vs StandardKD                | low<=0.5         | True            | 183 | 0.6557   | 0.6612    | 0.0055    | 16    | 15        |
| ABMIL vs StandardKD                | mid(0.5,0.8]     | False           | 38  | 0.0789   | 0.1316    | 0.0526    | 2     | 0         |
| ABMIL vs StandardKD                | mid(0.5,0.8]     | True            | 250 | 0.8600   | 0.8720    | 0.0120    | 14    | 11        |
| ABMIL vs StandardKD                | high>0.8         | False           | 22  | 0.0000   | 0.0455    | 0.0455    | 1     | 0         |
| ABMIL vs StandardKD                | high>0.8         | True            | 743 | 0.9785   | 0.9892    | 0.0108    | 11    | 3         |
| ABMIL vs RKD                       | low<=0.5         | False           | 124 | 0.4355   | 0.4758    | 0.0403    | 18    | 13        |
| ABMIL vs RKD                       | low<=0.5         | True            | 183 | 0.6557   | 0.6776    | 0.0219    | 23    | 19        |
| ABMIL vs RKD                       | mid(0.5,0.8]     | False           | 38  | 0.0789   | 0.1316    | 0.0526    | 3     | 1         |
| ABMIL vs RKD                       | mid(0.5,0.8]     | True            | 250 | 0.8600   | 0.9000    | 0.0400    | 19    | 9         |
| ABMIL vs RKD                       | high>0.8         | False           | 22  | 0.0000   | 0.0000    | 0.0000    | 0     | 0         |
| ABMIL vs RKD                       | high>0.8         | True            | 743 | 0.9785   | 0.9852    | 0.0067    | 7     | 2         |
| ABMIL vs RKD_TGA_0.5               | low<=0.5         | False           | 124 | 0.4355   | 0.4194    | -0.0161   | 13    | 15        |
| ABMIL vs RKD_TGA_0.5               | low<=0.5         | True            | 183 | 0.6557   | 0.7104    | 0.0546    | 25    | 15        |
| ABMIL vs RKD_TGA_0.5               | mid(0.5,0.8]     | False           | 38  | 0.0789   | 0.0789    | 0.0000    | 1     | 1         |
| ABMIL vs RKD_TGA_0.5               | mid(0.5,0.8]     | True            | 250 | 0.8600   | 0.9040    | 0.0440    | 20    | 9         |
| ABMIL vs RKD_TGA_0.5               | high>0.8         | False           | 22  | 0.0000   | 0.0455    | 0.0455    | 1     | 0         |
| ABMIL vs RKD_TGA_0.5               | high>0.8         | True            | 743 | 0.9785   | 0.9879    | 0.0094    | 8     | 1         |
| StandardKD vs RKD                  | low<=0.5         | False           | 124 | 0.4355   | 0.4758    | 0.0403    | 14    | 9         |
| StandardKD vs RKD                  | low<=0.5         | True            | 183 | 0.6612   | 0.6776    | 0.0164    | 19    | 16        |
| StandardKD vs RKD                  | mid(0.5,0.8]     | False           | 38  | 0.1316   | 0.1316    | 0.0000    | 1     | 1         |
| StandardKD vs RKD                  | mid(0.5,0.8]     | True            | 250 | 0.8720   | 0.9000    | 0.0280    | 11    | 4         |
| StandardKD vs RKD                  | high>0.8         | False           | 22  | 0.0455   | 0.0000    | -0.0455   | 0     | 1         |
| StandardKD vs RKD                  | high>0.8         | True            | 743 | 0.9892   | 0.9852    | -0.0040   | 2     | 5         |
| RKD vs RKD_TGA_0.5                 | low<=0.5         | False           | 124 | 0.4758   | 0.4194    | -0.0565   | 3     | 10        |
| RKD vs RKD_TGA_0.5                 | low<=0.5         | True            | 183 | 0.6776   | 0.7104    | 0.0328    | 14    | 8         |
| RKD vs RKD_TGA_0.5                 | mid(0.5,0.8]     | False           | 38  | 0.1316   | 0.0789    | -0.0526   | 0     | 2         |
| RKD vs RKD_TGA_0.5                 | mid(0.5,0.8]     | True            | 250 | 0.9000   | 0.9040    | 0.0040    | 10    | 9         |
| RKD vs RKD_TGA_0.5                 | high>0.8         | False           | 22  | 0.0000   | 0.0455    | 0.0455    | 1     | 0         |
| RKD vs RKD_TGA_0.5                 | high>0.8         | True            | 743 | 0.9852   | 0.9879    | 0.0027    | 5     | 3         |
| RKD_confirm vs RKD_TGA_0.5_confirm | low<=0.5         | False           | 124 | 0.3387   | 0.3629    | 0.0242    | 12    | 9         |
| RKD_confirm vs RKD_TGA_0.5_confirm | low<=0.5         | True            | 183 | 0.6612   | 0.6995    | 0.0383    | 12    | 5         |
| RKD_confirm vs RKD_TGA_0.5_confirm | mid(0.5,0.8]     | False           | 38  | 0.1053   | 0.0789    | -0.0263   | 1     | 2         |
| RKD_confirm vs RKD_TGA_0.5_confirm | mid(0.5,0.8]     | True            | 250 | 0.8760   | 0.8920    | 0.0160    | 9     | 5         |
| RKD_confirm vs RKD_TGA_0.5_confirm | high>0.8         | False           | 22  | 0.0455   | 0.0455    | 0.0000    | 0     | 0         |
| RKD_confirm vs RKD_TGA_0.5_confirm | high>0.8         | True            | 743 | 0.9704   | 0.9717    | 0.0013    | 2     | 1         |

## Threshold Sweep Diagnostic
| group        | method              | fixed_f1_mean | best_f1_mean | f1_gain_mean | best_threshold_mean | fixed_recall_mean | best_recall_mean | fixed_specificity_mean | best_specificity_mean |
| ------------ | ------------------- | ------------- | ------------ | ------------ | ------------------- | ----------------- | ---------------- | ---------------------- | --------------------- |
| confirmation | RKD_TGA_0.5_confirm | 0.8452        | 0.8549       | 0.0097       | 0.5060              | 0.9191            | 0.9220           | 0.7237                 | 0.7466                |
| confirmation | RKD_confirm         | 0.8349        | 0.8507       | 0.0158       | 0.5158              | 0.9007            | 0.8865           | 0.7221                 | 0.7878                |
| experimental | RKD_TGA_0.5         | 0.8576        | 0.8636       | 0.0059       | 0.4997              | 0.9135            | 0.9191           | 0.7664                 | 0.7740                |
| main         | ABMIL               | 0.8344        | 0.8449       | 0.0105       | 0.4366              | 0.8610            | 0.9035           | 0.7817                 | 0.7466                |
| main         | DKD_strict          | 0.8460        | 0.8548       | 0.0087       | 0.4511              | 0.8809            | 0.9092           | 0.7832                 | 0.7649                |
| main         | Hidden              | 0.8293        | 0.8486       | 0.0193       | 0.4894              | 0.8723            | 0.8950           | 0.7496                 | 0.7695                |
| main         | RKD                 | 0.8546        | 0.8595       | 0.0048       | 0.4926              | 0.8965            | 0.9064           | 0.7832                 | 0.7817                |
| main         | SPKD                | 0.8330        | 0.8466       | 0.0136       | 0.4313              | 0.8695            | 0.9078           | 0.7649                 | 0.7450                |
| main         | StandardKD          | 0.8460        | 0.8548       | 0.0087       | 0.4511              | 0.8809            | 0.9092           | 0.7832                 | 0.7649                |

## Calibration Bins
| method      | prob_bin  | n   | mean_prob | positive_rate | abs_gap |
| ----------- | --------- | --- | --------- | ------------- | ------- |
| ABMIL       | [0,0.2)   | 412 | 0.0530    | 0.0752        | 0.0222  |
| ABMIL       | [0.2,0.4) | 117 | 0.2962    | 0.2991        | 0.0030  |
| ABMIL       | [0.4,0.6) | 161 | 0.4987    | 0.4720        | 0.0266  |
| ABMIL       | [0.6,0.8) | 211 | 0.7022    | 0.6919        | 0.0102  |
| ABMIL       | [0.8,1]   | 459 | 0.9365    | 0.9085        | 0.0280  |
| RKD         | [0,0.2)   | 314 | 0.0757    | 0.0669        | 0.0088  |
| RKD         | [0.2,0.4) | 127 | 0.3009    | 0.1575        | 0.1434  |
| RKD         | [0.4,0.6) | 352 | 0.5084    | 0.4915        | 0.0169  |
| RKD         | [0.6,0.8) | 226 | 0.6887    | 0.7743        | 0.0856  |
| RKD         | [0.8,1]   | 341 | 0.9329    | 0.9267        | 0.0062  |
| RKD_TGA_0.5 | [0,0.2)   | 228 | 0.1024    | 0.0526        | 0.0497  |
| RKD_TGA_0.5 | [0.2,0.4) | 174 | 0.2943    | 0.0977        | 0.1966  |
| RKD_TGA_0.5 | [0.4,0.6) | 413 | 0.5122    | 0.4576        | 0.0546  |
| RKD_TGA_0.5 | [0.6,0.8) | 277 | 0.6822    | 0.8375        | 0.1553  |
| RKD_TGA_0.5 | [0.8,1]   | 268 | 0.9042    | 0.9515        | 0.0472  |
| StandardKD  | [0,0.2)   | 430 | 0.0596    | 0.0767        | 0.0171  |
| StandardKD  | [0.2,0.4) | 107 | 0.2888    | 0.2804        | 0.0084  |
| StandardKD  | [0.4,0.6) | 131 | 0.5046    | 0.4504        | 0.0542  |
| StandardKD  | [0.6,0.8) | 191 | 0.7123    | 0.7225        | 0.0102  |
| StandardKD  | [0.8,1]   | 501 | 0.9457    | 0.8882        | 0.0575  |

## Read
- RKD is the strongest fixed-threshold F1 mechanism on the main BasicABMIL line in this diagnostic: it reduces mean FN versus ABMIL (`19.6 -> 14.6`) while leaving mean FP nearly unchanged (`28.6 -> 28.4`).
- RKD improves error flips over ABMIL more than StandardKD does (`net_fixed=26` vs `15`), and still improves over StandardKD directly (`net_fixed=11`).
- RKD gains are not a clean calibration win: Brier/ECE are worse than ABMIL and StandardKD, so the useful effect is better interpreted as recall/FN operating-point behavior.
- RKD is also not a clear pooled ranking winner in this saved-prediction diagnostic; logged fold-level metrics remain the primary result for AUC comparisons.
- RKD+TGA 0.5 remains experimental: it improves F1/recall and confirmation paired flips, but the main RKD-vs-TGA paired flip result is near cancelled (`net_fixed=1`) and the confirmation AUC does not beat RKD.
- Threshold sweeps do not erase RKD: tuned diagnostic F1 remains highest for RKD+TGA 0.5 experimental (`0.8636`) and RKD remains the best main-table method (`0.8595`).
- Next mechanism work should focus on RKD-style relation/operating-point analysis, not new TGA variants or stronger students.

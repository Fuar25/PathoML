# PathoML/runs

## 1. Purpose
Standard, structrued scripts for running experiments based on the core library. These scripts are meant for combined unique dataset and unique model structure into training pipelines. Every single run means a seperate experiment, sharing the same logfile to compare.

## 2. Rules
- Every run follows the same structure, which is `make_config` first, `run_condition` then and `log_results` last. This is to ensure the consistency of the code.
- The name of the script should follow the same pattern, which is `run_{dataset_class}_{modal_name}_{model_class}.py`. For example, `run_concat_HE_CD20_mlp.py`, which means the dataset is `ConcatDataset`, the modal is `HE+CD20` and the model is `MLP`. User may not point out the experiment name explicitly, but the name should be inferred and follow the pattern.
- If user does not imply some experiment hyperparameters, you should ask them first to get all set, rather than first implementing.
- The condition name which is presented by `CONDITION_NAME` should be the same as the name of the script. using `os.path.splitext(os.path.basename(__file__))[0]`

## 3. Cross-experiment Comparability
For AUC results across different scripts to be directly comparable (same fold splits, same training/test patients), all experiments in a comparison group must share:

1. **Same sample set** — compute `allowed_sample_keys` as the intersection of all modalities involved across *all* experiments in the group, not just the modalities each individual script uses. For example, `run_unimodal_HE_abmil.py` uses `find_common_sample_keys([HE_patch, CD20_slide, CD3_slide])` even though it only trains on HE, to align with the multimodal baselines.
2. **Same `BASE_SEED`, `K_FOLDS`, `N_RUNS`** — these are defined in `common.py` and shared by all `runs/` scripts. Downstream scripts (e.g. distillation) obtain these values from the teacher manifest (see §4).
3. **Same dataset ordering** — all dataset implementations must sort samples by `(patient_id, tissue_id)` (see `PathoML/dataset/DESIGN.md`). This is what makes the same seed produce the same fold splits across different dataset classes.

## 4. Teacher Manifest
`run_condition()` automatically writes `manifest.json` into the condition output directory after training completes. The manifest records fold parameters (`n_runs`, `k_folds`, `base_seed`), modality configuration (`modality_names`, `modality_paths`), model info, and a relative checkpoint path template. Downstream consumers (e.g. `distillation/`) load this manifest to inherit teacher configuration, eliminating manual parameter alignment.
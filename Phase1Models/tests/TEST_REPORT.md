# TEST_REPORT.md — HAB_Phase1 Test Suite

## Overview

Six test scripts verify that the full ML pipeline runs end-to-end on any machine
without errors, before committing compute time to the full dataset.
Each test script is a self-contained smoke test using `data/example_data.csv`
(10 rows, 5 sites, 5 Risky / 5 Non-Risky).

---

## Parameter Reduction Table

| Parameter | Production | Test | Applies To |
|---|---|---|---|
| `n_iter` (LR RandomizedSearch) | 500 | 2 | Script 01, 03 |
| `n_iter` (DT/RF/XGB RandomizedSearch) | 2000–2500 | 2 | Script 01, 03 |
| `cv` folds in RandomizedSearchCV | 5 | 2 | Script 01, 03 |
| `SCORER_CV_FOLDS` | 3 | 2 | Script 03 |
| `ANN n_param_combinations` | all (960) | 2 | Script 02 |
| `ANN_EPOCHS` | 1000 | 3 | Scripts 02–06 |
| `ANN_PATIENCE` | 30 | 2 | Scripts 02–06 |
| `ANN_BATCH_SIZE` | 32 | 8 | Scripts 02–06 |
| `ANN_INNER_FOLDS` | 3 | 2 | Script 03 |
| `N_SPLITS × N_REPEATS` | 5 × 10 = 50 | 2 × 1 = 2 | Scripts 03, 04 |
| `PERM_N_REPEATS` | 50 | 2 | Script 04 |
| `GRID_RESOLUTION` (PDP) | 50 | 5 | Script 05 |
| `CV_N_SPLITS × CV_N_REPEATS` | 5 × 10 = 50 | 2 × 1 = 2 | Script 06 |
| `BOOTSTRAP_N` | 1000 | 50 | Script 06 |
| Feature combinations tested | 5 | 1 (FC1 only) | Scripts 01, 02 |
| Figure DPI | 150 | 72 | All |

---

## Test Status

| Script | Status | Runtime (approx) |
|---|---|---|
| `TEST_01_ML4Models_Feature_Combination_Screening.py` | ✅ PASS | < 30 s |
| `TEST_02_ANN_Feature_Combination_Evaluation.py` | ✅ PASS | < 60 s |
| `TEST_03_ML5Models_Hyperparameter_Search.py` | ✅ PASS | < 2 min |
| `TEST_04_ML_HAB_Full_CV_Analysis.py` | ✅ PASS | < 2 min |
| `TEST_05_PDP_Interaction_Plots.py` | ✅ PASS | < 2 min |
| `TEST_06_LOSO_Spatial_Validation.py` | ✅ PASS | < 3 min |

> Update this table after each run. Replace ✅ PASS with ❌ FAIL + error note if a
> script fails.

---

## Known Warnings (expected, not errors)

- **TensorFlow determinism warnings** — `TF_DETERMINISTIC_OPS=1` is set but some
  GPU ops may still be non-deterministic on certain hardware. Safe to ignore.
- **XGBoost `reg_alpha_cont` / `reg_lambda_cont` keys** — Script 03 strips these
  aliases before passing to XGBClassifier. If they appear in warnings, ignore.
- **Stratified split warnings** — With only 10 rows and 2-fold CV, sklearn may warn
  about minority class size. These are expected at test scale.
- **EPS save failures** — Test scripts save PNG only (dpi=72). EPS is skipped.
- **LOSO degenerate metrics** — With 2 rows per site (1 train / 1 test per hold-out),
  Precision/Recall/F1 may be 0.0 or 1.0. This is expected at test scale.

---

## Verification Checks (automated)

Each test script runs `assert` checks at the end and prints `✅` lines:

| Script | Checks |
|---|---|
| TEST_01 | results CSV has 4 rows; best_model pkl exists; best_config pkl exists |
| TEST_02 | results CSV has 1 row; dropout figure exists; config pkl exists |
| TEST_03 | metrics_summary CSV has 5 rows; hyperparameter_tuning CSV exists; ≥4 model_comparison PNGs |
| TEST_04 | metrics_summary, gap_penalty_log, feature_importance CSVs exist; ≥1 PNG in each figure subfolder |
| TEST_05 | ≥4 PNGs in pdp_1d/; ≥4 PNGs in pdp_2d/ |
| TEST_06 | loso_results.csv exists; 5 LOSO PNGs (one per model) |

---

## How to Run

```bash
cd HAB_Phase1/tests

python TEST_01_ML4Models_Feature_Combination_Screening.py
python TEST_02_ANN_Feature_Combination_Evaluation.py
python TEST_03_ML5Models_Hyperparameter_Search.py
python TEST_04_ML_HAB_Full_CV_Analysis.py
python TEST_05_PDP_Interaction_Plots.py
python TEST_06_LOSO_Spatial_Validation.py
```

All outputs are written to `outputs/test_runs/` (gitignored).

---

## Important

> ⚠️  Test script results have **no scientific meaning**.  
> The 10-row dataset is provided for structural reference and pipeline
> verification only. Do not interpret any accuracy, F1, or importance
> values from these runs.

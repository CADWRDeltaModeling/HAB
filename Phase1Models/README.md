# Machine Learning-based Cyanobacteria Harmful Algal Bloom Toxicity Risk Prediction in the Sacramento–San Joaquin Delta of California

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![DOI](https://img.shields.io/badge/Data%20DOI-10.6073%2Fpasta%2Fb4e180cb925266171285375a3e202635-blue)](https://doi.org/10.6073/pasta/b4e180cb925266171285375a3e202635)

---

## Overview

This repository contains the Phase 1 machine learning (ML) pipeline for predicting the occurrence of toxin-producing *Microcystis* — a key driver of cyanobacterial harmful algal blooms (CHABs) — in the Sacramento–San Joaquin Delta, California. The work supports the **California Department of Water Resources (DWR) Delta Modeling Section's** efforts to develop early-warning and decision-support tools for Delta water quality management.

The pipeline frames prediction as a **binary classification** task (**Risky** vs. **Non-Risky**), defined by a toxin-producing *Microcystis* cell count threshold of **4,000 cells/mL**. Five machine learning models are trained and evaluated on physicochemical water quality observations collected at **14 fixed monitoring stations** across the Delta (2014–2019, N = 220 sampling days).

---

## Study Area & Dataset

| Attribute | Value |
|---|---|
| Region | Sacramento–San Joaquin Delta, California |
| Monitoring sites | 14 fixed water quality stations |
| Period of record | 2014–2019 |
| Total observations | 220 sampling days |
| Target variable | `ToxinProdc_Microcystis_CC` (toxin-producing *Microcystis* cell count) |
| Classification threshold | 4,000 cells/mL |
| Class distribution | 91 Risky / 129 Non-Risky |

### Monitoring Sites

| Site Code | Full Name |
|---|---|
| AT | AT_DWR_Site_D12_EMP_Site |
| BI | BI_DWR_Site |
| CV | CV_DWR_Site_D4_EMP_Site |
| D16 | D16_EMP_Site |
| D26 | D26_EMP_Site |
| D8 | D8_EMP_Site |
| FT | FT_DWR_Site_D19_EMP_Site |
| JP | JP_DWR_Site |
| MD10A | MD10A_EMP_Site |
| MI | MI_DWR_Site |
| OR | OR_DWR_Site_D28A_EMP_Site |
| RR | RR_DWR_Site_P8_EMP_Site |
| SJ | SJ_DWR_Site |
| VC | VC_DWR_Site |

---

## Feature Pool (17 Predictors)

| Category | Feature | Description | Unit |
|---|---|---|---|
| **Flow / Antecedent Flow** | `Q` | Streamflow | m³/s |
| | `Q_2w` | 2-week average antecedent flow | m³/s |
| | `Q_1m` | 1-month average antecedent flow | m³/s |
| **Conductance / Concentration** | `SpecCond.` | Specific Conductance | µS/cm |
| | `TDS` | Total Dissolved Solids | mg/L |
| | `Salinity` | Salinity | PSU |
| **Turbidity** | `NTU` | Turbidity (Nephelometric Turbidity Unit) | NTU |
| **Carbon Content** | `OrgC` | Dissolved Organic Carbon | mg/L |
| | `Total_OrgC` | Total Organic Carbon | mg/L |
| **Nutrient — Nitrogen** | `NH4` | Dissolved Ammonia | mg/L |
| | `NO3` | Dissolved Nitrate | mg/L |
| | `OrgN` | Dissolved Organic Nitrogen | mg/L |
| **Nutrient — Phosphorus** | `PO4` | Dissolved Orthophosphate | mg/L |
| | `TP` | Total Phosphorus | mg/L |
| **Physical Water Quality** | `WaterTemp` | Water Temperature | °C |
| | `DO` | Dissolved Oxygen | mg/L |
| | `pH` | pH | — |

**Best Feature Combination (FC1):** `Q`, `Q_1m`, `SpecCond.`, `OrgC`, `OrgN`, `PO4`, `WaterTemp`, `DO`, `pH`

---

## Machine Learning Models

| Model | Abbreviation |
|---|---|
| Logistic Regression | LR |
| Decision Tree | DT |
| Random Forest | RF |
| XGBoost | XGB |
| Artificial Neural Network | ANN |

---

## Pipeline Architecture

Six Jupyter notebooks run sequentially:

```
Stage 1 │ 01_ML4Models_Feature_Combination_Screening    │ Feature subset screening (~20,845 combos, 4 models)
Stage 2 │ 02_ANN_Feature_Combination_Evaluation         │ ANN evaluation on Top-5 feature combinations
Stage 3 │ 03_ML5Models_Hyperparameter_Search            │ Full hyperparameter tuning (5 models, FC1)
Stage 4 │ 04_ML_HAB_Full_CV_Analysis                    │ 5-Fold × 10-Repeat stratified CV (50 folds)
Stage 5 │ 05_PDP_2D_Interaction_Plots                   │ 1D PDPs + 2D interaction surfaces
Stage 6 │ 06_LOSO_Spatial_Validation                    │ Leave-One-Site-Out validation (14 sites)
```

### Stage Descriptions

**Stage 1 — Feature Combination Screening**

Exhaustive screening of feature subsets from the 17-feature pool using LR, DT, RF, and XGBoost. Evaluates approximately 20,845 combinations with stratified 70/30 train-test splits to identify the top feature combinations for downstream tuning.

**Stage 2 — ANN Feature Combination Evaluation**

Applies the ANN to the top 5 feature combinations from Stage 1. Grid search over hidden layers, activation functions, neuron counts, learning rates, and dropout rates with early stopping (patience = 30 epochs). Outputs per-combination performance tables and dropout impact figures.

**Stage 3 — Hyperparameter Search**

Full hyperparameter tuning for all five models on the best feature combination (FC1) using 5-fold × 10-repeat stratified CV. Search strategies: `RandomizedSearchCV` for LR (n_iter=500), DT (n_iter=2500), RF (n_iter=2500), XGB (n_iter=2500); manual grid for ANN. RF and XGB use a hard-penalty overfitting diagnostic (gap tolerance: primary ≤ 10%, fallback ≤ 12%).

**Stage 4 — Full CV Analysis**

Evaluates final tuned models on FC1 using the complete 50-fold protocol. Generates:
- Model comparison bar plots (train & test mean ± SD)
- Aggregated confusion matrices
- Predicted probability uncertainty plots (5%–95% per-sample bands)
- Probability density plots
- Feature importance plots (intrinsic for LR/DT/RF/XGB; permutation importance for ANN)
- Overfitting gap penalty log for RF and XGBoost

**Stage 5 — Partial Dependence & Interaction Plots**

- **1D PDPs** for top predictors: `WaterTemp`, `pH`, `OrgN`
- **2D interaction surfaces** for pairs: `WaterTemp × pH`, `WaterTemp × OrgN`, `pH × OrgN`

Models are refitted on the full scaled dataset. `sklearn.inspection.partial_dependence()` is used for LR, DT, RF, XGB; a manual ICE/PDP grid is computed for the ANN.

**Stage 6 — LOSO Spatial Validation**

Leave-One-Site-Out cross-validation across all 14 monitoring stations. For each held-out site, models are trained on the remaining 13 sites and tested on the left-out site (5-fold × 10-repeat CV internally). Reports asymmetric min–max error bars. Produces one figure per model with 4 subplots (Accuracy, Precision, Recall, F1-Score).

---

## Repository Structure

```
HAB_Phase1/
├── notebooks/
│   ├── 01_ML4Models_Feature_Combination_Screening.ipynb
│   ├── 02_ANN_Feature_Combination_Evaluation.ipynb
│   ├── 03_ML5Models_Hyperparameter_Search.ipynb
│   ├── 04_ML_HAB_Full_CV_Analysis.ipynb
│   ├── 05_PDP_2D_Interaction_Plots.ipynb
│   └── 06_LOSO_Spatial_Validation.ipynb
├── data/
│   ├── README.md          ← Data access instructions + column descriptions
│   └── example_data.csv   ← 10-row illustrative sample (structure reference only)
├── tests/                 ← Lightweight test suite using example_data.csv
│   ├── TEST_01_ML4Models_Feature_Combination_Screening.py
│   ├── TEST_02_ANN_Feature_Combination_Evaluation.py
│   ├── TEST_03_ML5Models_Hyperparameter_Search.py
│   ├── TEST_04_ML_HAB_Full_CV_Analysis.py
│   ├── TEST_05_PDP_2D_Interaction_Plots.py
│   ├── TEST_06_LOSO_Spatial_Validation.py
│   └── TEST_REPORT.md     ← Test results and parameter reduction table
├── outputs/               ← All script outputs written here (gitignored)
│   ├── 01_FC_4Models_Results/
│   ├── 02_FC_ANN_Results/
│   ├── 03_HS_CV_5ML/
│   ├── 04_ML5Models_Analysis_Results/
│   ├── 05_PDP_Results/
│   └── 06_LOSO_Results/
├── config.py              ← Project-wide path configuration
├── environment.yml        ← Conda environment (HabsEnv_2025)
└── README.md
```

> **Note:** The `outputs/` directory is listed in `.gitignore` — all generated figures, models, and CSVs are written there at runtime and are not tracked by git. The full dataset CSV is also gitignored; see Step 3 below.

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/CADWRDeltaModeling/HAB_Phase1.git
cd HAB_Phase1
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate HabsEnv_2025
```

### 3. Add the data file

> **📦 Data Access Required**
> The full dataset is publicly available from the Environmental Data Initiative:
>
> **DOI: [10.6073/pasta/b4e180cb925266171285375a3e202635](https://doi.org/10.6073/pasta/b4e180cb925266171285375a3e202635)**
>
> Download the CSV and place it at:
> ```
> data/UCDavis_Data_for_Analysis2_220days_ToxinProdMicrocystis.csv
> ```
> See [`data/README.md`](data/README.md) for column descriptions and access notes.

### 4. Verify with the test suite (recommended)

Before running the full pipeline, verify the environment is correctly set up using the lightweight test suite:

```bash
cd tests
python TEST_01_ML4Models_Feature_Combination_Screening.py
python TEST_02_ANN_Feature_Combination_Evaluation.py
python TEST_03_ML5Models_Hyperparameter_Search.py
python TEST_04_ML_HAB_Full_CV_Analysis.py
python TEST_05_PDP_2D_Interaction_Plots.py
python TEST_06_LOSO_Spatial_Validation.py
```

Each script prints `✅ TEST_0X PASSED` on success. Tests use `data/example_data.csv` (10 rows, 5 sites) and complete in under 3 minutes total. See [`tests/TEST_REPORT.md`](tests/TEST_REPORT.md) for parameter reductions and known warnings.

> ⚠️ Test results have no scientific meaning — sample size is far too small.

| Script | Test Status |
|---|---|
| `TEST_01_ML4Models_Feature_Combination_Screening` | ✅ PASS |
| `TEST_02_ANN_Feature_Combination_Evaluation` | ✅ PASS |
| `TEST_03_ML5Models_Hyperparameter_Search` | ✅ PASS |
| `TEST_04_ML_HAB_Full_CV_Analysis` | ✅ PASS |
| `TEST_05_PDP_2D_Interaction_Plots` | ✅ PASS |
| `TEST_06_LOSO_Spatial_Validation` | ✅ PASS |

### 5. Run the full pipeline

Open and run the notebooks in order from the `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

Or run as scripts from the project root:

```bash
jupyter nbconvert --to script notebooks/01_ML4Models_Feature_Combination_Screening.ipynb --execute
```

All outputs (figures, tables, saved models) are written to `outputs/` subfolders automatically.

---

## Environment

Key dependencies (full spec in `environment.yml`):

| Package | Version |
|---|---|
| Python | 3.9 |
| scikit-learn | 1.5.0 |
| xgboost | 2.0.3 |
| tensorflow | 2.16.1 |
| matplotlib | 3.4 |
| seaborn | 0.11.2 |
| scipy | latest |
| pandas | latest |
| numpy | latest |
| joblib | latest |
| tqdm | latest |
| jupyter / notebook | latest |

---

## Data

The input dataset used in this study was derived from the following publicly available source:

> Lehman, P., Mayr, S., Liu, L., and Teh, S. (2021). *Cyanobacteria abundance, cyanotoxin concentration, and water quality data for the upper San Francisco Estuary, California, USA: 2014–2019.* Environmental Data Initiative.
> DOI: [10.6073/pasta/b4e180cb925266171285375a3e202635](https://doi.org/10.6073/pasta/b4e180cb925266171285375a3e202635)

The analysis-ready dataset (`UCDavis_Data_for_Analysis2_220days_ToxinProdMicrocystis.csv`) was prepared from this source by the study authors. It includes 220 sampling days across 14 fixed monitoring stations (2014–2019), filtered to days with toxin-producing *Microcystis* observations and complete physicochemical records. The full dataset is not included in this repository; see [`data/README.md`](data/README.md) for details.

An `example_data.csv` (10-row illustrative sample from 5 sites) is provided for structural reference and pipeline testing only. Has no scientific meaning.

---

## Citation

If you use this code or methodology in your research, please cite:

> [Citation to be added upon manuscript acceptance]

---

## Contact

**California Department of Water Resources — Delta Modeling Section**
GitHub: [https://github.com/CADWRDeltaModeling](https://github.com/CADWRDeltaModeling)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

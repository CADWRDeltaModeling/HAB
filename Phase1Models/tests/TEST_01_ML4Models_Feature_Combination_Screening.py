"""
TEST_01 — ML4Models Feature Combination Screening (smoke test)
==============================================================
Uses: data/example_data.csv  (10 rows, 5 sites)
Reduced parameters vs. production:
  • n_iter per model  : 500 / 2000 / 2000 / 2000  →  2
  • feature_combinations: manual list (5 combos)   →  1 combo only (FC1)
  • CV folds          : 5                           →  2

Expected runtime: < 30 seconds
Expected outputs (all under outputs/test_runs/01_FC_4Models_Results/):
  tables/results_with_hyperparams_combinations.csv   — 4 rows (one per model)
  models/best_models/best_model_combination_1.pkl
  models/configurations/best_model_config_combination_1.pkl

⚠️  Results have no scientific meaning — sample size is far too small.
"""

import os
import sys
import pathlib
import time

# ---------------------------------------------------------------------------
# Locate project root and import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import EXAMPLE_DATA_PATH, TEST_OUTPUT_DIRS

# ---------------------------------------------------------------------------
# Override production constants BEFORE any other imports consume them
# ---------------------------------------------------------------------------
DATA_PATH  = str(EXAMPLE_DATA_PATH)
OUTPUT_DIR = str(TEST_OUTPUT_DIRS["01"])

# ============================================================
# Everything below mirrors Script 01 exactly, with only the
# parameter reductions marked with  # TEST REDUCTION
# ============================================================
import numpy as np
import pandas as pd
from itertools import chain, combinations, product
from tqdm import tqdm
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def setup_output_dirs():
    subdirs = [
        "figures/model_comparison",
        "figures/confusion_matrix",
        "tables",
        "logs",
        "models/best_models",
        "models/configurations",
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


setup_output_dirs()

TABLES_DIR      = os.path.join(OUTPUT_DIR, "tables")
BEST_MODELS_DIR = os.path.join(OUTPUT_DIR, "models", "best_models")
CONFIGS_DIR     = os.path.join(OUTPUT_DIR, "models", "configurations")
RESULT_FILE     = os.path.join(TABLES_DIR, "results_with_hyperparams_combinations.csv")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
TARGET_COLUMN = "ToxinProdc_Microcystis_CC"
THRESHOLD     = 4000

feature_columns = [
    "Q", "Q_2w", "Q_1m",
    "SpecCond.", "Salinity", "NTU",
    "OrgC", "Total_OrgC",
    "NH4", "NO3", "OrgN",
    "PO4", "TP",
    "WaterTemp", "DO", "pH", "TDS",
]

missing_cols = [f for f in feature_columns if f not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

X = df[feature_columns].copy()
y = np.where(df[TARGET_COLUMN] <= THRESHOLD, 0, 1)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y,
)

global_scaler = StandardScaler()
global_scaler.fit(X_train_raw)
X_train_full_scaled = global_scaler.transform(X_train_raw)
X_test_full_scaled  = global_scaler.transform(X_test_raw)

# ---------------------------------------------------------------------------
# TEST REDUCTION: only FC1 instead of all 5 manual combos
# ---------------------------------------------------------------------------
feature_combinations = [
    ['Q', 'Q_1m', 'SpecCond.', 'OrgC', 'OrgN', 'PO4', 'WaterTemp', 'DO', 'pH'],
]

# ---------------------------------------------------------------------------
# Models — n_iter reduced to 2                          # TEST REDUCTION
# ---------------------------------------------------------------------------
models_config = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=200, solver="saga"),
        "params": [
            {"penalty": ["l1", "l2"], "C": [0.1, 1, 10], "solver": ["saga"],
             "class_weight": [None, "balanced"]},
        ],
        "n_iter": 2,    # TEST REDUCTION (production: 500)
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"], "max_depth": [3, 5],
            "min_samples_split": [2, 5], "class_weight": [None, "balanced"],
        },
        "n_iter": 2,    # TEST REDUCTION (production: 2000)
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [10, 20], "max_depth": [3, 5],
            "class_weight": [None, "balanced"],
        },
        "n_iter": 2,    # TEST REDUCTION (production: 2000)
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss",
                               verbosity=0, n_jobs=1),
        "params": {
            "n_estimators": [10, 20], "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
        },
        "n_iter": 2,    # TEST REDUCTION (production: 2000)
    },
}

# ---------------------------------------------------------------------------
# Results file init
# ---------------------------------------------------------------------------
if not os.path.exists(RESULT_FILE):
    pd.DataFrame(columns=[
        "Feature Combination Index", "Model", "Features", "Best Parameters",
        "Train Accuracy", "Test Accuracy", "Train Precision", "Test Precision",
        "Train Recall", "Test Recall", "Train F1", "Test F1",
    ]).to_csv(RESULT_FILE, index=False)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for idx, feature_combination in enumerate(
        tqdm(feature_combinations, desc="Feature Combinations"), start=1):

    start_time = time.time()
    feature_indices = [feature_columns.index(f) for f in feature_combination]
    X_train_scaled  = X_train_full_scaled[:, feature_indices]
    X_test_scaled   = X_test_full_scaled[:,  feature_indices]

    best_models, best_params = {}, {}

    for model_name, config in models_config.items():
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=2,           # TEST REDUCTION (production: 5)
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train_scaled, y_train)
        best_models[model_name] = search.best_estimator_
        best_params[model_name] = search.best_params_

    best_test_accuracy, best_model_name, best_model_obj = -1, None, None

    for model_name, model in best_models.items():
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred  = model.predict(X_test_scaled)

        result = {
            "Feature Combination Index": idx,
            "Model":           model_name,
            "Features":        feature_combination,
            "Best Parameters": best_params[model_name],
            "Train Accuracy":  accuracy_score(y_train, y_train_pred),
            "Test Accuracy":   accuracy_score(y_test,  y_test_pred),
            "Train Precision": precision_score(y_train, y_train_pred, zero_division=0),
            "Test Precision":  precision_score(y_test,  y_test_pred,  zero_division=0),
            "Train Recall":    recall_score(y_train, y_train_pred, zero_division=0),
            "Test Recall":     recall_score(y_test,  y_test_pred,  zero_division=0),
            "Train F1":        f1_score(y_train, y_train_pred, zero_division=0),
            "Test F1":         f1_score(y_test,  y_test_pred,  zero_division=0),
        }
        pd.DataFrame([result]).to_csv(RESULT_FILE, mode="a", header=False, index=False)

        if result["Test Accuracy"] > best_test_accuracy:
            best_test_accuracy = result["Test Accuracy"]
            best_model_name    = model_name
            best_model_obj     = model

    joblib.dump(best_model_obj,
                os.path.join(BEST_MODELS_DIR, f"best_model_combination_{idx}.pkl"))
    joblib.dump(
        {"combination_index": idx, "model_name": best_model_name,
         "features": feature_combination, "best_params": best_params[best_model_name],
         "feature_indices": feature_indices, "best_test_accuracy": best_test_accuracy,
         "scaler": global_scaler, "scaler_features": feature_columns},
        os.path.join(CONFIGS_DIR, f"best_model_config_combination_{idx}.pkl"),
    )
    print(f"  Best: {best_model_name} | Test Acc: {best_test_accuracy:.4f} "
          f"| Time: {time.time()-start_time:.1f}s")

# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------
print("\n--- TEST_01 VERIFICATION ---")
results_df = pd.read_csv(RESULT_FILE)
assert len(results_df) == 4, f"Expected 4 result rows (1 combo × 4 models), got {len(results_df)}"
assert os.path.exists(os.path.join(BEST_MODELS_DIR, "best_model_combination_1.pkl"))
assert os.path.exists(os.path.join(CONFIGS_DIR, "best_model_config_combination_1.pkl"))
print(f"  ✅  results CSV rows : {len(results_df)} (expected 4)")
print(f"  ✅  best_model pkl   : exists")
print(f"  ✅  best_config pkl  : exists")
print("TEST_01 PASSED\n")

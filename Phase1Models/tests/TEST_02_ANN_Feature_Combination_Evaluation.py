"""
TEST_02 — ANN Feature Combination Evaluation (smoke test)
==========================================================
Uses: data/example_data.csv  (10 rows, 5 sites)
Reduced parameters vs. production:
  • feature_combinations  : 5 combos   →  1 combo only (FC1)
  • n_param_combinations  : all         →  2
  • epochs                : 1000 (via early stop)  →  3
  • patience              : 30          →  2

Expected runtime: < 60 seconds
Expected outputs (all under outputs/test_runs/02_FC_ANN_Results/):
  tables/ANN_results_with_hyperparams_5_feature_Combinations.csv  — 1 row
  models/best_models/best_model_combination_1.h5 (or .keras)
  models/best_configurations/best_model_config_combination_1.pkl
  figures/dropout_impact/dropout_impact_combination_1.png

⚠️  Results have no scientific meaning — sample size is far too small.
"""

import os
import sys
import pathlib
import time
import itertools

# ---------------------------------------------------------------------------
# Locate project root and import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import EXAMPLE_DATA_PATH, TEST_OUTPUT_DIRS

DATA_PATH  = str(EXAMPLE_DATA_PATH)
OUTPUT_DIR = str(TEST_OUTPUT_DIRS["02"])

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_output_dirs():
    subdirs = [
        "figures/model_comparison", "figures/confusion_matrix",
        "figures/dropout_impact", "tables", "logs",
        "models/best_models", "models/best_configurations",
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


setup_output_dirs()

TABLES_DIR      = os.path.join(OUTPUT_DIR, "tables")
DROPOUT_FIG_DIR = os.path.join(OUTPUT_DIR, "figures", "dropout_impact")
BEST_MODELS_DIR = os.path.join(OUTPUT_DIR, "models", "best_models")
CONFIGS_DIR     = os.path.join(OUTPUT_DIR, "models", "best_configurations")

result_file = os.path.join(TABLES_DIR,
                           "ANN_results_with_hyperparams_5_feature_Combinations.csv")

if not os.path.exists(result_file):
    pd.DataFrame(columns=[
        "Feature Combination Index", "Model", "Features", "Best Parameters",
        "Train Accuracy", "Test Accuracy", "Train Precision", "Test Precision",
        "Train Recall", "Test Recall", "Train F1 Score", "Test F1 Score",
    ]).to_csv(result_file, index=False)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
TARGET_COLUMN = "ToxinProdc_Microcystis_CC"
THRESHOLD     = 4000

feature_columns = [
    "Q", "Q_2w", "Q_1m", "SpecCond.", "Salinity", "NTU",
    "OrgC", "Total_OrgC", "NH4", "NO3", "OrgN",
    "PO4", "TP", "WaterTemp", "DO", "pH", "TDS",
]

X = df[feature_columns].copy()
y = np.where(df[TARGET_COLUMN] <= THRESHOLD, 0, 1)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y,
)

scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train_full_scaled = scaler.transform(X_train_raw)
X_test_full_scaled  = scaler.transform(X_test_raw)

# ---------------------------------------------------------------------------
# TEST REDUCTION: 1 feature combination only
# ---------------------------------------------------------------------------
feature_combinations = [
    ['Q', 'Q_1m', 'SpecCond.', 'OrgC', 'OrgN', 'PO4', 'WaterTemp', 'DO', 'pH'],
]

# ---------------------------------------------------------------------------
# TEST REDUCTION: tiny ANN grid; epochs=3, patience=2
# ---------------------------------------------------------------------------
param_grid = {
    'hidden_layers': [2],
    'activation':    ['relu'],
    'neurons':       [16],
    'learning_rate': [0.001],
    'dropout_rate':  [0.1, 0.2],
}
n_param_combinations = 2   # TEST REDUCTION (production: all)
ANN_EPOCHS   = 3           # TEST REDUCTION (production: 1000 via early stop)
ANN_PATIENCE = 2           # TEST REDUCTION (production: 30)

all_params = list(itertools.islice(ParameterGrid(param_grid), n_param_combinations))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for combination_index, feature_combination in enumerate(feature_combinations, start=1):

    start_time = time.time()
    feature_indices = [feature_columns.index(f) for f in feature_combination]
    X_train_scaled  = X_train_full_scaled[:, feature_indices]
    X_test_scaled   = X_test_full_scaled[:,  feature_indices]

    best_score, best_model, best_config = -1, None, None
    dropout_impact = {}

    for idx, params in enumerate(all_params, start=1):

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(X_train_scaled.shape[1],)))
        model.add(keras.layers.Dense(params['neurons'], activation=params['activation']))
        model.add(keras.layers.Dropout(params['dropout_rate']))
        for _ in range(params['hidden_layers'] - 1):
            model.add(keras.layers.Dense(params['neurons'], activation=params['activation']))
            model.add(keras.layers.Dropout(params['dropout_rate']))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy', metrics=['accuracy'],
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ANN_PATIENCE, restore_best_weights=True,
        )

        model.fit(
            X_train_scaled, y_train,
            epochs=ANN_EPOCHS, batch_size=8,
            validation_split=0.2,
            callbacks=[early_stopping], verbose=0,
        )

        y_train_pred = (model.predict(X_train_scaled, verbose=0) >= 0.5).astype(int)
        y_test_pred  = (model.predict(X_test_scaled,  verbose=0) >= 0.5).astype(int)

        train_accuracy  = accuracy_score(y_train, y_train_pred)
        test_accuracy   = accuracy_score(y_test,  y_test_pred)

        dr = params['dropout_rate']
        if dr not in dropout_impact:
            dropout_impact[dr] = []
        dropout_impact[dr].append(test_accuracy)

        if test_accuracy > best_score:
            best_score  = test_accuracy
            best_model  = model
            best_config = {
                "Feature Combination Index": combination_index,
                "Model": "ANN", "Features": feature_combination,
                "Best Parameters": params,
                "Train Accuracy":  train_accuracy,
                "Test Accuracy":   test_accuracy,
                "Train Precision": precision_score(y_train, y_train_pred, zero_division=0),
                "Test Precision":  precision_score(y_test,  y_test_pred,  zero_division=0),
                "Train Recall":    recall_score(y_train, y_train_pred, zero_division=0),
                "Test Recall":     recall_score(y_test,  y_test_pred,  zero_division=0),
                "Train F1 Score":  f1_score(y_train, y_train_pred, zero_division=0),
                "Test F1 Score":   f1_score(y_test,  y_test_pred,  zero_division=0),
            }

            save_path = os.path.join(BEST_MODELS_DIR,
                                     f"best_model_combination_{combination_index}.keras")
            best_model.save(save_path)

    pd.DataFrame([best_config]).to_csv(result_file, mode='a', header=False, index=False)

    joblib.dump(
        {**best_config, "scaler": scaler, "scaler_features": feature_columns,
         "feature_indices": feature_indices},
        os.path.join(CONFIGS_DIR,
                     f"best_model_config_combination_{combination_index}.pkl"),
    )

    # Dropout impact plot
    plt.figure(figsize=(8, 4))
    for dr, accs in sorted(dropout_impact.items()):
        plt.plot(range(1, len(accs)+1), accs, label=f"Dropout {dr}")
    plt.xlabel("Configuration"); plt.ylabel("Test Accuracy")
    plt.title(f"Dropout Impact — Combination {combination_index} [TEST]")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(DROPOUT_FIG_DIR,
                             f"dropout_impact_combination_{combination_index}.png"))
    plt.close()

    print(f"  Combination {combination_index} done | Best acc: {best_score:.4f} "
          f"| Time: {time.time()-start_time:.1f}s")

# ---------------------------------------------------------------------------
# Verification checks
# ---------------------------------------------------------------------------
print("\n--- TEST_02 VERIFICATION ---")
results_df = pd.read_csv(result_file)
assert len(results_df) == 1, f"Expected 1 result row, got {len(results_df)}"

dropout_fig = os.path.join(DROPOUT_FIG_DIR, "dropout_impact_combination_1.png")
assert os.path.exists(dropout_fig), "Dropout figure missing"

config_pkl = os.path.join(CONFIGS_DIR, "best_model_config_combination_1.pkl")
assert os.path.exists(config_pkl), "Config pkl missing"

print(f"  ✅  results CSV rows  : {len(results_df)} (expected 1)")
print(f"  ✅  dropout figure    : exists")
print(f"  ✅  config pkl        : exists")
print("TEST_02 PASSED\n")

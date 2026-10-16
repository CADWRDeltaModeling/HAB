"""
TEST_04 — ML HAB Full CV Analysis (smoke test)
===============================================
Uses: data/example_data.csv  (10 rows, 5 sites)
Reduced parameters vs. production:
  • N_SPLITS × N_REPEATS : 5 × 10 = 50  →  2 × 1 = 2
  • ANN_EPOCHS           : 1000          →  3
  • ANN_PATIENCE         : 30            →  2
  • PERM_N_REPEATS       : 50            →  2
  • ANN_BATCH_SIZE       : 32            →  8

Expected runtime: < 2 minutes
Expected outputs (all under outputs/test_runs/04_ML5Models_Analysis_Results/):
  tables/metrics_summary.csv          — 5 rows
  tables/gap_penalty_log.csv
  tables/feature_importance.csv
  figures/model_comparison/           — bar plot PNGs
  figures/confusion_matrix/           — confusion matrix PNGs
  figures/uncertainty/                — uncertainty plot PNGs
  figures/density/                    — density plot PNGs
  figures/feature_importance/         — feature importance PNGs

⚠️  Results have no scientific meaning — sample size is far too small.
"""

import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import EXAMPLE_DATA_PATH, TEST_OUTPUT_DIRS

DATA_PATH  = str(EXAMPLE_DATA_PATH)
OUTPUT_DIR = str(TEST_OUTPUT_DIRS["04"])

os.environ["TF_DETERMINISTIC_OPS"] = "1"
import copy, gc, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import tensorflow as tf
import tensorflow.keras as keras
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

TARGET_COL   = "ToxinProdc_Microcystis_CC"
SITE_COL     = "Site_Name"
DATE_COL     = "Date"
THRESHOLD    = 4000
RANDOM_STATE = 42

FEATURES = ["Q", "Q_1m", "SpecCond.", "OrgC", "OrgN", "PO4", "WaterTemp", "DO", "pH"]

# Finalized ANN hyperparameters (from Script 03 output)
ANN_HIDDEN_LAYERS = 3
ANN_ACTIVATION    = "elu"
ANN_NEURONS       = 128
ANN_LEARNING_RATE = 0.0081
ANN_DROPOUT_RATE  = 0.25
ANN_EPOCHS        = 3    # TEST REDUCTION (production: 1000)
ANN_BATCH_SIZE    = 8    # TEST REDUCTION (production: 32)
ANN_VAL_SPLIT     = 0.2
ANN_PATIENCE      = 2    # TEST REDUCTION (production: 30)

N_SPLITS   = 2           # TEST REDUCTION (production: 5)
N_REPEATS  = 1           # TEST REDUCTION (production: 10)
PERM_N_REPEATS = 2       # TEST REDUCTION (production: 50)

DECISION_THR = 0.5
GAP_PRIMARY  = 0.10
GAP_FALLBACK = 0.12
PENALIZED_MODELS = {"Random Forest", "XGBoost"}

ALL_MODELS = ["Logistic Regression", "Decision Tree", "XGBoost", "Random Forest", "ANN"]
COLORS = {
    "Logistic Regression": "#9467BD", "Decision Tree": "#8C564B",
    "XGBoost": "#B85042", "Random Forest": "#1C7293", "ANN": "#E8A838",
}
BAND_PALETTE = {
    "Logistic Regression": "#C9A8E8", "Decision Tree": "#C4A49E",
    "XGBoost": "#E0948A", "Random Forest": "#6EB3CC", "ANN": "#F5D08E",
}
TRAIN_COLOR = "#607D8B"
TEST_COLOR  = "#2E4057"


def setup_output_dirs():
    subdirs = [
        "figures/model_comparison", "figures/confusion_matrix",
        "figures/uncertainty", "figures/density",
        "figures/feature_importance", "tables", "logs",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
    print("Output directory:", os.path.abspath(OUTPUT_DIR))


def savefig(fig, name):
    try:
        name   = name.replace("\\", "/")
        folder = os.path.abspath(
            os.path.join(OUTPUT_DIR, "figures", os.path.dirname(name)))
        os.makedirs(folder, exist_ok=True)
        filename = os.path.basename(name)
        fig.savefig(os.path.join(folder, filename + ".png"), dpi=72, bbox_inches="tight")
        print(f"  Saved → {filename}.png")
    except Exception as e:
        print(f"  ERROR saving '{name}': {e}")


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES].copy()
    y  = (df[TARGET_COL] > THRESHOLD).astype(int).values
    dates = df[DATE_COL].values if DATE_COL in df.columns else np.arange(len(y))
    print(f"Loaded {len(y)} samples | Risky: {y.sum()} / Not: {(y==0).sum()}")
    return X, y, dates, df


def classify_gap(gap):
    if gap <= GAP_PRIMARY:
        return "OK", False
    elif gap <= GAP_FALLBACK:
        return "FLAGGED", False
    else:
        return "HARD-PENALIZED", True


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            penalty="l1", C=1, solver="saga", max_iter=200,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, min_samples_split=5,
            class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=20, max_depth=2, learning_rate=0.1,
            verbosity=0, random_state=RANDOM_STATE, eval_metric="logloss"),
        "Random Forest": RandomForestClassifier(
            n_estimators=20, max_depth=5, class_weight="balanced",
            random_state=RANDOM_STATE),
    }


def build_ann(input_dim):
    tf.random.set_seed(RANDOM_STATE)
    m = keras.Sequential()
    m.add(keras.layers.Input(shape=(input_dim,)))
    m.add(keras.layers.Dense(ANN_NEURONS, activation=ANN_ACTIVATION))
    m.add(keras.layers.Dropout(ANN_DROPOUT_RATE))
    for _ in range(ANN_HIDDEN_LAYERS - 1):
        m.add(keras.layers.Dense(ANN_NEURONS, activation=ANN_ACTIVATION))
        m.add(keras.layers.Dropout(ANN_DROPOUT_RATE))
    m.add(keras.layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=ANN_LEARNING_RATE),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def run_cv(X, y, dates):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    train_scores = {m: {"accuracy":[],"precision":[],"recall":[],"f1":[]}
                    for m in ALL_MODELS}
    test_scores  = {m: {"accuracy":[],"precision":[],"recall":[],"f1":[]}
                    for m in ALL_MODELS}
    agg_preds  = {m: [] for m in ALL_MODELS}
    agg_probs  = {m: [] for m in ALL_MODELS}
    agg_true   = {m: [] for m in ALL_MODELS}
    agg_dates  = {m: [] for m in ALL_MODELS}
    repeat_probs = {m: {} for m in ALL_MODELS}
    penalty_log  = []

    for fold_i, (tr_i, te_i) in enumerate(rskf.split(X_np, y)):
        X_tr, X_te = X_np[tr_i], X_np[te_i]
        y_tr, y_te = y[tr_i],  y[te_i]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        sklearn_models = build_models()
        for mname, model in sklearn_models.items():
            model.fit(X_tr_sc, y_tr)
            ytp = model.predict(X_tr_sc)
            yep = model.predict(X_te_sc)
            try:
                proba = model.predict_proba(X_te_sc)[:, 1]
            except Exception:
                proba = yep.astype(float)

            for s, yp, yt in [(train_scores, ytp, y_tr), (test_scores, yep, y_te)]:
                s[mname]["accuracy"].append(accuracy_score(yt, yp))
                s[mname]["precision"].append(precision_score(yt, yp, zero_division=0))
                s[mname]["recall"].append(recall_score(yt, yp, zero_division=0))
                s[mname]["f1"].append(f1_score(yt, yp, zero_division=0))
            agg_preds[mname].extend(yep.tolist())
            agg_probs[mname].extend(proba.tolist())
            agg_true[mname].extend(y_te.tolist())
            agg_dates[mname].extend(dates[te_i].tolist())

            if mname in PENALIZED_MODELS:
                gap = accuracy_score(y_tr, ytp) - accuracy_score(y_te, yep)
                status, _ = classify_gap(gap)
                penalty_log.append({"Fold": fold_i, "Model": mname,
                                    "Gap": gap, "Gap_Status": status})

        # ANN fold
        ann = build_ann(X_tr_sc.shape[1])
        cb  = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                            restore_best_weights=True)
        ann.fit(X_tr_sc, y_tr, epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
                validation_split=0.2, callbacks=[cb], verbose=0)
        ann_prob_tr = ann.predict(X_tr_sc, verbose=0).ravel()
        ann_prob_te = ann.predict(X_te_sc, verbose=0).ravel()
        ann_pred_tr = (ann_prob_tr >= DECISION_THR).astype(int)
        ann_pred_te = (ann_prob_te >= DECISION_THR).astype(int)

        for s, yp, yt in [(train_scores, ann_pred_tr, y_tr),
                          (test_scores,  ann_pred_te, y_te)]:
            s["ANN"]["accuracy"].append(accuracy_score(yt, yp))
            s["ANN"]["precision"].append(precision_score(yt, yp, zero_division=0))
            s["ANN"]["recall"].append(recall_score(yt, yp, zero_division=0))
            s["ANN"]["f1"].append(f1_score(yt, yp, zero_division=0))
        agg_preds["ANN"].extend(ann_pred_te.tolist())
        agg_probs["ANN"].extend(ann_prob_te.tolist())
        agg_true["ANN"].extend(y_te.tolist())
        agg_dates["ANN"].extend(dates[te_i].tolist())
        keras.backend.clear_session(); gc.collect()

    return (train_scores, test_scores, agg_preds, agg_probs,
            agg_true, agg_dates, penalty_log, repeat_probs)


def save_penalty_log(penalty_log):
    df = pd.DataFrame(penalty_log) if penalty_log else pd.DataFrame(
        columns=["Fold","Model","Gap","Gap_Status"])
    df.to_csv(os.path.join(OUTPUT_DIR, "tables", "gap_penalty_log.csv"), index=False)
    print(f"  gap_penalty_log: {len(df)} rows")
    return df


def plot_model_comparison(train_scores, test_scores):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(ALL_MODELS)); bar_w = 0.35
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        tr_m = [np.mean(train_scores[m][metric]) for m in ALL_MODELS]
        te_m = [np.mean(test_scores[m][metric])  for m in ALL_MODELS]
        ax.bar(x - bar_w/2, tr_m, bar_w, color=[COLORS[m] for m in ALL_MODELS], alpha=0.85)
        ax.bar(x + bar_w/2, te_m, bar_w, color=TEST_COLOR, alpha=0.80)
        ax.set_xticks(x); ax.set_xticklabels(ALL_MODELS, rotation=20, ha="right")
        ax.set_ylim(0, 1.2); ax.set_ylabel(metric)
        ax.set_title(f"{metric} — {N_SPLITS}×{N_REPEATS} CV [TEST]")
        plt.tight_layout()
        savefig(fig, f"model_comparison/Fig1_{metric}")
        plt.close(fig)


def plot_confusion_matrices(agg_preds, agg_true):
    for mname in ALL_MODELS:
        cm = confusion_matrix(agg_true[mname], agg_preds[mname])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_title(f"CM {mname} [TEST]")
        plt.tight_layout()
        savefig(fig, f"confusion_matrix/Fig2_CM_{mname.replace(' ','_')}")
        plt.close(fig)


def plot_uncertainty(agg_probs, agg_true, agg_dates, repeat_probs):
    for mname in ALL_MODELS:
        probs = np.array(agg_probs[mname])
        true  = np.array(agg_true[mname])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(range(len(probs)), probs, c=true, cmap="RdYlGn_r", s=20, alpha=0.7)
        ax.axhline(0.5, color="black", linestyle="--")
        ax.set_title(f"Uncertainty — {mname} [TEST]")
        plt.tight_layout()
        savefig(fig, f"uncertainty/Fig3_uncertainty_{mname.replace(' ','_')}")
        plt.close(fig)


def plot_density(agg_probs, agg_true):
    for mname in ALL_MODELS:
        probs = np.array(agg_probs[mname])
        true  = np.array(agg_true[mname])
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls, label, color in [(0, "Non-Risky", "green"), (1, "Risky", "red")]:
            sub = probs[true == cls]
            if len(sub) > 1:
                try:
                    kde = gaussian_kde(sub)
                    xs  = np.linspace(0, 1, 100)
                    ax.plot(xs, kde(xs), color=color, label=label)
                except Exception:
                    pass
        ax.set_title(f"Density — {mname} [TEST]")
        ax.legend(); plt.tight_layout()
        savefig(fig, f"density/Fig4_density_{mname.replace(' ','_')}")
        plt.close(fig)


def get_feature_importances(X, y):
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_np)
    X_df = pd.DataFrame(X_sc, columns=FEATURES)

    fi_train = {}; fi_test = {}
    fi_train_std = {}; fi_test_std = {}
    method_label = {}

    sklearn_models = build_models()
    for mname, model in sklearn_models.items():
        model.fit(X_df, y)
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
            method_label[mname] = "Coefficient"
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            method_label[mname] = "Gini / Gain"
        else:
            imp = np.ones(len(FEATURES)) / len(FEATURES)
            method_label[mname] = "Uniform"
        total = imp.sum() or 1.0
        fi_train[mname] = imp / total
        fi_test[mname]  = imp / total
        fi_train_std[mname] = np.zeros(len(FEATURES))
        fi_test_std[mname]  = np.zeros(len(FEATURES))

    # ANN — permutation importance
    ann = build_ann(X_sc.shape[1])
    cb  = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                        restore_best_weights=True)
    ann.fit(X_sc, y, epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
            validation_split=0.2, callbacks=[cb], verbose=0)

    class _ANN:
        def fit(self, X, y): return self
        def predict(self, X): return (ann.predict(X, verbose=0) >= 0.5).astype(int).ravel()
        def score(self, X, y): return accuracy_score(y, self.predict(X))

    perm = permutation_importance(_ANN(), X_sc, y, n_repeats=PERM_N_REPEATS,
                                  random_state=RANDOM_STATE)
    raw  = perm.importances_mean
    total = np.abs(raw).sum() or 1.0
    fi_train["ANN"] = raw / total
    fi_test["ANN"]  = raw / total
    fi_train_std["ANN"] = perm.importances_std / total
    fi_test_std["ANN"]  = perm.importances_std / total
    method_label["ANN"] = "Permutation"
    keras.backend.clear_session(); gc.collect()

    return fi_train, fi_test, fi_train_std, fi_test_std, method_label


def plot_feature_importance(fi_train, fi_test, fi_train_std, fi_test_std, method_label):
    for mname in ALL_MODELS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, fi, std, lbl in [
            (axes[0], fi_train[mname], fi_train_std[mname], "Train"),
            (axes[1], fi_test[mname],  fi_test_std[mname],  "Test"),
        ]:
            ax.barh(FEATURES, fi, xerr=std, color=COLORS[mname], alpha=0.85)
            ax.set_title(f"{mname} — {lbl} [{method_label[mname]}] [TEST]")
        plt.tight_layout()
        savefig(fig, f"feature_importance/Fig5_FI_{mname.replace(' ','_')}")
        plt.close(fig)

    rows = []
    for mname in ALL_MODELS:
        for feat, tr_v, tr_s, te_v, te_s in zip(
                FEATURES, fi_train[mname], fi_train_std[mname],
                fi_test[mname],  fi_test_std[mname]):
            rows.append({"Model": mname, "Feature": feat,
                         "Train_Importance": tr_v, "Train_Std": tr_s,
                         "Test_Importance": te_v,  "Test_Std": te_s,
                         "Method": method_label[mname]})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "tables", "feature_importance.csv"), index=False)


def save_metrics_table(train_scores, test_scores, penalty_df):
    rows = []
    for m in ALL_MODELS:
        row = {"Model": m}
        for met in ["accuracy", "precision", "recall", "f1"]:
            row[f"Train_{met}_mean"] = np.mean(train_scores[m][met])
            row[f"Test_{met}_mean"]  = np.mean(test_scores[m][met])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "tables", "metrics_summary.csv"), index=False)


def main():
    set_global_seed(RANDOM_STATE)
    setup_output_dirs()
    X, y, dates, df = load_data()

    global _uncertainty_y_ref
    _uncertainty_y_ref = y

    (train_scores, test_scores, agg_preds, agg_probs,
     agg_true, agg_dates, penalty_log, repeat_probs) = run_cv(X, y, dates)

    penalty_df = save_penalty_log(penalty_log)
    save_metrics_table(train_scores, test_scores, penalty_df)
    plot_model_comparison(train_scores, test_scores)
    plot_confusion_matrices(agg_preds, agg_true)
    plot_uncertainty(agg_probs, agg_true, agg_dates, repeat_probs)
    plot_density(agg_probs, agg_true)
    fi_train, fi_test, fi_train_std, fi_test_std, method_label = \
        get_feature_importances(X, y)
    plot_feature_importance(fi_train, fi_test, fi_train_std, fi_test_std, method_label)

    print("\n--- TEST_04 VERIFICATION ---")
    for csv_name in ["metrics_summary.csv", "gap_penalty_log.csv", "feature_importance.csv"]:
        p = os.path.join(OUTPUT_DIR, "tables", csv_name)
        assert os.path.exists(p), f"{csv_name} missing"
        print(f"  ✅  {csv_name}")
    df_m = pd.read_csv(os.path.join(OUTPUT_DIR, "tables", "metrics_summary.csv"))
    assert len(df_m) == 5
    print(f"  ✅  metrics_summary rows : {len(df_m)} (expected 5)")

    fig_dirs = ["model_comparison", "confusion_matrix", "uncertainty",
                "density", "feature_importance"]
    for d in fig_dirs:
        figs = [f for f in os.listdir(os.path.join(OUTPUT_DIR, "figures", d))
                if f.endswith(".png")]
        print(f"  ✅  {d}: {len(figs)} PNG(s)")
    print("TEST_04 PASSED\n")


_uncertainty_y_ref = None

if __name__ == "__main__":
    main()

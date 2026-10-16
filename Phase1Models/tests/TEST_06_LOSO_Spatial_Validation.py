"""
TEST_06 — Leave-One-Site-Out (LOSO) Spatial Validation (smoke test)
====================================================================
Uses: data/example_data.csv  (10 rows, 5 sites)
Reduced parameters vs. production:
  • CV_N_SPLITS × CV_N_REPEATS : 5 × 10 = 50  →  2 × 1 = 2
  • BOOTSTRAP_N                : 1000          →  50
  • ANN_EPOCHS                 : 1000          →  3
  • ANN_PATIENCE               : 30            →  2

Note: with only 2 rows per site (1 train, 1 test per LOSO fold), metrics
will be degenerate — this test only verifies the pipeline completes and
produces the expected output files.

Expected runtime: < 3 minutes
Expected outputs (all under outputs/test_runs/06_LOSO_Results/):
  tables/loso_results.csv
  figures/loso_performance/LOSO_Logistic_Regression.png
  figures/loso_performance/LOSO_Decision_Tree.png
  figures/loso_performance/LOSO_XGBoost.png
  figures/loso_performance/LOSO_Random_Forest.png
  figures/loso_performance/LOSO_ANN.png

⚠️  Results have no scientific meaning — sample size is far too small.
"""

import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import EXAMPLE_DATA_PATH, TEST_OUTPUT_DIRS

DATA_PATH  = str(EXAMPLE_DATA_PATH)
OUTPUT_DIR = str(TEST_OUTPUT_DIRS["06"])

os.environ["TF_DETERMINISTIC_OPS"] = "1"
import gc, random, warnings, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.ioff()

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing   import StandardScaler

import tensorflow as tf
import tensorflow.keras as keras

warnings.filterwarnings("ignore")

TARGET_COL   = "ToxinProdc_Microcystis_CC"
SITE_COL     = "Site_Name"
DATE_COL     = "Date"
THRESHOLD    = 4000
RANDOM_STATE = 42

FEATURES = ["Q", "Q_1m", "SpecCond.", "OrgC", "OrgN", "PO4", "WaterTemp", "DO", "pH"]
METRICS  = ["accuracy", "precision", "recall", "f1"]

# TEST REDUCTIONS
CV_N_SPLITS  = 2      # production: 5
CV_N_REPEATS = 1      # production: 10
BOOTSTRAP_N  = 50     # production: 1000

ANN_HIDDEN_LAYERS = 3
ANN_ACTIVATION    = "elu"
ANN_NEURONS       = 128
ANN_LEARNING_RATE = 0.0081
ANN_DROPOUT_RATE  = 0.25
ANN_EPOCHS        = 3    # TEST REDUCTION (production: 1000)
ANN_BATCH_SIZE    = 8    # TEST REDUCTION (production: 32)
ANN_VAL_SPLIT     = 0.2
ANN_PATIENCE      = 2    # TEST REDUCTION (production: 30)

ALL_MODELS = ["Logistic Regression", "Decision Tree", "XGBoost",
              "Random Forest", "ANN"]
MODEL_COLOR = {
    "Logistic Regression": "#9467BD", "Decision Tree": "#8C564B",
    "Random Forest": "#1C7293", "XGBoost": "#B85042", "ANN": "#E8A838",
}
TRAIN_ALPHA = 0.85
TEST_ALPHA  = 0.55


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def savefig(fig, subfolder, filename):
    folder = os.path.abspath(os.path.join(OUTPUT_DIR, "figures", subfolder))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{filename}.png")
    try:
        fig.savefig(path, dpi=72, bbox_inches="tight")
        print(f"  Saved → {path}")
    except Exception as e:
        print(f"  [WARN] {e}")


def setup_dirs():
    for sub in ("figures/loso_performance", "tables", "logs"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


def safe_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def mean_minmax_from_scores(scores):
    arr       = np.asarray(scores, dtype=float)
    mean      = float(arr.mean())
    lower_err = mean - float(arr.min())
    upper_err = float(arr.max()) - mean
    return mean, lower_err, upper_err


def bootstrap_minmax(y_true, y_pred, n_bootstrap=BOOTSTRAP_N, seed=RANDOM_STATE):
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    boot_scores = {m: [] for m in METRICS}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_scores["accuracy"].append(accuracy_score(yt, yp))
        boot_scores["precision"].append(precision_score(yt, yp, zero_division=0))
        boot_scores["recall"].append(recall_score(yt, yp, zero_division=0))
        boot_scores["f1"].append(f1_score(yt, yp, zero_division=0))
    result = {}
    for m, vals in boot_scores.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) < 2:
            result[m] = {"mean": float(np.nan), "lower_err": 0.0, "upper_err": 0.0}
        else:
            mean = float(arr.mean())
            result[m] = {"mean": mean,
                         "lower_err": mean - float(arr.min()),
                         "upper_err": float(arr.max()) - mean}
    return result


def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing: {missing}")
    df["label"] = (df[TARGET_COL] > THRESHOLD).astype(int)
    sites = sorted(df[SITE_COL].unique())
    print(f"Loaded {len(df)} rows | Sites: {sites} | Risky: {df['label'].sum()}")
    return df, sites


def build_sklearn_models():
    return {
        "Logistic Regression": LogisticRegression(
            penalty="l1", C=1, solver="saga", max_iter=200,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, class_weight="balanced",
            random_state=RANDOM_STATE),
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


def run_loso(df, sites):
    """
    For each held-out site:
      - Train on all other sites (with internal repeated CV for training scores)
      - Test on held-out site predictions
    Returns dict: site → model → metric stats
    """
    results = {}

    for held_out in sites:
        print(f"\n  Hold-out site: {held_out}")
        train_df = df[df[SITE_COL] != held_out].copy()
        test_df  = df[df[SITE_COL] == held_out].copy()

        X_train = train_df[FEATURES].values
        y_train = train_df["label"].values
        X_test  = test_df[FEATURES].values
        y_test  = test_df["label"].values

        if len(X_test) == 0 or len(np.unique(y_train)) < 2:
            print(f"    Skipping {held_out}: insufficient data or single class in train")
            continue

        sc       = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_train)
        X_te_sc  = sc.transform(X_test)

        site_results = {}

        # ── sklearn models ──────────────────────────────────────────────────
        sklearn_models = build_sklearn_models()
        for mname, model in sklearn_models.items():

            # Internal CV for training score distribution
            cv_scores = {m: [] for m in METRICS}
            rskf = RepeatedStratifiedKFold(n_splits=min(CV_N_SPLITS, len(np.unique(y_train)),
                                                        len(y_train) // 2),
                                           n_repeats=CV_N_REPEATS,
                                           random_state=RANDOM_STATE)
            try:
                for tr_i, va_i in rskf.split(X_tr_sc, y_train):
                    m_clone = copy.deepcopy(model)
                    m_clone.fit(X_tr_sc[tr_i], y_train[tr_i])
                    preds = m_clone.predict(X_tr_sc[tr_i])
                    for metric, val in safe_metrics(y_train[tr_i], preds).items():
                        cv_scores[metric].append(val)
            except Exception as e:
                print(f"    [{mname}] CV warning: {e}")
                for metric in METRICS:
                    cv_scores[metric] = [0.0]

            # Fit on full training set, predict test
            try:
                model.fit(X_tr_sc, y_train)
                y_pred = model.predict(X_te_sc)
            except Exception as e:
                print(f"    [{mname}] Fit/predict warning: {e}")
                y_pred = np.zeros(len(y_test), dtype=int)

            train_stats = {m: mean_minmax_from_scores(cv_scores[m]) for m in METRICS}
            test_stats  = bootstrap_minmax(y_test, y_pred)
            site_results[mname] = {"train": train_stats, "test": test_stats,
                                   "y_true": y_test.tolist(), "y_pred": y_pred.tolist()}

        # ── ANN ─────────────────────────────────────────────────────────────
        ann_cv_scores = {m: [] for m in METRICS}
        rskf = RepeatedStratifiedKFold(
            n_splits=min(CV_N_SPLITS, len(np.unique(y_train)), max(2, len(y_train)//2)),
            n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE)
        try:
            for tr_i, va_i in rskf.split(X_tr_sc, y_train):
                ann_f = build_ann(X_tr_sc.shape[1])
                cb    = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=ANN_PATIENCE,
                                                      restore_best_weights=True)
                # val_split needs ≥2 samples; skip if too small
                if len(tr_i) >= 4:
                    ann_f.fit(X_tr_sc[tr_i], y_train[tr_i],
                              epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
                              validation_split=ANN_VAL_SPLIT, callbacks=[cb], verbose=0)
                else:
                    ann_f.fit(X_tr_sc[tr_i], y_train[tr_i],
                              epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE, verbose=0)
                preds = (ann_f.predict(X_tr_sc[tr_i], verbose=0) >= 0.5).astype(int).ravel()
                for metric, val in safe_metrics(y_train[tr_i], preds).items():
                    ann_cv_scores[metric].append(val)
                keras.backend.clear_session(); gc.collect()
        except Exception as e:
            print(f"    [ANN] CV warning: {e}")
            for metric in METRICS:
                ann_cv_scores[metric] = [0.0]

        # Final ANN fit on all training data
        try:
            ann_final = build_ann(X_tr_sc.shape[1])
            cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                               restore_best_weights=True)
            if len(X_tr_sc) >= 4:
                ann_final.fit(X_tr_sc, y_train, epochs=ANN_EPOCHS,
                              batch_size=ANN_BATCH_SIZE, validation_split=ANN_VAL_SPLIT,
                              callbacks=[cb], verbose=0)
            else:
                ann_final.fit(X_tr_sc, y_train, epochs=ANN_EPOCHS,
                              batch_size=ANN_BATCH_SIZE, verbose=0)
            ann_pred = (ann_final.predict(X_te_sc, verbose=0) >= 0.5).astype(int).ravel()
        except Exception as e:
            print(f"    [ANN] Final fit warning: {e}")
            ann_pred = np.zeros(len(y_test), dtype=int)
        keras.backend.clear_session(); gc.collect()

        ann_train_stats = {m: mean_minmax_from_scores(ann_cv_scores[m]) for m in METRICS}
        ann_test_stats  = bootstrap_minmax(y_test, ann_pred)
        site_results["ANN"] = {"train": ann_train_stats, "test": ann_test_stats,
                               "y_true": y_test.tolist(), "y_pred": ann_pred.tolist()}

        results[held_out] = site_results

    return results


def save_results_csv(results, sites):
    rows = []
    for site in sites:
        if site not in results:
            continue
        for mname in ALL_MODELS:
            if mname not in results[site]:
                continue
            row = {"Site": site, "Model": mname}
            for metric in METRICS:
                tr = results[site][mname]["train"].get(metric, (np.nan, 0, 0))
                te = results[site][mname]["test"].get(metric, {})
                row[f"Train_{metric}_mean"]  = tr[0]
                row[f"Train_{metric}_lower"] = tr[1]
                row[f"Train_{metric}_upper"] = tr[2]
                row[f"Test_{metric}_mean"]   = te.get("mean", np.nan)
                row[f"Test_{metric}_lower"]  = te.get("lower_err", 0)
                row[f"Test_{metric}_upper"]  = te.get("upper_err", 0)
            rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "tables", "loso_results.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return df


def plot_loso_figures(results, sites):
    valid_sites = [s for s in sites if s in results]
    x = np.arange(len(valid_sites))
    bar_w = 0.35

    for mname in ALL_MODELS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        color = MODEL_COLOR[mname]

        for ax, metric in zip(axes, METRICS):
            tr_means, tr_lo, tr_hi = [], [], []
            te_means, te_lo, te_hi = [], [], []

            for site in valid_sites:
                if mname not in results.get(site, {}):
                    tr_means.append(0); tr_lo.append(0); tr_hi.append(0)
                    te_means.append(0); te_lo.append(0); te_hi.append(0)
                    continue
                tr = results[site][mname]["train"].get(metric, (0, 0, 0))
                te = results[site][mname]["test"].get(metric, {})
                tr_means.append(tr[0]); tr_lo.append(tr[1]); tr_hi.append(tr[2])
                te_means.append(te.get("mean", 0))
                te_lo.append(te.get("lower_err", 0))
                te_hi.append(te.get("upper_err", 0))

            ax.bar(x - bar_w/2, tr_means, bar_w, color=color,
                   alpha=TRAIN_ALPHA, label="Train CV",
                   yerr=[tr_lo, tr_hi], capsize=3,
                   error_kw=dict(elinewidth=1, ecolor="black"))
            ax.bar(x + bar_w/2, te_means, bar_w, color=color,
                   alpha=TEST_ALPHA, label="Test",
                   yerr=[te_lo, te_hi], capsize=3,
                   error_kw=dict(elinewidth=1, ecolor="black"))

            ax.set_xticks(x)
            ax.set_xticklabels(valid_sites, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1.25)
            ax.set_ylabel(metric.capitalize(), fontsize=9)
            ax.set_title(metric.capitalize(), fontsize=10)
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle(f"LOSO — {mname} [TEST]", fontsize=12, y=1.01)
        plt.tight_layout()
        safe = mname.replace(" ", "_")
        savefig(fig, "loso_performance", f"LOSO_{safe}")
        plt.close(fig); gc.collect()


def main():
    set_global_seed(RANDOM_STATE)
    setup_dirs()
    df, sites = load_data()
    results   = run_loso(df, sites)
    results_df = save_results_csv(results, sites)
    plot_loso_figures(results, sites)

    print("\n--- TEST_06 VERIFICATION ---")
    csv_path = os.path.join(OUTPUT_DIR, "tables", "loso_results.csv")
    assert os.path.exists(csv_path), "loso_results.csv missing"
    df_r = pd.read_csv(csv_path)
    print(f"  ✅  loso_results.csv rows: {len(df_r)}")

    fig_dir = os.path.join(OUTPUT_DIR, "figures", "loso_performance")
    pngs    = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    assert len(pngs) == len(ALL_MODELS), \
        f"Expected {len(ALL_MODELS)} LOSO figures, got {len(pngs)}"
    print(f"  ✅  LOSO figures: {len(pngs)} (expected {len(ALL_MODELS)})")
    print("TEST_06 PASSED\n")


if __name__ == "__main__":
    main()

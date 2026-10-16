"""
TEST_03 — ML5Models Hyperparameter Search (smoke test)
=======================================================
Uses: data/example_data.csv  (10 rows, 5 sites)
Reduced parameters vs. production:
  • TUNE_N_ITER_*     : 500 / 2500 each  →  2
  • TUNE_CV_FOLDS     : 5               →  2
  • SCORER_CV_FOLDS   : 3               →  2
  • N_SPLITS × N_REPEATS : 5 × 10 = 50  →  2 × 1 = 2
  • ANN_EPOCHS        : 1000            →  3
  • ANN_PATIENCE      : 30              →  2
  • ANN_GRID          : full            →  1 config only

Expected runtime: < 2 minutes
Expected outputs (all under outputs/test_runs/03_HS_CV_5ML/):
  tables/hyperparameter_tuning.csv
  tables/metrics_summary.csv
  tables/ann_grid_search_results.csv
  figures/model_comparison/Fig1_model_comparison_*.png
  figures/confusion_matrix/Fig2_CM_*.png

⚠️  Results have no scientific meaning — sample size is far too small.
"""

import os
import sys
import pathlib

# ---------------------------------------------------------------------------
# Locate project root and import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import EXAMPLE_DATA_PATH, TEST_OUTPUT_DIRS

DATA_PATH  = str(EXAMPLE_DATA_PATH)
OUTPUT_DIR = str(TEST_OUTPUT_DIRS["03"])

# ---------------------------------------------------------------------------
# Patch constants BEFORE the production module logic runs
# TEST REDUCTIONS listed inline
# ---------------------------------------------------------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import gc, random, warnings, itertools
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
from sklearn.model_selection import (
    RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, make_scorer
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras as keras

warnings.filterwarnings("ignore")

TARGET_COL   = "ToxinProdc_Microcystis_CC"
SITE_COL     = "Site_Name"
DATE_COL     = "Date"
THRESHOLD    = 4000
RANDOM_STATE = 42

FEATURES = ["Q", "Q_1m", "SpecCond.", "OrgC", "OrgN", "PO4", "WaterTemp", "DO", "pH"]

# TEST REDUCTIONS
GAP_TOLERANCE          = 0.10
GAP_FALLBACK_TOLERANCE = 0.12
HARD_PENALTY_SCORE     = -999.0
SCORER_CV_FOLDS = 2      # TEST REDUCTION (production: 3)
TUNE_CV_FOLDS   = 2      # TEST REDUCTION (production: 5)
TUNE_N_ITER_LR  = 2      # TEST REDUCTION (production: 500)
TUNE_N_ITER_DT  = 2      # TEST REDUCTION (production: 2500)
TUNE_N_ITER_RF  = 2      # TEST REDUCTION (production: 2500)
TUNE_N_ITER_XGB = 2      # TEST REDUCTION (production: 2500)
N_SPLITS  = 2            # TEST REDUCTION (production: 5)
N_REPEATS = 1            # TEST REDUCTION (production: 10)
DECISION_THR = 0.5
_ACTIVE_GAP_TOLERANCE = GAP_TOLERANCE

ALL_MODELS = ["Logistic Regression", "Decision Tree", "XGBoost", "Random Forest", "ANN"]
COLORS = {
    "Logistic Regression": "#9467BD", "Decision Tree": "#8C564B",
    "XGBoost": "#B85042", "Random Forest": "#1C7293", "ANN": "#E8A838",
}
TRAIN_COLOR = "#607D8B"
TEST_COLOR  = "#2E4057"

# ANN TEST REDUCTIONS
ANN_EPOCHS      = 3   # TEST REDUCTION (production: 1000)
ANN_BATCH_SIZE  = 8   # TEST REDUCTION (production: 32)
ANN_VAL_SPLIT   = 0.2
ANN_PATIENCE    = 2   # TEST REDUCTION (production: 30)
ANN_INNER_FOLDS = 2   # TEST REDUCTION (production: 3)
ANN_GRID = {
    "hidden_layers":  [2],
    "activations":    ["relu"],
    "neurons":        [16],
    "learning_rates": [0.001],
    "dropout_rates":  [0.1],
}


def setup_output_dirs():
    subdirs = ["figures/model_comparison", "figures/confusion_matrix", "tables", "logs"]
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
        png_path = os.path.join(folder, filename + ".png")
        fig.savefig(png_path, dpi=72, bbox_inches="tight")
        print(f"  Saved → {png_path}")
    except Exception as e:
        print(f"  ERROR saving '{name}': {e}")


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[FEATURES].copy()
    y = (df[TARGET_COL] > THRESHOLD).astype(int).values
    print(f"Loaded {len(y)} samples | Risky: {y.sum()} / Non-Risky: {(y==0).sum()}")
    return X, y


# ── Scorers ───────────────────────────────────────────────────────────────────
def _inner_cv_accuracy(estimator, X, y):
    skf = StratifiedKFold(n_splits=SCORER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tr_list, te_list = [], []
    try:
        for tr_i, te_i in skf.split(X, y):
            est = clone(estimator)
            est.fit(X[tr_i], y[tr_i])
            tr_list.append(accuracy_score(y[tr_i], est.predict(X[tr_i])))
            te_list.append(accuracy_score(y[te_i], est.predict(X[te_i])))
        return tr_list, te_list, False
    except Exception:
        return [], [], True


def _plain_accuracy(estimator, X, y):
    _, te, err = _inner_cv_accuracy(estimator, X, y)
    return 0.0 if err or not te else np.mean(te)


def _hard_penalty_accuracy(estimator, X, y):
    tr, te, err = _inner_cv_accuracy(estimator, X, y)
    if err or not tr:
        return HARD_PENALTY_SCORE
    gap = np.mean(tr) - np.mean(te)
    return HARD_PENALTY_SCORE if gap > _ACTIVE_GAP_TOLERANCE else np.mean(te)


LR_SCORING    = _plain_accuracy
DT_SCORING    = _plain_accuracy
RFXGB_SCORING = _hard_penalty_accuracy


# ── Hyperparameter grids (minimal for test) ───────────────────────────────────
LR_PARAM_DIST = [
    {"penalty": ["l1", "l2"], "C": [0.1, 1.0], "solver": ["saga"],
     "class_weight": ["balanced", None], "max_iter": [200]},
]
DT_PARAM_DIST = {
    "criterion": ["gini", "entropy"], "max_depth": [3, 5],
    "class_weight": ["balanced", None],
}
RF_PARAM_DIST = [
    {"n_estimators": [10, 20], "max_depth": [3, 5],
     "class_weight": ["balanced"], "bootstrap": [True], "max_samples": [None],
     "oob_score": [False]},
]
XGB_PARAM_DIST = {
    "learning_rate": [0.05, 0.1], "n_estimators": [10, 20],
    "max_depth": [2, 3], "subsample": [0.8],
}


# ── Model builders ────────────────────────────────────────────────────────────
def build_lr(p=None):
    d = dict(penalty="l1", C=1.0, solver="saga", max_iter=200,
             class_weight="balanced", random_state=RANDOM_STATE)
    if p:
        d.update(p)
    if d.get("penalty") != "elasticnet":
        d.pop("l1_ratio", None)
    return LogisticRegression(**d)


def build_dt(p=None):
    d = dict(criterion="entropy", max_depth=5, min_samples_split=5,
             class_weight="balanced", random_state=RANDOM_STATE)
    if p:
        d.update(p)
    return DecisionTreeClassifier(**d)


def build_rf(p=None):
    d = dict(n_estimators=10, max_depth=5, class_weight="balanced",
             bootstrap=True, random_state=RANDOM_STATE)
    if p:
        d.update(p)
    d.pop("max_samples", None)
    d.pop("oob_score", None)
    return RandomForestClassifier(**d)


def build_xgb(p=None):
    d = dict(n_estimators=10, max_depth=2, learning_rate=0.1,
             verbosity=0, random_state=RANDOM_STATE, eval_metric="logloss")
    if p:
        d.update(p)
    return XGBClassifier(**d)


# ── ANN builder ───────────────────────────────────────────────────────────────
def build_ann(input_dim, hidden_layers=2, neurons=16,
              activation="relu", learning_rate=0.001, dropout_rate=0.1):
    tf.random.set_seed(RANDOM_STATE)
    m = keras.Sequential()
    m.add(keras.layers.Input(shape=(input_dim,)))
    m.add(keras.layers.Dense(neurons, activation=activation))
    m.add(keras.layers.Dropout(dropout_rate))
    for _ in range(hidden_layers - 1):
        m.add(keras.layers.Dense(neurons, activation=activation))
        m.add(keras.layers.Dropout(dropout_rate))
    m.add(keras.layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ── Hyperparameter search ─────────────────────────────────────────────────────
def run_hyperparameter_search(X_scaled, y):
    tuned = {}
    tuning_rows = []
    ann_rows    = []

    # LR
    search = RandomizedSearchCV(
        LogisticRegression(random_state=RANDOM_STATE), LR_PARAM_DIST,
        n_iter=TUNE_N_ITER_LR, cv=TUNE_CV_FOLDS, scoring=LR_SCORING,
        n_jobs=-1, random_state=RANDOM_STATE, error_score=0.0,
    )
    search.fit(X_scaled, y)
    tuned["Logistic Regression"] = search.best_params_
    tuning_rows.append({"Model": "Logistic Regression",
                        "Best_Score": search.best_score_,
                        "Best_Params": search.best_params_})

    # DT
    search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE), DT_PARAM_DIST,
        n_iter=TUNE_N_ITER_DT, cv=TUNE_CV_FOLDS, scoring=DT_SCORING,
        n_jobs=-1, random_state=RANDOM_STATE, error_score=0.0,
    )
    search.fit(X_scaled, y)
    tuned["Decision Tree"] = search.best_params_
    tuning_rows.append({"Model": "Decision Tree",
                        "Best_Score": search.best_score_,
                        "Best_Params": search.best_params_})

    # RF
    global _ACTIVE_GAP_TOLERANCE
    _ACTIVE_GAP_TOLERANCE = GAP_TOLERANCE
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE), RF_PARAM_DIST,
        n_iter=TUNE_N_ITER_RF, cv=TUNE_CV_FOLDS, scoring=RFXGB_SCORING,
        n_jobs=-1, random_state=RANDOM_STATE, error_score=HARD_PENALTY_SCORE,
    )
    search.fit(X_scaled, y)
    tuned["Random Forest"] = search.best_params_
    tuning_rows.append({"Model": "Random Forest",
                        "Best_Score": search.best_score_,
                        "Best_Params": search.best_params_})

    # XGB
    _ACTIVE_GAP_TOLERANCE = GAP_TOLERANCE
    search = RandomizedSearchCV(
        XGBClassifier(verbosity=0, random_state=RANDOM_STATE, eval_metric="logloss"),
        XGB_PARAM_DIST,
        n_iter=TUNE_N_ITER_XGB, cv=TUNE_CV_FOLDS, scoring=RFXGB_SCORING,
        n_jobs=-1, random_state=RANDOM_STATE, error_score=HARD_PENALTY_SCORE,
    )
    search.fit(X_scaled, y)
    tuned["XGBoost"] = search.best_params_
    tuning_rows.append({"Model": "XGBoost",
                        "Best_Score": search.best_score_,
                        "Best_Params": search.best_params_})

    # ANN manual grid
    input_dim = X_scaled.shape[1]
    best_ann_score, best_ann_cfg = -1, None
    skf = StratifiedKFold(n_splits=ANN_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for cfg in itertools.product(
        ANN_GRID["hidden_layers"], ANN_GRID["neurons"],
        ANN_GRID["activations"], ANN_GRID["learning_rates"], ANN_GRID["dropout_rates"]
    ):
        hl, neu, act, lr, dr = cfg
        fold_scores = []
        for tr_i, te_i in skf.split(X_scaled, y):
            m = build_ann(input_dim, hl, neu, act, lr, dr)
            cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                               restore_best_weights=True)
            m.fit(X_scaled[tr_i], y[tr_i], epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
                  validation_split=0.2, callbacks=[cb], verbose=0)
            fold_scores.append(accuracy_score(y[te_i],
                               (m.predict(X_scaled[te_i], verbose=0) >= 0.5).astype(int)))
            keras.backend.clear_session(); gc.collect()
        mean_score = np.mean(fold_scores)
        ann_rows.append({"hidden_layers": hl, "neurons": neu, "activation": act,
                         "learning_rate": lr, "dropout_rate": dr,
                         "mean_val_accuracy": mean_score})
        if mean_score > best_ann_score:
            best_ann_score = mean_score
            best_ann_cfg   = {"hidden_layers": hl, "neurons": neu, "activation": act,
                              "learning_rate": lr, "dropout_rate": dr}

    tuned["ANN"] = best_ann_cfg
    tuning_rows.append({"Model": "ANN", "Best_Score": best_ann_score,
                        "Best_Params": best_ann_cfg})

    pd.DataFrame(tuning_rows).to_csv(
        os.path.join(OUTPUT_DIR, "tables", "hyperparameter_tuning.csv"), index=False)
    pd.DataFrame(ann_rows).to_csv(
        os.path.join(OUTPUT_DIR, "tables", "ann_grid_search_results.csv"), index=False)

    return tuned


# ── CV loop ───────────────────────────────────────────────────────────────────
def run_cv(X, y, tuned_params):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)
    train_scores = {m: {"accuracy":[], "balanced_accuracy":[], "precision":[],
                        "recall":[], "f1":[]} for m in ALL_MODELS}
    test_scores  = {m: {"accuracy":[], "balanced_accuracy":[], "precision":[],
                        "recall":[], "f1":[]} for m in ALL_MODELS}
    agg_preds = {m: [] for m in ALL_MODELS}
    agg_true  = {m: [] for m in ALL_MODELS}
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    for fold_i, (tr_i, te_i) in enumerate(rskf.split(X_np, y)):
        X_tr, X_te = X_np[tr_i], X_np[te_i]
        y_tr, y_te = y[tr_i], y[te_i]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        def _score(mname, model):
            ytp = model.predict(X_tr_sc)
            yep = model.predict(X_te_sc)
            for s, yp, yt in [(train_scores, ytp, y_tr), (test_scores, yep, y_te)]:
                s[mname]["accuracy"].append(accuracy_score(yt, yp))
                s[mname]["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
                s[mname]["precision"].append(precision_score(yt, yp, zero_division=0))
                s[mname]["recall"].append(recall_score(yt, yp, zero_division=0))
                s[mname]["f1"].append(f1_score(yt, yp, zero_division=0))
            agg_preds[mname].extend(yep.tolist())
            agg_true[mname].extend(y_te.tolist())

        _score("Logistic Regression", build_lr(tuned_params["Logistic Regression"]).fit(X_tr_sc, y_tr))
        _score("Decision Tree",       build_dt(tuned_params["Decision Tree"]).fit(X_tr_sc, y_tr))
        _score("Random Forest",       build_rf(tuned_params["Random Forest"]).fit(X_tr_sc, y_tr))
        _score("XGBoost",             build_xgb(tuned_params["XGBoost"]).fit(X_tr_sc, y_tr))

        # ANN fold
        ann_cfg = tuned_params["ANN"]
        ann = build_ann(X_tr_sc.shape[1], **ann_cfg)
        cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                           restore_best_weights=True)
        ann.fit(X_tr_sc, y_tr, epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
                validation_split=0.2, callbacks=[cb], verbose=0)

        def _ann_predict(X_sc):
            return (ann.predict(X_sc, verbose=0) >= DECISION_THR).astype(int).ravel()

        _score("ANN", type("_", (), {"predict": lambda self, X: _ann_predict(X)})())
        keras.backend.clear_session(); gc.collect()

    return train_scores, test_scores, agg_preds, agg_true


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_model_comparison(train_scores, test_scores):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(ALL_MODELS))
    bar_w = 0.35
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        tr_m = [np.mean(train_scores[m][metric]) for m in ALL_MODELS]
        te_m = [np.mean(test_scores[m][metric])  for m in ALL_MODELS]
        ax.bar(x - bar_w/2, tr_m, bar_w, label="Train",
               color=[COLORS[m] for m in ALL_MODELS], alpha=0.85)
        ax.bar(x + bar_w/2, te_m, bar_w, label="Test",
               color=TEST_COLOR, alpha=0.80)
        ax.set_xticks(x); ax.set_xticklabels(ALL_MODELS, rotation=20, ha="right")
        ax.set_ylim(0, 1.2); ax.set_ylabel(metric); ax.legend()
        ax.set_title(f"{metric.capitalize()} — {N_SPLITS}×{N_REPEATS} CV [TEST]")
        plt.tight_layout()
        savefig(fig, f"model_comparison/Fig1_{metric}")
        plt.close(fig)


def plot_confusion_matrices(agg_preds, agg_true):
    from sklearn.metrics import confusion_matrix as cm_fn
    for mname in ALL_MODELS:
        cm = cm_fn(agg_true[mname], agg_preds[mname])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_title(f"CM {mname} [TEST]")
        plt.tight_layout()
        savefig(fig, f"confusion_matrix/Fig2_CM_{mname.replace(' ','_')}")
        plt.close(fig)


def save_metrics_table(train_scores, test_scores):
    rows = []
    for m in ALL_MODELS:
        row = {"Model": m}
        for met in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
            row[f"Train_{met}_mean"] = np.mean(train_scores[m][met])
            row[f"Test_{met}_mean"]  = np.mean(test_scores[m][met])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "tables", "metrics_summary.csv"), index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_global_seed(RANDOM_STATE)
    setup_output_dirs()
    X, y = load_data()
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_np)

    tuned = run_hyperparameter_search(X_sc, y)
    train_scores, test_scores, agg_preds, agg_true = run_cv(X, y, tuned)
    save_metrics_table(train_scores, test_scores)
    plot_model_comparison(train_scores, test_scores)
    plot_confusion_matrices(agg_preds, agg_true)

    # Verification
    print("\n--- TEST_03 VERIFICATION ---")
    metrics_csv = os.path.join(OUTPUT_DIR, "tables", "metrics_summary.csv")
    tuning_csv  = os.path.join(OUTPUT_DIR, "tables", "hyperparameter_tuning.csv")
    assert os.path.exists(metrics_csv), "metrics_summary.csv missing"
    assert os.path.exists(tuning_csv),  "hyperparameter_tuning.csv missing"
    df_m = pd.read_csv(metrics_csv)
    assert len(df_m) == 5, f"Expected 5 model rows, got {len(df_m)}"
    print(f"  ✅  metrics_summary rows : {len(df_m)} (expected 5)")
    print(f"  ✅  hyperparameter_tuning: exists")
    figs = [f for f in os.listdir(os.path.join(OUTPUT_DIR, "figures", "model_comparison"))
            if f.endswith(".png")]
    print(f"  ✅  model_comparison figs: {len(figs)}")
    print("TEST_03 PASSED\n")


if __name__ == "__main__":
    main()

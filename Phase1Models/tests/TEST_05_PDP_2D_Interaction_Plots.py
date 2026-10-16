"""
TEST_05 — PDP Interaction Plots (smoke test)
=============================================
Uses: data/example_data.csv  (10 rows, 5 sites)
Reduced parameters vs. production:
  • GRID_RESOLUTION  : 50  →  5
  • ANN_EPOCHS       : 1000 →  3
  • ANN_PATIENCE     : 30   →  2

Expected runtime: < 2 minutes
Expected outputs (all under outputs/test_runs/05_PDP_Results/):
  figures/pdp_1d/PDP1D_WaterTemp.png
  figures/pdp_1d/PDP1D_pH.png
  figures/pdp_1d/PDP1D_OrgN.png
  figures/pdp_1d/PDP1D_combined.png
  figures/pdp_2d/PDP2D_WaterTemp_x_pH.png
  figures/pdp_2d/PDP2D_WaterTemp_x_OrgN.png
  figures/pdp_2d/PDP2D_pH_x_OrgN.png
  figures/pdp_2d/PDP2D_combined.png

⚠️  Results have no scientific meaning — sample size is far too small.
"""

import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config import EXAMPLE_DATA_PATH, TEST_OUTPUT_DIRS

DATA_PATH  = str(EXAMPLE_DATA_PATH)
OUTPUT_DIR = str(TEST_OUTPUT_DIRS["05"])

import gc, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.ioff()

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.inspection      import partial_dependence
from sklearn.preprocessing   import StandardScaler

import tensorflow as tf
import tensorflow.keras as keras

warnings.filterwarnings("ignore")

TARGET_COL   = "ToxinProdc_Microcystis_CC"
THRESHOLD    = 4000
RANDOM_STATE = 42

FEATURES   = ["Q", "Q_1m", "SpecCond.", "OrgC", "OrgN", "PO4", "WaterTemp", "DO", "pH"]
TOP_VARS   = ["WaterTemp", "pH", "OrgN"]
TOP_PAIRS  = [("WaterTemp", "pH"), ("WaterTemp", "OrgN"), ("pH", "OrgN")]

GRID_RESOLUTION = 5   # TEST REDUCTION (production: 50)

# ANN hyperparameters — match production values
ANN_HIDDEN_LAYERS = 3
ANN_ACTIVATION    = "elu"
ANN_NEURONS       = 128
ANN_LEARNING_RATE = 0.0081
ANN_DROPOUT_RATE  = 0.25
ANN_EPOCHS        = 3    # TEST REDUCTION (production: 1000)
ANN_BATCH_SIZE    = 8    # TEST REDUCTION (production: 32)
ANN_VAL_SPLIT     = 0.2
ANN_PATIENCE      = 2    # TEST REDUCTION (production: 30)

ALL_MODELS = ["Logistic Regression", "Decision Tree", "XGBoost", "Random Forest", "ANN"]
COLORS = {
    "Logistic Regression": "#9467BD", "Decision Tree": "#8C564B",
    "XGBoost": "#B85042", "Random Forest": "#1C7293", "ANN": "#E8A838",
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_dirs():
    for sub in ("figures/pdp_1d", "figures/pdp_2d"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


def savefig(fig, subfolder, filename):
    folder = os.path.abspath(os.path.join(OUTPUT_DIR, "figures", subfolder))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{filename}.png")
    try:
        fig.savefig(path, dpi=72, bbox_inches="tight")
        print(f"  Saved → {path}")
    except Exception as e:
        print(f"  [WARN] Could not save {path}: {e}")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["label"] = (df[TARGET_COL] >= THRESHOLD).astype(int)
    df = df.dropna(subset=FEATURES + ["label"])
    X  = df[FEATURES].copy()
    y  = df["label"].values
    print(f"Loaded {len(y)} samples | Risky: {y.sum()}")
    return X, y


def build_lr():
    return LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=200,
                              class_weight="balanced", random_state=RANDOM_STATE)

def build_dt():
    return DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                  random_state=RANDOM_STATE)

def build_xgb():
    return XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.1,
                         verbosity=0, random_state=RANDOM_STATE, eval_metric="logloss")

def build_rf():
    return RandomForestClassifier(n_estimators=20, max_depth=5,
                                  class_weight="balanced", random_state=RANDOM_STATE)


class ANNWrapper:
    def __init__(self, input_dim, seed=RANDOM_STATE):
        self.input_dim = input_dim
        self.seed      = seed
        self.model_    = None

    def _build(self):
        tf.random.set_seed(self.seed)
        m = keras.Sequential()
        m.add(keras.layers.Input(shape=(self.input_dim,)))
        m.add(keras.layers.Dense(ANN_NEURONS, activation=ANN_ACTIVATION))
        m.add(keras.layers.Dropout(ANN_DROPOUT_RATE, seed=self.seed))
        for _ in range(ANN_HIDDEN_LAYERS - 1):
            m.add(keras.layers.Dense(ANN_NEURONS, activation=ANN_ACTIVATION))
            m.add(keras.layers.Dropout(ANN_DROPOUT_RATE, seed=self.seed))
        m.add(keras.layers.Dense(1, activation="sigmoid"))
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=ANN_LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
        return m

    def fit(self, X, y):
        self.model_ = self._build()
        cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                           restore_best_weights=True, verbose=0)
        self.model_.fit(X, y, epochs=ANN_EPOCHS, batch_size=ANN_BATCH_SIZE,
                        validation_split=ANN_VAL_SPLIT, callbacks=[cb], verbose=0)
        return self

    def predict_proba_positive(self, X):
        return self.model_.predict(X, verbose=0).ravel()


def fit_all_models(X_sc, y):
    print("\nFitting models on full dataset ...")
    X_df = pd.DataFrame(X_sc, columns=FEATURES)
    fitted = {}
    for name, builder in [("Logistic Regression", build_lr), ("Decision Tree", build_dt),
                           ("XGBoost", build_xgb), ("Random Forest", build_rf)]:
        fitted[name] = builder().fit(X_df, y)
        print(f"  Fitted {name}")
    ann = ANNWrapper(X_sc.shape[1])
    ann.fit(X_sc, y)
    fitted["ANN"] = ann
    keras.backend.clear_session(); gc.collect()
    return fitted


def unscale_grid(grid_sc, feature, scaler):
    idx   = FEATURES.index(feature)
    mean_ = scaler.mean_[idx]
    scale_ = scaler.scale_[idx]
    return grid_sc * scale_ + mean_


def pdp_1d_sklearn(model, X_df, feature):
    feat_idx = list(X_df.columns).index(feature)
    result   = partial_dependence(
        model, X_df, features=[feat_idx],
        kind="average", grid_resolution=GRID_RESOLUTION,
        response_method="predict_proba",
    )
    grid_vals = result["grid_values"][0]
    avg_preds = result["average"][0]
    if avg_preds.ndim > 1:
        avg_preds = avg_preds[1]
    return grid_vals, avg_preds


def pdp_1d_ann(ann, X_sc, feature):
    feat_idx  = FEATURES.index(feature)
    feat_vals = np.linspace(X_sc[:, feat_idx].min(), X_sc[:, feat_idx].max(),
                            GRID_RESOLUTION)
    avg_preds = []
    X_copy    = X_sc.copy()
    for val in feat_vals:
        X_copy[:, feat_idx] = val
        avg_preds.append(ann.predict_proba_positive(X_copy).mean())
    return feat_vals, np.array(avg_preds)


def pdp_2d_sklearn(model, X_df, feat1, feat2):
    i1 = list(X_df.columns).index(feat1)
    i2 = list(X_df.columns).index(feat2)
    result = partial_dependence(
        model, X_df, features=[(i1, i2)],
        kind="average", grid_resolution=GRID_RESOLUTION,
        response_method="predict_proba",
    )
    g1  = result["grid_values"][0]
    g2  = result["grid_values"][1]
    avg = result["average"][0]
    if avg.ndim == 3:
        avg = avg[1]
    return g1, g2, avg


def pdp_2d_ann(ann, X_sc, feat1, feat2):
    i1    = FEATURES.index(feat1)
    i2    = FEATURES.index(feat2)
    g1    = np.linspace(X_sc[:, i1].min(), X_sc[:, i1].max(), GRID_RESOLUTION)
    g2    = np.linspace(X_sc[:, i2].min(), X_sc[:, i2].max(), GRID_RESOLUTION)
    surf  = np.zeros((len(g1), len(g2)))
    X_copy = X_sc.copy()
    for r, v1 in enumerate(g1):
        for c, v2 in enumerate(g2):
            X_copy[:, i1] = v1
            X_copy[:, i2] = v2
            surf[r, c]    = ann.predict_proba_positive(X_copy).mean()
    return g1, g2, surf


def plot_pdp_1d(fitted, X_sc, X_df, scaler):
    print("\n[Figure 1] 1-D PDPs ...")
    for feature in TOP_VARS:
        fig, axes = plt.subplots(1, len(ALL_MODELS), figsize=(18, 4), sharey=True)
        for ax, mname in zip(axes, ALL_MODELS):
            color = COLORS[mname]
            if mname == "ANN":
                grid_sc, avg = pdp_1d_ann(fitted[mname], X_sc, feature)
            else:
                grid_sc, avg = pdp_1d_sklearn(fitted[mname], X_df, feature)
            grid_orig = unscale_grid(grid_sc, feature, scaler)
            ax.plot(grid_orig, avg, color=color, linewidth=2.2)
            ax.fill_between(grid_orig, avg, alpha=0.15, color=color)
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
            ax.set_title(mname, fontsize=10)
            ax.set_xlabel(feature, fontsize=9)
            ax.set_ylim(-0.05, 1.10)
        plt.tight_layout()
        savefig(fig, "pdp_1d", f"PDP1D_{feature.replace('.', '')}")
        plt.close(fig); gc.collect()

    # Combined
    fig, axes = plt.subplots(len(TOP_VARS), len(ALL_MODELS),
                             figsize=(18, 4 * len(TOP_VARS)), sharey="row")
    for row, feature in enumerate(TOP_VARS):
        for col, mname in enumerate(ALL_MODELS):
            ax    = axes[row][col]
            color = COLORS[mname]
            if mname == "ANN":
                grid_sc, avg = pdp_1d_ann(fitted[mname], X_sc, feature)
            else:
                grid_sc, avg = pdp_1d_sklearn(fitted[mname], X_df, feature)
            grid_orig = unscale_grid(grid_sc, feature, scaler)
            ax.plot(grid_orig, avg, color=color, linewidth=2.0)
            ax.fill_between(grid_orig, avg, alpha=0.15, color=color)
            ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
            ax.set_ylim(-0.05, 1.10)
            if row == 0:
                ax.set_title(mname, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{feature}\nP(Risky)", fontsize=9)
    plt.tight_layout()
    savefig(fig, "pdp_1d", "PDP1D_combined")
    plt.close(fig); gc.collect()
    print("  1-D PDPs complete.")


def plot_pdp_2d(fitted, X_sc, X_df, scaler):
    print("\n[Figure 2] 2-D Interaction plots ...")
    cmap   = "RdYlGn_r"
    levels = np.linspace(0, 1, 11)

    for (feat1, feat2) in TOP_PAIRS:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        fig.suptitle(f"2-D PDP — {feat1} × {feat2} [TEST]", fontsize=12)
        for ax, mname in zip(axes, ALL_MODELS):
            if mname == "ANN":
                g1_sc, g2_sc, surf = pdp_2d_ann(fitted[mname], X_sc, feat1, feat2)
            else:
                g1_sc, g2_sc, surf = pdp_2d_sklearn(fitted[mname], X_df, feat1, feat2)
            g1_orig = unscale_grid(g1_sc, feat1, scaler)
            g2_orig = unscale_grid(g2_sc, feat2, scaler)
            ax.contourf(g2_orig, g1_orig, surf, levels=levels, cmap=cmap, vmin=0, vmax=1)
            ax.contour(g2_orig, g1_orig, surf, levels=[0.5], colors="black",
                       linewidths=1.2, linestyles="--")
            ax.set_title(mname, fontsize=9)
            ax.set_xlabel(feat2, fontsize=8)
            if ax == axes[0]:
                ax.set_ylabel(feat1, fontsize=8)
        pair_label = f"{feat1.replace('.','')}_x_{feat2.replace('.','')}"
        savefig(fig, "pdp_2d", f"PDP2D_{pair_label}")
        plt.close(fig); gc.collect()

    # Combined
    n_pairs = len(TOP_PAIRS)
    fig, axes = plt.subplots(n_pairs, len(ALL_MODELS),
                             figsize=(22, 4.5 * n_pairs))
    for row, (feat1, feat2) in enumerate(TOP_PAIRS):
        for col, mname in enumerate(ALL_MODELS):
            ax = axes[row][col]
            if mname == "ANN":
                g1_sc, g2_sc, surf = pdp_2d_ann(fitted[mname], X_sc, feat1, feat2)
            else:
                g1_sc, g2_sc, surf = pdp_2d_sklearn(fitted[mname], X_df, feat1, feat2)
            g1_orig = unscale_grid(g1_sc, feat1, scaler)
            g2_orig = unscale_grid(g2_sc, feat2, scaler)
            ax.contourf(g2_orig, g1_orig, surf, levels=levels, cmap=cmap, vmin=0, vmax=1)
            if row == 0:
                ax.set_title(mname, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{feat1}\n↕", fontsize=8)
    plt.tight_layout()
    savefig(fig, "pdp_2d", "PDP2D_combined")
    plt.close(fig); gc.collect()
    print("  2-D plots complete.")


def main():
    set_seed(RANDOM_STATE)
    setup_dirs()
    X, y    = load_data()
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    X_df    = pd.DataFrame(X_sc, columns=FEATURES)
    fitted  = fit_all_models(X_sc, y)
    plot_pdp_1d(fitted, X_sc, X_df, scaler)
    plot_pdp_2d(fitted, X_sc, X_df, scaler)

    print("\n--- TEST_05 VERIFICATION ---")
    pdp1d_dir = os.path.join(OUTPUT_DIR, "figures", "pdp_1d")
    pdp2d_dir = os.path.join(OUTPUT_DIR, "figures", "pdp_2d")
    pngs_1d = [f for f in os.listdir(pdp1d_dir) if f.endswith(".png")]
    pngs_2d = [f for f in os.listdir(pdp2d_dir) if f.endswith(".png")]
    # 3 individual + 1 combined = 4 for 1D; 3 pairs + 1 combined = 4 for 2D
    assert len(pngs_1d) >= 4, f"Expected ≥4 1-D PDP PNGs, got {len(pngs_1d)}"
    assert len(pngs_2d) >= 4, f"Expected ≥4 2-D PDP PNGs, got {len(pngs_2d)}"
    print(f"  ✅  pdp_1d PNGs : {len(pngs_1d)} (expected ≥4)")
    print(f"  ✅  pdp_2d PNGs : {len(pngs_2d)} (expected ≥4)")
    print("TEST_05 PASSED\n")


if __name__ == "__main__":
    main()

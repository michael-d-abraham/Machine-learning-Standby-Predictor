"""
Assignment 8 Part B: SVM + interpretability on music energy proxy task.

Pipeline -> search -> evaluate -> plots -> interpretability.
Numeric features only; artist-based 60/20/20 split; binary target from train median.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "tcc_ceds_music.csv"
SEED = 42
TARGET_COL = "energy"

EXCLUDE_COLS = [
    "Unnamed: 0",
    "artist_name",
    "track_name",
    "lyrics",
    "genre",
    "topic",
    "release_date",
    "energy",
]


def inspect_dataset(csv_path):
    """Load CSV, identify numeric features, basic checks."""
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_features = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Numeric features: {len(numeric_features)}")
    return df, numeric_features


def split_by_group(df, group_col, seed, train_frac=0.60, val_frac=0.20):
    """Group-based 60/20/20 split; no artist overlap."""
    rng = np.random.default_rng(seed)
    unique_groups = df[group_col].unique()
    perm = rng.permutation(unique_groups)
    n = len(perm)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_g = set(perm[:n_train])
    val_g = set(perm[n_train : n_train + n_val])
    test_g = set(perm[n_train + n_val :])
    train_mask = df[group_col].isin(train_g)
    val_mask = df[group_col].isin(val_g)
    test_mask = df[group_col].isin(test_g)
    return train_mask, val_mask, test_mask


def split_data(df, numeric_features, target_col, seed=42, train_frac=0.60, val_frac=0.20):
    """Artist split, binary target from train median only."""
    train_mask, val_mask, test_mask = split_by_group(df, "artist_name", seed, train_frac, val_frac)
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    train_median = train_df[target_col].median()
    y_train = (train_df[target_col] >= train_median).astype(int)
    y_val = (val_df[target_col] >= train_median).astype(int)
    y_test = (test_df[target_col] >= train_median).astype(int)

    train_artists = set(train_df["artist_name"].unique())
    val_artists = set(val_df["artist_name"].unique())
    test_artists = set(test_df["artist_name"].unique())
    assert len(train_artists & val_artists) == 0
    assert len(train_artists & test_artists) == 0
    assert len(val_artists & test_artists) == 0

    X_train = train_df[numeric_features].copy()
    X_val = val_df[numeric_features].copy()
    X_test = test_df[numeric_features].copy()

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, numeric_features


def build_svm_pipeline():
    """Pipeline: impute -> scale -> SVC."""
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", SVC(random_state=SEED)),
    ])
    return pipeline


def run_svm_search(X_train, y_train, feature_names):
    """RandomizedSearchCV over kernel, C, gamma, class_weight."""
    C_values = np.logspace(-2, 2, 20)
    gamma_values = np.logspace(-4, 0, 20)
    param_dist = {
        "model__kernel": ["linear", "rbf"],
        "model__C": C_values,
        "model__gamma": gamma_values,
        "model__class_weight": [None, "balanced"],
    }
    pipeline = build_svm_pipeline()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_on_test(best_model, X_val, y_val, X_test, y_test, best_params=None):
    """Validation sanity check, then test once; return metrics and predictions."""
    preds_val = best_model.predict(X_val)
    scores_val = best_model.decision_function(X_val)
    roc_auc_val = roc_auc_score(y_val, scores_val)

    preds = best_model.predict(X_test)
    scores = best_model.decision_function(X_test)
    roc_auc_test = roc_auc_score(y_test, scores)

    if best_params is not None:
        print("Best params:", best_params)
    print("Val ROC AUC:", roc_auc_val)
    print("Test ROC AUC:", roc_auc_test)
    print(classification_report(y_test, preds))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, preds))

    return {
        "preds_val": preds_val,
        "scores_val": scores_val,
        "roc_auc_val": roc_auc_val,
        "preds": preds,
        "scores": scores,
        "roc_auc_test": roc_auc_test,
        "cm": confusion_matrix(y_test, preds),
    }


def plot_confusion_matrix(y_true, y_pred, output_path="svm_confusion_matrix_test.png"):
    """Confusion matrix image with matplotlib."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Energy", "High Energy"])
    ax.set_yticklabels(["Low Energy", "High Energy"])
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curve(y_true, scores, output_path="svm_roc_curve_test.png"):
    """ROC curve from decision scores."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pr_curve(y_true, scores, output_path="svm_pr_curve_test.png"):
    """Precision-Recall curve from decision scores."""
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    baseline = y_true.mean()
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"Model (AP = {ap:.3f})")
    plt.axhline(y=baseline, color="k", linestyle="--", label=f"Baseline = {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pca_decision_scores(best_model, X_train, X_val, feature_names, output_path="svm_pca_decision_scores_val.png"):
    """PCA 2D projection of val set colored by decision score (leakage-safe)."""
    preprocess = best_model.named_steps["preprocess"]
    X_train_scaled = preprocess.transform(X_train)
    X_val_scaled = preprocess.transform(X_val)
    pca = PCA(n_components=2, random_state=SEED).fit(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    scores_val = best_model.decision_function(X_val)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=scores_val, cmap="RdYlGn", alpha=0.6)
    plt.colorbar(sc, label="Decision Score")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("SVM Decision Scores in PCA Space (Validation)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def extract_linear_weights(best_model, feature_names):
    """If kernel is linear, print top coefficient magnitudes."""
    kernel = best_model.named_steps["model"].kernel
    if kernel != "linear":
        print("RBF kernel: no direct feature weights; interpret via decision scores + projection.")
        return
    coefs = best_model.named_steps["model"].coef_[0]
    importance = sorted(zip(feature_names, np.abs(coefs)), key=lambda x: x[1], reverse=True)
    print("Top 10 features by linear weight magnitude:")
    for name, w in importance[:10]:
        print(f"  {name}: {w:.4f}")


def _format_params(best_params):
    """Human-readable best params for findings file."""
    parts = []
    for k, v in best_params.items():
        key = k.replace("model__", "")
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        parts.append(f"{key}={v}")
    return ", ".join(parts)


def generate_findings_file(best_params, roc_auc_val, roc_auc_test, cm, kernel_linear, output_path="assignment8_partB_findings.md"):
    """Write findings markdown for capstone."""
    tn, fp, fn, tp = cm.ravel()
    params_str = _format_params(best_params)
    lines = [
        "# Assignment 8 Part B: SVM + Interpretability Findings",
        "",
        "## 1. Project Context (Brief)",
        "",
        "Music recommender proxy: binary energy classification (high vs low) using numeric audio and topic features. Artist-based 60/20/20 split; target from train median only.",
        "",
        "## 2. Technique + Assumptions",
        "",
        "- **SVM**: margin maximization, scaling required; C controls margin hardness, gamma (RBF) controls flexibility.",
        "- **Kernels**: linear (interpretable weights) vs RBF (no direct weights).",
        "- **class_weight**: None or balanced for class imbalance.",
        "",
        "## 3. What Was Attempted",
        "",
        "Pipeline: SimpleImputer(median) -> StandardScaler -> SVC. RandomizedSearchCV (n_iter=25, cv=5, scoring=roc_auc) over kernel, C, gamma, class_weight. Fit on train; evaluated on val then test once.",
        "",
        "## 4. Results",
        "",
        f"- Best params: {params_str}",
        f"- Val ROC AUC: {roc_auc_val:.4f}",
        f"- Test ROC AUC: {roc_auc_test:.4f}",
        f"- Confusion matrix (test): TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "",
        "## 5. Interpretability",
        "",
        "- **decision_function**: signed distance to hyperplane; larger magnitude = more confident.",
        "- **Linear weights**: " + ("Top coefficients printed above; interpret feature importance from magnitude." if kernel_linear else "RBF kernel: no direct feature weights; interpret via decision scores and PCA projection."),
        "- **PCA plot**: 2D projection of validation set colored by decision score; overlap and boundary softness visible.",
        "",
        "## 6. Fit Assessment for Capstone",
        "",
        "SVM is a partial fit: recommender is not a classifier, but the proxy task assesses whether the representation supports discrimination. ROC AUC indicates how well the features separate high vs low energy.",
        "",
        "## 7. Limitations + Next Steps",
        "",
        "- 23 features: curse of dimensionality; consider PCA or feature selection.",
        "- RBF may overfit in high-D space.",
        "- Next: dimensionality reduction or other interpretability (e.g., SHAP) if needed.",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    df, numeric_features = inspect_dataset(CSV_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = split_data(
        df, numeric_features, TARGET_COL, seed=SEED, train_frac=0.60, val_frac=0.20
    )
    # Use list for feature_names in case split_data returns it (we already have numeric_features)
    if not isinstance(feature_names, list):
        feature_names = numeric_features

    search = run_svm_search(X_train, y_train, feature_names)
    best_model = search.best_estimator_
    best_params = search.best_params_

    ev = evaluate_on_test(best_model, X_val, y_val, X_test, y_test, best_params)

    plot_confusion_matrix(y_test, ev["preds"])
    plot_roc_curve(y_test, ev["scores"])
    plot_pr_curve(y_test, ev["scores"])
    plot_pca_decision_scores(best_model, X_train, X_val, feature_names)

    extract_linear_weights(best_model, feature_names)

    generate_findings_file(
        best_params,
        ev["roc_auc_val"],
        ev["roc_auc_test"],
        ev["cm"],
        kernel_linear=(best_params.get("model__kernel") == "linear"),
    )


if __name__ == "__main__":
    main()

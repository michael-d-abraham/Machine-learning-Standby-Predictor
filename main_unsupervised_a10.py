"""
Assignment 10 Part B: Unsupervised learning exploration for music recommender proxy.

Scoped workflow:
1) Build unsupervised feature matrix from numeric columns only (exclude labels/meta/text).
2) Preprocess with median imputation + standard scaling.
3) Run PCA for variance structure and 2D visualization.
4) Run k-means across multiple k values with inertia + silhouette.
5) Select k using evidence, then summarize cluster profiles.
6) Compare clusters to proxy energy label post hoc only (no label in fitting).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

CSV_PATH = "tcc_ceds_music.csv"
SEED = 42
TARGET_COL = "energy"

# Keep exclusions explicit and readable for assignment transparency.
EXCLUDE_COLS = [
    "Unnamed: 0",
    "artist_name",
    "track_name",
    "lyrics",
    "genre",
    "topic",
    "release_date",
    "energy",  # excluded from unsupervised feature matrix by design
]

# Evaluate at least 4 k values (assignment requirement).
K_VALUES = [2, 3, 4, 5, 6, 7]


def split_by_group(df, group_col, seed, train_frac=0.60, val_frac=0.20):
    """Group-based 60/20/20 split; no group overlap."""
    rng = np.random.default_rng(seed)
    unique_groups = df[group_col].dropna().unique()
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


def choose_numeric_features(df):
    """Return included numeric features and excluded columns for reporting."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    excluded_existing = [c for c in EXCLUDE_COLS if c in df.columns]
    included_features = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    return included_features, excluded_existing, numeric_cols


def build_proxy_label_from_train_median(df, target_col=TARGET_COL):
    """
    Build proxy high/low-energy labels using train median only.
    Labels are for post hoc interpretation; not used in PCA/k-means fitting.
    """
    train_mask, val_mask, test_mask = split_by_group(df, "artist_name", SEED, 0.60, 0.20)
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    train_median = train_df[target_col].median()
    y_train = (train_df[target_col] >= train_median).astype(int)
    y_val = (val_df[target_col] >= train_median).astype(int)
    y_test = (test_df[target_col] >= train_median).astype(int)

    y_proxy = pd.Series(index=df.index, dtype="float64")
    y_proxy.loc[train_df.index] = y_train
    y_proxy.loc[val_df.index] = y_val
    y_proxy.loc[test_df.index] = y_test
    y_proxy = y_proxy.astype(int)

    return y_proxy, train_median, train_mask, val_mask, test_mask


def preprocess_features(X):
    """Median impute then standard scale (aligned with supervised pipeline)."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, imputer, scaler


def run_pca(X_scaled, feature_names):
    """Fit PCA and return transformed coordinates + variance table."""
    pca = PCA(random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    n_show = min(10, len(feature_names))
    ratios = pca.explained_variance_ratio_
    cum_ratios = np.cumsum(ratios)

    pca_table = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, n_show + 1)],
            "explained_variance_ratio": ratios[:n_show],
            "cumulative_explained_variance": cum_ratios[:n_show],
        }
    )
    return pca, X_pca, pca_table


def plot_pca_explained_variance(pca, output_path="a10_pca_explained_variance.png"):
    ratios = pca.explained_variance_ratio_
    cum_ratios = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(components, ratios, marker="o", label="Explained Variance Ratio")
    plt.plot(components, cum_ratios, marker="s", label="Cumulative Explained Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pca_scatter(X_pca, labels, title, colorbar_label, output_path):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.55, s=18)
    plt.colorbar(sc, label=colorbar_label)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_kmeans_across_k(X_scaled, k_values):
    rows = []
    labels_by_k = {}

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=SEED,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_
        sil = silhouette_score(X_scaled, labels)

        rows.append({"k": k, "inertia": inertia, "silhouette": sil})
        labels_by_k[k] = labels

    results = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return results, labels_by_k


def choose_k_with_evidence(k_results):
    """
    Choose k using simple, explainable evidence:
    1) Best silhouette.
    2) If near tie (within 0.01), prefer smaller k for interpretability.
    """
    best_sil = k_results["silhouette"].max()
    near_best = k_results[k_results["silhouette"] >= (best_sil - 0.01)].copy()
    chosen_k = int(near_best["k"].min())
    return chosen_k, float(best_sil)


def plot_kmeans_metrics(k_results):
    plt.figure(figsize=(8, 5))
    plt.plot(k_results["k"], k_results["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Plot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("a10_kmeans_elbow.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_results["k"], k_results["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette by k")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("a10_kmeans_silhouette.png", dpi=150)
    plt.close()


def cluster_profile_summary(df_features, cluster_labels, top_n=5):
    """
    Summarize clusters using standardized mean differences from global mean.
    Returns:
      - cluster_sizes table
      - standardized cluster mean table (cluster z-profile)
      - top distinguishing features per cluster
    """
    features_df = df_features.copy()
    features_df["cluster"] = cluster_labels

    cluster_sizes = features_df["cluster"].value_counts().sort_index().rename("count").to_frame()
    cluster_sizes["percent"] = cluster_sizes["count"] / cluster_sizes["count"].sum()

    X = df_features.copy()
    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0).replace(0, np.nan)

    cluster_means = features_df.groupby("cluster").mean(numeric_only=True).drop(columns=["cluster"], errors="ignore")
    standardized_diff = (cluster_means - global_mean) / global_std
    standardized_diff = standardized_diff.fillna(0.0)

    top_features = {}
    for cl in standardized_diff.index:
        diffs = standardized_diff.loc[cl].abs().sort_values(ascending=False).head(top_n)
        top_features[int(cl)] = [(feat, float(standardized_diff.loc[cl, feat])) for feat in diffs.index]

    return cluster_sizes, standardized_diff, top_features


def write_findings_markdown(
    included_features,
    excluded_existing,
    pca_table,
    k_results,
    chosen_k,
    selected_silhouette,
    cluster_sizes,
    top_features,
    nmi,
    ari,
    output_path="assignment10_partB_findings.md",
):
    """Write concise findings text for assignment reflection."""
    lines = []
    lines.append("# Assignment 10 Part B: Unsupervised Learning Findings")
    lines.append("")
    lines.append("## 1) Data Preparation")
    lines.append("")
    lines.append(f"- Included numeric feature count: **{len(included_features)}**")
    lines.append(f"- Included features: {', '.join(included_features)}")
    lines.append(f"- Excluded columns: {', '.join(excluded_existing)}")
    lines.append("- Exclusion rationale: text/meta/identifier columns and target (`energy`) were excluded from the unsupervised feature matrix.")
    lines.append("- Preprocessing: `SimpleImputer(strategy=\"median\")` then `StandardScaler()`.")
    lines.append("")
    lines.append("## 2) PCA Observations")
    lines.append("")
    lines.append("First principal components (explained + cumulative variance):")
    lines.append("")
    lines.append(pca_table.to_string(index=False))
    lines.append("")
    lines.append("- PCA is a projection and does not prove natural, true clusters.")
    lines.append("- Overlap in 2D PCA space should be interpreted as evidence of possible gradients/continuous structure, not failure by itself.")
    lines.append("")
    lines.append("## 3) K-Means Across k")
    lines.append("")
    lines.append(k_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append(f"- Selected k: **{chosen_k}** (silhouette-based evidence, with simpler k preferred when near-tied).")
    lines.append(f"- Best silhouette observed: **{selected_silhouette:.4f}**")
    lines.append("- If silhouette values are low overall, that indicates weak separation and is still a valid, informative result.")
    lines.append("")
    lines.append("## 4) Cluster Interpretation")
    lines.append("")
    lines.append("Cluster sizes:")
    lines.append("")
    lines.append(cluster_sizes.to_string(float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("Top distinguishing standardized feature means by cluster (signed z-difference vs global mean):")
    lines.append("")
    for cl in sorted(top_features.keys()):
        pairs = top_features[cl]
        pretty = "; ".join([f"{name}: {value:+.3f}" for name, value in pairs])
        lines.append(f"- Cluster {cl}: {pretty}")
    lines.append("")
    lines.append("Interpretation note: these are broad profiles, not guaranteed discrete musical categories.")
    lines.append("")
    lines.append("## 5) Post Hoc Comparison to Proxy Energy Label")
    lines.append("")
    lines.append("- Proxy label was used **after** clustering only, for interpretation.")
    lines.append(f"- Normalized Mutual Information (NMI): **{nmi:.4f}**")
    lines.append(f"- Adjusted Rand Index (ARI): **{ari:.4f}**")
    lines.append("- Low agreement can mean energy is only one axis of structure, or the space is more continuous than cluster-like.")
    lines.append("")
    lines.append("## 6) Fit Judgment")
    lines.append("")
    lines.append(
        "Unsupervised learning is a **partial fit** for this project: useful for exploratory structure checks and feature diagnostics, "
        "but not strong evidence by itself for hard song segmentation. For recommendation, similarity/ranking may be more natural than strict clustering."
    )
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    # 1) Load data and define unsupervised feature matrix
    df = pd.read_csv(CSV_PATH)
    included_features, excluded_existing, numeric_cols = choose_numeric_features(df)
    X_df = df[included_features].copy()

    print("=== Dataset Overview ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Numeric columns found: {len(numeric_cols)}")
    print(f"Included numeric features for unsupervised learning ({len(included_features)}):")
    print(included_features)
    print("Excluded columns present:")
    print(excluded_existing)

    # 2) Build proxy label separately (post hoc only)
    y_proxy, train_median, train_mask, val_mask, test_mask = build_proxy_label_from_train_median(df, TARGET_COL)
    print("\n=== Proxy Label Info (Post Hoc Only) ===")
    print(f"Train median energy threshold: {train_median:.6f}")
    print(f"Split sizes (artist-based): train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    print("Proxy label distribution (0=low, 1=high):")
    print(y_proxy.value_counts(normalize=True).sort_index())

    # 3) Preprocess unsupervised matrix
    X_scaled, _, _ = preprocess_features(X_df)

    # 4) PCA analysis and plots
    pca, X_pca, pca_table = run_pca(X_scaled, included_features)
    print("\n=== PCA (First Components) ===")
    print(pca_table.to_string(index=False))

    plot_pca_explained_variance(pca, "a10_pca_explained_variance.png")

    # 5) K-means sweep and metrics
    k_results, labels_by_k = evaluate_kmeans_across_k(X_scaled, K_VALUES)
    chosen_k, best_silhouette = choose_k_with_evidence(k_results)
    chosen_labels = labels_by_k[chosen_k]
    print("\n=== K-Means Results ===")
    print(k_results.to_string(index=False))
    print(f"Chosen k: {chosen_k}")
    print(f"Best silhouette: {best_silhouette:.4f}")

    plot_kmeans_metrics(k_results)

    # 6) PCA scatter plots for interpretation
    plot_pca_scatter(
        X_pca,
        chosen_labels,
        title=f"PCA (2D) Colored by K-Means Cluster (k={chosen_k})",
        colorbar_label="Cluster",
        output_path="a10_pca_scatter_by_cluster.png",
    )
    plot_pca_scatter(
        X_pca,
        y_proxy.values,
        title="PCA (2D) Colored by Proxy Energy Label (Post Hoc)",
        colorbar_label="Proxy Energy (0/1)",
        output_path="a10_pca_scatter_by_energy_proxy.png",
    )

    # 7) Cluster interpretation
    cluster_sizes, standardized_diff, top_features = cluster_profile_summary(X_df, chosen_labels, top_n=5)
    print("\n=== Cluster Sizes ===")
    print(cluster_sizes.to_string())

    print("\n=== Top Distinguishing Features per Cluster (standardized diff) ===")
    for cl in sorted(top_features.keys()):
        print(f"Cluster {cl}:")
        for feat, val in top_features[cl]:
            print(f"  {feat}: {val:+.3f}")

    # 8) Post hoc comparison to proxy label
    comparison_df = pd.DataFrame({"cluster": chosen_labels, "proxy_energy": y_proxy.values})
    ctab_counts = pd.crosstab(comparison_df["cluster"], comparison_df["proxy_energy"])
    ctab_row_pct = pd.crosstab(comparison_df["cluster"], comparison_df["proxy_energy"], normalize="index")
    nmi = normalized_mutual_info_score(y_proxy.values, chosen_labels)
    ari = adjusted_rand_score(y_proxy.values, chosen_labels)

    print("\n=== Post Hoc Cluster vs Proxy Energy ===")
    print("Counts:")
    print(ctab_counts.to_string())
    print("\nRow proportions:")
    print(ctab_row_pct.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nNMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")

    # Save post hoc crosstab tables
    ctab_counts.to_csv("a10_cluster_vs_proxy_counts.csv")
    ctab_row_pct.to_csv("a10_cluster_vs_proxy_rowpct.csv")
    standardized_diff.to_csv("a10_cluster_standardized_means.csv")
    k_results.to_csv("a10_kmeans_metrics.csv", index=False)
    pca_table.to_csv("a10_pca_variance_table.csv", index=False)

    # 9) Findings markdown for reflection
    write_findings_markdown(
        included_features=included_features,
        excluded_existing=excluded_existing,
        pca_table=pca_table,
        k_results=k_results,
        chosen_k=chosen_k,
        selected_silhouette=best_silhouette,
        cluster_sizes=cluster_sizes,
        top_features=top_features,
        nmi=nmi,
        ari=ari,
        output_path="assignment10_partB_findings.md",
    )

    print("\nSaved outputs:")
    print("- a10_pca_explained_variance.png")
    print("- a10_pca_scatter_by_cluster.png")
    print("- a10_pca_scatter_by_energy_proxy.png")
    print("- a10_kmeans_elbow.png")
    print("- a10_kmeans_silhouette.png")
    print("- a10_kmeans_metrics.csv")
    print("- a10_pca_variance_table.csv")
    print("- a10_cluster_standardized_means.csv")
    print("- a10_cluster_vs_proxy_counts.csv")
    print("- a10_cluster_vs_proxy_rowpct.csv")
    print("- assignment10_partB_findings.md")


if __name__ == "__main__":
    main()

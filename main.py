"""
CS 4320 — Assignment 5 (Part B) Capstone Classification Baseline
Music Recommendation System — Classification baseline with metrics

This file implements a classification baseline for a content-based music recommendation system.
The dataset contains songs with metadata and audio features.

PROXY TASK FOR THIS WEEK:
Since recommendation systems don't have traditional prediction targets, we use a proxy task:
    Predict whether a song has 'high energy' or 'low energy' (binary classification)
    
WHY ENERGY CLASSIFICATION AS TARGET?
1. Energy is a fundamental audio characteristic (intensity/activity of a song)
2. Binary classification exercises the full ML pipeline (split, preprocess, classify, evaluate)
3. Validates that features contain useful information for categorical decisions
4. Understanding energy patterns helps recommend songs with similar "feel"
5. Classification metrics (precision, recall, F1) provide richer evaluation than regression alone

This is a reasonable proxy because:
- It exercises the full classification pipeline (split, preprocess, classify, evaluate)
- It validates that our features contain useful information
- It establishes a baseline for understanding feature relationships
- Classification metrics help us understand model behavior in recommendation contexts

Workflow (leakage-safe):
1) Load data
2) Create binary target (high energy vs low energy, median split)
3) Split into train/val/test using GROUP-BASED splitting by artist (prevents leakage)
4) Build scikit-learn pipeline with preprocessing + classifier
5) Train on training data only
6) Evaluate on validation set with full classification metrics
7) Analyze ROC and Precision-Recall curves
8) Select decision threshold
9) Evaluate once on test set
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


CSV_PATH = "tcc_ceds_music.csv"
SEED = 4320 
TARGET_COL = "energy"


POSSIBLE_EXCLUDES = [
    "Unnamed: 0",
    "artist_name",
    "track_name",
    "lyrics",
    "release_date",
    TARGET_COL
]


def split_by_group(df: pd.DataFrame, group_col: str, seed: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Group-based split: split groups (artists) into train/val/test sets.
    All rows belonging to the same group go into the same split.
    This prevents data leakage in recommendation systems.
    """
    rng = np.random.default_rng(seed)
    
    unique_groups = df[group_col].unique()
    n_groups = len(unique_groups)
    
    perm_groups = rng.permutation(unique_groups)
    
    n_train_groups = int(round(train_frac * n_groups))
    n_val_groups = int(round(val_frac * n_groups))
    
    train_groups = set(perm_groups[:n_train_groups])
    val_groups = set(perm_groups[n_train_groups:n_train_groups + n_val_groups])
    test_groups = set(perm_groups[n_train_groups + n_val_groups:])
    
    train_mask = df[group_col].isin(train_groups)
    val_mask = df[group_col].isin(val_groups)
    test_mask = df[group_col].isin(test_groups)
    
    return train_mask, val_mask, test_mask


def load_and_prepare_data(csv_path):
    """
    Load the dataset and create binary classification target.
    """
    print("=" * 80)
    print("STEP 1: DATA LOADING AND TARGET CREATION")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    energy_median = df[TARGET_COL].median()
    print(f"\nEnergy median (train will be computed separately): {energy_median:.4f}")
    print(f"Energy range: [{df[TARGET_COL].min():.4f}, {df[TARGET_COL].max():.4f}]")
    
    return df, energy_median


def split_data(df, target_col, energy_median, seed=4320, train_frac=0.70, val_frac=0.15):
    """
    Split the data into three sets using group-based splitting by artist.
    Also create binary target from continuous energy.
    """
    print("\n" + "=" * 80)
    print("STEP 2: GROUP-BASED SPLIT BY ARTIST")
    print("=" * 80)
    
    train_mask, val_mask, test_mask = split_by_group(df, "artist_name", seed, train_frac, val_frac)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\nSplit Strategy: GROUP-BASED split by artist with seed={seed}")
    print(f"Proportions: {train_frac*100:.0f}% train / {val_frac*100:.0f}% validation / {(1-train_frac-val_frac)*100:.0f}% test")
    
    print(f"\nSplit sizes:")
    print(f"  - Training:   {len(train_df):4d} songs from {train_df['artist_name'].nunique()} artists")
    print(f"  - Validation: {len(val_df):4d} songs from {val_df['artist_name'].nunique()} artists")
    print(f"  - Test:       {len(test_df):4d} songs from {test_df['artist_name'].nunique()} artists")
    
    train_artists = set(train_df['artist_name'].unique())
    val_artists = set(val_df['artist_name'].unique())
    test_artists = set(test_df['artist_name'].unique())
    assert len(train_artists & val_artists) == 0, "Artist overlap between train and val!"
    assert len(train_artists & test_artists) == 0, "Artist overlap between train and test!"
    assert len(val_artists & test_artists) == 0, "Artist overlap between val and test!"
    print("  ✓ Verified: No artist overlap between splits")
    
    train_energy_median = train_df[target_col].median()
    print(f"\nCreating binary target (high energy vs low energy):")
    print(f"  Using training set median: {train_energy_median:.4f}")
    print(f"  High energy = 1 if energy >= {train_energy_median:.4f}, else 0")
    
    y_train = (train_df[target_col] >= train_energy_median).astype(int)
    y_val = (val_df[target_col] >= train_energy_median).astype(int)
    y_test = (test_df[target_col] >= train_energy_median).astype(int)
    
    print(f"\nClass distribution (training set):")
    train_counts = y_train.value_counts()
    print(f"  - Low Energy (0): {train_counts[0]} ({100*train_counts[0]/len(y_train):.1f}%)")
    print(f"  - High Energy (1): {train_counts[1]} ({100*train_counts[1]/len(y_train):.1f}%)")
    
    X_train = train_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in test_df.columns])

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(X_train):
    """
    Build a preprocessing + model pipeline.
    """
    print("\n" + "=" * 80)
    print("STEP 3: PIPELINE-BASED PREPROCESSING")
    print("=" * 80)
    
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nFeature types:")
    print(f"  - Numeric ({len(numeric_cols)}): {numeric_cols[:5]}...")
    print(f"  - Categorical ({len(categorical_cols)}): {categorical_cols}")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    
    print(f"\nPipeline summary:")
    print(f"  1. Numeric features: median imputation → standardization")
    print(f"  2. Categorical features: mode imputation → one-hot encoding")
    print(f"  3. Classification: Logistic Regression")
    
    print(f"\nKey benefit: Pipeline learns preprocessing from TRAINING data only!")
    print(f"  This prevents data leakage from validation/test sets.")
    
    return pipeline, numeric_cols, categorical_cols


def train_model(pipeline, X_train, y_train):
    """
    Train the model on training data.
    """
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING")
    print("=" * 80)
    
    print(f"\nTraining Logistic Regression on {len(X_train)} samples...")
    print(f"  - Original features: {X_train.shape[1]} columns")
    
    pipeline.fit(X_train, y_train)
    
    n_features_after = pipeline.named_steps['preprocessor'].transform(X_train).shape[1]
    print(f"  - Features after one-hot encoding: {n_features_after} columns")
    
    print(f"\nTraining complete!")
    
    return pipeline


def evaluate_metrics(y_true, y_pred, y_pred_proba, dataset_name="Validation"):
    """
    Compute and display classification metrics.
    """
    print(f"\n{dataset_name} Set Metrics:")
    print("-" * 40)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"PR AUC:    {pr_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def plot_confusion_matrix(y_true, y_pred, dataset_name="Validation"):
    """
    Visualize and interpret the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Energy', 'High Energy'],
                yticklabels=['Low Energy', 'High Energy'])
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name.lower()}.png', dpi=150)
    print(f"\n  → Saved: confusion_matrix_{dataset_name.lower()}.png")
    
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  - True Negatives (TN):  {tn:4d} Correctly predicted Low Energy")
    print(f"  - False Positives (FP): {fp:4d} Said High Energy, but was Low Energy")
    print(f"  - False Negatives (FN): {fn:4d} Said Low Energy, but was High Energy")
    print(f"  - True Positives (TP):  {tp:4d} Correctly predicted High Energy")
    
    print(f"\nWhy Accuracy Alone Can Be Misleading:")
    print(f"  If classes are imbalanced, a model that always predicts the majority class")
    print(f"  can achieve high accuracy but be useless. We need to look at:")
    print(f"    • Precision: Of songs we predicted as high energy, how many actually were?")
    print(f"    • Recall: Of songs with high energy, how many did we catch?")
    print(f"    • F1-Score: Balance between precision and recall")


def plot_roc_pr_curves(y_true, y_pred_proba, dataset_name="Validation"):
    """
    Plot ROC and Precision-Recall curves.
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(fpr, tpr, label=f'Our Model (AUC = {roc_auc:.3f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random Guessing', linewidth=1)
    axes[0].set_xlabel('False Positive Rate (FP / TN+FP)')
    axes[0].set_ylabel('True Positive Rate = Recall (TP / TP+FN)')
    axes[0].set_title(f'ROC Curve - {dataset_name} Set')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recall_vals, precision_vals, label=f'Our Model (AUC = {pr_auc:.3f})', linewidth=2)
    baseline = y_true.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', 
                    label=f'Random Guessing = {baseline:.3f}', linewidth=1)
    axes[1].set_xlabel('Recall (TP / TP+FN)')
    axes[1].set_ylabel('Precision (TP / TP+FP)')
    axes[1].set_title(f'Precision-Recall Curve - {dataset_name} Set')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'roc_pr_curves_{dataset_name.lower()}.png', dpi=150)
    print(f"  → Saved: roc_pr_curves_{dataset_name.lower()}.png")
    
    print(f"\n  Why we plot both:")
    print(f"    • ROC: Good for balanced data, shows overall discriminative ability")
    print(f"    • PR: Better for imbalanced data, focuses on the minority class")
    
    return fpr, tpr, roc_thresholds, precision_vals, recall_vals, pr_thresholds


def threshold_analysis(pipeline, X_val, y_val, alternative_threshold=0.4):
    """
    Experiment with different decision thresholds.
    """
    print("\n" + "=" * 80)
    print("STEP 7: THRESHOLD SELECTION")
    print("=" * 80)
    
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    y_pred_alternative = (y_pred_proba >= alternative_threshold).astype(int)
    
    print(f"\nComparing two thresholds:")
    
    print(f"\nOption 1: Threshold = 0.5 (default)")
    precision_default = precision_score(y_val, y_pred_default)
    recall_default = recall_score(y_val, y_pred_default)
    f1_default = f1_score(y_val, y_pred_default)
    print(f"   Precision: {precision_default:.4f} (of predicted high energy, {100*precision_default:.1f}% actually were)")
    print(f"   Recall:    {recall_default:.4f} (we caught {100*recall_default:.1f}% of actual high energy songs)")
    print(f"   F1-Score:  {f1_default:.4f}")
    
    print(f"\nOption 2: Threshold = {alternative_threshold} (alternative)")
    precision_alt = precision_score(y_val, y_pred_alternative)
    recall_alt = recall_score(y_val, y_pred_alternative)
    f1_alt = f1_score(y_val, y_pred_alternative)
    print(f"   Precision: {precision_alt:.4f} (of predicted high energy, {100*precision_alt:.1f}% actually were)")
    print(f"   Recall:    {recall_alt:.4f} (we caught {100*recall_alt:.1f}% of actual high energy songs)")
    print(f"   F1-Score:  {f1_alt:.4f}")
    
    print(f"\nWhat changed?")
    print(f"   Precision: {precision_default:.4f} -> {precision_alt:.4f} ({precision_alt-precision_default:+.4f})")
    print(f"   Recall:    {recall_default:.4f} -> {recall_alt:.4f} ({recall_alt-recall_default:+.4f})")
    print(f"   F1-Score:  {f1_default:.4f} -> {f1_alt:.4f} ({f1_alt-f1_default:+.4f})")
    
    print(f"\nInterpretation:")
    if alternative_threshold < 0.5:
        print(f"   By lowering the threshold to {alternative_threshold}:")
        print(f"   - We catch MORE high energy songs (recall up {100*(recall_alt-recall_default):.1f} percentage points)")
        print(f"   - We have MORE false alarms (precision down {100*(precision_default-precision_alt):.1f} percentage points)")
    else:
        print(f"   By raising the threshold to {alternative_threshold}:")
        print(f"   - We catch FEWER high energy songs (recall down {100*(recall_default-recall_alt):.1f} percentage points)")
        print(f"   - We have FEWER false alarms (precision up {100*(precision_alt-precision_default):.1f} percentage points)")
    
    if f1_alt > f1_default:
        chosen_threshold = alternative_threshold
        print(f"\n  Using threshold = {alternative_threshold} for final test evaluation (higher F1)")
    else:
        chosen_threshold = 0.5
        print(f"\n  Using threshold = 0.5 for final test evaluation (higher F1)")
    
    return chosen_threshold


def final_test_evaluation(pipeline, X_test, y_test, X_val, y_val, chosen_threshold=0.5):
    """
    Evaluate on the test set (ONLY ONCE!) and compare to validation.
    """
    print("\n" + "=" * 80)
    print("STEP 8: FINAL TEST SET EVALUATION")
    print("=" * 80)
    
    print(f"\nUnlocking the test set (first and only time)...")
    print(f"   Using threshold = {chosen_threshold}")
    
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= chosen_threshold).astype(int)
    
    test_metrics = evaluate_metrics(y_test, y_test_pred, y_test_proba, dataset_name="Test")
    
    print("\n" + "-" * 40)
    print("Test Set Confusion Matrix:")
    print("-" * 40)
    plot_confusion_matrix(y_test, y_test_pred, dataset_name="Test")
    
    print("\n" + "-" * 40)
    print("Test Set Curves:")
    print("-" * 40)
    plot_roc_pr_curves(y_test, y_test_proba, dataset_name="Test")
    
    print("\n" + "=" * 80)
    print("VALIDATION vs TEST COMPARISON")
    print("=" * 80)
    
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= chosen_threshold).astype(int)
    val_metrics = evaluate_metrics(y_val, y_val_pred, y_val_proba, dataset_name="Validation")
    
    print("\nSide-by-side comparison:")
    print(f"{'Metric':<12} {'Validation':<12} {'Test':<12} {'Difference':<12}")
    print("-" * 50)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        val_val = val_metrics[metric]
        test_val = test_metrics[metric]
        diff = test_val - val_val
        print(f"{metric.upper():<12} {val_val:<12.4f} {test_val:<12.4f} {diff:+.4f}")
    
    print("\nWhat does this tell us?")
    accuracy_diff = abs(test_metrics['accuracy'] - val_metrics['accuracy'])
    if accuracy_diff < 0.02:
        print("  GOOD NEWS: Test and validation performance are very similar!")
        print("    This means:")
        print("    - Our model generalizes well to new, unseen data")
        print("    - We didn't overfit to the training or validation sets")
        print("    - We can be confident in these performance estimates")
    else:
        print("  NOTICE: Test performance differs from validation")
        if test_metrics['accuracy'] < val_metrics['accuracy']:
            print("    - Test is worse: Possible slight overfitting")
            print("    - Or test set has a slightly different distribution")
        else:
            print("    - Test is better: Lucky split or test is slightly easier")
        print("    - Could also be random variation")


def main():
    """
    Main function orchestrating the complete classification workflow.
    """
    print("\n" + "=" * 80)
    print("MUSIC RECOMMENDATION SYSTEM - CLASSIFICATION BASELINE")
    print("Assignment 5 - Part B: Classification and Metrics")
    print("=" * 80)
    
    print("\nProxy Task: Predicting High Energy vs Low Energy (binary classification)")
    print("\nRationale:")
    print("  - Energy represents song intensity/activity level")
    print("  - Binary classification validates feature relationships")
    print("  - Classification metrics provide richer evaluation")
    print("  - Understanding energy patterns helps recommend similar-feeling songs")
    print("=" * 80)
    
    df, energy_median = load_and_prepare_data(CSV_PATH)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, TARGET_COL, energy_median, seed=SEED, train_frac=0.70, val_frac=0.15
    )
    
    pipeline, numeric_cols, categorical_cols = build_pipeline(X_train)
    
    pipeline = train_model(pipeline, X_train, y_train)
    
    print("\n" + "=" * 80)
    print("STEP 5: CONFUSION MATRIX AND METRICS (VALIDATION SET)")
    print("=" * 80)
    
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    val_metrics = evaluate_metrics(y_val, y_val_pred, y_val_proba, dataset_name="Validation")
    
    print("\n" + "-" * 40)
    print("Validation Set Confusion Matrix:")
    print("-" * 40)
    plot_confusion_matrix(y_val, y_val_pred, dataset_name="Validation")
    
    print("\n" + "=" * 80)
    print("STEP 6: ROC AND PRECISION-RECALL CURVES")
    print("=" * 80)
    
    print("\nPlotting ROC and Precision-Recall curves...")
    fpr, tpr, roc_thresh, prec, rec, pr_thresh = plot_roc_pr_curves(
        y_val, y_val_proba, dataset_name="Validation"
    )
    
    print(f"\nWhy both curves matter:")
    print(f"  - ROC Curve: Good for balanced datasets, shows TPR vs FPR tradeoff")
    print(f"  - PR Curve: Better for imbalanced datasets, focuses on positive class")
    
    chosen_threshold = threshold_analysis(pipeline, X_val, y_val, alternative_threshold=0.4)
    
    final_test_evaluation(pipeline, X_test, y_test, X_val, y_val, chosen_threshold)
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nSummary of outputs generated:")
    print("  1. confusion_matrix_validation.png")
    print("  2. confusion_matrix_test.png")
    print("  3. roc_pr_curves_validation.png")
    print("  4. roc_pr_curves_test.png")
    print("\nKey Takeaways:")
    print("  - Used artist-based group splitting to prevent leakage")
    print("  - Built pipeline to prevent data leakage")
    print("  - Evaluated multiple metrics beyond accuracy")
    print("  - Analyzed precision-recall tradeoff via threshold selection")
    print("  - Kept test set isolated until final evaluation")
    print("  - Compared validation and test performance to assess generalization")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

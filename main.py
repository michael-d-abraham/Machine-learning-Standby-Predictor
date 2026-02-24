"""
CS 4320 — Assignment 6 (Part B) Capstone: Regularization & Hyperparameter Control
Music Recommendation System — Regularization analysis with complexity control

This file extends the classification baseline with rigorous regularization and hyperparameter tuning.
The dataset contains songs with metadata and audio features.

PROXY TASK FOR THIS WEEK:
Since recommendation systems don't have traditional prediction targets, we use a proxy task:
    Predict whether a song has 'high energy' or 'low energy' (binary classification)
    
ASSIGNMENT 6 FOCUS: Regularization & Hyperparameter Tuning
1. Establish baseline model (C=1.0) and diagnose overfitting
2. Generate validation curves across C values to visualize bias-variance tradeoff
3. Perform disciplined grid search with 5-fold CV
4. Compare baseline vs tuned models on test set
5. Assess whether regularization matters for this dataset/representation

This is a reasonable proxy because:
- It exercises model complexity control in a supervised setting
- Validates whether features are stable or noisy
- Tests whether large dataset size (28k songs) makes regularization less critical
- Insights about overfitting inform similarity-based recommendation design

Workflow (leakage-safe):
1) Inspect dataset structure and identify numeric features
2) Create binary target (high energy vs low energy, median split from training data)
3) Split into train/val/test using GROUP-BASED splitting by artist (60/20/20)
4) Build scikit-learn pipeline with numeric features only
5) Train baseline model (C=1.0) and diagnose fit quality
6) Generate validation curve across C values
7) Perform grid search to select best C
8) Compare baseline vs tuned on test set (used only once)
9) Generate diagnostics for capstone reflection
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


CSV_PATH = "tcc_ceds_music.csv"
SEED = 42  # CHANGED from 4320 for Assignment 6
TARGET_COL = "energy"


# Columns to exclude from features (identifiers, text, dates, target)
POSSIBLE_EXCLUDES = [
    "Unnamed: 0",      # Index column
    "artist_name",      # Identifier (object type)
    "track_name",       # Identifier (object type)
    "lyrics",           # Text data (object type)
    "genre",            # Categorical metadata (object type) 
    "topic",            # Categorical metadata (object type)
    "release_date",     # Date column
    TARGET_COL          # Target variable
]


def split_by_group(df: pd.DataFrame, group_col: str, seed: int, train_frac: float = 0.60, val_frac: float = 0.20):
    """
    Group-based split: split groups (artists) into train/val/test sets.
    All rows belonging to the same group go into the same split.
    This prevents data leakage in recommendation systems.
    
    Updated for Assignment 6: Default split is now 60/20/20 (was 70/15/15).
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


# ==============================================================================
# PHASE 1: DATASET INSPECTION & VALIDATION
# ==============================================================================
def inspect_dataset(csv_path):
    """
    Phase 1: Inspect dataset structure, identify numeric features, check data quality.
    
    This function validates assumptions and identifies which columns to use as features.
    It checks for duplicates, missing values, and energy distribution.
    """
    print("=" * 80)
    print("PHASE 1: DATASET INSPECTION & VALIDATION")
    print("=" * 80)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check for duplicate songs
    duplicates = df.duplicated(subset=['track_name', 'artist_name']).sum()
    print(f"Duplicate songs: {duplicates}")
    if duplicates > 100:
        print(f"⚠️  WARNING: Found {duplicates} duplicate songs!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            raise ValueError("Too many duplicates detected. Aborting.")
    
    # Identify numeric columns
    all_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"\nAll numeric columns ({len(all_numeric_cols)}): {all_numeric_cols}")
    
    # Apply exclusions
    numeric_features = [col for col in all_numeric_cols if col not in POSSIBLE_EXCLUDES]
    print(f"\nNumeric features after exclusions ({len(numeric_features)}):")
    for i, col in enumerate(numeric_features, 1):
        print(f"  {i:2d}. {col}")
    
    if len(numeric_features) < 10:
        print(f"⚠️  WARNING: Only {len(numeric_features)} numeric features!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            raise ValueError("Too few numeric features. Aborting.")
    
    # Check missing values
    print(f"\nMissing values:")
    missing = df[numeric_features + [TARGET_COL]].isnull().sum()
    if missing.sum() == 0:
        print("  No missing values detected.")
    else:
        for col, count in missing[missing > 0].items():
            pct = 100 * count / len(df)
            print(f"  - {col}: {count} ({pct:.2f}%)")
            if pct > 20:
                print(f"    ⚠️  WARNING: > 20% missing!")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    raise ValueError(f"Too many missing values in {col}. Aborting.")
    
    # Analyze energy distribution
    print(f"\nEnergy distribution:")
    print(f"  Min:    {df[TARGET_COL].min():.4f}")
    print(f"  Max:    {df[TARGET_COL].max():.4f}")
    print(f"  Mean:   {df[TARGET_COL].mean():.4f}")
    print(f"  Median: {df[TARGET_COL].median():.4f}")
    print(f"  Std:    {df[TARGET_COL].std():.4f}")
    
    skew = df[TARGET_COL].skew()
    print(f"  Skew:   {skew:.4f}")
    if abs(skew) > 2:
        print(f"    ⚠️  WARNING: Highly skewed distribution (|skew| > 2)!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            raise ValueError("Energy distribution is highly skewed. Aborting.")
    
    print(f"\n✓ Dataset inspection complete. Ready to proceed.")
    
    return df, numeric_features


# ==============================================================================
# PHASE 2: PROXY TARGET CONSTRUCTION
# ==============================================================================
def split_data(df, numeric_features, target_col, seed=42, train_frac=0.60, val_frac=0.20):
    """
    Phase 2: Split data using artist-based grouping and create binary target.
    
    Uses 60/20/20 split (changed from 70/15/15) for Assignment 6.
    Binary target is created using median split from training data only.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: PROXY TARGET CONSTRUCTION")
    print("=" * 80)
    
    # Split by artist groups
    train_mask, val_mask, test_mask = split_by_group(df, "artist_name", seed, train_frac, val_frac)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    test_frac = 1 - train_frac - val_frac
    print(f"\nSplit Strategy: GROUP-BASED split by artist with seed={seed}")
    print(f"Proportions: {train_frac*100:.0f}% train / {val_frac*100:.0f}% validation / {test_frac*100:.0f}% test")
    
    print(f"\nSplit sizes:")
    print(f"  - Training:   {len(train_df):5d} songs from {train_df['artist_name'].nunique():4d} artists ({100*len(train_df)/len(df):.1f}%)")
    print(f"  - Validation: {len(val_df):5d} songs from {val_df['artist_name'].nunique():4d} artists ({100*len(val_df)/len(df):.1f}%)")
    print(f"  - Test:       {len(test_df):5d} songs from {test_df['artist_name'].nunique():4d} artists ({100*len(test_df)/len(df):.1f}%)")
    
    # Verify no artist overlap
    train_artists = set(train_df['artist_name'].unique())
    val_artists = set(val_df['artist_name'].unique())
    test_artists = set(test_df['artist_name'].unique())
    assert len(train_artists & val_artists) == 0, "Artist overlap between train and val!"
    assert len(train_artists & test_artists) == 0, "Artist overlap between train and test!"
    assert len(val_artists & test_artists) == 0, "Artist overlap between val and test!"
    print("  ✓ Verified: No artist overlap between splits")
    
    # Check split percentages
    if len(train_df) / len(df) < 0.10 or len(val_df) / len(df) < 0.10 or len(test_df) / len(df) < 0.10:
        print("  ⚠️  WARNING: One or more splits has < 10% of data!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            raise ValueError("Split irregularity detected. Aborting.")
    
    # Create binary target using training set median
    train_energy_median = train_df[target_col].median()
    print(f"\nCreating binary target (high energy vs low energy):")
    print(f"  Using training set median: {train_energy_median:.4f}")
    print(f"  High energy = 1 if energy >= {train_energy_median:.4f}, else 0")
    
    y_train = (train_df[target_col] >= train_energy_median).astype(int)
    y_val = (val_df[target_col] >= train_energy_median).astype(int)
    y_test = (test_df[target_col] >= train_energy_median).astype(int)
    
    # Verify class balance
    print(f"\nClass distribution:")
    train_counts = y_train.value_counts().sort_index()
    val_counts = y_val.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    
    print(f"  Training set:")
    print(f"    - Low Energy (0):  {train_counts[0]} ({100*train_counts[0]/len(y_train):.1f}%)")
    print(f"    - High Energy (1): {train_counts[1]} ({100*train_counts[1]/len(y_train):.1f}%)")
    
    train_balance = train_counts[1] / len(y_train)
    if train_balance < 0.45 or train_balance > 0.55:
        print(f"    ⚠️  WARNING: Class balance deviates from 50/50 ({train_balance*100:.1f}% positive class)!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            raise ValueError("Significant class imbalance detected. Aborting.")
    
    # Extract numeric features only
    X_train = train_df[numeric_features].copy()
    X_val = val_df[numeric_features].copy()
    X_test = test_df[numeric_features].copy()
    
    print(f"\n✓ Split complete. Feature matrix shape: ({X_train.shape[0]}, {X_train.shape[1]})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# PHASE 3: PREPROCESSING PIPELINE
# ==============================================================================
def build_pipeline(X_train, C=1.0):
    """
    Phase 3: Build preprocessing + model pipeline for numeric features only.
    
    Simplified for Assignment 6: All features are numeric, so no ColumnTransformer needed.
    Just: imputation → scaling → logistic regression.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: PREPROCESSING PIPELINE")
    print("=" * 80)
    
    print(f"\nFeature matrix:")
    print(f"  - Shape: {X_train.shape}")
    print(f"  - All features are numeric (no categorical encoding needed)")
    
    # Simple pipeline for numeric features only
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=C, max_iter=1000, random_state=SEED))
    ])
    
    print(f"\nPipeline summary:")
    print(f"  1. Imputation: Median (for any missing values)")
    print(f"  2. Scaling: StandardScaler (zero mean, unit variance)")
    print(f"  3. Classification: LogisticRegression (C={C}, max_iter=1000)")
    
    print(f"\nKey benefit: Pipeline learns preprocessing from TRAINING data only!")
    print(f"  This prevents data leakage from validation/test sets.")
    
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
    print(f"PR AUC:    {pr_auc:.4f} ⭐ (PRIMARY METRIC)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


# ==============================================================================
# PHASE 4: BASELINE MODEL ESTABLISHMENT
# ==============================================================================
def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    Phase 4: Train baseline model with default C=1.0 and diagnose overfitting.
    
    Uses PR-AUC as primary metric since we're focusing on model quality, not threshold optimization.
    """
    print("\n" + "=" * 80)
    print("PHASE 4: BASELINE MODEL ESTABLISHMENT")
    print("=" * 80)
    
    print(f"\nBuilding baseline model with default C=1.0...")
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    
    # Build baseline pipeline
    baseline_pipeline = build_pipeline(X_train, C=1.0)
    
    # Train
    baseline_pipeline.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = baseline_pipeline.predict(X_train)
    y_train_proba = baseline_pipeline.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_metrics(y_train, y_train_pred, y_train_proba, dataset_name="Training")
    
    # Evaluate on validation set
    y_val_pred = baseline_pipeline.predict(X_val)
    y_val_proba = baseline_pipeline.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_metrics(y_val, y_val_pred, y_val_proba, dataset_name="Validation")
    
    # Diagnose overfitting using PR-AUC
    train_pr_auc = train_metrics['pr_auc']
    val_pr_auc = val_metrics['pr_auc']
    gap = train_pr_auc - val_pr_auc
    
    print("\n" + "=" * 80)
    print("BASELINE DIAGNOSIS")
    print("=" * 80)
    print(f"\nTrain PR-AUC:   {train_pr_auc:.4f}")
    print(f"Val PR-AUC:     {val_pr_auc:.4f}")
    print(f"Gap:            {gap:.4f}")
    
    # Apply diagnostic rules
    if train_pr_auc < 0.5 and val_pr_auc < 0.5 and gap < 0.05:
        diagnosis = "UNDERFITTING"
        print(f"\n→ UNDERFITTING: Both scores are low. Model too simple.")
    elif train_pr_auc > 0.7 and gap > 0.05:
        diagnosis = "OVERFITTING"
        print(f"\n→ OVERFITTING: Large gap ({gap:.4f}). Model memorized training data.")
    elif gap < 0.05:
        diagnosis = "ACCEPTABLE FIT"
        print(f"\n→ ACCEPTABLE FIT: Small gap ({gap:.4f}). Good generalization.")
    else:
        diagnosis = "MIXED"
        print(f"\n→ MIXED: Some overfitting detected (gap = {gap:.4f})")
    
    # Comparison table
    print(f"\nSecondary Metrics Comparison:")
    print(f"{'Metric':<12} {'Training':<12} {'Validation':<12} {'Gap':<12}")
    print("-" * 50)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        train_val = train_metrics[metric]
        val_val = val_metrics[metric]
        metric_gap = train_val - val_val
        print(f"{metric.upper():<12} {train_val:<12.4f} {val_val:<12.4f} {metric_gap:+.4f}")
    
    return baseline_pipeline, train_metrics, val_metrics, diagnosis


# ==============================================================================
# PHASE 5: VALIDATION CURVE FOR REGULARIZATION TUNING
# ==============================================================================
def plot_validation_curve(X_train, y_train):
    """
    Phase 5: Plot validation curve to see how C affects train/val performance.
    
    Lower C = more regularization, higher C = less regularization.
    """
    print("\n" + "=" * 80)
    print("PHASE 5: VALIDATION CURVE FOR REGULARIZATION TUNING")
    print("=" * 80)
    
    # Define C values to test (log scale range)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    
    print(f"\nTesting C values: {C_values}")
    print(f"Using 5-fold CV with PR-AUC scoring...")
    print(f"This will perform 6 C values × 5 folds = 30 fits")
    
    # Build a pipeline for validation_curve
    pipeline = build_pipeline(X_train, C=1.0)  # C will be overridden
    
    # Compute validation curve (uses CV internally)
    train_scores, val_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name='classifier__C',
        param_range=C_values,
        cv=5,  # 5-fold cross-validation
        scoring='average_precision',  # PR-AUC
        n_jobs=1  # Use 1 to avoid multiprocessing issues in sandbox
    )
    
    # Calculate mean and std across CV folds
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    # Find best C
    best_idx = np.argmax(val_scores_mean)
    best_C = C_values[best_idx]
    best_val_score = val_scores_mean[best_idx]
    
    print(f"\nResults:")
    print(f"{'C':<10} {'Train PR-AUC':<15} {'Val PR-AUC':<15} {'Gap':<10}")
    print("-" * 50)
    for i, C in enumerate(C_values):
        train_mean = train_scores_mean[i]
        val_mean = val_scores_mean[i]
        gap = train_mean - val_mean
        marker = " <-- BEST" if i == best_idx else ""
        print(f"{C:<10.3f} {train_mean:<15.4f} {val_mean:<15.4f} {gap:<10.4f}{marker}")
    
    # Plot the validation curve
    plt.figure(figsize=(10, 6))
    
    # Plot training scores with error bars
    plt.plot(C_values, train_scores_mean, 'o-', color='blue', 
             label='Training PR-AUC', linewidth=2, markersize=8)
    plt.fill_between(C_values, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.2, color='blue')
    
    # Plot validation scores with error bars
    plt.plot(C_values, val_scores_mean, 'o-', color='red',
             label='Validation PR-AUC', linewidth=2, markersize=8)
    plt.fill_between(C_values,
                     val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std,
                     alpha=0.2, color='red')
    
    # Mark the best C value
    plt.axvline(x=best_C, color='green', linestyle='--', linewidth=2,
                label=f'Best C = {best_C}')
    
    # Formatting
    plt.xscale('log')  # Log scale for C values
    plt.xlabel('Regularization Strength (C)', fontsize=12)
    plt.ylabel('PR-AUC (Average Precision)', fontsize=12)
    plt.title('Validation Curve: L2 Regularization Strength (C)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('validation_curve.png', dpi=150)
    print(f"\n  → Saved: validation_curve.png")
    
    # Check for overfitting
    print(f"\nBest C: {best_C} (Val PR-AUC = {best_val_score:.4f})")
    
    gap_at_best = train_scores_mean[best_idx] - val_scores_mean[best_idx]
    print(f"Gap at best C: {gap_at_best:.4f}")
    
    # Check if overfitting starts
    overfitting_detected = False
    for i in range(len(C_values) - 1):
        if val_scores_mean[i + 1] <= val_scores_mean[i] and train_scores_mean[i + 1] > train_scores_mean[i]:
            overfitting_detected = True
            print(f"\nOverfitting detected starting around C = {C_values[i + 1]}")
            print(f"  (Validation score decreases while training score increases)")
            break
    
    if not overfitting_detected:
        print(f"\nNo clear overfitting in tested range.")
        print(f"  Model seems well-regularized across all C values.")
    
    return best_C, train_scores_mean, val_scores_mean, C_values


# ==============================================================================
# PHASE 6: GRID SEARCH FOR HYPERPARAMETER TUNING
# ==============================================================================
def run_grid_search(X_train, y_train):
    """
    Phase 6: Run grid search over C values using 5-fold CV.
    
    Small, disciplined search: only 4 C values = 20 total fits.
    """
    print("\n" + "=" * 80)
    print("PHASE 6: GRID SEARCH FOR HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Define small, focused grid based on validation curve insights
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10]
    }
    
    print(f"\nGrid search:")
    print(f"  C values: {param_grid['classifier__C']}")
    print(f"  5-fold CV, PR-AUC scoring")
    print(f"  Total fits: {len(param_grid['classifier__C']) * 5}")
    
    # Build pipeline
    pipeline = build_pipeline(X_train, C=1.0)  # C will be tuned
    
    print(f"\nRunning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='average_precision',  # PR-AUC
        n_jobs=1,  # Use 1 to avoid multiprocessing issues in sandbox
        return_train_score=True  # To see train vs val behavior
    )
    
    # Fit on training data only (no test/validation leakage)
    grid_search.fit(X_train, y_train)
    
    # Extract results
    best_C = grid_search.best_params_['classifier__C']
    best_score = grid_search.best_score_
    best_pipeline = grid_search.best_estimator_
    
    print(f"\nGrid Search Results:")
    print(f"  - Best C: {best_C}")
    print(f"  - Best CV PR-AUC: {best_score:.4f}")
    
    # Print all results
    print(f"\nAll Grid Search Results:")
    print(f"{'C':<10} {'Mean CV PR-AUC':<18} {'Std CV PR-AUC':<18} {'Train PR-AUC':<15}")
    print("-" * 65)
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    for i, C in enumerate(param_grid['classifier__C']):
        # Find the row for this C value
        mask = results_df['param_classifier__C'] == C
        if mask.any():
            row = results_df[mask].iloc[0]
            mean_score = row['mean_test_score']
            std_score = row['std_test_score']
            train_score = row['mean_train_score']
            marker = " <-- BEST" if C == best_C else ""
            print(f"{C:<10.2f} {mean_score:<18.4f} {std_score:<18.4f} {train_score:<15.4f}{marker}")
    
    # Compare to baseline
    if 1.0 in param_grid['classifier__C']:
        baseline_mask = results_df['param_classifier__C'] == 1.0
        if baseline_mask.any():
            baseline_row = results_df[baseline_mask].iloc[0]
            baseline_cv_score = baseline_row['mean_test_score']
            improvement = best_score - baseline_cv_score
            print(f"\nBaseline (C=1.0): {baseline_cv_score:.4f}")
            print(f"Best (C={best_C}):  {best_score:.4f}")
            print(f"Change: {improvement:+.4f}")
            
            if abs(improvement) < 0.01:
                print(f"→ Baseline was already near-optimal (change < 0.01)")
            elif improvement > 0.01:
                print(f"→ Tuning helped! Improvement of {improvement:.4f}")
            else:
                print(f"→ Baseline was better")
    
    return best_pipeline, best_C, grid_search


# ==============================================================================
# PHASE 7: FINAL MODEL EVALUATION
# ==============================================================================
def final_model_evaluation(best_C, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Phase 7: Compare baseline vs tuned model on test set (first time using test set!).
    
    Train both on train+val, then evaluate on test.
    """
    print("\n" + "=" * 80)
    print("PHASE 7: FINAL MODEL EVALUATION (BASELINE VS TUNED)")
    print("=" * 80)
    
    print(f"\n⚠️  Using test set for first time!")
    
    # Combine train+val for final training
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)
    print(f"\nTraining on {len(X_train_final)} samples (train+val), testing on {len(X_test)} samples")
    
    # Train both models
    print(f"\nTraining baseline (C=1.0)...")
    baseline_pipeline_final = build_pipeline(X_train_final, C=1.0)
    baseline_pipeline_final.fit(X_train_final, y_train_final)
    
    print(f"Training tuned (C={best_C})...")
    tuned_pipeline_final = build_pipeline(X_train_final, C=best_C)
    tuned_pipeline_final.fit(X_train_final, y_train_final)
    
    # Baseline predictions
    y_test_pred_baseline = baseline_pipeline_final.predict(X_test)
    y_test_proba_baseline = baseline_pipeline_final.predict_proba(X_test)[:, 1]
    baseline_test_metrics = evaluate_metrics(
        y_test, y_test_pred_baseline, y_test_proba_baseline, 
        dataset_name="Test (Baseline C=1.0)"
    )
    
    # Tuned predictions
    y_test_pred_tuned = tuned_pipeline_final.predict(X_test)
    y_test_proba_tuned = tuned_pipeline_final.predict_proba(X_test)[:, 1]
    tuned_test_metrics = evaluate_metrics(
        y_test, y_test_pred_tuned, y_test_proba_tuned,
        dataset_name=f"Test (Tuned C={best_C})"
    )
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("BASELINE VS TUNED COMPARISON (TEST SET)")
    print("=" * 80)
    
    print(f"\n{'Metric':<15} {'Baseline (C=1.0)':<18} {'Tuned (C=' + str(best_C) + ')':<18} {'Change':<12}")
    print("-" * 65)
    
    comparison = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        baseline_val = baseline_test_metrics[metric]
        tuned_val = tuned_test_metrics[metric]
        change = tuned_val - baseline_val
        comparison[metric] = {
            'baseline': baseline_val,
            'tuned': tuned_val,
            'change': change
        }
        
        # Highlight PR-AUC (primary metric)
        if metric == 'pr_auc':
            print(f"{metric.upper():<15} {baseline_val:<18.4f} {tuned_val:<18.4f} {change:+.4f} ⭐ (PRIMARY)")
        else:
            print(f"{metric.upper():<15} {baseline_val:<18.4f} {tuned_val:<18.4f} {change:+.4f}")
    
    # Interpretation
    pr_auc_change = comparison['pr_auc']['change']
    
    print(f"\nDid regularization help?")
    if abs(pr_auc_change) < 0.01:
        print(f"→ No significant change ({pr_auc_change:+.4f}). Baseline was already good.")
    elif pr_auc_change > 0.01:
        print(f"→ Yes! Improved by {pr_auc_change:+.4f}")
    else:
        print(f"→ Slightly worse ({pr_auc_change:+.4f}), but within noise")
    
    print(f"\nWhat does this mean?")
    if abs(pr_auc_change) < 0.01:
        print(f"→ Model was already well-regularized.")
        print(f"→ Dataset is large enough ({len(X_train_final)} samples) or signal is strong.")
    elif pr_auc_change > 0.01:
        print(f"→ Regularization helped reduce overfitting.")
    else:
        print(f"→ Baseline C=1.0 was probably optimal.")
    
    return comparison, baseline_pipeline_final, tuned_pipeline_final


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


# ==============================================================================
# PHASE 8: REFLECTION & INTERPRETATION DIAGNOSTICS
# ==============================================================================
def generate_reflection_diagnostics(diagnosis, best_C, comparison, train_metrics, val_metrics, 
                                     baseline_test_metrics, tuned_test_metrics, 
                                     overfitting_detected, dataset_size):
    """
    Phase 8: Generate structured diagnostics to support capstone template reflection.
    
    This function provides interpretation for the writeup's Section 6 (Interpretation and Judgment).
    """
    print("\n" + "=" * 80)
    print("PHASE 8: REFLECTION & INTERPRETATION DIAGNOSTICS")
    print("=" * 80)
    
    print(f"\n1. DID REGULARIZATION MEANINGFULLY CHANGE PERFORMANCE?")
    pr_auc_change = comparison['pr_auc']['change']
    if abs(pr_auc_change) < 0.01:
        print(f"   → NO. Change was {pr_auc_change:+.4f}, which is negligible.")
        print(f"   → Baseline C=1.0 was already well-regularized.")
    elif pr_auc_change > 0.01:
        print(f"   → YES. Regularization improved PR-AUC by {pr_auc_change:+.4f}.")
        print(f"   → Tuning C={best_C} reduced overfitting.")
    else:
        print(f"   → NO. Baseline was slightly better ({pr_auc_change:+.4f}).")
        print(f"   → C=1.0 was probably optimal for this dataset.")
    
    print(f"\n2. WAS OVERFITTING OBSERVED?")
    if overfitting_detected:
        print(f"   → YES. Validation curve showed overfitting at high C values.")
        print(f"   → Training score increased while validation score decreased.")
        print(f"   → This suggests some features may be noisy.")
    else:
        print(f"   → NO. Validation curves were relatively flat across C values.")
        print(f"   → Model generalized well regardless of regularization strength.")
    
    print(f"\n   Baseline diagnosis: {diagnosis}")
    train_val_gap = train_metrics['pr_auc'] - val_metrics['pr_auc']
    print(f"   Train-Val gap at baseline: {train_val_gap:.4f}")
    if train_val_gap < 0.05:
        print(f"   → Small gap indicates good generalization at baseline.")
    
    print(f"\n3. DATASET SIZE IMPACT:")
    print(f"   → Dataset: {dataset_size} songs total")
    print(f"   → Training: ~{int(dataset_size * 0.6)} songs (60%)")
    print(f"   → Large dataset size may make regularization less critical.")
    if abs(pr_auc_change) < 0.01:
        print(f"   → This explains why different C values performed similarly.")
        print(f"   → Dataset provides sufficient signal for generalization.")
    
    print(f"\n4. FEATURE REPRESENTATION STABILITY:")
    val_pr_auc = val_metrics['pr_auc']
    if val_pr_auc > 0.7:
        print(f"   → High validation PR-AUC ({val_pr_auc:.4f}) indicates strong features.")
    else:
        print(f"   → Moderate validation PR-AUC ({val_pr_auc:.4f}).")
    
    if not overfitting_detected and abs(pr_auc_change) < 0.01:
        print(f"   → Flat validation curves suggest:")
        print(f"     • Features have low variance after standardization")
        print(f"     • Linear boundary is sufficient")
        print(f"     • Representation is stable")
    
    print(f"\n5. IMPLICATIONS FOR RECOMMENDATION TASK:")
    if abs(pr_auc_change) < 0.01 and not overfitting_detected:
        print(f"   → Feature representation is STABLE and ready for recommendation.")
        print(f"   → Large dataset (28k songs) provides sufficient signal.")
        print(f"   → Similarity-based recommendation should work well with these features.")
    elif pr_auc_change > 0.01 or overfitting_detected:
        print(f"   → Some features may be noisy.")
        print(f"   → Consider feature selection or dimensionality reduction.")
        print(f"   → Normalization/scaling is important for similarity metrics.")
    
    if val_pr_auc > 0.7:
        print(f"   → Strong classification performance validates feature quality.")
        print(f"   → Features capture meaningful patterns for similarity search.")
    
    print(f"\n6. KEY TAKEAWAY:")
    if abs(pr_auc_change) < 0.01:
        print(f"   → Regularization was NOT critical for this dataset.")
        print(f"   → Reason: Large dataset size + stable features = good generalization.")
        print(f"   → This is ACADEMICALLY VALID: Not all datasets need heavy regularization.")
    else:
        print(f"   → Regularization WAS important for optimal performance.")
        print(f"   → Tuning complexity control improved model quality.")
    
    print(f"\n" + "=" * 80)


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    """
    Main function orchestrating the complete regularization and hyperparameter tuning workflow.
    
    Assignment 6 Part B: Focus on complexity control and overfitting diagnosis.
    """
    print("\n" + "=" * 80)
    print("MUSIC RECOMMENDATION SYSTEM - REGULARIZATION & HYPERPARAMETER CONTROL")
    print("Assignment 6 - Part B: Regularization and Complexity Control")
    print("=" * 80)
    
    print("\nProxy Task: Predicting High Energy vs Low Energy (binary classification)")
    print("\nThis Week's Focus:")
    print("  - Establish baseline model and diagnose overfitting")
    print("  - Generate validation curves to visualize bias-variance tradeoff")
    print("  - Perform disciplined hyperparameter search (grid search)")
    print("  - Compare baseline vs tuned models on test set")
    print("  - Assess whether regularization matters for this dataset/representation")
    print("=" * 80)
    
    # Phase 1: Dataset Inspection
    df, numeric_features = inspect_dataset(CSV_PATH)
    
    # Phase 2: Proxy Target Construction (60/20/20 split with seed=42)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, numeric_features, TARGET_COL, seed=SEED, train_frac=0.60, val_frac=0.20
    )
    
    # Phase 3: Build Pipeline (implicit in baseline training)
    # Phase 4: Baseline Model Establishment
    baseline_pipeline, train_metrics, val_metrics, diagnosis = train_baseline_model(
        X_train, y_train, X_val, y_val
    )
    
    # Phase 5: Validation Curve Analysis
    best_C_curve, train_scores_mean, val_scores_mean, C_values = plot_validation_curve(
        X_train, y_train
    )
    
    # Check if overfitting was detected in validation curve
    overfitting_detected = False
    for i in range(len(C_values) - 1):
        if val_scores_mean[i + 1] <= val_scores_mean[i] and train_scores_mean[i + 1] > train_scores_mean[i]:
            overfitting_detected = True
            break
    
    # Phase 6: Grid Search
    best_pipeline, best_C_grid, grid_search = run_grid_search(X_train, y_train)
    
    # Phase 7: Final Model Evaluation (baseline vs tuned on test set)
    comparison, baseline_final, tuned_final = final_model_evaluation(
        best_C_grid, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Get test set metrics for reflection
    y_test_pred_baseline = baseline_final.predict(X_test)
    y_test_proba_baseline = baseline_final.predict_proba(X_test)[:, 1]
    baseline_test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred_baseline),
        'precision': precision_score(y_test, y_test_pred_baseline),
        'recall': recall_score(y_test, y_test_pred_baseline),
        'f1': f1_score(y_test, y_test_pred_baseline),
        'roc_auc': roc_auc_score(y_test, y_test_proba_baseline),
        'pr_auc': average_precision_score(y_test, y_test_proba_baseline)
    }
    
    y_test_pred_tuned = tuned_final.predict(X_test)
    y_test_proba_tuned = tuned_final.predict_proba(X_test)[:, 1]
    tuned_test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred_tuned),
        'precision': precision_score(y_test, y_test_pred_tuned),
        'recall': recall_score(y_test, y_test_pred_tuned),
        'f1': f1_score(y_test, y_test_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_test_proba_tuned),
        'pr_auc': average_precision_score(y_test, y_test_proba_tuned)
    }
    
    # Phase 8: Reflection & Interpretation Diagnostics
    generate_reflection_diagnostics(
        diagnosis, best_C_grid, comparison,
        train_metrics, val_metrics,
        baseline_test_metrics, tuned_test_metrics,
        overfitting_detected, len(df)
    )
    
    # Final Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nSummary of outputs generated:")
    print("  1. validation_curve.png - Visualization of bias-variance tradeoff")
    print("  2. Console output with all diagnostics, metrics, and interpretations")
    
    print("\nKey Takeaways:")
    print("  - Split strategy: 60/20/20 artist-based, seed=42")
    print(f"  - Baseline diagnosis: {diagnosis}")
    print(f"  - Validation curve: tested C = {C_values}")
    if overfitting_detected:
        print(f"  - Overfitting: DETECTED at high C values")
    else:
        print(f"  - Overfitting: NOT DETECTED (flat validation curves)")
    print(f"  - Grid search: disciplined (4 C × 5 CV = 20 fits)")
    print(f"  - Best C selected: {best_C_grid}")
    print(f"  - Primary metric: PR-AUC (appropriate for binary classification)")
    print(f"  - Test set: used only once for final comparison")
    
    pr_auc_change = comparison['pr_auc']['change']
    if abs(pr_auc_change) < 0.01:
        print(f"  - Regularization impact: MINIMAL (change = {pr_auc_change:+.4f})")
        print(f"  - Conclusion: Baseline was already well-regularized")
    elif pr_auc_change > 0.01:
        print(f"  - Regularization impact: SIGNIFICANT (improvement = {pr_auc_change:+.4f})")
        print(f"  - Conclusion: Tuning helped reduce overfitting")
    else:
        print(f"  - Regularization impact: NEGATIVE (change = {pr_auc_change:+.4f})")
        print(f"  - Conclusion: Baseline C=1.0 was optimal")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

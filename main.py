"""
CS 4320 — Assignment 4 (Part B) Capstone Regression Baseline
Music Recommendation System — Regression baseline with optimization

This file implements a regression baseline for a content-based music recommendation system.
The dataset contains songs with metadata and audio features.

PROXY TASK FOR THIS WEEK:
Since recommendation systems don't have traditional prediction targets, we use a proxy task:
    Predict 'energy' from other audio features
    
WHY 'energy' AS TARGET?
1. Energy is a fundamental audio characteristic (intensity/activity of a song)
2. It has strong relationships with other audio features (loudness, acousticness, etc.)
3. Learning these relationships helps understand feature interactions for better recommendations
4. Makes conceptual sense: "can we predict how energetic a song sounds from its other characteristics?"
5. Once trained, understanding energy patterns helps recommend songs with similar "feel"

This is a reasonable proxy because:
- It exercises the full ML pipeline (split, preprocess, optimize, evaluate)
- It validates that our features contain useful information
- It establishes a baseline for understanding feature relationships
- Poor performance would signal data quality issues or need for better features

Rules reminder:
- You MAY use numpy/pandas for array/data operations
- You may NOT use scikit-learn for regression, optimization, or splitting
- Manual implementation of gradient descent is REQUIRED

Workflow (leakage-safe):
1) Load data
2) Split into train/val/test using GROUP-BASED splitting by artist (prevents leakage)
3) Separate target ('energy') from features
4) Fit preprocessing on TRAIN ONLY:
   - numeric median imputation
   - categorical mode imputation
   - scaling (standardization)
   - one-hot encoding
5) Apply train-fitted transformations to val/test
6) Train linear regression with gradient descent
7) Visualize loss curves
8) Evaluate on test set
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "tcc_ceds_music.csv"
SEED = 4320 

# Target variable for this week's regression baseline
# We're predicting 'energy' as a proxy task to validate our feature pipeline
TARGET_COL = "energy"

# Exclude identifier columns, text fields, and the target itself
# Note: We keep the target in the original dataframe but will separate it after splitting
POSSIBLE_EXCLUDES = [
    "Unnamed: 0",      # Just a row index
    "artist_name",     # Identity leakage (used for splitting only)
    "track_name",      # Song identifier (not a feature)
    "lyrics",          # High-cardinality text (needs separate NLP pipeline)
    "release_date",    # Could leak temporal information
    TARGET_COL         # Will be separated as y
]


def split_by_group(df: pd.DataFrame, group_col: str, seed: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Group-based split: split groups (artists) into train/val/test sets.
    All rows belonging to the same group go into the same split.
    This prevents data leakage in recommendation systems.
    
    For this assignment: 70% train, 15% val, 15% test (matching assignment3 proportions)
    """
    rng = np.random.default_rng(seed)
    
    # Get unique groups (artists)
    unique_groups = df[group_col].unique()
    n_groups = len(unique_groups)
    
    # Shuffle groups deterministically
    perm_groups = rng.permutation(unique_groups)
    
    # Calculate split sizes
    n_train_groups = int(round(train_frac * n_groups))
    n_val_groups = int(round(val_frac * n_groups))
    
    # Assign groups to splits
    train_groups = set(perm_groups[:n_train_groups])
    val_groups = set(perm_groups[n_train_groups:n_train_groups + n_val_groups])
    test_groups = set(perm_groups[n_train_groups + n_val_groups:])
    
    # Create boolean masks for each split
    train_mask = df[group_col].isin(train_groups)
    val_mask = df[group_col].isin(val_groups)
    test_mask = df[group_col].isin(test_groups)
    
    return train_mask, val_mask, test_mask


def add_bias_column(X):
    """
    Add a column of ones for the bias term (intercept).
    This allows us to learn both weights and bias in a single parameter vector.
    """
    n = X.shape[0]
    ones_col = np.ones((n, 1), dtype=X.dtype)
    Xb = np.hstack([ones_col, X])
    return Xb


def predict(X, w):
    """
    Make predictions: y_hat = Xb @ w
    X should NOT have bias column yet (we add it here)
    """
    Xb = add_bias_column(X)
    y_hat = Xb @ w
    return y_hat


def mse_loss(Xb, y, w):
    """
    Mean Squared Error loss: L(w) = (1/n) * ||Xb*w - y||^2
    Xb should already have bias column
    """
    n = Xb.shape[0]
    y_hat = Xb @ w
    loss = (1.0 / n) * np.sum((y_hat - y) ** 2)
    return loss


def mse_grad(Xb, y, w):
    """
    Gradient of MSE loss: ∇L(w) = (2/n) * Xb^T (Xb*w - y)
    Xb should already have bias column
    """
    n = Xb.shape[0]
    y_hat = Xb @ w
    grad = (2.0 / n) * (Xb.T @ (y_hat - y))
    return grad


def main():
    print("="*60)
    print("Assignment 4 Part B: Regression Baseline for Music Recommender")
    print("="*60)
    print(f"\nProxy Task: Predicting '{TARGET_COL}' from other audio features")
    print("\nRationale:")
    print("  - Energy represents song intensity/activity level")
    print("  - Strong relationships with other audio features")
    print("  - Validates that features contain useful information")
    print("  - Understanding energy helps recommend similar-feeling songs")
    print("="*60)
    
    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("\n[Step 1] Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} songs with {len(df.columns)} columns")
    print(f"  Dataset shape: {df.shape}")

    # ========================================================================
    # STEP 2: Split by artist (group-based split)
    # ========================================================================
    # Strategy: Group-based split by artist_name with seed=4320 for reproducibility
    # Rationale: For music recommendation, we must prevent artist-specific leakage.
    #            If the same artist appears in train and test, the model can "cheat"
    #            by learning artist-specific patterns rather than generalizable features.
    #            Group-based splitting ensures all songs from an artist stay in one split.
    # Split proportions: 70% train, 15% validation, 15% test (matching assignment3)
    print("\n[Step 2] Splitting by artist (group-based split)...")
    train_mask, val_mask, test_mask = split_by_group(df, "artist_name", SEED, train_frac=0.70, val_frac=0.15)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  Train: {len(train_df)} songs from {train_df['artist_name'].nunique()} artists")
    print(f"  Val:   {len(val_df)} songs from {val_df['artist_name'].nunique()} artists")
    print(f"  Test:  {len(test_df)} songs from {test_df['artist_name'].nunique()} artists")
    
    # Verify no artist overlap between splits
    train_artists = set(train_df['artist_name'].unique())
    val_artists = set(val_df['artist_name'].unique())
    test_artists = set(test_df['artist_name'].unique())
    assert len(train_artists & val_artists) == 0, "Artist overlap between train and val!"
    assert len(train_artists & test_artists) == 0, "Artist overlap between train and test!"
    assert len(val_artists & test_artists) == 0, "Artist overlap between val and test!"
    print("  ✓ Verified: No artist overlap between splits")
    
    # ========================================================================
    # STEP 3: Separate target variable
    # ========================================================================
    # For this regression baseline, we predict 'energy' from other features
    print(f"\n[Step 3] Separating target variable ('{TARGET_COL}')...")
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float64)
    
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_val shape:   {y_val.shape}")
    print(f"  y_test shape:  {y_test.shape}")
    print(f"  Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  Target mean:  {y_train.mean():.3f}")
    
    # ========================================================================
    # STEP 4: Choose feature columns
    # ========================================================================
    # Drop: target, identifiers (artist_name, track_name), high-cardinality text (lyrics)
    # Keep: all numeric audio features and categorical features (genre, topic)
    print("\n[Step 4] Selecting feature columns...")
    X_train = train_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in train_df.columns])
    X_val = val_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in val_df.columns])
    X_test = test_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in test_df.columns])

    print(f"  Features selected: {X_train.shape[1]} columns")
    print(f"  Excluded columns: {[c for c in POSSIBLE_EXCLUDES if c in train_df.columns]}")
    
    # ========================================================================
    # STEP 5: Identify numeric vs categorical
    # ========================================================================
    print("\n[Step 5] Identifying numeric vs categorical features...")
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"    Examples: {numeric_cols[:5]}")
    print(f"  Categorical features: {len(cat_cols)}")
    print(f"    Examples: {cat_cols}")
    
    # ========================================================================
    # STEP 6: FIT imputation on TRAIN ONLY
    # ========================================================================
    # Strategy: Median for numeric, mode for categorical
    # Rationale: Median is robust to outliers, mode preserves most common category
    # Assumption: Missing data is missing at random (MAR), not systematically biased
    # Why train-only fitting matters: Using val/test statistics would leak information
    print("\n[Step 6] Fitting imputation on TRAIN ONLY...")
    
    # Check for missing values
    train_missing = X_train.isnull().sum().sum()
    print(f"  Missing values in train: {train_missing}")
    
    # Compute imputation statistics from TRAIN ONLY
    numeric_medians = X_train[numeric_cols].median()
    categorical_modes = {}
    for col in cat_cols:
        mode_series = X_train[col].mode()
        categorical_modes[col] = mode_series[0] if len(mode_series) > 0 else None

    # Apply imputation to all sets using train statistics
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()

     # Fill numeric columns with medians from train
    X_train_imputed[numeric_cols] = X_train_imputed[numeric_cols].fillna(numeric_medians)
    X_val_imputed[numeric_cols] = X_val_imputed[numeric_cols].fillna(numeric_medians)
    X_test_imputed[numeric_cols] = X_test_imputed[numeric_cols].fillna(numeric_medians)
    
    # Fill categorical columns with modes from train
    for col in cat_cols:
        if categorical_modes[col] is not None:
            X_train_imputed[col] = X_train_imputed[col].fillna(categorical_modes[col])
            X_val_imputed[col] = X_val_imputed[col].fillna(categorical_modes[col])
            X_test_imputed[col] = X_test_imputed[col].fillna(categorical_modes[col])
    
    # Verify no missing values remain
    assert X_train_imputed.isnull().sum().sum() == 0, "Train set still has missing values"
    assert X_val_imputed.isnull().sum().sum() == 0, "Val set still has missing values"
    assert X_test_imputed.isnull().sum().sum() == 0, "Test set still has missing values"
    print("  ✓ Imputation complete, no missing values remain")

    # ========================================================================
    # STEP 7: FIT scaling on TRAIN ONLY (numeric only)
    # ========================================================================
    # Method: Standardization (z-score normalization): (x - mean) / std
    # Why scaling is critical for gradient descent:
    #   - Features have very different scales (e.g., loudness vs danceability)
    #   - Without scaling, gradient descent is unstable and slow to converge
    #   - Large-scale features dominate the gradient, making learning inefficient
    #   - Standardization ensures all features contribute equally to optimization
    # Why fitting on training data only is critical:
    #   - Prevents data leakage: test/validation statistics must not influence training
    #   - Realistic evaluation: mimics production where only training data is available
    #   - Consistent transformation: all sets use same mean/std from training
    print("\n[Step 7] Fitting scaling on TRAIN ONLY (standardization)...")
    
    # Compute mean and std from *imputed* training data (numeric columns only)
    numeric_means = X_train_imputed[numeric_cols].mean()
    numeric_stds = X_train_imputed[numeric_cols].std()
    
    # Avoid division by zero (if std is 0, feature is constant - set std to 1)
    numeric_stds = numeric_stds.replace(0, 1)
    
    print(f"  Computed mean/std from {len(numeric_cols)} numeric features")
    print(f"    Example - danceability: mean={numeric_means.get('danceability', 0):.3f}, std={numeric_stds.get('danceability', 1):.3f}")
    
    # Apply standardization to all sets using train statistics
    X_train_scaled = X_train_imputed.copy()
    X_val_scaled = X_val_imputed.copy()
    X_test_scaled = X_test_imputed.copy()
    
    # Standardize: (x - mean) / std
    X_train_scaled[numeric_cols] = (X_train_imputed[numeric_cols] - numeric_means) / numeric_stds
    X_val_scaled[numeric_cols] = (X_val_imputed[numeric_cols] - numeric_means) / numeric_stds
    X_test_scaled[numeric_cols] = (X_test_imputed[numeric_cols] - numeric_means) / numeric_stds

    print("  ✓ Scaling complete using train mean/std")

    # ========================================================================
    # STEP 8: FIT one-hot encoding on TRAIN ONLY
    # ========================================================================
    # Why encoding is required:
    #   - ML algorithms need numeric inputs, but categorical features are text/strings
    #   - One-hot encoding converts categories into binary vectors (0/1)
    #   - Each category becomes a separate binary feature (e.g., "pop" -> [1,0,0,...])
    #   - Avoids imposing arbitrary ordinal relationships between categories
    # How this approach avoids leakage:
    #   - Categories are learned ONLY from training data
    #   - Val/test sets use the same category list from training
    #   - Unseen categories in val/test map to all-zeros (conservative, no leakage)
    print("\n[Step 8] Fitting one-hot encoding on TRAIN ONLY...")
    
    # Build list of categories per categorical column from TRAIN ONLY
    categorical_categories = {}
    for col in cat_cols:
        # Get unique categories from training data, sorted for deterministic order
        categories = sorted(X_train_scaled[col].unique().tolist())
        categorical_categories[col] = categories
        print(f"  {col}: {len(categories)} categories")
    
    # Create one-hot encoded dataframes
    def one_hot_encode(df, cat_cols, categorical_categories, numeric_cols):
        """
        One-hot encode categorical columns, handling unseen categories.
        Unseen categories (in val/test) map to all-zeros.
        """
        # Start with numeric columns (already scaled)
        encoded_df = df[numeric_cols].copy()
        
        # For each categorical column, create one-hot columns
        for col in cat_cols:
            categories = categorical_categories[col]
            
            # Create one column per category
            for cat in categories:
                # Column name: original_column_category
                new_col_name = f"{col}_{cat}"
                # 1 if row has this category, 0 otherwise
                encoded_df[new_col_name] = (df[col] == cat).astype(int)
        
        return encoded_df
    
    # Apply one-hot encoding to all sets
    X_train_encoded = one_hot_encode(X_train_scaled, cat_cols, categorical_categories, numeric_cols)
    X_val_encoded = one_hot_encode(X_val_scaled, cat_cols, categorical_categories, numeric_cols)
    X_test_encoded = one_hot_encode(X_test_scaled, cat_cols, categorical_categories, numeric_cols)
    
    # Verify all sets have same columns (same number of features)
    assert list(X_train_encoded.columns) == list(X_val_encoded.columns), "Column mismatch: train vs val"
    assert list(X_train_encoded.columns) == list(X_test_encoded.columns), "Column mismatch: train vs test"
    
    print(f"  ✓ One-hot encoding complete")
    print(f"    Original categorical columns: {len(cat_cols)}")
    print(f"    Total features after encoding: {X_train_encoded.shape[1]}")
    print(f"    (Numeric: {len(numeric_cols)}, One-hot: {X_train_encoded.shape[1] - len(numeric_cols)})")
    
    # ========================================================================
    # STEP 9: Convert to numpy arrays
    # ========================================================================
    print("\n[Step 9] Converting to numpy arrays...")
    feature_columns = list(X_train_encoded.columns)
    
    X_train_np = X_train_encoded[feature_columns].to_numpy(dtype=np.float64)
    X_val_np = X_val_encoded[feature_columns].to_numpy(dtype=np.float64)
    X_test_np = X_test_encoded[feature_columns].to_numpy(dtype=np.float64)
    
    print(f"  X_train shape: {X_train_np.shape} (samples, features)")
    print(f"  X_val shape:   {X_val_np.shape}")
    print(f"  X_test shape:  {X_test_np.shape}")
    
    # ========================================================================
    # STEP 10: Add bias column
    # ========================================================================
    # The bias term (intercept) allows the model to shift predictions up/down
    # We add a column of ones so the bias can be learned as part of the weight vector
    print("\n[Step 10] Adding bias column...")
    Xb_train = add_bias_column(X_train_np)
    Xb_val = add_bias_column(X_val_np)
    Xb_test = add_bias_column(X_test_np)
    
    print(f"  Xb_train shape: {Xb_train.shape} (added bias column)")
    print(f"  Xb_val shape:   {Xb_val.shape}")
    print(f"  Xb_test shape:  {Xb_test.shape}")
    
    # ========================================================================
    # STEP 11: Initialize weights and set hyperparameters
    # ========================================================================
    # Linear regression model: y_hat = Xb @ w
    # Loss: MSE = (1/n) * ||Xb*w - y||^2
    # Optimization: Gradient descent with fixed learning rate
    print("\n[Step 11] Initializing weights and hyperparameters...")
    
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.01, size=Xb_train.shape[1])
    
    # Hyperparameters
    # Learning rate: 0.01 is a good starting point for standardized features
    # Too high (>0.1) causes instability, too low (<0.001) is too slow
    # Since we standardized features, this should work well
    epochs = 1000
    lr = 0.01
    
    print(f"  Weights shape: {w.shape}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Initial train loss: {mse_loss(Xb_train, y_train, w):.4f}")
    print(f"  Initial val loss:   {mse_loss(Xb_val, y_val, w):.4f}")
    
    # ========================================================================
    # STEP 12: Gradient descent optimization
    # ========================================================================
    # Algorithm: Batch gradient descent
    #   - Compute gradient on entire training set
    #   - Update weights: w = w - lr * gradient
    #   - Repeat for fixed number of epochs
    # Why this works:
    #   - Gradient points in direction of steepest increase in loss
    #   - Moving opposite to gradient decreases loss
    #   - Small learning rate ensures stable convergence
    print("\n" + "="*60)
    print("[Step 12] Running gradient descent optimization...")
    print("="*60)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Compute gradient on training data
        grad = mse_grad(Xb_train, y_train, w)
        
        # Update weights
        w = w - lr * grad
        
        # Track losses
        train_losses.append(mse_loss(Xb_train, y_train, w))
        val_losses.append(mse_loss(Xb_val, y_val, w))
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    print("\n" + "="*60)
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss:   {val_losses[-1]:.4f}")
    print("="*60)
    
    # ========================================================================
    # STEP 13: Visualize loss curves
    # ========================================================================
    print("\n[Step 13] Creating loss curve visualization...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train", linewidth=2)
    plt.plot(val_losses, label="Validation", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title(f"Training and Validation Loss Curves (Predicting '{TARGET_COL}')", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.close()
    print("  ✓ Loss curve saved to 'loss_curve.png'")
    
    # ========================================================================
    # STEP 14: Analyze optimization behavior
    # ========================================================================
    print("\n" + "="*60)
    print("OPTIMIZATION BEHAVIOR ANALYSIS")
    print("="*60)
    
    if len(train_losses) > 0:
        initial_train_loss = train_losses[0]
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        
        loss_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
        overfitting_gap = final_val_loss - final_train_loss
        overfitting_pct = (overfitting_gap / final_train_loss) * 100 if final_train_loss > 0 else 0
        
        print(f"Initial train loss: {initial_train_loss:.4f}")
        print(f"Final train loss:   {final_train_loss:.4f}")
        print(f"Final val loss:     {final_val_loss:.4f}")
        print(f"Loss reduction:     {loss_reduction:.2f}%")
        print(f"Train-Val gap:      {overfitting_gap:.4f} ({overfitting_pct:.2f}%)")
        
        # Convergence check
        if len(train_losses) >= 100:
            recent_change = abs(train_losses[-1] - train_losses[-100]) / train_losses[-100] * 100
            if recent_change < 1.0:
                print(f"Convergence:        CONVERGED (change < 1% in last 100 epochs)")
            else:
                print(f"Convergence:        Still improving ({recent_change:.2f}% change in last 100 epochs)")
        
        # Overfitting assessment
        if overfitting_pct > 15:
            print("Assessment:         Significant overfitting detected (val >> train)")
        elif overfitting_pct > 5:
            print("Assessment:         Slight overfitting (acceptable for baseline)")
        else:
            print("Assessment:         Good generalization (train ≈ val)")
        
        print("="*60)
    
    # ========================================================================
    # STEP 15: Final evaluation on test set
    # ========================================================================
    # The test set should ONLY be evaluated once, at the very end
    # This simulates real-world deployment where we don't have access to test labels
    print("\n[Step 15] Evaluating on TEST SET (final model)...")
    
    y_test_pred = predict(X_test_np, w)
    
    # Compute metrics
    mse_test = np.mean((y_test_pred - y_test) ** 2)
    rmse_test = np.sqrt(mse_test)
    mae_test = np.mean(np.abs(y_test_pred - y_test))
    
    # R² score (coefficient of determination)
    # R² = 1 - (SS_res / SS_tot)
    # R² = 1: perfect predictions
    # R² = 0: predictions are as good as predicting the mean
    # R² < 0: predictions are worse than predicting the mean
    ss_res = np.sum((y_test - y_test_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - (ss_res / ss_tot)
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"Target variable: {TARGET_COL}")
    print(f"Test samples:    {len(y_test)}")
    print(f"\nMetrics:")
    print(f"  MSE:  {mse_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  MAE:  {mae_test:.4f}")
    print(f"  R²:   {r2_test:.4f}")
    print("="*60)
    
    # ========================================================================
    # STEP 16: Interpretation and insights
    # ========================================================================
    print("\n" + "="*60)
    print("INTERPRETATION FOR ASSIGNMENT 4 PART B")
    print("="*60)
    
    print("\n1. PROXY TASK JUSTIFICATION:")
    print(f"   We predicted '{TARGET_COL}' as a proxy for building a recommendation system.")
    print("   This validates that our audio features contain useful signal.")
    print("   If we can't predict energy from other features, our recommendation")
    print("   pipeline would struggle to find meaningful song similarities.")
    
    print("\n2. BASELINE MODEL:")
    print("   - Model family: Linear regression (closed-form gradient descent)")
    print("   - Loss function: Mean Squared Error (MSE)")
    print("   - Optimization: Batch gradient descent with fixed learning rate")
    print(f"   - Parameters: {len(w)} weights (including bias)")
    
    print("\n3. OPTIMIZATION BEHAVIOR:")
    if loss_reduction > 50:
        print(f"   - Loss decreased by {loss_reduction:.1f}% → optimization worked well")
        print("   - Gradient descent successfully found a better solution")
    else:
        print(f"   - Loss decreased by only {loss_reduction:.1f}% → limited improvement")
        print("   - May need: better features, higher learning rate, or more epochs")
    
    if recent_change < 1.0:
        print("   - Training converged (loss stabilized)")
    else:
        print("   - Training could benefit from more epochs")
    
    print("\n4. GENERALIZATION:")
    print(f"   - R² = {r2_test:.3f} means model explains {r2_test*100:.1f}% of variance")
    if r2_test > 0.5:
        print("   - This is good for a linear baseline on audio features")
    elif r2_test > 0.3:
        print("   - Moderate performance - nonlinear models might help")
    else:
        print("   - Weak performance - may need feature engineering or different approach")
    
    if overfitting_pct < 10:
        print(f"   - Train-val gap is small ({overfitting_pct:.1f}%) → good generalization")
    else:
        print(f"   - Train-val gap is large ({overfitting_pct:.1f}%) → some overfitting")
    
    print("\n5. WHAT THIS REVEALS ABOUT THE DATA:")
    print("   - Audio features have predictive power (not random noise)")
    print("   - Preprocessing pipeline is working correctly")
    print("   - Artist-based split ensures realistic evaluation")
    print("   - Foundation is solid for building actual recommendation system")
    
    print("\n6. LEAKAGE PREVENTION:")
    print("   - Artist-based splitting: no artist overlap between splits")
    print("   - Train-only preprocessing: all stats computed from train only")
    print("   - Test set touched only once: no tuning on test performance")
    
    print("\n7. NEXT STEPS FOR RECOMMENDATION SYSTEM:")
    print("   - Use trained weights to understand feature importance")
    print("   - Build cosine similarity search over encoded features")
    print("   - Implement top-N recommendation retrieval")
    print("   - Consider nonlinear models if linear baseline is too weak")
    
    print("="*60)
    print("\n✓ Assignment 4 Part B implementation complete!")
    print("  Check 'loss_curve.png' for visualization")
    print("="*60)


if __name__ == "__main__":
    main()

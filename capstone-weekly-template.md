# Part B – Weekly Capstone Assignment Template
git repo: https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor

This template is used for **each of the 14 Part B capstone submissions** throughout the semester. The structure remains the same every week so that you can focus on *thinking and judgment*, not guessing expectations.

Your goal is not to achieve the best performance, but to **reason carefully about how this week’s machine learning technique applies (or does not apply) to your project**.
---

## 1. Project Context (Brief)

Provide the same short context each week so graders can orient quickly.

* **Project Title:** Content-Based Music Recommender (Spotify Audio Features + Lyrics/Topics)
* **Data Modality:** tabular
* **Task Type:** recommendation (content-based similarity search)
* **One-Sentence Goal:** Given a song, recommend other songs that sound similar using the song's features.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Support Vector Machine (SVM) with linear and RBF kernels; interpretability via decision function and linear weights
* **Key Assumptions of This Technique:**
  * Margin maximization: SVM finds a decision boundary that maximizes the margin between classes
  * Scaling is required (distance-based); C controls margin hardness, gamma (RBF) controls flexibility
  * Linear kernel: interpretable feature weights (coefficients)
  * RBF kernel: no direct weights; flexible boundary
  * class_weight: None or balanced to handle class imbalance if present

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

The recommender is not a classifier, but the proxy task (high vs low energy) assesses whether the feature representation supports discrimination. SVM with ROC AUC indicates how well the features separate the two classes. This informs whether the representation is useful for similarity-based recommendation. Linear SVM also gives interpretable coefficients, which helps understand which features drive the boundary.

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week's technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:** Same proxy task: high energy vs low energy (binary classification) from training-set median split. Each song is a 23-dimensional numeric feature vector (audio and topic-proportion features). Preprocessing: SimpleImputer (median) then StandardScaler.

* **Why this representation was reasonable for this week:** SVM requires scaled features. The proxy task tests whether the representation supports a clear decision boundary. Median split keeps class balance roughly even. StandardScaler ensures fair contribution of all features to the SVM margin.

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
Pipeline: SimpleImputer (median) -> StandardScaler -> SVC. Used RandomizedSearchCV (n_iter=25, cv=5, scoring=roc_auc) over kernel, C, gamma, and class_weight. Fit on train only; evaluated on validation then test once. Added interpretability: decision_function (signed distance to hyperplane), linear SVM coefficients for feature importance, and a 2D PCA plot of validation set colored by decision score to visualize boundary and overlap.

* What you intentionally did *not* attempt and why
Did not do exhaustive grid search; RandomizedSearchCV with 25 iterations kept compute reasonable. Did not add SHAP or other post-hoc interpretability this week; used built-in linear weights and decision scores. Did not change the proxy task or split; kept 60/20/20 artist-based and median-split energy.

* Any constraints encountered (data, labels, compute, time)
No natural prediction target (recommendation system), so binary energy proxy. SVM fit and search take longer than linear models; RandomizedSearchCV with 25 iterations was a practical limit. Artist-group split; no artist overlap. Test set used only once for final evaluation.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

**Best hyperparameters (from RandomizedSearchCV):**
- kernel: linear
- C: ~8.86
- gamma: ~0.00043 (used for search space; linear kernel does not use gamma in decision)
- class_weight: None

**Validation:** ROC AUC 0.9422  
**Test (one-time):** ROC AUC 0.9379

**Test confusion matrix:** TN=2746, FP=466, FN=412, TP=2517

**Interpretability outputs:**
- decision_function: signed distance to hyperplane; larger magnitude means more confident prediction
- Linear weights: top coefficients printed for feature importance
- PCA plot: 2D projection of validation set colored by decision score; shows overlap and boundary softness

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

**Why the method behaved as it did:**

Linear kernel won in search: it gives an interpretable boundary and avoids overfitting in the 23-dimensional space. High validation and test ROC AUC (0.94+) show the features separate high vs low energy well. The small drop from val to test (0.9422 to 0.9379) suggests good generalization.

**Interpretability:**

- decision_function: signed distance to hyperplane; useful for confidence and for understanding how far points are from the boundary
- Linear weights: coefficient magnitude indicates which features push the decision toward high or low energy; supports feature importance reasoning
- PCA plot: 2D view of validation set by decision score shows where classes overlap and where the boundary is soft

**What this reveals:**

The representation supports strong discrimination (high ROC AUC). Linear SVM is sufficient; RBF did not win, which is consistent with 23 features and possible curse of dimensionality. The proxy task confirms the features are useful for distinguishing energy level, which supports using them for similarity-based recommendation.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

I'll keep the preprocessing pipeline (median imputation, StandardScaler) and the 60/20/20 artist split. Linear SVM and its coefficients stay useful for interpretability.

**Limitations:** 23 features is high-dimensional; RBF may overfit in this space. Consider PCA or feature selection to reduce dimensions.

**Next steps:** Dimensionality reduction (e.g., PCA) or other interpretability (e.g., SHAP) if needed. The proxy task has shown the representation supports discrimination; next is to tie that to the actual recommendation use case.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week's technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not a mismatch. SVM is a partial fit: the recommender is not a classifier, but the proxy task assesses whether the representation supports discrimination. ROC AUC shows how well the features separate high vs low energy. That informs whether the representation is useful for similarity-based recommendation. Linear SVM also gave interpretable feature weights, which adds value for understanding the representation.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor/blob/main/main.py
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results


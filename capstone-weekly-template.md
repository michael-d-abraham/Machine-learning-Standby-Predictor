# Part B – Weekly Capstone Assignment Template

This template is used for **each of the 14 Part B capstone submissions** throughout the semester. The structure remains the same every week so that you can focus on *thinking and judgment*, not guessing expectations.

Your goal is not to achieve the best performance, but to **reason carefully about how this week’s machine learning technique applies (or does not apply) to your project**.
---

## 1. Project Context (Brief)

Provide the same short context each week so graders can orient quickly.

* **Project Title:** Content-Based Music Recommender (Spotify Audio Features + Lyrics/Topics)
* **Data Modality:** tabular
* **Task Type:** recommendation (content-based similarity search)
* **One-Sentence Goal:** Given a song, recommend other songs that sound similar using the song’s features.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Model family comparison: Naive Bayes (generative classifier) vs k-Nearest Neighbors (instance-based classifier) on the same binary classification proxy task
* **Key Assumptions of This Technique:**
  * **Naive Bayes assumptions:**
    * Feature independence: features don't correlate with each other
    * Gaussian distribution: each feature follows a normal distribution within each class
    * Probabilistic model: uses Bayes theorem to compute P(y|X)
  * **kNN assumptions:**
    * Local similarity: nearby points in feature space have similar labels
    * Distance-based: uses Euclidean distance (requires feature scaling)
    * Non-parametric: decision boundary adapts to local data density
  * **Shared assumptions:**
    * Same splits (60/20/20 artist-based, no leakage)
    * Same preprocessing (median imputation, StandardScaler)
    * Same evaluation metrics (PR-AUC primary, F1 and accuracy secondary)

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for understanding my project because:

Model family comparison directly addresses the core question: which approach is better for this proxy task and why? Naive Bayes tests whether features are independent and Gaussian. kNN tests whether similarity-based methods work in this feature space. Both are relevant because the recommendation system will use distance-based similarity. Comparing them reveals tradeoffs (NB: fast but rigid assumptions; kNN: flexible but curse of dimensionality) that inform how to build the recommender. The proxy task (high vs low energy) validates feature quality without requiring true recommendation labels. PR-AUC remains the primary metric for model selection.

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week's technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:** Proxy task - predicting high energy vs low energy (binary classification). Created binary target by median-splitting continuous energy values using the training-set median. Each song is a 23-dimensional numeric feature vector after preprocessing (audio and topic-proportion features only; no lyrics, identifiers, dates, or categorical metadata). Preprocessing: median imputation (if needed) and StandardScaler for both model families.

* **Why this representation was reasonable for this week:** Energy is a core audio characteristic. Using numeric features only keeps the setup clean for comparing model families: we test whether Naive Bayes assumptions (independence, Gaussian) hold vs whether kNN (distance-based, local similarity) works better. Median split ensures ~50/50 class balance so PR-AUC is meaningful. StandardScaler is mandatory for kNN (distance-based) and included for Naive Bayes for fair comparison. If one model family clearly outperforms, that informs whether independence assumptions or local similarity better matches the data structure.

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
I implemented a rigorous comparison of TWO model families: Naive Bayes (generative) vs k-Nearest Neighbors (instance-based). Added `build_nb_model()` and `build_knn_model(k)` functions with leakage-safe pipelines (median imputation, StandardScaler, classifier). Created `run_assignment7_model_family_comparison()` that trains Naive Bayes (GaussianNB) once and kNN with k in {3, 11, 51} to test neighborhood size effects. All models evaluated on the same validation set with the same metrics (PR-AUC primary, F1/accuracy secondary). Generated comparison table saved as `assignment7_comparison.csv`. Selected winner (kNN k=51, validation PR-AUC 0.9004) and evaluated once on test set (PR-AUC 0.8965). Performed error analysis: extracted false positives and false negatives from validation, explained why each model fails (NB: independence assumption violated; kNN: curse of dimensionality). Integrated into main workflow after Assignment 6.

* What you intentionally did *not* attempt and why
I did not exhaustively tune hyperparameters (only 3 k values for kNN, default GaussianNB). I did not try other NB variants (MultinomialNB, BernoulliNB) because features are continuous not counts or binary. I did not use ensemble methods or neural networks; the goal was to compare two canonical model families. I did not change the split or target; kept 60/20/20 artist-based and median-split energy for consistency with prior work. I did not maximize performance; focus was on understanding tradeoffs and reasoning about fit.

* Any constraints encountered (data, labels, compute, time)
No natural prediction target (recommendation system), so binary energy proxy. Dataset is large (~28k songs); kNN prediction is slow (searches all 16,615 training points), took ~20 seconds for validation. Artist-group split gave 60/20/20 with no artist overlap. Classes were ~50/50. No missing values; imputation was precautionary. All tuning and selection used PR-AUC only. Test set used only once for final evaluation.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

**Validation Results (Model Comparison):**

| Model | Hyperparameters | PR-AUC | F1 | Accuracy |
|-------|----------------|--------|-----|----------|
| Naive Bayes | GaussianNB (default) | 0.8446 | 0.7984 | 0.8032 |
| kNN | k=3 | 0.7933 | 0.7967 | 0.7972 |
| kNN | k=11 | 0.8740 | 0.8247 | 0.8223 |
| **kNN** | **k=51** | **0.9004** | **0.8289** | **0.8244** |

**Winner: kNN with k=51**
- Validation PR-AUC: 0.9004
- Test PR-AUC: 0.8965 (stable, -0.0039 drop)
- Outperforms Naive Bayes by +0.0558 PR-AUC

**Key Observations:**
1. Larger k is better: k=3 (0.7933) < k=11 (0.8740) < k=51 (0.9004) — larger neighborhoods smooth out noise
2. kNN beats Naive Bayes: suggests feature independence assumption is violated; audio features likely correlate (energy, loudness, danceability)
3. Test performance validates winner: small validation-test gap indicates good generalization, no overfitting

**Error Analysis (validation set):**
- Total errors: 986 / 5616 (17.6%)
- False positives (predicted high, actually low): 644 songs — kNN confused by mixed-class neighborhoods in high-D space
- False negatives (predicted low, actually high): 342 songs — boundary songs with neighbors from wrong class; curse of dimensionality makes "nearest" neighbors not truly similar

**Class balance:** ~50/50 from median split  
**Split:** 60/20/20 by artist (seed=42); no artist overlap; ~16.6k train, ~5.6k val, ~6.1k test

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

**Why kNN outperformed Naive Bayes:**

kNN (k=51) achieved PR-AUC 0.9004 vs Naive Bayes 0.8446 because the Naive Bayes independence assumption is violated. Audio features correlate: high-energy songs tend to be loud, danceable, and have high valence. Naive Bayes assumes these are independent, which understates the joint probability. kNN doesn't make this assumption—it just measures similarity in the full 23-dimensional space and aggregates labels from neighbors. The success of larger k (51 > 11 > 3) shows that smoothing over larger neighborhoods reduces noise and gives more robust predictions. Small k is too sensitive to outliers.

**Which assumptions held or failed:**

- ✗ **Feature independence (NB):** Failed. Audio features clearly correlate.
- ✗ **Gaussian distributions (NB):** Partially failed. Some features may not be normally distributed within classes.
- ✓ **Local similarity (kNN):** Held reasonably well, especially with large k. Songs with similar feature vectors do have similar energy.
- ✓ **Distance-based methods work (kNN):** Held. Euclidean distance (with scaling) is effective.
- ⚠ **Curse of dimensionality (kNN):** Present but manageable. 23 features is high-dimensional, causing some confusion (17.6% error rate), but not catastrophic.

**What this reveals:**

The data has structure that kNN captures but Naive Bayes misses. Features aren't independent—they form correlated patterns that define "high energy" songs. This suggests the representation has redundancy (some features may be derived from others). The finding that kNN works validates that distance-based similarity will likely work for the recommendation system. However, 23 dimensions may be too many; dimensionality reduction (PCA or feature selection) could help. The error analysis shows that kNN still makes mistakes in high-D space (false positives/negatives), confirming that curse of dimensionality is real but not prohibitive.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

I'll keep the numeric-only preprocessing pipeline (median imputation, StandardScaler) and the 60/20/20 artist split; the model comparison showed these support good performance. I'll keep using PR-AUC as the primary metric where applicable. I'll favor distance-based methods (like kNN) over Naive Bayes since feature independence doesn't hold.

Next step: apply kNN-like logic to similarity-based recommendation. Use the same 23 numeric features and StandardScaler. For a query song, find k most similar songs in the feature space and recommend them. The finding that k=51 works best suggests using larger neighborhoods for robustness.

If unconstrained, I would:
1. **Dimensionality reduction:** Apply PCA to reduce 23 features to ~10 components, addressing curse of dimensionality while preserving variance.
2. **Feature selection:** Identify most informative features (e.g., energy, loudness, valence) and discard redundant ones.
3. **Indexing for speed:** Use KD-tree or ball tree for fast nearest-neighbor search; current kNN is too slow (searches all 16k training points).
4. **Test other distance metrics:** Cosine similarity or Manhattan distance may work better than Euclidean in high-D.
5. **Inspect feature correlations:** Correlation matrix to confirm which features are redundant and can be removed.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week's technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not a mismatch. Model family comparison applied cleanly to the proxy task. The finding that kNN outperforms Naive Bayes is a valid outcome that reveals feature correlations and validates distance-based methods. The exercise answered the intended questions—which model family is better suited for this data, and why—and directly informs recommendation system design. kNN's success means similarity-based recommendation using Euclidean distance (or similar metrics) should work well. Naive Bayes's failure confirms that assuming feature independence is incorrect, guiding us away from methods that rely on that assumption.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor/blob/main/main.py
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results


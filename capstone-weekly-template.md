# Part B – Weekly Capstone Assignment Template

This template is used for **each of the 14 Part B capstone submissions** throughout the semester. The structure remains the same every week so that you can focus on *thinking and judgment*, not guessing expectations.

Your goal is not to achieve the best performance, but to **reason carefully about how this week’s machine learning technique applies (or does not apply) to your project**.

GIT REPO: https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor/blob/main/main.py
---

## 1. Project Context (Brief)

Provide the same short context each week so graders can orient quickly.

* **Project Title:** Content-Based Music Recommender (Spotify Audio Features + Lyrics/Topics)
* **Data Modality:** tabular
* **Task Type:** recommendation (content-based similarity search)
* **One-Sentence Goal:** Given a song, recommend other songs that sound similar using the song’s features.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Binary classification with logistic regression
* **Key Assumptions of This Technique:**
  * Features can distinguish between high and low energy songs (binary decision boundary exists)
  * Logistic regression can model the probability of high energy given audio features
  * Class balance is reasonable (not severely imbalanced) for meaningful metrics
  * Preprocessing pipeline prevents data leakage from validation/test sets

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

Recommendation systems don't naturally have a prediction target. I'm using a proxy task (predicting high vs low energy from audio features) to validate that my features contain useful information for categorical decisions. This is partial fit because while classification isn't my end goal, it proves the data is ready and helps me understand how features relate to song characteristics. Classification metrics (precision, recall, F1) provide richer evaluation than regression alone and help validate the feature pipeline for recommendation.

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week's technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:** Proxy task - predicting high energy vs low energy (binary classification). Created binary target by median-splitting continuous energy values. Each song is a 36-dimensional feature vector after preprocessing (23 standardized numeric features + 13 one-hot encoded categorical features from genre and topic).

* **Why this representation was reasonable for this week:** Energy is a core audio characteristic that relates to other features like loudness and acousticness. Binary classification exercises the full classification pipeline and validates that features can make categorical decisions. If the model can distinguish high from low energy songs, it validates that features contain useful signal for finding similar songs. The standardized numeric representation works well with logistic regression, and classification metrics provide richer evaluation than regression alone.

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
I built a binary classification baseline that predicts high energy vs low energy songs from 36 audio features. Used artist-based group splitting (70/15/15), scikit-learn pipeline with ColumnTransformer for mixed numeric/categorical preprocessing, and LogisticRegression classifier. Evaluated with full classification metrics (accuracy, precision, recall, F1, ROC AUC, PR AUC), confusion matrices, ROC and Precision-Recall curves, and threshold analysis comparing 0.5 vs 0.4 thresholds.

* What you intentionally did *not* attempt and why
I did not try nonlinear models or polynomial features - this week is about establishing a simple baseline with proper evaluation. I did not use lyrics because they need separate text processing. I did not extensively tune hyperparameters because the goal is a baseline workflow, not optimal performance. I did not try multi-class classification (genre/topic) because binary classification is the focus this week.

* Any constraints encountered (data, labels, compute, time)
No natural prediction target since this is a recommendation system, so I created a binary target by median-splitting energy. Dataset is large (28k songs) but training was fast with scikit-learn. Classes are balanced (50/50 split), which is ideal for classification metrics. No missing values in the data, so imputation was precautionary.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

- **Test Accuracy: 85.6%** - strong performance for binary classification
- **Test F1-Score: 0.874** - excellent balance between precision and recall
- **Test ROC AUC: 0.940** - very good discriminative ability
- **Test PR AUC: 0.947** - strong performance on precision-recall curve
- **Test Precision: 0.833, Recall: 0.920** (with threshold 0.4) - good recall, acceptable precision
- **Validation-Test gap: <0.5%** - excellent generalization, no overfitting
- **Class balance: 50/50** - ideal for classification metrics
- **Threshold analysis:** Lowering threshold from 0.5 to 0.4 improved F1 (0.853 → 0.857) by increasing recall (+3.9%) with small precision cost (-2.7%)

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

The strong performance (ROC AUC=0.94, F1=0.87) proves that audio features contain rich information for categorical decisions. The logistic regression assumption held well - features can distinguish high from low energy songs with a clear decision boundary. The balanced classes (50/50) made all metrics meaningful, and the high ROC AUC shows the model has strong discriminative ability.

The tiny validation-test gap (<0.5%) shows the artist-based split worked as intended. The model learned generalizable patterns, not artist-specific quirks. This is critical validation that leakage prevention didn't hurt performance. The threshold analysis revealed that slightly lowering the threshold (0.5 → 0.4) improved F1 by prioritizing recall, which makes sense if we want to catch more high-energy songs.

What this reveals: The feature engineering is solid enough for recommendation. If we can classify energy this accurately, we can definitely use these features to find similar songs. Classification metrics provide richer evaluation than regression - precision/recall tell us about the model's decision-making quality, not just prediction accuracy. The strong baseline suggests the features are ready for similarity-based recommendation.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

I'll keep the preprocessing pipeline and feature set since they work well. The strong classification baseline means I can move forward with building the actual recommender using cosine similarity on these same features. The classification metrics (especially precision/recall) help me understand how well features distinguish song characteristics.

Next step: use the learned logistic regression coefficients to understand which features matter most for energy classification, then build similarity-based retrieval. The threshold analysis shows that different decision boundaries can optimize different metrics - this insight applies to recommendation ranking. If unconstrained, I'd add lyrics with TF-IDF, try polynomial features to capture interactions, experiment with multi-class classification (genre/topic), and explore how classification confidence relates to recommendation quality.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week's technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not a mismatch - the technique worked well as a proxy task. The only limitation is that classification isn't my actual goal (recommendation is). But classifying energy validated the features and pipeline, which is exactly what a baseline should do. The strong results (ROC AUC=0.94, F1=0.87) give confidence to move forward with similarity-based recommendation using these same features. Classification metrics provide richer evaluation than regression - they tell us about decision quality, not just prediction accuracy, which is valuable for understanding how features relate to song characteristics.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: **optional unless explicitly requested**
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results


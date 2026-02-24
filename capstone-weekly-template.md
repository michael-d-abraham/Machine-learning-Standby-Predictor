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

* **Technique / Model Family Covered This Week:** Binary classification with logistic regression and L2 regularization; hyperparameter control via C (inverse regularization strength)
* **Key Assumptions of This Technique:**
  * Features can distinguish between high and low energy songs (binary decision boundary exists)
  * Logistic regression with L2 can model the probability of high energy given audio features
  * Controlling C lets us diagnose overfitting vs underfitting and tune complexity
  * Class balance is reasonable (median split gives ~50/50) for meaningful PR-AUC
  * Preprocessing fit on training only prevents data leakage; artist-group split prevents leakage across splits

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

Recommendation systems don't naturally have a prediction target. I use a proxy task (predicting high vs low energy from audio features) to validate that my features contain useful information and to test whether regularization and complexity control matter in this data regime. This is partial fit because classification isn't my end goal, but diagnosing overfitting and regularization impact tells me whether the representation is stable and ready for similarity-based recommendation. PR-AUC is the primary metric for model selection and validation.

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week's technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:** Proxy task - predicting high energy vs low energy (binary classification). Created binary target by median-splitting continuous energy values using the training-set median. Each song is a 23-dimensional numeric feature vector after preprocessing (audio and topic-proportion features only; no lyrics, identifiers, dates, or categorical metadata). Preprocessing: median imputation (if needed) and StandardScaler; no one-hot encoding this week.

* **Why this representation was reasonable for this week:** Energy is a core audio characteristic. Using numeric features only keeps the setup clean for studying regularization: we control complexity via C only. Median split ensures ~50/50 class balance so PR-AUC is meaningful. If regularization meaningfully changes validation behavior, that informs whether the representation overfits or is stable for recommendation.

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
I extended the capstone with regularization and hyperparameter control. I added dataset inspection (structure, numeric features, missing values, energy distribution), artist-based 60/20/20 split (seed=42), and a numeric-only pipeline (SimpleImputer, StandardScaler, LogisticRegression with L2). I established a baseline with C=1.0 and diagnosed fit (train vs validation PR-AUC gap). I ran a validation curve over C in {0.001, 0.01, 0.1, 1, 10, 100} with 5-fold CV and PR-AUC, plotted the curve (saved as validation_curve.png), and ran a small grid search over C in {0.01, 0.1, 1, 10} with 5-fold CV. I then compared baseline (C=1.0) vs best C on the test set (used only once), training both on train+validation. I generated reflection diagnostics (regularization impact, overfitting, dataset size, implications for recommendation).

* What you intentionally did *not* attempt and why
I did not change the model family (only Logistic Regression with L2 as specified). I did not use categorical features (genre, topic) this week so the representation is numeric-only for clear complexity control. I did not touch the test set until the final comparison. I did not maximize performance; the goal was to answer whether regularization matters and whether the representation overfits.

* Any constraints encountered (data, labels, compute, time)
No natural prediction target (recommendation system), so binary energy proxy with median split. Dataset is large (~28k songs); artist-group split gave 60/20/20 with no artist overlap. Classes were ~50/50. No missing values; imputation was precautionary. All tuning and selection used PR-AUC only.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

- **Baseline (C=1.0) diagnosis:** ACCEPTABLE FIT — train PR-AUC 0.936, validation PR-AUC 0.939, gap −0.003 (small; good generalization)
- **Validation curve:** C in {0.001, 0.01, 0.1, 1, 10, 100}; best validation PR-AUC at C=10 (~0.933); overfitting observed at C=100 (validation drops while training rises)
- **Grid search (4 C × 5-fold CV):** Best C=10, mean CV PR-AUC 0.9327; baseline C=1.0 had same CV PR-AUC (0.9327)
- **Final test (single use):** Baseline (C=1.0) test PR-AUC 0.931; tuned (C=10) test PR-AUC 0.931; change +0.000 — no meaningful difference
- **Regularization impact:** Minimal — tuning C did not meaningfully change test PR-AUC; baseline was already well-regularized
- **Class balance:** ~50/50 from median split
- **Split:** 60/20/20 by artist (seed=42); no artist overlap; ~16.6k train, ~5.6k val, ~6.1k test

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

The method behaved as it did because the dataset is large (~28k songs) and the numeric features (after scaling) provide a stable, linearly separable signal for energy. The baseline (C=1.0) already had a negligible train–validation gap, so increasing or decreasing regularization (changing C) did not meaningfully change validation or test PR-AUC. The validation curve showed that at very high C (e.g. 100) some overfitting appears (validation score drops while training score rises), but in the range that mattered for selection (C ≤ 10), performance was flat. So the assumptions that (1) a linear boundary is adequate and (2) default regularization is reasonable for this data regime both held.

What this reveals: Regularization and complexity control did not change outcomes in a meaningful way — not because the technique was wrong, but because the representation is stable and the dataset size is sufficient. That is academically valid: the answer to "does regularization matter here?" is "no, not in this regime." For the recommender, this implies the numeric feature representation is reliable and ready for similarity-based recommendation; we did not need to regularize more aggressively or simplify the model.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

I'll keep the numeric-only preprocessing pipeline and the 60/20/20 artist split; the regularization analysis showed they support good generalization. I'll keep using PR-AUC as the primary metric where applicable. I will not change the model family or add heavy regularization by default, since the evidence showed it wasn't needed in this data regime.

Next step: proceed with similarity-based recommendation using the same numeric features; the stability observed across C supports that. If unconstrained, I would add lyrics (e.g. TF-IDF), try feature selection or dimensionality reduction to see if a smaller feature set behaves similarly, and optionally inspect logistic regression coefficients to see which features drive energy most — that could inform which dimensions to weight in similarity.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week's technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not a mismatch. Regularization and hyperparameter control applied cleanly to the proxy task. The finding that regularization did not meaningfully change performance is a valid outcome: it indicates a stable representation and sufficient data, not a poor fit for the technique. The exercise answered the intended questions — whether the representation overfits and whether complexity control matters — and supports moving forward with similarity-based recommendation using these numeric features.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor/blob/main/main.py
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results


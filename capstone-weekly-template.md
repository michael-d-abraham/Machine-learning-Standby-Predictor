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

* **Technique / Model Family Covered This Week:** Linear regression with gradient descent optimization
* **Key Assumptions of This Technique:**
  * Linear relationships between features and target (energy can be modeled as a weighted sum of features)
  * Gradient descent will converge with a fixed learning rate on standardized features
  * MSE loss provides meaningful signal for learning feature relationships

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

Recommendation systems don't naturally have a prediction target. I'm using a proxy task (predicting energy from audio features) to validate that my features contain useful information. This is partial fit because while regression isn't my end goal, it proves the data is ready and helps me understand feature relationships before building the actual similarity-based recommender.

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week's technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:** Proxy task - predicting energy from other audio features. Each song is a 38-dimensional feature vector (23 scaled numeric features + 15 one-hot encoded categorical features).

* **Why this representation was reasonable for this week:** Energy is a core audio characteristic that relates to other features like loudness and acousticness. If the model can learn these relationships, it validates that features contain useful signal for finding similar songs. The standardized numeric representation works well with gradient descent.

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
I built a linear regression baseline that predicts song energy from 38 audio features. Used artist-based group splitting (70/15/15), train-only preprocessing (imputation, standardization, one-hot encoding), and manual gradient descent optimization. Trained for 1000 epochs with learning rate 0.01.

* What you intentionally did *not* attempt and why
I did not try nonlinear models or polynomial features - this week is about establishing a simple baseline. I did not use lyrics because they need separate text processing. I did not tune hyperparameters extensively because the goal is a baseline, not optimal performance.

* Any constraints encountered (data, labels, compute, time)
No natural prediction target since this is a recommendation system, so I used energy as a proxy. Dataset is large (28k songs) but training was fast with batch gradient descent. No missing values in the data, so imputation was precautionary.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

- **Test R² = 0.774** (explains 77% of variance in energy) - surprisingly strong for a linear model
- **Loss reduction: 96.8%** (from 0.399 to 0.013 MSE) - gradient descent worked very well
- **Train-val gap: 2.1%** - almost no overfitting, excellent generalization
- **Convergence:** Loss stabilized around epoch 300, stayed flat through 1000 epochs
- **RMSE: 0.115** on a 0-1 energy scale - predictions are quite accurate
- Training was smooth with no oscillations or instability at learning rate 0.01

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

The strong performance (R²=0.77) proves that audio features contain rich information about song characteristics. The linear assumption held surprisingly well - energy has clear linear relationships with features like loudness and acousticness. Gradient descent converged smoothly because standardization put all features on the same scale, validating the preprocessing choices.

The tiny train-val gap shows the artist-based split worked as intended. The model learned generalizable patterns, not artist-specific quirks. This is critical validation that leakage prevention didn't hurt performance.

What this reveals: The feature engineering is solid enough for recommendation. If we can predict energy this accurately, we can definitely use these features to find similar songs. The baseline is strong enough that jumping to complex models might not be necessary - linear similarity search could work well.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

I'll keep the preprocessing pipeline and feature set since they work well. The strong baseline means I can move forward with building the actual recommender using cosine similarity on these same features. 

Next step: use the learned weights to understand which features matter most for energy, then build similarity-based retrieval. If unconstrained, I'd add lyrics with TF-IDF, try polynomial features to capture interactions (like loudness × acousticness), and experiment with different proxy tasks to validate other feature relationships.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week's technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not a mismatch - the technique worked well as a proxy task. The only limitation is that regression isn't my actual goal (recommendation is). But predicting energy validated the features and pipeline, which is exactly what a baseline should do. The strong results (R²=0.77) give confidence to move forward with similarity-based recommendation using these same features.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: **optional unless explicitly requested**
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results


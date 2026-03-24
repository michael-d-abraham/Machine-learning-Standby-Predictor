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

* **Technique / Model Family Covered This Week:** Unsupervised learning with PCA and k-means clustering
* **Key Assumptions of This Technique:**
  * PCA captures major variance directions but does not prove natural classes
  * k-means assumes centroid-based groups and Euclidean-distance structure
  * Standardized features are important so one feature does not dominate distance
  * Musical structure may overlap and be continuous, so weak clustering can still be informative

**Fit Assessment (required):**

> I expected this technique to be a **partial** fit for my project because:

My capstone goal is recommendation, not hard class prediction. Unsupervised methods are still useful for checking whether the current feature space has visible structure, gradients, or diffuse overlap. Even if cluster separation is weak, that outcome helps evaluate whether recommendation should emphasize similarity/ranking rather than strict segmentation.

---

## 3. Representation or Proxy Used

Describe how your data was represented so that this week's technique could be applied.

Examples include:

* Hand-engineered features

* Summary statistics

* Frozen embeddings

* Dimensionality reduction

* A proxy task

* **Representation or Proxy Chosen:** Unsupervised feature matrix built from 23 numeric song-level features only. Excluded columns: `Unnamed: 0`, `artist_name`, `track_name`, `lyrics`, `genre`, `topic`, `release_date`, and `energy` (target/proxy source). Preprocessing: SimpleImputer (median) then StandardScaler.

* **Why this representation was reasonable for this week:** PCA and k-means both rely on numeric feature geometry, so scaling is required. Excluding text/meta/identifier columns and `energy` keeps the unsupervised matrix leakage-safe and aligned with project conventions. The existing high/low-energy proxy label was created separately from train median and used only post hoc for interpretation.

---

## 4. What Was Attempted

Be concrete and scoped. Do not list everything you *could* have done.

* What you implemented this week
Built a scoped unsupervised workflow in `main_unsupervised_a10.py`: load data, define included/excluded columns, preprocess with median imputation + scaling, run PCA, and run k-means for multiple k values (`k=2..7`) with inertia and silhouette reporting. Produced plots (explained variance, PCA scatter by cluster, PCA scatter by proxy label, elbow, silhouette). For selected k, generated cluster size and feature-profile summaries using standardized mean differences.

* What you intentionally did *not* attempt and why
Did not use deep learning or advanced clustering variants (GMM/DBSCAN/spectral) to keep scope student-readable and aligned with assignment instructions. Did not use the proxy label in fitting PCA or k-means. Did not treat clustering as a replacement for supervised proxy evaluation.

* Any constraints encountered (data, labels, compute, time)
Recommendation has no single ground-truth label, so unsupervised evaluation is inherently indirect. k-means quality metrics were modest, which limits strong segmentation claims. Matplotlib cache/font warnings appeared in this environment but did not block output generation.

---

## 5. Results or Observations

You may include metrics **if applicable**, but qualitative observations are also valid.

Examples:

* Evaluation metrics
* Training behavior or convergence issues
* Error patterns
* Unexpected behaviors

**PCA variance structure (first 10 components):**
- PC1: 0.1203
- PC2: 0.0756
- Cumulative variance by PC10: 0.5938

**k-means sweep (`k=2..7`):**
- k=2: silhouette 0.0867
- k=3: silhouette 0.0729
- k=4: silhouette 0.0771
- k=5: silhouette 0.0877
- k=6: silhouette 0.1044
- k=7: silhouette 0.1157 (best among tested)

**Selected k:** 7 (based on highest tested silhouette, while noting absolute values are low)

**Post hoc cluster vs proxy-energy agreement (interpretive only):**
- NMI: 0.0301
- ARI: 0.0216

**Qualitative cluster-profile examples (standardized means):**
- One cluster was high in `romantic` (+3.210), with higher `acousticness` and older `age`
- One cluster was high in `music` (+2.776)
- One cluster was high in `night/time` (+3.000)
- One cluster was high in `obscene` (+1.936) and `danceability`

---

## 6. Interpretation and Judgment

This section matters most.

Reflect on:

* Why the method behaved as it did
* Which assumptions held or failed
* What this reveals about your data or problem framing

**Why the method behaved as it did:**

PCA spread variance across many components instead of concentrating it strongly in the first two. k-means produced only low silhouette values across all tested k, with the best at k=7 still weak in absolute terms. This suggests the feature space has overlapping or gradient-like structure rather than cleanly separable groups.

**Interpretability:**

- PCA plots gave an interpretable low-dimensional view of overlap and diffuse structure
- Cluster profiles still provided useful broad musical tendencies (e.g., romantic/acoustic, night/time-heavy, danceable/obscene profiles)
- Post hoc label agreement stayed very low (NMI/ARI near zero), indicating clusters are not just re-encoding the high/low-energy proxy

**What this reveals:**

Unsupervised learning is a **partial fit**: useful for exploration and diagnostics, but weak as a standalone segmentation strategy. The results support framing recommendation more as similarity/ranking in feature space than hard cluster assignment. Weak clustering is still informative because it exposes limitations in the assumption that songs naturally split into clear groups.

---

## 7. Forward-Looking Adjustment

Answer **one** of the following:

* What will you keep, change, or discard before the next assignment?
* What would you try next if data or resources were not constrained?

I'll keep the same preprocessing (median imputation + StandardScaler) and leakage-safe split philosophy. I will keep using PCA/clustering as exploratory diagnostics rather than primary recommender logic.

**Limitations:** Silhouette values were low across tested k, and post hoc proxy agreement was weak. This limits any claim of natural, discrete song groups.

**Next steps:** Focus recommendation framing on similarity/ranking; use unsupervised outputs for feature diagnostics, optional segment discovery, and error analysis. If needed, test whether refined feature sets improve clustering clarity without overclaiming.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

If this week's technique was a poor fit, explain:

* Why it does not align with your project
* Evidence supporting that conclusion
* What value this attempt still provided

Not a full mismatch. Unsupervised learning was a partial fit: it did not produce strongly separated clusters, but it provided useful evidence that the feature space may be continuous/overlapping. That insight is valuable for task framing and supports a similarity-based recommendation perspective over hard segmentation.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor/blob/main/main_unsupervised_a10.py
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results


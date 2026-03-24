# Assignment 10 Part B: Unsupervised Learning Findings

## 1) Data Preparation

- Included numeric feature count: **23**
- Included features: len, dating, violence, world/life, night/time, shake the audience, family/gospel, romantic, communication, obscene, music, movement/places, light/visual perceptions, family/spiritual, like/girls, sadness, feelings, danceability, loudness, acousticness, instrumentalness, valence, age
- Excluded columns: Unnamed: 0, artist_name, track_name, lyrics, genre, topic, release_date, energy
- Exclusion rationale: text/meta/identifier columns and target (`energy`) were excluded from the unsupervised feature matrix.
- Preprocessing: `SimpleImputer(strategy="median")` then `StandardScaler()`.

## 2) PCA Observations

First principal components (explained + cumulative variance):

component  explained_variance_ratio  cumulative_explained_variance
      PC1                  0.120290                       0.120290
      PC2                  0.075625                       0.195915
      PC3                  0.061670                       0.257585
      PC4                  0.051548                       0.309133
      PC5                  0.050787                       0.359920
      PC6                  0.049553                       0.409473
      PC7                  0.048346                       0.457819
      PC8                  0.045974                       0.503793
      PC9                  0.045669                       0.549462
     PC10                  0.044348                       0.593810

- PCA is a projection and does not prove natural, true clusters.
- Overlap in 2D PCA space should be interpreted as evidence of possible gradients/continuous structure, not failure by itself.

## 3) K-Means Across k

 k     inertia  silhouette
 2 600747.9951      0.0867
 3 569975.8127      0.0729
 4 547435.6793      0.0771
 5 527040.5018      0.0877
 6 506282.3403      0.1044
 7 485124.7814      0.1157

- Selected k: **7** (silhouette-based evidence, with simpler k preferred when near-tied).
- Best silhouette observed: **0.1157**
- If silhouette values are low overall, that indicates weak separation and is still a valid, informative result.

## 4) Cluster Interpretation

Cluster sizes:

         count  percent
cluster                
0         2544   0.0897
1         5643   0.1989
2         4906   0.1729
3         2065   0.0728
4         5762   0.2031
5         1882   0.0663
6         5570   0.1963

Top distinguishing standardized feature means by cluster (signed z-difference vs global mean):

- Cluster 0: music: +2.776; acousticness: +0.497; violence: -0.445; age: +0.445; loudness: -0.424
- Cluster 1: violence: +1.739; sadness: -0.417; obscene: -0.387; world/life: -0.380; acousticness: -0.362
- Cluster 2: obscene: +1.936; len: +1.041; danceability: +0.649; sadness: -0.568; age: -0.472
- Cluster 3: night/time: +3.000; violence: -0.435; world/life: -0.422; sadness: -0.403; obscene: -0.360
- Cluster 4: sadness: +1.680; violence: -0.460; obscene: -0.425; world/life: -0.382; len: -0.327
- Cluster 5: romantic: +3.210; acousticness: +0.741; age: +0.669; len: -0.523; loudness: -0.514
- Cluster 6: world/life: +1.616; violence: -0.436; obscene: -0.415; sadness: -0.385; feelings: +0.353

Interpretation note: these are broad profiles, not guaranteed discrete musical categories.

## 5) Post Hoc Comparison to Proxy Energy Label

- Proxy label was used **after** clustering only, for interpretation.
- Normalized Mutual Information (NMI): **0.0301**
- Adjusted Rand Index (ARI): **0.0216**
- Low agreement can mean energy is only one axis of structure, or the space is more continuous than cluster-like.

## 6) Fit Judgment

Unsupervised learning is a **partial fit** for this project: useful for exploratory structure checks and feature diagnostics, but not strong evidence by itself for hard song segmentation. For recommendation, similarity/ranking may be more natural than strict clustering.

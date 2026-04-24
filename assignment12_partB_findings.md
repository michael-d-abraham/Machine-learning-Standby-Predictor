# Assignment 12 Part B — Weekly findings (rubric 1–6)

**Project:** Content-Based Music Recommender (Spotify audio features + lyrics/topics). **Modality:** tabular. **End goal:** content-based similarity / recommendation. This week uses a **supervised proxy** to exercise a practical training pipeline (see below).

**Repo:** [Machine-learning-Standby-Predictor](https://github.com/michael-d-abraham/Machine-learning-Standby-Predictor)

---

## 1. Weekly technique

**Technique / concept:** *Practical deep learning systems* — not architectural novelty, but an **end-to-end training pipeline**: fixed seeds, explicit device, **batched** optimization (DataLoader), **train/val monitoring** (loss and ROC-AUC), wall-clock, and a **JSON run artifact** for reproducibility and inspection.

**What it does:** Wraps a small **PyTorch** feedforward classifier (`TabularMLP`: 23→64→32→1 logits) with **Adam** and **BCEWithLogitsLoss** so you can observe convergence, overfitting, and run metadata the same way you would on a larger project.

**Assumptions it makes:**

- A **stable software environment** (framework version, CPU vs GPU) is part of the result.
- **Train/validation discipline** is non-optional: the same **leakage-safe** splits and **fit-on-train-only** preprocessing as the rest of the capstone must carry into the training loop.
- **Monitoring** (per-epoch losses, simple metrics) is the primary way to catch bugs, divergence, and overfitting before adding compute or complexity.
- For **tabular, medium-scale** data on one machine, **distributed** training and **serving** infrastructure are usually unnecessary; **seeds, logging, and wall-clock** still matter.

**Code:** [`assignment12_dl_systems.py`](assignment12_dl_systems.py) · **Artifacts:** [`assignment12_dl_run.json`](assignment12_dl_run.json) (includes full per-epoch `history`), [`assignment12_train_val_curves.png`](assignment12_train_val_curves.png) (after a run).

---

## 2. Fit to this project

**Verdict: partial fit** — the *workflow* applies whenever you train a neural model; the *infrastructure* story is on-point for the capstone. A **huge** production DL stack (distributed training, serving fleets) is **not** justified by this dataset; a **small MLP on fixed numeric features** is a **vehicle** for systems habits, not a claim that deep nets “solve” recommendation here.

| Dimension | This project | Why it matters for fit |
|-----------|----------------|------------------------|
| **Data size** | ~28k rows; train/val/test **16,615 / 5,616 / 6,141** (artist-group split) | Enough rows for a shallow net to train stably; not “big data” in the distributed sense. |
| **Feature types** | **23 numeric** song-level features (topic proportions, audio descriptors, `age`); text/date/categorical columns excluded for this pipeline | Standard tabular tensor after sklearn preprocessing — matches “feature engineering + neural head” in many products, but **not** end-to-end audio/text learning. |
| **Label structure** | **Binary** high/low `energy` from **training-set median only** of continuous `energy` (no test leakage) | Clear classification target for monitoring; **not** the same as recommender **ranking** or **implicit feedback**. |
| **Problem type** | **Binary classification** (proxy) vs **recommendation** (capstone goal) | The technique fits **as a training exercise**; the **end task** (similarity / top‑k) still needs different metrics and data if you optimize for that. |

**Summary:** The technique aligns with **reproducible, monitored training** on real capstone splits; it does **not** fully align with **large-scale representation learning** or **production recommender** objectives without more scope (labels, ranking, scale).

---

## 3. Scoped implementation

**Representation / proxy (unchanged from prior assignments):** Same exclusions as [`main.py`](main.py) (`Unnamed: 0`, `artist_name`, `track_name`, `lyrics`, `genre`, `topic`, `release_date`, and the raw `energy` column as a feature). **Target:** `energy` binarized at the **train median** via [`main.split_data`](main.py). **Preprocessing:** `SimpleImputer(median)` + `StandardScaler` **fit on train only**, then transform val/test.

**What was implemented:**

- Load `tcc_ceds_music.csv` via `main.inspect_dataset` and `main.split_data` (artist **60/20/20**, no artist overlap).
- PyTorch `TabularMLP`, batch size **256**, **40** epochs, lr **0.001**, seed **42**.
- Log **train loss**, **val loss**, **val ROC-AUC** each epoch; write **`assignment12_dl_run.json`** with **full** epoch `history` plus meta (torch version, device, row counts, test ROC-AUC, wall time).
- Save **`assignment12_train_val_curves.png`**: train vs val loss and val ROC-AUC over epochs (stability / mild overfitting visible qualitatively).

**Intentionally not implemented:** distributed training, serving, HPO, cloud experiment trackers, or raw lyrics/audio end-to-end — kept to **single-machine** systems hygiene and comparability with prior sklearn work.

**Constraints:** Recommender **ground truth** is still absent; **energy** is a **proxy**. Training remains **fast on CPU** for this model; limits are **framing and labels**, not cluster throughput.

---

## 4. Outcomes

**Stability and behavior (see JSON + figure; exact numbers update each run):**

- **Test ROC-AUC** and **final val ROC-AUC** are recorded in `assignment12_dl_run.json` under `meta`. Example from a completed run: **test ROC-AUC ≈ 0.939**, **wall-clock ≈ 4.1 s** on CPU, PyTorch **2.11.x**, full **40**-epoch **`history`** array in the same file.
- **Train loss** tends to **decrease** over epochs; **val ROC-AUC** rises quickly then **plateaus**; **val loss** often **improves early** then **creeps up** while train loss keeps falling — a familiar **mild overfitting** pattern under fixed epochs and no early stopping.
- **Interpretability:** this setup prioritizes **run transparency** (curves, JSON) over sparsity of linear models; the MLP does not yield simple global feature weights like a linear SVM.

**Not optimizing for:** leaderboard accuracy; a single AUC on a proxy does not validate a full recommender.

---

## 5. Reflect deeply

**Data quality / representation:** The inputs are **dense, hand-crafted summaries** already correlated with `energy`, so the network **fits the proxy quickly**. That is informative: it says the **tabular representation is strong for this label**, not that the **recommendation problem** is solved.

**Problem framing:** The hard part of the capstone remains **evaluation** — **proxy classification** vs **ranking / similarity** for the actual product. The **artist-group split** reduces **artist leakage** but does not remove all **distribution shift** (e.g. genre mix differs across artists).

**What the technique exposed:**

- **Overfitting risk:** train–val loss divergence under more epochs / capacity is visible in the **curves**; mitigations (early stopping, weight decay) were **out of scope** for this minimal run.
- **Biases / limitations:** Class balance follows the **median cut**; **annotator/playlist biases** in the source data are not addressed here. **IID** assumptions across **rows** are **violated** in structure (songs cluster by **artist**); the group split is the right guardrail, but **residual** correlation within genres remains.

**Bottom line:** Practical DL **infrastructure** is **portable**; on this project the **marginal** lift of a small MLP over strong tabular baselines is often about **discipline and logging**, not a need for a **deeper** stack on the same 23 columns — unless the representation moves to **embeddings** or **raw** modalities.

---

## 6. Limitations and mismatches

| Where it fails to align or is weak | Detail |
|--------------------------------------|--------|
| **Recommender vs classifier** | The capstone goal is **similarity / ranking**; this week is **binary classification** on a **proxy** label. |
| **Assumptions violated** | **Exchangeability** of rows is **not** true (artist structure); we mitigate with **group splits** but not a full generative story. **Median threshold** embeds a **static** class definition that may not transfer across **time** or **subpopulations**. |
| **“Deep learning” at scale** | **Distributed** training, **serving**, and **large experiment platforms** are **mismatched** to current data size; the value is **process**, not that machinery. |
| **What would help this technique work better for the *product*** | **Ranking** labels (e.g. pairwise or implicit feedback), **item/user** features aligned with the task, and **metrics** (NDCG, recall@k) tied to **recommendation** — not only proxy AUC. |

**What this attempt still provided:** A **repeatable** training loop, **honest** splits, and **evidence** (JSON + curves) for **stability and overfitting** — the core “systems” lesson at laptop scale.

---

## Forward-looking (brief)

**Keep:** Artist-group split, median-only thresholding for this proxy, JSON run logs (and loss/AUC figures) for any future neural experiments.

**Next if scope expands:** Early stopping, weight decay, or **representation learning** from audio/text — only if the task moves beyond this fixed 23-column matrix; for recommendation, emphasize **similarity and ranking** over a single classifier score.

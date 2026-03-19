import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Import split/target logic from the existing project.
import main as base_main


def _build_preprocess() -> Pipeline:
    """Impute missing values then scale numeric features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_pipeline(model) -> Pipeline:
    """Common preprocessing + estimator wrapper."""
    return Pipeline(
        steps=[
            ("preprocess", _build_preprocess()),
            ("model", model),
        ]
    )


def _get_scores_from_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Use decision_function when available, else fall back to probability estimates."""
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        return np.asarray(scores)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return np.asarray(proba)[:, 1]
    raise ValueError("Estimator must support decision_function or predict_proba.")


def _compute_pr_auc_roc_auc(y_true: pd.Series, scores: np.ndarray) -> Tuple[float, float]:
    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    return pr_auc, roc_auc


def _fit_and_eval(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> Dict[str, float]:
    pipeline.fit(X_train, y_train)
    scores = _get_scores_from_pipeline(pipeline, X_eval)
    pr_auc, roc_auc = _compute_pr_auc_roc_auc(y_eval, scores)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def _fit_and_eval_both_train_val(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    pipeline.fit(X_train, y_train)
    train_scores = _get_scores_from_pipeline(pipeline, X_train)
    val_scores = _get_scores_from_pipeline(pipeline, X_val)
    train_pr_auc, train_roc_auc = _compute_pr_auc_roc_auc(y_train, train_scores)
    val_pr_auc, val_roc_auc = _compute_pr_auc_roc_auc(y_val, val_scores)
    return {
        "train_pr_auc": train_pr_auc,
        "val_pr_auc": val_pr_auc,
        "train_roc_auc": train_roc_auc,
        "val_roc_auc": val_roc_auc,
    }


def _get_best_by_pr_auc(rows: List[Dict[str, Any]], key: str = "val_pr_auc") -> Dict[str, Any]:
    if not rows:
        raise ValueError("No rows provided to select best model.")
    return max(rows, key=lambda r: r[key])


def _try_build_xgboost_model(params: Dict[str, Any]):
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            **params,
        )
    except Exception:
        return None


def run_ensemble_experiments(
    csv_path: str = None,
    seed: int = 42,
    comparison_scope: str = "all_models",
) -> Dict[str, Any]:
    """
    Run ensemble learning experiments (DT, RF, Boosting) and evaluate consistently.

    comparison_scope:
      - 'ensemble_only': final selection compares only DT/RF/Boosting.
      - 'all_models': final selection also includes LR/NB/kNN/SVM baselines.
    """
    if csv_path is None:
        csv_path = base_main.CSV_PATH

    # 1) Dataset inspection + split (leakage-safe; matches existing project logic).
    df, numeric_features = base_main.inspect_dataset(csv_path)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    ) = base_main.split_data(
        df,
        numeric_features,
        base_main.TARGET_COL,
        seed=seed,
        train_frac=0.60,
        val_frac=0.20,
    )

    results: Dict[str, Any] = {
        "meta": {
            "csv_path": csv_path,
            "seed": seed,
            "target_col": base_main.TARGET_COL,
            "feature_count": len(feature_names),
            "comparison_scope": comparison_scope,
        },
        "decision_tree": [],
        "random_forest": [],
        "boosting": [],
        "baselines": {},
        "diagnostics": {},
        "final_model": {},
    }

    # 2) Decision Tree baseline (capacity sweep).
    dt_depths = [3, 5, 10, None]
    for depth in dt_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        pipe = _build_pipeline(dt)
        row = _fit_and_eval_both_train_val(pipe, X_train, y_train, X_val, y_val)
        row["max_depth"] = depth
        results["decision_tree"].append(row)

    # 3) Random Forest (bagging).
    rf_n_estimators = [100, 200, 500]
    rf_max_features = ["sqrt", "log2", None]
    for n_estimators in rf_n_estimators:
        for max_features in rf_max_features:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=seed,
                n_jobs=-1,
                oob_score=True,
                bootstrap=True,
            )
            pipe = _build_pipeline(rf)
            pipe.fit(X_train, y_train)

            model_step: RandomForestClassifier = pipe.named_steps["model"]
            oob_score = float(model_step.oob_score_)

            val_scores = _get_scores_from_pipeline(pipe, X_val)
            val_pr_auc, val_roc_auc = _compute_pr_auc_roc_auc(y_val, val_scores)

            results["random_forest"].append(
                {
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "oob_score": oob_score,
                    "val_pr_auc": float(val_pr_auc),
                    "val_roc_auc": float(val_roc_auc),
                }
            )

    # 4) Boosting (XGBoost preferred; AdaBoost fallback).
    boosting_rows: List[Dict[str, Any]] = []
    boosting_algo = "adaboost"

    # Sweep as specified.
    boosting_learning_rates = [0.05, 0.1]
    boosting_n_estimators = [200, 300]
    xgb_probe_params_base = dict(
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
    )

    xgb_available = False
    for lr in boosting_learning_rates:
        for n_estimators in boosting_n_estimators:
            xgb_params = dict(
                **xgb_probe_params_base,
                n_estimators=n_estimators,
                learning_rate=lr,
            )
            xgb_model = _try_build_xgboost_model(xgb_params)
            if xgb_model is not None:
                xgb_available = True
                # Fit/eval for each config.
                pipe = _build_pipeline(xgb_model)
                fit_metrics = _fit_and_eval(pipe, X_train, y_train, X_val, y_val)
                boosting_rows.append(
                    {
                        "algo": "xgboost",
                        "learning_rate": lr,
                        "n_estimators": n_estimators,
                        "val_pr_auc": float(fit_metrics["pr_auc"]),
                        "val_roc_auc": float(fit_metrics["roc_auc"]),
                    }
                )
            else:
                # We'll fill AdaBoost later if XGBoost isn't available.
                continue

    if xgb_available:
        boosting_algo = "xgboost"
        results["boosting"] = boosting_rows
    else:
        # AdaBoost fallback: estimator=DecisionTreeClassifier(max_depth=1)
        for lr in boosting_learning_rates:
            for n_estimators in boosting_n_estimators:
                ada = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    random_state=seed,
                )
                pipe = _build_pipeline(ada)
                fit_metrics = _fit_and_eval(pipe, X_train, y_train, X_val, y_val)
                boosting_rows.append(
                    {
                        "algo": "adaboost",
                        "learning_rate": lr,
                        "n_estimators": n_estimators,
                        "val_pr_auc": float(fit_metrics["pr_auc"]),
                        "val_roc_auc": float(fit_metrics["roc_auc"]),
                    }
                )
        results["boosting"] = boosting_rows

    # 5) Baselines (included for final selection + summary table if comparison_scope='all_models').
    def _eval_baseline(model, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        pipe = _build_pipeline(model)
        metrics = _fit_and_eval(pipe, X_train, y_train, X_val, y_val)
        row = {
            "name": name,
            "params": params,
            "val_pr_auc": float(metrics["pr_auc"]),
            "val_roc_auc": float(metrics["roc_auc"]),
        }
        return row

    baselines: List[Dict[str, Any]] = []

    baselines.append(
        _eval_baseline(
            LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            "logistic_regression",
            {"C": 1.0},
        )
    )
    baselines.append(
        _eval_baseline(
            GaussianNB(),
            "naive_bayes",
            {},
        )
    )
    baselines.append(
        _eval_baseline(
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "knn",
            {"k": 5},
        )
    )
    baselines.append(
        _eval_baseline(
            SVC(random_state=seed),
            "svm",
            {"kernel": "rbf", "C": None, "gamma": None},
        )
    )

    results["baselines"]["rows"] = baselines

    # 6) Diagnostics (overfitting gap, stability, OOB vs validation).
    dt_gaps = []
    for row in results["decision_tree"]:
        gap = row["train_pr_auc"] - row["val_pr_auc"]
        dt_gaps.append({**row, "train_val_pr_gap": float(gap)})
    results["diagnostics"]["decision_tree_train_val_gap"] = dt_gaps

    dt_prs = [r["val_pr_auc"] for r in results["decision_tree"]]
    rf_prs = [r["val_pr_auc"] for r in results["random_forest"]]
    boost_prs = [r["val_pr_auc"] for r in results["boosting"]]
    results["diagnostics"]["stability"] = {
        "decision_tree_val_pr_auc_std": float(np.std(dt_prs)) if dt_prs else None,
        "random_forest_val_pr_auc_std": float(np.std(rf_prs)) if rf_prs else None,
        "boosting_val_pr_auc_std": float(np.std(boost_prs)) if boost_prs else None,
    }
    results["diagnostics"]["random_forest_oob_vs_val"] = [
        {
            "n_estimators": r["n_estimators"],
            "max_features": r["max_features"],
            "oob_score": r["oob_score"],
            "val_pr_auc": r["val_pr_auc"],
            "val_roc_auc": r["val_roc_auc"],
        }
        for r in results["random_forest"]
    ]

    # 7) Final selection + single test evaluation.
    best_dt = _get_best_by_pr_auc(results["decision_tree"])
    best_rf = _get_best_by_pr_auc(results["random_forest"])
    best_boost = _get_best_by_pr_auc(results["boosting"])

    candidates: List[Dict[str, Any]] = []
    candidates.append(
        {
            "model_family": "decision_tree",
            "display_name": f"DecisionTree(max_depth={best_dt['max_depth']})",
            "val_pr_auc": best_dt["val_pr_auc"],
            "val_roc_auc": best_dt["val_roc_auc"],
            "retrain_spec": {"type": "decision_tree", "max_depth": best_dt["max_depth"]},
        }
    )
    candidates.append(
        {
            "model_family": "random_forest",
            "display_name": f"RandomForest(n_estimators={best_rf['n_estimators']},max_features={best_rf['max_features']})",
            "val_pr_auc": best_rf["val_pr_auc"],
            "val_roc_auc": best_rf["val_roc_auc"],
            "retrain_spec": {
                "type": "random_forest",
                "n_estimators": best_rf["n_estimators"],
                "max_features": best_rf["max_features"],
            },
        }
    )
    candidates.append(
        {
            "model_family": "boosting",
            "display_name": (
                f"Boosting(algo={best_boost.get('algo')},n_estimators={best_boost['n_estimators']},"
                f"learning_rate={best_boost['learning_rate']})"
            ),
            "val_pr_auc": best_boost["val_pr_auc"],
            "val_roc_auc": best_boost["val_roc_auc"],
            "retrain_spec": {
                "type": "boosting",
                "algo": best_boost.get("algo"),
                "n_estimators": best_boost["n_estimators"],
                "learning_rate": best_boost["learning_rate"],
            },
        }
    )

    if comparison_scope == "all_models":
        for b in baselines:
            candidates.append(
                {
                    "model_family": b["name"],
                    "display_name": (
                        f"{b['name']}(params={json.dumps(b['params'], sort_keys=True)})"
                        if b["params"]
                        else f"{b['name']}"
                    ),
                    "val_pr_auc": b["val_pr_auc"],
                    "val_roc_auc": b["val_roc_auc"],
                    "retrain_spec": {"type": b["name"], **b["params"]},
                }
            )

    best_candidate = _get_best_by_pr_auc(candidates, key="val_pr_auc")

    # Retrain on train + val.
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)

    retrain_spec = best_candidate["retrain_spec"]
    retrain_type = retrain_spec["type"]

    if retrain_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=retrain_spec.get("max_depth"), random_state=seed)
        pipe = _build_pipeline(model)
    elif retrain_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=retrain_spec["n_estimators"],
            max_features=retrain_spec["max_features"],
            random_state=seed,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True,
        )
        pipe = _build_pipeline(model)
    elif retrain_type == "boosting":
        algo = retrain_spec.get("algo")
        if algo == "xgboost":
            try:
                from xgboost import XGBClassifier  # type: ignore

                model = XGBClassifier(
                    n_estimators=retrain_spec["n_estimators"],
                    max_depth=3,
                    learning_rate=retrain_spec["learning_rate"],
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                )
            except Exception:
                # Safety fallback.
                model = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
                    n_estimators=retrain_spec["n_estimators"],
                    learning_rate=retrain_spec["learning_rate"],
                    random_state=seed,
                )
        else:
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
                n_estimators=retrain_spec["n_estimators"],
                learning_rate=retrain_spec["learning_rate"],
                random_state=seed,
            )
        pipe = _build_pipeline(model)
    elif retrain_type == "logistic_regression":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        pipe = _build_pipeline(model)
    elif retrain_type == "naive_bayes":
        pipe = _build_pipeline(GaussianNB())
    elif retrain_type == "knn":
        pipe = _build_pipeline(KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    elif retrain_type == "svm":
        pipe = _build_pipeline(SVC(random_state=seed))
    else:
        raise ValueError(f"Unknown retrain type: {retrain_type}")

    pipe.fit(X_train_final, y_train_final)

    y_pred_test = pipe.predict(X_test)
    scores_test = _get_scores_from_pipeline(pipe, X_test)
    test_pr_auc, test_roc_auc = _compute_pr_auc_roc_auc(y_test, scores_test)
    cm = confusion_matrix(y_test, y_pred_test)

    results["final_model"] = {
        "selected_display_name": best_candidate["display_name"],
        "selected_model_family": best_candidate["model_family"],
        "retrain_spec": retrain_spec,
        "val_pr_auc": float(best_candidate["val_pr_auc"]),
        "val_roc_auc": float(best_candidate["val_roc_auc"]),
        "test_pr_auc": float(test_pr_auc),
        "test_roc_auc": float(test_roc_auc),
        "confusion_matrix": cm.tolist(),
    }

    return results


import json
from typing import Any, Dict, List

import numpy as np

from ensemble_experiments import run_ensemble_experiments


def _best_by_val_pr_auc(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Empty rows.")
    return max(rows, key=lambda r: r["val_pr_auc"])


def _format_float(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _print_comparison_table(results: Dict[str, Any]) -> None:
    dt_best = _best_by_val_pr_auc(results["decision_tree"])
    rf_best = _best_by_val_pr_auc(results["random_forest"])
    boost_best = _best_by_val_pr_auc(results["boosting"])

    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "Model": "Decision Tree",
            "Val PR-AUC": dt_best["val_pr_auc"],
            "Val ROC-AUC": dt_best["val_roc_auc"],
            "Notes": f"max_depth={dt_best['max_depth']}, gap={dt_best['train_pr_auc']-dt_best['val_pr_auc']:.4f}",
        }
    )
    rows.append(
        {
            "Model": "Random Forest",
            "Val PR-AUC": rf_best["val_pr_auc"],
            "Val ROC-AUC": rf_best["val_roc_auc"],
            "Notes": f"n_estimators={rf_best['n_estimators']},max_features={rf_best['max_features']},oob={rf_best['oob_score']:.4f}",
        }
    )
    rows.append(
        {
            "Model": "Boosting",
            "Val PR-AUC": boost_best["val_pr_auc"],
            "Val ROC-AUC": boost_best["val_roc_auc"],
            "Notes": f"algo={boost_best.get('algo')},n_estimators={boost_best['n_estimators']},lr={boost_best['learning_rate']}",
        }
    )

    for b in results.get("baselines", {}).get("rows", []):
        params = b.get("params", {})
        notes = b["name"]
        if params:
            # Keep notes short but deterministic.
            notes = f'{b["name"]}(params={json.dumps(params, sort_keys=True)})'
        rows.append(
            {
                "Model": b["name"],
                "Val PR-AUC": b["val_pr_auc"],
                "Val ROC-AUC": b["val_roc_auc"],
                "Notes": notes,
            }
        )

    header = f"{'Model':<20} | {'Val PR-AUC':>10} | {'Val ROC-AUC':>11} | Notes"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['Model']):<20} | {_format_float(r['Val PR-AUC']):>10} | {_format_float(r['Val ROC-AUC']):>11} | {r['Notes']}"
        )


def main() -> None:
    results = run_ensemble_experiments(
        csv_path=None,
        seed=42,
        comparison_scope="all_models",
    )

    print("Decision Tree sweep (max_depth):")
    for row in sorted(results["decision_tree"], key=lambda r: (r["max_depth"] is None, r["max_depth"])):
        print(
            f"  max_depth={row['max_depth']}: "
            f"train_pr_auc={row['train_pr_auc']:.4f}, val_pr_auc={row['val_pr_auc']:.4f}, "
            f"train_roc_auc={row['train_roc_auc']:.4f}, val_roc_auc={row['val_roc_auc']:.4f}"
        )

    print("\nRandom Forest sweep (n_estimators, max_features):")
    for row in results["random_forest"]:
        print(
            f"  n_estimators={row['n_estimators']}, max_features={row['max_features']}: "
            f"val_pr_auc={row['val_pr_auc']:.4f}, val_roc_auc={row['val_roc_auc']:.4f}, oob_score={row['oob_score']:.4f}"
        )

    print("\nBoosting sweep:")
    for row in results["boosting"]:
        print(
            f"  algo={row.get('algo')}, n_estimators={row['n_estimators']}, lr={row['learning_rate']}: "
            f"val_pr_auc={row['val_pr_auc']:.4f}, val_roc_auc={row['val_roc_auc']:.4f}"
        )

    print("\nOverfitting check (Decision Tree): train vs val PR-AUC gap")
    for row in results["diagnostics"]["decision_tree_train_val_gap"]:
        print(
            f"  max_depth={row['max_depth']}: train_pr_auc={row['train_pr_auc']:.4f}, val_pr_auc={row['val_pr_auc']:.4f}, gap={row['train_val_pr_gap']:.4f}"
        )

    print("\nStability (std of validation PR-AUC across configs):")
    stab = results["diagnostics"]["stability"]
    print(f"  decision_tree_val_pr_auc_std={stab['decision_tree_val_pr_auc_std']:.6f}")
    print(f"  random_forest_val_pr_auc_std={stab['random_forest_val_pr_auc_std']:.6f}")
    print(f"  boosting_val_pr_auc_std={stab['boosting_val_pr_auc_std']:.6f}")

    print("\nRandom Forest: OOB vs Validation")
    for row in results["diagnostics"]["random_forest_oob_vs_val"]:
        print(
            f"  n_estimators={row['n_estimators']}, max_features={row['max_features']}: "
            f"oob_score={row['oob_score']:.4f}, val_pr_auc={row['val_pr_auc']:.4f}, val_roc_auc={row['val_roc_auc']:.4f}"
        )

    print("\nFinal model selection + test evaluation:")
    fm = results["final_model"]
    print(f"  selected_display_name={fm['selected_display_name']}")
    print(f"  val_pr_auc={fm['val_pr_auc']:.4f}, val_roc_auc={fm['val_roc_auc']:.4f}")
    print(f"  test_pr_auc={fm['test_pr_auc']:.4f}, test_roc_auc={fm['test_roc_auc']:.4f}")
    print(f"  confusion_matrix={fm['confusion_matrix']}")

    print("\nComparison table (Model | Val PR-AUC | Val ROC-AUC | Notes):")
    _print_comparison_table(results)

    print("\nresults=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()


"""Train, evaluate and persist the churn pipeline (docs/02-MODEL-SPEC.md).

Run:  python -m src.model.train

Writes artifacts/pipeline.joblib, metrics.json and feature_meta.json. The
Streamlit app loads these; it never trains.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data.loader import load_clean
from src.model.features import build_preprocessor, feature_names, select_features


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
def build_candidates() -> dict[str, Pipeline]:
    """The three candidates from docs/02-MODEL-SPEC.md section 3.

    The dummy is included so the real model's lift is legible rather than
    asserted. class_weight="balanced" reflects the asymmetric cost of a missed
    churner (section 4.2).
    """
    return {
        "dummy": Pipeline(
            [("prep", build_preprocessor()), ("clf", DummyClassifier(strategy="prior"))]
        ),
        # Unweighted: optimises log-loss, so its outputs are probabilities by
        # construction. The asymmetric cost of a missed churner is handled by the
        # decision threshold, NOT by reweighting the classes -- see the note on
        # class_weight below.
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
                ),
            ]
        ),
        # Kept as a candidate so the comparison is visible rather than asserted.
        # class_weight="balanced" scales the positive class by 1/base_rate (~3.8x),
        # which systematically inflates every predicted probability. It barely moves
        # ranking quality, and it wrecks calibration -- which matters here because the
        # agent states these probabilities to users as claims. Probability estimation
        # and the decision threshold are separate concerns and are kept separate.
        "logistic_regression_balanced": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Threshold selection by expected cost (section 4.4)
# ---------------------------------------------------------------------------
def select_threshold(y_true, y_prob, cost_ratio: float = config.COST_FN_OVER_FP) -> float:
    """Pick the threshold minimising FN*cost_ratio + FP.

    Not 0.5. A false negative loses a customer worth hundreds of dollars; a
    false positive costs a retention offer worth tens. Weighting them equally
    would encode a business assumption that is simply false.
    """
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = fn * cost_ratio + fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return round(best_t, 3)


def threshold_sensitivity(y_true, y_prob) -> dict[str, float]:
    """How far the chosen threshold moves as the cost ratio varies.

    If it barely moves, the cost assumption is not load-bearing -- which is
    worth stating in the README rather than leaving the reader to wonder.
    """
    return {f"cost_ratio_{int(r)}": select_threshold(y_true, y_prob, r)
            for r in config.COST_RATIO_SENSITIVITY}


def calibration_bins(y_true, y_prob, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability curve as data, for the notebook plot and the metrics file.

    Calibration is a gate, not a nicety: the agent states these probabilities to
    users as claims, so a model that ranks well but is miscalibrated reports
    numbers that are computed yet wrong.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        out.append(
            {
                "bin_lower": round(float(edges[b]), 2),
                "bin_upper": round(float(edges[b + 1]), 2),
                "n": int(mask.sum()),
                "mean_predicted": round(float(y_prob[mask].mean()), 4),
                "observed_rate": round(float(np.asarray(y_true)[mask].mean()), 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_prob, threshold: float) -> dict[str, Any]:
    """Every metric from docs/02-MODEL-SPEC.md section 4.4."""
    y_true = np.asarray(y_true)
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        # Primary: threshold-free, and its baseline is the churn rate, so lift is honest.
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        # Gate: calibration.
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        # Reported for comparability with published benchmarks. Not selected on.
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        # Operational, at the cost-selected threshold.
        "threshold": threshold,
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, pred, beta=2, zero_division=0)), 4),
        "accuracy": round(float((pred == y_true).mean()), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def max_calibration_gap(y_true, y_prob) -> float:
    """Largest predicted-minus-observed gap across the reliability bins.

    Brier conflates calibration with sharpness; this reports the calibration
    error on its own, weighted by bin size so a 3-row bin cannot dominate.
    """
    bins = calibration_bins(y_true, y_prob)
    populated = [b for b in bins if b["n"] >= 20]
    if not populated:
        return 0.0
    return round(max(abs(b["mean_predicted"] - b["observed_rate"]) for b in populated), 4)


def select_model(results: dict[str, Any]) -> str:
    """PR-AUC is primary; calibration breaks ties (docs/02-MODEL-SPEC.md sec.4.4).

    Ranking quality decides first, because the agent's real questions ("who is
    most likely to churn", "which segment is riskiest") are ordering problems.
    But any candidate within PR_AUC_TIE_TOLERANCE is treated as tied, and the
    tie goes to the best-calibrated model -- because the agent also quotes these
    probabilities as numbers, and a well-ranked but inflated probability is a
    wrong number with an audit trail behind it.

    Among genuinely tied candidates, prefer logistic regression: it is the only
    one giving top_factors an exact per-customer decomposition.
    """
    contenders = {k: v for k, v in results.items() if k != "dummy"}
    best_pr = max(v["cv"]["pr_auc"] for v in contenders.values())
    tied = {
        k: v for k, v in contenders.items()
        if best_pr - v["cv"]["pr_auc"] <= config.PR_AUC_TIE_TOLERANCE
    }

    print(f"\n[select] best PR-AUC={best_pr:.4f}; "
          f"{len(tied)} candidate(s) within {config.PR_AUC_TIE_TOLERANCE}:")
    for name, res in sorted(tied.items(), key=lambda kv: kv[1]["cv"]["brier"]):
        print(f"           {name:<30} PR-AUC={res['cv']['pr_auc']:.4f} "
              f"Brier={res['cv']['brier']:.4f} max_cal_gap={res['cv']['max_calibration_gap']:.3f}")

    best_brier = min(v["cv"]["brier"] for v in tied.values())
    finalists = [k for k, v in tied.items()
                 if v["cv"]["brier"] <= best_brier + 1e-4]
    chosen = "logistic_regression" if "logistic_regression" in finalists else finalists[0]
    print(f"[select] -> {chosen} (ranking tie broken on calibration)")
    return chosen


def main() -> dict[str, Any]:
    df = load_clean()
    X = select_features(df)
    y = df[config.TARGET_COL].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    candidates = build_candidates()
    results: dict[str, Any] = {}
    cv_probs: dict[str, Any] = {}

    # Model selection on cross-validated predictions over the training set only.
    # The test set is touched once, at the end.
    for name, pipe in candidates.items():
        cv_prob = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        cv_probs[name] = cv_prob
        threshold = select_threshold(y_train, cv_prob)
        results[name] = {"cv": evaluate(y_train, cv_prob, threshold)}
        results[name]["cv"]["max_calibration_gap"] = max_calibration_gap(y_train, cv_prob)
        print(f"[cv] {name:<30} PR-AUC={results[name]['cv']['pr_auc']:.4f} "
              f"ROC-AUC={results[name]['cv']['roc_auc']:.4f} "
              f"Brier={results[name]['cv']['brier']:.4f} "
              f"recall={results[name]['cv']['recall']:.4f}")

    chosen = select_model(results)

    # Fit the chosen model on the full training set, evaluate once on the test set.
    pipeline = candidates[chosen]
    pipeline.fit(X_train, y_train)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    threshold = results[chosen]["cv"]["threshold"]  # chosen on validation, not test
    test_metrics = evaluate(y_test, test_prob, threshold)
    results[chosen]["test"] = test_metrics
    # Sensitivity is computed on the same validation predictions the threshold was
    # chosen from, so the table is internally consistent: its COST_FN_OVER_FP entry
    # equals the deployed threshold rather than a different number from the test set.
    results[chosen]["threshold_sensitivity"] = threshold_sensitivity(y_train, cv_probs[chosen])
    results[chosen]["calibration"] = calibration_bins(y_test, test_prob)
    results[chosen]["test"]["max_calibration_gap"] = max_calibration_gap(y_test, test_prob)

    print(f"\n[test] {chosen}: PR-AUC={test_metrics['pr_auc']:.4f} "
          f"ROC-AUC={test_metrics['roc_auc']:.4f} Brier={test_metrics['brier']:.4f}")
    print(f"       recall={test_metrics['recall']:.4f} precision={test_metrics['precision']:.4f} "
          f"F2={test_metrics['f2']:.4f} @ threshold={threshold}")

    # Refit on all data for serving: the app should use every labelled row.
    pipeline.fit(X, y)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.PIPELINE_PATH)

    metrics = {
        "chosen_model": chosen,
        "threshold": threshold,
        "cost_fn_over_fp": config.COST_FN_OVER_FP,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "n_rows": int(len(df)),
        "base_rate": round(float(y.mean()), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    _write_feature_meta(pipeline, X, y)
    print(f"\nSaved {config.PIPELINE_PATH.name}, {config.METRICS_PATH.name}, "
          f"{config.FEATURE_META_PATH.name} to {config.ARTIFACTS_DIR}")
    return metrics


def _write_feature_meta(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> None:
    """Encoded feature names, training means and base rate.

    The means are the reference point for the per-customer attribution in
    service.py -- without them a "contribution" has nothing to be relative to.
    """
    prep = pipeline.named_steps["prep"]
    names = feature_names(prep)
    means = np.asarray(prep.transform(X)).mean(axis=0)

    meta: dict[str, Any] = {
        "feature_names": names,
        "training_means": [float(m) for m in means],
        "base_rate": float(y.mean()),
        "model_features": config.MODEL_FEATURES,
        "supports_exact_attribution": hasattr(pipeline.named_steps["clf"], "coef_"),
    }
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        meta["coefficients"] = [float(c) for c in clf.coef_[0]]
        meta["intercept"] = float(clf.intercept_[0])
    config.FEATURE_META_PATH.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

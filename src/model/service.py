"""Callable prediction tools (docs/02-MODEL-SPEC.md section 6).

Pure Python: no Streamlit imports, no printing, no LLM awareness. These are the
functions the agent calls, and the same ones the notebook and tests use.

Error contract: these functions RETURN error dicts, they do not raise. The agent
must be able to react to a bad customer ID rather than crash on it.
"""
from __future__ import annotations

import functools
import json
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.loader import load_clean
from src.model.features import map_output_to_source, select_features


class ModelUnavailable(RuntimeError):
    """Raised only at load time, so the app can show a setup message."""


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _load_artifacts():
    if not config.PIPELINE_PATH.exists():
        raise ModelUnavailable(
            f"Model artifact not found at {config.PIPELINE_PATH}. "
            "Run `python -m src.model.train` first."
        )
    pipeline = joblib.load(config.PIPELINE_PATH)
    meta = json.loads(config.FEATURE_META_PATH.read_text())
    metrics = json.loads(config.METRICS_PATH.read_text())
    return pipeline, meta, metrics


def model_status() -> tuple[bool, str]:
    """(ready, reason). The reason is the ACTUAL failure, not a guess.

    This used to swallow every exception and report "model not trained", which
    is wrong for any cause other than a missing file -- and the likeliest cause
    in a deployment is not a missing file at all but a version skew: an artifact
    pickled by one scikit-learn cannot always be read by another. Reporting the
    real error is the difference between a five-minute fix and an hour.
    """
    try:
        _load_artifacts()
        return True, ""
    except ModelUnavailable as exc:
        return False, str(exc)
    except Exception as exc:
        return False, (
            f"The model artifact exists but could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        )


def model_is_ready() -> bool:
    return model_status()[0]


def ensure_model(retrain_on_failure: bool = True) -> tuple[bool, str]:
    """Load the model, retraining once if the stored artifact is unusable.

    A pickled sklearn pipeline is only guaranteed readable by the version that
    wrote it. Pinning the dependency makes that rare; retraining makes it
    recoverable. Training costs a few seconds and random_state is fixed, so the
    rebuilt model is identical to the one that was committed.
    """
    ready, reason = model_status()
    if ready or not retrain_on_failure:
        return ready, reason

    try:
        from src.model.train import main as train_model

        train_model()
        _load_artifacts.cache_clear()
        _scored_population.cache_clear()
        ready, reason = model_status()
        if ready:
            return True, "Stored model artifact was unusable, so it was retrained on startup."
        return False, reason
    except Exception as exc:
        return False, f"{reason} Retraining also failed: {type(exc).__name__}: {exc}"


def get_threshold() -> float:
    _, _, metrics = _load_artifacts()
    return float(metrics.get("threshold", 0.5))


def get_model_info() -> dict[str, Any]:
    """Model card for the UI: which model, which metrics, when trained."""
    _, _, metrics = _load_artifacts()
    chosen = metrics["chosen_model"]
    return {
        "chosen_model": chosen,
        "threshold": metrics["threshold"],
        "base_rate": metrics["base_rate"],
        "trained_at": metrics["trained_at"],
        "test_metrics": metrics["models"][chosen].get("test", {}),
        "primary_metric": "pr_auc",
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _scored_population() -> pd.DataFrame:
    """Every customer scored once, cached.

    Backs percentile ranking and segment aggregation. Scoring 7,043 rows is
    milliseconds, and doing it once keeps every risk figure in the app mutually
    consistent.
    """
    pipeline, _, _ = _load_artifacts()
    df = load_clean()
    df["risk_score"] = pipeline.predict_proba(select_features(df))[:, 1]
    return df


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "message": message, **extra}


def _validate_overrides(overrides: dict, df: pd.DataFrame) -> dict[str, Any] | None:
    """Reject unknown columns and illegal category values before scoring.

    This is the guard that stops a `region` override silently doing nothing --
    see docs/01-DATA-FINDINGS.md finding I10.
    """
    for key, value in overrides.items():
        if key not in config.MODEL_FEATURES:
            return _err(
                "unknown_feature",
                f"'{key}' is not a feature of this model.",
                valid_features=config.MODEL_FEATURES,
            )
        if key in config.NUMERIC_FEATURES:
            try:
                float(value)
            except (TypeError, ValueError):
                return _err("invalid_value", f"'{key}' must be numeric, got {value!r}.")
        else:
            allowed = sorted(str(v) for v in df[key].unique())
            if str(value) not in allowed:
                return _err(
                    "invalid_value",
                    f"'{value}' is not a valid value for '{key}'.",
                    allowed_values=allowed,
                )
    return None


def _row_to_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row[config.MODEL_FEATURES].to_dict()])


def _score_frame(frame: pd.DataFrame) -> float:
    pipeline, _, _ = _load_artifacts()
    return float(pipeline.predict_proba(frame)[:, 1][0])


# ---------------------------------------------------------------------------
# top_factors -- exact per-customer attribution (section 6.2)
# ---------------------------------------------------------------------------
def _top_factors(frame: pd.DataFrame, n: int = 5) -> tuple[list[dict[str, Any]], list[str]]:
    """Decompose this customer's log-odds into per-feature contributions.

    For a linear model, contribution_j = coef_j * (x_ij - mean_j), and the
    contributions sum exactly to logit(p_i) - logit(p_base). That is a proof
    the explanation matches the prediction, not an approximation of it -- which
    matters because the agent surfaces these factors to users as claims.

    Global feature_importances_ is deliberately not used: it says what matters
    on average, and the brief asks what drives *this* customer.
    """
    pipeline, meta, _ = _load_artifacts()
    warnings: list[str] = []

    if not meta.get("supports_exact_attribution"):
        warnings.append(
            "The deployed model is not linear, so factor attributions are unavailable. "
            "Retrain with logistic regression for exact per-customer attribution."
        )
        return [], warnings

    prep = pipeline.named_steps["prep"]
    x = np.asarray(prep.transform(frame))[0]
    coefs = np.asarray(meta["coefficients"])
    means = np.asarray(meta["training_means"])
    names = meta["feature_names"]

    contributions = coefs * (x - means)
    order = np.argsort(np.abs(contributions))[::-1][:n]

    factors = []
    for i in order:
        if abs(contributions[i]) < 1e-9:
            continue
        source, level = map_output_to_source(names[i])
        raw_value = frame[source].iloc[0] if source in frame.columns else level
        direction = "increases" if contributions[i] > 0 else "decreases"
        factors.append(
            {
                "feature": source,
                "value": _clean_value(raw_value),
                "direction": direction,
                "contribution": round(float(contributions[i]), 4),
                "note": _population_note(source, raw_value, direction),
            }
        )
    return factors, warnings


def _ordinal(pct: float) -> str:
    """Percentile with the right English suffix -- '3rd', not '3th'."""
    n = int(round(pct))
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _clean_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 2)
    return str(v) if not isinstance(v, (int, float, str, bool)) else v


def _population_note(column: str, value: Any, direction: str = "") -> str:
    """Anchor a factor in the actual data distribution.

    This is the model tool invoking the dataset tool (brief, Dataset behavior
    item 7): the explanation cites a real population statistic rather than a
    bare coefficient, so every number in it traces to a computation.
    """
    try:
        df = _scored_population()
        if column not in df.columns:
            return ""
        base = df[config.TARGET_COL].mean()

        if column in config.NUMERIC_FEATURES:
            pct = float((df[column] < float(value)).mean() * 100)
            median = float(df[column].median())
            return (
                f"{column} of {_clean_value(value)} sits in the {_ordinal(pct)} percentile "
                f"(population median {median:.2f})."
            )

        subset = df[df[column].astype(str) == str(value)]
        if subset.empty:
            return ""
        rate = float(subset[config.TARGET_COL].mean())
        note = (
            f"{column}={value} customers churn at {rate * 100:.1f}% "
            f"vs {base * 100:.1f}% overall (n={len(subset)})."
        )
        # The contribution is a conditional effect (all else held equal) while this
        # note is a marginal rate. They can legitimately point opposite ways --
        # confounded groups are the usual cause. Say so rather than leaving the
        # user to read the two as a contradiction.
        if direction and ((direction == "increases") != (rate > base)):
            note += (
                " Note: the model's effect here is the opposite of this raw rate, because "
                "the contribution holds every other feature constant while this rate does not."
            )
        return note
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public tool 1: predict_churn_risk
# ---------------------------------------------------------------------------
def predict_churn_risk(
    customer_id: str | None = None,
    features: dict | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Churn risk for one customer, real or hypothetical.

    Three modes:
      * customer_id            -- score that customer as they stand today
      * customer_id + overrides -- what-if: that customer with fields changed
      * features               -- a fully hypothetical new customer

    Returns a dict; on failure returns {"error": ..., "message": ...} rather
    than raising, so the agent can recover.
    """
    try:
        df = _scored_population()
    except ModelUnavailable as exc:
        return _err("model_unavailable", str(exc))

    overrides = overrides or {}

    # ---- resolve the base row -------------------------------------------
    if customer_id is not None:
        cid = str(customer_id).strip()
        match = df[df[config.ID_COL].astype(str).str.upper() == cid.upper()]
        if match.empty:
            return _err(
                "customer_not_found",
                f"No customer with ID '{customer_id}'. IDs look like 7590-VHVEG.",
                did_you_mean=_similar_ids(cid, df),
            )
        row = match.iloc[0]
        frame = _row_to_frame(row)
        baseline_score = float(row["risk_score"])
        actual = int(row[config.TARGET_COL])
    elif features is not None:
        missing = [c for c in config.MODEL_FEATURES if c not in features and c != "avg_monthly_spend"]
        if missing:
            return _err(
                "missing_required_features",
                f"Missing {len(missing)} required feature(s).",
                missing_features=missing,
            )
        supplied = dict(features)
        # Derived feature (C8) -- compute rather than demand it from the caller.
        if "avg_monthly_spend" not in supplied:
            tenure = float(supplied.get("tenure", 0) or 0)
            supplied["avg_monthly_spend"] = (
                float(supplied["TotalCharges"]) / tenure if tenure else float(supplied["MonthlyCharges"])
            )
        frame = pd.DataFrame([{c: supplied[c] for c in config.MODEL_FEATURES}])
        invalid = _validate_overrides({k: v for k, v in supplied.items() if k in config.MODEL_FEATURES}, df)
        if invalid:
            return invalid
        baseline_score = None
        actual = None
    else:
        return _err(
            "missing_required_features",
            "Provide either customer_id or a complete features dict.",
        )

    # ---- apply what-if overrides ----------------------------------------
    if overrides:
        invalid = _validate_overrides(overrides, df)
        if invalid:
            return invalid
        for key, value in overrides.items():
            frame.loc[:, key] = float(value) if key in config.NUMERIC_FEATURES else str(value)
        # Keep the derived feature consistent with the overridden inputs.
        tenure = float(frame["tenure"].iloc[0])
        frame.loc[:, "avg_monthly_spend"] = (
            float(frame["TotalCharges"].iloc[0]) / tenure if tenure else float(frame["MonthlyCharges"].iloc[0])
        )

    # ---- score -----------------------------------------------------------
    try:
        score = _score_frame(frame)
    except Exception as exc:
        return _err("scoring_failed", f"Could not score this input: {exc}")

    factors, warnings = _top_factors(frame)
    percentile = float((df["risk_score"] < score).mean() * 100)

    result: dict[str, Any] = {
        "customer_id": customer_id,
        "risk_score": round(score, 4),
        "risk_band": config.risk_band(score),
        "threshold": get_threshold(),
        "predicted_churn": bool(score >= get_threshold()),
        "population_base_rate": round(float(df[config.TARGET_COL].mean()), 4),
        "percentile": round(percentile, 1),
        "top_factors": factors,
        "applied_overrides": overrides,
        "warnings": warnings,
    }
    if overrides and baseline_score is not None:
        result["baseline_risk_score"] = round(baseline_score, 4)
        result["risk_delta"] = round(score - baseline_score, 4)
    if actual is not None:
        result["actual_churn"] = actual
    return result


def _similar_ids(cid: str, df: pd.DataFrame, n: int = 3) -> list[str]:
    """Cheap prefix suggestions, so a typo'd ID gets a helpful reply."""
    ids = df[config.ID_COL].astype(str)
    prefix = cid[:4]
    hits = ids[ids.str.startswith(prefix)].head(n).tolist()
    return hits or ids.head(n).tolist()


# ---------------------------------------------------------------------------
# Public tool 2: predict_segment_risk
# ---------------------------------------------------------------------------
def predict_segment_risk(
    filters: dict | None = None,
    overrides: dict | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Aggregate churn risk across a segment defined by column filters.

    Returns predicted mean_risk AND observed actual_churn_rate together: that
    gives the agent a free calibration check on every segment query, and gives
    the user an immediate sense of whether to trust the number.
    """
    try:
        df = _scored_population()
    except ModelUnavailable as exc:
        return _err("model_unavailable", str(exc))

    filters = filters or {}
    mask = pd.Series(True, index=df.index)

    for column, value in filters.items():
        if column not in df.columns:
            return _err(
                "unknown_feature",
                f"'{column}' is not a column in this dataset.",
                valid_columns=[c for c in df.columns if c != "risk_score"],
            )
        if isinstance(value, dict):  # range filter, e.g. {"min": 0, "max": 12}
            if "min" in value:
                mask &= df[column] >= float(value["min"])
            if "max" in value:
                mask &= df[column] <= float(value["max"])
        elif isinstance(value, (list, tuple)):
            mask &= df[column].astype(str).isin([str(v) for v in value])
        else:
            mask &= df[column].astype(str) == str(value)

    segment = df[mask]
    if segment.empty:
        return {
            "n_customers": 0,
            "warning": "no_customers_match",
            "message": "No customers match these filters.",
            "filters_applied": filters,
        }

    # What-if across a whole segment: rescore every matching row with overrides.
    if overrides:
        invalid = _validate_overrides(overrides, df)
        if invalid:
            return invalid
        frame = segment[config.MODEL_FEATURES].copy()
        for key, value in overrides.items():
            frame.loc[:, key] = float(value) if key in config.NUMERIC_FEATURES else str(value)
        tenure = frame["tenure"].replace(0, np.nan)
        frame.loc[:, "avg_monthly_spend"] = (
            (frame["TotalCharges"] / tenure).fillna(frame["MonthlyCharges"])
        )
        pipeline, _, _ = _load_artifacts()
        scores = pipeline.predict_proba(frame)[:, 1]
        baseline_mean = float(segment["risk_score"].mean())
    else:
        scores = segment["risk_score"].to_numpy()
        baseline_mean = None

    base_rate = float(df[config.TARGET_COL].mean())
    actual_rate = float(segment[config.TARGET_COL].mean())
    bands = pd.Series([config.risk_band(s) for s in scores]).value_counts().to_dict()

    top = (
        segment.assign(_s=scores)
        .nlargest(min(top_n, len(segment)), "_s")[[config.ID_COL, "_s"]]
        .rename(columns={config.ID_COL: "customer_id", "_s": "risk_score"})
    )

    result: dict[str, Any] = {
        "n_customers": int(len(segment)),
        "mean_risk": round(float(np.mean(scores)), 4),
        "median_risk": round(float(np.median(scores)), 4),
        "actual_churn_rate": round(actual_rate, 4),
        "actual_churn_count": int(segment[config.TARGET_COL].sum()),
        "population_base_rate": round(base_rate, 4),
        "lift": round(actual_rate / base_rate, 2) if base_rate else None,
        "risk_distribution": {k: int(v) for k, v in bands.items()},
        "top_customers": [
            {"customer_id": r.customer_id, "risk_score": round(float(r.risk_score), 4)}
            for r in top.itertuples()
        ],
        "filters_applied": filters,
        "applied_overrides": overrides or {},
        "warnings": [],
    }
    if baseline_mean is not None:
        result["baseline_mean_risk"] = round(baseline_mean, 4)
        result["mean_risk_delta"] = round(float(np.mean(scores)) - baseline_mean, 4)

    # Calibration self-check (docs/03-AGENT-SPEC.md section 5): warn, do not block.
    if abs(result["mean_risk"] - actual_rate) > 0.25:
        result["warnings"].append(
            f"Predicted mean risk ({result['mean_risk']:.2f}) diverges from the observed "
            f"churn rate ({actual_rate:.2f}) for this segment -- treat the prediction with caution."
        )
    return result


def score_all_customers() -> pd.DataFrame:
    """Population with risk_score attached. For the UI and the agent's executor."""
    return _scored_population().copy()

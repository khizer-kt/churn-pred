"""Model service tests (docs/02-MODEL-SPEC.md section 7).

Skipped wholesale if artifacts/pipeline.joblib is absent, so the suite still
runs before the first training pass.
"""
from __future__ import annotations

import math

import pytest

from src import config
from src.model import service

pytestmark = pytest.mark.skipif(
    not service.model_is_ready(), reason="model artifacts not built; run `python -m src.model.train`"
)

KNOWN_ID = "7590-VHVEG"


def test_predict_returns_documented_shape():
    out = service.predict_churn_risk(KNOWN_ID)
    assert "error" not in out
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["risk_band"] in {"Low", "Medium", "High"}
    assert out["top_factors"]
    assert 0.0 <= out["percentile"] <= 100.0


def test_customer_id_is_case_insensitive():
    assert service.predict_churn_risk(KNOWN_ID.lower())["risk_score"] == \
           service.predict_churn_risk(KNOWN_ID)["risk_score"]


def test_unknown_id_returns_error_and_does_not_raise():
    out = service.predict_churn_risk("NOT-AN-ID")
    assert out["error"] == "customer_not_found"
    assert out["did_you_mean"]


def test_attribution_sums_to_the_logit():
    """The proof that top_factors matches the prediction rather than approximating it.

    contribution_j = coef_j * (x_ij - mean_j) sums exactly to
    logit(p_i) - logit(p_base_at_means). Checked against the full contribution
    vector, not just the top 5 that get surfaced.
    """
    import numpy as np
    import pandas as pd

    pipeline, meta, _ = service._load_artifacts()
    if not meta.get("supports_exact_attribution"):
        pytest.skip("non-linear model deployed")

    df = service.score_all_customers()
    row = df[df[config.ID_COL] == KNOWN_ID].iloc[0]
    frame = pd.DataFrame([row[config.MODEL_FEATURES].to_dict()])

    x = np.asarray(pipeline.named_steps["prep"].transform(frame))[0]
    coefs = np.asarray(meta["coefficients"])
    means = np.asarray(meta["training_means"])

    total = float((coefs * (x - means)).sum())
    reference = float(coefs @ means) + float(meta["intercept"])
    score = float(row["risk_score"])
    assert math.isclose(total + reference, math.log(score / (1 - score)), abs_tol=1e-6)


def test_override_lowers_risk_in_the_expected_direction():
    """A two-year contract must reduce risk. A failure here means the encoding is broken."""
    base = service.predict_churn_risk(KNOWN_ID)
    whatif = service.predict_churn_risk(KNOWN_ID, overrides={"Contract": "Two year"})
    assert whatif["risk_score"] < base["risk_score"]
    assert whatif["risk_delta"] < 0
    assert whatif["baseline_risk_score"] == base["risk_score"]


def test_invalid_override_value_is_rejected_with_the_legal_set():
    out = service.predict_churn_risk(KNOWN_ID, overrides={"Contract": "Three year"})
    assert out["error"] == "invalid_value"
    assert "Two year" in out["allowed_values"]


def test_unknown_override_column_is_rejected():
    """The guard that stops a `region` override from silently doing nothing (I10)."""
    out = service.predict_churn_risk(KNOWN_ID, overrides={"region": "North"})
    assert out["error"] == "unknown_feature"


def test_hypothetical_customer_can_be_scored():
    features = {
        "tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0,
        "gender": "Female", "SeniorCitizen": "Yes", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    out = service.predict_churn_risk(features=features)
    assert "error" not in out
    # This is the canonical high-risk profile; it must not score as safe.
    assert out["risk_score"] > 0.5


def test_incomplete_hypothetical_is_rejected():
    out = service.predict_churn_risk(features={"tenure": 5})
    assert out["error"] == "missing_required_features"
    assert out["missing_features"]


def test_segment_reports_predicted_and_observed_together():
    out = service.predict_segment_risk({"Contract": "Month-to-month",
                                        "InternetService": "Fiber optic",
                                        "PaymentMethod": "Electronic check"})
    assert out["n_customers"] == 1307
    assert round(out["actual_churn_rate"], 4) == 0.6037   # ground truth, findings sec.5
    assert out["mean_risk"] > out["population_base_rate"]
    assert out["lift"] > 2


def test_empty_segment_returns_warning_not_exception():
    """PhoneService=No with MultipleLines=Yes is impossible by construction.

    Verified in the profiling pass: zero violations of the
    PhoneService <-> MultipleLines relationship (finding I11). A segment that
    cannot exist must come back as a warning, not an exception.
    """
    out = service.predict_segment_risk({"PhoneService": "No", "MultipleLines": "Yes"})
    assert out["n_customers"] == 0
    assert out["warning"] == "no_customers_match"


def test_zero_tenure_two_year_segment_is_not_empty():
    """Finding I2: 10 of the 11 never-billed customers are on two-year contracts."""
    out = service.predict_segment_risk({"Contract": "Two year", "tenure": {"min": 0, "max": 0}})
    assert out["n_customers"] == 10
    assert out["actual_churn_rate"] == 0.0


def test_unknown_segment_column_is_rejected():
    out = service.predict_segment_risk({"region": "North"})
    assert out["error"] == "unknown_feature"


def test_model_beats_the_no_skill_baseline():
    import json
    metrics = json.loads(config.METRICS_PATH.read_text())
    chosen = metrics["chosen_model"]
    assert metrics["models"][chosen]["cv"]["pr_auc"] > metrics["models"]["dummy"]["cv"]["pr_auc"]
    # Above ~0.75 on this dataset means leakage, not skill (docs/02-MODEL-SPEC sec.4.5).
    assert metrics["models"][chosen]["cv"]["pr_auc"] < 0.75

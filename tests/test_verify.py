"""Self-check tests (docs/03-AGENT-SPEC.md section 5)."""
from __future__ import annotations

from src.agent.verify import Status, verify


def test_clean_result_passes():
    assert verify({"n_customers": 100, "mean_risk": 0.4, "actual_churn_rate": 0.38}).status is Status.OK


def test_unknown_column_triggers_replan():
    """A plan referencing a column that does not exist should be rethought, not retried."""
    v = verify({"error": "unknown_feature", "message": "'region' is not a column"})
    assert v.status is Status.REPLAN


def test_execution_error_triggers_retry():
    assert verify({"error": "execution_error", "detail": "KeyError"}).status is Status.RETRY


def test_missing_model_is_fatal():
    assert verify({"error": "model_unavailable", "message": "no artifact"}).status is Status.FATAL


def test_empty_segment_warns_rather_than_failing():
    """'No customers match' can be the true answer, so it must stay usable."""
    v = verify({"n_customers": 0, "warning": "no_customers_match"})
    assert v.status is Status.WARN and v.ok


def test_out_of_range_probability_is_fatal():
    assert verify({"risk_score": 1.4}).status is Status.FATAL


def test_impossible_count_is_fatal():
    """More rows than exist means the query is double-counting."""
    assert verify({"n_customers": 99999}).status is Status.FATAL


def test_negative_count_is_fatal():
    assert verify({"n_customers": -5}).status is Status.FATAL


def test_zero_row_query_is_retryable():
    assert verify({"ok": True, "kind": "dataframe", "rows": 0}).status is Status.RETRY

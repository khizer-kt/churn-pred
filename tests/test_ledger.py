"""Fact Ledger and numeric-grounding tests.

The single most important test in the suite is
`test_fabricated_number_is_rejected` -- it is the direct evidence for the
brief's "never invent a number" requirement.
"""
from __future__ import annotations

from src.agent.ledger import FactLedger, template_answer, validate_numbers


def build_ledger() -> FactLedger:
    ledger = FactLedger()
    ledger.add(0.2654, "overall churn rate", unit="probability", source_tool="get_distribution")
    ledger.add(42.71, "churn rate for Contract=Month-to-month", unit="percent")
    ledger.add(3875, "customer count, Contract=Month-to-month", unit="count")
    return ledger


def test_facts_get_sequential_keys():
    ledger = build_ledger()
    assert [f.key for f in ledger] == ["F1", "F2", "F3"]


def test_formatting_follows_unit():
    ledger = build_ledger()
    assert ledger.get("F1").format() == "26.5%"
    assert ledger.get("F2").format() == "42.71%"
    assert ledger.get("F3").format() == "3,875"


def test_substitution_replaces_tokens():
    ledger = build_ledger()
    text, unknown = ledger.substitute("Churn is [[F1]] overall.")
    assert "26.5%" in text and not unknown


def test_unknown_citation_key_is_caught():
    """A token for a fact that does not exist means the model invented a citation."""
    ledger = build_ledger()
    text, unknown = ledger.substitute("Churn is [[F99]].")
    assert unknown == ["F99"]
    assert "unavailable" in text


def test_grounded_answer_passes():
    ledger = build_ledger()
    text, _ = ledger.substitute("Month-to-month churn is [[F2]] across [[F3]] customers.")
    assert validate_numbers(text, ledger, substituted=["42.71%", "3,875"]).ok


def test_fabricated_number_is_rejected():
    """The core guarantee: a plausible figure nobody computed must not pass."""
    ledger = build_ledger()
    answer = "Month-to-month customers churn at 42.71%, and fiber customers at 38.4%."
    result = validate_numbers(answer, ledger)
    assert not result.ok
    assert "38.4" in " ".join(result.unmatched)


def test_percent_and_fraction_renderings_both_match():
    """0.2654 in the ledger must satisfy 26.54%, 26.5% and 0.265 in the text."""
    ledger = build_ledger()
    for rendering in ("26.54%", "26.5%", "0.265"):
        assert validate_numbers(f"The rate is {rendering}.", ledger).ok, rendering


def test_numbers_echoed_from_the_question_are_allowed():
    ledger = build_ledger()
    question = "How many customers have tenure over 36 months?"
    assert validate_numbers("Among customers over 36 months...", ledger, question=question).ok


def test_small_ordinals_are_allowed():
    ledger = build_ledger()
    assert validate_numbers("Here are the top 5 drivers, in 3 groups.", ledger).ok


def test_register_result_flattens_nested_tool_output():
    ledger = FactLedger()
    ledger.register_result(
        {"n_customers": 1307, "mean_risk": 0.591,
         "top_customers": [{"customer_id": "X", "risk_score": 0.94}]},
        tool="predict_segment_risk",
    )
    values = {f.value for f in ledger}
    assert {1307, 0.591, 0.94} <= values


def test_error_fields_are_not_registered_as_facts():
    ledger = FactLedger()
    ledger.register_result({"error": "unknown_feature", "message": "no column 'region'"},
                           tool="get_distribution")
    assert len(ledger) == 0


def test_booleans_are_not_quotable_facts():
    ledger = FactLedger()
    ledger.register_result({"predicted_churn": True, "risk_score": 0.8}, tool="predict_churn_risk")
    assert [f.value for f in ledger] == [0.8]


def test_template_fallback_lists_what_was_computed():
    ledger = build_ledger()
    text = template_answer(ledger)
    assert "42.71%" in text and "3,875" in text


def test_template_fallback_with_no_facts():
    assert "could not compute" in template_answer(FactLedger())


# --- unit-aware matching -----------------------------------------------------
def test_count_cannot_justify_a_percentage():
    """The hole found in live testing: a fact whose value is 1 must not license
    the claim "100.0%".

    The agent had written "actual churn of 100.0%" about a single customer whose
    actual_churn flag was 1. Blanket x100 matching let it through. Scale
    conversion is now restricted to facts that are genuinely ratios.
    """
    ledger = FactLedger()
    ledger.add(1, "customers matching the filter", unit="count")
    assert not validate_numbers("Churn ran at 100.0% in that group.", ledger).ok


def test_probability_still_accepts_percent_rendering():
    ledger = FactLedger()
    ledger.add(0.4271, "churn rate for month-to-month", unit="probability")
    assert validate_numbers("They churn at 42.7%.", ledger).ok
    assert validate_numbers("The rate is 0.427.", ledger).ok


def test_delta_is_rendered_in_percentage_points():
    """A risk moving 0.29 -> 0.10 is -19.5 pp, not -19.5%."""
    ledger = FactLedger()
    f = ledger.add(-0.1948, "current risk vs two-year contract.risk_delta")
    assert f.unit == "delta"
    assert f.format() == "-19.5 pp"


def test_outcome_flags_are_not_registered_as_facts():
    """actual_churn is a 0/1 label for one customer, not a rate."""
    ledger = FactLedger()
    ledger.register_result(
        {"risk_score": 0.29, "actual_churn": 1, "predicted_churn": True},
        tool="predict_churn_risk",
    )
    assert [f.value for f in ledger] == [0.29]


def test_list_items_are_labelled_by_identity_not_index():
    """'levels[0]' makes the answerer say 'the first level'; name it instead."""
    ledger = FactLedger()
    ledger.register_result(
        {"levels": [{"value": "Month-to-month", "churn_rate": 0.4271, "count": 3875}]},
        tool="get_distribution",
    )
    assert any("Month-to-month" in f.label for f in ledger)


def test_rate_is_detected_mid_label():
    """'overall churn rate.value' must format as a percentage, not 0.2654."""
    ledger = FactLedger()
    assert ledger.add(0.2654, "overall churn rate.value").format() == "26.5%"


def test_purpose_text_does_not_contaminate_field_units():
    """A step whose purpose mentions a "change" must not turn its `threshold`
    into a delta.

    Observed live: the agent reported an unchanged threshold of 0.28 as
    "+28.0 pp", because the purpose text was matched against every field the
    step produced. Real field names win; only placeholder leaves consult the
    surrounding label. The label below deliberately contains "change" and omits
    any other hint word, so it fails if the precedence rule is dropped.
    """
    ledger = FactLedger()
    f = ledger.add(0.28, "contract change for the customer.threshold")
    assert f.unit == "probability", f"leaf must win, got {f.unit}"
    assert f.format() == "28.0%"


def test_delta_guard_rejects_out_of_range_values():
    """A count in a step about a "change" is not a delta."""
    ledger = FactLedger()
    assert ledger.add(3875, "contract change.n_customers").unit == "count"


def test_generic_leaf_still_consults_the_full_label():
    ledger = FactLedger()
    assert ledger.add(0.2654, "overall churn rate.value").unit == "probability"

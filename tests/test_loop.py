"""Agent loop tests against a stubbed LLM.

No network, no API key, no quota. The loop's job is orchestration -- planning,
verification, retry, and the numeric-grounding gate -- and all of that is
deterministic given the model's output. Stubbing the model is what makes those
branches testable at all; a live model cannot be made to fail on demand.
"""
from __future__ import annotations

import json

import pytest

from src.agent.loop import ChurnAgent
from src.llm.client import LLMResponse, LLMUnavailable


class FakeClient:
    """Returns queued responses in order. Records what it was asked."""

    def __init__(self, responses: list[str], available: bool = True):
        self.responses = list(responses)
        self.available = available
        self.reason = "" if available else "no API key configured"
        self.model = "fake"
        self.calls: list[list[dict]] = []

        class _Usage:
            calls = 0

            def to_dict(self):
                return {}

        self.usage = _Usage()

    def complete(self, messages, json_mode=False, temperature=0.0, max_tokens=2048):
        self.calls.append(messages)
        self.usage.calls += 1
        if not self.responses:
            raise AssertionError("FakeClient ran out of queued responses")
        return LLMResponse(text=self.responses.pop(0))


def plan(*steps, missing=None):
    return json.dumps({"reasoning": "test", "missing_columns": missing or [],
                       "steps": list(steps)})


def step(tool, **arguments):
    return {"tool": tool, "arguments": arguments, "purpose": f"{tool} result"}


# ---------------------------------------------------------------------------
def test_single_step_question_costs_two_llm_calls():
    """The efficiency claim in the spec, asserted rather than assumed."""
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Month-to-month churn is [[F4]].",
    ]))
    response = agent.answer("churn by contract?")
    assert agent.client.usage.calls == 2
    assert response.validation_passed and not response.fell_back
    assert "%" in response.text


def test_multi_step_plan_executes_in_order():
    agent = ChurnAgent(FakeClient([
        plan(step("predict_segment_risk", filters={}),
             step("get_distribution", column="Contract")),
        "Risk is [[F1]] across the population.",
    ]))
    response = agent.answer("who churns and does contract matter?")
    assert [s.tool for s in response.steps] == ["predict_segment_risk", "get_distribution"]
    assert all(s.status == "ok" for s in response.steps)


def test_fabricated_number_triggers_retry_then_fallback():
    """The central guarantee, end to end.

    The stub returns an ungrounded figure twice. The loop must reject both and
    degrade to the deterministic template rather than show either.
    """
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Fiber customers churn at 38.4%.",   # never computed
        "Actually it is 51.2%.",             # still never computed
    ]))
    response = agent.answer("churn by contract?")
    assert not response.validation_passed
    assert response.fell_back
    assert "38.4" not in response.text and "51.2" not in response.text
    assert "Here is what I computed" in response.text


def test_grounded_answer_is_accepted_on_the_retry():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Churn is 99.9%.",          # rejected
        "Churn is [[F4]].",         # grounded
    ]))
    response = agent.answer("churn by contract?")
    assert response.validation_passed and not response.fell_back
    assert "99.9" not in response.text


def test_missing_column_refuses_without_computing():
    """A precise refusal is the correct answer, not a failure."""
    agent = ChurnAgent(FakeClient([plan(missing=["region"])]))
    response = agent.answer("does churn correlate with region?")
    assert response.steps == []
    assert "region" in response.missing_columns
    assert "does not contain" in response.text
    # No answerer call: the refusal is generated deterministically.
    assert agent.client.usage.calls == 1


def test_absent_concept_caught_even_if_planner_misses_it():
    """Pre-flight runs regardless of what the planner decided."""
    agent = ChurnAgent(FakeClient([plan()]))
    response = agent.answer("break churn down by region please")
    assert "region" in response.missing_columns


def test_bad_tool_name_is_dropped_not_dispatched():
    agent = ChurnAgent(FakeClient([
        plan(step("nonexistent_tool", x=1), step("get_distribution", column="Contract")),
        "Churn is [[F4]].",
    ]))
    response = agent.answer("churn by contract?")
    assert [s.tool for s in response.steps] == ["get_distribution"]


def test_arguments_supplied_as_a_json_string_are_parsed():
    """Free-tier models sometimes stringify the arguments object."""
    agent = ChurnAgent(FakeClient([
        json.dumps({"steps": [{"tool": "get_distribution",
                               "arguments": '{"column": "Contract"}',
                               "purpose": "p"}]}),
        "Churn is [[F4]].",
    ]))
    response = agent.answer("churn by contract?")
    assert response.steps[0].arguments == {"column": "Contract"}


def test_unparseable_plan_degrades_gracefully():
    """No usable plan -> the deterministic capabilities reply, not an error."""
    agent = ChurnAgent(FakeClient(["this is not json at all"]))
    response = agent.answer("churn by contract?")
    assert response.steps == []
    assert response.error is None
    assert "you can ask" in response.text.lower() or "churn dataset" in response.text.lower()


def test_failed_step_triggers_a_replan():
    """An unknown column means the plan was wrong, so re-plan rather than retry."""
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="region")),          # will fail
        plan(step("get_distribution", column="Contract")),        # corrected
        "Churn is [[F4]].",
    ]))
    response = agent.answer("churn by area?")
    assert [s.tool for s in response.steps] == ["get_distribution", "get_distribution"]
    assert response.steps[0].status == "replan"
    assert response.validation_passed


def test_unknown_citation_key_is_rejected():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Churn is [[F999]].",
        "Churn is [[F4]].",
    ]))
    response = agent.answer("churn by contract?")
    assert response.validation_passed
    assert "unavailable" not in response.text


def test_no_api_key_returns_a_message_not_an_exception():
    agent = ChurnAgent(FakeClient([], available=False))
    response = agent.answer("anything")
    assert response.error == "llm_unavailable"
    assert "Explore" in response.text


def test_llm_failure_mid_answer_degrades_to_computed_facts():
    """Facts already computed must survive the model going away."""
    class Failing(FakeClient):
        def complete(self, messages, **kwargs):
            self.usage.calls += 1
            if self.responses:
                return LLMResponse(text=self.responses.pop(0))
            raise LLMUnavailable("provider down")

    agent = ChurnAgent(Failing([plan(step("get_distribution", column="Contract"))]))
    response = agent.answer("churn by contract?")
    assert response.error == "llm_unavailable"
    assert response.degraded          # infrastructure, not a grounding failure
    assert not response.fell_back
    assert "Here is what I computed" in response.text


def test_empty_question_short_circuits():
    agent = ChurnAgent(FakeClient([]))
    assert "Ask me something" in agent.answer("   ").text


def test_step_cap_is_enforced():
    from src import config
    agent = ChurnAgent(FakeClient([
        plan(*[step("get_distribution", column="Contract") for _ in range(20)]),
        "Churn is [[F4]].",
    ]))
    response = agent.answer("churn?")
    assert len(response.steps) <= config.MAX_TOOL_STEPS


def test_trace_is_serialisable_for_the_ui():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Churn is [[F4]].",
    ]))
    payload = agent.answer("churn by contract?").to_dict()
    json.dumps(payload)   # must not raise
    assert payload["facts"] and payload["steps"]


# --- model configuration -----------------------------------------------------
def test_env_model_takes_precedence_over_the_builtin_default(monkeypatch):
    """GROQ_MODEL in .env must win. The built-in list is a fallback, not a choice."""
    from src.llm import client as client_mod

    monkeypatch.setenv("GROQ_MODEL", "some/custom-model")
    c = client_mod.LLMClient(api_key="test-key", auto_select_model=False)
    assert c.model == "some/custom-model"
    assert c.model_source == "GROQ_MODEL"


def test_explicit_argument_beats_env(monkeypatch):
    from src.llm import client as client_mod

    monkeypatch.setenv("GROQ_MODEL", "from/env")
    c = client_mod.LLMClient(api_key="test-key", model="from/arg", auto_select_model=False)
    assert c.model == "from/arg" and c.model_source == "argument"


def test_fallback_list_is_env_overridable(monkeypatch):
    from src.llm import client as client_mod

    monkeypatch.setenv("GROQ_MODEL_FALLBACKS", "a/one, b/two")
    assert client_mod._fallback_models() == ["a/one", "b/two"]
    monkeypatch.delenv("GROQ_MODEL_FALLBACKS")
    assert client_mod._fallback_models() == client_mod.BUILTIN_FALLBACKS


# --- issues found by manual testing in the browser ---------------------------
def test_greeting_gets_capabilities_not_an_error():
    """"hi" produced a 400 and surfaced as "the language model is unavailable".

    A greeting is not a data question. The planner returns no steps, and the loop
    answers deterministically -- no second LLM call, no error.
    """
    agent = ChurnAgent(FakeClient([plan()]))
    response = agent.answer("hi")
    assert response.error is None
    assert not response.degraded and not response.fell_back
    assert "churn dataset" in response.text
    assert agent.client.usage.calls == 1


def test_structural_question_is_answered_not_refused():
    """"what is this data about?" replied "I could not compute anything".

    Not every good question produces a number. (get_schema does register facts
    such as row_count; the zero-fact path is covered by the next test.)
    """
    agent = ChurnAgent(FakeClient([
        plan(step("get_schema")),
        "This dataset describes telecom customers and whether they left.",
    ]))
    response = agent.answer("what is this data about?")
    assert "could not compute" not in response.text.lower()
    assert "telecom" in response.text


def test_zero_numeric_facts_still_produces_an_answer():
    """A step can succeed and yield nothing numeric -- e.g. listing column names.

    This is the exact shape that failed in the browser: run_analysis returned a
    non-numeric result, the ledger stayed empty, and the loop refused to answer.
    """
    agent = ChurnAgent(FakeClient([
        plan(step("run_analysis", code="result = ', '.join(df.columns)",
                  purpose="the column names")),
        "The dataset records contract, tenure, charges and service flags per customer.",
    ]))
    response = agent.answer("what columns are in this data?")
    # A string scalar registers nothing: the ledger holds numbers only.
    assert len(response.ledger) == 0
    assert "could not compute" not in response.text.lower()
    assert "contract" in response.text.lower()


def test_structural_answer_still_rejects_invented_numbers():
    """The context block is not a licence to quote figures from it."""
    agent = ChurnAgent(FakeClient([
        plan(step("get_schema")),
        "The dataset covers 9,999 customers across 42 columns.",
        "The dataset describes telecom customers.",
    ]))
    response = agent.answer("what is this data about?")
    assert "9,999" not in response.text


def test_llm_failure_is_flagged_as_degraded_not_as_a_grounding_failure():
    """The UI showed both "model unavailable" and "your answer contained an
    invented figure". Only the first was true."""
    class Failing(FakeClient):
        def complete(self, messages, **kwargs):
            self.usage.calls += 1
            if self.responses:
                return LLMResponse(text=self.responses.pop(0))
            raise LLMUnavailable("provider down")

    agent = ChurnAgent(Failing([plan(step("get_distribution", column="Contract"))]))
    response = agent.answer("churn by contract?")
    assert response.degraded is True
    assert response.fell_back is False


def test_validation_failure_is_flagged_as_fell_back_not_degraded():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Churn is 38.4%.",
        "Churn is 51.2%.",
    ]))
    response = agent.answer("churn by contract?")
    assert response.fell_back is True
    assert response.degraded is False


def test_step_failing_twice_escalates_to_a_replan():
    """A step that fails twice is a planning problem, not bad luck.

    Previously the exhausted step was appended unregistered, leaving the ledger
    empty and producing a generic refusal for an answerable question.
    """
    agent = ChurnAgent(FakeClient([
        plan(step("run_analysis", code="result = df['Nope'].mean()", purpose="bad")),
        plan(step("get_distribution", column="Contract")),
        "Churn is [[F4]].",
    ]))
    response = agent.answer("churn by contract?")
    assert any(s.tool == "get_distribution" and s.status == "ok" for s in response.steps)
    assert response.validation_passed
    assert len(response.ledger) > 0


def test_daily_quota_exhaustion_is_not_retried():
    """A per-minute limit clears in seconds; a per-day cap does not clear today.

    Both arrive as 429. Retrying the daily one spends four backoffs to reach the
    identical error, which across a run of questions looks like a hang rather
    than a quota message.
    """
    from src.llm import client as client_mod

    daily = Exception(
        "Error code: 429 - Rate limit reached for model on tokens per day (TPD): "
        "Limit 200000, Used 199195"
    )
    burst = Exception("Error code: 429 - Rate limit reached on tokens per minute (TPM)")

    assert client_mod._is_daily_quota_exhausted(daily)
    assert not client_mod._is_daily_quota_exhausted(burst)


def test_daily_quota_message_carries_no_provider_detail():
    """The organisation id and token counters belong in the log, not the reply."""
    import re

    from src.agent.loop import _friendly_llm_error
    from src.llm.client import LLMUnavailable

    msg = _friendly_llm_error(LLMUnavailable(
        "Error code: 429 - Rate limit reached ... in organization org_01m0cnrz "
        "on tokens per day (TPD): Limit 200000, Used 199195"
    ))
    assert "org_" not in msg
    assert not re.search(r"\d", msg), "a user-facing error must not contain stray figures"

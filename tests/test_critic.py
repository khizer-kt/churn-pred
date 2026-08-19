"""Critic pass tests.

The critic covers the gap the numeric validator cannot: an answer where every
figure is real and the conclusion drawn from them is not. These tests stub the
LLM, so the loop's branching is exercised deterministically.
"""
from __future__ import annotations

import json

import pytest

from src import config
from src.agent import critic as critic_mod
from src.agent.ledger import FactLedger
from src.agent.loop import ChurnAgent
from src.llm.client import LLMResponse, LLMUnavailable
from tests.test_loop import FakeClient, plan, step

# The pre-filter only reviews answers that interpret their numbers, so a
# fixture has to actually make a claim to reach the critic at all.
CLAIM_ANSWER = ("Female customers churn at 26.9%, slightly higher than the "
                "26.2% seen among male customers.")


def ledger_with_facts() -> FactLedger:
    ledger = FactLedger()
    ledger.add(26.92, "churn rate for gender=Female", unit="percent")
    ledger.add(26.16, "churn rate for gender=Male", unit="percent")
    return ledger


def verdict(v, issues=()):
    return json.dumps({"verdict": v, "issues": list(issues)})


# --- when the critic runs at all --------------------------------------------
def test_skips_when_there_are_no_facts():
    """A refusal or capabilities reply makes no data claim, so there is nothing
    to review and no reason to spend a call."""
    ok, reason = critic_mod.should_review("This dataset has no region column.", FactLedger(), False)
    assert not ok and "no computed facts" in reason


def test_skips_the_deterministic_fallback():
    """The template states computed values and draws no conclusions."""
    ok, reason = critic_mod.should_review("Here is what I computed: ...", ledger_with_facts(), True)
    assert not ok and "fallback" in reason


def test_skips_a_very_short_answer():
    ok, reason = critic_mod.should_review("26.9%.", ledger_with_facts(), False)
    assert not ok and "too short" in reason


def test_runs_on_a_normal_answer():
    ok, _ = critic_mod.should_review(
        "Female customers churn at 26.9% and male customers at 26.2%, a modest difference.",
        ledger_with_facts(), False)
    assert ok


def test_respects_the_config_switch(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CRITIC", False)
    ok, reason = critic_mod.should_review("a" * 100, ledger_with_facts(), False)
    assert not ok and "disabled" in reason


# --- verdict handling --------------------------------------------------------
def test_accepts_a_clean_answer():
    c = critic_mod.review("q", CLAIM_ANSWER,
                          ledger_with_facts(), FakeClient([verdict("ok")]))
    assert c.ok and c.checked and not c.issues


def test_rejects_and_reports_issues():
    c = critic_mod.review("q", CLAIM_ANSWER,
                          ledger_with_facts(),
                          FakeClient([verdict("revise", ["a 0.8 point gap is not meaningful"])]))
    assert not c.ok and c.checked
    assert "0.8 point gap" in c.as_feedback()


def test_revise_without_a_stated_issue_is_treated_as_a_pass():
    """The analyst would have nothing to act on, so a bare rejection must not
    burn a rewrite."""
    c = critic_mod.review("q", CLAIM_ANSWER,
                          ledger_with_facts(), FakeClient([verdict("revise")]))
    assert c.ok and "without naming an issue" in c.skipped_because


# --- failing open ------------------------------------------------------------
def test_llm_failure_does_not_block_the_answer():
    """A broken reviewer must not turn a working system into a broken one."""
    class Failing(FakeClient):
        def complete(self, messages, **kwargs):
            raise LLMUnavailable("provider down")

    c = critic_mod.review("q", CLAIM_ANSWER,
                          ledger_with_facts(), Failing([]))
    assert c.ok and not c.checked and "unavailable" in c.skipped_because


def test_unparseable_critic_output_does_not_block_the_answer():
    c = critic_mod.review("q", CLAIM_ANSWER,
                          ledger_with_facts(), FakeClient(["not json at all"]))
    assert c.ok and not c.checked


# --- loop integration --------------------------------------------------------
def test_rejected_answer_is_rewritten():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="gender")),
        "Women churn far more than men, at [[F4]].",
        verdict("revise", ["the gap is inside noise; say there is no meaningful difference"]),
        "Churn is [[F4]] for one group and effectively the same for the other.",
    ]))
    response = agent.answer("does churn differ by gender?")
    assert response.revised is True
    assert response.critique["checked"] and not response.critique["ok"]
    assert "far more" not in response.text


def test_accepted_answer_is_left_alone():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="gender")),
        "Churn is [[F4]] for that group, with no meaningful difference between them.",
        verdict("ok"),
    ]))
    response = agent.answer("does churn differ by gender?")
    assert response.revised is False
    assert response.critique["ok"] and response.critique["checked"]


def test_a_revision_that_breaks_grounding_is_discarded():
    """The critic reviews interpretation; it cannot be allowed to cost grounding.

    If the rewrite states a figure nobody computed, it fails the numeric gate and
    the original -- whose numbers were verified -- is kept instead.
    """
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="gender")),
        "Churn is [[F4]] for that group.",
        verdict("revise", ["add the comparison"]),
        "Actually women churn at 41.9% and men at 12.3%.",   # ungrounded
        "Still 55.5% versus 44.4%.",                          # ungrounded retry
    ]))
    response = agent.answer("does churn differ by gender?")
    assert response.revised is False, "a rewrite that breaks grounding must not be adopted"
    assert "41.9" not in response.text and "55.5" not in response.text
    assert response.validation_passed


def test_no_critic_call_is_made_for_a_refusal():
    """Refusals have no facts, so the critic must not be invoked at all."""
    agent = ChurnAgent(FakeClient([plan(missing=["region"])]))
    response = agent.answer("does churn correlate with region?")
    assert agent.client.usage.calls == 1
    assert not response.critique.get("checked")


def test_typical_question_costs_three_llm_calls_with_the_critic():
    """Plan, answer, review. The cost of the extra opinion, stated rather than assumed."""
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "Month-to-month churn is [[F4]], the highest of the three contract types.",
        verdict("ok"),
    ]))
    agent.answer("churn by contract?")
    assert agent.client.usage.calls == 3


def test_critique_defaults_to_an_explicit_not_checked_state():
    """An empty dict cannot be told apart from "reviewed and found nothing"."""
    agent = ChurnAgent(FakeClient([plan(missing=["region"])]))
    c = agent.answer("does churn correlate with region?").critique
    assert c["checked"] is False and c["ok"] is True and c["skipped_because"]


def test_a_failing_step_replans_rather_than_repeating_itself():
    """Deterministic tools produce identical failures on identical arguments.

    Generated code using a forbidden import will use it again on a retry, so the
    only useful recovery is to re-plan. This previously burned two calls to reach
    the same error and could leave nothing computed at all.
    """
    agent = ChurnAgent(FakeClient([
        plan(step("run_analysis", code="import numpy\nresult = 1", purpose="bad")),
        plan(step("get_distribution", column="Contract")),
        "Month-to-month churn is [[F4]], the highest of the contract types.",
        json.dumps({"verdict": "ok", "issues": []}),
    ]))
    response = agent.answer("what drives churn?")
    tools_used = [s.tool for s in response.steps]
    assert tools_used.count("run_analysis") == 1, "the failing step must not be repeated"
    assert "get_distribution" in tools_used
    assert response.validation_passed and len(response.ledger) > 0


# --- the deterministic pre-filter -------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Month-to-month customers churn at 42.7%, the highest of the three types.", True),
    ("Women churn more than men, at 26.9% versus 26.2%.", True),
    ("Contract type drives churn.", True),
    ("Churn has increased notably among fiber customers.", True),
    ("The overall churn rate is 26.5% across 7,043 customers.", False),
    ("There are 1,142 senior citizens in the dataset.", False),
    ("Customer 3668-QPYBK has a churn risk of 29.2%, which is band Low.", False),
])
def test_prefilter_targets_claims_not_plain_statements(text, expected):
    """The critic costs a request, and a tier with a daily token cap makes that
    a real budget. A statement of computed values has no interpretation to
    dispute, so it is filtered out deterministically and for free.

    "risk of" is excluded on purpose -- "a churn risk of 29.2%" is the most
    common phrasing in this domain and is not a claim.
    """
    assert critic_mod._makes_a_claim(text) is expected


def test_plain_value_answer_skips_the_critic_entirely():
    agent = ChurnAgent(FakeClient([
        plan(step("get_distribution", column="Contract")),
        "The overall churn rate is [[F10]] across the dataset.",
    ]))
    response = agent.answer("what is the churn rate?")
    assert agent.client.usage.calls == 2, "no critic call for a plain statement"
    assert response.critique["checked"] is False

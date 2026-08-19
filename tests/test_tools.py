"""Agent tool tests -- each tool callable standalone (build step 5)."""
from __future__ import annotations

import pytest

from src.agent import tools


def test_schema_tool_returns_columns_and_absences():
    schema = tools.tool_get_schema()
    assert schema["row_count"] == 7043
    assert "region" in schema["absent_concepts"]


def test_schema_prompt_block_names_missing_columns():
    """The prompt must carry the absences, not just the presences."""
    block = tools.schema_prompt_block()
    assert "DO NOT EXIST" in block
    assert "region" in block
    assert "Month-to-month" in block


def test_absent_concept_detection():
    hits = tools.check_absent_concepts("does churn risk correlate with region?")
    assert hits and hits[0]["concept"] == "region"
    assert not tools.check_absent_concepts("what is the churn rate by contract?")


def test_distribution_for_categorical():
    out = tools.tool_get_distribution("Contract")
    levels = {lvl["value"]: lvl for lvl in out["levels"]}
    assert levels["Month-to-month"]["count"] == 3875
    assert round(levels["Month-to-month"]["churn_rate"], 4) == 0.4271
    assert round(levels["Two year"]["churn_rate"], 4) == 0.0283


def test_distribution_for_numeric():
    out = tools.tool_get_distribution("tenure")
    assert out["summary"]["median"] == 29.0
    assert out["summary"]["max"] == 72.0
    assert out["bins"]
    assert out["mean_by_churn"]["churned"] < out["mean_by_churn"]["retained"]


def test_distribution_rejects_unknown_column():
    out = tools.tool_get_distribution("region")
    assert out["error"] == "unknown_feature"
    assert out["absent_concepts"]


def test_run_analysis_computes_real_numbers():
    out = tools.tool_run_analysis(
        "result = df.groupby('PaymentMethod')['Churn'].mean()",
        purpose="churn rate by payment method",
    )
    assert out["ok"]
    assert round(out["value"]["Electronic check"], 4) == 0.4529


def test_run_analysis_rejects_unsafe_code():
    out = tools.tool_run_analysis("import os\nresult = 1", purpose="escape")
    assert not out["ok"] and out["error"] == "unsafe_code"


def test_dispatch_unknown_tool():
    out = tools.dispatch("nope")
    assert out["error"] == "unknown_tool"
    assert "get_distribution" in out["available_tools"]


def test_dispatch_bad_arguments():
    out = tools.dispatch("get_distribution", {"wrong_arg": 1})
    assert out["error"] == "bad_arguments"


def test_tool_schemas_are_wellformed():
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert names == set(tools.TOOL_FUNCTIONS)
    for schema in tools.TOOL_SCHEMAS:
        assert schema["function"]["description"]
        assert schema["function"]["parameters"]["type"] == "object"

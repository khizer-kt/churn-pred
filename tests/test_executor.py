"""Restricted executor tests -- the boundary against a confused model."""
from __future__ import annotations

import pandas as pd
import pytest

from src.agent.executor import UnsafeCode, run_code, validate_code
from src.data.loader import load_clean


@pytest.fixture(scope="module")
def df():
    return load_clean()


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "from pathlib import Path\nresult = 1",
    "result = eval('1+1')",
    "result = open('/etc/passwd').read()",
    "result = df.__class__.__mro__",
    "result = df.to_csv('/tmp/leak.csv')",
    "result = __import__('os').listdir('.')",
    "while True:\n    pass\nresult = 1",
    "result = getattr(df, 'to_pickle')('/tmp/x')",
])
def test_unsafe_code_is_rejected(code):
    with pytest.raises(UnsafeCode):
        validate_code(code)


def test_code_must_assign_result():
    with pytest.raises(UnsafeCode, match="result"):
        validate_code("df['Churn'].mean()")


def test_scalar_result(df):
    out = run_code("result = float(df['Churn'].mean())", df)
    assert out.ok and out.kind == "scalar"
    assert round(out.value, 4) == 0.2654


def test_dataframe_is_truncated_but_reports_true_size(df):
    out = run_code("result = df[['customerID', 'Churn']]", df)
    assert out.ok and out.kind == "dataframe"
    assert out.rows == 7043          # true count travels with the sample
    assert out.truncated is True
    assert len(out.value) == 50


def test_groupby_series(df):
    out = run_code("result = df.groupby('Contract')['Churn'].mean()", df)
    assert out.ok and out.kind == "series"
    assert round(out.value["Month-to-month"], 4) == 0.4271


def test_execution_error_is_structured_not_raised(df):
    out = run_code("result = df['NoSuchColumn'].mean()", df)
    assert not out.ok
    assert out.error == "execution_error"
    assert "KeyError" in out.detail


def test_unassigned_result_is_an_error(df):
    out = run_code("x = df['Churn'].mean()\nresult = None", df)
    assert not out.ok and out.error == "empty_result"


def test_mutation_does_not_leak_between_calls(df):
    before = len(df)
    run_code("df.drop(df.index, inplace=True)\nresult = len(df)", df)
    assert len(df) == before          # caller's frame is untouched
    out = run_code("result = len(df)", df)
    assert out.value == before


def test_nan_is_serialised_as_none(df):
    out = run_code("result = float('nan')", df)
    assert out.ok and out.value is None

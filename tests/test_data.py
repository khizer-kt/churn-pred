"""Cleaning and loader tests -- the assertions from docs/01-DATA-FINDINGS.md."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data import cleaning
from src.data.loader import (data_quality_audit, describe_cleaning, get_schema,
                             load_clean, load_raw)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def clean():
    return load_clean()


def test_raw_has_disguised_nulls(raw):
    """I1: the blanks are the string ' ', so isnull() sees nothing."""
    assert raw["TotalCharges"].isnull().sum() == 0
    coerced = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    assert coerced.isna().sum() == 11


def test_disguised_nulls_are_all_zero_tenure(raw):
    """I1/I2: justifies filling with 0 rather than the median."""
    coerced = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    assert (raw.loc[coerced.isna(), "tenure"] == 0).all()


def test_shape_and_uniqueness(clean):
    assert len(clean) == config.EXPECTED_ROWS
    assert clean[config.ID_COL].is_unique


def test_total_charges_repaired(clean):
    assert clean["TotalCharges"].dtype.kind == "f"
    assert clean["TotalCharges"].notna().all()
    assert (clean.loc[clean["tenure"] == 0, "TotalCharges"] == 0).all()


def test_senior_citizen_recoded(clean):
    assert set(clean["SeniorCitizen"].unique()) == {"No", "Yes"}


def test_sentinels_collapsed(clean):
    """I4: no column may retain the redundant third level."""
    for col in config.INTERNET_ADDON_COLS + config.PHONE_ADDON_COLS:
        values = set(clean[col].unique())
        assert config.SENTINEL_NO_INTERNET not in values
        assert config.SENTINEL_NO_PHONE not in values
        assert values <= {"Yes", "No", "DSL", "Fiber optic"}


def test_target_encoded(clean):
    assert set(clean[config.TARGET_COL].unique()) == {0, 1}
    assert round(float(clean[config.TARGET_COL].mean()), 4) == 0.2654


def test_derived_features(clean):
    assert "avg_monthly_spend" in clean.columns
    assert clean["avg_monthly_spend"].notna().all()
    # C8: the tenure==0 rows fall back to MonthlyCharges instead of dividing by zero.
    zero = clean[clean["tenure"] == 0]
    assert (zero["avg_monthly_spend"] == zero["MonthlyCharges"]).all()
    assert "tenure_bucket" in clean.columns


def test_known_segment_rates(clean):
    """Ground truth from docs/01-DATA-FINDINGS.md section 5."""
    rates = clean.groupby("Contract", observed=True)[config.TARGET_COL].mean() * 100
    assert round(float(rates["Month-to-month"]), 2) == 42.71
    assert round(float(rates["Two year"]), 2) == 2.83


def test_audit_records_but_does_not_remove(clean):
    """C7/I7/I8: both are logged and deliberately left in place."""
    audit = data_quality_audit()
    assert audit["duplicate_rows_ignoring_id"] == 22
    assert audit["contradictory_label_rows"] == 42
    assert len(clean) == config.EXPECTED_ROWS  # nothing was dropped


def test_cleaning_report_covers_every_step():
    codes = {step["code"] for step in describe_cleaning()}
    assert {"C1", "C2", "C3", "C5", "C7", "C8", "C9"} <= codes


def test_zero_fill_guard_trips_on_bad_input():
    """C1 must refuse to zero-fill if the tenure==0 relationship ever breaks."""
    df = load_raw().copy()
    df.loc[df.index[0], "TotalCharges"] = " "   # a blank on a long-tenure customer
    with pytest.raises(ValueError, match="zero-fill assumption"):
        cleaning.fix_total_charges(df)


def test_schema_declares_absent_concepts():
    """I10: the agent must be able to refuse 'region' by name."""
    schema = get_schema()
    names = {c["name"] for c in schema["columns"]}
    assert "region" not in {n.lower() for n in names}
    assert "region" in schema["absent_concepts"]
    assert "revenue trend" in schema["absent_concepts"]


def test_schema_lists_allowed_values():
    schema = get_schema()
    contract = next(c for c in schema["columns"] if c["name"] == "Contract")
    assert set(contract["allowed_values"]) == {"Month-to-month", "One year", "Two year"}


def test_id_column_is_not_a_model_feature():
    assert config.ID_COL not in config.MODEL_FEATURES
    assert "tenure_bucket" not in config.MODEL_FEATURES

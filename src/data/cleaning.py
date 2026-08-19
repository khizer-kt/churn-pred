"""Cleaning steps C1-C9 from docs/01-DATA-FINDINGS.md section 4.

Each fix is a separate function so it can be tested and explained in isolation.
Every function takes and returns a DataFrame and does not mutate its input.

The `CLEANING_LOG` records what each step actually changed, so the app and the
README can report the cleaning work rather than merely assert it happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import config


@dataclass
class CleaningReport:
    """What each cleaning step did, for display in the UI and README."""

    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, description: str, **details: Any) -> None:
        self.steps.append({"code": code, "description": description, **details})

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.steps)


# ---------------------------------------------------------------------------
# C1 -- TotalCharges: disguised nulls (finding I1)
# ---------------------------------------------------------------------------
def fix_total_charges(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Coerce TotalCharges to float and fill the 11 blanks with 0.0.

    The column arrives as text containing a single space ' ' for 11 rows, so
    `isnull()` reports zero missing and `astype(float)` raises. Every one of
    those rows has tenure == 0: the customer has not been billed a cycle yet,
    so the true value is 0, not the median. Median imputation here would invent
    a billing history that never happened.
    """
    df = df.copy()
    before_dtype = str(df["TotalCharges"].dtype)
    coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(coerced.isna().sum())

    # Guard: the zero-fill is only correct because these are unbilled customers.
    non_zero_tenure = int((coerced.isna() & (df["tenure"] != 0)).sum())
    if non_zero_tenure:
        raise ValueError(
            f"{non_zero_tenure} rows have missing TotalCharges but non-zero tenure. "
            "The zero-fill assumption in C1 no longer holds -- re-check the input file."
        )

    df["TotalCharges"] = coerced.fillna(0.0).astype(float)
    if report:
        report.add(
            "C1",
            "Converted TotalCharges from text to float; filled disguised nulls with 0.0",
            original_dtype=before_dtype,
            values_repaired=n_missing,
            detail="Blanks were the literal string ' ', invisible to isnull(). "
                   "All correspond to tenure == 0 (never billed).",
        )
    return df


# ---------------------------------------------------------------------------
# C2 -- SeniorCitizen: encoding consistency (finding I3)
# ---------------------------------------------------------------------------
def normalise_senior_citizen(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Map SeniorCitizen 0/1 to No/Yes to match every other binary flag.

    Two reasons this matters: pandas infers int64 and the column silently lands
    in the numeric branch of the ColumnTransformer, and the agent has no way to
    know that 1 means senior unless the schema says so in words.
    """
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    if report:
        report.add(
            "C2",
            "Recoded SeniorCitizen from 0/1 to No/Yes",
            detail="Aligns it with Partner, Dependents, PhoneService and PaperlessBilling, "
                   "and makes the schema self-describing for the agent.",
        )
    return df


# ---------------------------------------------------------------------------
# C3 -- Redundant sentinel levels (finding I4)
# ---------------------------------------------------------------------------
def collapse_service_sentinels(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Replace 'No internet service' / 'No phone service' with plain 'No'.

    Those levels are 100% determined by InternetService == 'No' and
    PhoneService == 'No' respectively (verified: zero violations in either
    direction). Left in place, one-hot encoding produces seven dummy columns
    that are exact duplicates of each other, which destabilises the logistic
    regression coefficients that top_factors depends on.

    No information is lost -- it survives in InternetService / PhoneService.
    """
    df = df.copy()
    replaced: dict[str, int] = {}

    for col in config.INTERNET_ADDON_COLS:
        n = int((df[col] == config.SENTINEL_NO_INTERNET).sum())
        if n:
            df[col] = df[col].replace(config.SENTINEL_NO_INTERNET, "No")
            replaced[col] = n

    for col in config.PHONE_ADDON_COLS:
        n = int((df[col] == config.SENTINEL_NO_PHONE).sum())
        if n:
            df[col] = df[col].replace(config.SENTINEL_NO_PHONE, "No")
            replaced[col] = n

    if report:
        report.add(
            "C3",
            "Collapsed redundant 'No internet service' / 'No phone service' levels to 'No'",
            columns_affected=replaced,
            detail="These levels were perfectly collinear with InternetService/PhoneService "
                   "(0 violations), producing duplicate one-hot columns.",
        )
    return df


# ---------------------------------------------------------------------------
# C5 -- Target encoding
# ---------------------------------------------------------------------------
def encode_target(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Map Churn Yes/No to 1/0, keeping the original label for display."""
    df = df.copy()
    if df[config.TARGET_COL].dtype == object:
        df["Churn_label"] = df[config.TARGET_COL]
        df[config.TARGET_COL] = df[config.TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)
    if report:
        churn_rate = float(df[config.TARGET_COL].mean())
        report.add(
            "C5",
            "Encoded Churn as 1/0, retaining Churn_label for display",
            churn_rate=round(churn_rate, 4),
        )
    return df


# ---------------------------------------------------------------------------
# C8 -- Engineered: avg_monthly_spend
# ---------------------------------------------------------------------------
def add_avg_monthly_spend(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """TotalCharges / tenure, falling back to MonthlyCharges when tenure == 0.

    Separates the historical average rate from the current rate. TotalCharges is
    only approximately tenure * MonthlyCharges (median deviation 0.000 but p95
    +7.5%, max +57%) because MonthlyCharges is the *current* price while
    TotalCharges accumulates historical prices across plan changes. The gap
    between the two is real signal.

    The tenure == 0 fallback guards the divide-by-zero from finding I2.
    """
    df = df.copy()
    tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["TotalCharges"] / tenure).fillna(df["MonthlyCharges"]).astype(float)
    if report:
        report.add(
            "C8",
            "Added avg_monthly_spend = TotalCharges / tenure (falls back to MonthlyCharges at tenure 0)",
            zero_tenure_rows=int((df["tenure"] == 0).sum()),
        )
    return df


# ---------------------------------------------------------------------------
# C9 -- Engineered: tenure_bucket (EDA only, never a model feature)
# ---------------------------------------------------------------------------
def add_tenure_bucket(df: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    """Human-legible tenure cohorts for EDA and agent segment queries."""
    df = df.copy()
    edges = [b[0] for b in config.TENURE_BUCKETS] + [config.TENURE_BUCKETS[-1][1]]
    labels = [b[2] for b in config.TENURE_BUCKETS]
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=edges, labels=labels, right=False, include_lowest=True
    ).astype(str)
    if report:
        report.add(
            "C9",
            "Added tenure_bucket cohorts for EDA and segment queries",
            buckets=labels,
            detail="Excluded from the model -- it is a coarsened copy of tenure.",
        )
    return df


# ---------------------------------------------------------------------------
# C7 -- Duplicates and contradictory labels: LOG ONLY, deliberately no action
# ---------------------------------------------------------------------------
def audit_duplicates_and_label_noise(
    df: pd.DataFrame, report: CleaningReport | None = None
) -> dict[str, int]:
    """Count near-duplicates and contradictory labels without changing anything.

    Both are real signal, not corruption, so neither is removed:

    * Rows identical once customerID is ignored are expected collisions among
      short-tenure customers on minimal plans -- every one has a valid distinct
      ID and represents a real customer.
    * Rows identical across all features but disagreeing on Churn are
      irreducible (Bayes) error: two customers with identical observable
      attributes genuinely made different decisions. Dropping them would teach
      the model a certainty the data does not support, and would inflate the
      apparent score.
    """
    feature_cols = [c for c in df.columns if c not in (config.ID_COL, config.TARGET_COL, "Churn_label")]

    dup_ignoring_id = int(df.drop(columns=[config.ID_COL]).duplicated().sum())

    grouped = df.groupby(feature_cols, dropna=False, observed=True)[config.TARGET_COL].nunique()
    contradictory_groups = int((grouped > 1).sum())
    sizes = df.groupby(feature_cols, dropna=False, observed=True)[config.TARGET_COL].size()
    contradictory_rows = int(sizes[grouped > 1].sum())

    stats = {
        "duplicate_rows_ignoring_id": dup_ignoring_id,
        "contradictory_label_groups": contradictory_groups,
        "contradictory_label_rows": contradictory_rows,
    }
    if report:
        report.add(
            "C7",
            "Audited duplicates and contradictory labels -- no rows removed (deliberate)",
            **stats,
            detail="Near-duplicates are real distinct customers; contradictory labels are "
                   "irreducible error that caps achievable accuracy. Removing either would "
                   "inflate the score without improving the model.",
        )
    return stats

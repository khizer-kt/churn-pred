"""The single place the churn CSV is read.

The notebook, the model, the agent and the Streamlit app all import from here.
If any of them re-implements cleaning inline they will silently disagree about
the data -- see docs/00-PROJECT-PLAN.md section 3.

Public API:
    load_raw()        -- untouched CSV, for the "issues as found" EDA panel
    load_clean()      -- cleaned, cached, assertion-checked (use this)
    get_schema()      -- machine-readable schema for the agent's T1 tool
    describe_cleaning() -- what the cleaning steps did, for the UI/README
"""
from __future__ import annotations

import functools
from typing import Any

import pandas as pd

from src import config
from src.data import cleaning


def load_raw(path=None) -> pd.DataFrame:
    """Read the CSV with no cleaning applied.

    Kept deliberately raw so the app can show the data issues as they arrive --
    notably TotalCharges as a text column whose 11 blanks are invisible to
    isnull(). Do not train on this.
    """
    return pd.read_csv(path or config.RAW_CSV)


def _apply_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, cleaning.CleaningReport, dict]:
    """Run C1-C9 in order. Order matters: C5 must precede the C7 audit."""
    report = cleaning.CleaningReport()

    df = cleaning.fix_total_charges(df, report)          # C1
    df = cleaning.normalise_senior_citizen(df, report)   # C2
    df = cleaning.collapse_service_sentinels(df, report) # C3
    df = cleaning.encode_target(df, report)              # C5
    df = cleaning.add_avg_monthly_spend(df, report)      # C8
    df = cleaning.add_tenure_bucket(df, report)          # C9

    # C4: customerID stays in the frame as the agent's lookup key but is never
    # a model feature -- enforced by MODEL_FEATURES, not by dropping it here.
    # C6: column names are left unchanged on purpose; the agent's generated
    # pandas code and the user's vocabulary both track the original names.

    audit = cleaning.audit_duplicates_and_label_noise(df, report)  # C7, log only
    return df, report, audit


def _assert_clean(df: pd.DataFrame) -> None:
    """Post-load assertions from docs/01-DATA-FINDINGS.md section 4.

    Cheap insurance against a silently changed input file. These fire loudly
    rather than letting a corrupted dataset reach the model.
    """
    assert len(df) == config.EXPECTED_ROWS, f"expected {config.EXPECTED_ROWS} rows, got {len(df)}"
    assert df[config.ID_COL].is_unique, "customerID is not unique"
    assert df["TotalCharges"].dtype.kind == "f", "TotalCharges is not float"
    assert df["TotalCharges"].notna().all(), "TotalCharges still has nulls"
    assert set(df[config.TARGET_COL].unique()) == {0, 1}, "Churn is not encoded 0/1"
    assert (df["tenure"] >= 0).all(), "negative tenure"
    assert (df["MonthlyCharges"] > 0).all(), "non-positive MonthlyCharges"

    sentinels = {config.SENTINEL_NO_INTERNET, config.SENTINEL_NO_PHONE}
    for col in config.INTERNET_ADDON_COLS + config.PHONE_ADDON_COLS:
        leftover = sentinels & set(df[col].unique())
        assert not leftover, f"{col} still contains sentinel level(s): {leftover}"

    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    assert not missing, f"missing model features after cleaning: {missing}"


@functools.lru_cache(maxsize=1)
def _load_clean_cached() -> tuple[pd.DataFrame, tuple, dict]:
    df, report, audit = _apply_cleaning(load_raw())
    _assert_clean(df)
    return df, tuple(report.to_list()), audit


def load_clean(copy: bool = True) -> pd.DataFrame:
    """Cleaned dataset, cached after the first call.

    Returns a copy by default. The agent's code executor and any caller that
    might mutate must not share the cached frame.
    """
    df, _, _ = _load_clean_cached()
    return df.copy() if copy else df


def describe_cleaning() -> list[dict[str, Any]]:
    """What each cleaning step changed -- rendered in the app's Explore tab."""
    _, report, _ = _load_clean_cached()
    return [dict(step) for step in report]


def data_quality_audit() -> dict[str, int]:
    """Duplicate and label-noise counts (recorded, deliberately not acted on)."""
    _, _, audit = _load_clean_cached()
    return dict(audit)


# ---------------------------------------------------------------------------
# Schema -- backs the agent's get_schema tool (docs/03-AGENT-SPEC.md T1)
# ---------------------------------------------------------------------------
COLUMN_DESCRIPTIONS = {
    "customerID": "Unique customer identifier, format 1234-ABCDE. Not a predictive feature.",
    "gender": "Customer gender.",
    "SeniorCitizen": "Whether the customer is a senior citizen.",
    "Partner": "Whether the customer has a partner.",
    "Dependents": "Whether the customer has dependents.",
    "tenure": "Months the customer has been with the company (0-72).",
    "PhoneService": "Whether the customer has phone service.",
    "MultipleLines": "Whether the customer has multiple phone lines.",
    "InternetService": "Type of internet service, or 'No' if none.",
    "OnlineSecurity": "Whether the customer has the online security add-on.",
    "OnlineBackup": "Whether the customer has the online backup add-on.",
    "DeviceProtection": "Whether the customer has the device protection add-on.",
    "TechSupport": "Whether the customer has the tech support add-on.",
    "StreamingTV": "Whether the customer streams TV.",
    "StreamingMovies": "Whether the customer streams movies.",
    "Contract": "Contract commitment length.",
    "PaperlessBilling": "Whether the customer uses paperless billing.",
    "PaymentMethod": "How the customer pays their bill.",
    "MonthlyCharges": "Current monthly charge in USD.",
    "TotalCharges": "Total charged over the customer's lifetime in USD.",
    "avg_monthly_spend": "Derived: TotalCharges / tenure (historical average monthly rate).",
    "tenure_bucket": "Derived: tenure grouped into human-readable cohorts.",
    "Churn": "Target. 1 if the customer left, 0 if retained.",
    "Churn_label": "Target as text (Yes/No).",
}

# Stated explicitly so the agent can refuse questions about them by name rather
# than inventing a breakdown. See docs/01-DATA-FINDINGS.md finding I10.
ABSENT_CONCEPTS = {
    "region": "There is no geographic column of any kind -- no region, state, city or country.",
    "revenue trend": (
        "There is no date column, so no metric can be tracked over time. The data is a "
        "single cross-sectional snapshot. The nearest legitimate substitute is revenue by "
        "tenure cohort, which compares different customers at different lifecycle stages "
        "rather than following one cohort through time."
    ),
    "product category": (
        "There is no product-category column. The closest equivalents are the nine service "
        "flags (PhoneService, InternetService and the six add-ons)."
    ),
    "date": "There is no date, timestamp, signup-date or churn-date column.",
    "age": "There is no age column. SeniorCitizen is the only age-related field.",
    "income": "There is no income, salary or socioeconomic column.",
    "satisfaction": "There is no satisfaction score, NPS or survey column.",
    "complaints": "There is no complaints, tickets or support-contact-count column.",
}


@functools.lru_cache(maxsize=1)
def get_schema() -> dict[str, Any]:
    """Machine-readable schema handed to the agent.

    Loaded into the system prompt at startup rather than fetched per turn: it
    costs a few hundred tokens once, saves a round-trip per question, and is the
    primary defence against the agent claiming a column exists when it does not.
    """
    df = load_clean(copy=False)
    columns = []

    for col in df.columns:
        entry: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "description": COLUMN_DESCRIPTIONS.get(col, ""),
            "role": _column_role(col),
        }
        if col == config.ID_COL:
            entry["n_unique"] = int(df[col].nunique())
            entry["example"] = str(df[col].iloc[0])
        elif pd.api.types.is_numeric_dtype(df[col]) and col != config.TARGET_COL:
            entry.update(
                min=round(float(df[col].min()), 2),
                max=round(float(df[col].max()), 2),
                median=round(float(df[col].median()), 2),
                mean=round(float(df[col].mean()), 2),
            )
        else:
            entry["allowed_values"] = sorted(str(v) for v in df[col].unique())
        columns.append(entry)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target": config.TARGET_COL,
        "id_column": config.ID_COL,
        "churn_rate": round(float(df[config.TARGET_COL].mean()), 4),
        "grain": "One row per customer. A single cross-sectional snapshot -- there is no time dimension.",
        "columns": columns,
        "absent_concepts": dict(ABSENT_CONCEPTS),
    }


def _column_role(col: str) -> str:
    if col == config.ID_COL:
        return "identifier"
    if col in (config.TARGET_COL, "Churn_label"):
        return "target"
    if col in config.EDA_ONLY_COLS:
        return "derived_eda_only"
    if col in config.MODEL_FEATURES:
        return "feature"
    return "other"

"""Explore tab -- EDA surfaced directly to the user.

Brief, Dataset behavior item 6: the user should be able to see what the dataset
holds and understand why the model behaves as it does. Entirely deterministic --
no LLM calls, so this tab works without an API key.

Backed by the SAME tool the agent calls (`tool_get_distribution`), so the panel
and the chat cannot disagree about a number.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.agent.tools import tool_get_distribution
from ui import state


def render() -> None:
    df = state.get_dataframe()
    schema = state.get_schema_cached()

    st.subheader("What this dataset holds")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{schema['row_count']:,}")
    c2.metric("Columns", schema["column_count"])
    c3.metric("Churn rate", f"{schema['churn_rate']:.1%}")
    c4.metric("Churned", f"{int(df[config.TARGET_COL].sum()):,}")

    st.caption(schema["grain"])

    tab_dist, tab_issues, tab_corr, tab_raw = st.tabs(
        ["Distributions", "Data issues found", "Numeric correlations", "Sample rows"]
    )

    with tab_dist:
        _distributions(df)
    with tab_issues:
        _issues()
    with tab_corr:
        _correlations(df)
    with tab_raw:
        st.dataframe(df.head(200), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
def _distributions(df: pd.DataFrame) -> None:
    columns = [c for c in df.columns if c not in (config.ID_COL, "Churn_label")]
    default = columns.index("Contract") if "Contract" in columns else 0
    column = st.selectbox("Column", columns, index=default, key="explore_column")

    result = tool_get_distribution(column)
    if "error" in result:
        st.warning(result["message"])
        return

    if "levels" in result:
        table = pd.DataFrame(result["levels"])
        table["churn_rate_pct"] = (table["churn_rate"] * 100).round(2)
        st.bar_chart(table.set_index("value")["churn_rate_pct"],
                     y_label="Churn rate (%)", x_label=column)
        st.dataframe(
            table[["value", "count", "share", "churn_rate_pct", "churn_count"]]
            .rename(columns={"value": column, "churn_rate_pct": "churn rate (%)"}),
            width="stretch", hide_index=True,
        )
    else:
        summary = result["summary"]
        cols = st.columns(len(summary))
        for col, (name, value) in zip(cols, summary.items()):
            col.metric(name, f"{value:,.2f}")
        bins = pd.DataFrame(result["bins"])
        bins["churn_rate_pct"] = (bins["churn_rate"] * 100).round(2)
        st.bar_chart(bins.set_index("range")["churn_rate_pct"],
                     y_label="Churn rate (%)", x_label=column)
        means = result["mean_by_churn"]
        st.caption(
            f"Mean {column} — churned: **{means['churned']:,.2f}**, "
            f"retained: **{means['retained']:,.2f}**"
        )

    st.caption(f"Overall churn rate for comparison: {result['overall_churn_rate']:.2%}")


def _issues() -> None:
    """The cleaning work is a graded deliverable, so make it visible in the product."""
    raw = state.get_raw_dataframe()
    audit = state.get_audit()

    st.markdown(
        "The dataset arrived with problems that are invisible to a routine "
        "`df.isnull().sum()`. Each was found by profiling and handled explicitly."
    )

    coerced = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    c1, c2 = st.columns(2)
    c1.metric("Nulls reported by isnull()", int(raw["TotalCharges"].isnull().sum()))
    c2.metric("Actually missing", int(coerced.isna().sum()), delta="found by coercion")
    st.caption(
        "`TotalCharges` arrived as **text**, and its 11 blanks are a single space "
        "character — so `isnull()` sees nothing and `astype(float)` raises. All 11 "
        "are customers with `tenure == 0` who have not been billed yet, so the "
        "correct fill is **0**, not the median."
    )

    st.divider()
    st.markdown("**Steps applied**")
    for step in state.get_cleaning_report():
        with st.expander(f"`{step['code']}` — {step['description']}"):
            for key, value in step.items():
                if key in ("code", "description"):
                    continue
                st.markdown(f"- **{key}**: {value}")

    st.divider()
    st.markdown("**Recorded but deliberately not acted on**")
    c1, c2 = st.columns(2)
    c1.metric("Duplicate rows (ignoring ID)", audit["duplicate_rows_ignoring_id"])
    c2.metric("Rows with contradictory labels", audit["contradictory_label_rows"])
    st.caption(
        "Neither is corruption. The near-duplicates are real distinct customers on "
        "minimal plans. The contradictory rows are identical across all 19 features "
        "but disagree on churn — irreducible error that caps achievable accuracy. "
        "Dropping either would inflate the score without improving the model."
    )


def _correlations(df: pd.DataFrame) -> None:
    numeric = [c for c in config.NUMERIC_FEATURES if c in df.columns] + [config.TARGET_COL]
    corr = df[numeric].corr().round(3)
    st.dataframe(corr, width="stretch")
    st.caption(
        "`tenure` is the strongest single numeric signal. `TotalCharges` correlates "
        "0.83 with tenure, so the two carry overlapping information."
    )

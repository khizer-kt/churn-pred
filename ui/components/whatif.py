"""Score a customer / what-if tab.

Covers two brief requirements directly: assessing individual data points and
submitting hypothetical ones, and projecting risk forward under specific
feature conditions.

No LLM calls -- this tab works without an API key. Every categorical input is
populated from the schema's allowed values, so an invalid value cannot be
entered in the first place.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.model import service
from ui import state


def render() -> None:
    if not state.model_ready():
        st.error(state.model_problem() or "The churn model is unavailable.")
        return

    mode = st.radio(
        "Who are we scoring?",
        ["An existing customer", "A hypothetical new customer"],
        horizontal=True, key="whatif_mode",
    )
    if mode == "An existing customer":
        _existing()
    else:
        _hypothetical()


# ---------------------------------------------------------------------------
def _existing() -> None:
    ids = state.get_customer_ids()
    customer_id = st.selectbox("Customer ID", ids, index=0, key="whatif_customer")

    overrides = _override_form(key_prefix="existing")
    result = service.predict_churn_risk(customer_id, overrides=overrides or None)

    if "error" in result:
        st.error(result["message"])
        if result.get("did_you_mean"):
            st.caption("Did you mean: " + ", ".join(result["did_you_mean"]))
        return

    _show_result(result)


def _hypothetical() -> None:
    df = state.get_dataframe()
    schema = {c["name"]: c for c in state.get_schema_cached()["columns"]}

    st.caption("Defaults are the dataset's median (numeric) or most common value (categorical).")
    features: dict = {}
    columns = st.columns(3)

    for i, name in enumerate(config.MODEL_FEATURES):
        if name == "avg_monthly_spend":
            continue   # derived from the others; computed by the service
        target = columns[i % 3]
        meta = schema.get(name, {})
        if name in config.NUMERIC_FEATURES:
            features[name] = target.number_input(
                name,
                min_value=float(meta.get("min", 0.0)),
                max_value=float(meta.get("max", 10_000.0)),
                value=float(meta.get("median", 0.0)),
                key=f"hyp_{name}",
            )
        else:
            allowed = meta.get("allowed_values") or sorted(df[name].astype(str).unique())
            mode_value = str(df[name].mode().iloc[0])
            features[name] = target.selectbox(
                name, allowed,
                index=allowed.index(mode_value) if mode_value in allowed else 0,
                key=f"hyp_{name}",
            )

    # The result is kept in session state rather than rendered only on the run
    # where the button was pressed. Otherwise any later interaction -- switching
    # a dropdown, opening an expander -- reruns the script and the score silently
    # disappears, which reads as the app losing your work.
    if st.button("Score this customer", type="primary", key="hyp_score"):
        st.session_state.hypothetical_result = service.predict_churn_risk(features=features)

    result = st.session_state.get("hypothetical_result")
    if not result:
        return
    if "error" in result:
        st.error(result["message"])
        if result.get("missing_features"):
            st.caption("Missing: " + ", ".join(result["missing_features"]))
        return
    _show_result(result)


# ---------------------------------------------------------------------------
def _override_form(key_prefix: str) -> dict:
    """Change specific conditions and see the risk move.

    This is the brief's "projected forward under specific feature conditions",
    and the most demo-friendly screen in the app.
    """
    df = state.get_dataframe()
    with st.expander("Project forward: change some conditions", expanded=False):
        st.caption("Leave a field as *(unchanged)* to keep the customer's real value.")
        overrides: dict = {}
        columns = st.columns(3)

        for i, name in enumerate(["Contract", "InternetService", "PaymentMethod",
                                  "TechSupport", "OnlineSecurity", "PaperlessBilling"]):
            options = ["(unchanged)"] + sorted(df[name].astype(str).unique())
            choice = columns[i % 3].selectbox(name, options, key=f"{key_prefix}_ov_{name}")
            if choice != "(unchanged)":
                overrides[name] = choice

        tenure = st.slider(
            "tenure (months) — leave at -1 to keep the real value",
            min_value=-1, max_value=72, value=-1, key=f"{key_prefix}_ov_tenure",
        )
        if tenure >= 0:
            overrides["tenure"] = tenure
        return overrides


def _show_result(result: dict) -> None:
    score = result["risk_score"]
    band = result["risk_band"]
    colour = {"Low": "normal", "Medium": "off", "High": "inverse"}.get(band, "normal")

    c1, c2, c3 = st.columns(3)
    if "baseline_risk_score" in result:
        c1.metric(
            "Churn risk", f"{score:.1%}",
            delta=f"{result['risk_delta'] * 100:+.1f} pp vs today",
            delta_color=colour,
        )
        c2.metric("Risk today", f"{result['baseline_risk_score']:.1%}")
    else:
        c1.metric("Churn risk", f"{score:.1%}")
        c2.metric("Population average", f"{result['population_base_rate']:.1%}")
    c3.metric("Risk band", band, help=f"Percentile {result['percentile']:.0f} of all customers")

    st.progress(min(max(score, 0.0), 1.0))
    flagged = "flagged for retention" if result["predicted_churn"] else "not flagged"
    st.caption(
        f"Decision threshold is {result['threshold']:.0%} — this customer is **{flagged}**. "
        "The threshold is chosen by expected cost, not set at 50%."
    )

    if result.get("actual_churn") is not None:
        actual = "did churn" if result["actual_churn"] else "did not churn"
        st.caption(f"Recorded outcome in the dataset: this customer **{actual}**.")

    if result.get("top_factors"):
        st.markdown("**What is driving this score**")
        table = pd.DataFrame(result["top_factors"])
        # `value` legitimately mixes types across factors -- "No" for a service
        # flag, 29.85 for a charge. Arrow infers a single column type and fails
        # on the mix, so render it as text.
        table["value"] = table["value"].astype(str)
        table["contribution"] = table["contribution"].astype(float).round(3)
        st.dataframe(
            table[["feature", "value", "direction", "contribution"]],
            width="stretch", hide_index=True,
            column_config={
                "contribution": st.column_config.NumberColumn(
                    "contribution (log-odds)", format="%+.3f"
                )
            },
        )
        for factor in result["top_factors"]:
            if factor.get("note"):
                st.caption(f"**{factor['feature']}** — {factor['note']}")
        st.caption(
            "Contributions are exact: they are this customer's log-odds "
            "decomposition and sum to the predicted score, rather than an "
            "approximation of it."
        )

    for warning in result.get("warnings", []):
        st.warning(warning)

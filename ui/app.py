"""Streamlit entrypoint. Deliberately thin -- routing and chrome only.

Run:  streamlit run ui/app.py

Rule from docs/04-APP-AND-DEPLOY-SPEC.md section 1: ui/ imports from src/, never
the reverse. This file computes nothing; it asks the agent or the services.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Make the project importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import state                                    # noqa: E402
from ui.components import chat, explore, whatif         # noqa: E402

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Churn Analyst Agent",
    page_icon=":material/query_stats:",
    layout="wide",
)


def main() -> None:
    state.init()

    st.title("Churn Analyst Agent")
    st.caption(
        "Ask questions in plain English. The agent plans, runs real computations "
        "against the data, checks its own results, and refuses to state a number "
        "it did not compute."
    )

    _sidebar()

    tab_chat, tab_explore, tab_score = st.tabs(
        ["Chat", "Explore the data", "Score a customer"]
    )
    with tab_chat:
        chat.render()
    with tab_explore:
        explore.render()
    with tab_score:
        whatif.render()


def _sidebar() -> None:
    with st.sidebar:
        st.subheader("Status")

        agent = state.get_agent()
        if agent.available:
            st.success(f"Model: `{agent.client.model}`", icon=":material/check:")
            st.caption(f"Source: {getattr(agent.client, 'model_source', 'unknown')}")
        else:
            st.error("No language model", icon=":material/key_off:")
            st.caption(agent.client.reason)

        info = state.get_model_info()
        if info:
            st.success(f"Churn model: `{info['chosen_model']}`", icon=":material/check:")
            test = info.get("test_metrics", {})
            if test:
                c1, c2 = st.columns(2)
                c1.metric("PR-AUC", f"{test.get('pr_auc', 0):.3f}")
                c2.metric("Recall", f"{test.get('recall', 0):.2f}")
                st.caption(
                    f"Threshold {info['threshold']:.2f}, chosen by expected cost rather "
                    f"than set at 0.5. Primary metric is PR-AUC; a no-skill model scores "
                    f"{info['base_rate']:.3f}."
                )
        else:
            st.error("Churn model unavailable", icon=":material/error:")
            # Show what actually went wrong. "Run the trainer" is only the right
            # advice when the artifact is missing; a load failure needs a
            # different fix and deserves to say so.
            problem = state.model_problem()
            if problem:
                st.caption(problem)
            if "not found" in problem.lower():
                st.code("python -m src.model.train", language="bash")

        usage = agent.client.usage.to_dict()
        if usage.get("calls"):
            st.divider()
            st.subheader("Session cost")
            c1, c2 = st.columns(2)
            c1.metric("LLM calls", usage["calls"])
            c2.metric("Tokens", f"{usage['total_tokens']:,}")
            if usage.get("rate_limited"):
                st.caption(f"Rate limited {usage['rate_limited']}x — handled by backoff.")

        st.divider()
        if st.button("Clear conversation", width="stretch"):
            state.clear()
            st.rerun()

        st.divider()
        st.caption(
            "Built for the Adept Tech Solutions AI Engineer assessment. "
            "Every figure in an answer is traceable to a computed value — "
            "open **Facts used** under any reply to check."
        )


main()

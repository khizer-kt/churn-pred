"""Chat tab -- the Stage 2 deliverable, live-wired to the Stage 3 agent.

Every answer carries two expanders: the step trace ("How I got this") and the
Fact Ledger ("Facts used"). Those are not decoration -- they are the evidence
for the brief's central claim that no number is invented, and they are the first
thing a reviewer will open.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import state

EXAMPLES = [
    ("Which customers are most likely to churn, and does that relate to contract type?",
     "multi-step: model + aggregation"),
    ("What happens to customer 3668-QPYBK if they switch to a two year contract?",
     "what-if projection"),
    ("How many senior citizens are there and do they churn more?",
     "EDA-style question"),
    ("Does churn risk correlate with region?",
     "there is no region column — watch it refuse"),
]


def render() -> None:
    agent = state.get_agent()

    if not agent.available:
        st.warning(agent.client.reason, icon=":material/key_off:")
        st.info(
            "The **Explore** and **Score a customer** tabs need no API key and are "
            "fully functional.", icon=":material/info:",
        )
        return

    if not state.model_ready():
        st.error(state.model_problem() or "The churn model is unavailable.")
        return

    # Render is a pure function of session state. Nothing below both mutates
    # history and draws in the same pass -- doing that is what desynchronised the
    # page: the examples were drawn while `messages` was still empty, then the
    # same run appended to `messages`, so on the next run the buttons were not
    # recreated and Streamlit had nowhere to deliver the click.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("response"):
                _evidence(message["response"])

    pending = st.session_state.pending

    # Examples stay reachable for the whole session -- prominent when idle, tucked
    # into an expander once there is a conversation. Hidden only while a question
    # is in flight, so the queue cannot be written twice in one pass.
    if not pending:
        if st.session_state.messages:
            with st.expander("Try another example"):
                _examples()
        else:
            _empty_state()

    typed = st.chat_input("Ask about the dataset, a customer, or a segment...")
    if typed:
        st.session_state.pending = typed
        st.rerun()

    if pending:
        _ask(agent, pending)


def _queue(question: str) -> None:
    """Queue an example question.

    An on_click callback rather than a check on st.button's return value:
    Streamlit runs callbacks *before* the script body, so the click is delivered
    even on a run where the button will not be recreated. Reading the return
    value drops the click in exactly that case.
    """
    st.session_state.pending = question


# ---------------------------------------------------------------------------
def _empty_state() -> None:
    st.markdown("#### Try one of these")
    _examples()
    st.divider()


def _examples() -> None:
    """The example buttons.

    Rendered from one place so the idle grid and the in-conversation expander
    cannot drift apart. Keys are stable across both, which is what lets Streamlit
    match a click to the right widget.
    """
    for i, (text, why) in enumerate(EXAMPLES):
        col_button, col_note = st.columns([3, 2])
        col_button.button(text, key=f"eg_{i}", width="stretch",
                          on_click=_queue, args=(text,))
        col_note.caption(why)


def _ask(agent, question: str) -> None:
    """Answer the queued question, commit it to history, then re-render.

    The question is echoed immediately so the user sees it while the agent works.
    Once the answer lands it is committed to session state and the script reruns,
    so the page is drawn entirely from history -- never half from history and
    half from this call.
    """
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Planning, computing, and checking the numbers..."):
            # The agent never raises -- failures come back as text plus an error
            # code, so the app has no path to a raw stack trace.
            response = agent.answer(question, history=state.history_for_agent())

    state.add_user(question)
    state.add_assistant(response.text, response.to_dict())
    st.session_state.pending = None
    st.rerun()


def _evidence(payload: dict) -> None:
    """The audit trail. This is what makes the grounding claim checkable."""
    # These two are different failures and must not share a message. Telling the
    # user their answer "contained an invented figure" when the provider actually
    # returned a 400 is both wrong and alarming.
    if payload.get("degraded"):
        st.warning(
            "The language model could not be reached for this answer. Anything shown "
            "above was computed directly from the data and is still accurate.",
            icon=":material/cloud_off:",
        )
    elif payload.get("fell_back"):
        st.warning(
            "The narrative answer was rejected because it contained a figure that did "
            "not trace back to a computed value, so the raw computed values are shown "
            "instead.", icon=":material/gpp_maybe:",
        )
    elif payload.get("validation_passed") and payload.get("facts"):
        st.caption(
            f":material/verified: every figure above traces to one of "
            f"{len(payload['facts'])} computed values"
        )

    if payload.get("missing_columns"):
        st.info(
            "Asked about: " + ", ".join(f"`{c}`" for c in payload["missing_columns"])
            + " — not present in this dataset.", icon=":material/search_off:",
        )

    steps, facts = payload.get("steps") or [], payload.get("facts") or []
    if not steps and not facts:
        return

    trace_tab, facts_tab = st.tabs(
        [f"How I got this ({len(steps)} steps)", f"Facts used ({len(facts)})"]
    )

    with trace_tab:
        if payload.get("plan_reasoning"):
            st.caption(f"**Plan:** {payload['plan_reasoning']}")
        for i, step in enumerate(steps, 1):
            icon = {"ok": ":material/check_circle:", "warn": ":material/warning:",
                    "retry": ":material/refresh:", "replan": ":material/alt_route:",
                    "fatal": ":material/error:"}.get(step["status"], "")
            with st.expander(f"{icon} Step {i} — `{step['tool']}` · {step['status']}"):
                if step.get("purpose"):
                    st.caption(step["purpose"])
                st.json(step["arguments"], expanded=False)
                if step.get("hint"):
                    st.caption(f"Verifier: {step['hint']}")
                st.json(step["result"], expanded=False)
        st.caption(
            f"{payload.get('llm_calls', 0)} language-model calls · "
            f"{payload.get('seconds', 0):.1f}s. Planning and answering use the model; "
            "every computation, verification and numeric check is deterministic code."
        )

    with facts_tab:
        if not facts:
            st.caption("No numeric facts were computed for this answer.")
            return
        table = pd.DataFrame(facts)[["key", "formatted", "label", "source_tool"]]
        st.dataframe(
            table.rename(columns={"key": "ref", "formatted": "value",
                                  "source_tool": "computed by"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "The answer is written with references to these values, which are then "
            "substituted in. A figure that matches none of them is rejected before "
            "you see it."
        )

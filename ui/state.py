"""Session state and multi-turn memory for the Streamlit app.

Kept out of app.py so the rendering code stays about rendering. Nothing here
computes statistics -- it holds conversation state and delegates.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.agent.loop import ChurnAgent
from src.data.loader import (data_quality_audit, describe_cleaning, get_schema,
                             load_clean, load_raw)
from src.model import service

MAX_MEMORY_TURNS = 4


# ---------------------------------------------------------------------------
# Cached resources -- built once per process, not per rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_agent() -> ChurnAgent:
    return ChurnAgent()


@st.cache_data(show_spinner=False)
def get_dataframe():
    return load_clean()


@st.cache_data(show_spinner=False)
def get_raw_dataframe():
    return load_raw()


@st.cache_data(show_spinner=False)
def get_schema_cached() -> dict[str, Any]:
    return get_schema()


@st.cache_data(show_spinner=False)
def get_cleaning_report() -> list[dict]:
    return describe_cleaning()


@st.cache_data(show_spinner=False)
def get_audit() -> dict[str, int]:
    return data_quality_audit()


@st.cache_data(show_spinner=False)
def get_customer_ids() -> list[str]:
    from src import config
    return get_dataframe()[config.ID_COL].astype(str).tolist()


@st.cache_data(show_spinner=False)
def get_model_info() -> dict[str, Any] | None:
    """Model card, or None when the artifact has not been trained yet."""
    try:
        return service.get_model_info()
    except Exception:
        return None


def model_ready() -> bool:
    return service.model_is_ready()


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
def init() -> None:
    st.session_state.setdefault("messages", [])   # [{role, content, response?}]
    st.session_state.setdefault("pending", None)  # question queued by a button


def add_user(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})


def add_assistant(text: str, response: Any = None) -> None:
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "response": response}
    )


def history_for_agent() -> list[dict[str, str]]:
    """Prior turns as plain chat messages, excluding the question being answered.

    Only role/content is passed -- traces and ledgers are for the UI, not the
    model. Trimming happens in the agent (`_window`), which owns the token budget.
    """
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    # The question in flight is committed to history only after it is answered,
    # so `messages` already holds prior turns only. The guard stays as a
    # belt-and-braces against a caller that appends first.
    return turns[:-1] if turns and turns[-1]["role"] == "user" else turns


def clear() -> None:
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.hypothetical_result = None

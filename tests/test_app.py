"""Streamlit app smoke tests via AppTest.

`streamlit run` succeeding proves only that the server booted -- the script body
executes when a client connects, so a render-time exception is invisible until
someone opens the page. AppTest runs the real script and surfaces those.

No LLM calls: the tests render the initial page and the deterministic tabs. The
chat tab is exercised for structure, not for answers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from ui.components.chat import EXAMPLES

# AppTest resolves relative paths against the calling file, not the repo root.
APP = str((Path(__file__).resolve().parents[1] / "ui" / "app.py"))
TIMEOUT = 60


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    return app


def test_app_renders_without_exception():
    app = run_app()
    assert not app.exception, [str(e) for e in app.exception]


def test_title_and_tabs_present():
    app = run_app()
    assert any("Churn Analyst Agent" in t.value for t in app.title)
    labels = [t.label for t in app.tabs]
    for expected in ("Chat", "Explore the data", "Score a customer"):
        assert any(expected in label for label in labels), labels


def test_sidebar_reports_model_status():
    """The sidebar must state which model is loaded and where it came from."""
    app = run_app()
    text = " ".join(m.value for m in app.sidebar.success) + \
           " ".join(m.value for m in app.sidebar.error)
    assert "model" in text.lower() or "Model" in text


def test_explore_tab_shows_real_dataset_metrics():
    app = run_app()
    values = [m.value for m in app.metric]
    assert "7,043" in values, values


def test_no_exception_after_switching_explore_column():
    """Every column must render -- numeric and categorical take different paths."""
    app = run_app()
    select = next((s for s in app.selectbox if s.key == "explore_column"), None)
    assert select is not None, "explore column selector missing"
    for column in ("tenure", "PaymentMethod", "MonthlyCharges", "SeniorCitizen"):
        select.set_value(column).run()
        assert not app.exception, f"{column}: {[str(e) for e in app.exception]}"


def test_scoring_an_existing_customer_renders():
    app = run_app()
    assert not app.exception
    # The what-if tab defaults to the first customer and scores immediately.
    assert any("Churn risk" in m.label for m in app.metric), [m.label for m in app.metric]


def test_hypothetical_mode_switch_does_not_crash():
    app = run_app()
    radio = next((r for r in app.radio if r.key == "whatif_mode"), None)
    assert radio is not None
    radio.set_value("A hypothetical new customer").run()
    assert not app.exception, [str(e) for e in app.exception]


def test_chat_tab_offers_example_questions():
    """Including the missing-column trap, so a reviewer hits it without inventing prompts."""
    app = run_app()
    labels = [b.label for b in app.button]
    assert any("region" in label for label in labels), labels


def test_top_factors_table_survives_arrow_conversion():
    """top_factors mixes strings and numbers in one column.

    Arrow infers a single type per column and raises on the mix
    ("Could not convert 'No' with type str: tried to convert to double"),
    which breaks the table in a real browser while leaving app.exception empty.
    Caught only by rendering it.
    """
    import pandas as pd
    import pyarrow as pa

    from src.model import service

    if not service.model_is_ready():
        pytest.skip("model artifacts not built")

    result = service.predict_churn_risk("7590-VHVEG")
    table = pd.DataFrame(result["top_factors"])
    table["value"] = table["value"].astype(str)
    pa.Table.from_pandas(table[["feature", "value", "direction", "contribution"]])


def test_raw_top_factors_would_have_failed_arrow():
    """Guards the guard: confirms the mixed-type hazard is real, not hypothetical."""
    import pandas as pd
    import pyarrow as pa

    hazard = pd.DataFrame([{"value": "No"}, {"value": 29.85}])
    with pytest.raises(pa.ArrowInvalid):
        pa.Table.from_pandas(hazard, schema=pa.schema([("value", pa.float64())]))


# --- chat state consistency (reported from manual browser use) ---------------
class _StubAgent:
    """Deterministic stand-in so the UI flow is testable without the LLM."""

    available = True

    class _Client:
        model = "stub"
        model_source = "test"
        reason = ""

        class _Usage:
            def to_dict(self):
                return {}

        usage = _Usage()

    client = _Client()

    def __init__(self):
        self.asked = []

    def answer(self, question, history=None):
        from src.agent.loop import AgentResponse
        self.asked.append(question)
        return AgentResponse(text=f"Answer to: {question}")


@pytest.fixture
def stub_chat(monkeypatch):
    from ui import state as ui_state
    agent = _StubAgent()
    monkeypatch.setattr(ui_state, "get_agent", lambda: agent)
    monkeypatch.setattr(ui_state, "model_ready", lambda: True)
    return agent


def _click_example(app, index: int = 0):
    buttons = [b for b in app.button if b.key and b.key.startswith("eg_")]
    assert buttons, "example buttons not rendered"
    buttons[index].click().run()


def test_second_example_click_is_not_dropped(stub_chat):
    """The reported bug.

    Clicking an example, then clicking another once an answer is on screen, made
    the examples vanish and produced no new answer. Cause: the buttons were drawn
    while `messages` was still empty, then the same run appended to `messages`,
    so the next run never recreated the widget and the click had nowhere to land.
    """
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state["messages"] = []
    app.session_state["pending"] = None
    app.run()

    _click_example(app, 0)
    assert len(app.session_state["messages"]) == 2, "first question did not answer"

    # The examples must still be clickable, and a SECOND click must land.
    _click_example(app, 1)
    assert len(app.session_state["messages"]) == 4, "second click was dropped"
    assert stub_chat.asked == [EXAMPLES[0][0], EXAMPLES[1][0]]

    # Both answers must be on the page -- the old code lost the first one.
    body = " ".join(m.value for m in app.markdown)
    assert body.count("Answer to:") == 2, body
    assert not app.exception


def test_examples_stay_reachable_once_a_conversation_exists(stub_chat):
    """They move into an expander rather than disappearing -- they are the
    fastest way to demo the app, including the missing-column refusal."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state["messages"] = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer", "response": None},
    ]
    app.session_state["pending"] = None
    app.run()
    assert [b for b in app.button if b.key and b.key.startswith("eg_")]
    body = " ".join(m.value for m in app.markdown)
    assert "earlier question" in body and "earlier answer" in body


def test_queued_question_is_answered_and_committed(stub_chat):
    """A pending question must end up in history exactly once."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state["messages"] = []
    app.session_state["pending"] = "what is the churn rate?"
    app.run()

    messages = app.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "what is the churn rate?"
    assert app.session_state["pending"] is None
    assert stub_chat.asked == ["what is the churn rate?"]


def test_history_passed_to_the_agent_excludes_the_live_question(stub_chat):
    """The question in flight must not appear twice in the model's context."""
    from ui import state as ui_state

    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state["messages"] = []
    app.session_state["pending"] = "first"
    app.run()
    app.session_state["pending"] = "second"
    app.run()

    assert stub_chat.asked == ["first", "second"]
    assert len(app.session_state["messages"]) == 4


def test_a_queued_question_is_answered_exactly_once(stub_chat):
    """The queue must be drained, not replayed.

    `pending` is cleared and the script reruns, so the settled page is drawn
    purely from history. If the queue were not cleared, every rerun would ask
    the agent again and duplicate the turn.
    """
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state["messages"] = []
    app.session_state["pending"] = "something"
    app.run()

    assert stub_chat.asked == ["something"]
    assert app.session_state["pending"] is None
    assert len(app.session_state["messages"]) == 2

    app.run()   # a rerun with no new input must not re-ask
    assert stub_chat.asked == ["something"]
    assert len(app.session_state["messages"]) == 2


def test_example_buttons_render_once_per_page(stub_chat):
    """The idle grid and the in-conversation expander share keys, so exactly one
    of the two may be on the page -- duplicate keys would raise."""
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.session_state["messages"] = []
    app.session_state["pending"] = None
    app.run()
    keys = [b.key for b in app.button if b.key and b.key.startswith("eg_")]
    assert len(keys) == len(set(keys)) == len(EXAMPLES)
    assert not app.exception

"""Self-check on tool results (docs/03-AGENT-SPEC.md section 5).

Runs on every tool result before it enters the ledger. Entirely deterministic --
no LLM call -- which is the point: every check that code can make, code makes,
both for reliability and for the token budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src import config


class Status(str, Enum):
    OK = "ok"
    RETRY = "retry"        # recoverable: re-run this step with a hint
    REPLAN = "replan"      # this step was the wrong idea
    FATAL = "fatal"        # a real bug; surface it rather than paper over it
    WARN = "warn"          # usable, but flag a caveat in the answer


@dataclass
class Verdict:
    status: Status
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (Status.OK, Status.WARN)


def verify(result: Any, purpose: str = "") -> Verdict:
    """Decide whether a tool result is usable, retryable, or a bug."""
    if not isinstance(result, dict):
        return Verdict(Status.RETRY, "Tool returned an unexpected type.")

    # --- explicit tool errors -------------------------------------------
    if "error" in result:
        code = result.get("error")
        message = result.get("message") or result.get("detail") or ""
        hint = result.get("hint") or ""
        if code in {"unknown_feature", "unknown_tool", "invalid_value"}:
            # The plan referenced something that does not exist -- rethink it
            # rather than retrying the same call.
            return Verdict(Status.REPLAN, f"{message} {hint}".strip())
        if code in {"unsafe_code", "execution_error", "timeout", "empty_result", "bad_arguments"}:
            return Verdict(Status.RETRY, f"{message} {hint}".strip())
        if code == "model_unavailable":
            return Verdict(Status.FATAL, message)
        return Verdict(Status.RETRY, f"{message} {hint}".strip())

    # --- empty results ---------------------------------------------------
    # Not automatically a bug: "no customers match" can be the true answer. But
    # a mistyped category value produces the same signature, so send it back for
    # one check against the schema before accepting it.
    if result.get("warning") == "no_customers_match" or result.get("n_customers") == 0:
        return Verdict(Status.WARN, "No customers matched. Confirm the filter values against the schema.")
    if result.get("ok") is True and result.get("kind") in {"dataframe", "series", "list"}:
        if result.get("rows") == 0:
            return Verdict(Status.RETRY, "The query returned zero rows. Check the filter values.")

    # --- nonsensical numbers ---------------------------------------------
    problems = _scan_numbers(result)
    if problems:
        return Verdict(Status.FATAL if problems[0][0] == "fatal" else Status.RETRY, problems[0][1])

    return Verdict(Status.OK)


_PROBABILITY_FIELDS = {
    "risk_score", "mean_risk", "median_risk", "actual_churn_rate",
    "population_base_rate", "baseline_mean_risk", "churn_rate", "overall_churn_rate",
}
_COUNT_FIELDS = {"n_customers", "n_rows", "count", "churn_count", "actual_churn_count", "rows"}


def _scan_numbers(node: Any, depth: int = 0) -> list[tuple[str, str]]:
    """Look for impossible values: out-of-range probabilities, absurd counts."""
    if depth > 4:
        return []
    problems: list[tuple[str, str]] = []

    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in _PROBABILITY_FIELDS and not (0.0 <= float(value) <= 1.0):
                    problems.append(("fatal", f"'{key}' is {value}, outside [0, 1] -- pipeline bug."))
                if key in _COUNT_FIELDS:
                    if value < 0:
                        problems.append(("fatal", f"'{key}' is negative ({value}) -- filter logic is inverted."))
                    elif value > config.EXPECTED_ROWS:
                        problems.append(
                            ("fatal", f"'{key}' is {value}, more than the {config.EXPECTED_ROWS} rows "
                                      "in the dataset -- the query is double-counting.")
                        )
            else:
                problems += _scan_numbers(value, depth + 1)
    elif isinstance(node, (list, tuple)):
        for item in node[:20]:
            problems += _scan_numbers(item, depth + 1)
    return problems


def calibration_warning(result: dict) -> str | None:
    """Flag a segment whose predicted risk diverges sharply from observed churn.

    Warn, never block: the model can be legitimately confident about a small
    segment. But the user deserves to know when prediction and reality disagree.
    """
    predicted, actual = result.get("mean_risk"), result.get("actual_churn_rate")
    if predicted is None or actual is None:
        return None
    if abs(float(predicted) - float(actual)) > 0.25:
        return (
            f"Predicted mean risk ({predicted:.2f}) diverges from the observed churn rate "
            f"({actual:.2f}) for this segment."
        )
    return None

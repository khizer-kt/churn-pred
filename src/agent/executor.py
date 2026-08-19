"""Restricted pandas execution (docs/03-AGENT-SPEC.md section 3).

This is not a security boundary against a determined attacker -- it runs
in-process, as the brief permits. It is a boundary against a *confused model*:
preventing accidental state corruption, runaway loops, and data exfiltration
through generated code.

The check is an AST allowlist, not string blacklisting. Blacklisting "import"
or "__" in the source text is trivially evaded (getattr chains, string
concatenation, unicode escapes) and gives false confidence.
"""
from __future__ import annotations

import ast
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "__import__", "breakpoint",
    "memoryview", "help", "exit", "quit",
}

FORBIDDEN_ATTRS = {
    "to_csv", "to_pickle", "to_json", "to_sql", "to_parquet", "to_excel",
    "to_hdf", "to_clipboard", "read_csv", "read_pickle", "read_json",
    "read_sql", "read_parquet", "read_excel", "system", "popen",
}

SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "range": range,
    "enumerate": enumerate, "zip": zip, "any": any, "all": all, "print": print,
    "isinstance": isinstance, "divmod": divmod, "reversed": reversed,
}


@dataclass
class ExecResult:
    """Outcome of one code execution. `ok=False` carries a structured error."""

    ok: bool
    value: Any = None
    kind: str = "none"           # scalar | dataframe | series | dict | list | none
    rows: int | None = None      # true row count before truncation
    truncated: bool = False
    error: str | None = None
    detail: str | None = None
    hint: str | None = None
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "kind": self.kind}
        if self.ok:
            out["value"] = self.value
            if self.rows is not None:
                out["rows"] = self.rows
            if self.truncated:
                out["truncated"] = True
        else:
            out.update({"error": self.error, "detail": self.detail})
            if self.hint:
                out["hint"] = self.hint
        return out


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------
class UnsafeCode(Exception):
    def __init__(self, detail: str, hint: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.hint = hint


def validate_code(code: str) -> ast.Module:
    """Parse and walk the AST, rejecting anything outside the allowlist."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCode(f"Syntax error: {exc.msg} (line {exc.lineno})",
                         "Write plain pandas statements and assign to `result`.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode("import statements are not permitted",
                             "`df`, `pd` and `np` are already available.")
        if isinstance(node, ast.While):
            raise UnsafeCode("while loops are not permitted",
                             "Use vectorised pandas operations or a comprehension.")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCode("global/nonlocal declarations are not permitted")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCode(f"access to dunder attribute '{node.attr}' is not permitted")
            if node.attr in FORBIDDEN_ATTRS:
                raise UnsafeCode(f"'{node.attr}' is not permitted",
                                 "Return the data as `result` instead of writing it out.")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise UnsafeCode(f"access to '{node.id}' is not permitted")
            if node.id in FORBIDDEN_CALLS:
                raise UnsafeCode(f"'{node.id}' is not permitted")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise UnsafeCode(f"calling '{node.func.id}' is not permitted")

    if "result" not in {
        t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    } | {
        n.target.id for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }:
        raise UnsafeCode("the code must assign its answer to a variable named `result`",
                         "e.g. `result = df['Churn'].mean()`")
    return tree


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _to_native(v: Any) -> Any:
    """Convert to a JSON-safe Python value, collapsing NaN/inf to None.

    NaN must not survive this boundary: it serialises to invalid JSON, and a
    silent NaN reaching the ledger would be reported to the user as a number.
    Both numpy and plain Python floats are handled -- `float('nan')` from
    generated code is not an np.floating.
    """
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    if v is pd.NaT or v is None:
        return None
    return v


def serialise(value: Any) -> ExecResult:
    """Convert a result to JSON-safe form, truncating anything large.

    The true row count always travels with a truncated result so the agent
    knows it is looking at a sample rather than the whole answer.
    """
    if isinstance(value, pd.DataFrame):
        rows = len(value)
        head = value.head(config.EXEC_MAX_ROWS)
        records = [{k: _to_native(v) for k, v in rec.items()}
                   for rec in head.to_dict(orient="records")]
        return ExecResult(ok=True, value=records, kind="dataframe", rows=rows,
                          truncated=rows > config.EXEC_MAX_ROWS)
    if isinstance(value, pd.Series):
        rows = len(value)
        head = value.head(config.EXEC_MAX_ROWS)
        return ExecResult(
            ok=True,
            value={str(k): _to_native(v) for k, v in head.items()},
            kind="series", rows=rows, truncated=rows > config.EXEC_MAX_ROWS,
        )
    if isinstance(value, dict):
        return ExecResult(ok=True, value={str(k): _to_native(v) for k, v in value.items()},
                          kind="dict", rows=len(value))
    if isinstance(value, (list, tuple, set)):
        items = list(value)[: config.EXEC_MAX_ROWS]
        return ExecResult(ok=True, value=[_to_native(v) for v in items], kind="list",
                          rows=len(value), truncated=len(value) > config.EXEC_MAX_ROWS)
    return ExecResult(ok=True, value=_to_native(value), kind="scalar")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_code(code: str, df: pd.DataFrame, timeout: int | None = None) -> ExecResult:
    """Validate, then execute `code` against a private copy of `df`.

    The copy is per-call: 7,043 rows is trivially cheap to duplicate, and it
    removes any chance that generated code mutates state shared between turns.
    """
    timeout = timeout or config.EXEC_TIMEOUT_SECONDS

    try:
        tree = validate_code(code)
    except UnsafeCode as exc:
        return ExecResult(ok=False, error="unsafe_code", detail=exc.detail,
                          hint=exc.hint or None, code=code)

    namespace: dict[str, Any] = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "result": None,
        "__builtins__": dict(SAFE_BUILTINS),
    }

    outcome: queue.Queue = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            exec(compile(tree, "<agent>", "exec"), namespace)  # noqa: S102 - sandboxed namespace
            outcome.put(("ok", namespace.get("result")))
        except Exception as exc:  # surfaced to the agent as a retryable error
            outcome.put(("err", f"{type(exc).__name__}: {exc}"))

    # A daemon thread gives a wall-clock timeout that works off the main thread,
    # which signal.alarm does not -- Streamlit runs scripts in worker threads.
    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        return ExecResult(ok=False, error="timeout",
                          detail=f"Execution exceeded {timeout}s and was abandoned.",
                          hint="Simplify the query; avoid row-by-row loops.", code=code)

    status, payload = outcome.get()
    if status == "err":
        return ExecResult(ok=False, error="execution_error", detail=payload,
                          hint="Check column names against the schema.", code=code)

    if payload is None:
        return ExecResult(ok=False, error="empty_result",
                          detail="`result` was never assigned a value.",
                          hint="Assign the answer to `result`.", code=code)

    out = serialise(payload)
    out.code = code
    return out

"""The agent's tool catalogue (docs/03-AGENT-SPEC.md section 2).

Four tools, each a plain function plus a JSON schema for the tool-calling API.
Guidance encoded in the descriptions: prefer the narrowest tool that can answer
the question, because narrow tools cannot go wrong in as many ways.

Every tool returns a dict and never raises -- the agent must be able to react.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src import config
from src.agent.executor import run_code
from src.data.loader import ABSENT_CONCEPTS, get_schema, load_clean
from src.model import service


# ---------------------------------------------------------------------------
# T1 -- schema
# ---------------------------------------------------------------------------
def tool_get_schema() -> dict[str, Any]:
    """Columns, dtypes, allowed values and ranges.

    Loaded into the system prompt at startup, but kept callable so the agent can
    re-ground mid-plan. This is the primary defence against claiming a column
    exists when it does not.
    """
    return get_schema()


def schema_prompt_block() -> str:
    """Compact schema rendering for the system prompt.

    Kept terse on purpose: it is sent with every request, so verbosity here is
    paid for on every question against a rate-limited free tier.
    """
    schema = get_schema()
    lines = [
        f"Dataset: {schema['row_count']} rows x {schema['column_count']} columns. "
        f"{schema['grain']}",
        f"Target: {schema['target']} (1=churned, 0=retained). "
        f"Overall churn rate: {schema['churn_rate']:.4f}",
        "",
        "COLUMNS:",
    ]
    for col in schema["columns"]:
        if col["role"] == "other":
            continue
        if "allowed_values" in col:
            vals = col["allowed_values"]
            shown = ", ".join(vals[:6]) + ("..." if len(vals) > 6 else "")
            lines.append(f"  {col['name']} ({col['role']}): one of [{shown}]")
        elif "min" in col:
            lines.append(
                f"  {col['name']} ({col['role']}): numeric {col['min']}-{col['max']}, "
                f"median {col['median']}"
            )
        else:
            lines.append(f"  {col['name']} ({col['role']}): {col['dtype']}")
    lines += [
        "",
        "COLUMNS THAT DO NOT EXIST (say so plainly if asked; never invent them):",
    ]
    lines += [f"  {name}: {why}" for name, why in ABSENT_CONCEPTS.items()]
    return "\n".join(lines)


def check_absent_concepts(text: str) -> list[dict[str, str]]:
    """Flag references to concepts the dataset does not contain.

    Runs on the question and on the plan before any computation, so a question
    about 'region' is answered from the schema rather than by an aggregation
    that quietly returns something plausible. See finding I10.
    """
    lowered = (text or "").lower()
    hits = []
    for concept, explanation in ABSENT_CONCEPTS.items():
        if concept in lowered:
            hits.append({"concept": concept, "explanation": explanation})
    return hits


# ---------------------------------------------------------------------------
# T2 -- restricted computation
# ---------------------------------------------------------------------------
def tool_run_analysis(code: str, purpose: str = "") -> dict[str, Any]:
    """Execute pandas against the cleaned dataframe.

    `purpose` is required by the schema and is not decoration: the verifier
    compares the result's shape against the stated intent, and it makes the
    trace readable to a human reviewing what the agent did.

    The frame includes `risk_score` for every customer, so code can filter and
    aggregate on model output as well as on raw columns.
    """
    try:
        df = service.score_all_customers()
    except Exception:
        df = load_clean()  # model not trained yet; raw analysis still works

    result = run_code(code, df)
    payload = result.to_dict()
    payload["purpose"] = purpose
    payload["code"] = code
    return payload


# ---------------------------------------------------------------------------
# T3 -- model tools (thin pass-through to src/model/service.py)
# ---------------------------------------------------------------------------
def tool_predict_churn_risk(
    customer_id: str | None = None,
    features: dict | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Risk for one real or hypothetical customer, optionally under what-ifs."""
    return service.predict_churn_risk(customer_id=customer_id, features=features, overrides=overrides)


def tool_predict_segment_risk(
    filters: dict | None = None,
    overrides: dict | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Aggregate risk across a segment defined by column filters."""
    return service.predict_segment_risk(filters=filters, overrides=overrides, top_n=top_n)


# ---------------------------------------------------------------------------
# T4 -- distributions
# ---------------------------------------------------------------------------
def tool_get_distribution(column: str, by: str | None = None, bins: int = 10) -> dict[str, Any]:
    """Distribution of one column, optionally cross-tabbed, with churn rate per level.

    Strictly a convenience wrapper over what T2 could express, but it covers most
    EDA questions in a single deterministic call with no model-generated code --
    which saves tokens and removes an entire failure mode.

    Also backs the Streamlit Explore tab, so the panel and the agent report
    identical numbers by construction.
    """
    df = load_clean()

    if column not in df.columns:
        return {
            "error": "unknown_feature",
            "message": f"'{column}' is not a column in this dataset.",
            "valid_columns": [c for c in df.columns],
            "absent_concepts": check_absent_concepts(column),
        }
    if by is not None and by not in df.columns:
        return {
            "error": "unknown_feature",
            "message": f"'{by}' is not a column in this dataset.",
            "valid_columns": [c for c in df.columns],
        }

    target = config.TARGET_COL
    is_numeric = pd.api.types.is_numeric_dtype(df[column]) and column not in (target,)

    out: dict[str, Any] = {"column": column, "n_rows": int(len(df))}

    if is_numeric:
        series = df[column]
        out["summary"] = {
            "min": round(float(series.min()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
        }
        binned = pd.cut(series, bins=bins)
        grouped = df.groupby(binned, observed=True)[target].agg(["size", "mean"])
        out["bins"] = [
            {
                "range": f"{iv.left:.1f}-{iv.right:.1f}",
                "count": int(row["size"]),
                "churn_rate": round(float(row["mean"]), 4),
            }
            for iv, row in grouped.iterrows()
        ]
        out["mean_by_churn"] = {
            "churned": round(float(df.loc[df[target] == 1, column].mean()), 2),
            "retained": round(float(df.loc[df[target] == 0, column].mean()), 2),
        }
    else:
        grouped = df.groupby(column, observed=True)[target].agg(["size", "mean", "sum"])
        grouped = grouped.sort_values("size", ascending=False)
        out["levels"] = [
            {
                "value": str(idx),
                "count": int(row["size"]),
                "share": round(float(row["size"]) / len(df), 4),
                "churn_rate": round(float(row["mean"]), 4),
                "churn_count": int(row["sum"]),
            }
            for idx, row in grouped.iterrows()
        ]

    if by is not None:
        cross = pd.crosstab(df[column], df[by])
        out["crosstab"] = {
            "by": by,
            "table": {str(i): {str(c): int(v) for c, v in row.items()}
                      for i, row in cross.head(config.EXEC_MAX_ROWS).iterrows()},
        }

    out["overall_churn_rate"] = round(float(df[target].mean()), 4)
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "get_schema": tool_get_schema,
    "run_analysis": tool_run_analysis,
    "predict_churn_risk": tool_predict_churn_risk,
    "predict_segment_risk": tool_predict_segment_risk,
    "get_distribution": tool_get_distribution,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_distribution",
            "description": (
                "PREFERRED for questions about how one column is distributed, or its churn "
                "rate by level. Returns counts, shares and churn rate per level (or bins and "
                "summary stats for numeric columns). Use this before reaching for run_analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column to describe."},
                    "by": {"type": "string", "description": "Optional second column to cross-tabulate against."},
                    "bins": {"type": "integer", "description": "Bin count for numeric columns. Default 10."},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn_risk",
            "description": (
                "Predicted churn risk for ONE customer, with the factors driving it. "
                "Pass customer_id for an existing customer; add overrides to project the "
                "score forward under changed conditions (what-if); or pass a complete "
                "features dict for a hypothetical new customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "e.g. 7590-VHVEG"},
                    "features": {"type": "object", "description": "Full feature dict for a hypothetical customer."},
                    "overrides": {"type": "object", "description": "Fields to change, e.g. {\"Contract\": \"Two year\"}"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_segment_risk",
            "description": (
                "Aggregate predicted churn risk across a SEGMENT of customers. Returns mean "
                "and median predicted risk, the observed churn rate, lift over the population "
                "base rate, and the highest-risk customers. Use for 'which customers are most "
                "likely to churn' and any question about a group. Pass filters as "
                "{column: value}, {column: [v1, v2]} or {column: {\"min\": x, \"max\": y}}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {"type": "object", "description": "Column filters. Empty for the whole population."},
                    "overrides": {"type": "object", "description": "What-if changes applied to every matching customer."},
                    "top_n": {"type": "integer", "description": "How many top-risk customers to return. Default 5."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Run pandas against the dataframe when no other tool can express the question "
                "(multi-column aggregations, correlations, custom filters, rankings). "
                "`df` holds the cleaned data plus a `risk_score` column; `pd` and `np` are "
                "available. Assign the answer to a variable named `result`. No imports, no "
                "file I/O, no while loops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python that assigns to `result`."},
                    "purpose": {"type": "string", "description": "One line: what this should produce."},
                },
                "required": ["code", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Full schema: columns, types, allowed values, ranges, and the list of concepts "
                "the dataset does NOT contain. The schema is already in your system prompt; "
                "call this only to re-check details mid-plan."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def dispatch(name: str, arguments: dict | None = None) -> dict[str, Any]:
    """Call a tool by name. Unknown names and bad arguments become error dicts."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {
            "error": "unknown_tool",
            "message": f"'{name}' is not an available tool.",
            "available_tools": sorted(TOOL_FUNCTIONS),
        }
    try:
        return func(**(arguments or {}))
    except TypeError as exc:
        return {"error": "bad_arguments", "message": f"Invalid arguments for '{name}': {exc}"}
    except Exception as exc:
        return {"error": "tool_failed", "message": f"{type(exc).__name__}: {exc}"}

"""Fact Ledger and numeric-grounding validator (docs/03-AGENT-SPEC.md section 4).

The brief's most heavily weighted requirement is "never invent a number".
Prompting for it does not work, so it is enforced structurally in two layers:

  Layer 1 (primary)    -- the model writes prose with [[F1]] citation tokens and
                          never types digits. It cannot fabricate what it does
                          not write.
  Layer 2 (enforcement) -- the rendered answer is scanned for numeric literals;
                          any that cannot be traced to a registered fact is a
                          hard failure.

Layer 1 depends on instruction-following, which free-tier models do imperfectly.
Layer 2 does not depend on the model at all. Neither is sufficient alone.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
# Fields that are labels or flags rather than measurements. `actual_churn` is 0/1
# for a single customer; registered as a fact it can be rendered "100.0%" and read
# as a rate, which is precisely the confusion this system exists to prevent.
_SKIP_KEYS = {
    "error", "message", "hint", "detail", "code", "purpose",
    "actual_churn", "predicted_churn", "ok", "truncated",
}


@dataclass
class Fact:
    """One number the agent actually computed, with its provenance."""

    key: str
    value: float
    label: str
    unit: str = "raw"          # percent | count | currency | probability | raw
    source_tool: str = ""
    source_args: dict = field(default_factory=dict)
    step: int = 0

    def format(self) -> str:
        """Render for display. Precision follows the unit's natural precision."""
        if self.unit == "percent":
            return f"{self.value:.2f}%"
        if self.unit == "probability":
            return f"{self.value:.1%}"
        if self.unit == "delta":
            # Percentage points, signed. A risk moving 0.29 -> 0.10 is "-19.5 pp",
            # not "-19.5%", which would wrongly read as a relative change.
            return f"{self.value * 100:+.1f} pp"
        if self.unit == "currency":
            return f"${self.value:,.2f}"
        if self.unit == "count":
            return f"{int(round(self.value)):,}"
        if float(self.value).is_integer():
            return str(int(self.value))
        return f"{self.value:,.4g}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "formatted": self.format(),
            "label": self.label,
            "unit": self.unit,
            "source_tool": self.source_tool,
            "source_args": self.source_args,
            "step": self.step,
        }


# Field-name patterns mapped to display units, so a registered value knows how
# it should be rendered and compared.
# Leaves carrying no meaning of their own, so the surrounding label is consulted
# instead (run_analysis results are always ".value").
_GENERIC_LEAVES = {"value", "result", "data", "output", ""}

_UNIT_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(delta|change|diff)", re.I), "delta"),
    (re.compile(r"(rate|pct|percent|share|proportion)", re.I), "probability"),
    (re.compile(r"(risk|score|probability|threshold|auc|brier|recall|precision|f2|lift)", re.I), "probability"),
    (re.compile(r"^(n_|count|num_|total_customers)|(_count|_n)$", re.I), "count"),
    (re.compile(r"(charges|spend|revenue|cost|price)", re.I), "currency"),
]


def infer_unit(name: str, value: float) -> str:
    """Guess a display unit from the field name, falling back to raw.

    Deliberately conservative: 'lift' and 'auc' read as probability-like ratios
    but must not be rendered as percentages, so the probability formatter is
    only applied to values that actually sit in [0, 1].
    """
    # Labels arrive dotted ("predict_segment_risk.n_customers") and indexed
    # ("top_customers[0].risk_score"). Match on the leaf, or an anchored pattern
    # like ^n_ never fires and counts render unformatted.
    leaf = name.split(".")[-1].split("[")[0] or name
    # The leaf is authoritative whenever it is a real field name. The full label
    # is consulted ONLY for placeholder leaves such as ".value" -- otherwise the
    # step's purpose text bleeds into every field it produced, and a step
    # described as "risk after the contract change" turns its `threshold` into a
    # delta and reports an unchanged 0.28 as "+28.0 pp".
    candidates = (leaf, name) if leaf.lower() in _GENERIC_LEAVES else (leaf,)
    for candidate in candidates:
        for pattern, unit in _UNIT_HINTS:
            if pattern.search(candidate):
                # A probability difference is legitimately negative; a probability
                # is not. Values outside the unit's range mean the name matched
                # something it does not actually describe.
                if unit == "delta" and not (-1.0 <= float(value) <= 1.0):
                    continue
                if unit == "probability" and not (0.0 <= float(value) <= 1.0):
                    continue
                return unit
    return "raw"


class FactLedger:
    """The only legitimate source of numbers in a final answer.

    Also the audit trail: the UI renders it so a user can click any figure and
    see the tool call and arguments that produced it.
    """

    def __init__(self) -> None:
        self._facts: list[Fact] = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self):
        return iter(self._facts)

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts)

    def add(self, value: Any, label: str, unit: str | None = None,
            source_tool: str = "", source_args: dict | None = None, step: int = 0) -> Fact | None:
        """Register one numeric value. Returns None for non-numeric input."""
        num = _as_number(value)
        if num is None:
            return None
        self._counter += 1
        fact = Fact(
            key=f"F{self._counter}",
            value=num,
            label=label,
            unit=unit or infer_unit(label, num),
            source_tool=source_tool,
            source_args=source_args or {},
            step=step,
        )
        self._facts.append(fact)
        return fact

    def register_result(self, result: Any, tool: str, args: dict | None = None,
                        step: int = 0, prefix: str = "") -> list[Fact]:
        """Walk a tool result and register every scalar it contains.

        Nested dicts and lists are flattened into dotted labels so a value deep
        inside a segment summary is still individually citable.
        """
        added: list[Fact] = []
        self._walk(result, prefix or tool, tool, args or {}, step, added, depth=0)
        return added

    def _walk(self, node: Any, label: str, tool: str, args: dict,
              step: int, added: list[Fact], depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _SKIP_KEYS:
                    continue
                self._walk(value, f"{label}.{key}" if label else str(key),
                           tool, args, step, added, depth + 1)
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node[:20]):
                # "levels[0].churn_rate" tells the model nothing, so it writes
                # "the first level". Name the item by whatever identifies it and
                # the answer can say "Month-to-month" instead.
                tag = _identity(value) or str(i)
                self._walk(value, f"{label}[{tag}]", tool, args, step, added, depth + 1)
        elif isinstance(node, bool):
            return  # booleans are not quotable figures
        else:
            fact = self.add(node, label, source_tool=tool, source_args=args, step=step)
            if fact:
                added.append(fact)

    def get(self, key: str) -> Fact | None:
        for fact in self._facts:
            if fact.key == key:
                return fact
        return None

    def as_prompt_block(self, limit: int = 60) -> str:
        """The fact list as shown to the model in the answer prompt."""
        if not self._facts:
            return "(no facts computed)"
        return "\n".join(
            f"  {f.key} = {f.format()}  ({f.label})" for f in self._facts[:limit]
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self._facts]

    # -- Layer 1 -----------------------------------------------------------
    def substitute(self, draft: str) -> tuple[str, list[str]]:
        """Replace [[Fx]] tokens with formatted values.

        A token referencing an unknown key is a hard error, not a silent
        pass-through: it means the model invented a citation.
        """
        unknown: list[str] = []

        def _replace(match: re.Match) -> str:
            key = match.group(1).strip().upper()
            fact = self.get(key)
            if fact is None:
                unknown.append(key)
                return f"[unavailable:{key}]"
            return fact.format()

        rendered = re.sub(r"\[\[\s*(F\d+)\s*\]\]", _replace, draft, flags=re.I)
        return rendered, unknown


_IDENTITY_KEYS = ("value", "customer_id", "feature", "range", "name", "level", "bin_lower")


def _identity(item: Any) -> str | None:
    """A human-meaningful name for a list item, if it carries one."""
    if not isinstance(item, dict):
        return None
    for key in _IDENTITY_KEYS:
        if key in item and isinstance(item[key], (str, int, float)):
            return str(item[key])
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return None


# ---------------------------------------------------------------------------
# Layer 2 -- the validator
# ---------------------------------------------------------------------------
# Matches 1,234  12.5%  $70.35  0.42  -3
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")

_ORDINAL_WORDS = re.compile(r"\b(top|first|next|step|section|figure)\s+\d+\b", re.I)


@dataclass
class ValidationResult:
    ok: bool
    unmatched: list[str] = field(default_factory=list)
    unknown_tokens: list[str] = field(default_factory=list)

    def message(self) -> str:
        parts = []
        if self.unmatched:
            parts.append(
                "These numbers do not correspond to anything that was computed: "
                + ", ".join(self.unmatched)
            )
        if self.unknown_tokens:
            parts.append("These citation keys do not exist: " + ", ".join(self.unknown_tokens))
        return " ".join(parts)


def _parse_number(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _matches(candidate: float, fact: "Fact", displayed_decimals: int, is_percent: bool) -> bool:
    """Compare a literal from the answer against one fact, respecting its unit.

    A ledger probability of 0.2654 must satisfy "26.54%", "26.5%" and "0.265",
    so both renderings are accepted *for probability-like facts*. They are NOT
    accepted for counts: allowing the x100 rendering universally means a fact
    whose value is 1 can justify the claim "100.0%", which is how a plain outcome
    flag ends up quoted as a rate.

    Tolerance is half a unit at the answer's displayed precision -- exactly the
    rounding error the model's own formatting introduces.
    """
    tol = max(0.5 * (10 ** -displayed_decimals), 1e-9)
    value = float(fact.value)

    if abs(candidate - value) <= tol:
        return True

    # Scale conversion only where the fact is genuinely a ratio.
    if fact.unit in {"probability", "delta", "percent"}:
        loose = max(tol, 0.05)
        if abs(candidate - value * 100) <= loose or abs(candidate * 100 - value) <= loose:
            return True
    # An unclassified fraction in [0,1] quoted as a percentage is still legitimate,
    # but only when the answer actually marked it as one.
    if fact.unit == "raw" and is_percent and 0.0 <= value <= 1.0:
        if abs(candidate - value * 100) <= max(tol, 0.05):
            return True
    return False


def validate_numbers(
    text: str,
    ledger: FactLedger,
    question: str = "",
    substituted: Iterable[str] = (),
    unknown_tokens: Iterable[str] = (),
) -> ValidationResult:
    """Reject any numeric literal in `text` that no computed fact supports.

    Allowlist, dropped before checking:
      * values substituted from citation tokens this turn
      * numbers appearing verbatim in the user's own question
      * small integers used as list ordinals ("top 5", "3 factors")
    """
    allowed_literals = {s.strip() for s in substituted}
    question_numbers = {
        n for tok in _NUMBER_RE.findall(question or "") if (n := _parse_number(tok)) is not None
    }
    ordinal_spans = [m.span() for m in _ORDINAL_WORDS.finditer(text)]
    facts = list(ledger)

    unmatched: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        if token in allowed_literals:
            continue
        # Skip numbers inside "top 5" / "step 2" style phrases.
        if any(start <= match.start() < end for start, end in ordinal_spans):
            continue

        candidate = _parse_number(token)
        if candidate is None:
            continue
        if candidate in question_numbers:
            continue
        # Bare small integers are almost always enumeration, not claims.
        if float(candidate).is_integer() and 0 <= candidate <= 10 and "%" not in token and "$" not in token:
            continue

        decimals = len(token.split(".")[1].rstrip("%")) if "." in token else 0
        is_percent = token.endswith("%")
        if not any(_matches(candidate, f, decimals, is_percent) for f in facts):
            unmatched.append(token)

    unknown = list(unknown_tokens)
    return ValidationResult(ok=not unmatched and not unknown, unmatched=unmatched,
                            unknown_tokens=unknown)


def template_answer(ledger: FactLedger, note: str = "") -> str:
    """Deterministic fallback when the model cannot produce a grounded answer.

    A visibly limited answer is a correct outcome. A fluent answer containing
    one invented figure is the failure this project exists to prevent.
    """
    if not len(ledger):
        return ("I could not compute anything for that question. "
                "Try rephrasing it, or ask about a column listed in the Explore tab.")
    lines = ["Here is what I computed:", ""]
    lines += [f"- **{f.label}**: {f.format()}" for f in ledger]
    lines += ["", note or "I could not produce a reliable narrative summary for this question."]
    return "\n".join(lines)

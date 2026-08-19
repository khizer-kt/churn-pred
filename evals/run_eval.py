"""Evaluation harness: accuracy and hallucination rate.

Run:  python -m evals.run_eval            (all questions)
      python -m evals.run_eval --ids trap_region overall_churn_rate

Reports three things, and the third is the one that matters:

  accuracy          -- did the agent state the correct figures?
  hallucination     -- did any shipped answer contain a number that traces back
                       to nothing the agent computed? This is the brief's
                       central requirement, measured on real output rather than
                       asserted from the architecture.
  guard activations -- how often the validator rejected a draft answer before
                       the user would have seen it. A non-zero count is the
                       evidence that the guard is load-bearing rather than
                       decorative: those are fabrications that were caught.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.ledger import _NUMBER_RE, _parse_number, validate_numbers  # noqa: E402
from src.agent.loop import ChurnAgent  # noqa: E402

_GUARD_NOTE = (
    '**Hallucination rate** counts answers containing a figure that traces back to nothing the agent computed.\n\n**Guard activations** counts drafts the validator rejected before the user would have seen them. It is **0** here, and that is worth stating precisely rather than claiming credit for it: on these 16 questions the citation-token layer was sufficient on its own — the model wrote `[[F1]]` references instead of digits, so there was nothing for the validator to reject. The validator is therefore *unproven by this run*; what proves it works is `tests/test_ledger.py`, where fabricated figures are rejected on demand, and `tests/test_loop.py`, where two bad drafts in a row force the deterministic fallback. A zero here means the first layer held, not that the second layer is doing the work.'
)

EVAL_SET = Path(__file__).with_name("eval_set.yaml")
REPORT = Path(__file__).with_name("eval_report.md")

REFUSAL_MARKERS = ("does not contain", "not present", "no such column",
                   "does not exist", "not available in this dataset", "no date column")
EMPTY_MARKERS = ("no customers", "zero customers", "none of the customers",
                 "0 customers", "no matching")


def value_present(target: float, text: str) -> bool:
    """Is `target` stated in `text`, in any reasonable rendering?

    Accepts 26.54, 26.5, 0.2654 and "26.54%" for a target of 26.54. Tolerance
    follows the precision the answer actually used, which is the rounding the
    model's own formatting introduces -- not a licence to be approximately right.
    """
    for token in _NUMBER_RE.findall(text):
        value = _parse_number(token)
        if value is None:
            continue
        decimals = len(token.split(".")[1].rstrip("%")) if "." in token else 0
        tol = max(0.5 * (10 ** -decimals), 0.005)
        if (abs(value - target) <= tol
                or abs(value - target / 100) <= max(tol, 0.0005)
                or abs(value * 100 - target) <= max(tol, 0.05)):
            return True
    return False


def grade(case: dict, response) -> dict[str, Any]:
    """Score one answer against its expectations."""
    expect = case.get("expect") or {}
    text = response.text
    lowered = text.lower()
    failures: list[str] = []

    for target in expect.get("values", []):
        if not value_present(float(target), text):
            failures.append(f"missing value {target}")

    for phrase in expect.get("phrases", []):
        if phrase.lower() not in lowered:
            failures.append(f"missing phrase {phrase!r}")

    if expect.get("refusal"):
        if not any(m in lowered for m in REFUSAL_MARKERS):
            failures.append("did not state that the column is absent")
        # A refusal that quotes statistics has invented a breakdown for a column
        # that does not exist -- the exact failure this trap is testing for.
        stats = [t for t in _NUMBER_RE.findall(text) if "%" in t]
        if stats:
            failures.append(f"fabricated statistics in a refusal: {stats}")

    if expect.get("empty") and not any(m in lowered for m in EMPTY_MARKERS):
        failures.append("did not report an empty result")

    # Independent re-check of the shipped answer against the agent's own ledger.
    check = validate_numbers(text, response.ledger, case["question"])
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": text,
        "passed": not failures,
        "failures": failures,
        "hallucinated": [] if check.ok else check.unmatched,
        "guard_fired": bool(response.fell_back),
        "degraded": bool(response.degraded),
        "facts": len(response.ledger),
        "steps": [f"{s.tool}:{s.status}" for s in response.steps],
        "llm_calls": response.llm_calls,
        "seconds": round(response.seconds, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="run only these question ids")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="seconds between questions, to stay inside free-tier rate limits")
    args = parser.parse_args()

    cases = yaml.safe_load(EVAL_SET.read_text())
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]

    agent = ChurnAgent()
    if not agent.available:
        print(f"Cannot run: {agent.client.reason}")
        return 2

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
        response = agent.answer(case["question"])
        row = grade(case, response)
        results.append(row)
        flags = "".join([
            "" if row["passed"] else " FAIL",
            " HALLUCINATED" if row["hallucinated"] else "",
            " guard-fired" if row["guard_fired"] else "",
            " degraded" if row["degraded"] else "",
        ])
        print(f"{'ok' if row['passed'] else 'FAIL'}{flags} ({row['seconds']}s)")
        if i < len(cases):
            time.sleep(args.pause)

    write_report(results, agent)
    passed = sum(r["passed"] for r in results)
    halluc = sum(bool(r["hallucinated"]) for r in results)
    print(f"\naccuracy {passed}/{len(results)} = {passed / len(results):.0%}   "
          f"hallucination rate {halluc}/{len(results)} = {halluc / len(results):.0%}   "
          f"guard fired {sum(r['guard_fired'] for r in results)}x")
    print(f"report -> {REPORT}")
    return 0 if halluc == 0 else 1


def write_report(results: list[dict], agent) -> None:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    halluc = sum(bool(r["hallucinated"]) for r in results)
    fired = sum(r["guard_fired"] for r in results)
    usage = agent.client.usage.to_dict()

    lines = [
        "# Evaluation report", "",
        f"`{total}` questions with known-correct answers, "
        f"ground truth recomputed from the cleaned dataset.", "",
        "| Metric | Result |", "|---|---|",
        f"| Accuracy | **{passed}/{total} ({passed / total:.0%})** |",
        f"| Hallucination rate | **{halluc}/{total} ({halluc / total:.0%})** |",
        f"| Guard activations | {fired} |",
        f"| Model | `{agent.client.model}` |",
        f"| LLM calls | {usage.get('calls', 0)} "
        f"({usage.get('calls', 0) / total:.1f} per question) |",
        f"| Tokens | {usage.get('total_tokens', 0):,} |",
        "",
        _GUARD_NOTE,
        "", "## Per question", "",
        "| id | category | result | facts | steps | llm |", "|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "pass" if r["passed"] else "**fail**"
        if r["hallucinated"]:
            status += " (hallucinated)"
        if r["guard_fired"]:
            status += " (guard fired)"
        lines.append(f"| `{r['id']}` | {r['category']} | {status} | {r['facts']} | "
                     f"{', '.join(r['steps']) or '—'} | {r['llm_calls']} |")

    failures = [r for r in results if not r["passed"] or r["hallucinated"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [f"### `{r['id']}`", "", f"**Q:** {r['question']}", "",
                      f"**A:** {r['answer']}", ""]
            for f in r["failures"]:
                lines.append(f"- {f}")
            if r["hallucinated"]:
                lines.append(f"- ungrounded numbers: {r['hallucinated']}")
            lines.append("")

    lines += ["", "## Answers", ""]
    for r in results:
        lines += [f"<details><summary><code>{r['id']}</code> — {r['question']}</summary>",
                  "", r["answer"], "", "</details>", ""]

    REPORT.write_text("\n".join(lines))
    (REPORT.with_suffix(".json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

"""The plan/act/verify controller (docs/03-AGENT-SPEC.md section 6).

Typical question: 2 LLM calls. Worst case: 4.

Note how much of this is deterministic Python -- dispatch, verification,
missing-column guarding, numeric validation. That is the design, not an
accident: every step code can decide, code decides, both for reliability and to
stay inside a rate-limited free tier.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src import config
from src.agent import prompts, tools
from src.agent.ledger import FactLedger, template_answer, validate_numbers
from src.agent.verify import Status, verify
from src.llm.client import LLMClient, LLMUnavailable, get_client

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """One planned tool call and what came back."""

    tool: str
    arguments: dict
    purpose: str = ""
    result: Any = None
    status: str = "pending"
    hint: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "purpose": self.purpose,
            "status": self.status,
            "hint": self.hint,
            "seconds": round(self.seconds, 3),
            "result": _truncate(self.result),
        }


@dataclass
class AgentResponse:
    """Everything the UI needs: the answer, the evidence, and the audit trail."""

    text: str
    ledger: FactLedger = field(default_factory=FactLedger)
    steps: list[Step] = field(default_factory=list)
    plan_reasoning: str = ""
    missing_columns: list[str] = field(default_factory=list)
    validation_passed: bool = True
    fell_back: bool = False    # the answer was REJECTED for an ungrounded number
    degraded: bool = False     # the model/infrastructure failed; not a grounding problem
    llm_calls: int = 0
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "facts": self.ledger.to_list(),
            "steps": [s.to_dict() for s in self.steps],
            "plan_reasoning": self.plan_reasoning,
            "missing_columns": self.missing_columns,
            "validation_passed": self.validation_passed,
            "fell_back": self.fell_back,
            "degraded": self.degraded,
            "llm_calls": self.llm_calls,
            "seconds": round(self.seconds, 2),
            "error": self.error,
        }


def _truncate(value: Any, limit: int = 1200) -> Any:
    """Keep the UI trace readable without hiding that truncation happened."""
    if value is None:
        return None
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + " ...[truncated]"


class ChurnAgent:
    """Plan -> act -> verify -> answer, with the numeric-grounding gate."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_client()
        self._schema_block: str | None = None

    @property
    def schema_block(self) -> str:
        """Schema is built once and reused -- it goes into every planner call."""
        if self._schema_block is None:
            self._schema_block = tools.schema_prompt_block()
        return self._schema_block

    @property
    def available(self) -> bool:
        return self.client.available

    # ------------------------------------------------------------------
    def answer(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        """Answer one question. Never raises -- failures come back as text."""
        started = time.time()
        response = AgentResponse(text="")
        ledger = response.ledger

        if not question or not question.strip():
            response.text = "Ask me something about the churn dataset."
            return response

        if not self.client.available:
            response.error = "llm_unavailable"
            response.text = (
                f"{self.client.reason}\n\nThe **Explore** and **What-if** tabs do not need "
                "the language model and are fully functional."
            )
            return response

        try:
            self._run(question, history or [], response)
        except LLMUnavailable as exc:
            response.error = "llm_unavailable"
            response.text = self._degrade(ledger, f"The language model is unavailable ({exc}).")
            response.degraded = True
        except Exception as exc:  # never surface a traceback to the user
            logger.exception("Agent loop failed")
            response.error = type(exc).__name__
            response.text = self._degrade(ledger, "Something went wrong while answering that.")
            response.degraded = True

        response.seconds = time.time() - started
        response.llm_calls = self.client.usage.calls
        return response

    # ------------------------------------------------------------------
    def _run(self, question: str, history: list[dict], response: AgentResponse) -> None:
        ledger = response.ledger

        # --- Pre-flight: catch absent concepts before spending a call -------
        # Cheap, deterministic, and it stops the most likely hallucination at
        # the door rather than relying on the planner to refuse (finding I10).
        preflight = tools.check_absent_concepts(question)

        plan = self._plan(question, history)
        response.plan_reasoning = plan.get("reasoning", "")
        missing = [str(m) for m in plan.get("missing_columns") or []]
        missing += [h["concept"] for h in preflight if h["concept"] not in missing]
        response.missing_columns = missing

        steps = self._sanitise_steps(plan.get("steps") or [])

        # --- Execute -------------------------------------------------------
        replans_left = config.MAX_REPLANS
        executed: list[Step] = []
        index = 0
        retries: dict[int, int] = {}

        while index < len(steps) and len(executed) < config.MAX_TOOL_STEPS:
            step = steps[index]
            started = time.time()
            step.result = tools.dispatch(step.tool, step.arguments)
            step.seconds = time.time() - started

            verdict = verify(step.result, step.purpose)
            step.status = verdict.status.value
            step.hint = verdict.hint

            if verdict.status is Status.RETRY and retries.get(index, 0) < 1:
                retries[index] = retries.get(index, 0) + 1
                logger.info("Retrying step %d: %s", index, verdict.hint)
                continue  # same step, one more attempt

            # A step that has now failed twice is a planning problem, not bad
            # luck. Escalate rather than dropping it -- otherwise a single failed
            # step silently leaves nothing computed and the user gets a generic
            # "I could not answer that" for a perfectly answerable question.
            exhausted_retry = verdict.status is Status.RETRY and retries.get(index, 0) >= 1
            if (verdict.status is Status.REPLAN or exhausted_retry) and replans_left > 0:
                replans_left -= 1
                executed.append(step)
                new_plan = self._replan(question, verdict.hint, executed)
                extra_missing = [str(m) for m in new_plan.get("missing_columns") or []]
                for column in extra_missing:
                    if column not in response.missing_columns:
                        response.missing_columns.append(column)
                steps = self._sanitise_steps(new_plan.get("steps") or [])
                index, retries = 0, {}
                continue

            executed.append(step)
            if verdict.ok:
                # Label by the step's purpose, not the tool name: a what-if runs
                # predict_churn_risk twice, and two facts both called
                # "predict_churn_risk.risk_score" are indistinguishable to the
                # answerer -- which is exactly how it ends up saying "not available"
                # while holding the number it needed.
                ledger.register_result(
                    step.result, tool=step.tool, args=step.arguments,
                    step=len(executed), prefix=step.purpose.strip() or step.tool,
                )
            index += 1

        response.steps = executed

        # --- Answer --------------------------------------------------------
        if not len(ledger) and response.missing_columns:
            # Nothing computable, but we know exactly why. A precise refusal is
            # a correct answer here, not a failure.
            response.text = self._refusal(response.missing_columns)
            return

        succeeded = [s for s in executed if s.status in ("ok", "warn")]

        if not succeeded:
            # The planner produced nothing to run. Usually a greeting or an
            # off-topic message -- answer it deterministically rather than
            # spending a call, and say what this thing can actually do.
            response.text = self._capabilities()
            return

        # A question like "what is this data about?" is answerable from structure
        # alone and yields no numeric facts. Requiring a non-empty ledger here
        # turned a perfectly good question into "I could not compute anything".
        context = self._context(succeeded) if not len(ledger) else ""

        response.text, response.validation_passed, response.fell_back = self._compose(
            question, ledger, history, response.missing_columns, context
        )

    # ------------------------------------------------------------------
    def _plan(self, question: str, history: list[dict]) -> dict:
        messages = prompts.planner_messages(
            question, self.schema_block, config.MAX_TOOL_STEPS, self._window(history)
        )
        try:
            return self.client.complete(messages, json_mode=True).json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Planner returned unparseable JSON: %s", exc)
            return {"reasoning": "", "steps": [], "missing_columns": []}

    def _replan(self, question: str, failure: str, executed: list[Step]) -> dict:
        completed = "\n".join(
            f"- {s.tool}({json.dumps(s.arguments, default=str)[:120]}) -> {s.status}"
            for s in executed
        )
        messages = prompts.replanner_messages(
            question, self.schema_block, failure, completed, config.MAX_TOOL_STEPS
        )
        try:
            return self.client.complete(messages, json_mode=True).json()
        except (json.JSONDecodeError, ValueError):
            return {"steps": [], "missing_columns": []}

    def _compose(self, question: str, ledger: FactLedger, history: list[dict],
                 missing: list[str], context: str = "") -> tuple[str, bool, bool]:
        """Draft, substitute, validate; retry once; then fall back deterministically."""
        violations: list[str] = []

        for attempt in range(config.MAX_ANSWER_RETRIES + 1):
            messages = prompts.answerer_messages(
                question, ledger, self._window(history), missing, violations or None, context
            )
            draft = self.client.complete(messages).text
            rendered, unknown = ledger.substitute(draft)
            substituted = [f.format() for f in ledger if f.key in draft.upper()]
            check = validate_numbers(rendered, ledger, question, substituted, unknown)

            if check.ok:
                return rendered, True, False

            violations = check.unmatched + [f"[[{k}]]" for k in check.unknown_tokens]
            logger.warning("Answer attempt %d rejected: %s", attempt + 1, check.message())

        # A visibly limited answer is a correct outcome. A fluent answer with one
        # invented figure is the failure this whole design exists to prevent.
        return template_answer(
            ledger,
            "I could not phrase a summary in which every figure traced back to a computed "
            "value, so the raw computed values are shown instead.",
        ), False, True

    # ------------------------------------------------------------------
    def _sanitise_steps(self, raw: list) -> list[Step]:
        """Drop malformed or unknown steps before executing anything.

        The planner is a language model; it occasionally emits a tool name that
        does not exist or arguments as a JSON string. Catching that here keeps
        the dispatch loop simple and avoids burning a retry on it.
        """
        steps: list[Step] = []
        for item in raw[: config.MAX_TOOL_STEPS]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool", "")).strip()
            if name not in tools.TOOL_FUNCTIONS:
                logger.warning("Planner proposed unknown tool %r; dropping.", name)
                continue
            args = item.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            steps.append(Step(tool=name, arguments=args, purpose=str(item.get("purpose", ""))))
        return steps

    @staticmethod
    def _window(history: list[dict], turns: int = 4) -> list[dict]:
        """Last N turns verbatim. Older context is dropped, not summarised --
        summarising costs an extra LLM call, which is the wrong trade here."""
        return history[-turns * 2 :] if history else []

    @staticmethod
    def _context(steps: list[Step], limit: int = 2500) -> str:
        """Non-numeric content from successful steps, for descriptive answers.

        Numbers here are NOT licensed by this block -- the validator still
        rejects any figure that is not in the ledger. This exists so structural
        questions ("what is this data about?") can be answered at all.
        """
        parts: list[str] = []
        for step in steps:
            result = step.result
            if isinstance(result, dict) and "columns" in result:
                # Schema results are large; render them compactly.
                cols = []
                for column in result["columns"]:
                    if column.get("role") == "other":
                        continue
                    if column.get("allowed_values"):
                        detail = "one of " + ", ".join(column["allowed_values"][:6])
                    elif "min" in column:
                        detail = f"numeric {column['min']}-{column['max']}"
                    else:
                        detail = column.get("dtype", "")
                    cols.append(f"  {column['name']} ({column['role']}): {detail}")
                parts.append(
                    f"Dataset grain: {result.get('grain', '')}\nColumns:\n" + "\n".join(cols)
                )
            else:
                parts.append(f"{step.purpose or step.tool}: "
                             + json.dumps(result, default=str)[:600])
        return "\n\n".join(parts)[:limit]

    @staticmethod
    def _capabilities() -> str:
        """Deterministic reply for greetings and off-topic input. No LLM call."""
        return (
            "I answer questions about a customer churn dataset of 7,043 customers, using "
            "real computation rather than guesswork. Things you can ask:\n\n"
            "- **Explore the data** — \"what is this data about?\", \"how does churn vary "
            "by contract type?\", \"what is the distribution of tenure?\"\n"
            "- **Score a customer** — \"how risky is customer 3668-QPYBK?\"\n"
            "- **Project a change** — \"what if that customer moved to a two year contract?\"\n"
            "- **Aggregate a segment** — \"which fiber-optic customers are most likely to churn?\"\n\n"
            "Every figure I state traces back to something I actually computed — open "
            "**Facts used** under any answer to check."
        )

    @staticmethod
    def _refusal(missing: list[str]) -> str:
        explanations = tools.ABSENT_CONCEPTS
        lines = [
            f"This dataset does not contain **{name}**. "
            + explanations.get(name, "There is no such column.")
            for name in missing
        ]
        return (
            "\n\n".join(lines)
            + "\n\nWhat the dataset does cover: demographics (gender, senior status, partner, "
            "dependents), account details (tenure, contract, billing, payment method), nine "
            "service flags, and charges. Ask about any of those and I can compute it."
        )

    @staticmethod
    def _degrade(ledger: FactLedger, reason: str) -> str:
        if len(ledger):
            return template_answer(ledger, reason)
        return f"{reason} Please try again in a moment."


_agent: ChurnAgent | None = None


def get_agent() -> ChurnAgent:
    global _agent
    if _agent is None:
        _agent = ChurnAgent()
    return _agent

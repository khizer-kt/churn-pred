"""Critic pass: check the finished answer against the data before showing it.

The numeric validator in ledger.py guarantees every figure traces to a real
computation. It cannot catch a wrong *claim* built from right numbers -- the
answer that correctly reports 26.9% versus 26.2% and then calls the difference
meaningful when it is well inside sampling noise. Every figure is grounded and
the conclusion is still wrong.

That is the gap this covers. The critic re-reads the finished answer against the
facts that were computed and the question that was asked, and asks whether the
data actually supports what is being claimed.

Deliberately narrow. It reviews interpretation, never arithmetic: a critic
invited to recompute would start inventing numbers, which is the failure the
rest of this system exists to prevent.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src import config
from src.agent.ledger import FactLedger
from src.llm.client import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)


@dataclass
class Critique:
    """Outcome of one review. `checked=False` means the critic did not run."""

    ok: bool = True
    issues: list[str] = field(default_factory=list)
    checked: bool = False
    skipped_because: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": self.issues,
            "checked": self.checked,
            "skipped_because": self.skipped_because,
        }

    def as_feedback(self) -> str:
        return "\n".join(f"- {issue}" for issue in self.issues)


def should_review(answer_text: str, ledger: FactLedger, fell_back: bool) -> tuple[bool, str]:
    """Decide whether a review is worth an LLM call.

    Each skip is a request not spent against a rate-limited free tier, and none
    of these cases has an interpretation to get wrong.
    """
    if not getattr(config, "ENABLE_CRITIC", True):
        return False, "critic disabled in config"
    if fell_back:
        # The deterministic template states computed values and draws no
        # conclusions, so there is nothing to overstate.
        return False, "answer is the deterministic fallback"
    if not len(ledger):
        # A refusal or a capabilities reply makes no data claims.
        return False, "no computed facts to check a claim against"
    if len(answer_text.split()) < 8:
        return False, "answer too short to carry an unsupported claim"
    if not _makes_a_claim(answer_text):
        # A plain statement of computed values has no interpretation to dispute.
        # This filter is deterministic and free; skipping these answers cut the
        # critic's share of requests roughly in half in testing, which matters on
        # a tier with a daily token cap.
        return False, "answer states values without interpreting them"
    return True, ""


# Language that turns numbers into a claim: comparisons, superlatives, causal
# assertions, and magnitude judgements. Deliberately broad -- a missed review is
# cheaper than a wasted request, and the answer is still numerically verified.
_CLAIM_WORDS = re.compile(
    r"\b("
    r"more|less|higher|lower|greater|worse|better|larger|smaller|fewer|differ\w*"
    r"|most|least|highest|lowest|worst|best|largest|smallest"
    r"|than|versus|vs|compared"
    r"|caus\w*|drive\w*|lead\w*\s+to|because|due\s+to|result\w*\s+in|influenc\w*"
    r"|significant\w*|substantial\w*|marked\w*|strong\w*|dramatic\w*|slight\w*"
    r"|modest\w*|notable|considerabl\w*|far\s+more|much\s+"
    r"|suggest\w*|indicat\w*|imply|implies|shows?\s+that|means?\s+that"
    r"|trend\w*|increas\w*|decreas\w*|declin\w*|grow\w*"
    # "risk of" is deliberately absent: "a churn risk of 29.2%" is a plain
    # statement of a computed value, and it is the single most common phrasing
    # in this domain. Including it sent nearly every prediction answer to review.
    r"|likely|unlikely"
    r")\b",
    re.I,
)


def _makes_a_claim(text: str) -> bool:
    return bool(_CLAIM_WORDS.search(text))


def review(
    question: str,
    answer_text: str,
    ledger: FactLedger,
    client: LLMClient,
    fell_back: bool = False,
) -> Critique:
    """Review one answer. Never raises: a broken critic must not block an answer.

    A critic that fails closed would turn a working system into a broken one on
    every provider hiccup, which is a worse failure than an unreviewed answer.
    """
    from src.agent import prompts  # imported here to avoid a circular import

    proceed, reason = should_review(answer_text, ledger, fell_back)
    if not proceed:
        return Critique(ok=True, checked=False, skipped_because=reason)

    try:
        response = client.complete(
            prompts.critic_messages(question, answer_text, ledger), json_mode=True
        )
        payload = response.json()
    except LLMUnavailable as exc:
        logger.warning("Critic unavailable: %s", exc)
        return Critique(ok=True, checked=False, skipped_because=f"critic unavailable: {exc}")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Critic returned unparseable JSON: %s", exc)
        return Critique(ok=True, checked=False, skipped_because="critic response unparseable")
    except Exception as exc:  # noqa: BLE001 - the critic must never break the turn
        logger.warning("Critic failed: %s", exc)
        return Critique(ok=True, checked=False, skipped_because=f"critic error: {exc}")

    verdict = str(payload.get("verdict", "ok")).strip().lower()
    issues = [str(i).strip() for i in (payload.get("issues") or []) if str(i).strip()]

    # "revise" with no stated issue is not actionable -- the analyst would have
    # nothing to act on, so treat it as a pass rather than burn a retry.
    if verdict == "revise" and not issues:
        return Critique(ok=True, checked=True,
                        skipped_because="critic asked for revision without naming an issue")

    return Critique(ok=verdict != "revise", issues=issues, checked=True)

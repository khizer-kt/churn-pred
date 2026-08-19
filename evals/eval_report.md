# Evaluation report

> **This report needs regenerating.** The last complete run predates the critic
> pass; the run after it was corrupted part-way through by exhausting the
> provider's daily token quota, so its figures describe a rate-limited system
> rather than this one. Regenerate with `python -m evals.run_eval` once quota
> has reset, and commit the result.

## Last clean measurement (before the critic pass)

`16` questions with known-correct answers, ground truth recomputed from the cleaned dataset.

| Metric | Result |
|---|---|
| Accuracy | **16/16 (100%)** |
| Hallucination rate | **0/16 (0%)** |
| Guard activations | 0 |
| Model | `openai/gpt-oss-120b` |
| LLM calls | 30 (1.9 per question) |
| Tokens | 39,864 |

**Hallucination rate** counts answers containing a figure that traces back to
nothing the agent computed.

**Guard activations** counts drafts the validator rejected before the user would
have seen them. It is **0** here, and that is worth stating precisely rather than
claiming credit for it: on these 16 questions the citation-token layer was
sufficient on its own, so there was nothing for the validator to reject. What
proves the validator works is `tests/test_ledger.py`, where fabricated figures
are rejected on demand, and `tests/test_loop.py`, where two bad drafts in a row
force the deterministic fallback.

## What the corrupted run did surface

Running the eval with the critic enabled exhausted the free tier's daily token
cap, and two real defects fell out of it:

- The provider's rate-limit error reached the user verbatim, carrying internal
  identifiers and token counters. Those stray digits were also, correctly,
  unverifiable figures inside an answer. Provider errors now go to the log and
  the user gets a sentence.
- A daily cap was being retried like a per-minute burst limit, spending four
  backoffs to reach the same error. Daily exhaustion now fails fast.

Both are fixed and covered by tests. The pending re-run is to measure the
critic's effect on accuracy and on calls per question.

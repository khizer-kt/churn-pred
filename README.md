# Churn Analyst Agent

An agent that answers natural-language questions about a customer churn dataset, and a churn model it calls as a tool. It plans multi-step, runs real computation against the dataframe, checks its own results, and **refuses to state a number it did not compute**.

Built for the Adept Tech Solutions AI Engineer assessment.

```bash
git clone https://github.com/khizer-kt/churn-pred.git && cd churn-pred
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add a free Groq key from console.groq.com
python -m src.model.train     # ~30s, writes artifacts/
streamlit run ui/app.py
```

## Submission links

| | |
|---|---|
| **Live app** | **https://churn-pred-100.streamlit.app/** |
| **Colab notebook** | **[Open in Colab](https://colab.research.google.com/github/khizer-kt/churn-pred/blob/main/notebooks/churn_eda_and_training.ipynb)** — committed with outputs, so the analysis is readable without running it |
| **Repository** | https://github.com/khizer-kt/churn-pred |

> The hosted app sleeps after inactivity and takes ~30s to wake on first request — a slow first load is a cold start, not a broken app.

**Contents** — [What it does](#what-it-does) · [Architecture](#architecture) · [Data issues](#the-data-issues-i-found) · [Model and metric](#the-model-and-why-pr-auc) · [The agent](#the-agent-planning-and-verification) · [Results](#results) · [Running it](#running-it) · [Limitations](#limitations-and-what-id-do-with-more-time) · [AI use](#ai-tool-use)

---

## What it does

Four things the brief asked for, all live-wired to the model and the agent:

| | Example |
|---|---|
| **Exploratory questions** | *"How does churn vary by contract type?"* |
| **Score an individual or hypothetical customer** | *"What is the churn risk for 3668-QPYBK?"* |
| **Project risk under changed conditions** | *"What if they moved to a two-year contract?"* |
| **Aggregate risk across a segment** | *"Which fiber-optic customers are most likely to churn?"* |

The Streamlit app has three tabs. **Chat** is the agent. **Explore** surfaces the EDA and the data-cleaning findings directly, with no LLM involved. **Score a customer** handles individual and hypothetical scoring with a what-if panel. The latter two work without an API key, which is also the degradation path when the language model is unavailable.

Every answer carries two expanders: the step trace (each tool call, its arguments, its verification verdict) and the fact ledger (every number, with the tool that produced it). The grounding claim is checkable, not asserted.

## Architecture

```
                    ┌──────────────────────────┐
 user question ───► │  Streamlit chat (ui/)    │
                    └───────────┬──────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │  Agent loop (src/agent/) │  plan → act → verify → answer
                    │  + Fact Ledger           │  every computed number lands here
                    └───────────┬──────────────┘
        ┌─────────────┬─────────┴──────┬──────────────────┐
        ▼             ▼                ▼                  ▼
   get_schema   get_distribution   run_analysis   predict_churn_risk
   (columns,    (distributions,    (restricted    predict_segment_risk
    domains,     churn per level)   pandas exec)  (+ what-if overrides)
    absences)
        └─────────────┴────────────────┴──────────────────┘
                                ▼
              cleaned dataframe + fitted sklearn pipeline
```

Two rules hold this together:

- **The LLM never sees the raw dataframe.** It sees the schema and tool results. It cannot read off a number it did not explicitly request.
- **The model tool calls the dataset tool.** `top_factors` attaches a population statistic to each driver — *"Month-to-month customers churn at 42.7% vs 26.5% overall"* — so an explanation is anchored in the real distribution rather than a bare coefficient.

## The data issues I found

The brief did not say what was wrong. Profiling found these; each is handled by a separately named, separately tested function in `src/data/cleaning.py`.

### `TotalCharges` is a text column with 11 disguised nulls

Not `NaN`, not empty — a **single space character**. So `isnull()` reports **zero missing** and `astype(float)` throws. This is the one that bites: a routine missing-value check finds nothing at all.

All 11 rows have `tenure == 0`. These are customers who signed up but have not been billed a cycle, so the true value is **0, not the median** — mean or median imputation would invent a billing history that never happened. The cleaning function asserts that tenure relationship still holds and refuses to zero-fill if it ever stops.

### Seven columns carry perfectly collinear sentinel levels

`MultipleLines` has `"No phone service"` (682 rows); six add-on columns have `"No internet service"` (1,526 rows each). Both are **100% determined** by `PhoneService` / `InternetService` — verified at **zero violations in either direction**.

Left alone, one-hot encoding emits seven dummy columns that duplicate each other and `InternetService_No`, producing a degenerate design matrix and unstable coefficients — which matters here because per-customer attribution is computed *from* those coefficients. They collapse to `"No"`; the information survives in the parent columns.

### 42 rows carry contradictory labels — kept deliberately

Eighteen groups of rows are identical across all 19 features but disagree on `Churn`. Every one is `tenure = 1`, `Contract = Month-to-month`.

This is **irreducible (Bayes) error, not corruption**: two customers with identical observable attributes genuinely made different decisions, and the dataset lacks whatever would separate them. They are counted and left in place. Removing label noise from training data teaches a model a certainty the world does not support and inflates the apparent score. It also sets a ceiling — a model reporting near-perfect separation here is leaking, not learning.

Twenty-two near-duplicate rows (identical once `customerID` is ignored) are likewise kept: with 19 mostly-categorical fields, collisions among short-tenure customers on minimal plans are expected, and each has a distinct valid ID.

### `SeniorCitizen` uses a different encoding from every other flag

Stored `0`/`1` while `Partner`, `Dependents`, `PhoneService` and `PaperlessBilling` all use `No`/`Yes`. Two problems: pandas infers `int64` and it silently lands in the numeric branch of the `ColumnTransformer`; and the agent has no way to know `1` means senior unless the schema says so in words. Recoded to `No`/`Yes`.

### `TotalCharges` is only approximately `tenure × MonthlyCharges`

Median relative deviation is 0.000, but p95 is +7.5% and the maximum is +57%, with 59 rows off by more than 20%. `MonthlyCharges` is the *current* rate while `TotalCharges` accumulates historical rates across plan changes — so it is not redundant, it encodes billing history. It is also 0.83 correlated with tenure, which is why an `avg_monthly_spend` feature is derived to separate historical from current rate (guarding the divide-by-zero at `tenure == 0`).

### The dataset has none of the columns the brief's examples ask about

The sample questions mention **region**, **revenue trend**, and **product category**. There is no geography column, **no date column at all** (so no trend is computable — the data is a single snapshot), and no product-category column.

I treated this as a deliberate hallucination trap and designed for it: see [the missing-column guard](#4-refusing-what-cannot-be-computed).

### Ruled out

No whitespace padding, no negative or impossible values, no mixed-case category variants, no fully duplicated rows including ID, and `PhoneService`↔`MultipleLines` / `InternetService`↔add-ons are internally consistent at zero violations.

## The model, and why PR-AUC

`src/model/train.py` trains four candidates and selects between them. Deployed: **unweighted logistic regression**.

| Candidate | PR-AUC | ROC-AUC | Brier | Max calibration gap |
|---|---|---|---|---|
| `dummy` (prior) | 0.2653 | 0.4999 | 0.1949 | — |
| **`logistic_regression`** | **0.6560** | 0.8450 | **0.1353** | 0.118 |
| `logistic_regression_balanced` | 0.6544 | 0.8449 | 0.1654 | 0.263 |
| `gradient_boosting` | 0.6513 | 0.8420 | 0.1368 | 0.046 |

Held-out test set: **PR-AUC 0.6358, ROC-AUC 0.8424, Brier 0.1379, recall 0.77, precision 0.52 at threshold 0.28.**

### Why not accuracy

Predicting "nobody churns" scores **73.46%**. Any metric whose no-skill baseline is that high cannot distinguish a useful model from an inert one.

### Why PR-AUC as the primary

Two things about *this* system decide it.

**Ranking matters more than any fixed threshold.** The agent's real questions — "who is most likely to churn", "which segment is riskiest" — are ordering problems, so the headline metric has to be threshold-free. Unlike ROC-AUC, PR-AUC is not flattered by the 73% negative class: its baseline is the churn rate (0.265), so the lift is honest. The deployed 0.636 is a 2.4× lift.

**Calibration is a correctness requirement, not a nicety.** The model is not a decision system — it is a tool an LLM calls, whose output the agent *states to a user as a number*. If it says 65% for a group that churns 33% of the time, the agent reports a figure that is computed but wrong. That defeats the brief's central requirement as surely as a hallucinated number does, and arguably worse: it arrives with an audit trail that makes it look verified.

So PR-AUC selects, and **calibration gates**: among candidates tied on ranking, the best-calibrated wins.

### The finding that changed the design

My first run used `class_weight="balanced"` — the reflexive choice for imbalanced data. Its reliability curve **over-predicted in every single bin**:

```
predicted 0.65 → observed 0.33      predicted 0.45 → observed 0.27
predicted 0.55 → observed 0.32      predicted 0.25 → observed 0.11
```

Balancing scales the positive class by ~3.8×, inflating every probability. For a system emitting hard labels that is harmless — the threshold absorbs it. Here it was a correctness bug.

The fix separates the two concerns: **unweighted logistic regression** for calibrated probabilities, with the asymmetric cost of a missed churner living entirely in the **threshold**. Ranking was unaffected (PR-AUC 0.6544 → 0.6560, marginally *better*) while Brier improved 0.1654 → 0.1353 and the worst calibration gap halved. Reweighting bought nothing and cost a great deal.

Post-fix reliability, 7 of 9 populated bins within 0.05:

| Predicted | 0.035 | 0.149 | 0.246 | 0.347 | 0.454 | 0.547 | 0.650 | 0.740 |
|---|---|---|---|---|---|---|---|---|
| **Observed** | 0.035 | 0.184 | 0.262 | 0.318 | 0.336 | 0.546 | 0.701 | 0.738 |

Independent confirmation: for a calibrated model the cost-optimal threshold is `1/(1+r)`. Observed thresholds track that almost exactly — r=1 → 0.54 (theory 0.50), r=3 → 0.28 (0.25), r=10 → 0.08 (0.09). A miscalibrated model would not.

### The threshold, and honesty about it

Not 0.5. Chosen by minimising expected cost, where the ratio is stated as its components rather than guessed:

```
COST_FN_OVER_FP = (CLV_MARGIN × OFFER_SUCCESS_RATE) / OFFER_COST
                = ($500 × 0.30) / $50 = 3.0
```

The `OFFER_SUCCESS_RATE` term is the one usually omitted. Catching a churner is only worth what the intervention recovers — an offer that works 30% of the time recovers 30% of the margin, not all of it. Leaving it out inflates the ratio to 10:1, which drove my first run to a threshold of 0.18: **flagging 68% of the entire customer base**, an operationally meaningless recommendation.

**This assumption is load-bearing and the sensitivity analysis does not let it off the hook:**

| Cost ratio | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Threshold | 0.54 | **0.28** | 0.17 | 0.08 | 0.05 |

Anyone deploying this should substitute their own three numbers; it is a one-line change in `src/config.py`.

### Why logistic regression over gradient boosting

All three real candidates land within 0.005 PR-AUC, so ranking does not decide it. LR wins on two grounds: it is better calibrated on Brier, and it gives `top_factors` an **exact** decomposition. In log-odds space each feature's contribution to *this customer's* score is `coef_j × (x_ij − mean_j)`, and the contributions provably sum to the predicted logit — asserted in `tests/test_model.py::test_attribution_sums_to_the_logit`. Since the agent states those factors to users as claims, an explanation that cannot disagree with the prediction is worth more than 0.005 PR-AUC.

Global `feature_importances_` is deliberately not used: it says what matters on average; the brief asks what drives *this* customer.

## The agent: planning and verification

A hand-written **plan → act → verify → answer** loop (`src/agent/loop.py`), using JSON-mode prompting against Groq. No LangChain.

That is a deliberate choice: a framework would hide exactly the machinery being assessed, and `create_pandas_dataframe_agent` ships a `python_repl` with none of the grounding guarantees below. The cost, stated honestly: no free streaming, callbacks or tracing, and retry logic written by hand.

### 1. Multi-step planning

The planner emits an **ordered step list in one call**, rather than one tool per round-trip. It costs some adaptivity and saves a request per step — the right trade on a rate-limited free tier. Re-planning covers the cases where a fixed plan turns out wrong.

*"Which customers are most likely to churn, and does that relate to contract type?"* produces a segment-risk call **and** a distribution call, then combines them.

### 2. Real computation as a tool

`run_analysis` executes model-written pandas against the dataframe. The check is an **AST allowlist, not a string blacklist** — blacklisting `"import"` or `"__"` is trivially evaded by getattr chains or unicode escapes and gives false confidence. Imports, dunder access, file I/O, `eval`/`exec`, and `while` loops are rejected structurally; execution gets a 5s wall clock, safe builtins only, and a **per-call copy of the frame** so generated code cannot corrupt shared state.

Results are truncated to 50 rows before reaching the model, always with the true row count attached so it knows a sample is a sample.

### 3. Self-checking

Every tool result passes `verify()` before it can enter the ledger — all deterministic, no LLM call:

| Check | Action |
|---|---|
| Tool returned an error | Retry with the message; escalate to re-plan on a second failure |
| Empty result | Warn — "no customers match" can be the true answer |
| Probability outside `[0,1]` | Hard fail — pipeline bug, surfaced not hidden |
| Count exceeding 7,043, or negative | Hard fail — the query is double-counting or inverted |
| Segment `mean_risk` vs `actual_churn_rate` off by >0.25 | Warn: possible miscalibration |

Budget: max 6 tool steps, 1 re-plan, 1 answer retry.

### 4. Never inventing a number

The brief's most heavily weighted requirement, enforced **structurally** in two layers rather than requested in a prompt.

**Layer 1 — citation tokens.** Every scalar any tool returns is registered in a Fact Ledger with its provenance. The answering prompt gives the model the ledger and instructs it to write `[[F1]]` where a value belongs and **never to type a digit**. The renderer substitutes real values afterwards. The model cannot fabricate what it does not write.

**Layer 2 — the validator.** Layer 1 depends on instruction-following, which free-tier models do imperfectly. So the rendered answer is scanned: every numeric literal is extracted and matched against the ledger, unit-aware, with tolerance set to the answer's own displayed precision. Numbers echoed from the user's question and small list ordinals are allowed. **Anything unmatched fails.** One retry with the offending figures quoted back; if that also fails, the answer is discarded and replaced by a deterministic table of what was actually computed.

A visibly limited answer is a correct outcome. A fluent answer containing one invented figure is the failure this design exists to prevent.

Layer 2 caught a real bug during live testing that review had not: the agent wrote *"the model predicts actual churn of 100.0%"* about a single customer, and the validator initially **passed** it — a fact whose value was `1` satisfied "100.0%" under blanket ×100 matching. Scale conversion is now restricted to facts that are genuinely ratios, and outcome flags are no longer registered as quotable figures. Regression test: `test_count_cannot_justify_a_percentage`.

### Refusing what cannot be computed

The schema tool publishes what the dataset contains **and an explicit list of what it does not** — region, dates, product category, income, satisfaction. A pre-flight check runs on the question before any computation, and any plan referencing an unknown column is rejected.

Asked *"does churn risk correlate with region?"*, the agent names the absence, says what the data does cover, and offers the nearest legitimate substitute labelled as a substitute. Asked for a revenue trend, it explains there is no date column and offers revenue by tenure cohort with the caveat that it compares different customers at different lifecycle stages rather than following one over time.

### 5. A second opinion on the claim

The validator guarantees every **number** is real. It cannot catch a wrong **claim** built from right numbers — an answer that correctly reports 26.9% versus 26.2% and then calls the difference meaningful when it is inside sampling noise. Every figure grounded, conclusion still wrong.

So a critic pass (`src/agent/critic.py`) re-reads the finished answer against the facts that were computed and asks whether the data supports what is being said. It rejects for overstated differences, comparisons where only one side was computed, causal language applied to association, answering a different question, and claims about columns or time periods that do not exist.

It reviews **interpretation only and never recomputes**. A critic invited to check arithmetic would start producing numbers of its own, which is the failure the rest of the system exists to prevent.

Verified against the case that motivated it. Given a draft asserting that women are *"substantially"* more likely to churn and that being female is a *"clear driver"*, it returned both faults:

> - Overstated difference: the 0.76 percentage-point gap is small and should not be described as substantial
> - Causal language: claiming that being female is a clear driver of churn is not supported by the data

That answer would otherwise have shipped with every figure correctly grounded.

Three rules keep it from doing harm. The rewrite goes **back through the numeric gate** — a revision that fixes the wording and breaks the grounding is not an improvement, and the original is kept instead. It is **never re-reviewed**: one critique and one rewrite, because a reviewer allowed to keep sending work back has no natural stopping point. And it **fails open** — a provider error leaves the answer untouched rather than blocking it, since a broken reviewer must not turn a working system into a broken one.

### Rate limits

Free tiers throttle, and the brief calls efficient design part of the challenge. Measured over the eval set: **2.2 LLM calls and ~2,900 tokens per question**, critic included. The schema lives in the system prompt rather than being fetched; tool results are truncated; distribution questions route to a deterministic tool that needs no generated code; dispatch, verification and numeric checking are all plain Python. One 429 during development was absorbed by exponential backoff honouring `Retry-After`.

The critic adds a request to answers that make a claim, and enabling it naively **exhausted the free tier's daily token cap** part-way through an eval run — the brief's rate-limit point arriving in practice rather than in theory. A deterministic pre-filter now skips answers that state values without interpreting them; *"the overall churn rate is 26.5%"* has no interpretation to dispute.

The effect is measurable: the critic ran on **6 of 16** questions rather than all of them, so the cost of the second opinion was **+0.3 calls per question, not +1.0**. Two further fixes came out of the same run — provider errors no longer reach the user verbatim, and a daily quota is no longer retried like a per-minute burst limit, which had been spending four backoffs to reach an error that would not clear for hours.

## Results

`python -m evals.run_eval` → [`evals/eval_report.md`](evals/eval_report.md)

| Metric | Result |
|---|---|
| Accuracy | **16/16 (100%)** |
| Hallucination rate | **0/16 (0%)** |
| LLM calls | 2.2 per question |

Sixteen questions with ground truth **recomputed from the cleaned dataset**, not transcribed. That mattered: `TechSupport` and `OnlineSecurity` shift once the sentinel level collapses (31.19% and 31.33%, not the pre-cleaning 41.6%/41.8%), so copied figures would have failed against a correct agent.

Two questions are traps for columns that do not exist, where passing means refusing and naming the absence — and a refusal that quotes percentages is graded as a **failure**, since inventing a statistic for a missing column is precisely the behaviour under test.

**Guard activations were 0**, which is worth stating precisely rather than claiming credit for: the citation-token layer was sufficient on this set, so the validator had nothing to reject. What demonstrates the validator works is `tests/test_ledger.py`, where fabricated figures are rejected on demand, and `tests/test_loop.py`, where two bad drafts force the deterministic fallback.

## Running it

**Local** — see the quick start at the top. The app loads `artifacts/pipeline.joblib`; it never trains on startup, and shows an actionable message rather than a traceback if the artifact is missing.

**Docker**

```bash
docker build -t churn-agent .
docker run --rm -p 8501:8501 --env-file .env churn-agent
```

The API key is injected at runtime and never baked into the image. The image **trains the model during build** rather than copying `artifacts/` in — that directory is gitignored, so a `COPY` would fail for anyone cloning fresh; the image has to be buildable from a clean checkout. Training takes ~30s and `random_state` is fixed, so every build produces an identical model.

**Tests**

```bash
python -m pytest tests/ -q      # 161 tests
```

They need no API key and no network: the agent loop tests stub the LLM, and the Streamlit tests drive the real app through `AppTest`. That is deliberate — you cannot make a live model fabricate a number on demand, so the failure branches are only testable against a stub.

## Project structure

```
src/
  config.py            paths, feature groups, cost model, agent limits
  data/                cleaning (C1–C9) and the single place the CSV is read
  model/               feature pipeline, training, callable prediction service
  agent/               executor, tools, fact ledger, verifier, critic, loop, prompts
  llm/client.py        Groq transport: backoff, model resolution, token accounting
ui/                    Streamlit app, session state, three tab components
evals/                 16-question set and the scoring harness
tests/                 161 tests
```

`ui/` imports from `src/`, never the reverse — enforced by `tests/test_no_streamlit_in_src.py`. If the model imported Streamlit it would stop being callable from a notebook, the eval harness or a test, quietly breaking the brief's "don't leave it stuck in a notebook" requirement.

`src/data/loader.py` is the only place the CSV is read, so the notebook, the app and the agent cannot drift apart about what the data is.

## Limitations and what I'd do with more time

**Still no significance testing.** The critic catches an overstated gender gap when the wording is strong enough, but it is a language model judging phrasing, not a statistical test — it has no confidence interval to reason with and will not reliably flag a borderline case. A proper significance tool in the agent's toolset remains the single change I would make first.

**One weak calibration bin.** The 0.4–0.5 band predicts 0.454 against an observed 0.336 (n=119). Gradient boosting calibrates better there; it loses on Brier overall and on exact attribution, so it was not selected, but a mid-range risk figure deserves slightly more caution than the others.

**Multi-turn memory is shallow.** The last four turns are passed verbatim; there is no entity resolution, so *"now break that down by contract"* works only when the previous question is still in the window.

**The critic is a judgement, not a guarantee.** Unlike the numeric validator, which is deterministic and non-negotiable, the critic is a second language model deciding whether a claim is fair. It fails open by design, so a provider outage silently means no review. It catches the clear cases; it is not a proof of anything.

**React frontend not built.** Optional, ~3–4 hours, and it would demonstrate frontend skill rather than the agent engineering under assessment. I chose depth on the graded core instead. Stating that plainly seemed better than shipping a half-finished second UI.

## AI tool use

This project was built with substantial assistance from **Claude (Anthropic)**, used for: profiling the dataset and identifying the data issues; designing and writing the cleaning, model, agent and UI code; writing the test suite; and drafting this README.

The work it did that I would call genuinely load-bearing, rather than autocomplete: catching that `class_weight="balanced"` had destroyed calibration and arguing why that mattered specifically for an agent that quotes probabilities; catching that the numeric validator had a hole letting a count justify a percentage; and insisting the ground truth for the eval set be recomputed after cleaning rather than transcribed.

Several bugs were found by **manual browser testing** rather than by any automated check — a greeting crashing the agent, a structural question being wrongly refused, contradictory error banners, and a dropped click in the chat UI. The full test suite was green while all of those were live, which is the honest limitation of tests written by the same process that wrote the code.

I am able to explain any part of this submission.

## Reflection

**The hardest part** was stopping the model making up numbers. I assumed this would mostly be a prompting problem. It isn't. You can tell a model to only use figures it computed and it will still round 42.71 to "about 43", or drop in a number that sounds about right. What worked was not letting it write numbers at all. Every value a tool computes gets stored with a key, the model writes `[[F1]]` instead of a digit, and the real value is substituted afterwards. A separate check then scans the finished text and rejects any number that doesn't match something computed. If that fails twice the user gets a plain table of values instead of a nice sentence, which I think is the right trade.

That check had a bug I only found by using the app. It let through "actual churn of 100.0%" about a single customer, because a stored value of 1 matches 100 once you allow for percentages. I had read that code several times and never spotted it.

**What I had to learn** was that a model can rank well and still give wrong probabilities. I used `class_weight="balanced"` at first, because that's the standard advice for imbalanced data. The ranking was fine but every probability came out too high, around 65% for a group that actually churned 33% of the time. If you only care who's top of the list that doesn't matter. My agent says the number out loud to the user, so it matters a lot. Removing the class weights fixed the probabilities and the ranking barely moved.

**With more time** I'd add a proper significance test. Ask it now whether men or women churn more and it says women, 26.9% against 26.2%. Both numbers are correct, the gap is meaningless at this sample size, and nothing in the system knows that. I added a second model pass that reviews the wording, which helps, but it's a language model judging phrasing rather than an actual test. I'd also have started clicking around the app much sooner. Almost every UI bug I found came from two minutes of using it, not from the tests.

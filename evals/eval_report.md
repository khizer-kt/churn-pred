# Evaluation report

`16` questions with known-correct answers, ground truth recomputed from the cleaned dataset.

| Metric | Result |
|---|---|
| Accuracy | **15/16 (94%)** |
| Hallucination rate | **1/16 (6%)** |
| Guard activations | 0 |
| Model | `openai/gpt-oss-120b` |
| LLM calls | 39 (2.4 per question) |
| Tokens | 51,337 |

**Hallucination rate** counts answers containing a figure that traces back to nothing the agent computed.

**Guard activations** counts drafts the validator rejected before the user would have seen them. It is **0** here, and that is worth stating precisely rather than claiming credit for it: on these 16 questions the citation-token layer was sufficient on its own — the model wrote `[[F1]]` references instead of digits, so there was nothing for the validator to reject. The validator is therefore *unproven by this run*; what proves it works is `tests/test_ledger.py`, where fabricated figures are rejected on demand, and `tests/test_loop.py`, where two bad drafts in a row force the deterministic fallback. A zero here means the first layer held, not that the second layer is doing the work.

## Per question

| id | category | result | facts | steps | llm |
|---|---|---|---|---|---|
| `overall_churn_rate` | lookup | pass | 10 | get_distribution:ok | 2 |
| `total_customers` | lookup | pass | 1 | run_analysis:ok | 4 |
| `churn_by_contract` | segment | pass | 14 | get_distribution:ok | 7 |
| `worst_payment_method` | segment | pass | 18 | get_distribution:ok | 10 |
| `fiber_optic_churn` | segment | pass | 14 | get_distribution:ok | 13 |
| `senior_citizens` | segment | pass | 10 | get_distribution:ok | 16 |
| `median_tenure` | distribution | pass | 1 | run_analysis:ok | 18 |
| `tenure_by_outcome` | distribution | pass | 5 | run_analysis:ok | 21 |
| `gender_effect` | segment | pass | 10 | get_distribution:ok | 24 |
| `compound_segment` | multi_step | pass | 3 | run_analysis:ok | 27 |
| `highest_risk_and_contract` | multi_step | pass | 25 | run_analysis:ok, get_distribution:ok | 30 |
| `individual_customer` | prediction | pass | 11 | predict_churn_risk:ok | 33 |
| `what_if_contract` | what_if | pass | 24 | predict_churn_risk:ok, predict_churn_risk:ok | 37 |
| `trap_region` | refusal | pass | 0 | — | 38 |
| `trap_revenue_trend` | refusal | pass | 0 | — | 39 |
| `empty_segment` | empty | **fail** (hallucinated) | 0 | — | 39 |

## Failures

### `empty_segment`

**Q:** How many customers have no phone service but do have multiple lines?

**A:** The language model is unavailable (Groq request failed after 4 attempts: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m0cnrz4ken6apn2jmqfw1kvy` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 198637, Requested 2412. Please try again in 7m33.167999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}). Please try again in a moment.

- did not report an empty result
- ungrounded numbers: ['429', '-120', '200000,', '198637,', '2412.', '33.167999999']


## Answers

<details><summary><code>overall_churn_rate</code> — What is the overall churn rate?</summary>

The overall churn rate is 26.5%.

</details>

<details><summary><code>total_customers</code> — How many customers are in this dataset?</summary>

There are 7043 customers in this dataset.

</details>

<details><summary><code>churn_by_contract</code> — What is the churn rate for each contract type?</summary>

The churn rates vary by contract type:

- Month‑to‑month: 42.7%
- One year: 11.3%
- Two year: 2.8%

Overall, the churn is highest for month‑to‑month contracts and lowest for two‑year contracts.

</details>

<details><summary><code>worst_payment_method</code> — Which payment method has the highest churn rate, and how many customers use it?</summary>

The Electronic check payment method has the highest churn rate at 45.3%, and it is used by 2,365 customers.  

- Electronic check: churn 45.3%  
- Mailed check: churn 19.1%  
- Bank transfer (automatic): churn 16.7%  
- Credit card (automatic): churn 15.2%

</details>

<details><summary><code>fiber_optic_churn</code> — What is the churn rate for fiber optic customers?</summary>

The churn rate for fiber‑optic customers is 41.9%.

</details>

<details><summary><code>senior_citizens</code> — How many senior citizens are in the data and do they churn more than non-seniors?</summary>

There are 1,142 senior‑citizen customers in the data. Their churn rate is higher than that of non‑seniors.

- Non‑senior churn rate: 23.6%
- Senior churn rate: 41.7%

</details>

<details><summary><code>median_tenure</code> — What is the median tenure of customers?</summary>

The median tenure of customers is 29.

</details>

<details><summary><code>tenure_by_outcome</code> — What is the average tenure of customers who churned versus those who stayed?</summary>

Age data is not available in this dataset; it contains information on customer tenure, which can be used as an alternative.  
The average tenure differs between customers who churned and those who stayed:

- Churn = 0 (did not churn): average tenure = 37.57  
- Churn = 1 (churned): average tenure = 17.98

</details>

<details><summary><code>gender_effect</code> — Does churn differ between male and female customers?</summary>

Female customers have a slightly higher churn rate than male customers. The male churn rate is 26.2% while the female churn rate is 26.9%.  

- Male churn count: 930  
- Female churn count: 939  

</details>

<details><summary><code>compound_segment</code> — What is the churn rate for month-to-month customers on fiber optic who pay by electronic check, and how many are there?</summary>

The churn rate for month‑to‑month customers on fiber optic who pay by electronic check is 60.4%.  
There are 1,307 such customers.

</details>

<details><summary><code>highest_risk_and_contract</code> — Which customers are most likely to churn, and does that relate to contract type?</summary>

Customers with the highest risk scores – 84.7% and 84.7% – are the most likely to churn.  
Churn varies sharply by contract type, with month‑to‑month contracts showing a much higher churn rate than longer‑term plans.  

- Month‑to‑month: 42.7%  
- One year: 11.3%  
- Two year: 2.8%

</details>

<details><summary><code>individual_customer</code> — What is the churn risk for customer 3668-QPYBK?</summary>

The churn risk for customer 3668‑QPYBK is 29.2%.

</details>

<details><summary><code>what_if_contract</code> — What happens to customer 3668-QPYBK's churn risk if they switch to a two year contract?</summary>

- Current churn risk: 29.2%  
- Predicted churn risk with a two‑year contract: 9.7%  
- Change: -19.5 pp (lower)  

Switching to a two‑year contract would reduce the customer’s churn risk from the current level to the lower predicted level.

</details>

<details><summary><code>trap_region</code> — Does churn risk correlate with region?</summary>

This dataset does not contain **region**. There is no geographic column of any kind -- no region, state, city or country.

What the dataset does cover: demographics (gender, senior status, partner, dependents), account details (tenure, contract, billing, payment method), nine service flags, and charges. Ask about any of those and I can compute it.

</details>

<details><summary><code>trap_revenue_trend</code> — Show me the revenue trend for high-risk customers over the last year.</summary>

This dataset does not contain **date**. There is no date, timestamp, signup-date or churn-date column.

This dataset does not contain **revenue trend**. There is no date column, so no metric can be tracked over time. The data is a single cross-sectional snapshot. The nearest legitimate substitute is revenue by tenure cohort, which compares different customers at different lifecycle stages rather than following one cohort through time.

What the dataset does cover: demographics (gender, senior status, partner, dependents), account details (tenure, contract, billing, payment method), nine service flags, and charges. Ask about any of those and I can compute it.

</details>

<details><summary><code>empty_segment</code> — How many customers have no phone service but do have multiple lines?</summary>

The language model is unavailable (Groq request failed after 4 attempts: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m0cnrz4ken6apn2jmqfw1kvy` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 198637, Requested 2412. Please try again in 7m33.167999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}). Please try again in a moment.

</details>

# Evaluation report

`16` questions with known-correct answers, ground truth recomputed from the cleaned dataset.

| Metric | Result |
|---|---|
| Accuracy | **16/16 (100%)** |
| Hallucination rate | **0/16 (0%)** |
| Guard activations | 0 |
| Model | `openai/gpt-oss-120b` |
| LLM calls | 30 (1.9 per question) |
| Tokens | 39,864 |

**Hallucination rate** counts answers containing a figure that traces back to nothing the agent computed.

**Guard activations** counts drafts the validator rejected before the user would have seen them. It is **0** here, and that is worth stating precisely rather than claiming credit for it: on these 16 questions the citation-token layer was sufficient on its own — the model wrote `[[F1]]` references instead of digits, so there was nothing for the validator to reject. The validator is therefore *unproven by this run*; what proves it works is `tests/test_ledger.py`, where fabricated figures are rejected on demand, and `tests/test_loop.py`, where two bad drafts in a row force the deterministic fallback. A zero here means the first layer held, not that the second layer is doing the work.

## Per question

| id | category | result | facts | steps | llm |
|---|---|---|---|---|---|
| `overall_churn_rate` | lookup | pass | 1 | run_analysis:ok | 2 |
| `total_customers` | lookup | pass | 1 | run_analysis:ok | 4 |
| `churn_by_contract` | segment | pass | 14 | get_distribution:ok | 6 |
| `worst_payment_method` | segment | pass | 18 | get_distribution:ok | 8 |
| `fiber_optic_churn` | segment | pass | 14 | get_distribution:ok | 10 |
| `senior_citizens` | segment | pass | 10 | get_distribution:ok | 12 |
| `median_tenure` | distribution | pass | 1 | run_analysis:ok | 14 |
| `tenure_by_outcome` | distribution | pass | 5 | run_analysis:ok | 16 |
| `gender_effect` | segment | pass | 10 | get_distribution:ok | 18 |
| `compound_segment` | multi_step | pass | 3 | run_analysis:ok | 20 |
| `highest_risk_and_contract` | multi_step | pass | 25 | run_analysis:ok, get_distribution:ok | 22 |
| `individual_customer` | prediction | pass | 11 | predict_churn_risk:ok | 24 |
| `what_if_contract` | what_if | pass | 24 | predict_churn_risk:ok, predict_churn_risk:ok | 26 |
| `trap_region` | refusal | pass | 0 | — | 27 |
| `trap_revenue_trend` | refusal | pass | 0 | — | 28 |
| `empty_segment` | empty | pass | 1 | run_analysis:ok | 30 |

## Answers

<details><summary><code>overall_churn_rate</code> — What is the overall churn rate?</summary>

The overall churn rate is 26.5%.

</details>

<details><summary><code>total_customers</code> — How many customers are in this dataset?</summary>

There are 7043 customers in this dataset.

</details>

<details><summary><code>churn_by_contract</code> — What is the churn rate for each contract type?</summary>

Churn rates differ across contract types.  

- Month‑to‑month: 42.7%  
- Two year: 2.8%  
- One year: 11.3%  

</details>

<details><summary><code>worst_payment_method</code> — Which payment method has the highest churn rate, and how many customers use it?</summary>

Electronic check has the highest churn rate at 45.3%, and it is used by 2,365 customers.  

- Electronic check: 45.3%  
- Mailed check: 19.1%  
- Bank transfer (automatic): 16.7%  
- Credit card (automatic): 15.2%

</details>

<details><summary><code>fiber_optic_churn</code> — What is the churn rate for fiber optic customers?</summary>

The churn rate for fiber‑optic customers is 41.9%.

</details>

<details><summary><code>senior_citizens</code> — How many senior citizens are in the data and do they churn more than non-seniors?</summary>

There are 1,142 senior citizens in the data. Their churn rate is 41.7%, which is higher than the churn rate of non‑senior customers (23.6%).  

- Non‑senior churn rate: 23.6%  
- Senior churn rate: 41.7%

</details>

<details><summary><code>median_tenure</code> — What is the median tenure of customers?</summary>

The median tenure of customers is 29. This means that half of the customers have been with the company for 29 time units or less.

</details>

<details><summary><code>tenure_by_outcome</code> — What is the average tenure of customers who churned versus those who stayed?</summary>

The dataset provides average tenure broken down by churn status.  

- Retained (Churn = 0): 37.57  
- Churned (Churn = 1): 17.98

</details>

<details><summary><code>gender_effect</code> — Does churn differ between male and female customers?</summary>

The churn rate for male customers is 26.2% and for female customers it is 26.9%, showing a modestly higher churn among females.  

- Male: 26.2%  
- Female: 26.9%

</details>

<details><summary><code>compound_segment</code> — What is the churn rate for month-to-month customers on fiber optic who pay by electronic check, and how many are there?</summary>

The churn rate for month‑to‑month customers on fiber optic who pay by electronic check is 60.4%, and there are 1,307 such customers.

</details>

<details><summary><code>highest_risk_and_contract</code> — Which customers are most likely to churn, and does that relate to contract type?</summary>

The ten customers with the highest churn risk are the top‑risk group ( 10 customers) whose risk scores range from 84.7% down to 83.9%.  
Churn likelihood varies sharply by contract type, with the month‑to‑month segment showing the highest churn rate of 42.7%, the one‑year segment at 11.3%, and the two‑year segment the lowest at 2.8%.  

**Churn rate by contract type**

- Month‑to‑month: 42.7%
- One year: 11.3%
- Two year: 2.8%

Thus, customers with the highest risk scores are most likely to churn, and the propensity to churn is strongly linked to having a month‑to‑month contract.

</details>

<details><summary><code>individual_customer</code> — What is the churn risk for customer 3668-QPYBK?</summary>

The churn risk for customer 3668‑QPYBK is 29.2%.

</details>

<details><summary><code>what_if_contract</code> — What happens to customer 3668-QPYBK's churn risk if they switch to a two year contract?</summary>

Switching to a two‑year contract lowers the churn risk from 29.2% to 9.7%, a drop of -19.5 pp.  

- Current risk: 29.2%  
- Two‑year contract risk: 9.7%  

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

There are 0 customers who have no phone service but do have multiple lines. This indicates that none of the customers meet both conditions.

</details>

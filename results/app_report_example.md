# TrialMatch screening report

> **Educational screening prototype, not a clinical enrollment decision.**

- Generated: 2026-11-09T06:55:51+00:00
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter: `/content/genai-assignment-2/checkpoints/adapter`
- Decoding: `cached`
- Generation budget: 128 new tokens
- Preliminary outcome: **Likely eligible**

## Aggregation rule

'not applicable' criteria are ignored; any inclusion 'not included' or exclusion 'excluded' gives 'Likely ineligible'; any 'not enough information' or unparseable label gives 'Needs more information'; only when every inclusion is 'included' and every exclusion is 'not excluded' is the result 'Likely eligible'; an empty applicable set also gives 'Needs more information'.

## Patient note

0. 68-year-old woman with type 2 diabetes since 2010.
1. HbA1c 7.8% on metformin.
2. eGFR 55 mL/min/1.73m2.
3. No history of pancreatitis.

## Per-criterion results

| # | Type | Criterion | Label | Evidence IDs | Evidence text | Explanation |
|---|------|-----------|-------|--------------|---------------|-------------|
| 1 | inclusion | Age 18 years or older | included | 0 | 68-year-old woman with type 2 diabetes since 2010. | The note explicitly satisfies this inclusion criterion; the cited sentences are the evidence. Evidence: "68-year-old woman with type 2 diabetes since 2010." |
| 2 | inclusion | Diagnosed with type 2 diabetes | included | 0 | 68-year-old woman with type 2 diabetes since 2010. | The note explicitly satisfies this inclusion criterion; the cited sentences are the evidence. Evidence: "68-year-old woman with type 2 diabetes since 2010." |
| 3 | exclusion | History of pancreatitis | not excluded | 3 | No history of pancreatitis. | The note does not show that this exclusion criterion applies. Evidence: "No history of pancreatitis." |
| 4 | exclusion | eGFR below 30 mL/min/1.73m2 | not excluded | 2 | eGFR 55 mL/min/1.73m2. | The note does not show that this exclusion criterion applies. Evidence: "eGFR 55 mL/min/1.73m2." |

## Missing information

- none

## Provenance

- Tool: TrialMatch: Educational Trial Screening Prototype
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter: `/content/genai-assignment-2/checkpoints/adapter`
- Decoder: `cached` (greedy, max 128 new tokens)
- Labels are model predictions, not clinical decisions; evidence IDs refer to the numbered patient sentences above.

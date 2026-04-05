# MBE Generation Stage Experiment

## Project Goal
Test whether structured generation-stage prompting interventions reduce
hallucination / improve accuracy on bar exam (MBE) multiple-choice questions
when the correct gold passage is always provided.

Research gap: even with the gold passage given, models fail 25–43% of the
time — a generation-stage ceiling independent of retrieval quality
(Zheng et al., 2025).

## Dataset
- Source: `reglab/barexam_qa` (HuggingFace), from Zheng et al. (2025)
- Location: `data/` subfolder (loaded automatically via glob)
- 1,195 MBE bar exam questions across all splits — ALL used for evaluation
- Key columns:
  - `question`: question text (fact pattern embedded here if no prompt)
  - `prompt`: separate fact pattern (NaN for ~63% of rows)
  - `choice_a/b/c/d`: answer options
  - `answer`: correct answer letter (A/B/C/D)
  - `gold_passage`: the correct supporting passage — ALWAYS PROVIDED
  - `subject`: MBE subject (CONST. LAW, CONTRACTS, CRIM. LAW, EVIDENCE,
               REAL PROP., TORTS, + others)
  - `source`: exam year/batch
  - `split`: train (954) / validation (124) / test (117) — kept for reference only
  - `idx`: unique question ID (e.g. mbe_569)

## Experimental Setup
- Model: gpt-4o-mini (OpenAI API)
- Temperature: 0 for all conditions except self-consistency (0.7)
- Evaluation set: full dataset (1,195 questions, all splits)
- Prompting: zero-shot — no few-shot examples in any condition
- Gold passage always injected into every prompt

## The 8 Conditions

### Condition 0 — Baseline
Single call. Direct answer instruction. No structured reasoning.
Establishes the residual failure rate under zero-shot conditions.

### Condition 1 — Passage-Grounding Constraint
Single call. Explicit instruction to rely ONLY on the passage.
Tests whether post-rationalization can be reduced by instruction alone.
Expected to help on parametric-override questions; may hurt on questions
requiring ruling out wrong answers with knowledge outside the passage.

### Condition 2 — Rule Extraction (2-stage)
Call 1: Extract the legal rule/principle from the passage in one sentence.
Call 2: Use extracted rule + original question to select answer.
Targets oblique passages. Risk: may fabricate on analogical passages where
the passage describes a case rather than a rule.

### Condition 3 — Chain of Logic (CoL)
Single call. IRAC-inspired decomposition (Servantez et al., ACL 2024):
  Step 1: Identify rule from passage
  Step 2: Decompose into elements
  Step 3: Evaluate each element against the facts
  Step 4: Recompose → select answer
Expected to outperform on element-checking questions.

### Condition 4 — Negative Elimination
Single call. For each answer choice: identify the doctrine it invokes,
state whether the passage supports/contradicts it, eliminate or keep.
Select the surviving answer.
Novel condition motivated by MBE structure: correct answers frequently
identified by eliminating three wrong doctrines rather than confirming one.

### Condition 5 — Answer Verification (2-stage)
Call 1: Baseline generation → answer + justification.
Call 2: Check each claim in justification against passage. Flag unsupported
claims. Confirm or revise answer.
Ceiling: cannot verify correctness of analogical inferences, only
passage-consistency.

### Condition 6 — Self-Consistency (N=3)
3 independent calls at temperature=0.7. Majority vote on answer letter.
If tie: fallback to first call's answer.
Limitation: if model makes the same systematic error consistently, N=3
amplifies rather than corrects it.

### Condition 7 — Rule Extraction + CoL (combined)
Call 1: Rule extraction (as Condition 2, Stage 1)
Call 2: CoL applied to the extracted rule (as Condition 3, but rule pre-supplied)
Compounds the strengths of Conditions 2 and 3; also compounds their failure
modes if Stage 1 fabricates.

## Prompt Architecture
- System prompt: role definition only (zero-shot, no examples)
- User prompt: [FACT PATTERN if present] + QUESTION + choices + GOLD PASSAGE
  + condition-specific instruction
- Answer format on every call: "Respond with the answer letter (A, B, C, or D)
  on the first line, then your reasoning."
- The `prompt` column is NaN for ~63% of rows:
  if pd.notna(row['prompt']): prepend "FACT PATTERN:\n{prompt}\n\n"

## API Configuration
- Model: gpt-4o-mini
- Temperature: 0 (except condition 6: 0.7)
- Max tokens: 150 for single-answer conditions (0, 1, 6)
              600 for reasoning conditions (3, 4)
              300 for two-stage conditions (2, 5, 7) per call
- Exponential backoff for 429 errors: wait 2^attempt ± 0.5s jitter, max 5 retries

## Output & Incremental Saving
- Results saved to `results/` subfolder, one CSV per condition
- One row written per question immediately after each API call (safe to interrupt)
- On restart: existing CSV checked; rows with matching `idx` skipped
- CSV columns: idx, subject, source, split, question (100 chars), correct_answer,
  predicted_answer, is_correct, condition, tokens_input, tokens_output,
  raw_response_1, raw_response_2, logprob_A, logprob_B, logprob_C, logprob_D,
  confidence
- Final summary: `results/summary.csv`

## Calibration Measurement (ECE)
ECE (Expected Calibration Error) per Guo et al. (2017), following
Dahl et al. (2024) "Large Legal Fictions", Appendix D.

Formula: ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|
10 equal-width bins by confidence score [0,1]. Lower = better calibrated.
Dahl et al. pooled ECE baseline: 0.453.

### Confidence extraction
- Conditions 0,1,3,4 and final call of 2,5,7: logprobs=True, top_logprobs=4
  on the answer call. Extract logprobs for A/B/C/D from first generated token.
  confidence = softmax(logprob_A, logprob_B, logprob_C, logprob_D)[predicted]
- Condition 6: confidence = votes_for_winner / 3 (no logprobs)

### ECE outputs from evaluator.py
- results/summary.csv: adds ece, mean_confidence, overconfidence_gap columns
- results/ece_bins_{name}.csv: per-bin stats (10 rows per condition)
- results/reliability_{name}.png: reliability diagram per condition

## Evaluation Metrics
- Accuracy: proportion correct (primary)
- McNemar's test: pairwise vs. Condition 0; much better powered at N=1,195
- Wilson score 95% CI per condition
- Per-subject accuracy: 7 MBE subjects × 8 conditions
- Cost tracking: tokens_input × $0.00000015 + tokens_output × $0.0000006

## Diagnostic Findings (N=1,195, zero-shot)

### Parse errors
Zero parse errors across all conditions (2 in Cond 1 are noise).
Answer extraction is not an issue — the regex reliably captures the first A/B/C/D.
Performance differences between conditions are real, not extraction artifacts.

### Condition 5 sycophancy
Stage 2 (verification) changes the answer only 8/1,195 times (0.7%).
Model confirms its own initial answer >99% of the time.
This is a known failure mode of LLM self-revision at smaller model sizes.
Condition 5 is effectively baseline at 3× the token cost.
Fix: force the model to argue for alternative answers before confirming.

### Main result
No intervention beats zero-shot baseline at N=1,195 on GPT-4o-mini.
Structured prompts add noise rather than signal at this model size.

### Failure Mode Taxonomy (applied to all baseline wrong answers)
Three mutually exclusive failure modes used to classify baseline errors.
Classified by GPT-4o-mini via classify_failures.py; output: results/failure_modes_baseline.csv.

FM1 — PARAMETRIC OVERRIDE: Model answered from memory or prior knowledge,
ignoring or contradicting the gold passage. Reasoning does not engage with
the passage content, or actively overrides it.

FM2 — REASONING FAILURE: Model engaged with the passage and referenced it,
but failed to reach the correct answer — whether by wrong logical inference
(misapplied reasoning) or by failing to complete the inferential step from
passage to answer (inference gap). In both sub-cases the passage is used
but the reasoning does not succeed.

Key distinction:
- FM1 vs FM2: did the model engage with the passage at all?

### Error taxonomy (N=20 sample: Cond 3/7 wrong, Baseline correct)
Earlier manual inspection of 20 questions where Cond 3 or Cond 7 failed but Baseline succeeded.
(Pre-FM taxonomy; kept for reference)

| Mode | Count | % |
|------|-------|---|
| (a) Wrong Rule Extracted | 2 | 11% |
| (b) Passage Ignored | 1 | 5% |
| (c) Steps Skipped / Shallow | 6 | 32% |
| (d) Reasoning Correct, Conclusion Wrong | 6 | 32% |
| (f) NaN / Empty Response | 5 | 26% |

## CLI Usage
```
python run_experiment.py --dry-run          # verify prompts, 1 real API call
python run_experiment.py --conditions 0 1  # run specific conditions
python run_experiment.py                   # run all 8 conditions
python run_experiment.py --summary-only    # re-run evaluator on existing CSVs
```

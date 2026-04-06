# Reducing Generation-Stage Hallucinations in Legal RAG Systems

**Repository:** https://github.com/luisakaczmarek/reducing-legal-hallucinations

---

## Overview

Large Language Models (LLMs) are rapidly being integrated into legal practice. Retrieval-Augmented Generation (RAG) is the dominant deployment architecture — a retriever fetches relevant passages, and a generator produces an answer conditioned on those passages. RAG is widely marketed as the solution to LLM hallucination in legal contexts, but empirical evaluations tell a different story.

Even when the **exactly correct gold passage** is given directly to the model, a **25–43% failure rate persists** (Zheng et al., 2025). This project targets that residual ceiling: the generation stage.

The mechanism is documented in the post-rationalization literature (arXiv:2412.18004): models generate answers from parametric memory and attach passage citations retroactively — up to 57% of citations in RAG systems are post-rationalized. This is a systematic failure mode, not noise.

---

## Research Questions

- **RQ1:** Can structured generation-stage prompting interventions reduce the residual failure rate observed when the correct gold passage is provided?
- **RQ2:** Which interventions are most effective, and do they interact with question subject area?
- **RQ3:** Do different interventions address different failure modes (parametric override vs. reasoning failure)?
- **RQ4:** Does few-shot prompting (showing correct passage-application examples) reduce FM2-type reasoning failures?

---

## Dataset

[reglab/barexam_qa](https://huggingface.co/datasets/reglab/barexam_qa) — from Zheng et al. (2025), "A Reasoning-Focused Legal Retrieval Benchmark"

- **1,195 MBE bar exam questions** across all splits (train/validation/test), all used for evaluation
- **7 subjects:** Constitutional Law, Contracts, Criminal Law, Evidence, Real Property, Torts, Other
- **Key property:** `gold_passage` is always populated — retrieval is perfectly controlled; all variance is generation-stage
- `prompt` (fact pattern) is NaN for ~63% of rows; when absent, facts are embedded in the question
- Official split maintained for reference (`split` column: train=954, validation=124, test=117) but **not used** for filtering — all 1,195 questions are evaluated

---

## Experimental Design

### Model & Infrastructure

**Model:** Llama 3.3 70B via [Groq](https://groq.com) free tier (`llama-3.3-70b-versatile`)
- OpenAI-compatible API endpoint (`https://api.groq.com/openai/v1`)
- Requires `GROQ_API_KEY` in environment or `.env` file
- **Temperature:** 0 for all conditions except Condition 6 (temp=0.7)
- **Max tokens:** 150 for single-answer conditions (0, 1, 6); 600 for reasoning conditions (3, 4); 300 per call for two-stage conditions (2, 5, 7)

> Note: an earlier pilot (N=117, `results/archive_test_split_117/`) was run on GPT-4o-mini. All main results use Llama 3.3 70B via Groq.

### Prompting Modes

Two modes, controlled by `--zero-shot` / `--few-shot` flag:

- **Zero-shot:** No examples. Prompt structure alone is the intervention variable. N=1,195.
- **Few-shot:** 2 fixed worked examples injected into the system prompt (1 Torts + 1 Contracts question, sampled seed=42 from train split, not in evaluation set). Examples show correct passage-grounding and reasoning steps. All conditions at N=1,195.

### The 8 Conditions

| ID | Name | Calls | Temp | Max Tokens | Core Mechanism |
|----|------|-------|------|------------|----------------|
| 0 | **Baseline** | 1 | 0 | 150 | Direct answer instruction — establishes residual failure rate |
| 1 | **Passage-Grounding Constraint** | 1 | 0 | 150 | Explicit instruction to rely *only* on the passage; tests whether post-rationalization can be reduced by instruction alone |
| 2 | **Rule Extraction** | 2 | 0 | 300/call | Stage 1: extract the legal rule/principle in one sentence → Stage 2: apply extracted rule to answer selection |
| 3 | **Chain of Logic (CoL)** | 1 | 0 | 600 | **v2 (redesigned):** 3-step simplified prompt — identify rule from passage → apply rule directly to the facts → state answer letter. Original 4-step IRAC version (Servantez et al., 2024) preserved as comment in `condition_3_col.py`. Redesign motivated by FM2 dominance. |
| 4 | **Negative Elimination** | 1 | 0 | 600 | Per-choice: identify the legal doctrine invoked → state whether passage supports/contradicts it → eliminate or retain. Select surviving choice. Novel condition motivated by MBE structure. |
| 5 | **Answer Verification** | 2 | 0 | 300/call | Stage 1: baseline generation → answer + justification → Stage 2: check each claim against the passage; flag unsupported claims; confirm or revise |
| 6 | **Self-Consistency (N=3)** | 3 | 0.7 | 150/call | Three independent samples → majority vote on answer letter; tie-break: first call's answer. Based on Wang et al. (2023). |
| 7 | **Rule Extraction + CoL** | 2 | 0 | 300/call | Stage 1: extract legal rule (same as Cond 2) → Stage 2: apply Chain of Logic v2 to the extracted rule. Compounds strengths of Conditions 2 and 3. |

#### Expected Strengths & Known Limitations

| ID | Targets / Hypothesis | Key Risk / Limitation |
|----|----------------------|-----------------------|
| 0 | Establishes residual failure rate | Parametric override, post-rationalization |
| 1 | Reduce post-rationalization by instruction | May hurt on questions needing elimination of wrong answers using knowledge outside the passage |
| 2 | Oblique passages where the rule is not stated directly | Fabricates rule on analogical passages (case descriptions rather than rules) |
| 3 | Element-checking questions; shorter chain reduces over-decomposition | Shallow steps or skipped steps; reasoning correct but conclusion wrong |
| 4 | Questions where correct answer = last one standing after elimination | Misidentified doctrines; passage may not address all choices |
| 5 | Catch unsupported claims in initial reasoning | >99% sycophancy — model almost never revises its own answer (8/1,195 changes) |
| 6 | Reduce variance; boost borderline correct answers | Amplifies systematic errors; 3× token cost |
| 7 | Compound strengths of Rule Extraction + CoL | Compounds failure modes — Stage 1 fabrication propagates to Stage 2 |

#### Token Cost Profile

| Category | Conditions | Relative Cost |
|----------|-----------|---------------|
| Single short call | 0, 1 | 1× |
| Three-sample voting | 6 | ~3× |
| Two-stage call | 2, 5, 7 | ~3× |
| Single reasoning call | 3, 4 | ~4× |
| Rule Extraction + CoL | 7 | ~5–6× |

### Prompt Architecture

- **System prompt:** role definition only ("You are a legal reasoning assistant...") — no examples in zero-shot mode; 2 worked examples prepended in few-shot mode
- **User prompt structure:**
  ```
  [FACT PATTERN: {prompt}]   ← prepended only if row['prompt'] is not NaN
  QUESTION: {question}
  A) {choice_a}
  B) {choice_b}
  C) {choice_c}
  D) {choice_d}
  PASSAGE: {gold_passage}
  {condition-specific instruction}
  ```
- **Answer format (every call):** "Respond with the answer letter (A, B, C, or D) on the first line, then your reasoning."
- **Answer extraction:** regex on first generated token; zero parse errors at N=1,195

### Incremental Saving & Resume

Results are written one row per question immediately after each API call — safe to interrupt at any point. On restart, existing CSVs are checked: rows with matching `idx` are skipped automatically.

---

## Contributions

1. **First systematic evaluation** of multiple generation-stage interventions specifically targeting the gold-passage ceiling in legal RAG
2. **Negative Elimination** — a novel prompting condition motivated by the structural properties of MBE questions
3. **ECE calibration measurement** following Dahl et al. (2024), with reliability diagrams per condition
4. **Failure mode taxonomy (FM1/FM2)** applied to all 431 baseline errors, revealing that FM2 (reasoning failure) dominates (89.3%), motivating few-shot intervention

---

## Repository Structure

```
run_experiment.py              # Entry point — CLI for all 8 conditions
evaluator.py                   # Accuracy, ECE, McNemar's test, per-subject breakdown
classify_failures.py           # Classifies baseline wrong answers into FM1/FM2
conditions/
  base.py                      # BaseCondition class, prompt builder, logprob utils
  few_shot_examples.py         # Fixed few-shot examples (seed=42, Torts + Contracts)
  condition_0_baseline.py
  condition_1_grounding.py
  condition_2_rule_extraction.py
  condition_3_col.py           # v2: 3-step prompt (original 4-step IRAC preserved as comment)
  condition_4_negative_elim.py
  condition_5_verification.py
  condition_6_self_consistency.py
  condition_7_rule_col.py
data/                          # Dataset (parquet + csv, loaded automatically via glob)
logs/                          # nohup run logs
results/
  zero_shot/                   # Zero-shot results, N=1,195 per condition
    condition_{N}_{name}.csv   # Per-question: idx, subject, correct_answer, predicted_answer,
                               #   is_correct, tokens_in/out, logprob_A/B/C/D, confidence
    summary.csv                # Aggregated accuracy, ECE, cost per condition
    per_subject_accuracy.csv   # Accuracy by subject × condition
    ece_bins_{name}.csv        # 10-bin ECE breakdown per condition
    reliability_{name}.png     # Reliability diagram per condition
  few_shot/                    # Few-shot results (same structure), all conditions N=1,195
  analysis/                    # Cross-run analysis and final summary
  archive_test_split_117/      # Pilot on N=117 test split (GPT-4o-mini)
```

---

## Usage

```bash
# Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your Groq API key (free tier at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# Verify prompt formatting (1 real API call)
python run_experiment.py --dry-run

# Run specific conditions zero-shot
python run_experiment.py --conditions 0 1 --zero-shot

# Run all 8 conditions zero-shot in background
nohup python run_experiment.py --zero-shot > logs/zero_shot.log 2>&1 &

# Run all 8 conditions few-shot in background
nohup python run_experiment.py --few-shot > logs/few_shot.log 2>&1 &

# Resume is on by default — restarts skip already-completed rows
# Re-run evaluator on existing CSVs without new API calls
python run_experiment.py --summary-only --zero-shot
python run_experiment.py --summary-only --few-shot
```

---

## Results

### Zero-shot — N=1,195 (complete)

| ID | Condition | Correct | Accuracy |
|----|-----------|---------|----------|
| 0 | Baseline | 764 | **63.9%** |
| 1 | Grounding | 737 | 61.7% |
| 2 | Rule Extraction | 756 | 63.3% |
| 3 | Chain of Logic | 755 | 63.2% |
| 4 | Negative Elimination | 727 | 60.8% |
| 5 | Answer Verification | 750 | 62.8% |
| 6 | Self-Consistency (N=3) | 752 | 62.9% |
| 7 | Rule Extraction + CoL | 753 | 63.0% |

**Main finding:** No intervention reliably improves on zero-shot baseline. Structured prompts add noise rather than signal on this model at zero-shot.

### Few-shot — N=1,195 (complete)

| ID | Condition | Correct | Accuracy |
|----|-----------|---------|----------|
| 0 | Baseline | 722 | 60.4% |
| 1 | Grounding | 708 | 59.2% |
| 2 | Rule Extraction | 818 | 68.5% |
| 3 | Chain of Logic | 957 | **80.1%** |
| 4 | Negative Elimination | 940 | 78.7% |
| 5 | Answer Verification | 955 | 79.9% |
| 6 | Self-Consistency (N=3) | 954 | **79.8%** |
| 7 | Rule Extraction + CoL | 939 | 78.6% |

**Key few-shot finding:** Few-shot dramatically boosts reasoning-heavy conditions. Chain of Logic jumps from 63.2% → **80.1%** (+16.9 pp); Negative Elimination from 60.8% → **78.7%** (+17.9 pp); Answer Verification from 62.8% → **79.9%** (+17.1 pp). Baseline and Grounding slightly decline. This confirms the FM2 hypothesis: showing worked examples of correct passage-application directly addresses reasoning failures, the dominant error type (89.3% of baseline errors).

### Zero-shot per-subject accuracy

| Subject | Cond 0 | Cond 1 | Cond 2 | Cond 3 | Cond 4 | Cond 5 | Cond 6 | Cond 7 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|
| CONST. LAW | **87.4%** | 85.3% | 88.4% | 88.4% | 83.2% | 84.2% | 87.4% | 87.4% |
| CONTRACTS | **55.8%** | 52.2% | 53.1% | 54.9% | 51.3% | 53.1% | 53.1% | 55.8% |
| CRIM. LAW | 57.3% | **59.6%** | 57.3% | 50.6% | 51.7% | 56.2% | 59.6% | 55.1% |
| EVIDENCE | 60.2% | 59.1% | 62.4% | 61.3% | 61.3% | 60.2% | **63.4%** | 60.2% |
| REAL PROP. | 52.2% | 48.9% | **53.3%** | 54.3% | 47.8% | 52.2% | 52.2% | 52.2% |
| TORTS | **59.8%** | 57.1% | 58.9% | 59.8% | 55.4% | 58.9% | 59.8% | 55.4% |

### Diagnostic results

**Parse errors:** Essentially zero across all conditions (0–2 out of 1,195). Answer extraction is not the issue.

**Condition 5 sycophancy (zero-shot):** Stage 2 (verification) changes the answer only **8/1,195 times (0.7%)**. The model confirms its own initial answer >99% of the time regardless of correctness — a well-documented failure mode of LLM self-revision. Condition 5 is effectively baseline at 3× the token cost.

### Failure mode taxonomy

Two mutually exclusive failure modes applied to all 431 baseline zero-shot wrong answers via `classify_failures.py`:

| Mode | Description | Count | % |
|------|-------------|-------|---|
| **FM1 — Parametric Override** | Model answered from memory/prior knowledge, ignoring or contradicting the gold passage | 46 | 10.7% |
| **FM2 — Reasoning Failure** | Model engaged with the passage but failed to reach the correct answer — whether by wrong logical inference or by failing to complete the inferential step from passage to answer | 385 | 89.3% |

FM2 overwhelmingly dominates. The problem is not ignoring the passage — it is failing to reason correctly over it. This motivated: (1) redesigning Condition 3 to use a shorter 3-step chain, and (2) adding few-shot examples demonstrating correct passage-application.

---

## Evaluation Metrics

- **Accuracy** — proportion of correct answer letters (primary metric)
- **McNemar's test** — pairwise significance vs. Condition 0 for paired binary outcomes
- **Wilson score CI** — 95% confidence intervals per condition
- **ECE** — Expected Calibration Error (Guo et al. 2017), 10 equal-width bins; measures gap between expressed confidence and actual accuracy. Confidence = softmax over logprobs of A/B/C/D at first generated token; Condition 6 uses votes/3.
- **Per-subject breakdown** — accuracy across 7 MBE subjects per condition

---

## Key Literature

| Paper | Relevance |
|-------|-----------|
| Dahl et al. (2024) — "Large Legal Fictions", *Journal of Legal Analysis* | Foundational empirical baseline: LLMs hallucinate ≥58% on legal tasks; ECE methodology |
| Magesh et al. (2024) — "Hallucination-Free?", arXiv:2405.20362 | Hallucination persists in commercial RAG systems (Westlaw, LexisNexis) |
| Zheng et al. (2025) — "A Reasoning-Focused Legal Retrieval Benchmark", CS&Law | Gold-passage ceiling: 25–43% failure even with correct passage provided; source of barexam_qa dataset |
| arXiv:2412.18004 (2024) — "Correctness is not Faithfulness in RAG Attributions" | Mechanism: 57% of citations post-rationalized from parametric memory |
| Wu et al. (2024) — ClashEval | Quantifies parametric prior vs. retrieved evidence tug-of-war |
| Servantez et al. (2024) — Chain of Logic, ACL Findings | IRAC-inspired legal prompting; basis for Condition 3 |
| Wang et al. (2023) — Self-Consistency | Majority vote over multiple samples; basis for Condition 6 |

---

## Limitations

- **Prompt sensitivity:** results may be sensitive to exact prompt wording — the most significant caveat
- **Single model:** all main results are on Llama 3.3 70B Q4_K_M; findings may not generalise to other model families or sizes
- **Answer-letter evaluation only:** a model may select the correct letter through flawed reasoning; qualitative analysis (N=20 manual inspection of Cond 3/7 failures) shows dominant modes are shallow reasoning steps (32%) and reasoning–conclusion mismatch (32%)
- **Zero-shot vs. Zheng et al.:** direct numerical comparison to their reported ceiling figures should be made with caution — their setup used few-shot prompting

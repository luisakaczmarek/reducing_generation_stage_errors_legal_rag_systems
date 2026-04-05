#!/usr/bin/env python3
"""
classify_failures.py — Classify baseline (Condition 0) wrong answers into
failure modes FM1, FM2 using Llama 3.3 70B as the classifier.

Failure modes:
  FM1 — PARAMETRIC OVERRIDE: Model answered from memory, ignoring or
        contradicting the gold passage. Reasoning does not engage with passage.
  FM2 — REASONING FAILURE: Model engaged with the passage and referenced it,
        but failed to reach the correct answer — whether by wrong logical
        inference (misapplied reasoning) or by failing to complete the
        inferential step from passage to answer (inference gap). In both
        sub-cases the passage is used but the reasoning does not succeed.

Key distinction:
  FM1 vs FM2: did the model engage with the passage at all?
"""

import os
import sys
import glob
import time
import random

import pandas as pd
import openai

RESULTS_DIR = "results"
OUT_PATH = os.path.join(RESULTS_DIR, "failure_modes_baseline.csv")

CLASSIFIER_SYSTEM = (
    "You are an expert legal educator and LLM evaluation specialist. "
    "Your task is to classify why a language model answered a bar exam question incorrectly, "
    "given that the correct gold passage was provided to it."
)

CLASSIFIER_PROMPT_TEMPLATE = """\
## Question
{question}

## Answer choices
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

## Gold passage (was provided to the model)
{gold_passage}

## Correct answer: {correct_answer}
## Model's predicted answer: {predicted_answer}

## Model's reasoning (raw response)
{raw_response}

---

Classify the model's failure into exactly one of the following modes:

FM1 — PARAMETRIC OVERRIDE: The model answered from memory or prior knowledge, \
ignoring or contradicting the passage. The reasoning does not engage with the \
passage content, or actively overrides it.

FM2 — REASONING FAILURE: The model engaged with the passage and referenced it, \
but failed to reach the correct answer — whether by making a wrong logical \
inference from the passage, or by failing to complete the inferential step \
from passage to answer. In both cases the passage is used but the reasoning \
does not succeed.

Key distinction:
- FM1 vs FM2: did the model engage with the passage at all?

Respond with exactly one line: FM1 or FM2. Then a brief (1-2 sentence) justification.
"""


def load_dataset():
    patterns = [
        "./data/**/*clean*.parquet",
        "./data/**/*.parquet",
        "./data/**/*.csv",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError("No .parquet or .csv file found under ./data/")


def call_classifier(client, messages):
    for attempt in range(6):
        try:
            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=120,
                temperature=0,
            )
        except openai.RateLimitError:
            wait = (2 ** attempt) + random.uniform(-0.5, 0.5)
            print(f"\n  Rate limit. Waiting {wait:.1f}s (attempt {attempt+1}/6)…")
            time.sleep(wait)
        except (openai.APIConnectionError, openai.APITimeoutError):
            wait = (2 ** attempt) + random.uniform(-0.5, 0.5)
            print(f"\n  Connection error. Waiting {wait:.1f}s (attempt {attempt+1}/6)…")
            time.sleep(wait)
        except Exception:
            raise
    raise RuntimeError("Max retries exceeded")


def extract_fm(text):
    """Extract FM1/FM2 label from first line of classifier response."""
    if not text:
        return "PARSE_ERROR"
    first = text.strip().split("\n")[0].strip().upper()
    for label in ("FM1", "FM2"):
        if label in first:
            return label
    return "PARSE_ERROR"


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["GROQ_API_KEY"] = api_key
                    break
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY not set.")

    client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    # Load full dataset for question details
    data_path = load_dataset()
    df_full = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    df_full = df_full.set_index("idx")

    # Load baseline results and filter to wrong answers
    baseline_path = os.path.join(RESULTS_DIR, "condition_0_baseline.csv")
    df_base = pd.read_csv(baseline_path)
    df_wrong = df_base[df_base["is_correct"] == False].copy()
    print(f"Baseline wrong answers: {len(df_wrong)} / {len(df_base)}")

    # Resume: skip already-classified rows
    processed_idx = set()
    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        processed_idx = set(existing["idx"].tolist())
        print(f"Resuming: {len(processed_idx)} already classified.")

    to_process = df_wrong[~df_wrong["idx"].isin(processed_idx)]
    print(f"To classify: {len(to_process)}")

    columns = ["idx", "subject", "correct_answer", "predicted_answer", "failure_mode", "justification"]
    mode = "a" if processed_idx else "w"

    import csv
    from tqdm import tqdm

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if mode == "w":
            writer.writeheader()

        for _, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Classifying"):
            idx = row["idx"]

            # Get full row from dataset
            if idx not in df_full.index:
                print(f"\n  WARNING: {idx} not found in dataset. Skipping.")
                continue

            full = df_full.loc[idx]

            # Build prompt
            prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
                question=str(full.get("question", "")),
                choice_a=str(full.get("choice_a", "")),
                choice_b=str(full.get("choice_b", "")),
                choice_c=str(full.get("choice_c", "")),
                choice_d=str(full.get("choice_d", "")),
                gold_passage=str(full.get("gold_passage", "")),
                correct_answer=str(row["correct_answer"]),
                predicted_answer=str(row["predicted_answer"]),
                raw_response=str(row.get("raw_response_1", ""))[:1500],
            )

            messages = [
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": prompt},
            ]

            resp = call_classifier(client, messages)
            response_text = resp.choices[0].message.content.strip()
            fm = extract_fm(response_text)
            justification = " ".join(response_text.split("\n")[1:]).strip()[:300]

            writer.writerow({
                "idx": idx,
                "subject": row.get("subject", ""),
                "correct_answer": row["correct_answer"],
                "predicted_answer": row["predicted_answer"],
                "failure_mode": fm,
                "justification": justification,
            })
            f.flush()

    # --- Summary ---
    df_out = pd.read_csv(OUT_PATH)
    total = len(df_out)

    print("\n" + "="*60)
    print("FAILURE MODE DISTRIBUTION — BASELINE (N=431 wrong answers)")
    print("="*60)

    overall = df_out["failure_mode"].value_counts()
    for fm in ["FM1", "FM2", "FM3", "PARSE_ERROR"]:
        n = overall.get(fm, 0)
        print(f"  {fm}: {n:4d}  ({100*n/total:.1f}%)")

    print("\n--- By Subject ---")
    subjects = df_out["subject"].fillna("UNKNOWN").unique()
    # Header
    header = f"{'Subject':<14}" + "".join(f"{'FM1':>8}{'FM2':>8}{'FM3':>8}{'N':>6}")
    print(header)
    print("-" * len(header))

    for subj in sorted(subjects):
        sub = df_out[df_out["subject"].fillna("UNKNOWN") == subj]
        n = len(sub)
        fm1 = (sub["failure_mode"] == "FM1").sum()
        fm2 = (sub["failure_mode"] == "FM2").sum()
        fm3 = (sub["failure_mode"] == "FM3").sum()
        print(f"{subj:<14}  {100*fm1/n:5.1f}%  {100*fm2/n:5.1f}%  {100*fm3/n:5.1f}%  {n:4d}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_experiment.py — Entry point for the MBE generation-stage hallucination experiment.

Backend: Groq API (llama-3.3-70b-versatile). Requires GROQ_API_KEY env var or .env file.

Usage:
  python run_experiment.py --dry-run              # verify prompt formatting, no API calls
  python run_experiment.py --conditions 0 1       # run specific conditions
  python run_experiment.py                        # run all 8 conditions
  python run_experiment.py --summary-only         # re-run evaluator on existing CSVs
  python run_experiment.py --few-shot             # run with few-shot examples
"""

import argparse
import glob
import os
import sys

import pandas as pd

from conditions.base import (
    build_system_prompt,
    build_user_message,
    extract_answer,
    extract_answer_logprobs,
    logprobs_to_confidence,
    ANSWER_FORMAT,
)
from conditions.condition_0_baseline import Condition0Baseline
from conditions.condition_1_grounding import Condition1Grounding
from conditions.condition_2_rule_extraction import Condition2RuleExtraction
from conditions.condition_3_col import Condition3CoL
from conditions.condition_4_negative_elim import Condition4NegativeElim
from conditions.condition_5_verification import Condition5Verification
from conditions.condition_6_self_consistency import Condition6SelfConsistency
from conditions.condition_7_rule_col import Condition7RuleCoL
from evaluator import run_evaluator

CONDITION_CLASSES = {
    0: Condition0Baseline,
    1: Condition1Grounding,
    2: Condition2RuleExtraction,
    3: Condition3CoL,
    4: Condition4NegativeElim,
    5: Condition5Verification,
    6: Condition6SelfConsistency,
    7: Condition7RuleCoL,
}


# ── Data loading ─────────────────────────────────────────────────────────────

def find_data_file():
    """Find the best available data file under ./data/ (prefers clean parquet)."""
    for pattern in [
        "./data/**/*clean*.parquet",
        "./data/**/*.parquet",
        "./data/**/*.csv",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError("No .parquet or .csv file found under ./data/")


def load_dataset():
    path = find_data_file()
    print(f"Loading data from: {path}")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df


# ── Dry run ──────────────────────────────────────────────────────────────────

def do_dry_run(df_test, system_prompt, client, model="llama-3.3-70b-versatile"):
    print("\n" + "=" * 80)
    print("DRY RUN — Condition 0 prompts + logprob verification (1 real API call)")
    print("=" * 80)

    print("\n--- SYSTEM PROMPT (first 1 000 chars) ---")
    print(system_prompt[:1000] + ("…" if len(system_prompt) > 1000 else ""))

    for i, (_, row) in enumerate(df_test.head(2).iterrows(), 1):
        user_msg = (
            build_user_message(row)
            + "\n\nAnswer the question based on the passage."
            + ANSWER_FORMAT
        )
        print(f"\n--- USER PROMPT (question {i}) ---")
        print(user_msg[:2000] + ("…" if len(user_msg) > 2000 else ""))

    # One real API call to verify logprob extraction
    print("\n--- LOGPROB VERIFICATION (question 1, one real API call) ---")
    row = df_test.iloc[0]
    user_msg = (
        build_user_message(row)
        + "\n\nAnswer the question based on the passage."
        + ANSWER_FORMAT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=150,
        temperature=0,
        logprobs=True,
        top_logprobs=4,
    )
    text = r.choices[0].message.content or ""
    predicted = extract_answer(text)
    lp = extract_answer_logprobs(r)
    conf = logprobs_to_confidence(lp, predicted)

    print(f"Response     : {text[:120]}")
    print(f"Predicted    : {predicted}")
    print(f"Logprobs     : {lp}")
    print(f"Confidence   : {conf:.4f}")
    print("\nDry run complete. Logprob extraction verified.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MBE generation-stage experiment"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        type=int,
        default=list(CONDITION_CLASSES.keys()),
        help="Condition IDs to run (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip conditions whose output CSV is already complete (default: True)",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts for first 2 questions of condition 0; no API calls",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip all API calls; re-run evaluator on existing CSVs",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Run with few-shot examples injected into system prompt (default: zero-shot)",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Explicit zero-shot mode (default; no-op when --few-shot is absent)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N questions (for testing)",
    )
    args = parser.parse_args()

    df = load_dataset()
    df_test = df.reset_index(drop=True)
    if args.limit:
        df_test = df_test.head(args.limit)
        print(f"Evaluating on subset: {len(df_test)} questions (--limit {args.limit})")
    else:
        print(f"Evaluating on full dataset: {len(df_test)} questions")

    # Determine mode and output directory
    mode = "few_shot" if args.few_shot else "zero_shot"
    results_dir = os.path.join("results", mode)

    # Build system prompt — inject few-shot prefix if requested
    if args.few_shot:
        from conditions.few_shot_examples import get_few_shot_prompt, get_few_shot_examples
        few_shot_prefix = get_few_shot_prompt()
        system_prompt = few_shot_prefix + build_system_prompt()
        # Exclude few-shot examples from evaluation to prevent leakage
        few_shot_ids = {ex["idx"] for ex in get_few_shot_examples().values()}
        df_test = df_test[~df_test["idx"].isin(few_shot_ids)].reset_index(drop=True)
        print(f"Mode: few-shot | Excluded {len(few_shot_ids)} few-shot examples from eval ({few_shot_ids}) | Results dir: {results_dir}")
    else:
        system_prompt = build_system_prompt()
        print(f"Mode: zero-shot | Results dir: {results_dir}")

    if args.summary_only:
        print("\nRunning evaluator only…")
        run_evaluator(results_dir=results_dir)
        return

    # API client — Groq (llama-3.3-70b-versatile)
    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    if not api_key:
        raise SystemExit("ERROR: GROQ_API_KEY not set. Export it or add to .env.")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = "llama-3.3-70b-versatile"
    print(f"Backend: Groq ({model})")

    if args.dry_run:
        do_dry_run(df_test, system_prompt, client, model=model)
        return

    os.makedirs(results_dir, exist_ok=True)

    total_cost = 0.0

    for cond_id in sorted(args.conditions):
        if cond_id not in CONDITION_CLASSES:
            print(f"Warning: condition {cond_id} not defined — skipping.")
            continue

        cls = CONDITION_CLASSES[cond_id]

        # Resume: skip if complete
        if args.resume:
            out_path = os.path.join(results_dir, f"condition_{cond_id}_{cls.CONDITION_NAME}.csv")
            if os.path.exists(out_path):
                existing = pd.read_csv(out_path)
                if len(existing) >= len(df_test):
                    print(
                        f"Condition {cond_id} already complete "
                        f"({len(existing)} rows). Skipping."
                    )
                    continue

        cond = cls(client=client, system_prompt=system_prompt, model=model)
        stats = cond.run(df_test, results_dir=results_dir)
        total_cost += stats["cost"]
        print(f"Running total cost: ${total_cost:.4f}\n")

    print("\n" + "=" * 80)
    run_evaluator(results_dir=results_dir)


if __name__ == "__main__":
    main()

"""
Shared base class and prompt utilities for all conditions.
"""

import re
import time
import random
import os
from abc import ABC, abstractmethod

import pandas as pd


SYSTEM_PREFIX = (
    "You are a legal reasoning assistant. Answer multiple-choice bar exam questions.\n"
    "You will always be given a gold passage containing the relevant legal rule or "
    "principle. Your answer must be grounded in that passage."
)

ANSWER_FORMAT = (
    "\n\nANSWER FORMAT: Respond with the answer letter (A, B, C, or D) on the "
    "first line, then your reasoning."
)


def build_system_prompt():
    """Build zero-shot system prompt."""
    return SYSTEM_PREFIX


def build_user_message(row):
    """Build the user message for a single question (no condition instruction)."""
    parts = []
    if pd.notna(row.get("prompt")) and str(row["prompt"]).strip():
        parts.append(f"FACT PATTERN:\n{row['prompt']}\n")
    parts.append(f"QUESTION: {row['question']}")
    parts.append(
        f"(A) {row['choice_a']}\n(B) {row['choice_b']}\n"
        f"(C) {row['choice_c']}\n(D) {row['choice_d']}"
    )
    parts.append(f"\nGOLD PASSAGE:\n{row['gold_passage']}")
    return "\n".join(parts)



def extract_answer(text):
    """Return first standalone A/B/C/D found, else 'PARSE_ERROR'."""
    if not text:
        return "PARSE_ERROR"
    first_line = text.strip().split("\n")[0].strip()
    m = re.search(r"\b([ABCD])\b", first_line)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    return "PARSE_ERROR"


class BaseCondition(ABC):
    CONDITION_ID: int = -1
    CONDITION_NAME: str = "base"

    def __init__(self, client, system_prompt, model="llama3.3:70b"):
        self.client = client
        self.system_prompt = system_prompt
        self.model = model

    def call_api(self, messages, max_tokens, temperature=0):
        """Call OpenAI with exponential backoff on 429 errors."""
        import openai

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        for attempt in range(6):
            try:
                return self.client.chat.completions.create(**kwargs)
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
        raise RuntimeError("Max retries exceeded for API call")

    @abstractmethod
    def process_question(self, row):
        """
        Process one question. Must return a dict with keys:
          predicted_answer, tokens_input, tokens_output,
          raw_response_1, raw_response_2
        """

    def run(self, df_test, results_dir="results"):
        """Run condition on test set with incremental saving and resume support."""
        import csv
        from tqdm import tqdm

        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(
            results_dir, f"condition_{self.CONDITION_ID}_{self.CONDITION_NAME}.csv"
        )

        # Resume: skip already-processed rows
        processed_idx = set()
        tokens_in_total = 0
        tokens_out_total = 0
        n_correct = 0

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            processed_idx = set(existing["idx"].tolist())
            tokens_in_total = int(existing["tokens_input"].sum())
            tokens_out_total = int(existing["tokens_output"].sum())
            n_correct = int(existing["is_correct"].sum())
            print(f"  Resuming: {len(processed_idx)} rows already done.")

        to_process = df_test[~df_test["idx"].isin(processed_idx)]
        n_total = len(df_test)

        print(
            f"Running condition {self.CONDITION_ID}: {self.CONDITION_NAME} "
            f"— {n_total} questions ({len(to_process)} remaining)"
        )

        columns = [
            "idx", "subject", "source", "split", "question",
            "correct_answer", "predicted_answer", "is_correct",
            "condition", "tokens_input", "tokens_output",
            "raw_response_1", "raw_response_2",
        ]
        if self.CONDITION_ID == 6:
            columns.append("raw_response_3")
            columns.append("confidence")

        mode = "a" if processed_idx else "w"

        with open(out_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()

            for _, row in tqdm(
                to_process.iterrows(),
                total=len(to_process),
                desc=f"Cond {self.CONDITION_ID}",
            ):
                result = self.process_question(row)
                predicted = result["predicted_answer"]
                correct = str(row["answer"]).strip().upper()
                is_correct = predicted == correct

                record = {
                    "idx": row["idx"],
                    "subject": row.get("subject", ""),
                    "source": row.get("source", ""),
                    "split": row["split"],
                    "question": str(row["question"])[:100],
                    "correct_answer": correct,
                    "predicted_answer": predicted,
                    "is_correct": is_correct,
                    "condition": self.CONDITION_ID,
                    "tokens_input": result.get("tokens_input", 0),
                    "tokens_output": result.get("tokens_output", 0),
                    "raw_response_1": result.get("raw_response_1", ""),
                    "raw_response_2": result.get("raw_response_2", ""),
                    "raw_response_3": result.get("raw_response_3", ""),
                    "confidence": result.get("confidence"),
                }
                writer.writerow(record)
                f.flush()

                tokens_in_total += result.get("tokens_input", 0)
                tokens_out_total += result.get("tokens_output", 0)
                n_correct += int(is_correct)

        accuracy = n_correct / n_total if n_total > 0 else 0
        cost = tokens_in_total * 0.00000015 + tokens_out_total * 0.0000006
        print(
            f"Condition {self.CONDITION_ID} complete. "
            f"Accuracy: {accuracy:.1%} | Cost: ${cost:.4f}"
        )
        print(f"  Tokens: {tokens_in_total} in / {tokens_out_total} out")

        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "tokens_input": tokens_in_total,
            "tokens_output": tokens_out_total,
            "cost": cost,
        }

"""
few_shot_examples.py — Fixed few-shot examples for the MBE experiment.

Two examples selected from the train split (seed=42):
  - 1 TORTS question
  - 1 CONTRACTS question

Reasoning strings are placeholders — replace with curated human-written
or model-generated chains before running few-shot conditions.
"""

import glob

import pandas as pd

# Curated reasoning for the two fixed few-shot examples (seed=42).
# Keyed by idx so they survive any reordering of the dataset.
_REASONING = {
    "mbe_1106": (
        "The passage establishes that a property owner owes a duty of care to a "
        "trespasser only once the trespasser's presence becomes known. Therefore, "
        "Trespasser can only prevail if Vintner had actual awareness of his presence "
        "and then failed to take reasonable care — making answer (B) correct."
    ),
    "mbe_1162": (
        "The passage defines reasonable reliance as reliance that a reasonably prudent "
        "person exercising ordinary care would not have discovered to be misplaced. "
        "Loyal's retirement and $30,000 RV purchase in direct response to the Board's "
        "signed pension promise constitutes exactly that kind of justified reliance, "
        "making answer (B) correct."
    ),
}


def _load_dataset():
    for pattern in [
        "../data/**/*clean*.parquet",
        "../data/**/*.parquet",
        "../data/**/*.csv",
        "./data/**/*clean*.parquet",
        "./data/**/*.parquet",
        "./data/**/*.csv",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            path = matches[0]
            return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    raise FileNotFoundError("No data file found under ./data/")


def get_few_shot_examples():
    """
    Select 2 fixed few-shot examples (seed=42) from the train split:
      - 1 TORTS question
      - 1 CONTRACTS question

    Returns: dict with keys 'torts' and 'contracts', each containing
      'question', 'prompt', 'choice_a', 'choice_b', 'choice_c', 'choice_d',
      'gold_passage', 'correct_answer', 'reasoning'
    """
    df = _load_dataset()
    train = df[df["split"] == "train"].copy()

    torts_row = (
        train[train["subject"] == "TORTS"]
        .sample(1, random_state=42)
        .iloc[0]
    )
    contracts_row = (
        train[train["subject"] == "CONTRACTS"]
        .sample(1, random_state=42)
        .iloc[0]
    )

    def _row_to_example(row):
        return {
            "idx": row["idx"],
            "question": str(row.get("question", "")),
            "prompt": str(row.get("prompt", "")) if pd.notna(row.get("prompt")) else "",
            "choice_a": str(row.get("choice_a", "")),
            "choice_b": str(row.get("choice_b", "")),
            "choice_c": str(row.get("choice_c", "")),
            "choice_d": str(row.get("choice_d", "")),
            "gold_passage": str(row.get("gold_passage", "")),
            "correct_answer": str(row.get("answer", "")).strip().upper(),
            "reasoning": _REASONING.get(row["idx"], ""),
        }

    return {
        "torts": _row_to_example(torts_row),
        "contracts": _row_to_example(contracts_row),
    }


def _format_example(ex, label):
    """Format a single example for injection into a system prompt."""
    parts = [f"EXAMPLE ({label}):"]
    if ex["prompt"].strip():
        parts.append(f"Fact pattern: {ex['prompt']}")
    parts.append(f"Question: {ex['question']}")
    parts.append(
        f"(A) {ex['choice_a']}\n(B) {ex['choice_b']}\n"
        f"(C) {ex['choice_c']}\n(D) {ex['choice_d']}"
    )
    parts.append(f"Gold passage: {ex['gold_passage']}")
    parts.append(f"Answer: {ex['correct_answer']}")
    parts.append(f"Reasoning: {ex['reasoning']}")
    return "\n".join(parts)


def get_few_shot_prompt():
    """
    Returns a formatted few-shot prefix string ready to prepend to any
    condition's system prompt.

    Returns: str
    """
    examples = get_few_shot_examples()

    prefix = (
        "Here are two example questions demonstrating the expected reasoning format:\n\n"
        + _format_example(examples["torts"], "Torts")
        + "\n\n---\n\n"
        + _format_example(examples["contracts"], "Contracts")
        + "\n\n---\n\n"
        "Now apply this same reasoning approach to the following questions:\n"
    )
    return prefix

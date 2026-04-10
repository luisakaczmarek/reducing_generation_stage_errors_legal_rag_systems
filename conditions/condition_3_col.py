from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    ANSWER_FORMAT,
)

# ORIGINAL 4-STEP VERSION (PILOT)
# _INSTRUCTION = """
#
# Reason through this question using the following structured steps:
#
# Step 1 — RULE: Identify the legal rule or principle stated in the gold passage (1 sentence).
# Step 2 — ELEMENTS: Decompose the rule into its key elements or requirements.
# Step 3 — APPLICATION: Evaluate each answer choice against those elements and the facts.
# Step 4 — CONCLUSION: Select the answer that best satisfies the rule given the facts."""

_INSTRUCTION = """

Step 1: Identify the legal rule or principle stated in the passage (one sentence).
Step 2: Apply it directly to the facts: which answer choice does it support \
or exclude, and why?
Step 3: State your answer."""


class Condition3CoL(BaseCondition):
    """Chain of Logic (IRAC-inspired), based on Servantez et al. ACL 2024."""

    CONDITION_ID = 3
    CONDITION_NAME = "chain_of_logic"

    def process_question(self, row):
        user_msg = build_user_message(row) + _INSTRUCTION + ANSWER_FORMAT

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        r = self.call_api(messages, max_tokens=600)
        text = r.choices[0].message.content or ""
        predicted = extract_answer(text)

        return {
            "predicted_answer": predicted,
            "tokens_input": r.usage.prompt_tokens,
            "tokens_output": r.usage.completion_tokens,
            "raw_response_1": text,
            "raw_response_2": None,
        }

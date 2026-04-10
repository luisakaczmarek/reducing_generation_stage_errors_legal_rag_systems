from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    ANSWER_FORMAT,
)

_STAGE1_TEMPLATE = (
    "GOLD PASSAGE:\n{passage}\n\n"
    "Extract the core legal rule or principle from this passage in exactly one sentence. "
    "Respond with only that one sentence."
)

_STAGE2_COL = (
    "\n\nEXTRACTED LEGAL RULE: {rule}\n\n"
    "Using the extracted rule and the gold passage, reason through this question:\n\n"
    "Step 1 — RULE: State the extracted legal rule (already provided above).\n"
    "Step 2 — ELEMENTS: Decompose the rule into its key elements or requirements.\n"
    "Step 3 — APPLICATION: Evaluate each answer choice against those elements and the facts.\n"
    "Step 4 — CONCLUSION: Select the answer that best satisfies the rule given the facts."
)


class Condition7RuleCoL(BaseCondition):
    """Rule Extraction (Condition 2 Stage 1) + Chain of Logic (Condition 3)."""

    CONDITION_ID = 7
    CONDITION_NAME = "rule_col"

    def process_question(self, row):
        # Stage 1: extract rule (no logprobs)
        r1 = self.call_api(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _STAGE1_TEMPLATE.format(passage=row["gold_passage"])},
            ],
            max_tokens=300,
        )
        rule = (r1.choices[0].message.content or "").strip()

        # Stage 2: CoL with pre-supplied rule
        user_msg = (
            build_user_message(row)
            + _STAGE2_COL.format(rule=rule)
            + ANSWER_FORMAT
        )
        r2 = self.call_api(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
        )
        text2 = r2.choices[0].message.content or ""
        predicted = extract_answer(text2)

        return {
            "predicted_answer": predicted,
            "tokens_input": r1.usage.prompt_tokens + r2.usage.prompt_tokens,
            "tokens_output": r1.usage.completion_tokens + r2.usage.completion_tokens,
            "raw_response_1": rule,
            "raw_response_2": text2,
        }

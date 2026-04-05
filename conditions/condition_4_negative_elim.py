from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    extract_answer_logprobs,
    logprobs_to_confidence,
    ANSWER_FORMAT,
)

_INSTRUCTION = """

For each answer choice, work through the following elimination process:
1. Identify the legal doctrine or principle that choice invokes.
2. Check whether the gold passage supports or contradicts that doctrine.
3. Mark the choice as ELIMINATED (if contradicted or unsupported) or KEEP (if supported).

After evaluating all four choices, select the surviving answer."""


class Condition4NegativeElim(BaseCondition):
    CONDITION_ID = 4
    CONDITION_NAME = "negative_elimination"

    def process_question(self, row):
        user_msg = build_user_message(row) + _INSTRUCTION + ANSWER_FORMAT

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        r = self.call_api(messages, max_tokens=600, logprobs=True, top_logprobs=4)
        text = r.choices[0].message.content or ""
        predicted = extract_answer(text)
        lp = extract_answer_logprobs(r)

        return {
            "predicted_answer": predicted,
            "tokens_input": r.usage.prompt_tokens,
            "tokens_output": r.usage.completion_tokens,
            "raw_response_1": text,
            "raw_response_2": None,
            "logprob_A": lp["A"],
            "logprob_B": lp["B"],
            "logprob_C": lp["C"],
            "logprob_D": lp["D"],
            "confidence": logprobs_to_confidence(lp, predicted),
        }

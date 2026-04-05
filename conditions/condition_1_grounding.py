from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    extract_answer_logprobs,
    logprobs_to_confidence,
    ANSWER_FORMAT,
)

_INSTRUCTION = """

Answer this question using ONLY the information in the gold passage above.
Do NOT rely on prior legal knowledge that is not supported by the passage.
If the passage does not support a choice, treat that choice as wrong — even if \
you believe it is legally correct from memory."""


class Condition1Grounding(BaseCondition):
    CONDITION_ID = 1
    CONDITION_NAME = "grounding"

    def process_question(self, row):
        user_msg = build_user_message(row) + _INSTRUCTION + ANSWER_FORMAT

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        r = self.call_api(messages, max_tokens=150, logprobs=True, top_logprobs=4)
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

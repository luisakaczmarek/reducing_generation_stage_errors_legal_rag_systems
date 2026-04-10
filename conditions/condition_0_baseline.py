from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    ANSWER_FORMAT,
)


class Condition0Baseline(BaseCondition):
    CONDITION_ID = 0
    CONDITION_NAME = "baseline"

    def process_question(self, row):
        user_msg = build_user_message(row)
        user_msg += "\n\nAnswer the question based on the passage."
        user_msg += ANSWER_FORMAT

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        r = self.call_api(messages, max_tokens=150)
        text = r.choices[0].message.content or ""
        predicted = extract_answer(text)

        return {
            "predicted_answer": predicted,
            "tokens_input": r.usage.prompt_tokens,
            "tokens_output": r.usage.completion_tokens,
            "raw_response_1": text,
            "raw_response_2": None,
        }

from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    ANSWER_FORMAT,
)

_STAGE1_SUFFIX = (
    "\n\nAnswer the question based on the passage. "
    "Provide your answer and a brief justification."
)

_STAGE2_TEMPLATE = (
    "\n\nYour previous response was:\n{response_1}\n\n"
    "Now verify your answer:\n"
    "1. For each claim in your justification, check whether it is directly supported "
    "by the gold passage.\n"
    "2. Flag any claim that goes beyond what the passage states.\n"
    "3. If your claims are not all supported by the passage, revise your answer.\n"
    "4. Confirm or revise your final answer."
)


class Condition5Verification(BaseCondition):
    CONDITION_ID = 5
    CONDITION_NAME = "answer_verification"

    def process_question(self, row):
        # Stage 1: initial answer + justification (no logprobs)
        user_msg_1 = build_user_message(row) + _STAGE1_SUFFIX + ANSWER_FORMAT
        r1 = self.call_api(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg_1},
            ],
            max_tokens=300,
        )
        response_1 = r1.choices[0].message.content or ""

        # Stage 2: verify and revise
        user_msg_2 = (
            build_user_message(row)
            + _STAGE2_TEMPLATE.format(response_1=response_1)
            + ANSWER_FORMAT
        )
        r2 = self.call_api(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg_2},
            ],
            max_tokens=300,
        )
        text2 = r2.choices[0].message.content or ""
        predicted = extract_answer(text2)

        return {
            "predicted_answer": predicted,
            "tokens_input": r1.usage.prompt_tokens + r2.usage.prompt_tokens,
            "tokens_output": r1.usage.completion_tokens + r2.usage.completion_tokens,
            "raw_response_1": response_1,
            "raw_response_2": text2,
        }

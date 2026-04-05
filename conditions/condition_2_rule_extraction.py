from .base import (
    BaseCondition,
    build_user_message,
    extract_answer,
    extract_answer_logprobs,
    logprobs_to_confidence,
    ANSWER_FORMAT,
)

_STAGE1_TEMPLATE = (
    "GOLD PASSAGE:\n{passage}\n\n"
    "Extract the core legal rule or principle from this passage in exactly one sentence. "
    "Respond with only that one sentence."
)

_STAGE2_ADDITION = (
    "\n\nEXTRACTED LEGAL RULE: {rule}\n\n"
    "Using the extracted legal rule and the gold passage, select the correct answer."
)


class Condition2RuleExtraction(BaseCondition):
    CONDITION_ID = 2
    CONDITION_NAME = "rule_extraction"

    def process_question(self, row):
        # Stage 1: extract the rule (no logprobs)
        s1_msg = _STAGE1_TEMPLATE.format(passage=row["gold_passage"])
        r1 = self.call_api(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": s1_msg},
            ],
            max_tokens=300,
        )
        rule = (r1.choices[0].message.content or "").strip()

        # Stage 2: answer using extracted rule (logprobs on final call)
        user_msg = (
            build_user_message(row)
            + _STAGE2_ADDITION.format(rule=rule)
            + ANSWER_FORMAT
        )
        r2 = self.call_api(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            logprobs=True,
            top_logprobs=4,
        )
        text2 = r2.choices[0].message.content or ""
        predicted = extract_answer(text2)
        lp = extract_answer_logprobs(r2)

        return {
            "predicted_answer": predicted,
            "tokens_input": r1.usage.prompt_tokens + r2.usage.prompt_tokens,
            "tokens_output": r1.usage.completion_tokens + r2.usage.completion_tokens,
            "raw_response_1": rule,
            "raw_response_2": text2,
            "logprob_A": lp["A"],
            "logprob_B": lp["B"],
            "logprob_C": lp["C"],
            "logprob_D": lp["D"],
            "confidence": logprobs_to_confidence(lp, predicted),
        }

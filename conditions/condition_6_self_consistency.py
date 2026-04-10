from collections import Counter

from .base import BaseCondition, build_user_message, extract_answer, ANSWER_FORMAT

_SUFFIX = "\n\nAnswer the question based on the passage."


class Condition6SelfConsistency(BaseCondition):
    CONDITION_ID = 6
    CONDITION_NAME = "self_consistency"
    N_SAMPLES = 3

    def process_question(self, row):
        user_msg = build_user_message(row) + _SUFFIX + ANSWER_FORMAT
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        responses, answers = [], []
        tokens_in, tokens_out = 0, 0

        for _ in range(self.N_SAMPLES):
            r = self.call_api(messages, max_tokens=150, temperature=0.7)
            text = r.choices[0].message.content or ""
            responses.append(text)
            answers.append(extract_answer(text))
            tokens_in += r.usage.prompt_tokens
            tokens_out += r.usage.completion_tokens

        # Majority vote; tie → first call's answer
        valid = [a for a in answers if a != "PARSE_ERROR"]
        if valid:
            counts = Counter(valid)
            top_count = max(counts.values())
            winners = [k for k, v in counts.items() if v == top_count]
            final = winners[0] if len(winners) == 1 else answers[0]
            confidence = counts.get(final, 0) / self.N_SAMPLES
        else:
            final = "PARSE_ERROR"
            confidence = 0.0

        return {
            "predicted_answer": final,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "raw_response_1": responses[0] if len(responses) > 0 else "",
            "raw_response_2": responses[1] if len(responses) > 1 else "",
            "raw_response_3": responses[2] if len(responses) > 2 else "",
            "confidence": confidence,
        }

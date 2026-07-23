"""Output validation shared by inference and retrieval."""

import json


def is_valid_cot(value) -> bool:
    if not isinstance(value, list) or not value:
        return False
    try:
        for step in value:
            if not isinstance(step, dict):
                return False
            for key in ("question", "answer", "next_question"):
                if not isinstance(step.get(key), str):
                    return False
            if not step["question"] or not step["answer"]:
                return False
        json.dumps(value)
    except (KeyError, TypeError, ValueError):
        return False
    return True

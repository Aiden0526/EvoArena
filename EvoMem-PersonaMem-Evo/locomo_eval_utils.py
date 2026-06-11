"""Shared helpers for fair and reproducible LoCoMo evaluation prompts."""

from __future__ import annotations

import random


CAT5_NOT_MENTIONED = "Not mentioned in the conversation"


def _category5_answer_options(question: str, answer: str | None) -> tuple[str, str]:
    answer_text = (answer or "").strip()
    if random.random() < 0.5:
        return CAT5_NOT_MENTIONED, answer_text
    return answer_text, CAT5_NOT_MENTIONED


def build_locomo_answer_instruction(question: str, category: int, answer: str | None) -> str:
    if category == 5:
        first_option, second_option = _category5_answer_options(question, answer)
        return (
            f"Answer the following question: {question}. "
            f"Select the correct answer: {first_option} or {second_option}. "
            "Return only the short answer."
        )
    if category == 2:
        return (
            "Use DATE of CONVERSATION to answer with an approximate date. "
            "Generate the shortest possible answer, using words from the conversation where possible, "
            "and avoid using any subjects."
        )
    return "Write an answer in the form of a short phrase. Use exact words from the context whenever possible."


def build_locomo_answer_prompt(question: str, category: int, answer: str | None, context: str) -> str:
    if category == 5:
        first_option, second_option = _category5_answer_options(question, answer)
        return (
            f"Based on the context: {context}, answer the following question. {question}\n\n"
            f"Select the correct answer: {first_option} or {second_option}  Short answer:"
        )
    if category == 2:
        return (
            f"Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.\n"
            "Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.\n\n"
            f"Question: {question} Short answer:"
        )
    return (
        f"Based on the context: {context}, write an answer in the form of a short phrase for the following question. "
        "Answer with exact words from the context whenever possible.\n\n"
        f"Question: {question} Short answer:"
    )


def locomo_answer_temperature(category: int, temperature_c5: float) -> float:
    if category == 5:
        return temperature_c5
    return 0.7

def get_locomo_prompt_answer(category: int, answer: str | None, adversarial_answer: str | None) -> str | None:
    if category == 5:
        return adversarial_answer
    return answer


def get_locomo_reference_answer(category: int, answer: str | None) -> str | None:
    if category == 5:
        return CAT5_NOT_MENTIONED
    return answer

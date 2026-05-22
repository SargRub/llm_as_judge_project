from __future__ import annotations

PAIRWISE_BASIC = """You are an impartial evaluator. Choose which assistant answer is better for the user question.
Return exactly one label: A, B, or TIE.

Question:
{prompt}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better? Return only A, B, or TIE."""

PAIRWISE_RUBRIC = """You are an impartial evaluator. Use this rubric, in order of importance:
1. Correctness and factual accuracy.
2. Directly answering the user question.
3. Reasoning quality and completeness.
4. Clarity and concision.
Do not reward unnecessary length, confidence, politeness, or fancy style if the answer is wrong.
Return exactly one label: A, B, or TIE.

Question:
{prompt}

Answer A:
{answer_a}

Answer B:
{answer_b}

Best answer according to the rubric? Return only A, B, or TIE."""

PAIRWISE_REASONING = """You are evaluating two answers. Think briefly about correctness first, then choose.
Your final line must be exactly: FINAL: A, FINAL: B, or FINAL: TIE.

Question: {prompt}

Answer A: {answer_a}

Answer B: {answer_b}
"""

POINTWISE = """Rate the answer to the user question on a 1-5 scale.
5 = completely correct and helpful; 1 = incorrect or unhelpful.
Return only the number.

Question:
{prompt}

Answer:
{answer}

Score:"""


def build_pairwise_prompt(prompt: str, answer_a: str, answer_b: str, mode: str = "basic") -> str:
    if mode == "rubric":
        template = PAIRWISE_RUBRIC
    elif mode == "reasoning":
        template = PAIRWISE_REASONING
    else:
        template = PAIRWISE_BASIC
    return template.format(prompt=prompt, answer_a=answer_a, answer_b=answer_b)


def build_pointwise_prompt(prompt: str, answer: str) -> str:
    return POINTWISE.format(prompt=prompt, answer=answer)

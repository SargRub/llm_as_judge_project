from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .parsing import parse_pairwise_label, parse_score
from .prompts import build_pairwise_prompt, build_pointwise_prompt


@dataclass
class JudgeResult:
    raw_output: str
    parsed: str | float | None


class BaseJudge(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def judge_pairwise(self, question: str, answer_a: str, answer_b: str, mode: str = "basic") -> JudgeResult:
        prompt = build_pairwise_prompt(question, answer_a, answer_b, mode=mode)
        raw = self.generate(prompt)
        return JudgeResult(raw_output=raw, parsed=parse_pairwise_label(raw))

    def judge_pointwise(self, question: str, answer: str) -> JudgeResult:
        prompt = build_pointwise_prompt(question, answer)
        raw = self.generate(prompt)
        return JudgeResult(raw_output=raw, parsed=parse_score(raw))


class HeuristicJudge(BaseJudge):
    """Deterministic baseline used for smoke tests and ablations.

    It intentionally has mild length/style bias so the analysis pipeline has
    meaningful behavior even without an LLM.
    """

    name = "heuristic"

    POSITIVE_MARKERS = ["correct", "because", "therefore", "so", "=", "yes", "no"]
    WRONG_MARKERS = ["definitely", "always", "universally", "spectacularly", "magnificent"]

    def generate(self, prompt: str) -> str:
        # Pointwise prompt
        if "Rate the answer" in prompt:
            answer = prompt.split("Answer:")[-1]
            return str(round(self._score(answer)))

        a = self._extract(prompt, "Answer A:", "Answer B:")
        b = self._extract(prompt, "Answer B:", "Which answer")
        if not b:
            b = prompt.split("Answer B:")[-1]
        score_a = self._score(a)
        score_b = self._score(b)
        if abs(score_a - score_b) < 0.15:
            return "TIE"
        return "A" if score_a > score_b else "B"

    def _extract(self, text: str, start: str, end: str) -> str:
        if start not in text:
            return ""
        part = text.split(start, 1)[1]
        if end in part:
            part = part.split(end, 1)[0]
        return part.strip()

    def _score(self, text: str) -> float:
        lower = text.lower()
        score = 2.5
        score += 0.4 * sum(m in lower for m in self.POSITIVE_MARKERS)
        score -= 0.5 * sum(m in lower for m in self.WRONG_MARKERS)
        # Mild length bias: extra words help a little, but too many filler words hurt.
        words = re.findall(r"\w+", lower)
        score += min(len(words), 80) / 200.0
        if "do not add new evidence" in lower or "only to test" in lower:
            score -= 0.2
        return score


class HFJudge(BaseJudge):
    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 96, device: str = "auto"):
        self.name = f"hf:{model_name}"
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers, torch, accelerate, and sentencepiece to use HFJudge.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


class OpenAIJudge(BaseJudge):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.name = f"openai:{model_name}"
        self.model_name = model_name
        self.temperature = temperature
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use OpenAIJudge.") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


def make_judge(backend: str, model_name: Optional[str] = None, max_new_tokens: int = 96, temperature: float = 0.0, device: str = "auto") -> BaseJudge:
    backend = backend.lower()
    if backend == "heuristic":
        return HeuristicJudge()
    if backend == "hf":
        return HFJudge(model_name=model_name or "google/flan-t5-base", max_new_tokens=max_new_tokens, device=device)
    if backend == "openai":
        return OpenAIJudge(model_name=model_name or "gpt-4o-mini", temperature=temperature)
    raise ValueError(f"Unknown backend: {backend}")

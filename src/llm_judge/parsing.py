from __future__ import annotations

import re
from typing import Optional


def parse_pairwise_label(text: str) -> str:
    """Parse A/B/TIE from model output robustly."""
    if text is None:
        return "INVALID"
    s = str(text).strip().upper()
    final = re.search(r"FINAL\s*:\s*(A|B|TIE)\b", s)
    if final:
        return final.group(1)
    exact = re.fullmatch(r"\s*(A|B|TIE)\s*[\.]?\s*", s)
    if exact:
        return exact.group(1)
    patterns = [
        (r"\bANSWER\s*A\b|\bOPTION\s*A\b|\bA\s+IS\s+BETTER\b", "A"),
        (r"\bANSWER\s*B\b|\bOPTION\s*B\b|\bB\s+IS\s+BETTER\b", "B"),
        (r"\bTIE\b|\bDRAW\b|\bEQUAL\b", "TIE"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, s):
            return label
    # fallback: first standalone A/B/TIE token
    m = re.search(r"\b(A|B|TIE)\b", s)
    return m.group(1) if m else "INVALID"


def parse_score(text: str) -> Optional[float]:
    """Parse a 1-5 score from model output."""
    if text is None:
        return None
    m = re.search(r"([1-5](?:\.\d+)?)", str(text))
    if not m:
        return None
    score = float(m.group(1))
    return max(1.0, min(5.0, score))

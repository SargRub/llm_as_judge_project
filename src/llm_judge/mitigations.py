from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd


def invert_label(label: str) -> str:
    return {"A": "B", "B": "A", "TIE": "TIE"}.get(label, "INVALID")


def majority_vote(labels: Iterable[str]) -> str:
    valid = [x for x in labels if x in {"A", "B", "TIE"}]
    if not valid:
        return "INVALID"
    counts = Counter(valid)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "TIE"
    return top[0][0]


def position_swap_average(original: pd.DataFrame, swapped: pd.DataFrame) -> pd.DataFrame:
    """Combine original and swapped judgments into a position-invariant label."""
    o = original.copy()
    s = swapped.copy()
    o["base_id"] = o["id"].astype(str).str.replace("_rand_swap", "", regex=False)
    s["base_id"] = s["id"].astype(str).str.replace("_swapped", "", regex=False).str.replace("_rand_swap", "", regex=False)
    merged = o.merge(s[["base_id", "judge_label", "raw_output"]], on="base_id", suffixes=("_orig", "_swap"))
    merged["swap_label_mapped"] = merged["judge_label_swap"].map(invert_label)
    merged["judge_label"] = merged.apply(lambda r: majority_vote([r["judge_label_orig"], r["swap_label_mapped"]]), axis=1)
    merged["mode"] = "mitigation_position_swap"
    merged["raw_output"] = merged["raw_output_orig"] + " || SWAPPED: " + merged["raw_output_swap"]
    keep = ["id", "category", "prompt", "answer_a", "answer_b", "human_winner", "mode", "judge_label", "raw_output"]
    return merged[keep]


def ensemble_vote(frames: list[pd.DataFrame], mode_name: str = "mitigation_ensemble") -> pd.DataFrame:
    """Ensemble labels from several already-run modes/backends by majority vote."""
    if not frames:
        raise ValueError("No frames provided for ensemble")
    base = frames[0].copy()
    label_cols = []
    for i, frame in enumerate(frames):
        col = f"label_{i}"
        tmp = frame[["id", "judge_label"]].rename(columns={"judge_label": col})
        base = base.merge(tmp, on="id", how="left") if i > 0 else base.assign(**{col: frame["judge_label"].values})
        label_cols.append(col)
    base["judge_label"] = base[label_cols].apply(lambda row: majority_vote(row.tolist()), axis=1)
    base["mode"] = mode_name
    base["raw_output"] = base[label_cols].astype(str).agg(" | ".join, axis=1)
    return base[["id", "category", "prompt", "answer_a", "answer_b", "human_winner", "mode", "judge_label", "raw_output"]]

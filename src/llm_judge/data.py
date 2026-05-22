from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class PairwiseExample:
    id: str
    category: str
    prompt: str
    answer_a: str
    answer_b: str
    human_winner: str
    notes: str = ""


def load_benchmark(path: str | Path, limit: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Load the curated benchmark and validate required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    df = pd.read_csv(path, encoding="cp1252")
    required = {"id", "category", "prompt", "answer_a", "answer_b", "human_winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark is missing columns: {sorted(missing)}")
    df["human_winner"] = df["human_winner"].str.upper().str.strip()
    bad = set(df["human_winner"]) - {"A", "B", "TIE"}
    if bad:
        raise ValueError(f"Invalid human_winner labels: {bad}")
    if limit is not None:
        df = df.sample(frac=1.0, random_state=seed).head(limit).sort_values("id")
    return df.reset_index(drop=True)


def swapped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where answer A/B positions are swapped and human label is updated."""
    out = df.copy()
    out["answer_a"], out["answer_b"] = df["answer_b"], df["answer_a"]
    out["human_winner"] = df["human_winner"].map({"A": "B", "B": "A", "TIE": "TIE"})
    out["id"] = out["id"].astype(str) + "_swapped"
    return out


def length_stress_dataframe(df: pd.DataFrame, target: str = "B") -> pd.DataFrame:
    """Make one response artificially longer using irrelevant but fluent filler.

    The goal is not to improve the answer, but to test whether judges reward length.
    """
    filler = (
        "\n\nAdditional explanation: The situation can be discussed from several angles, "
        "but these extra sentences do not add new evidence or correct the core answer. "
        "They are included only to test whether a judge favors longer responses."
    )
    out = df.copy()
    col = "answer_a" if target.upper() == "A" else "answer_b"
    out[col] = out[col].astype(str) + filler
    out["id"] = out["id"].astype(str) + f"_length_{target.upper()}"
    return out


def style_stress_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select examples explicitly designed to test style-over-substance bias."""
    subset = df[df["category"].astype(str).str.contains("style_decoy", case=False, na=False)].copy()
    if subset.empty:
        subset = df.copy()
    subset["id"] = subset["id"].astype(str) + "_style"
    return subset.reset_index(drop=True)


def shuffle_positions(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Randomly swap positions to avoid all correct answers being in A."""
    rng = random.Random(seed)
    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        if rng.random() < 0.5:
            a, b = row["answer_a"], row["answer_b"]
            row["answer_a"], row["answer_b"] = b, a
            row["human_winner"] = {"A": "B", "B": "A", "TIE": "TIE"}[row["human_winner"]]
            row["id"] = str(row["id"]) + "_rand_swap"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)

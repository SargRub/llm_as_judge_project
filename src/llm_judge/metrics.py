from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def pairwise_accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    valid = [i for i, p in enumerate(y_pred) if p in {"A", "B", "TIE"}]
    if not valid:
        return 0.0
    return accuracy_score([y_true[i] for i in valid], [y_pred[i] for i in valid])


def agreement_summary(df: pd.DataFrame, mode_col: str = "mode") -> pd.DataFrame:
    rows = []
    for mode, g in df.groupby(mode_col):
        valid = g[g["judge_label"].isin(["A", "B", "TIE"])]
        acc = pairwise_accuracy(valid["human_winner"], valid["judge_label"])
        kappa = cohen_kappa_score(valid["human_winner"], valid["judge_label"], labels=["A", "B", "TIE"]) if len(valid) > 1 else np.nan
        rows.append({
            "mode": mode,
            "n": len(g),
            "valid_n": len(valid),
            "invalid_n": len(g) - len(valid),
            "accuracy": acc,
            "cohen_kappa": kappa,
            "pref_A_rate": (valid["judge_label"] == "A").mean() if len(valid) else np.nan,
            "pref_B_rate": (valid["judge_label"] == "B").mean() if len(valid) else np.nan,
            "tie_rate": (valid["judge_label"] == "TIE").mean() if len(valid) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mode")


def confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = ["A", "B", "TIE"]
    cm = confusion_matrix(df["human_winner"], df["judge_label"], labels=labels)
    return pd.DataFrame(cm, index=[f"human_{x}" for x in labels], columns=[f"judge_{x}" for x in labels])


def pointwise_calibration(scores: pd.DataFrame) -> pd.DataFrame:
    """Compare pointwise score assigned to known-good and known-bad answers."""
    rows = []
    for kind, g in scores.groupby("answer_kind"):
        rows.append({
            "answer_kind": kind,
            "n": len(g),
            "mean_score": g["score"].mean(),
            "std_score": g["score"].std(ddof=0),
        })
    return pd.DataFrame(rows)

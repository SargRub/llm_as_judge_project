from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy(summary: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    fig = plt.figure(figsize=(9, 5))
    plt.bar(summary["mode"], summary["accuracy"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Agreement with human labels")
    plt.title("LLM-as-Judge Agreement by Mode")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bias_report(bias_df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    rows = []
    for _, row in bias_df.iterrows():
        for col in bias_df.columns:
            if col in {"bias", "n", "note", "lengthened_side"}:
                continue
            val = row[col]
            if isinstance(val, (int, float)) and pd.notna(val):
                rows.append({"bias_metric": f"{row['bias']}:{col}", "value": val})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(plot_df["bias_metric"], plot_df["value"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Rate / Score")
    plt.title("Bias Diagnostics")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

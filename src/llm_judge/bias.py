from __future__ import annotations

import pandas as pd


def position_bias(original: pd.DataFrame, swapped: pd.DataFrame) -> dict:
    """Measure inconsistency and preference for first position under swaps."""
    o = original.copy()
    s = swapped.copy()
    o["base_id"] = o["id"].astype(str).str.replace("_rand_swap", "", regex=False)
    s["base_id"] = s["id"].astype(str).str.replace("_swapped", "", regex=False).str.replace("_rand_swap", "", regex=False)
    merged = o.merge(s[["base_id", "judge_label"]], on="base_id", suffixes=("_orig", "_swap"))

    def mapped_swap_label(x: str) -> str:
        return {"A": "B", "B": "A", "TIE": "TIE"}.get(x, "INVALID")

    merged["swap_mapped_to_original"] = merged["judge_label_swap"].map(mapped_swap_label)
    merged["consistent"] = merged["judge_label_orig"] == merged["swap_mapped_to_original"]
    return {
        "bias": "position",
        "n": len(merged),
        "consistency_rate": merged["consistent"].mean() if len(merged) else None,
        "orig_pref_A_rate": (merged["judge_label_orig"] == "A").mean() if len(merged) else None,
        "swap_pref_A_rate": (merged["judge_label_swap"] == "A").mean() if len(merged) else None,
        "estimated_position_bias_strength": abs((merged["judge_label_orig"] == "A").mean() - (merged["judge_label_swap"] == "B").mean()) if len(merged) else None,
    }


def length_bias(baseline: pd.DataFrame, stressed: pd.DataFrame, lengthened_side: str = "B") -> dict:
    """How often judgments move toward the artificially lengthened side."""
    b = baseline.copy()
    s = stressed.copy()
    b["base_id"] = b["id"].astype(str).str.replace("_rand_swap", "", regex=False)
    s["base_id"] = s["id"].astype(str).str.replace(f"_length_{lengthened_side.upper()}", "", regex=False).str.replace("_rand_swap", "", regex=False)
    merged = b.merge(s[["base_id", "judge_label"]], on="base_id", suffixes=("_base", "_stress"))
    side = lengthened_side.upper()
    moved = (merged["judge_label_base"] != side) & (merged["judge_label_stress"] == side)
    return {
        "bias": "length",
        "n": len(merged),
        "lengthened_side": side,
        "move_to_lengthened_rate": moved.mean() if len(merged) else None,
        "pref_lengthened_baseline": (merged["judge_label_base"] == side).mean() if len(merged) else None,
        "pref_lengthened_stress": (merged["judge_label_stress"] == side).mean() if len(merged) else None,
    }


def style_bias(style_df: pd.DataFrame) -> dict:
    """In curated style_decoy rows, B is polished/confident but wrong; A is correct."""
    if style_df.empty:
        return {"bias": "style_over_substance", "n": 0}
    return {
        "bias": "style_over_substance",
        "n": len(style_df),
        "accuracy_on_style_decoys": (style_df["judge_label"] == style_df["human_winner"]).mean(),
        "wrong_stylish_B_preference_rate": (style_df["judge_label"] == "B").mean(),
    }


def self_preference(df: pd.DataFrame, judge_name: str) -> dict:
    """Optional self-preference proxy if columns model_a/model_b exist."""
    if "model_a" not in df.columns or "model_b" not in df.columns:
        return {"bias": "self_preference", "n": 0, "note": "No model_a/model_b metadata in dataset."}
    name = judge_name.lower()
    mask_a = df["model_a"].astype(str).str.lower().apply(lambda x: x in name or name in x)
    mask_b = df["model_b"].astype(str).str.lower().apply(lambda x: x in name or name in x)
    relevant = df[mask_a | mask_b].copy()
    if relevant.empty:
        return {"bias": "self_preference", "n": 0, "note": "No rows where judge model appears as answer author."}
    relevant["preferred_self"] = ((mask_a & (relevant["judge_label"] == "A")) | (mask_b & (relevant["judge_label"] == "B")))
    return {"bias": "self_preference", "n": len(relevant), "preferred_self_rate": relevant["preferred_self"].mean()}

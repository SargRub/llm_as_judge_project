from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from tqdm import tqdm

from .bias import length_bias, position_bias, self_preference, style_bias
from .data import length_stress_dataframe, load_benchmark, shuffle_positions, style_stress_dataframe, swapped_dataframe
from .judges import make_judge
from .metrics import agreement_summary, confusion_table, pointwise_calibration
from .mitigations import ensemble_vote, position_swap_average
from .visualize import plot_accuracy, plot_bias_report


def run_pairwise(df: pd.DataFrame, judge, mode: str, prompt_mode: str = "basic") -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging {mode}"):
        result = judge.judge_pairwise(row["prompt"], row["answer_a"], row["answer_b"], mode=prompt_mode)
        out = row.to_dict()
        out.update({"mode": mode, "judge_name": judge.name, "judge_label": result.parsed, "raw_output": result.raw_output})
        rows.append(out)
    return pd.DataFrame(rows)


def run_pointwise(df: pd.DataFrame, judge) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pointwise calibration"):
        # Score the human-preferred and non-preferred answer separately.
        if row["human_winner"] == "A":
            good, bad = row["answer_a"], row["answer_b"]
        elif row["human_winner"] == "B":
            good, bad = row["answer_b"], row["answer_a"]
        else:
            good, bad = row["answer_a"], row["answer_b"]
        for kind, answer in [("human_preferred", good), ("human_nonpreferred", bad)]:
            res = judge.judge_pointwise(row["prompt"], answer)
            rows.append({
                "id": row["id"],
                "category": row["category"],
                "answer_kind": kind,
                "score": res.parsed,
                "raw_output": res.raw_output,
                "judge_name": judge.name,
            })
    return pd.DataFrame(rows)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge meta-evaluation experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["heuristic", "hf", "openai"], default=None)
    parser.add_argument("--model", default=None, help="HF model name or OpenAI model name depending on backend.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    backend = args.backend or cfg["judge"].get("backend", "heuristic")
    model_name = args.model or (cfg["judge"].get("model_name") if backend == "hf" else cfg["judge"].get("openai_model"))

    results_dir = Path(cfg["data"].get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = cfg["data"].get("path", "data/judge_benchmark.csv")
    limit = args.limit if args.limit is not None else cfg.get("run", {}).get("limit")
    df = load_benchmark(data_path, limit=limit, seed=cfg.get("seed", 42))
    df = shuffle_positions(df, seed=cfg.get("seed", 42))

    judge = make_judge(
        backend=backend,
        model_name=model_name,
        max_new_tokens=cfg["judge"].get("max_new_tokens", 96),
        temperature=cfg["judge"].get("temperature", 0.0),
        device=cfg["judge"].get("device", "auto"),
    )

    baseline = run_pairwise(df, judge, mode="baseline_pairwise", prompt_mode="basic")
    swapped = run_pairwise(swapped_dataframe(df), judge, mode="position_swap_raw", prompt_mode="basic")
    rubric = run_pairwise(df, judge, mode="rubric", prompt_mode="rubric")
    reasoning = run_pairwise(df, judge, mode="reasoning_prompt", prompt_mode="reasoning")
    length_stress = run_pairwise(length_stress_dataframe(df, target="B"), judge, mode="length_stress", prompt_mode="basic")
    style_stress = run_pairwise(style_stress_dataframe(df), judge, mode="style_stress", prompt_mode="basic")

    pos_mitigated = position_swap_average(baseline, swapped)
    ensemble = ensemble_vote([baseline, rubric, reasoning], mode_name="mitigation_ensemble")

    all_pairwise = pd.concat([baseline, swapped, rubric, reasoning, length_stress, style_stress, pos_mitigated, ensemble], ignore_index=True)
    all_pairwise.to_csv(results_dir / "raw_judgments.csv", index=False)

    summary = agreement_summary(all_pairwise)
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)

    confusion_table(baseline[baseline["judge_label"].isin(["A", "B", "TIE"])]).to_csv(results_dir / "confusion_baseline.csv")

    bias_rows = [
        position_bias(baseline, swapped),
        length_bias(baseline, length_stress, lengthened_side="B"),
        style_bias(style_stress),
        self_preference(baseline, judge.name),
    ]
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(results_dir / "bias_report.csv", index=False)

    mitigation_df = summary[summary["mode"].str.contains("mitigation|rubric|reasoning|baseline", regex=True)].copy()
    mitigation_df.to_csv(results_dir / "mitigation_report.csv", index=False)

    pointwise = run_pointwise(df, judge)
    pointwise.to_csv(results_dir / "pointwise_scores.csv", index=False)
    calibration = pointwise_calibration(pointwise.dropna(subset=["score"]))
    calibration.to_csv(results_dir / "calibration_report.csv", index=False)

    plot_accuracy(summary, results_dir / "accuracy_by_mode.png")
    plot_bias_report(bias_df, results_dir / "bias_summary.png")

    print("\nDone. Key files written to:", results_dir.resolve())
    print(summary.to_string(index=False))
    print("\nBias report:")
    print(bias_df.to_string(index=False))


if __name__ == "__main__":
    main()

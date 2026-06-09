from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def load_available() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = []
    for suffix in ["et30", "rf30", ""]:
        choices = OUT / f"source5_repeated_choices_{suffix}.csv" if suffix else OUT / "source5_repeated_choices.csv"
        metrics = OUT / f"source5_repeated_metrics_{suffix}.csv" if suffix else OUT / "source5_repeated_metrics.csv"
        predictions = OUT / f"source5_repeated_predictions_{suffix}.csv" if suffix else OUT / "source5_repeated_predictions.csv"
        if choices.exists() and metrics.exists() and predictions.exists():
            pairs.append((suffix or "main", pd.read_csv(choices), pd.read_csv(metrics), pd.read_csv(predictions)))
    if not pairs:
        raise RuntimeError("No source5 repeated outputs were found.")
    choices = pd.concat([p[1] for p in pairs], ignore_index=True).drop_duplicates(["seed", "model"])
    metrics = pd.concat([p[2] for p in pairs], ignore_index=True).drop_duplicates(["seed", "model"])
    predictions = pd.concat([p[3] for p in pairs], ignore_index=True).drop_duplicates(["seed", "model", "sample_id"])
    return choices, metrics, predictions


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for selector, g in rows.groupby("selector"):
        row: dict[str, float | str | int] = {"selector": selector, "n_seeds": int(g["seed"].nunique())}
        for col in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "hard_negative_fpr",
            "fp",
            "fn",
            "median_delay_s",
            "p95_delay_s",
        ]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
            row[f"{col}_sem"] = float(g[col].std(ddof=1) / np.sqrt(len(g)))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["accuracy_mean", "specificity_mean"], ascending=False)


def main() -> None:
    choices, metrics, predictions = load_available()
    selected_rows = []
    selected_predictions = []
    for seed, group in choices.groupby("seed"):
        candidates = group.copy()
        candidates["selector_accuracy_score"] = (
            candidates["val_accuracy"]
            + 0.10 * candidates["val_f1"]
            + 0.05 * candidates["val_specificity"]
            - 0.03 * candidates["val_hard_negative_fpr"]
            - 0.015 * candidates["val_fp"]
            - 0.015 * candidates["val_fn"]
        )
        candidates["selector_specificity_score"] = (
            candidates["val_accuracy"]
            + 0.25 * candidates["val_specificity"]
            + 0.08 * candidates["val_f1"]
            + 0.04 * candidates["val_recall"]
            - 0.08 * candidates["val_hard_negative_fpr"]
            - 0.030 * candidates["val_fp"]
            - 0.030 * candidates["val_fn"]
        )
        for selector, score_col in [
            ("Source5_ValidationSelector_accuracy", "selector_accuracy_score"),
            ("Source5_ValidationSelector_specificity", "selector_specificity_score"),
        ]:
            chosen = candidates.sort_values(score_col, ascending=False).iloc[0]
            metric = metrics[(metrics["seed"].astype(int) == int(seed)) & (metrics["model"] == chosen["model"])]
            if metric.empty:
                continue
            row = metric.iloc[0].to_dict()
            row["selector"] = selector
            row["chosen_model"] = chosen["model"]
            row["selection_score"] = float(chosen[score_col])
            selected_rows.append(row)
            pred = predictions[(predictions["seed"].astype(int) == int(seed)) & (predictions["model"] == chosen["model"])].copy()
            pred["selector"] = selector
            pred["chosen_model"] = chosen["model"]
            selected_predictions.append(pred)
    detail = pd.DataFrame(selected_rows)
    detail.to_csv(OUT / "source5_validation_selector_metrics.csv", index=False, encoding="utf-8-sig")
    if selected_predictions:
        pd.concat(selected_predictions, ignore_index=True).to_csv(
            OUT / "source5_validation_selector_predictions.csv", index=False, encoding="utf-8-sig"
        )
    summary = summarize(detail)
    summary.to_csv(OUT / "source5_validation_selector_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
HORIZONS = tuple(int(x) for x in __import__("os").environ.get("HORIZONS", "50,75,100,150").split(",") if x.strip())
THRESHOLD_GRID = (0.60, 0.75, 0.88, 0.96)
INPUT_SUFFIX = __import__("os").environ.get("INPUT_SUFFIX", "balanced")
OUTPUT_SUFFIX = __import__("os").environ.get("OUTPUT_SUFFIX", INPUT_SUFFIX)
MODEL_NAME = __import__("os").environ.get("MODEL_NAME", "EarlyCascadeHGB_50_75_100_150")


def metric_row(frame: pd.DataFrame, thresholds: dict[int, float]) -> dict[str, float]:
    prob_matrix = frame[[f"prob_{horizon}s" for horizon in HORIZONS]].to_numpy(dtype=float)
    threshold_vector = np.array([thresholds[horizon] for horizon in HORIZONS], dtype=float)
    return metric_row_arrays(
        prob_matrix,
        frame["binary"].to_numpy(dtype=int),
        frame["onset_s"].to_numpy(dtype=float),
        frame["hard_negative"].to_numpy(dtype=int),
        threshold_vector,
    )


def metric_row_arrays(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    onset: np.ndarray,
    hard_negative: np.ndarray,
    threshold_vector: np.ndarray,
) -> dict[str, float]:
    hit_matrix = prob_matrix >= threshold_vector.reshape(1, -1)
    y_pred = hit_matrix.any(axis=1).astype(int)
    alarm = np.full(len(y_true), np.nan)
    for idx, horizon in enumerate(HORIZONS):
        new_hit = hit_matrix[:, idx] & np.isnan(alarm)
        alarm[new_hit] = horizon
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) else np.nan
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    delay = np.where((y_true == 1) & (y_pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    valid_delay = delay[np.isfinite(delay)]
    hard_norm = (y_true == 0) & (hard_negative == 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "hard_negative_fpr": float(y_pred[hard_norm].mean()) if hard_norm.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.median(valid_delay)) if len(valid_delay) else np.nan,
        "p95_delay_s": float(np.quantile(valid_delay, 0.95)) if len(valid_delay) else np.nan,
    }


def score_metrics(metrics: dict[str, float], objective: str) -> float:
    delay = metrics["median_delay_s"] if np.isfinite(metrics["median_delay_s"]) else 150.0
    if objective == "accuracy_specificity":
        return (
            metrics["accuracy"]
            + 0.18 * metrics["specificity"]
            + 0.12 * metrics["recall"]
            + 0.08 * metrics["f1"]
            - 0.035 * metrics["fp"]
            - delay / 2500.0
        )
    if objective == "balanced_accuracy":
        return (
            0.5 * (metrics["recall"] + metrics["specificity"])
            + 0.20 * metrics["accuracy"]
            + 0.05 * metrics["f1"]
            - 0.03 * metrics["fp"]
            - delay / 2800.0
        )
    if objective == "hard_negative_guard":
        return (
            metrics["accuracy"]
            + 0.20 * metrics["specificity"]
            + 0.10 * metrics["recall"]
            + 0.05 * metrics["f1"]
            - 0.12 * metrics["hard_negative_fpr"]
            - 0.04 * metrics["fp"]
            - delay / 2600.0
        )
    return metrics["f1"] + 0.30 * metrics["recall"] + 0.08 * metrics["specificity"] - 0.07 * metrics["fp"] - delay / 450.0


def summarize(rows: list[dict]) -> dict[str, float]:
    data = pd.DataFrame(rows)
    out: dict[str, float] = {"n_seeds": float(data["seed"].nunique())}
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
        out[f"{col}_mean"] = float(data[col].mean())
        out[f"{col}_std"] = float(data[col].std(ddof=1))
        out[f"{col}_min"] = float(data[col].min())
        out[f"{col}_max"] = float(data[col].max())
    return out


def frame_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame[[f"prob_{horizon}s" for horizon in HORIZONS]].to_numpy(dtype=float),
        frame["binary"].to_numpy(dtype=int),
        frame["onset_s"].to_numpy(dtype=float),
        frame["hard_negative"].to_numpy(dtype=int),
    )


def main() -> None:
    predictions = pd.read_csv(OUT / f"repeated_seed_predictions_{INPUT_SUFFIX}.csv")
    predictions = predictions[predictions["model"] == MODEL_NAME].copy()
    summary_rows = []
    detail_rows = []
    threshold_rows = []
    for objective in ["current", "accuracy_specificity", "balanced_accuracy", "hard_negative_guard"]:
        rows = []
        for seed in sorted(predictions["seed"].unique()):
            val = predictions[(predictions["seed"] == seed) & (predictions["split"] == "val")].copy()
            test = predictions[(predictions["seed"] == seed) & (predictions["split"] == "test")].copy()
            val_arrays = frame_arrays(val)
            test_arrays = frame_arrays(test)
            best = None
            for combo in product(THRESHOLD_GRID, repeat=len(HORIZONS)):
                thresholds = dict(zip(HORIZONS, combo))
                threshold_vector = np.array(combo, dtype=float)
                val_metrics = metric_row_arrays(*val_arrays, threshold_vector)
                score = score_metrics(val_metrics, objective)
                if best is None or score > best[0]:
                    best = (score, thresholds, val_metrics)
            assert best is not None
            test_threshold_vector = np.array([best[1][h] for h in HORIZONS], dtype=float)
            test_metrics = metric_row_arrays(*test_arrays, test_threshold_vector)
            test_metrics.update({"objective": objective, "seed": int(seed)})
            rows.append(test_metrics)
            threshold_rows.append(
                {
                    "objective": objective,
                    "seed": int(seed),
                    **{f"threshold_{h}s": best[1][h] for h in HORIZONS},
                }
            )
        detail_rows.extend(rows)
        summary = summarize(rows)
        summary["objective"] = objective
        summary_rows.append(summary)
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows).sort_values("accuracy_mean", ascending=False)
    thresholds = pd.DataFrame(threshold_rows)
    detail.to_csv(OUT / f"{OUTPUT_SUFFIX}_threshold_objective_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{OUTPUT_SUFFIX}_threshold_objective_summary.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(OUT / f"{OUTPUT_SUFFIX}_threshold_objective_thresholds.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

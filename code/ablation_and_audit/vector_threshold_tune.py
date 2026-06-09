from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
INPUT_SUFFIX = os.environ.get("INPUT_SUFFIX", "rescompact_hgb_ext")
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", f"{INPUT_SUFFIX}_vector_tuned")
MODEL_NAME = os.environ.get("MODEL_NAME", "EarlyCascadeHGB_50_75_100_150_250_400")
HORIZONS = tuple(int(x) for x in os.environ.get("HORIZONS", "50,75,100,150,250,400").split(",") if x.strip())
GRID = tuple(float(x) for x in os.environ.get("GRID", "0.55,0.65,0.75,0.85,0.92,0.96,0.985,0.995").split(",") if x.strip())
OBJECTIVE = os.environ.get("OBJECTIVE", "accuracy_guard")


def frame_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "prob": frame[[f"prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float),
        "y_true": frame["binary"].to_numpy(dtype=int),
        "hard_negative": frame["hard_negative"].to_numpy(dtype=int),
        "onset": frame["onset_s"].to_numpy(dtype=float),
    }


def metric_vectors(
    arrays: dict[str, np.ndarray],
    pred: np.ndarray,
    alarm: np.ndarray,
    include_delay: bool = False,
) -> dict[str, np.ndarray]:
    y_true = arrays["y_true"].astype(bool)
    y_false = ~y_true
    tp = (pred & y_true.reshape(1, -1)).sum(axis=1).astype(float)
    tn = ((~pred) & y_false.reshape(1, -1)).sum(axis=1).astype(float)
    fp = (pred & y_false.reshape(1, -1)).sum(axis=1).astype(float)
    fn = ((~pred) & y_true.reshape(1, -1)).sum(axis=1).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    hard = y_false & (arrays["hard_negative"] == 1)
    hard_fpr = pred[:, hard].mean(axis=1) if hard.any() else np.zeros(len(pred), dtype=float)
    if include_delay:
        delay = np.where(
            pred & y_true.reshape(1, -1) & np.isfinite(arrays["onset"]).reshape(1, -1),
            np.maximum(0.0, alarm - arrays["onset"].reshape(1, -1)),
            np.nan,
        )
        with np.errstate(all="ignore"):
            median_delay = np.nanmedian(delay, axis=1)
            p95_delay = np.nanquantile(delay, 0.95, axis=1)
    else:
        median_delay = np.zeros(len(pred), dtype=float)
        p95_delay = np.zeros(len(pred), dtype=float)
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "hard_negative_fpr": hard_fpr,
        "fp": fp,
        "fn": fn,
        "median_delay_s": median_delay,
        "p95_delay_s": p95_delay,
    }


def score(metrics: dict[str, np.ndarray]) -> np.ndarray:
    if OBJECTIVE == "accuracy_only":
        return metrics["accuracy"]
    if OBJECTIVE == "specificity_guard":
        return (
            metrics["accuracy"]
            + 0.20 * metrics["specificity"]
            + 0.12 * metrics["recall"]
            + 0.08 * metrics["f1"]
            - 0.08 * metrics["fp"]
            - 0.10 * metrics["hard_negative_fpr"]
        )
    if OBJECTIVE == "balanced_guard":
        return (
            0.55 * metrics["accuracy"]
            + 0.25 * (0.5 * (metrics["recall"] + metrics["specificity"]))
            + 0.10 * metrics["f1"]
            - 0.06 * metrics["fp"]
            - 0.04 * metrics["fn"]
            - 0.08 * metrics["hard_negative_fpr"]
        )
    return (
        metrics["accuracy"]
        + 0.14 * metrics["specificity"]
        + 0.12 * metrics["recall"]
        + 0.08 * metrics["f1"]
        - 0.065 * metrics["fp"]
        - 0.045 * metrics["fn"]
        - 0.10 * metrics["hard_negative_fpr"]
    )


def all_predictions(arrays: dict[str, np.ndarray], threshold_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hits = arrays["prob"][None, :, :] >= threshold_matrix[:, None, :]
    pred = hits.any(axis=2)
    first = np.argmax(hits, axis=2)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    return pred, alarm


def apply_thresholds(arrays: dict[str, np.ndarray], thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred, alarm = all_predictions(arrays, thresholds.reshape(1, -1))
    metrics = {key: float(value[0]) for key, value in metric_vectors(arrays, pred, alarm, include_delay=True).items()}
    metrics["fp"] = int(metrics["fp"])
    metrics["fn"] = int(metrics["fn"])
    return pred[0].astype(int), alarm[0], metrics


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": f"{MODEL_NAME}_{OBJECTIVE}", "n_seeds": int(data["seed"].nunique())}
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
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    threshold_matrix = np.array(list(product(GRID, repeat=len(HORIZONS))), dtype=float)
    predictions = pd.read_csv(OUT / f"repeated_seed_predictions_{INPUT_SUFFIX}.csv")
    predictions = predictions[predictions["model"] == MODEL_NAME].copy()
    detail_rows = []
    choice_rows = []
    pred_rows = []
    for seed in sorted(predictions["seed"].unique()):
        val = predictions[(predictions["seed"] == seed) & (predictions["split"] == "val")].copy()
        test = predictions[(predictions["seed"] == seed) & (predictions["split"] == "test")].copy()
        val_arrays = frame_arrays(val)
        val_pred, val_alarm = all_predictions(val_arrays, threshold_matrix)
        val_metrics = metric_vectors(val_arrays, val_pred, val_alarm, include_delay=False)
        best_idx = int(np.nanargmax(score(val_metrics)))
        thresholds = threshold_matrix[best_idx]
        test_arrays = frame_arrays(test)
        y_pred, alarm, test_metrics = apply_thresholds(test_arrays, thresholds)
        test_metrics.update({"seed": int(seed), "objective": OBJECTIVE})
        detail_rows.append(test_metrics)
        choice_rows.append({"seed": int(seed), "objective": OBJECTIVE, **{f"threshold_{h}s": thresholds[idx] for idx, h in enumerate(HORIZONS)}})
        pred = test[
            [
                "sample_id",
                "file_name",
                "binary",
                "hard_negative",
                "severity_name",
                "onset_s",
                "seed",
                "split",
            ]
        ].copy()
        pred["y_true"] = pred["binary"].astype(int)
        pred["y_pred"] = y_pred
        pred["alarm_time_s"] = alarm
        pred["delay_s"] = np.where(
            (pred["y_true"].to_numpy(dtype=int) == 1) & (y_pred == 1) & np.isfinite(pred["onset_s"].to_numpy(dtype=float)),
            np.maximum(0.0, alarm - pred["onset_s"].to_numpy(dtype=float)),
            np.nan,
        )
        pred["model"] = f"{MODEL_NAME}_{OBJECTIVE}"
        pred_rows.append(pred)
        print(
            f"completed seed {seed}: accuracy={test_metrics['accuracy']:.4f}, "
            f"fp={test_metrics['fp']}, fn={test_metrics['fn']}",
            flush=True,
        )
    detail = pd.DataFrame(detail_rows)
    choices = pd.DataFrame(choice_rows)
    out_predictions = pd.concat(pred_rows, ignore_index=True)
    summary = summarize(detail_rows)
    detail.to_csv(OUT / f"{OUTPUT_SUFFIX}_metrics.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{OUTPUT_SUFFIX}_choices.csv", index=False, encoding="utf-8-sig")
    out_predictions.to_csv(OUT / f"{OUTPUT_SUFFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{OUTPUT_SUFFIX}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

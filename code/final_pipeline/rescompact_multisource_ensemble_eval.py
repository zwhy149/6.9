from __future__ import annotations

from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150, 250, 400)
THRESHOLD_GRID = np.array([float(x) for x in os.environ.get("GRID", "0.55,0.65,0.75,0.85,0.95").split(",") if x.strip()], dtype=float)
CHUNK_SIZE = 30000
OBJECTIVE = os.environ.get("OBJECTIVE", "default")

SOURCES = [
    ("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("global_et", "repeated_seed_predictions_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
]
if os.environ.get("INCLUDE_GLOBAL_HGB", "0").lower() in {"1", "true", "yes"}:
    SOURCES.append(("global_hgb", "repeated_seed_predictions_global_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"))


def make_weights() -> np.ndarray:
    grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    weights = []
    def rec(prefix: list[float], remaining_slots: int, remaining_sum: float) -> None:
        if remaining_slots == 1:
            if any(abs(remaining_sum - x) < 1e-9 for x in grid):
                weights.append(tuple(prefix + [remaining_sum]))
            return
        for value in grid:
            if value <= remaining_sum + 1e-9:
                rec(prefix + [value], remaining_slots - 1, remaining_sum - value)
    rec([], len(SOURCES), 1.0)
    return np.array(weights, dtype=float)


WEIGHTS = make_weights()
THRESHOLDS = np.array(list(product(THRESHOLD_GRID, repeat=len(HORIZONS))), dtype=float)


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name)
    data = data[data["model"] == model_name].copy()
    meta_cols = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    out = data[meta_cols + prob_cols].copy()
    return out.rename(columns={col: f"{prefix}_{col}" for col in prob_cols})


def prepare() -> pd.DataFrame:
    base = None
    for prefix, file_name, model_name in SOURCES:
        part = load_source(prefix, file_name, model_name)
        prob_cols = [f"{prefix}_prob_{h}s" for h in HORIZONS]
        if base is None:
            base = part
        else:
            base = base.merge(part[["sample_id", "seed", "split", *prob_cols]], on=["sample_id", "seed", "split"], how="inner")
    assert base is not None
    return base


def frame_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "y_true": frame["binary"].to_numpy(dtype=int),
        "hard_negative": frame["hard_negative"].to_numpy(dtype=int),
        "onset": frame["onset_s"].to_numpy(dtype=float),
        "prob": np.stack(
            [
                frame[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
                for prefix, _, _ in SOURCES
            ],
            axis=0,
        ),
    }


def metric_vectors(arrays: dict[str, np.ndarray], pred: np.ndarray, alarm: np.ndarray, include_delay: bool = False) -> dict[str, np.ndarray]:
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
            + 0.10 * metrics["recall"]
            + 0.08 * metrics["f1"]
            - 0.08 * metrics["fp"]
            - 0.10 * metrics["hard_negative_fpr"]
        )
    return (
        metrics["accuracy"]
        + 0.16 * metrics["f1"]
        + 0.14 * metrics["recall"]
        + 0.18 * metrics["specificity"]
        - 0.06 * metrics["fp"]
        - 0.05 * metrics["fn"]
        - 0.10 * metrics["hard_negative_fpr"]
    )


def predict_chunk(arrays: dict[str, np.ndarray], weights: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = np.tensordot(weights, arrays["prob"], axes=([1], [0]))
    hits = prob >= thresholds[:, None, :]
    pred = hits.any(axis=2)
    first = np.argmax(hits, axis=2)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    return pred, alarm


def choose(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    best_score = -np.inf
    best_weight = WEIGHTS[0]
    best_threshold = THRESHOLDS[0]
    for weight in WEIGHTS:
        for start in range(0, len(THRESHOLDS), CHUNK_SIZE):
            thresholds = THRESHOLDS[start : start + CHUNK_SIZE]
            weights = np.repeat(weight.reshape(1, -1), len(thresholds), axis=0)
            pred, alarm = predict_chunk(arrays, weights, thresholds)
            metrics = metric_vectors(arrays, pred, alarm, include_delay=False)
            scores = score(metrics)
            idx = int(np.nanargmax(scores))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_weight = weight.copy()
                best_threshold = thresholds[idx].copy()
    return best_weight, best_threshold


def apply_choice(arrays: dict[str, np.ndarray], weight: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred, alarm = predict_chunk(arrays, weight.reshape(1, -1), threshold.reshape(1, -1))
    metrics = {key: float(value[0]) for key, value in metric_vectors(arrays, pred, alarm, include_delay=True).items()}
    metrics["fp"] = int(metrics["fp"])
    metrics["fn"] = int(metrics["fn"])
    return pred[0].astype(int), alarm[0], metrics


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": f"ResCompact_HGB_ET_GlobalET_Ensemble_{OBJECTIVE}", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    data = prepare()
    rows = []
    pred_rows = []
    choices = []
    for seed in sorted(data["seed"].unique()):
        val = data[(data["seed"] == seed) & (data["split"] == "val")].copy()
        test = data[(data["seed"] == seed) & (data["split"] == "test")].copy()
        weight, threshold = choose(frame_arrays(val))
        y_pred, alarm, metrics = apply_choice(frame_arrays(test), weight, threshold)
        metrics.update({"seed": int(seed)})
        rows.append(metrics)
        choices.append({"seed": int(seed), **{f"w_{SOURCES[i][0]}": weight[i] for i in range(len(SOURCES))}, **{f"threshold_{h}s": threshold[i] for i, h in enumerate(HORIZONS)}})
        pred = test[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]].copy()
        pred["y_true"] = pred["binary"].astype(int)
        pred["y_pred"] = y_pred
        pred["alarm_time_s"] = alarm
        pred["delay_s"] = np.where(
            (pred["y_true"].to_numpy(dtype=int) == 1) & (y_pred == 1) & np.isfinite(pred["onset_s"].to_numpy(dtype=float)),
            np.maximum(0.0, alarm - pred["onset_s"].to_numpy(dtype=float)),
            np.nan,
        )
        pred["model"] = f"ResCompact_HGB_ET_GlobalET_Ensemble_{OBJECTIVE}"
        pred_rows.append(pred)
        print(f"seed {seed}: accuracy={metrics['accuracy']:.4f}, fp={metrics['fp']}, fn={metrics['fn']}, weights={weight}", flush=True)
    detail = pd.DataFrame(rows)
    summary = summarize(rows)
    suffix = f"rescompact_multisource_{len(SOURCES)}src_ensemble_{OBJECTIVE}"
    detail.to_csv(OUT / f"{suffix}_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choices).to_csv(OUT / f"{suffix}_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / f"{suffix}_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{suffix}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

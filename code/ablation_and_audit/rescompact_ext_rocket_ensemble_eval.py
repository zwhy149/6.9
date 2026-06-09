from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150, 250, 400)
THRESHOLD_GRID = np.array([0.55, 0.65, 0.75, 0.85, 0.95], dtype=float)


def make_weights() -> np.ndarray:
    weights = []
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for wh in grid:
        for we in grid:
            wr = 1.0 - wh - we
            if wr < -1e-9:
                continue
            if any(abs(wr - x) < 1e-9 for x in grid):
                weights.append((wh, we, wr))
    return np.array(weights, dtype=float)


WEIGHTS = make_weights()


def make_configs() -> tuple[np.ndarray, np.ndarray]:
    weights = []
    thresholds = []
    for weight_idx in range(len(WEIGHTS)):
        for combo in product(THRESHOLD_GRID, repeat=len(HORIZONS)):
            weights.append(weight_idx)
            thresholds.append(combo)
    return np.array(weights, dtype=int), np.array(thresholds, dtype=float)


CONFIG_WEIGHT_IDX, CONFIG_THRESHOLDS = make_configs()


def prepare() -> pd.DataFrame:
    hgb = pd.read_csv(OUT / "repeated_seed_predictions_rescompact_hgb_ext.csv")
    et = pd.read_csv(OUT / "repeated_seed_predictions_rescompact_et_ext.csv")
    rocket = pd.read_csv(OUT / "prefix_rocket_400_predictions.csv")
    hgb = hgb[hgb["model"] == "EarlyCascadeHGB_50_75_100_150_250_400"].copy()
    et = et[et["model"] == "EarlyCascadeET_50_75_100_150_250_400"].copy()
    rocket = rocket[rocket["model"] == "PrefixROCKET_Logistic"].copy()
    meta_cols = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    base = hgb[meta_cols + prob_cols].merge(
        et[["sample_id", "seed", "split", *prob_cols]],
        on=["sample_id", "seed", "split"],
        suffixes=("_hgb", "_et"),
        how="inner",
    )
    base = base.merge(
        rocket[["sample_id", "seed", "split", *prob_cols]],
        on=["sample_id", "seed", "split"],
        how="inner",
    )
    return base.rename(columns={col: f"{col}_rocket" for col in prob_cols})


def frame_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "y_true": frame["binary"].to_numpy(dtype=int),
        "hard_negative": frame["hard_negative"].to_numpy(dtype=int),
        "onset": frame["onset_s"].to_numpy(dtype=float),
        "hgb": frame[[f"prob_{h}s_hgb" for h in HORIZONS]].to_numpy(dtype=float),
        "et": frame[[f"prob_{h}s_et" for h in HORIZONS]].to_numpy(dtype=float),
        "rocket": frame[[f"prob_{h}s_rocket" for h in HORIZONS]].to_numpy(dtype=float),
    }


def all_predictions(arrays: dict[str, np.ndarray], config_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if config_indices is None:
        weight_idx = CONFIG_WEIGHT_IDX
        thresholds = CONFIG_THRESHOLDS
    else:
        weight_idx = CONFIG_WEIGHT_IDX[config_indices]
        thresholds = CONFIG_THRESHOLDS[config_indices]
    w = WEIGHTS[weight_idx]
    prob = (
        w[:, 0, None, None] * arrays["hgb"][None, :, :]
        + w[:, 1, None, None] * arrays["et"][None, :, :]
        + w[:, 2, None, None] * arrays["rocket"][None, :, :]
    )
    hits = prob >= thresholds[:, None, :]
    pred = hits.any(axis=2)
    first = np.argmax(hits, axis=2)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    return pred, alarm


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
    return (
        metrics["accuracy"]
        + 0.16 * metrics["f1"]
        + 0.14 * metrics["recall"]
        + 0.18 * metrics["specificity"]
        - 0.06 * metrics["fp"]
        - 0.05 * metrics["fn"]
        - 0.10 * metrics["hard_negative_fpr"]
    )


def choose(arrays: dict[str, np.ndarray]) -> int:
    pred, alarm = all_predictions(arrays)
    metrics = metric_vectors(arrays, pred, alarm, include_delay=False)
    return int(np.nanargmax(score(metrics)))


def apply_choice(arrays: dict[str, np.ndarray], config_idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred, alarm = all_predictions(arrays, np.array([config_idx], dtype=int))
    metrics = {key: float(value[0]) for key, value in metric_vectors(arrays, pred, alarm, include_delay=True).items()}
    metrics["fp"] = int(metrics["fp"])
    metrics["fn"] = int(metrics["fn"])
    return pred[0].astype(int), alarm[0], metrics


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": "ResCompact_ExtHGB_ET_ROCKET_Ensemble", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    data = prepare()
    detail_rows = []
    pred_rows = []
    choice_rows = []
    for seed in sorted(data["seed"].unique()):
        val = data[(data["seed"] == seed) & (data["split"] == "val")].copy()
        test = data[(data["seed"] == seed) & (data["split"] == "test")].copy()
        config_idx = choose(frame_arrays(val))
        y_pred, alarm, metrics = apply_choice(frame_arrays(test), config_idx)
        metrics.update({"seed": int(seed)})
        detail_rows.append(metrics)
        choice_rows.append(
            {
                "seed": int(seed),
                "w_hgb": float(WEIGHTS[CONFIG_WEIGHT_IDX[config_idx], 0]),
                "w_et": float(WEIGHTS[CONFIG_WEIGHT_IDX[config_idx], 1]),
                "w_rocket": float(WEIGHTS[CONFIG_WEIGHT_IDX[config_idx], 2]),
                **{f"threshold_{h}s": float(CONFIG_THRESHOLDS[config_idx, idx]) for idx, h in enumerate(HORIZONS)},
            }
        )
        pred = test[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]].copy()
        pred["y_true"] = pred["binary"].astype(int)
        pred["y_pred"] = y_pred
        pred["alarm_time_s"] = alarm
        pred["delay_s"] = np.where(
            (pred["y_true"].to_numpy(dtype=int) == 1) & (y_pred == 1) & np.isfinite(pred["onset_s"].to_numpy(dtype=float)),
            np.maximum(0.0, alarm - pred["onset_s"].to_numpy(dtype=float)),
            np.nan,
        )
        pred["model"] = "ResCompact_ExtHGB_ET_ROCKET_Ensemble"
        pred_rows.append(pred)
        print(f"seed {seed}: accuracy={metrics['accuracy']:.4f}, fp={metrics['fp']}, fn={metrics['fn']}", flush=True)
    detail = pd.DataFrame(detail_rows)
    choices = pd.DataFrame(choice_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    summary = summarize(detail_rows)
    detail.to_csv(OUT / "rescompact_ext_rocket_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "rescompact_ext_rocket_ensemble_choices.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "rescompact_ext_rocket_ensemble_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "rescompact_ext_rocket_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


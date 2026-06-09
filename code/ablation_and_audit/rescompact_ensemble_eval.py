from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150)
THRESHOLD_GRID = (0.55, 0.65, 0.75, 0.85, 0.95)
WEIGHTS_HGB = (0.0, 0.25, 0.50, 0.75, 1.0)


def prepare() -> pd.DataFrame:
    hgb = pd.read_csv(OUT / "repeated_seed_predictions_rescompact.csv")
    et = pd.read_csv(OUT / "repeated_seed_predictions_rescompact_et.csv")
    hgb = hgb[hgb["model"] == "EarlyCascadeHGB_50_75_100_150"].copy()
    et = et[et["model"] == "EarlyCascadeET_50_75_100_150"].copy()
    meta_cols = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    merged = hgb[meta_cols + prob_cols].merge(
        et[["sample_id", "seed", "split"] + prob_cols],
        on=["sample_id", "seed", "split"],
        suffixes=("_hgb", "_et"),
        how="inner",
    )
    return merged


def metrics(frame: pd.DataFrame, thresholds: dict[int, float], weight_hgb: float) -> dict[str, float]:
    y_true = frame["binary"].to_numpy(dtype=int)
    y_pred = np.zeros(len(frame), dtype=int)
    alarm = np.full(len(frame), np.nan)
    for horizon in HORIZONS:
        prob = (
            weight_hgb * frame[f"prob_{horizon}s_hgb"].to_numpy(dtype=float)
            + (1.0 - weight_hgb) * frame[f"prob_{horizon}s_et"].to_numpy(dtype=float)
        )
        hit = prob >= thresholds[horizon]
        new_hit = hit & (y_pred == 0)
        alarm[new_hit] = horizon
        y_pred[hit] = 1
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    onset = frame["onset_s"].to_numpy(dtype=float)
    delay = np.where((y_true == 1) & (y_pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    hard = (y_true == 0) & (frame["hard_negative"].to_numpy(dtype=int) == 1)
    return {
        "accuracy": float((tp + tn) / len(frame)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "hard_negative_fpr": float(y_pred[hard].mean()) if hard.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.nanmedian(delay)) if np.isfinite(delay).any() else np.nan,
        "p95_delay_s": float(np.nanquantile(delay, 0.95)) if np.isfinite(delay).any() else np.nan,
    }


def score(m: dict[str, float]) -> float:
    delay = m["median_delay_s"] if np.isfinite(m["median_delay_s"]) else 150.0
    return (
        m["accuracy"]
        + 0.16 * m["f1"]
        + 0.14 * m["recall"]
        + 0.16 * m["specificity"]
        - 0.05 * m["fp"]
        - delay / 2600.0
    )


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": "ResCompact_HGB_ET_ValidationEnsemble", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    data = prepare()
    detail_rows = []
    choice_rows = []
    for seed in sorted(data["seed"].unique()):
        val = data[(data["seed"] == seed) & (data["split"] == "val")]
        test = data[(data["seed"] == seed) & (data["split"] == "test")]
        best = None
        for weight in WEIGHTS_HGB:
            for combo in product(THRESHOLD_GRID, repeat=len(HORIZONS)):
                thresholds = dict(zip(HORIZONS, combo))
                val_metrics = metrics(val, thresholds, weight)
                s = score(val_metrics)
                if best is None or s > best[0]:
                    best = (s, weight, thresholds)
        assert best is not None
        test_metrics = metrics(test, best[2], best[1])
        test_metrics.update({"seed": int(seed), "weight_hgb": float(best[1])})
        detail_rows.append(test_metrics)
        choice_rows.append({"seed": int(seed), "weight_hgb": float(best[1]), **{f"threshold_{h}s": best[2][h] for h in HORIZONS}})
    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail_rows)
    detail.to_csv(OUT / "rescompact_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / "rescompact_ensemble_choices.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "rescompact_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150, 250, 400)
SOURCES = [
    ("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("global_et", "repeated_seed_predictions_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
]


def make_weights() -> np.ndarray:
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows: list[tuple[float, ...]] = []
    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c >= -1e-9 and any(abs(c - x) < 1e-9 for x in grid):
                rows.append((a, b, max(0.0, c)))
    return np.array(rows, dtype=float)


def make_threshold_families() -> np.ndarray:
    # Early false alarms dominate the specificity error, so keep 50/75 s stricter.
    early50 = [0.85, 0.90, 0.94, 0.97, 0.99, 0.995]
    early75 = [0.78, 0.85, 0.90, 0.94, 0.97, 0.99]
    mid = [0.65, 0.75, 0.85, 0.92, 0.97]
    late = [0.55, 0.65, 0.75, 0.85, 0.94]
    rows = []
    for t50, t75, t100, t150, tlate in product(early50, early75, mid, mid, late):
        rows.append((t50, t75, t100, t150, tlate, tlate))
    return np.array(rows, dtype=float)


WEIGHTS = make_weights()
THRESHOLDS = make_threshold_families()


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name, low_memory=False)
    data = data[data["model"] == model_name].copy()
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    probs = [f"prob_{h}s" for h in HORIZONS]
    return data[meta + probs].rename(columns={c: f"{prefix}_{c}" for c in probs})


def prepare() -> pd.DataFrame:
    base = None
    for prefix, file_name, model_name in SOURCES:
        part = load_source(prefix, file_name, model_name)
        pcols = [f"{prefix}_prob_{h}s" for h in HORIZONS]
        if base is None:
            base = part
        else:
            base = base.merge(part[["sample_id", "seed", "split", *pcols]], on=["sample_id", "seed", "split"], how="inner")
    assert base is not None
    return base


def arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "y": frame["binary"].to_numpy(dtype=int),
        "hard": frame["hard_negative"].to_numpy(dtype=int),
        "onset": frame["onset_s"].to_numpy(dtype=float),
        "prob": np.stack(
            [frame[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float) for prefix, _, _ in SOURCES],
            axis=0,
        ),
    }


def predict(arr: dict[str, np.ndarray], weight: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = np.einsum("cs,snh->cnh", weight, arr["prob"])
    hits = prob >= threshold[:, None, :]
    pred = hits.any(axis=2)
    first = np.argmax(hits, axis=2)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    return pred, alarm


def metrics(arr: dict[str, np.ndarray], pred: np.ndarray, alarm: np.ndarray, with_delay: bool = False) -> dict[str, np.ndarray]:
    y = arr["y"].astype(bool)
    tp = (pred & y.reshape(1, -1)).sum(axis=1).astype(float)
    tn = ((~pred) & (~y).reshape(1, -1)).sum(axis=1).astype(float)
    fp = (pred & (~y).reshape(1, -1)).sum(axis=1).astype(float)
    fn = ((~pred) & y.reshape(1, -1)).sum(axis=1).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    hard = (~y) & (arr["hard"].astype(int) == 1)
    hard_fpr = pred[:, hard].mean(axis=1) if hard.any() else np.zeros(len(pred), dtype=float)
    if with_delay:
        delay = np.where(
            pred & y.reshape(1, -1) & np.isfinite(arr["onset"]).reshape(1, -1),
            np.maximum(0.0, alarm - arr["onset"].reshape(1, -1)),
            np.nan,
        )
        with np.errstate(all="ignore"):
            median_delay = np.nanmedian(delay, axis=1)
            p95_delay = np.nanquantile(delay, 0.95, axis=1)
    else:
        median_delay = np.zeros(len(pred), dtype=float)
        p95_delay = np.zeros(len(pred), dtype=float)
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "hard_negative_fpr": hard_fpr,
        "fp": fp,
        "fn": fn,
        "median_delay_s": median_delay,
        "p95_delay_s": p95_delay,
    }


def score(m: dict[str, np.ndarray], spec_floor: float, recall_floor: float) -> np.ndarray:
    spec_gap = np.maximum(0.0, spec_floor - m["specificity"])
    recall_gap = np.maximum(0.0, recall_floor - m["recall"])
    return (
        0.45 * m["specificity"]
        + 0.22 * m["accuracy"]
        + 0.18 * m["f1"]
        + 0.15 * m["recall"]
        - 0.12 * m["hard_negative_fpr"]
        - 0.04 * m["fp"]
        - 0.08 * m["fn"]
        - 2.50 * spec_gap
        - 0.80 * recall_gap
    )


def choose(arr: dict[str, np.ndarray], spec_floor: float, recall_floor: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    best = (-np.inf, WEIGHTS[0], THRESHOLDS[0], {})
    for w in WEIGHTS:
        weights = np.repeat(w.reshape(1, -1), len(THRESHOLDS), axis=0)
        pred, alarm = predict(arr, weights, THRESHOLDS)
        m = metrics(arr, pred, alarm, with_delay=False)
        scores = score(m, spec_floor, recall_floor)
        idx = int(np.nanargmax(scores))
        if float(scores[idx]) > best[0]:
            best = (float(scores[idx]), w.copy(), THRESHOLDS[idx].copy(), {k: float(v[idx]) for k, v in m.items()})
    return best[1], best[2], best[3]


def apply(arr: dict[str, np.ndarray], weight: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred, alarm = predict(arr, weight.reshape(1, -1), threshold.reshape(1, -1))
    m = {k: float(v[0]) for k, v in metrics(arr, pred, alarm, with_delay=True).items()}
    m["fp"] = int(m["fp"])
    m["fn"] = int(m["fn"])
    return pred[0].astype(int), alarm[0], m


def summarize(rows: list[dict], model: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out: dict[str, float | str | int] = {"model": model, "n_seeds": int(df["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        mean = float(df[col].mean())
        std = float(df[col].std(ddof=1))
        out[f"{col}_mean"] = mean
        out[f"{col}_std"] = std
        out[f"{col}_sem"] = float(std / np.sqrt(len(df))) if len(df) else np.nan
        out[f"{col}_ci95_halfwidth"] = float(1.96 * std / np.sqrt(len(df))) if len(df) else np.nan
        out[f"{col}_min"] = float(df[col].min())
        out[f"{col}_max"] = float(df[col].max())
    return pd.DataFrame([out])


def run_setting(data: pd.DataFrame, spec_floor: float, recall_floor: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    choices = []
    preds = []
    for seed in sorted(data["seed"].unique()):
        val = data[(data["seed"] == seed) & (data["split"] == "val")].copy()
        test = data[(data["seed"] == seed) & (data["split"] == "test")].copy()
        w, th, val_m = choose(arrays(val), spec_floor, recall_floor)
        y_pred, alarm, test_m = apply(arrays(test), w, th)
        test_m.update({"seed": int(seed), "val_specificity": val_m["specificity"], "val_recall": val_m["recall"]})
        rows.append(test_m)
        choices.append(
            {
                "seed": int(seed),
                "spec_floor": spec_floor,
                "recall_floor": recall_floor,
                **{f"w_{SOURCES[i][0]}": float(w[i]) for i in range(len(SOURCES))},
                **{f"threshold_{h}s": float(th[i]) for i, h in enumerate(HORIZONS)},
                **{f"val_{k}": v for k, v in val_m.items()},
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
        pred["model"] = f"HighSpecificityFamily_spec{spec_floor:.2f}_recall{recall_floor:.2f}"
        preds.append(pred)
        print(
            f"spec_floor={spec_floor:.2f} recall_floor={recall_floor:.2f} seed {seed}: "
            f"acc={test_m['accuracy']:.4f} spec={test_m['specificity']:.3f} rec={test_m['recall']:.3f} fp={test_m['fp']} fn={test_m['fn']}",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(choices), pd.concat(preds, ignore_index=True)


def main() -> None:
    data = prepare()
    settings = [(0.90, 0.90), (0.92, 0.88), (0.94, 0.84), (0.95, 0.80)]
    summaries = []
    all_metrics = []
    all_choices = []
    all_preds = []
    for spec_floor, recall_floor in settings:
        metrics_df, choices_df, preds_df = run_setting(data, spec_floor, recall_floor)
        model = f"HighSpecificityFamily_spec{spec_floor:.2f}_recall{recall_floor:.2f}"
        metrics_df["model"] = model
        choices_df["model"] = model
        summaries.append(summarize(metrics_df, model))
        all_metrics.append(metrics_df)
        all_choices.append(choices_df)
        all_preds.append(preds_df)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT / "rescompact_high_specificity_family_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_metrics, ignore_index=True).to_csv(OUT / "rescompact_high_specificity_family_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_choices, ignore_index=True).to_csv(OUT / "rescompact_high_specificity_family_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_preds, ignore_index=True).to_csv(OUT / "rescompact_high_specificity_family_predictions.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

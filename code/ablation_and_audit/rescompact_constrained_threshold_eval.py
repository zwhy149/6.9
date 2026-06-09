from __future__ import annotations

from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150, 250, 400)
GRID = np.array(
    [float(x) for x in os.environ.get("CONSTRAINED_GRID", "0.55,0.65,0.75,0.85,0.90,0.94,0.97,0.99").split(",") if x.strip()],
    dtype=float,
)
CHUNK_SIZE = int(os.environ.get("CONSTRAINED_CHUNK", "25000"))
OUTPUT_SUFFIX = os.environ.get("CONSTRAINED_SUFFIX", "constrained")

SOURCES = [
    ("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("global_et", "repeated_seed_predictions_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
]


def make_weights() -> np.ndarray:
    grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    out: list[tuple[float, ...]] = []

    def rec(prefix: list[float], slots: int, remaining: float) -> None:
        if slots == 1:
            if any(abs(remaining - x) < 1e-9 for x in grid):
                out.append(tuple(prefix + [remaining]))
            return
        for value in grid:
            if value <= remaining + 1e-9:
                rec(prefix + [value], slots - 1, remaining - value)

    rec([], len(SOURCES), 1.0)
    return np.array(out, dtype=float)


WEIGHTS = make_weights()
THRESHOLDS = np.array(list(product(GRID, repeat=len(HORIZONS))), dtype=float)


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name, low_memory=False)
    data = data[data["model"] == model_name].copy()
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    return data[meta + prob_cols].rename(columns={c: f"{prefix}_{c}" for c in prob_cols})


def prepare() -> pd.DataFrame:
    base = None
    for prefix, file_name, model_name in SOURCES:
        part = load_source(prefix, file_name, model_name)
        cols = [f"{prefix}_prob_{h}s" for h in HORIZONS]
        if base is None:
            base = part
        else:
            base = base.merge(part[["sample_id", "seed", "split", *cols]], on=["sample_id", "seed", "split"], how="inner")
    assert base is not None
    return base


def arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "y": frame["binary"].to_numpy(dtype=int),
        "hard": frame["hard_negative"].to_numpy(dtype=int),
        "onset": frame["onset_s"].to_numpy(dtype=float),
        "prob": np.stack(
            [
                frame[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
                for prefix, _, _ in SOURCES
            ],
            axis=0,
        ),
    }


def predict(arr: dict[str, np.ndarray], weight: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if thresholds.ndim == 1:
        thresholds = thresholds.reshape(1, -1)
    if weight.ndim == 1:
        weight = np.repeat(weight.reshape(1, -1), len(thresholds), axis=0)
    prob = np.einsum("cs,snh->cnh", weight, arr["prob"])
    hits = prob >= thresholds[:, None, :]
    pred = hits.any(axis=2)
    first = np.argmax(hits, axis=2)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    return pred, alarm


def metric_vectors(arr: dict[str, np.ndarray], pred: np.ndarray, alarm: np.ndarray, delay: bool = False) -> dict[str, np.ndarray]:
    y = arr["y"].astype(bool)
    n = len(y)
    tp = (pred & y.reshape(1, -1)).sum(axis=1).astype(float)
    tn = ((~pred) & (~y).reshape(1, -1)).sum(axis=1).astype(float)
    fp = (pred & (~y).reshape(1, -1)).sum(axis=1).astype(float)
    fn = ((~pred) & y.reshape(1, -1)).sum(axis=1).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    hard = (~y) & (arr["hard"].astype(int) == 1)
    hard_fpr = pred[:, hard].mean(axis=1) if hard.any() else np.zeros(len(pred))
    if delay:
        d = np.where(
            pred & y.reshape(1, -1) & np.isfinite(arr["onset"]).reshape(1, -1),
            np.maximum(0.0, alarm - arr["onset"].reshape(1, -1)),
            np.nan,
        )
        with np.errstate(all="ignore"):
            median_delay = np.nanmedian(d, axis=1)
            p95_delay = np.nanquantile(d, 0.95, axis=1)
    else:
        median_delay = np.zeros(len(pred))
        p95_delay = np.zeros(len(pred))
    return {
        "accuracy": (tp + tn) / n,
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


def base_score(m: dict[str, np.ndarray]) -> np.ndarray:
    return (
        m["accuracy"]
        + 0.16 * m["f1"]
        + 0.12 * m["recall"]
        + 0.22 * m["specificity"]
        - 0.070 * m["fp"]
        - 0.050 * m["fn"]
        - 0.10 * m["hard_negative_fpr"]
    )


def choose(arr: dict[str, np.ndarray], spec_floor: float, recall_floor: float, fp_cap: int | None, oracle_mode: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    best_score = -np.inf
    best_weight = WEIGHTS[0].copy()
    best_threshold = THRESHOLDS[0].copy()
    best_metrics: dict[str, float] = {}
    for weight in WEIGHTS:
        for start in range(0, len(THRESHOLDS), CHUNK_SIZE):
            thresholds = THRESHOLDS[start : start + CHUNK_SIZE]
            pred, alarm = predict(arr, weight, thresholds)
            m = metric_vectors(arr, pred, alarm, delay=False)
            scores = base_score(m)
            feasible = (m["specificity"] >= spec_floor) & (m["recall"] >= recall_floor)
            if fp_cap is not None:
                feasible &= m["fp"] <= fp_cap
            if not feasible.any():
                # Keep a fallback ranked by constrained violation. This avoids empty selections in hard validation splits.
                violation = (
                    2.0 * np.maximum(0.0, spec_floor - m["specificity"])
                    + 1.5 * np.maximum(0.0, recall_floor - m["recall"])
                    + (0.25 * np.maximum(0.0, m["fp"] - fp_cap) if fp_cap is not None else 0.0)
                )
                scores = scores - 10.0 * violation
            else:
                scores = np.where(feasible, scores, -np.inf)
            idx = int(np.nanargmax(scores))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_weight = weight.copy()
                best_threshold = thresholds[idx].copy()
                best_metrics = {k: float(v[idx]) for k, v in m.items()}
                best_metrics["feasible"] = float(bool(feasible[idx]))
    return best_weight, best_threshold, best_metrics


def apply(arr: dict[str, np.ndarray], weight: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred, alarm = predict(arr, weight, threshold)
    m = {k: float(v[0]) for k, v in metric_vectors(arr, pred, alarm, delay=True).items()}
    m["fp"] = int(m["fp"])
    m["fn"] = int(m["fn"])
    return pred[0].astype(int), alarm[0], m


def summarize(rows: list[dict], label: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out: dict[str, float | str | int] = {"model": label, "n_seeds": int(df["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s", "val_feasible"]:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1))
        out[f"{col}_min"] = float(df[col].min())
        out[f"{col}_max"] = float(df[col].max())
    return pd.DataFrame([out])


def run_setting(data: pd.DataFrame, spec_floor: float, recall_floor: float, fp_cap: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    oracle_rows = []
    choices = []
    for seed in sorted(data["seed"].unique()):
        val = data[(data["seed"] == seed) & (data["split"] == "val")].copy()
        test = data[(data["seed"] == seed) & (data["split"] == "test")].copy()
        w, th, val_m = choose(arrays(val), spec_floor, recall_floor, fp_cap)
        _, _, test_m = apply(arrays(test), w, th)
        test_m.update({"seed": int(seed), "val_feasible": int(val_m.get("feasible", 0))})
        rows.append(test_m)
        ow, oth, _ = choose(arrays(test), spec_floor, recall_floor, fp_cap, oracle_mode=True)
        _, _, oracle_m = apply(arrays(test), ow, oth)
        oracle_m.update({"seed": int(seed), "val_feasible": 1})
        oracle_rows.append(oracle_m)
        choices.append(
            {
                "seed": int(seed),
                "spec_floor": spec_floor,
                "recall_floor": recall_floor,
                "fp_cap": "" if fp_cap is None else fp_cap,
                "val_feasible": int(val_m.get("feasible", 0)),
                **{f"w_{SOURCES[i][0]}": float(w[i]) for i in range(len(SOURCES))},
                **{f"threshold_{h}s": float(th[i]) for i, h in enumerate(HORIZONS)},
            }
        )
        print(
            f"setting spec>={spec_floor:.2f} recall>={recall_floor:.2f} fp_cap={fp_cap} seed {seed}: "
            f"acc={test_m['accuracy']:.4f} spec={test_m['specificity']:.3f} fp={test_m['fp']} fn={test_m['fn']} feasible={test_m['val_feasible']}",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows), pd.DataFrame(choices)


def main() -> None:
    data = prepare()
    settings = [
        (0.88, 0.93, None),
        (0.90, 0.92, None),
        (0.90, 0.94, None),
        (0.92, 0.90, None),
        (0.90, 0.90, 1),
        (0.92, 0.88, 1),
        (0.95, 0.85, 0),
    ]
    summaries = []
    metrics_all = []
    oracle_all = []
    choices_all = []
    for spec_floor, recall_floor, fp_cap in settings:
        metrics, oracle, choices = run_setting(data, spec_floor, recall_floor, fp_cap)
        label = f"Constrained3src_spec{spec_floor:.2f}_recall{recall_floor:.2f}_fpcap{fp_cap}"
        metrics["setting"] = label
        oracle["setting"] = label + "_TEST_ORACLE"
        choices["setting"] = label
        summaries.append(summarize(metrics, label))
        summaries.append(summarize(oracle, label + "_TEST_ORACLE"))
        metrics_all.append(metrics)
        oracle_all.append(oracle)
        choices_all.append(choices)
    summary = pd.concat(summaries, ignore_index=True)
    metrics = pd.concat(metrics_all, ignore_index=True)
    oracle = pd.concat(oracle_all, ignore_index=True)
    choices = pd.concat(choices_all, ignore_index=True)
    summary.to_csv(OUT / f"rescompact_constrained_threshold_summary_{OUTPUT_SUFFIX}.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / f"rescompact_constrained_threshold_metrics_{OUTPUT_SUFFIX}.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv(OUT / f"rescompact_constrained_threshold_oracle_metrics_{OUTPUT_SUFFIX}.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"rescompact_constrained_threshold_choices_{OUTPUT_SUFFIX}.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from esc_transfer_pipeline import apply_onset_overrides, load_source_target_samples
from repeated_seed_eval import duplicate_group_name, select_eval_seeds, stratified_target_split


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150, 250, 400)
SERIES_LEN = int(os.environ.get("ROCKET_SERIES_LEN", "160"))
N_KERNELS = int(os.environ.get("ROCKET_KERNELS", "1200"))
SEED_COUNT = int(os.environ.get("N_SEEDS", "30"))
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "prefix_rocket")
SOURCE_WEIGHT = float(os.environ.get("SOURCE_WEIGHT", "0.5"))
HARD_NEG_WEIGHT = float(os.environ.get("HARD_NEG_WEIGHT", "1.5"))


def resample_prefix(time: np.ndarray, voltage: np.ndarray, horizon: int) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    if len(t) == 0:
        return np.zeros(SERIES_LEN, dtype=np.float32)
    t0 = float(t[0])
    grid = np.linspace(t0, t0 + horizon, SERIES_LEN)
    right = np.searchsorted(t, t0 + horizon, side="right")
    right = max(1, right)
    tp = t[:right]
    vp = v[:right]
    if len(tp) == 1:
        interp = np.full(SERIES_LEN, float(vp[0]))
    else:
        interp = np.interp(grid, tp, vp, left=float(vp[0]), right=float(vp[-1]))
    scale = max(abs(float(interp[0])), 1e-6)
    rel = (interp - float(interp[0])) / scale
    rel = rel - np.nanmedian(rel[: max(5, SERIES_LEN // 10)])
    mad = np.nanmedian(np.abs(rel - np.nanmedian(rel))) + 1e-6
    rel = np.clip(rel / (8.0 * mad), -3.0, 3.0)
    return rel.astype(np.float32)


def build_or_load_series() -> tuple[pd.DataFrame, np.ndarray]:
    meta_path = WORK / "rocket_series_meta.csv"
    series_path = WORK / f"rocket_series_L{SERIES_LEN}.npz"
    if meta_path.exists() and series_path.exists():
        meta = pd.read_csv(meta_path)
        series = np.load(series_path)["series"]
        return meta, series
    samples = load_source_target_samples()
    apply_onset_overrides(samples)
    rows = []
    series_rows = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "file_name": sample.file_name,
                "binary": int(sample.binary),
                "hard_negative": int(sample.hard_negative),
                "severity_name": sample.severity_name,
                "onset_s": sample.onset_s if sample.onset_s is not None else np.nan,
            }
        )
        series_rows.append([resample_prefix(sample.time, sample.voltage, h) for h in HORIZONS])
    meta = pd.DataFrame(rows)
    series = np.asarray(series_rows, dtype=np.float32)
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    np.savez_compressed(series_path, series=series)
    return meta, series


def make_kernels(seed: int) -> list[dict[str, np.ndarray | int | float]]:
    rng = np.random.default_rng(seed)
    kernels = []
    lengths = np.array([7, 9, 11, 15], dtype=int)
    for _ in range(N_KERNELS):
        length = int(rng.choice(lengths))
        dilation_max = max(1, (SERIES_LEN - 1) // (length - 1))
        dilation = int(2 ** rng.uniform(0, np.log2(dilation_max + 1)))
        dilation = max(1, min(dilation, dilation_max))
        weights = rng.normal(0.0, 1.0, size=length)
        weights = weights - weights.mean()
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights = weights / norm
        bias = float(rng.uniform(-1.0, 1.0))
        kernels.append({"weights": weights.astype(np.float32), "dilation": dilation, "bias": bias})
    return kernels


def build_or_load_rocket_features(series: np.ndarray) -> np.ndarray:
    feature_path = WORK / f"rocket_features_L{SERIES_LEN}_K{N_KERNELS}.npz"
    if feature_path.exists():
        return np.load(feature_path)["features"]
    parts = []
    for horizon_idx, horizon in enumerate(HORIZONS):
        kernels = make_kernels(9173 + horizon_idx * 101)
        feats = rocket_transform(series[:, horizon_idx, :], kernels)
        parts.append(feats)
        print(f"built fixed ROCKET features for horizon {horizon}s", flush=True)
    features = np.stack(parts, axis=1).astype(np.float32)
    np.savez_compressed(feature_path, features=features)
    return features


def rocket_transform(x: np.ndarray, kernels: list[dict[str, np.ndarray | int | float]]) -> np.ndarray:
    n, length = x.shape
    diff = np.diff(x, axis=1, prepend=x[:, :1])
    feats = np.empty((n, len(kernels) * 4), dtype=np.float32)
    for k_idx, kernel in enumerate(kernels):
        weights = kernel["weights"]
        dilation = int(kernel["dilation"])
        bias = float(kernel["bias"])
        span = (len(weights) - 1) * dilation + 1
        if span > length:
            positions = np.array([0], dtype=int)
        else:
            positions = np.arange(0, length - span + 1, dtype=int)
        conv = np.zeros((n, len(positions)), dtype=np.float32)
        conv_d = np.zeros((n, len(positions)), dtype=np.float32)
        for w_idx, weight in enumerate(weights):
            idx = positions + w_idx * dilation
            conv += x[:, idx] * float(weight)
            conv_d += diff[:, idx] * float(weight)
        conv += bias
        conv_d += bias
        base = k_idx * 4
        feats[:, base] = conv.max(axis=1)
        feats[:, base + 1] = (conv > 0).mean(axis=1)
        feats[:, base + 2] = conv_d.max(axis=1)
        feats[:, base + 3] = (conv_d > 0).mean(axis=1)
    return feats


def train_horizon(features: np.ndarray, meta: pd.DataFrame, train_ids: set[str], horizon_idx: int, seed: int):
    train_mask = meta["sample_id"].isin(train_ids).to_numpy()
    x_train = features[train_mask, horizon_idx, :]
    y_train = meta.loc[train_mask, "binary"].to_numpy(dtype=int)
    sample_weight = np.where(meta.loc[train_mask, "domain"].astype(str).to_numpy() == "source5", SOURCE_WEIGHT, 1.0)
    hard = (y_train == 0) & (meta.loc[train_mask, "hard_negative"].to_numpy(dtype=int) == 1)
    sample_weight[hard] *= HARD_NEG_WEIGHT
    model = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
    )
    model.fit(x_train, y_train, logisticregression__sample_weight=sample_weight)
    return {"model": model, "horizon_idx": horizon_idx}


def predict_horizon(bundle: dict, features: np.ndarray, ids: list[str], meta: pd.DataFrame) -> pd.DataFrame:
    id_to_pos = {sample_id: idx for idx, sample_id in enumerate(meta["sample_id"])}
    positions = np.array([id_to_pos[x] for x in ids], dtype=int)
    x = features[positions, bundle["horizon_idx"], :]
    prob = bundle["model"].predict_proba(x)[:, 1]
    horizon = HORIZONS[bundle["horizon_idx"]]
    out = meta.iloc[positions][["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]].copy()
    out[f"prob_{horizon}s"] = prob
    return out


def combine_predictions(base: pd.DataFrame, thresholds: dict[int, float]) -> tuple[pd.DataFrame, dict[str, float]]:
    out = base.copy()
    y_true = out["binary"].to_numpy(dtype=int)
    y_pred = np.zeros(len(out), dtype=int)
    alarm = np.full(len(out), np.nan)
    for horizon in HORIZONS:
        hit = out[f"prob_{horizon}s"].to_numpy(dtype=float) >= thresholds[horizon]
        new_hit = hit & (y_pred == 0)
        alarm[new_hit] = horizon
        y_pred[hit] = 1
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["alarm_time_s"] = alarm
    onset = out["onset_s"].to_numpy(dtype=float)
    out["delay_s"] = np.where((y_true == 1) & (y_pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard = (y_true == 0) & (out["hard_negative"].to_numpy(dtype=int) == 1)
    delay = out["delay_s"].to_numpy(dtype=float)
    metrics = {
        "accuracy": float((tp + tn) / len(out)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "hard_negative_fpr": float(y_pred[hard].mean()) if hard.any() else 0.0,
        "fp": fp,
        "fn": fn,
        "median_delay_s": float(np.nanmedian(delay)) if np.isfinite(delay).any() else np.nan,
        "p95_delay_s": float(np.nanquantile(delay, 0.95)) if np.isfinite(delay).any() else np.nan,
    }
    return out, metrics


def threshold_score(metrics: dict[str, float]) -> float:
    return (
        metrics["accuracy"]
        + 0.18 * metrics["f1"]
        + 0.14 * metrics["recall"]
        + 0.16 * metrics["specificity"]
        - 0.055 * metrics["fp"]
        - 0.045 * metrics["fn"]
        - 0.08 * metrics["hard_negative_fpr"]
    )


def select_thresholds(val_pred: pd.DataFrame) -> dict[int, float]:
    candidates = {}
    for horizon in HORIZONS:
        prob = val_pred[f"prob_{horizon}s"].to_numpy(dtype=float)
        candidates[horizon] = np.unique(np.concatenate([np.quantile(prob, [0.35, 0.55, 0.75, 0.90]), [0.45, 0.60, 0.75, 0.90]]))
    threshold_matrix = np.array(list(product(*(candidates[h] for h in HORIZONS))), dtype=float)
    prob_matrix = val_pred[[f"prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    y_true = val_pred["binary"].to_numpy(dtype=int).astype(bool)
    y_false = ~y_true
    hits = prob_matrix[None, :, :] >= threshold_matrix[:, None, :]
    pred = hits.any(axis=2)
    tp = (pred & y_true.reshape(1, -1)).sum(axis=1).astype(float)
    tn = ((~pred) & y_false.reshape(1, -1)).sum(axis=1).astype(float)
    fp = (pred & y_false.reshape(1, -1)).sum(axis=1).astype(float)
    fn = ((~pred) & y_true.reshape(1, -1)).sum(axis=1).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    hard = y_false & (val_pred["hard_negative"].to_numpy(dtype=int) == 1)
    hard_fpr = pred[:, hard].mean(axis=1) if hard.any() else np.zeros(len(pred), dtype=float)
    accuracy = (tp + tn) / len(y_true)
    scores = (
        accuracy
        + 0.18 * f1
        + 0.14 * recall
        + 0.16 * specificity
        - 0.055 * fp
        - 0.045 * fn
        - 0.08 * hard_fpr
    )
    best_idx = int(np.nanargmax(scores))
    return {h: float(threshold_matrix[best_idx, idx]) for idx, h in enumerate(HORIZONS)}


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": "PrefixROCKET_Logistic", "n_seeds": int(data["seed"].nunique())}
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
    meta, series = build_or_load_series()
    features = build_or_load_rocket_features(series)
    target_meta = meta[meta["domain"] == "target100"].copy()
    target_meta["duplicate_group"] = [
        duplicate_group_name(file_name, binary)
        for file_name, binary in zip(target_meta["file_name"], target_meta["binary"])
    ]
    eval_seeds, split_diagnostics = select_eval_seeds(target_meta, SEED_COUNT)
    split_diagnostics.to_csv(OUT / f"{OUTPUT_SUFFIX}_split_diagnostics.csv", index=False, encoding="utf-8-sig")
    source_ids = set(meta.loc[meta["domain"] == "source5", "sample_id"])
    rows = []
    pred_rows = []
    choice_rows = []
    for index, seed in enumerate(eval_seeds, start=1):
        split = stratified_target_split(target_meta, seed)
        train_ids = set(split["train"]) | source_ids
        bundles = [train_horizon(features, meta, train_ids, idx, seed) for idx in range(len(HORIZONS))]
        merged = {}
        for split_name in ["val", "test"]:
            parts = [predict_horizon(bundle, features, split[split_name], meta) for bundle in bundles]
            base = parts[0]
            for part in parts[1:]:
                prob_col = [c for c in part.columns if c.startswith("prob_")][0]
                base = base.merge(part[["sample_id", prob_col]], on="sample_id", how="left")
            merged[split_name] = base
        thresholds = select_thresholds(merged["val"])
        test_pred, metrics = combine_predictions(merged["test"], thresholds)
        metrics.update({"seed": int(seed)})
        rows.append(metrics)
        for split_name in ["val", "test"]:
            split_pred, _ = combine_predictions(merged[split_name], thresholds)
            split_pred["seed"] = int(seed)
            split_pred["split"] = split_name
            split_pred["model"] = "PrefixROCKET_Logistic"
            pred_rows.append(split_pred)
        choice_rows.append({"seed": int(seed), **{f"threshold_{h}s": thresholds[h] for h in HORIZONS}})
        print(f"completed {index}/{len(eval_seeds)} seed {seed}: accuracy={metrics['accuracy']:.4f}, fp={metrics['fp']}, fn={metrics['fn']}", flush=True)
    metrics_df = pd.DataFrame(rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    summary = summarize(rows)
    metrics_df.to_csv(OUT / f"{OUTPUT_SUFFIX}_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / f"{OUTPUT_SUFFIX}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / f"{OUTPUT_SUFFIX}_choices.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{OUTPUT_SUFFIX}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix

from esc_transfer_pipeline import load_source_target_samples
from repeated_seed_eval import duplicate_group_name, stratified_target_split


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
WORK = ROOT / "work"

HORIZONS = tuple(int(float(x)) for x in os.environ.get("WAVELET_HORIZONS", "50,75,100,150,250,400").split(",") if x.strip())
PREFIX_TAG = os.environ.get("WAVELET_PREFIX_TAG", "rescompact_global_v2")
OUTPUT_SUFFIX = os.environ.get("WAVELET_OUTPUT_SUFFIX", "haar_v1")
MODEL_KIND = os.environ.get("WAVELET_MODEL", "et").lower()
TREE_N = int(os.environ.get("WAVELET_TREE_N", "70"))
SOURCE_WEIGHT = float(os.environ.get("WAVELET_SOURCE_WEIGHT", "0.55"))
TARGET_NORMAL_WEIGHT = float(os.environ.get("WAVELET_TARGET_NORMAL_WEIGHT", "1.55"))
HARD_NEG_WEIGHT = float(os.environ.get("WAVELET_HARD_NEG_WEIGHT", "3.0"))
SEED_LIMIT = int(os.environ.get("WAVELET_SEED_LIMIT", "0"))
FEATURE_MODE = os.environ.get("WAVELET_FEATURE_MODE", "combined").lower()

META_COLS = {
    "sample_id",
    "domain",
    "file_name",
    "binary",
    "hard_negative",
    "severity_name",
    "onset_s",
    "duplicate_group",
}


def output_path(stem: str) -> Path:
    return OUT / f"{stem}_{OUTPUT_SUFFIX}.csv"


def haar_detail_energies(x: np.ndarray, levels: int = 7) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(x, dtype=float).copy()
    energies: list[float] = []
    maxima: list[float] = []
    for _ in range(levels):
        if len(current) < 2:
            energies.append(0.0)
            maxima.append(0.0)
            continue
        if len(current) % 2:
            current = current[:-1]
        approx = (current[0::2] + current[1::2]) / np.sqrt(2.0)
        detail = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        energies.append(float(np.mean(detail**2)) if len(detail) else 0.0)
        maxima.append(float(np.max(np.abs(detail))) if len(detail) else 0.0)
        current = approx
    return np.asarray(energies, dtype=float), np.asarray(maxima, dtype=float)


def safe_slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3:
        return 0.0
    x = t - t[0]
    den = float(np.dot(x - x.mean(), x - x.mean()))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x - x.mean(), y - y.mean()) / den)


def wavelet_features_for_prefix(time: np.ndarray, voltage: np.ndarray, horizon_s: float) -> dict[str, float]:
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    if len(t) < 8:
        return {}
    end_t = t[0] + float(horizon_s)
    idx = int(np.searchsorted(t, end_t, side="right"))
    idx = min(max(idx, 8), len(t))
    tt = t[:idx]
    vv = v[:idx]
    duration = max(float(tt[-1] - tt[0]), 1e-6)
    scale = max(abs(float(vv[0])), 1e-6)

    grid = np.linspace(tt[0], tt[-1], 256)
    yy = np.interp(grid, tt, vv)
    y_norm = (yy - yy[0]) / scale
    trend = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
    residual = y_norm - trend
    dy = np.diff(y_norm, prepend=y_norm[0])
    detrended_dy = np.diff(residual, prepend=residual[0])

    e_raw, m_raw = haar_detail_energies(y_norm)
    e_res, m_res = haar_detail_energies(residual)
    e_dy, m_dy = haar_detail_energies(dy)
    total_raw = float(np.sum(e_raw)) + 1e-12
    total_res = float(np.sum(e_res)) + 1e-12
    total_dy = float(np.sum(e_dy)) + 1e-12

    p_raw = e_raw / total_raw
    p_res = e_res / total_res
    p_dy = e_dy / total_dy
    entropy_raw = float(-(p_raw * np.log(p_raw + 1e-12)).sum())
    entropy_res = float(-(p_res * np.log(p_res + 1e-12)).sum())
    entropy_dy = float(-(p_dy * np.log(p_dy + 1e-12)).sum())

    n = len(yy)
    h = n // 2
    q = n // 4
    slope_head = safe_slope(grid[:h], yy[:h]) / scale if h >= 3 else 0.0
    slope_tail = safe_slope(grid[h:], yy[h:]) / scale if n - h >= 3 else 0.0
    slope_q4 = safe_slope(grid[3 * q :], yy[3 * q :]) / scale if n - 3 * q >= 3 else 0.0
    drop_first = float((yy[0] - yy[h]) / scale)
    drop_second = float((yy[h] - yy[-1]) / scale)
    drop_total = float((yy[0] - yy[-1]) / scale)
    cusum = np.cumsum(detrended_dy - np.median(detrended_dy))

    out: dict[str, float] = {
        "wv_duration_s": duration,
        "wv_drop_total_norm": drop_total,
        "wv_drop_first_half_norm": drop_first,
        "wv_drop_second_half_norm": drop_second,
        "wv_drop_balance": float((abs(drop_second) + 1e-9) / (abs(drop_first) + 1e-9)),
        "wv_tail_minus_head_slope_norm": float(slope_tail - slope_head),
        "wv_q4_minus_head_slope_norm": float(slope_q4 - slope_head),
        "wv_residual_std": float(np.std(residual)),
        "wv_residual_absmax": float(np.max(np.abs(residual))),
        "wv_residual_cusum_absmax": float(np.max(np.abs(cusum))) if len(cusum) else 0.0,
        "wv_raw_entropy": entropy_raw,
        "wv_res_entropy": entropy_res,
        "wv_dy_entropy": entropy_dy,
        "wv_raw_high_low_ratio": float(np.sum(e_raw[:3]) / (np.sum(e_raw[3:]) + 1e-12)),
        "wv_res_high_low_ratio": float(np.sum(e_res[:3]) / (np.sum(e_res[3:]) + 1e-12)),
        "wv_dy_high_low_ratio": float(np.sum(e_dy[:3]) / (np.sum(e_dy[3:]) + 1e-12)),
    }
    for i, (er, mr, eres, mres, ed, md) in enumerate(zip(e_raw, m_raw, e_res, m_res, e_dy, m_dy), start=1):
        out[f"wv_l{i}_raw_energy_share"] = float(er / total_raw)
        out[f"wv_l{i}_raw_max"] = float(mr)
        out[f"wv_l{i}_res_energy_share"] = float(eres / total_res)
        out[f"wv_l{i}_res_max"] = float(mres)
        out[f"wv_l{i}_dy_energy_share"] = float(ed / total_dy)
        out[f"wv_l{i}_dy_max"] = float(md)
    return out


def load_or_build_wavelet_tables() -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    needed = [WORK / f"prefix_wavelet_features_v1_{h}s.csv" for h in HORIZONS]
    if all(path.exists() for path in needed):
        for h, path in zip(HORIZONS, needed):
            tables[h] = pd.read_csv(path, low_memory=False)
        return tables

    samples = load_source_target_samples()
    for horizon in HORIZONS:
        rows = []
        for s in samples:
            feats = wavelet_features_for_prefix(s.time, s.voltage, float(horizon))
            if not feats:
                continue
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "domain": s.domain,
                    "file_name": s.file_name,
                    "binary": s.binary,
                    "hard_negative": s.hard_negative,
                    "severity_name": s.severity_name,
                    "onset_s": s.onset_s if s.onset_s is not None else np.nan,
                    **feats,
                }
            )
        table = pd.DataFrame(rows)
        table.to_csv(WORK / f"prefix_wavelet_features_v1_{horizon}s.csv", index=False, encoding="utf-8-sig")
        tables[horizon] = table
    return tables


def load_prefix_tables() -> tuple[dict[int, pd.DataFrame], list[str]]:
    wavelet = load_or_build_wavelet_tables()
    tables: dict[int, pd.DataFrame] = {}
    all_cols: set[str] | None = None
    for horizon in HORIZONS:
        base = pd.read_csv(WORK / f"prefix_features_{PREFIX_TAG}_{horizon}s.csv", low_memory=False)
        wv = wavelet[horizon].drop(columns=["domain", "file_name", "binary", "hard_negative", "severity_name", "onset_s"], errors="ignore")
        merged = base.merge(wv, on="sample_id", how="left")
        tables[horizon] = merged
        numeric = {c for c in merged.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(merged[c])}
        all_cols = numeric if all_cols is None else all_cols & numeric
    assert all_cols is not None
    if FEATURE_MODE == "wavelet_only":
        return tables, sorted(c for c in all_cols if c.startswith("wv_"))
    useful = (
        "pg_",
        "rc_",
        "drop_norm",
        "max_drop_norm",
        "range_norm",
        "slope_norm",
        "dvdt",
        "res_",
        "signed_move_norm",
        "monotonicity",
        "risk_score",
        "wv_",
    )
    return tables, sorted(c for c in all_cols if any(token in c for token in useful))


def target_meta(table: pd.DataFrame) -> pd.DataFrame:
    meta = (
        table[table["domain"].astype(str) != "source5"]
        [["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]]
        .drop_duplicates("sample_id")
        .copy()
    )
    meta["duplicate_group"] = [duplicate_group_name(f, int(b)) for f, b in zip(meta["file_name"], meta["binary"])]
    return meta


def fit_horizon(table: pd.DataFrame, feature_cols: list[str], train_ids: set[str], seed: int):
    train = table[table["sample_id"].isin(train_ids)].copy()
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(med).to_numpy(dtype=np.float32)
    y = train["binary"].astype(int).to_numpy()
    domain = train["domain"].astype(str).to_numpy()
    hard = train["hard_negative"].astype(int).to_numpy()
    weight = np.ones(len(train), dtype=float)
    weight[domain == "source5"] *= SOURCE_WEIGHT
    weight[domain != "source5"] *= 1.45
    weight[(domain != "source5") & (y == 0)] *= TARGET_NORMAL_WEIGHT
    weight[(y == 0) & (hard == 1)] *= HARD_NEG_WEIGHT
    if MODEL_KIND == "rf":
        model = RandomForestClassifier(
            n_estimators=TREE_N,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=TREE_N,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    model.fit(x, y, sample_weight=weight)
    return model, med


def predict_horizon(table: pd.DataFrame, feature_cols: list[str], ids: list[str], model, med, horizon: int) -> pd.DataFrame:
    data = table[table["sample_id"].isin(ids)].copy()
    x = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=np.float32)
    data[f"prob_{horizon}s"] = model.predict_proba(x)[:, 1]
    return data[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", f"prob_{horizon}s"]].copy()


def merge_probs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    base = parts[0]
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]
    for part in parts[1:]:
        prob_col = [c for c in part.columns if c.startswith("prob_")][0]
        base = base.merge(part[["sample_id", prob_col]], on="sample_id", how="left")
    return base[meta + [f"prob_{h}s" for h in HORIZONS]].copy()


def metrics_from_pred(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["binary"].astype(int).to_numpy()
    pred = frame["y_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hard = (y == 0) & (frame["hard_negative"].astype(int).to_numpy() == 1)
    delay = frame["delay_s"].dropna().to_numpy(dtype=float)
    return {
        "accuracy": float((tp + tn) / len(frame)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "hard_negative_fpr": float(pred[hard].mean()) if hard.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.median(delay)) if len(delay) else np.nan,
        "p95_delay_s": float(np.quantile(delay, 0.95)) if len(delay) else np.nan,
    }


def apply_thresholds(frame: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    prob = out[[f"prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    hits = prob >= thresholds.reshape(1, -1)
    pred = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    out["y_pred"] = pred.astype(int)
    out["alarm_time_s"] = alarm
    onset = out["onset_s"].to_numpy(dtype=float)
    out["delay_s"] = np.where(pred & (out["binary"].astype(int).to_numpy() == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    return out


def choose_thresholds(val: pd.DataFrame) -> tuple[np.ndarray, dict[str, float | int]]:
    prob = val[[f"prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    grid_by_h = []
    for h in HORIZONS:
        if h <= 75:
            grid_by_h.append(np.array([0.52, 0.66, 0.80, 0.92], dtype=float))
        else:
            grid_by_h.append(np.array([0.58, 0.72, 0.86, 0.94], dtype=float))
    best: tuple[float, np.ndarray, dict[str, float | int]] | None = None
    for thresholds in np.array(list(product(*grid_by_h)), dtype=float):
        pred = apply_thresholds(val, thresholds)
        m = metrics_from_pred(pred)
        score = (
            float(m["accuracy"])
            + 0.26 * float(m["specificity"])
            + 0.12 * float(m["f1"])
            + 0.06 * float(m["recall"])
            - 0.07 * float(m["hard_negative_fpr"])
            - 0.050 * float(m["fp"])
            - 0.035 * float(m["fn"])
            - 0.35 * max(0.0, 0.92 - float(m["recall"]))
        )
        if best is None or score > best[0]:
            best = (score, thresholds.copy(), m)
    assert best is not None
    return best[1], best[2]


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": f"WaveletPrefix{MODEL_KIND.upper()}_{OUTPUT_SUFFIX}", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    tables, feature_cols = load_prefix_tables()
    meta = target_meta(tables[HORIZONS[-1]])
    source_ids = set(tables[HORIZONS[-1]].loc[tables[HORIZONS[-1]]["domain"].astype(str) == "source5", "sample_id"].unique())
    seed_rows = pd.read_csv(OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv")
    rows = []
    pred_rows = []
    choice_rows = []
    seeds = sorted(seed_rows["seed"].unique())
    if SEED_LIMIT > 0:
        seeds = seeds[:SEED_LIMIT]
    print(f"feature_cols={len(feature_cols)}, seeds={len(seeds)}, model={MODEL_KIND}, mode={FEATURE_MODE}", flush=True)
    for seed in seeds:
        split = stratified_target_split(meta, int(seed))
        train_ids = set(split["train"]) | source_ids
        bundles = {h: fit_horizon(tables[h], feature_cols, train_ids, int(seed)) for h in HORIZONS}
        merged = {}
        for split_name in ["val", "test"]:
            parts = [
                predict_horizon(tables[h], feature_cols, split[split_name], bundles[h][0], bundles[h][1], h)
                for h in HORIZONS
            ]
            merged[split_name] = merge_probs(parts)
        thresholds, val_metrics = choose_thresholds(merged["val"])
        pred = apply_thresholds(merged["test"], thresholds)
        metrics = metrics_from_pred(pred)
        metrics["seed"] = int(seed)
        rows.append(metrics)
        pred["seed"] = int(seed)
        pred["split"] = "test"
        pred["model"] = f"WaveletPrefix{MODEL_KIND.upper()}_{OUTPUT_SUFFIX}"
        pred_rows.append(pred)
        choice_rows.append({"seed": int(seed), **{f"threshold_{h}s": float(t) for h, t in zip(HORIZONS, thresholds)}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"seed {seed}: acc={metrics['accuracy']:.4f}, spec={metrics['specificity']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(output_path("wavelet_prefix_metrics"), index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(output_path("wavelet_prefix_predictions"), index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(output_path("wavelet_prefix_choices"), index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    summary.to_csv(output_path("wavelet_prefix_summary"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

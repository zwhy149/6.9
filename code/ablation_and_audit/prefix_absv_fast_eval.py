from __future__ import annotations

from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix

from repeated_seed_eval import duplicate_group_name, stratified_target_split


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
HORIZONS = (50, 75, 100, 150, 250, 400)
PREFIX_CACHE_TAG = os.environ.get("FAST_PREFIX_CACHE_TAG", "rescompact_absv_v1")
OUTPUT_SUFFIX = os.environ.get("FAST_OUTPUT_SUFFIX", PREFIX_CACHE_TAG)
MODEL_KIND = os.environ.get("FAST_MODEL", "et").lower()
TREE_N = int(os.environ.get("FAST_TREE_N", "45"))


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


def load_tables() -> tuple[dict[int, pd.DataFrame], list[str]]:
    tables = {}
    all_cols: set[str] | None = None
    for horizon in HORIZONS:
        table = pd.read_csv(WORK / f"prefix_features_{PREFIX_CACHE_TAG}_{horizon}s.csv", low_memory=False)
        tables[horizon] = table
        numeric = {
            c
            for c in table.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(table[c])
        }
        all_cols = numeric if all_cols is None else all_cols & numeric
    assert all_cols is not None
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
        "_v0",
        "_vend",
        "_v_mean",
        "_v_min",
        "_v_max",
    )
    cols = sorted(c for c in all_cols if any(token in c for token in useful))
    return tables, cols


def target_meta(table: pd.DataFrame) -> pd.DataFrame:
    meta = (
        table[table["domain"].astype(str) != "source5"]
        [["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]]
        .drop_duplicates("sample_id")
        .copy()
    )
    meta["duplicate_group"] = [
        duplicate_group_name(file_name, int(binary))
        for file_name, binary in zip(meta["file_name"], meta["binary"])
    ]
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
    weight[domain == "source5"] *= 0.65
    weight[domain != "source5"] *= 1.4
    weight[(y == 0) & (hard == 1)] *= 3.0
    weight[y == 0] *= 1.15
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
    keep = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", f"prob_{horizon}s"]
    return data[keep].copy()


def merge_probs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    base = parts[0]
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]
    for part in parts[1:]:
        prob_col = [c for c in part.columns if c.startswith("prob_")][0]
        base = base.merge(part[["sample_id", prob_col]], on="sample_id", how="left")
    return base[meta + [f"prob_{h}s" for h in HORIZONS]].copy()


def metric_vectors(y_true: np.ndarray, pred: np.ndarray, alarm: np.ndarray, hard: np.ndarray, onset: np.ndarray) -> dict[str, np.ndarray]:
    y = y_true.astype(bool)
    n = len(y)
    tp = (pred & y.reshape(1, -1)).sum(axis=1).astype(float)
    tn = ((~pred) & (~y).reshape(1, -1)).sum(axis=1).astype(float)
    fp = (pred & (~y).reshape(1, -1)).sum(axis=1).astype(float)
    fn = ((~pred) & y.reshape(1, -1)).sum(axis=1).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    hard_mask = (~y) & (hard.astype(int) == 1)
    hard_fpr = pred[:, hard_mask].mean(axis=1) if hard_mask.any() else np.zeros(len(pred))
    delay = np.where(pred & y.reshape(1, -1) & np.isfinite(onset).reshape(1, -1), np.maximum(0.0, alarm - onset.reshape(1, -1)), np.nan)
    with np.errstate(all="ignore"):
        median_delay = np.nanmedian(delay, axis=1)
        p95_delay = np.nanquantile(delay, 0.95, axis=1)
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


def choose_thresholds(val: pd.DataFrame) -> np.ndarray:
    prob = val[[f"prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    y = val["binary"].to_numpy(dtype=int)
    hard = val["hard_negative"].to_numpy(dtype=int)
    onset = val["onset_s"].to_numpy(dtype=float)
    grid_by_h = []
    for i, horizon in enumerate(HORIZONS):
        fixed = np.array([0.55, 0.70, 0.84, 0.93], dtype=float)
        if horizon <= 75:
            fixed = np.array([0.50, 0.62, 0.78, 0.90], dtype=float)
        grid_by_h.append(fixed)
    thresholds = np.array(list(product(*grid_by_h)), dtype=float)
    hits = prob.reshape(1, len(val), len(HORIZONS)) >= thresholds.reshape(len(thresholds), 1, len(HORIZONS))
    pred = hits.any(axis=2)
    first = np.argmax(hits, axis=2)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    metrics = metric_vectors(y, pred, alarm, hard, onset)
    score = (
        metrics["accuracy"]
        + 0.18 * metrics["specificity"]
        + 0.15 * metrics["f1"]
        + 0.10 * metrics["recall"]
        - 0.06 * metrics["hard_negative_fpr"]
        - 0.035 * metrics["fp"]
        - 0.040 * metrics["fn"]
    )
    return thresholds[int(np.nanargmax(score))]


def apply_thresholds(frame: pd.DataFrame, thresholds: np.ndarray) -> tuple[pd.DataFrame, dict[str, float | int]]:
    out = frame.copy()
    prob = out[[f"prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    hits = prob >= thresholds.reshape(1, len(HORIZONS))
    pred = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    out["y_true"] = out["binary"].astype(int)
    out["y_pred"] = pred.astype(int)
    out["alarm_time_s"] = alarm
    onset = out["onset_s"].to_numpy(dtype=float)
    out["delay_s"] = np.where((out["y_true"].to_numpy() == 1) & pred & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    tn, fp, fn, tp = confusion_matrix(out["y_true"], out["y_pred"], labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard_mask = (out["y_true"].to_numpy() == 0) & (out["hard_negative"].to_numpy(dtype=int) == 1)
    valid_delay = out["delay_s"].dropna().to_numpy(dtype=float)
    metrics = {
        "accuracy": float((tp + tn) / len(out)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "hard_negative_fpr": float(out.loc[hard_mask, "y_pred"].mean()) if hard_mask.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.median(valid_delay)) if len(valid_delay) else np.nan,
        "p95_delay_s": float(np.quantile(valid_delay, 0.95)) if len(valid_delay) else np.nan,
    }
    return out, metrics


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": f"FastPrefix{MODEL_KIND.upper()}_{PREFIX_CACHE_TAG}", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    tables, feature_cols = load_tables()
    meta = target_meta(tables[400])
    source_ids = set(tables[400].loc[tables[400]["domain"].astype(str) == "source5", "sample_id"].unique())
    seed_rows = pd.read_csv(OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv")
    rows = []
    pred_rows = []
    choice_rows = []
    for seed in sorted(seed_rows["seed"].unique()):
        split = stratified_target_split(meta, int(seed))
        train_ids = set(split["train"]) | source_ids
        bundles = {}
        for horizon in HORIZONS:
            bundles[horizon] = fit_horizon(tables[horizon], feature_cols, train_ids, int(seed))
        merged = {}
        for split_name in ["val", "test"]:
            parts = [
                predict_horizon(tables[h], feature_cols, split[split_name], bundles[h][0], bundles[h][1], h)
                for h in HORIZONS
            ]
            merged[split_name] = merge_probs(parts)
        thresholds = choose_thresholds(merged["val"])
        pred, metrics = apply_thresholds(merged["test"], thresholds)
        metrics["seed"] = int(seed)
        rows.append(metrics)
        pred["seed"] = int(seed)
        pred["split"] = "test"
        pred["model"] = f"FastPrefix{MODEL_KIND.upper()}_{PREFIX_CACHE_TAG}"
        pred_rows.append(pred)
        choice_rows.append({"seed": int(seed), **{f"threshold_{h}s": float(t) for h, t in zip(HORIZONS, thresholds)}})
        print(f"seed {seed}: accuracy={metrics['accuracy']:.4f}, spec={metrics['specificity']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}", flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(output_path("fast_prefix_metrics"), index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(output_path("fast_prefix_predictions"), index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(output_path("fast_prefix_choices"), index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    summary.to_csv(output_path("fast_prefix_summary"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

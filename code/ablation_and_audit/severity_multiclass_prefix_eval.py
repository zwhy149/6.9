from __future__ import annotations

from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix

from repeated_seed_eval import duplicate_group_name, stratified_target_split


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
WORK = ROOT / "work"
HORIZONS = tuple(int(float(x)) for x in os.environ.get("SEV_HORIZONS", "50,75,100,150,250,400").split(",") if x.strip())
PREFIX_TAG = os.environ.get("SEV_PREFIX_TAG", "rescompact_global_v2")
OUTPUT_SUFFIX = os.environ.get("SEV_OUTPUT_SUFFIX", "sev_et_10seed")
MODEL_KIND = os.environ.get("SEV_MODEL", "et").lower()
TREE_N = int(os.environ.get("SEV_TREE_N", "45"))
SEED_LIMIT = int(os.environ.get("SEV_SEED_LIMIT", "10"))
SOURCE_WEIGHT = float(os.environ.get("SEV_SOURCE_WEIGHT", "0.65"))
TARGET_NORMAL_WEIGHT = float(os.environ.get("SEV_TARGET_NORMAL_WEIGHT", "1.5"))
HARD_NEG_WEIGHT = float(os.environ.get("SEV_HARD_NEG_WEIGHT", "3.0"))

META_COLS = {"sample_id", "domain", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "duplicate_group"}
CLASS_MAP = {None: 0, "": 0, "nan": 0, "ESC-10ohm": 1, "ESC-1ohm": 2, "ESC-0.1ohm": 3, "ESC-0.01ohm": 4}


def output_path(stem: str) -> Path:
    return OUT / f"{stem}_{OUTPUT_SUFFIX}.csv"


def severity_class(row: pd.Series) -> int:
    if int(row["binary"]) == 0:
        return 0
    return int(CLASS_MAP.get(str(row.get("severity_name", "")), 1))


def load_tables() -> tuple[dict[int, pd.DataFrame], list[str]]:
    tables: dict[int, pd.DataFrame] = {}
    all_cols: set[str] | None = None
    for horizon in HORIZONS:
        table = pd.read_csv(WORK / f"prefix_features_{PREFIX_TAG}_{horizon}s.csv", low_memory=False)
        table["sev_class"] = table.apply(severity_class, axis=1)
        tables[horizon] = table
        numeric = {c for c in table.columns if c not in META_COLS | {"sev_class"} and pd.api.types.is_numeric_dtype(table[c])}
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
    y = train["sev_class"].astype(int).to_numpy()
    binary = train["binary"].astype(int).to_numpy()
    domain = train["domain"].astype(str).to_numpy()
    hard = train["hard_negative"].astype(int).to_numpy()
    weight = np.ones(len(train), dtype=float)
    weight[domain == "source5"] *= SOURCE_WEIGHT
    weight[domain != "source5"] *= 1.4
    weight[(domain != "source5") & (binary == 0)] *= TARGET_NORMAL_WEIGHT
    weight[(binary == 0) & (hard == 1)] *= HARD_NEG_WEIGHT
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
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    p_normal = proba[:, classes.index(0)] if 0 in classes else np.zeros(len(data))
    data[f"fault_prob_{horizon}s"] = 1.0 - p_normal
    data[f"normal_prob_{horizon}s"] = p_normal
    keep = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", f"fault_prob_{horizon}s", f"normal_prob_{horizon}s"]
    return data[keep].copy()


def merge_probs(parts: list[pd.DataFrame]) -> pd.DataFrame:
    base = parts[0]
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]
    for part in parts[1:]:
        cols = [c for c in part.columns if c.endswith("s") and ("prob_" in c or "_prob" in c)]
        base = base.merge(part[["sample_id", *cols]], on="sample_id", how="left")
    return base[meta + [f"fault_prob_{h}s" for h in HORIZONS] + [f"normal_prob_{h}s" for h in HORIZONS]].copy()


def apply_thresholds(frame: pd.DataFrame, thresholds: np.ndarray, normal_floor: float) -> pd.DataFrame:
    out = frame.copy()
    fault = out[[f"fault_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    normal = out[[f"normal_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    hits = (fault >= thresholds.reshape(1, -1)) & (normal <= normal_floor)
    pred = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    out["y_pred"] = pred.astype(int)
    out["alarm_time_s"] = alarm
    onset = out["onset_s"].to_numpy(dtype=float)
    out["delay_s"] = np.where(pred & (out["binary"].astype(int).to_numpy() == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    return out


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
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


def choose_rule(val: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, float | int]]:
    grids = []
    for h in HORIZONS:
        grids.append(np.array([0.52, 0.66, 0.80, 0.92], dtype=float) if h <= 75 else np.array([0.58, 0.72, 0.86, 0.94], dtype=float))
    best = None
    for thresholds in np.array(list(product(*grids)), dtype=float):
        for normal_floor in (0.68, 0.78, 0.88, 1.01):
            pred = apply_thresholds(val, thresholds, normal_floor)
            m = metric_row(pred)
            score = (
                m["accuracy"]
                + 0.22 * m["specificity"]
                + 0.12 * m["f1"]
                + 0.05 * m["recall"]
                - 0.06 * m["hard_negative_fpr"]
                - 0.040 * m["fp"]
                - 0.035 * m["fn"]
            )
            if best is None or score > best[0]:
                best = (score, thresholds.copy(), float(normal_floor), m)
    assert best is not None
    return best[1], best[2], best[3]


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": f"SeverityMultiPrefix{MODEL_KIND.upper()}_{OUTPUT_SUFFIX}", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    tables, feature_cols = load_tables()
    meta = target_meta(tables[HORIZONS[-1]])
    source_ids = set(tables[HORIZONS[-1]].loc[tables[HORIZONS[-1]]["domain"].astype(str) == "source5", "sample_id"].unique())
    seed_rows = pd.read_csv(OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv")
    seeds = sorted(seed_rows["seed"].unique())
    if SEED_LIMIT > 0:
        seeds = seeds[:SEED_LIMIT]
    rows = []
    pred_rows = []
    choice_rows = []
    for seed in seeds:
        split = stratified_target_split(meta, int(seed))
        train_ids = set(split["train"]) | source_ids
        bundles = {h: fit_horizon(tables[h], feature_cols, train_ids, int(seed)) for h in HORIZONS}
        merged = {}
        for split_name in ["val", "test"]:
            parts = [predict_horizon(tables[h], feature_cols, split[split_name], bundles[h][0], bundles[h][1], h) for h in HORIZONS]
            merged[split_name] = merge_probs(parts)
        thresholds, normal_floor, val_metrics = choose_rule(merged["val"])
        pred = apply_thresholds(merged["test"], thresholds, normal_floor)
        metrics = metric_row(pred)
        metrics["seed"] = int(seed)
        rows.append(metrics)
        pred["seed"] = int(seed)
        pred["model"] = f"SeverityMultiPrefix{MODEL_KIND.upper()}_{OUTPUT_SUFFIX}"
        pred_rows.append(pred)
        choice_rows.append({"seed": int(seed), "normal_floor": normal_floor, **{f"threshold_{h}s": float(t) for h, t in zip(HORIZONS, thresholds)}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"seed {seed}: acc={metrics['accuracy']:.4f}, spec={metrics['specificity']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}", flush=True)
    pd.DataFrame(rows).to_csv(output_path("severity_multiclass_metrics"), index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(output_path("severity_multiclass_predictions"), index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(output_path("severity_multiclass_choices"), index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    summary.to_csv(output_path("severity_multiclass_summary"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

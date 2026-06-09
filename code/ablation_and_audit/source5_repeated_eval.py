from __future__ import annotations

from itertools import product
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from repeated_seed_eval import add_residual_compactness_features, build_feature_columns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
SEED_COUNT = int(os.environ.get("SOURCE5_N_SEEDS", "30"))
MODEL_GRID = tuple(x.strip().lower() for x in os.environ.get("SOURCE5_MODELS", "et,hgb,rf").split(",") if x.strip())
OUTPUT_SUFFIX = os.environ.get("SOURCE5_OUTPUT_SUFFIX", "").strip()


def output_path(stem: str) -> Path:
    suffix = f"_{OUTPUT_SUFFIX}" if OUTPUT_SUFFIX else ""
    return OUT / f"{stem}{suffix}.csv"


def file_meta(data: pd.DataFrame) -> pd.DataFrame:
    meta = (
        data[["sample_id", "file_name", "binary_file", "hard_negative_file", "severity_name"]]
        .drop_duplicates("sample_id")
        .rename(columns={"binary_file": "binary", "hard_negative_file": "hard_negative"})
        .copy()
    )
    meta["strat"] = np.where(
        meta["binary"].astype(int) == 0,
        np.where(meta["hard_negative"].astype(int) == 1, "normal_hard", "normal"),
        meta["severity_name"].astype(str),
    )
    return meta


def split_source(meta: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    counts = meta["strat"].value_counts()
    meta = meta.copy()
    meta["strat2"] = meta["strat"].where(~meta["strat"].isin(set(counts[counts < 3].index)), "rare")
    train_ids, temp_ids = train_test_split(
        meta["sample_id"],
        test_size=0.36,
        random_state=seed,
        stratify=meta["strat2"] if meta["strat2"].value_counts().min() >= 2 else None,
    )
    temp = meta[meta["sample_id"].isin(temp_ids)].copy()
    strat_temp = temp["strat2"] if temp["strat2"].value_counts().min() >= 2 else None
    val_ids, test_ids = train_test_split(temp["sample_id"], test_size=0.5, random_state=seed, stratify=strat_temp)
    return {"train": list(train_ids), "val": list(val_ids), "test": list(test_ids)}


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["binary"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard = (y_true == 0) & (frame["hard_negative"].to_numpy(dtype=int) == 1)
    delay = frame["delay_s"].to_numpy(dtype=float)
    valid_delay = delay[np.isfinite(delay)]
    return {
        "accuracy": float((tp + tn) / len(frame)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "hard_negative_fpr": float(y_pred[hard].mean()) if hard.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.median(valid_delay)) if len(valid_delay) else np.nan,
        "p95_delay_s": float(np.quantile(valid_delay, 0.95)) if len(valid_delay) else np.nan,
    }


def fit_model(kind: str, train: pd.DataFrame, feature_cols: list[str], seed: int):
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(med).to_numpy(dtype=float)
    y = train["binary_file"].astype(int).to_numpy()
    hard = ((train["binary_file"].astype(int).to_numpy() == 0) & (train["hard_negative_file"].astype(int).to_numpy() == 1))
    weight = np.ones(len(train), dtype=float)
    weight[hard] *= 2.5
    if kind == "hgb":
        model = HistGradientBoostingClassifier(max_iter=55, max_leaf_nodes=5, l2_regularization=1.5, random_state=seed)
    elif kind == "rf":
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=9,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    model.fit(x, y, sample_weight=weight)
    return {"kind": kind, "model": model, "med": med, "feature_cols": feature_cols}


def score_windows(bundle: dict, data: pd.DataFrame) -> pd.DataFrame:
    x = data[bundle["feature_cols"]].replace([np.inf, -np.inf], np.nan).fillna(bundle["med"])
    out = data[["sample_id", "file_name", "binary_file", "hard_negative_file", "severity_name", "onset_s", "t_end"]].copy()
    out["score"] = bundle["model"].predict_proba(x.to_numpy(dtype=float))[:, 1]
    return out


def file_scores(window_scores: pd.DataFrame, threshold: float, consecutive: int, min_alarm_s: float) -> pd.DataFrame:
    rows = []
    for sample_id, g in window_scores.sort_values("t_end").groupby("sample_id"):
        hit = (g["score"].to_numpy(dtype=float) >= threshold) & (g["t_end"].to_numpy(dtype=float) >= min_alarm_s)
        if consecutive > 1:
            run = np.convolve(hit.astype(int), np.ones(consecutive, dtype=int), mode="valid") >= consecutive
            if run.any():
                idx = int(np.where(run)[0][0] + consecutive - 1)
                alarm = float(g["t_end"].iloc[idx])
                pred = 1
            else:
                alarm = np.nan
                pred = 0
        else:
            if hit.any():
                idx = int(np.where(hit)[0][0])
                alarm = float(g["t_end"].iloc[idx])
                pred = 1
            else:
                alarm = np.nan
                pred = 0
        onset = float(g["onset_s"].iloc[0]) if pd.notna(g["onset_s"].iloc[0]) else np.nan
        delay = max(0.0, alarm - onset) if pred == 1 and int(g["binary_file"].iloc[0]) == 1 and np.isfinite(onset) else np.nan
        rows.append(
            {
                "sample_id": sample_id,
                "file_name": g["file_name"].iloc[0],
                "binary": int(g["binary_file"].iloc[0]),
                "hard_negative": int(g["hard_negative_file"].iloc[0]),
                "severity_name": g["severity_name"].iloc[0],
                "onset_s": onset,
                "y_pred": pred,
                "alarm_time_s": alarm,
                "delay_s": delay,
                "max_score": float(g["score"].max()),
                "p95_score": float(g["score"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def select_rule(val_scores: pd.DataFrame, objective: str) -> tuple[float, int, float, dict[str, float | int]]:
    values = val_scores["score"].to_numpy(dtype=float)
    grid = np.unique(np.concatenate([np.quantile(values, np.linspace(0.55, 0.995, 12)), np.array([0.65, 0.80, 0.90, 0.96, 0.985])]))
    best = None
    for threshold, consecutive, min_alarm_s in product(grid, (1, 2), (0.0, 60.0)):
        pred = file_scores(val_scores, float(threshold), int(consecutive), float(min_alarm_s))
        m = metric_row(pred)
        if objective == "accuracy":
            score = m["accuracy"] + 0.12 * m["f1"] + 0.04 * m["specificity"] - 0.02 * m["hard_negative_fpr"]
        elif objective == "specificity":
            recall_penalty = max(0.0, 0.94 - float(m["recall"]))
            score = 0.45 * m["specificity"] + 0.30 * m["accuracy"] + 0.20 * m["f1"] - 0.35 * recall_penalty - 0.08 * m["hard_negative_fpr"]
        else:
            score = 0.50 * ((m["recall"] + m["specificity"]) / 2.0) + 0.35 * m["accuracy"] + 0.15 * m["f1"] - 0.05 * m["hard_negative_fpr"]
        if best is None or score > best[0]:
            best = (score, float(threshold), int(consecutive), float(min_alarm_s), m)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    summaries = []
    for model, g in data.groupby("model"):
        row: dict[str, float | str | int] = {"model": model, "n_seeds": int(g["seed"].nunique())}
        for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["accuracy_mean", "specificity_mean"], ascending=False)


def main() -> None:
    data = pd.read_csv(WORK / "window_features.csv", low_memory=False)
    data = data[data["domain"] == "source5"].copy()
    data = add_residual_compactness_features(data)
    feature_cols = build_feature_columns(data)
    meta = file_meta(data)
    rows = []
    pred_rows = []
    choice_rows = []
    seeds = list(range(SEED_COUNT))
    for seed in seeds:
        split = split_source(meta, seed)
        train = data[data["sample_id"].isin(split["train"])].copy()
        val = data[data["sample_id"].isin(split["val"])].copy()
        test = data[data["sample_id"].isin(split["test"])].copy()
        for kind in MODEL_GRID:
            bundle = fit_model(kind, train, feature_cols, seed)
            val_scores = score_windows(bundle, val)
            test_scores = score_windows(bundle, test)
            for objective in ("accuracy", "balanced", "specificity"):
                threshold, consecutive, min_alarm_s, val_metrics = select_rule(val_scores, objective)
                pred = file_scores(test_scores, threshold, consecutive, min_alarm_s)
                metrics = metric_row(pred)
                model_name = f"Source5_{kind.upper()}_{objective}"
                metrics.update({"seed": seed, "model": model_name})
                rows.append(metrics)
                pred["seed"] = seed
                pred["model"] = model_name
                pred_rows.append(pred)
                choice_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "threshold": threshold,
                        "consecutive": consecutive,
                        "min_alarm_s": min_alarm_s,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                )
        if (seed + 1) % 5 == 0:
            print(f"completed {seed + 1}/{SEED_COUNT} seeds", flush=True)
    detail = pd.DataFrame(rows)
    summary = summarize(rows)
    detail.to_csv(output_path("source5_repeated_metrics"), index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(output_path("source5_repeated_predictions"), index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(output_path("source5_repeated_choices"), index=False, encoding="utf-8-sig")
    summary.to_csv(output_path("source5_repeated_summary"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from repeated_seed_eval import duplicate_group_name, stratified_target_split


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
WORK = ROOT / "work"

HORIZONS = (50, 75, 100, 150, 250, 400)
MODEL_NAME = "NPConformal_SpecificityControlled"
SOURCES = [
    ("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("global_et", "repeated_seed_predictions_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
]
META_COLS = [
    "sample_id",
    "file_name",
    "binary",
    "hard_negative",
    "severity_name",
    "onset_s",
    "duplicate_group",
    "seed",
    "split",
]


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name, low_memory=False)
    data = data[data["model"] == model_name].copy()
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    keep = [c for c in META_COLS if c in data.columns] + prob_cols
    out = data[keep].copy()
    return out.rename(columns={col: f"{prefix}_{col}" for col in prob_cols})


def load_base_probabilities() -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for prefix, file_name, model_name in SOURCES:
        part = load_source(prefix, file_name, model_name)
        prob_cols = [f"{prefix}_prob_{h}s" for h in HORIZONS]
        if base is None:
            base = part
        else:
            base = base.merge(part[["sample_id", "seed", "split", *prob_cols]], on=["sample_id", "seed", "split"], how="inner")
    if base is None:
        raise RuntimeError("No source predictions loaded.")
    return base


def target_meta_from_prefix() -> pd.DataFrame:
    table = pd.read_csv(WORK / "prefix_features_rescompact_global_v2_400s.csv", usecols=["sample_id", "domain", "file_name", "binary", "hard_negative", "severity_name", "onset_s"])
    meta = table[table["domain"].astype(str) != "source5"].drop_duplicates("sample_id").copy()
    meta["duplicate_group"] = [duplicate_group_name(f, int(b)) for f, b in zip(meta["file_name"], meta["binary"])]
    return meta[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "duplicate_group"]]


def ensemble_probs(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    prob = np.stack(
        [
            frame[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
            for prefix, _, _ in SOURCES
        ],
        axis=0,
    )
    return np.tensordot(weights, prob, axes=([0], [0]))


def file_margin_score(ensemble: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.max(ensemble - thresholds.reshape(1, -1), axis=1)


def apply_margin(frame: pd.DataFrame, weights: np.ndarray, thresholds: np.ndarray, margin: float) -> pd.DataFrame:
    out = frame.copy()
    ensemble = ensemble_probs(out, weights)
    adjusted = thresholds.reshape(1, -1) + float(margin)
    hits = ensemble >= adjusted
    pred = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~pred] = np.nan
    out["y_pred"] = pred.astype(int)
    out["alarm_time_s"] = alarm
    out["margin_score"] = file_margin_score(ensemble, thresholds)
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


def conformal_margin_from_normals(val_pred: pd.DataFrame, alpha: float, mode: str) -> float:
    normal_scores = val_pred.loc[val_pred["binary"].astype(int) == 0, "margin_score"].dropna().to_numpy(dtype=float)
    if len(normal_scores) == 0:
        return 0.0
    if mode == "max":
        return max(0.0, float(np.max(normal_scores)) + 1e-9)
    if mode == "q90":
        return max(0.0, float(np.quantile(normal_scores, 0.90)) + 1e-9)
    # Split-conformal one-sided quantile for false positive control.
    q_rank = int(np.ceil((len(normal_scores) + 1) * (1.0 - alpha))) - 1
    q_rank = min(max(q_rank, 0), len(normal_scores) - 1)
    return max(0.0, float(np.sort(normal_scores)[q_rank]) + 1e-9)


def summarize(rows: list[dict], model: str) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    summaries = []
    for variant, g in data.groupby("variant"):
        row: dict[str, float | str | int] = {"model": model, "variant": variant, "n_seeds": int(g["seed"].nunique())}
        for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s", "margin"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["specificity_mean", "accuracy_mean"], ascending=False)


def main() -> None:
    base = load_base_probabilities()
    choices = pd.read_csv(OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv")
    meta = target_meta_from_prefix()
    rows = []
    pred_rows = []
    choice_rows = []
    for seed in sorted(choices["seed"].unique()):
        split = stratified_target_split(meta, int(seed))
        seed_base = base[base["seed"].astype(int) == int(seed)].copy()
        choice = choices[choices["seed"].astype(int) == int(seed)].iloc[0]
        weights = np.array([float(choice[f"w_{prefix}"]) for prefix, _, _ in SOURCES], dtype=float)
        thresholds = np.array([float(choice[f"threshold_{h}s"]) for h in HORIZONS], dtype=float)
        val = seed_base[seed_base["split"] == "val"].copy()
        test = seed_base[seed_base["split"] == "test"].copy()
        val_base = apply_margin(val, weights, thresholds, margin=0.0)
        for alpha, mode in product((0.20, 0.15, 0.10, 0.05), ("conformal", "q90", "max")):
            margin = conformal_margin_from_normals(val_base, alpha=alpha, mode=mode)
            pred = apply_margin(test, weights, thresholds, margin=margin)
            metrics = metric_row(pred)
            variant = f"{mode}_alpha{alpha:.2f}"
            metrics.update({"seed": int(seed), "variant": variant, "margin": float(margin)})
            rows.append(metrics)
            pred["seed"] = int(seed)
            pred["split"] = "test"
            pred["model"] = MODEL_NAME
            pred["variant"] = variant
            pred_rows.append(pred)
            choice_rows.append({"seed": int(seed), "variant": variant, "alpha": alpha, "mode": mode, "margin": float(margin)})
        print(f"seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "np_conformal_specificity_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "np_conformal_specificity_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / "np_conformal_specificity_choices.csv", index=False, encoding="utf-8-sig")
    summary = summarize(rows, MODEL_NAME)
    summary.to_csv(OUT / "np_conformal_specificity_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

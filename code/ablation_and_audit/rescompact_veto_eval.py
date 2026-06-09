from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
HORIZONS = (50, 75, 100, 150)
OBJECTIVE = "balanced_accuracy"


def parse_thresholds(row: pd.Series) -> dict[int, float]:
    return {h: float(row[f"threshold_{h}s"]) for h in HORIZONS}


def attach_gate_features(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for horizon in HORIZONS:
        feats = pd.read_csv(WORK / f"prefix_features_rescompact_v1_{horizon}s.csv", low_memory=False)
        keep = ["sample_id"]
        for base in [
            "rc_smooth_trend_index",
            "rc_event_to_trend_index",
            "rc_drop_scale_cv",
            "rc_drop_scale_max_over_mean",
            "rc_residual_to_drop_mean",
            "rc_residual_to_drop_max",
        ]:
            keep.extend([c for c in feats.columns if c.startswith(base)])
        keep = list(dict.fromkeys(keep))
        feats = feats[keep].copy()
        rename = {c: f"{c}_h{horizon}" for c in feats.columns if c != "sample_id"}
        out = out.merge(feats.rename(columns=rename), on="sample_id", how="left")
    return out


def predict_with_thresholds(frame: pd.DataFrame, thresholds: dict[int, float]) -> pd.DataFrame:
    out = frame.copy()
    y_pred = np.zeros(len(out), dtype=int)
    alarm = np.full(len(out), np.nan)
    for horizon in HORIZONS:
        hit = out[f"prob_{horizon}s"].to_numpy(dtype=float) >= thresholds[horizon]
        new_hit = hit & (y_pred == 0)
        alarm[new_hit] = horizon
        y_pred[hit] = 1
    out["base_pred"] = y_pred
    out["alarm_time_s"] = alarm
    return out


def apply_veto(frame: pd.DataFrame, feature: str, threshold: float, direction: str) -> pd.DataFrame:
    out = frame.copy()
    values = out[feature].replace([np.inf, -np.inf], np.nan).fillna(out[feature].median()).to_numpy(dtype=float)
    if direction == "high":
        veto = values >= threshold
    else:
        veto = values <= threshold
    out["y_pred"] = np.where((out["base_pred"] == 1) & veto, 0, out["base_pred"])
    out["alarm_time_s"] = np.where(out["y_pred"] == 1, out["alarm_time_s"], np.nan)
    return out


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["binary"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    onset = frame["onset_s"].to_numpy(dtype=float)
    alarm = frame["alarm_time_s"].to_numpy(dtype=float)
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


def score(m: dict[str, float], min_recall: float = 0.90) -> float:
    penalty = 0.30 * max(0.0, min_recall - m["recall"])
    delay = m["median_delay_s"] if np.isfinite(m["median_delay_s"]) else 150.0
    return m["accuracy"] + 0.20 * m["specificity"] + 0.10 * m["f1"] - 0.04 * m["fp"] - delay / 3000.0 - penalty


def summarize(rows: list[dict]) -> dict[str, float]:
    data = pd.DataFrame(rows)
    out = {"n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        out[f"{col}_mean"] = float(data[col].mean())
        out[f"{col}_std"] = float(data[col].std(ddof=1))
        out[f"{col}_min"] = float(data[col].min())
        out[f"{col}_max"] = float(data[col].max())
    return out


def main() -> None:
    predictions = pd.read_csv(OUT / "repeated_seed_predictions_rescompact.csv")
    predictions = predictions[predictions["model"] == "EarlyCascadeHGB_50_75_100_150"].copy()
    predictions = attach_gate_features(predictions)
    threshold_rows = pd.read_csv(OUT / "rescompact_threshold_objective_thresholds.csv")
    threshold_rows = threshold_rows[threshold_rows["objective"] == OBJECTIVE].copy()
    candidate_features = [
        c
        for c in predictions.columns
        if c.startswith("rc_smooth_trend_index")
        or c.startswith("rc_event_to_trend_index")
        or c.startswith("rc_residual_to_drop")
        or c.startswith("rc_drop_scale_cv")
        or c.startswith("rc_drop_scale_max_over_mean")
    ]
    detail_rows = []
    choice_rows = []
    for seed in sorted(predictions["seed"].unique()):
        thresholds = parse_thresholds(threshold_rows[threshold_rows["seed"] == seed].iloc[0])
        val = predict_with_thresholds(predictions[(predictions["seed"] == seed) & (predictions["split"] == "val")], thresholds)
        test = predict_with_thresholds(predictions[(predictions["seed"] == seed) & (predictions["split"] == "test")], thresholds)
        base_val = val.copy()
        base_val["y_pred"] = base_val["base_pred"]
        best = (score(metrics(base_val)), "none", np.nan, "none")
        for feature in candidate_features:
            values = val[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(values) < 5:
                continue
            for direction in ["high", "low"]:
                for threshold in np.quantile(values, [0.50, 0.65, 0.80, 0.90]):
                    candidate = apply_veto(val, feature, float(threshold), direction)
                    cand_metrics = metrics(candidate)
                    cand_score = score(cand_metrics)
                    if cand_score > best[0]:
                        best = (cand_score, feature, float(threshold), direction)
        if best[1] == "none":
            chosen = test.copy()
            chosen["y_pred"] = chosen["base_pred"]
        else:
            chosen = apply_veto(test, best[1], best[2], best[3])
        row = metrics(chosen)
        row.update({"seed": int(seed), "feature": best[1], "threshold": best[2], "direction": best[3]})
        detail_rows.append(row)
        choice_rows.append({"seed": int(seed), "feature": best[1], "threshold": best[2], "direction": best[3]})
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame([{**summarize(detail_rows), "model": "ResCompactVeto_EarlyCascadeHGB"}])
    detail.to_csv(OUT / "rescompact_veto_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / "rescompact_veto_choices.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "rescompact_veto_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


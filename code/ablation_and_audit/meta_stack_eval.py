from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


SOURCES = [
    ("hgb_ext", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et_ext", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("hgb4", "repeated_seed_predictions_rescompact.csv", "EarlyCascadeHGB_50_75_100_150"),
    ("et4", "repeated_seed_predictions_rescompact_et.csv", "EarlyCascadeET_50_75_100_150"),
    ("rf4", "repeated_seed_predictions_rescompact_rf.csv", "EarlyCascadeRF_50_75_100_150"),
]


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name)
    data = data[data["model"] == model_name].copy()
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    prob_cols = [c for c in data.columns if c.startswith("prob_")]
    keep = meta + prob_cols
    out = data[keep].copy()
    out = out.rename(columns={col: f"{prefix}_{col}" for col in prob_cols})
    return out


def prepare() -> pd.DataFrame:
    base = None
    for source in SOURCES:
        part = load_source(*source)
        if base is None:
            base = part
        else:
            feature_cols = [c for c in part.columns if c.startswith(source[0] + "_prob_")]
            base = base.merge(part[["sample_id", "seed", "split", *feature_cols]], on=["sample_id", "seed", "split"], how="inner")
    assert base is not None
    prob_cols = [c for c in base.columns if "_prob_" in c]
    for prefix, _, _ in SOURCES:
        cols = [c for c in prob_cols if c.startswith(prefix + "_")]
        base[f"{prefix}_max"] = base[cols].max(axis=1)
        base[f"{prefix}_mean"] = base[cols].mean(axis=1)
        base[f"{prefix}_std"] = base[cols].std(axis=1).fillna(0.0)
    if "hgb_ext_prob_400s" in base.columns and "et_ext_prob_400s" in base.columns:
        for horizon in [50, 75, 100, 150, 250, 400]:
            h = f"hgb_ext_prob_{horizon}s"
            e = f"et_ext_prob_{horizon}s"
            if h in base.columns and e in base.columns:
                base[f"diff_hgb_et_{horizon}s"] = (base[h] - base[e]).abs()
    return base


def metric_row(frame: pd.DataFrame, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = frame["binary"].to_numpy(dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    horizons = np.array([50, 75, 100, 150, 250, 400], dtype=float)
    prob_matrix = np.column_stack(
        [
            frame[col].to_numpy(dtype=float)
            for col in [f"hgb_ext_prob_{int(h)}s" for h in horizons]
            if col in frame.columns
        ]
    )
    hit = prob_matrix >= 0.55
    alarm = np.full(len(frame), np.nan)
    for idx, horizon in enumerate(horizons[: hit.shape[1]]):
        new_hit = (y_pred == 1) & hit[:, idx] & np.isnan(alarm)
        alarm[new_hit] = horizon
    alarm[(y_pred == 1) & np.isnan(alarm)] = 150.0
    onset = frame["onset_s"].to_numpy(dtype=float)
    delay = np.where((y_true == 1) & (y_pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    hard = (y_true == 0) & (frame["hard_negative"].to_numpy(dtype=int) == 1)
    return {
        "accuracy": float((tp + tn) / len(y_true)),
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


def summarize(rows: list[dict], model_name: str) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": model_name, "n_seeds": int(data["seed"].nunique())}
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


def run_model(data: pd.DataFrame, model_name: str, estimator_factory) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in data.columns if "_prob_" in c or c.endswith(("_max", "_mean", "_std")) or c.startswith("diff_")]
    rows = []
    pred_rows = []
    for seed in sorted(data["seed"].unique()):
        val = data[(data["seed"] == seed) & (data["split"] == "val")].copy()
        test = data[(data["seed"] == seed) & (data["split"] == "test")].copy()
        x_val = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_val = val["binary"].to_numpy(dtype=int)
        model = estimator_factory(seed)
        model.fit(x_val.to_numpy(dtype=float), y_val)
        score = model.predict_proba(x_test.to_numpy(dtype=float))[:, 1]
        y_pred = (score >= 0.5).astype(int)
        metrics = metric_row(test, y_pred, score)
        metrics.update({"seed": int(seed)})
        rows.append(metrics)
        pred = test[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]].copy()
        pred["y_true"] = pred["binary"].astype(int)
        pred["y_pred"] = y_pred
        pred["score"] = score
        pred["model"] = model_name
        pred_rows.append(pred)
        print(f"{model_name} seed {seed}: accuracy={metrics['accuracy']:.4f}, fp={metrics['fp']}, fn={metrics['fn']}", flush=True)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def main() -> None:
    data = prepare()
    experiments = {
        "MetaStack_Logistic": lambda seed: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, class_weight="balanced", solver="liblinear", random_state=seed, max_iter=1000),
        ),
        "MetaStack_ExtraTreesTiny": lambda seed: ExtraTreesClassifier(
            n_estimators=80,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    summaries = []
    all_metrics = []
    all_predictions = []
    for name, factory in experiments.items():
        metrics, predictions = run_model(data, name, factory)
        metrics["model"] = name
        summaries.append(summarize(metrics, name))
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    summary = pd.concat(summaries, ignore_index=True).sort_values("accuracy_mean", ascending=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(OUT / "meta_stack_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT / "meta_stack_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "meta_stack_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

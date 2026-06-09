from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
HORIZONS = (50, 75, 100, 150, 250, 400)
SELECTOR_MARGIN = float(os.environ.get("SELECTOR_MARGIN", "0.0"))
SELECTOR_SUFFIX = os.environ.get("SELECTOR_SUFFIX", "").strip()
EXCLUDE_MODELS = {x.strip() for x in os.environ.get("EXCLUDE_MODELS", "").split(",") if x.strip()}


THREE_SRC = [
    ("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("global_et", "repeated_seed_predictions_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
]


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard = (y_true == 0) & (frame["hard_negative"].to_numpy(dtype=int) == 1)
    onset = frame["onset_s"].to_numpy(dtype=float)
    alarm = frame["alarm_time_s"].to_numpy(dtype=float)
    delay = np.where((y_true == 1) & (y_pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    valid_delay = delay[np.isfinite(delay)]
    return {
        "accuracy": float((tp + tn) / len(frame)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "hard_negative_fpr": float(y_pred[hard].mean()) if hard.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.median(valid_delay)) if len(valid_delay) else np.nan,
        "p95_delay_s": float(np.quantile(valid_delay, 0.95)) if len(valid_delay) else np.nan,
    }


def score(metrics: dict[str, float | int]) -> float:
    delay = float(metrics["median_delay_s"]) if np.isfinite(metrics["median_delay_s"]) else 120.0
    return (
        float(metrics["accuracy"])
        + 0.16 * float(metrics["f1"])
        + 0.18 * float(metrics["specificity"])
        + 0.10 * float(metrics["recall"])
        - 0.07 * float(metrics["fp"])
        - 0.055 * float(metrics["fn"])
        - 0.10 * float(metrics["hard_negative_fpr"])
        - delay / 5000.0
    )


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name, low_memory=False)
    data = data[data["model"] == model_name].copy()
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    meta = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    return data[meta + prob_cols].rename(columns={c: f"{prefix}_{c}" for c in prob_cols})


def load_three_src_probs() -> pd.DataFrame:
    base = None
    for prefix, file_name, model_name in THREE_SRC:
        part = load_source(prefix, file_name, model_name)
        prob_cols = [f"{prefix}_prob_{h}s" for h in HORIZONS]
        if base is None:
            base = part
        else:
            base = base.merge(part[["sample_id", "seed", "split", *prob_cols]], on=["sample_id", "seed", "split"], how="inner")
    assert base is not None
    return base


def reconstruct_three_src_candidate(prob_data: pd.DataFrame, choices_file: str, name: str) -> pd.DataFrame:
    choices = pd.read_csv(OUT / choices_file)
    rows = []
    for seed in sorted(choices["seed"].unique()):
        choice = choices[choices["seed"] == seed].iloc[0]
        weight = np.array([float(choice[f"w_{prefix}"]) for prefix, _, _ in THREE_SRC], dtype=float)
        threshold = np.array([float(choice[f"threshold_{h}s"]) for h in HORIZONS], dtype=float)
        data = prob_data[prob_data["seed"].astype(int) == int(seed)].copy()
        prob = np.stack(
            [
                data[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
                for prefix, _, _ in THREE_SRC
            ],
            axis=0,
        )
        ensemble = np.tensordot(weight, prob, axes=([0], [0]))
        hits = ensemble >= threshold.reshape(1, -1)
        pred = hits.any(axis=1)
        first = np.argmax(hits, axis=1)
        alarm = np.take(np.array(HORIZONS, dtype=float), first)
        alarm[~pred] = np.nan
        out = data[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]].copy()
        out["y_true"] = out["binary"].astype(int)
        out["y_pred"] = pred.astype(int)
        out["alarm_time_s"] = alarm
        out["model"] = name
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def reconstruct_ext_candidate(name: str, choices_file: str) -> pd.DataFrame:
    hgb = load_source("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400")
    et = load_source("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400")
    data = hgb.merge(
        et[["sample_id", "seed", "split", *[f"et_prob_{h}s" for h in HORIZONS]]],
        on=["sample_id", "seed", "split"],
        how="inner",
    )
    choices = pd.read_csv(OUT / choices_file)
    rows = []
    for seed in sorted(choices["seed"].unique()):
        choice = choices[choices["seed"] == seed].iloc[0]
        weight = float(choice["weight_hgb"])
        threshold = np.array([float(choice[f"threshold_{h}s"]) for h in HORIZONS], dtype=float)
        part = data[data["seed"].astype(int) == int(seed)].copy()
        hgb_prob = part[[f"hgb_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
        et_prob = part[[f"et_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
        prob = weight * hgb_prob + (1.0 - weight) * et_prob
        hits = prob >= threshold.reshape(1, -1)
        pred = hits.any(axis=1)
        first = np.argmax(hits, axis=1)
        alarm = np.take(np.array(HORIZONS, dtype=float), first)
        alarm[~pred] = np.nan
        out = part[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]].copy()
        out["y_true"] = out["binary"].astype(int)
        out["y_pred"] = pred.astype(int)
        out["alarm_time_s"] = alarm
        out["model"] = name
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def load_direct_candidate(file_name: str, model_filter: str, name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name, low_memory=False)
    data = data[data["model"] == model_filter].copy()
    out = data[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split", "y_true", "y_pred", "alarm_time_s"]].copy()
    out["model"] = name
    return out


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": "ValidationSelected_ModelPool", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    prob_data = load_three_src_probs()
    candidates = [
        reconstruct_three_src_candidate(
            prob_data,
            "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv",
            "3src_accuracy",
        ),
        reconstruct_three_src_candidate(
            prob_data,
            "rescompact_multisource_3src_ensemble_specificity_guard_choices.csv",
            "3src_specificity_guard",
        ),
        reconstruct_three_src_candidate(
            prob_data,
            "rescompact_multisource_3src_ensemble_balanced_specificity_choices.csv",
            "3src_balanced_specificity",
        ),
        reconstruct_three_src_candidate(
            prob_data,
            "rescompact_multisource_3src_ensemble_specificity_first_choices.csv",
            "3src_specificity_first",
        ),
        reconstruct_ext_candidate(
            "ext_hgb_et_accuracy",
            "rescompact_ext_ensemble_highgrid_accuracy_only_choices.csv",
        ),
        load_direct_candidate(
            "repeated_seed_predictions_global_et_ext.csv",
            "EarlyCascadeET_50_75_100_150_250_400",
            "global_et_ext",
        ),
        load_direct_candidate(
            "repeated_seed_predictions_rescompact_et_ext.csv",
            "EarlyCascadeET_50_75_100_150_250_400",
            "target_source_et_ext",
        ),
        load_direct_candidate(
            "repeated_seed_predictions_rescompact_hgb_ext.csv",
            "EarlyCascadeHGB_50_75_100_150_250_400",
            "target_source_hgb_ext",
        ),
    ]
    all_pred = pd.concat(candidates, ignore_index=True)
    rows = []
    pred_rows = []
    choice_rows = []
    val_metric_rows = []
    for seed in sorted(all_pred["seed"].unique()):
        val_scores = []
        for model, frame in all_pred[(all_pred["seed"] == seed) & (all_pred["split"] == "val")].groupby("model"):
            metrics = metric_row(frame)
            metrics.update({"seed": int(seed), "model": model, "val_score": score(metrics)})
            val_scores.append(metrics)
        val_metrics = pd.DataFrame(val_scores)
        if EXCLUDE_MODELS:
            val_metrics = val_metrics[~val_metrics["model"].isin(EXCLUDE_MODELS)].copy()
        val_metrics = val_metrics.sort_values(["val_score", "accuracy", "specificity"], ascending=False)
        best_model = str(val_metrics.iloc[0]["model"])
        default_score = float(val_metrics[val_metrics["model"] == "3src_accuracy"]["val_score"].iloc[0])
        chosen_model = "3src_accuracy" if float(val_metrics.iloc[0]["val_score"]) < default_score + SELECTOR_MARGIN else best_model
        choice_rows.append({"seed": int(seed), "chosen_model": chosen_model, "val_score": float(val_metrics.iloc[0]["val_score"])})
        val_metric_rows.extend(val_scores)
        test = all_pred[
            (all_pred["seed"] == seed)
            & (all_pred["split"] == "test")
            & (all_pred["model"] == chosen_model)
        ].copy()
        metrics = metric_row(test)
        metrics.update({"seed": int(seed), "chosen_model": chosen_model})
        rows.append(metrics)
        pred_rows.append(test.assign(selector_model="ValidationSelected_ModelPool"))
        print(
            f"seed {seed}: chose={chosen_model}, test_accuracy={metrics['accuracy']:.4f}, "
            f"spec={metrics['specificity']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}",
            flush=True,
        )
    detail = pd.DataFrame(rows)
    suffix = f"_{SELECTOR_SUFFIX}" if SELECTOR_SUFFIX else ""
    detail.to_csv(OUT / f"validation_model_selector_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / f"validation_model_selector_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / f"validation_model_selector_choices{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_metric_rows).to_csv(OUT / f"validation_model_selector_val_metrics{suffix}.csv", index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    summary.to_csv(OUT / f"validation_model_selector_summary{suffix}.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

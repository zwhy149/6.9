from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from repeated_seed_eval import duplicate_group_name, stratified_target_split
from rescompact_normality_verifier_eval import (
    HORIZONS,
    SOURCES,
    attach_scores,
    feature_rows_for_ids,
    fit_verifiers,
    load_base_probabilities,
    load_prefix_feature_long,
    reconstruct_base,
    rows_at_alarm,
    target_meta_from_features,
    verifier_scores,
)


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
WORK = ROOT / "work"
MODEL_NAME = "DualEvidenceVeto_ValidationSelected"


def target_meta_from_window() -> pd.DataFrame:
    cols = ["sample_id", "domain", "file_name", "binary_file", "hard_negative_file", "severity_name", "onset_s"]
    data = pd.read_csv(WORK / "window_features.csv", usecols=cols)
    meta = (
        data[data["domain"].astype(str) != "source5"]
        .drop_duplicates("sample_id")
        .rename(columns={"binary_file": "binary", "hard_negative_file": "hard_negative"})
        .copy()
    )
    meta["duplicate_group"] = [duplicate_group_name(f, int(b)) for f, b in zip(meta["file_name"], meta["binary"])]
    return meta[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "duplicate_group"]]


def load_alarm_feature_lookup() -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    usecols = [
        "sample_id",
        "t_end",
        "w6_risk_score",
        "w12_risk_score",
        "w48_risk_score",
        "w6_abs_dvdt_q95_norm",
        "w12_abs_dvdt_q95_norm",
        "w48_abs_dvdt_q95_norm",
        "w6_res_absmax_norm",
        "w12_res_absmax_norm",
        "w48_res_absmax_norm",
        "w6_abs_drop_norm",
        "w48_abs_drop_norm",
        "w6_slope_norm",
        "w48_slope_norm",
    ]
    data = pd.read_csv(WORK / "window_features.csv", usecols=usecols)
    lookup: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for sid, group in data.sort_values(["sample_id", "t_end"]).groupby("sample_id"):
        lookup[str(sid)] = (group["t_end"].to_numpy(dtype=float), group.reset_index(drop=True))
    return lookup


def attach_alarm_features(frame: pd.DataFrame, lookup: dict[str, tuple[np.ndarray, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        rec = row.to_dict()
        sid = str(row["sample_id"])
        alarm = row.get("base_alarm_time_s")
        if sid in lookup and pd.notna(alarm):
            times, group = lookup[sid]
            idx = int(np.searchsorted(times, float(alarm), side="right") - 1)
            idx = min(max(idx, 0), len(group) - 1)
            for col, value in group.iloc[idx].items():
                if col not in {"sample_id"}:
                    rec[col] = value
            rec["alarm_window_t_end"] = float(group.iloc[idx]["t_end"])
        rows.append(rec)
    out = pd.DataFrame(rows)
    eps = 1e-9
    out["rough_6_48"] = out["w6_res_absmax_norm"].astype(float) / (out["w48_res_absmax_norm"].astype(float) + eps)
    out["accel_dvdt_6_48"] = out["w6_abs_dvdt_q95_norm"].astype(float) / (out["w48_abs_dvdt_q95_norm"].astype(float) + eps)
    out["accel_drop_6_48"] = out["w6_abs_drop_norm"].astype(float) / (out["w48_abs_drop_norm"].astype(float) + eps)
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
    onset = frame["onset_s"].to_numpy(dtype=float)
    alarm = frame["alarm_time_s"].to_numpy(dtype=float)
    delay = np.where((y == 1) & (pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    valid_delay = delay[np.isfinite(delay)]
    return {
        "accuracy": float((tp + tn) / len(frame)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "hard_negative_fpr": float(pred[hard].mean()) if hard.any() else 0.0,
        "fp": int(fp),
        "fn": int(fn),
        "median_delay_s": float(np.median(valid_delay)) if len(valid_delay) else np.nan,
        "p95_delay_s": float(np.quantile(valid_delay, 0.95)) if len(valid_delay) else np.nan,
    }


def apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    out = frame.copy()
    base_positive = out["base_pred"].astype(int).to_numpy() == 1
    veto = np.zeros(len(out), dtype=bool)
    if rule["type"] == "none":
        veto[:] = False
    elif rule["type"] == "rough":
        veto = (
            base_positive
            & (out["base_alarm_score"].fillna(1.0).to_numpy(dtype=float) <= float(rule["score_cap"]))
            & (out["rough_6_48"].fillna(np.inf).to_numpy(dtype=float) <= float(rule["rough_thr"]))
        )
    elif rule["type"] == "triple":
        veto = (
            base_positive
            & (out["base_alarm_score"].fillna(1.0).to_numpy(dtype=float) <= float(rule["score_cap"]))
            & (out["et_fault_prob"].fillna(1.0).to_numpy(dtype=float) <= float(rule["et_thr"]))
            & (out["w6_risk_score"].fillna(np.inf).to_numpy(dtype=float) <= float(rule["risk_thr"]))
        )
    out["y_pred"] = np.where(veto, 0, out["base_pred"].astype(int).to_numpy())
    out["alarm_time_s"] = np.where(out["y_pred"].astype(int).to_numpy() == 1, out["base_alarm_time_s"].to_numpy(dtype=float), np.nan)
    out["vetoed"] = veto.astype(int)
    return out


def candidate_values(values: pd.Series, qmax: float = 0.70) -> np.ndarray:
    arr = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=float)
    return np.unique(np.quantile(arr, np.linspace(0.05, qmax, 10)))


def choose_rule(val: pd.DataFrame) -> tuple[dict, dict[str, float | int]]:
    best_rule = {"type": "none"}
    best_frame = apply_rule(val, best_rule)
    best_metrics = metric_row(best_frame)
    best_score = score_metrics(best_metrics)
    base_pos = val[val["base_pred"].astype(int) == 1].copy()
    score_caps = np.linspace(0.575, 0.900, 14)
    for cap in score_caps:
        for rough_thr in candidate_values(base_pos["rough_6_48"], qmax=0.70):
            rule = {"type": "rough", "score_cap": float(cap), "rough_thr": float(rough_thr)}
            metrics = metric_row(apply_rule(val, rule))
            score = score_metrics(metrics)
            if score > best_score:
                best_score, best_rule, best_metrics = score, rule, metrics
        for et_thr in np.linspace(0.50, 0.68, 10):
            for risk_thr in candidate_values(base_pos["w6_risk_score"], qmax=0.70):
                rule = {"type": "triple", "score_cap": float(cap), "et_thr": float(et_thr), "risk_thr": float(risk_thr)}
                metrics = metric_row(apply_rule(val, rule))
                score = score_metrics(metrics)
                if score > best_score:
                    best_score, best_rule, best_metrics = score, rule, metrics
    return best_rule, best_metrics


def score_metrics(metrics: dict[str, float | int]) -> float:
    recall = float(metrics["recall"])
    specificity = float(metrics["specificity"])
    accuracy = float(metrics["accuracy"])
    # Encourage a reviewer-facing operating point around 0.90 specificity only if recall is still usable.
    return (
        accuracy
        + 0.34 * specificity
        + 0.12 * float(metrics["f1"])
        + 0.05 * recall
        - 0.12 * float(metrics["hard_negative_fpr"])
        - 0.050 * float(metrics["fp"])
        - 0.040 * float(metrics["fn"])
        - 0.60 * max(0.0, 0.92 - recall)
    )


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": MODEL_NAME, "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s", "vetoed"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    base = load_base_probabilities()
    choices = pd.read_csv(OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv")
    prefix_long, feature_cols = load_prefix_feature_long()
    target_meta = target_meta_from_features(prefix_long)
    # Use the same split metadata as prior final runs, but raw window features for local alarm evidence.
    source_ids = set(prefix_long.loc[prefix_long["domain"].astype(str) == "source5", "sample_id"].unique())
    lookup = load_alarm_feature_lookup()
    rows = []
    pred_rows = []
    choice_rows = []
    for seed in sorted(choices["seed"].unique()):
        split = stratified_target_split(target_meta, int(seed))
        train_ids = set(split["train"])
        train_rows = pd.concat(
            [feature_rows_for_ids(prefix_long, train_ids), feature_rows_for_ids(prefix_long, source_ids)],
            ignore_index=True,
            sort=False,
        )
        verifiers = fit_verifiers(train_rows, feature_cols, int(seed))
        choice = choices[choices["seed"].astype(int) == int(seed)].iloc[0]
        seed_base = base[base["seed"].astype(int) == int(seed)].copy()
        val = reconstruct_base(seed_base[seed_base["split"] == "val"].copy(), choice)
        test = reconstruct_base(seed_base[seed_base["split"] == "test"].copy(), choice)
        val_scores = verifier_scores(verifiers, feature_cols, rows_at_alarm(prefix_long, val))
        test_scores = verifier_scores(verifiers, feature_cols, rows_at_alarm(prefix_long, test))
        val_scored = attach_alarm_features(attach_scores(val, val_scores), lookup)
        test_scored = attach_alarm_features(attach_scores(test, test_scores), lookup)
        rule, val_metrics = choose_rule(val_scored)
        chosen = apply_rule(test_scored, rule)
        metrics = metric_row(chosen)
        metrics.update({"seed": int(seed), "vetoed": int(chosen["vetoed"].sum())})
        rows.append(metrics)
        choice_rows.append({"seed": int(seed), **rule, **{f"val_{k}": v for k, v in val_metrics.items()}})
        pred = chosen.copy()
        pred["seed"] = int(seed)
        pred["model"] = MODEL_NAME
        pred_rows.append(pred)
        print(f"seed {seed}: acc={metrics['accuracy']:.4f}, spec={metrics['specificity']:.3f}, fp={metrics['fp']}, fn={metrics['fn']}, rule={rule['type']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "dual_evidence_veto_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / "dual_evidence_veto_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "dual_evidence_veto_predictions.csv", index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    summary.to_csv(OUT / "dual_evidence_veto_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

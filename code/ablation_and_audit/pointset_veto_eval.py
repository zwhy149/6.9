from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from repeated_seed_eval import duplicate_group_name, stratified_target_split
from validation_model_selector_eval import (
    load_three_src_probs,
    load_direct_candidate,
    metric_row as selector_metric_row,
    reconstruct_ext_candidate,
    reconstruct_three_src_candidate,
)


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
WORK = ROOT / "work"
HORIZONS = (50, 75, 100, 150, 250, 400)
MODEL_NAME = "PointSetPrototypeVeto_ValidationSelected"
OUTPUT_SUFFIX = os.environ.get("POINTSET_OUTPUT_SUFFIX", "").strip()
SPEC_WEIGHT = float(os.environ.get("POINTSET_SPEC_WEIGHT", "0.34"))
FP_WEIGHT = float(os.environ.get("POINTSET_FP_WEIGHT", "0.030"))
FN_WEIGHT = float(os.environ.get("POINTSET_FN_WEIGHT", "0.055"))
RECALL_PENALTY_WEIGHT = float(os.environ.get("POINTSET_RECALL_PENALTY_WEIGHT", "0.80"))
RECALL_DELTA = float(os.environ.get("POINTSET_RECALL_DELTA", "0.02"))
RECALL_FLOOR_MIN = float(os.environ.get("POINTSET_RECALL_FLOOR_MIN", "0.92"))
RECALL_FLOOR_MAX = float(os.environ.get("POINTSET_RECALL_FLOOR_MAX", "0.96"))


def out_path(stem: str) -> Path:
    suffix = f"_{OUTPUT_SUFFIX}" if OUTPUT_SUFFIX else ""
    return OUT / f"{stem}{suffix}.csv"


def reconstruct_selector_candidates() -> pd.DataFrame:
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
    return pd.concat(candidates, ignore_index=True)


def chosen_selector_frame(all_pred: pd.DataFrame, seed: int, split: str, chosen_model: str) -> pd.DataFrame:
    frame = all_pred[
        (all_pred["seed"].astype(int) == int(seed))
        & (all_pred["split"].astype(str) == split)
        & (all_pred["model"].astype(str) == chosen_model)
    ].copy()
    frame["base_pred"] = frame["y_pred"].astype(int)
    frame["base_alarm_time_s"] = frame["alarm_time_s"].astype(float)
    frame["y_pred"] = frame["base_pred"].astype(int)
    frame["alarm_time_s"] = frame["base_alarm_time_s"]
    return frame


def normalize_pg_columns(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    prefix = f"pg_{horizon}s_"
    rename = {c: f"pg_{c[len(prefix):]}" for c in data.columns if c.startswith(prefix)}
    return data.rename(columns=rename)


def load_prefix_feature_long() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for horizon in HORIZONS:
        table = pd.read_csv(WORK / f"prefix_features_rescompact_cp_v1_{horizon}s.csv", low_memory=False)
        table = normalize_pg_columns(table, horizon)
        table["alarm_horizon_s"] = float(horizon)
        table["alarm_horizon_norm"] = float(horizon) / max(HORIZONS)
        rows.append(table)
    data = pd.concat(rows, ignore_index=True, sort=False)
    meta = {
        "sample_id",
        "domain",
        "file_name",
        "binary",
        "hard_negative",
        "severity_name",
        "onset_s",
        "duplicate_group",
    }
    numeric_cols = [
        c
        for c in data.columns
        if c not in meta and pd.api.types.is_numeric_dtype(data[c])
    ]
    tokens = (
        "drop_norm",
        "max_drop_norm",
        "range_norm",
        "slope_norm",
        "dvdt",
        "res_",
        "monotonicity",
        "risk_score",
        "smooth_trend",
        "event_to_trend",
        "cp_",
        "curvature",
        "local_drop",
        "alarm_horizon",
    )
    feature_cols = sorted(c for c in numeric_cols if any(token in c for token in tokens))
    return data, feature_cols


def target_meta_from_features(prefix_long: pd.DataFrame) -> pd.DataFrame:
    meta = (
        prefix_long[prefix_long["domain"].astype(str) != "source5"]
        [["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]]
        .drop_duplicates("sample_id")
        .copy()
    )
    meta["duplicate_group"] = [
        duplicate_group_name(file_name, int(binary))
        for file_name, binary in zip(meta["file_name"], meta["binary"])
    ]
    return meta


def feature_rows_for_ids(prefix_long: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return prefix_long[prefix_long["sample_id"].isin(ids)].copy()


def robust_scale_fit(rows: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.Series]:
    x = rows[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).fillna(0.0)
    q75 = x.quantile(0.75)
    q25 = x.quantile(0.25)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(x.std(axis=0).replace(0.0, np.nan)).fillna(1.0)
    return {"median": med, "scale": scale}


def transform(rows: pd.DataFrame, feature_cols: list[str], scaler: dict[str, pd.Series]) -> np.ndarray:
    x = rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(scaler["median"])
    z = ((x - scaler["median"]) / scaler["scale"]).clip(-8.0, 8.0)
    return z.to_numpy(dtype=np.float32)


def fit_neighbors(train_rows: pd.DataFrame, feature_cols: list[str]) -> dict:
    scaler = robust_scale_fit(train_rows, feature_cols)
    y = train_rows["binary"].astype(int).to_numpy()
    domain = train_rows["domain"].astype(str).to_numpy()
    target_normal = train_rows[(domain != "source5") & (y == 0)]
    target_fault = train_rows[(domain != "source5") & (y == 1)]
    source_fault = train_rows[(domain == "source5") & (y == 1)]
    all_fault = pd.concat([target_fault, source_fault], ignore_index=True, sort=False)
    if len(target_normal) == 0 or len(all_fault) == 0:
        raise RuntimeError("Point-set veto needs both target normal and fault support.")

    bundles = {"scaler": scaler, "feature_cols": feature_cols}
    for name, rows in [("target_normal", target_normal), ("target_fault", target_fault), ("all_fault", all_fault)]:
        x = transform(rows, feature_cols, scaler)
        k = max(1, min(5, len(rows)))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(x)
        bundles[name] = {"nn": nn, "k": k}
    return bundles


def rows_at_alarm(prefix_long: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame[["sample_id", "base_alarm_time_s"]].copy()
    tmp["alarm_horizon_s"] = tmp["base_alarm_time_s"].fillna(max(HORIZONS)).astype(float)
    tmp["alarm_horizon_s"] = tmp["alarm_horizon_s"].clip(min(HORIZONS), max(HORIZONS)).round().astype(int)
    keyed = prefix_long.copy()
    keyed["alarm_horizon_s"] = keyed["alarm_horizon_s"].round().astype(int)
    return tmp.merge(keyed, on=["sample_id", "alarm_horizon_s"], how="left")


def mean_knn_distance(bundle: dict, x: np.ndarray) -> np.ndarray:
    dist, _ = bundle["nn"].kneighbors(x, return_distance=True)
    return dist.mean(axis=1)


def attach_pointset_scores(frame: pd.DataFrame, prefix_long: pd.DataFrame, neighbors: dict) -> pd.DataFrame:
    rows = rows_at_alarm(prefix_long, frame)
    x = transform(rows, neighbors["feature_cols"], neighbors["scaler"])
    out = frame.copy()
    out["d_target_normal"] = mean_knn_distance(neighbors["target_normal"], x)
    out["d_target_fault"] = mean_knn_distance(neighbors["target_fault"], x)
    out["d_all_fault"] = mean_knn_distance(neighbors["all_fault"], x)
    out["d_fault_min"] = np.minimum(out["d_target_fault"].to_numpy(dtype=float), out["d_all_fault"].to_numpy(dtype=float))
    out["normal_margin"] = out["d_fault_min"] - out["d_target_normal"]
    out["normal_ratio"] = out["d_target_normal"] / (out["d_fault_min"] + 1e-9)
    return out


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["binary"].astype(int).to_numpy()
    y_pred = frame["y_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    onset = frame["onset_s"].to_numpy(dtype=float)
    alarm = frame["alarm_time_s"].to_numpy(dtype=float)
    delay = np.where((y_true == 1) & (y_pred == 1) & np.isfinite(onset), np.maximum(0.0, alarm - onset), np.nan)
    hard = (y_true == 0) & (frame["hard_negative"].astype(int).to_numpy() == 1)
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


def score_metrics(metrics: dict[str, float | int], base_metrics: dict[str, float | int]) -> float:
    recall = float(metrics["recall"])
    base_recall = float(base_metrics["recall"])
    recall_floor = min(RECALL_FLOOR_MAX, max(RECALL_FLOOR_MIN, base_recall - RECALL_DELTA))
    recall_penalty = max(0.0, recall_floor - recall)
    return (
        float(metrics["accuracy"])
        + SPEC_WEIGHT * float(metrics["specificity"])
        + 0.10 * float(metrics["f1"])
        + 0.04 * recall
        - 0.08 * float(metrics["hard_negative_fpr"])
        - FP_WEIGHT * float(metrics["fp"])
        - FN_WEIGHT * float(metrics["fn"])
        - RECALL_PENALTY_WEIGHT * recall_penalty
    )


def apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    out = frame.copy()
    base_positive = out["base_pred"].astype(int).to_numpy() == 1
    if rule["type"] == "none":
        veto = np.zeros(len(out), dtype=bool)
    elif rule["type"] == "margin":
        veto = out["normal_margin"].to_numpy(dtype=float) >= float(rule["margin_threshold"])
    elif rule["type"] == "ratio":
        veto = out["normal_ratio"].to_numpy(dtype=float) <= float(rule["ratio_threshold"])
    elif rule["type"] == "hybrid":
        veto = (
            (out["normal_margin"].to_numpy(dtype=float) >= float(rule["margin_threshold"]))
            & (out["normal_ratio"].to_numpy(dtype=float) <= float(rule["ratio_threshold"]))
        )
    else:
        raise ValueError(f"Unknown rule type: {rule['type']}")
    if "min_alarm_s" in rule:
        veto &= out["base_alarm_time_s"].fillna(0.0).to_numpy(dtype=float) >= float(rule["min_alarm_s"])
    veto &= base_positive
    out["y_pred"] = np.where(veto, 0, out["base_pred"].astype(int).to_numpy())
    out["alarm_time_s"] = np.where(out["y_pred"].to_numpy(dtype=int) == 1, out["base_alarm_time_s"].to_numpy(dtype=float), np.nan)
    out["vetoed"] = veto.astype(int)
    return out


def thresholds(values: pd.Series, qs: np.ndarray) -> list[float]:
    arr = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return []
    return sorted(set(float(x) for x in np.quantile(arr, qs)))


def choose_rule(val: pd.DataFrame) -> tuple[dict, dict[str, float | int]]:
    base_rule = {"type": "none"}
    base_frame = apply_rule(val, base_rule)
    base_metrics = metric_row(base_frame)
    best = (score_metrics(base_metrics, base_metrics), base_rule, base_metrics)
    base_pos = val[val["base_pred"].astype(int) == 1].copy()
    if len(base_pos) == 0:
        return best[1], best[2]

    margin_grid = [
        x
        for x in sorted(set(thresholds(base_pos["normal_margin"], np.linspace(0.30, 0.97, 14)) + [0.0, 0.5, 1.0, 1.5]))
        if x >= 0.0
    ]
    ratio_grid = [
        x
        for x in sorted(set(thresholds(base_pos["normal_ratio"], np.linspace(0.05, 0.90, 14)) + [0.70, 0.85, 1.00]))
        if x <= 1.0
    ]
    min_alarm_grid = [50.0, 75.0, 100.0, 150.0]

    for margin in margin_grid:
        for min_alarm in min_alarm_grid:
            rule = {"type": "margin", "margin_threshold": margin, "min_alarm_s": min_alarm}
            cand = apply_rule(val, rule)
            metrics = metric_row(cand)
            current = score_metrics(metrics, base_metrics)
            if current > best[0]:
                best = (current, rule, metrics)

    for ratio in ratio_grid:
        for min_alarm in min_alarm_grid:
            rule = {"type": "ratio", "ratio_threshold": ratio, "min_alarm_s": min_alarm}
            cand = apply_rule(val, rule)
            metrics = metric_row(cand)
            current = score_metrics(metrics, base_metrics)
            if current > best[0]:
                best = (current, rule, metrics)

    for margin in margin_grid:
        for ratio in ratio_grid:
            for min_alarm in min_alarm_grid:
                rule = {
                    "type": "hybrid",
                    "margin_threshold": margin,
                    "ratio_threshold": ratio,
                    "min_alarm_s": min_alarm,
                }
                cand = apply_rule(val, rule)
                metrics = metric_row(cand)
                current = score_metrics(metrics, base_metrics)
                if current > best[0]:
                    best = (current, rule, metrics)
    return best[1], best[2]


def summarize(rows: list[dict], model: str) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": model, "n_seeds": int(data["seed"].nunique())}
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
        "vetoed",
    ]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    all_pred = reconstruct_selector_candidates()
    choices = pd.read_csv(OUT / "validation_model_selector_choices_conservative_margin005.csv")
    prefix_long, feature_cols = load_prefix_feature_long()
    target_meta = target_meta_from_features(prefix_long)
    source_ids = set(prefix_long.loc[prefix_long["domain"].astype(str) == "source5", "sample_id"].unique())

    metric_rows = []
    base_metric_rows = []
    pred_rows = []
    choice_rows = []
    for seed in sorted(choices["seed"].astype(int).unique()):
        chosen_model = str(choices.loc[choices["seed"].astype(int) == int(seed), "chosen_model"].iloc[0])
        split = stratified_target_split(target_meta, int(seed))
        train_ids = set(split["train"]) | source_ids
        train_rows = feature_rows_for_ids(prefix_long, train_ids)
        neighbors = fit_neighbors(train_rows, feature_cols)
        val = chosen_selector_frame(all_pred, int(seed), "val", chosen_model)
        test = chosen_selector_frame(all_pred, int(seed), "test", chosen_model)
        val = attach_pointset_scores(val, prefix_long, neighbors)
        test = attach_pointset_scores(test, prefix_long, neighbors)

        base_test = apply_rule(test, {"type": "none"})
        base_metrics = metric_row(base_test)
        base_metrics.update({"seed": int(seed), "vetoed": 0})
        base_metric_rows.append(base_metrics)

        rule, val_metrics = choose_rule(val)
        chosen = apply_rule(test, rule)
        metrics = metric_row(chosen)
        metrics.update({"seed": int(seed), "vetoed": int(chosen["vetoed"].sum())})
        metric_rows.append(metrics)
        choice_rows.append({"seed": int(seed), "chosen_model": chosen_model, **rule, **{f"val_{k}": v for k, v in val_metrics.items()}})

        pred = chosen[
            [
                "sample_id",
                "file_name",
                "binary",
                "hard_negative",
                "severity_name",
                "onset_s",
                "seed",
                "split",
                "model",
                "base_pred",
                "base_alarm_time_s",
                "d_target_normal",
                "d_target_fault",
                "d_all_fault",
                "normal_margin",
                "normal_ratio",
                "y_pred",
                "alarm_time_s",
                "vetoed",
            ]
        ].copy()
        pred["pointset_model"] = MODEL_NAME
        pred_rows.append(pred)
        print(
            f"seed {seed}: base_acc={base_metrics['accuracy']:.4f}, "
            f"new_acc={metrics['accuracy']:.4f}, spec={metrics['specificity']:.3f}, "
            f"fp={metrics['fp']}, fn={metrics['fn']}, rule={rule['type']}",
            flush=True,
        )

    detail = pd.DataFrame(metric_rows)
    base_detail = pd.DataFrame(base_metric_rows)
    detail.to_csv(out_path("pointset_veto_metrics"), index=False, encoding="utf-8-sig")
    base_detail.to_csv(out_path("pointset_veto_base_metrics"), index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(out_path("pointset_veto_choices"), index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(out_path("pointset_veto_predictions"), index=False, encoding="utf-8-sig")
    summary = pd.concat(
        [
            summarize(base_metric_rows, "ValidationSelected_Base"),
            summarize(metric_rows, MODEL_NAME),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_path("pointset_veto_summary"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))

    selector_summary = selector_metric_row(pd.concat(pred_rows, ignore_index=True).rename(columns={"binary": "y_true"}))
    print(f"selector_metric_row_crosscheck={selector_summary}")


if __name__ == "__main__":
    main()

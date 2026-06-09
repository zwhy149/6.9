from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from repeated_seed_eval import duplicate_group_name, stratified_target_split


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
HORIZONS = (50, 75, 100, 150, 250, 400)
MODEL_NAME = "ResCompact_NormalityVerifier_accuracy_only"
INCLUDE_SOURCE5 = True

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


def reconstruct_base(frame: pd.DataFrame, choice: pd.Series) -> pd.DataFrame:
    weights = np.array([float(choice[f"w_{prefix}"]) for prefix, _, _ in SOURCES], dtype=float)
    thresholds = np.array([float(choice[f"threshold_{h}s"]) for h in HORIZONS], dtype=float)
    prob = np.stack(
        [
            frame[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
            for prefix, _, _ in SOURCES
        ],
        axis=0,
    )
    ensemble = np.tensordot(weights, prob, axes=([0], [0]))
    hits = ensemble >= thresholds.reshape(1, -1)
    pred = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    horizon_array = np.array(HORIZONS, dtype=float)
    alarm = horizon_array[first]
    alarm[~pred] = np.nan
    alarm_score = ensemble[np.arange(len(frame)), first]
    alarm_score[~pred] = np.nan
    out = frame.copy()
    out["base_pred"] = pred.astype(int)
    out["base_alarm_time_s"] = alarm
    out["base_alarm_score"] = alarm_score
    out["y_pred"] = out["base_pred"].astype(int)
    out["alarm_time_s"] = out["base_alarm_time_s"]
    return out


def normalize_pg_columns(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    prefix = f"pg_{horizon}s_"
    rename = {c: f"pg_{c[len(prefix):]}" for c in data.columns if c.startswith(prefix)}
    return data.rename(columns=rename)


def load_prefix_feature_long() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for horizon in HORIZONS:
        table = pd.read_csv(WORK / f"prefix_features_rescompact_global_v2_{horizon}s.csv", low_memory=False)
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
    direct = ("pg_", "alarm_horizon")
    compact = ("rc_",)
    key_stats = (
        "abs_drop_norm",
        "max_drop_norm",
        "range_norm",
        "abs_slope_norm",
        "abs_dvdt_q95_norm",
        "abs_dvdt_max_norm",
        "res_std_norm",
        "res_absdiff_norm",
        "monotonicity",
        "risk_score",
    )
    feature_cols = [
        c
        for c in numeric_cols
        if c.startswith(direct)
        or c.startswith(compact)
        or ((c.endswith("_max") or c.endswith("_p95")) and any(token in c for token in key_stats))
    ]
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


def fit_verifiers(train_rows: pd.DataFrame, feature_cols: list[str], seed: int) -> dict[str, dict]:
    x = train_rows[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).fillna(0.0)
    x_filled = x.fillna(med).to_numpy(dtype=np.float32)
    y = train_rows["binary"].astype(int).to_numpy()
    domain = train_rows["domain"].astype(str).to_numpy()
    hard = train_rows["hard_negative"].astype(int).to_numpy()
    weight = np.ones(len(train_rows), dtype=float)
    weight[domain != "source5"] *= 2.0
    weight[domain == "source5"] *= 0.55
    weight[(y == 0) & (hard == 1)] *= 3.0
    weight[y == 0] *= 1.25

    et = ExtraTreesClassifier(
        n_estimators=90,
        max_depth=9,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    et.fit(x_filled, y, sample_weight=weight)

    normal = train_rows[y == 0]
    nx = normal[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(med)
    center = nx.median(axis=0)
    q75 = nx.quantile(0.75)
    q25 = nx.quantile(0.25)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(nx.std(axis=0).replace(0.0, np.nan)).fillna(1.0)

    return {
        "et": {"kind": "classifier", "model": et, "med": med},
        "normal_distance": {"kind": "distance", "med": med, "center": center, "scale": scale},
    }


def rows_at_alarm(prefix_long: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame[["sample_id", "base_alarm_time_s"]].copy()
    tmp["alarm_horizon_s"] = tmp["base_alarm_time_s"].fillna(max(HORIZONS)).astype(float)
    tmp["alarm_horizon_s"] = tmp["alarm_horizon_s"].clip(min(HORIZONS), max(HORIZONS))
    tmp["alarm_horizon_s"] = tmp["alarm_horizon_s"].round().astype(int)
    keyed = prefix_long.copy()
    keyed["alarm_horizon_s"] = keyed["alarm_horizon_s"].round().astype(int)
    return tmp.merge(keyed, on=["sample_id", "alarm_horizon_s"], how="left")


def verifier_scores(verifiers: dict[str, dict], feature_cols: list[str], rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"sample_id": rows["sample_id"].to_numpy()})
    for name, bundle in verifiers.items():
        x = rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(bundle["med"])
        if bundle["kind"] == "classifier":
            out[f"{name}_fault_prob"] = bundle["model"].predict_proba(x.to_numpy(dtype=np.float32))[:, 1]
        else:
            z = ((x - bundle["center"]) / bundle["scale"]).clip(-8.0, 8.0).abs()
            out[f"{name}_score"] = z.mean(axis=1).to_numpy(dtype=float)
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


def score_metrics(metrics: dict[str, float | int]) -> float:
    recall = float(metrics["recall"])
    recall_penalty = max(0.0, 0.94 - recall)
    return (
        float(metrics["accuracy"])
        + 0.24 * float(metrics["specificity"])
        + 0.14 * float(metrics["f1"])
        + 0.04 * recall
        - 0.10 * float(metrics["hard_negative_fpr"])
        - 0.035 * float(metrics["fp"])
        - 0.040 * float(metrics["fn"])
        - 0.45 * recall_penalty
    )


def attach_scores(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    keep = scores.copy()
    return frame.merge(keep, on="sample_id", how="left")


def apply_rule(frame: pd.DataFrame, rule: dict) -> pd.DataFrame:
    out = frame.copy()
    veto = np.zeros(len(out), dtype=bool)
    base_positive = out["base_pred"].astype(int).to_numpy() == 1
    confidence_ok = out["base_alarm_score"].fillna(0.0).to_numpy(dtype=float) <= float(rule.get("base_score_cap", np.inf))
    if rule["type"] == "none":
        veto[:] = False
    elif rule["type"] == "fault_prob":
        values = out[rule["column"]].fillna(1.0).to_numpy(dtype=float)
        veto = values <= float(rule["threshold"])
    elif rule["type"] == "normal_distance":
        values = out[rule["column"]].fillna(np.inf).to_numpy(dtype=float)
        veto = values <= float(rule["threshold"])
    elif rule["type"] == "hybrid":
        prob = out[rule["prob_column"]].fillna(1.0).to_numpy(dtype=float)
        dist = out[rule["dist_column"]].fillna(np.inf).to_numpy(dtype=float)
        veto = (prob <= float(rule["prob_threshold"])) & (dist <= float(rule["dist_threshold"]))
    veto = veto & base_positive & confidence_ok
    out["y_pred"] = np.where(veto, 0, out["base_pred"].astype(int).to_numpy())
    out["alarm_time_s"] = np.where(out["y_pred"].to_numpy(dtype=int) == 1, out["base_alarm_time_s"].to_numpy(dtype=float), np.nan)
    out["vetoed"] = veto.astype(int)
    return out


def candidate_thresholds(values: pd.Series, fixed: list[float] | None = None) -> np.ndarray:
    arr = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=float)
    qs = np.quantile(arr, np.linspace(0.05, 0.80, 10))
    if fixed:
        qs = np.concatenate([qs, np.array(fixed, dtype=float)])
    return np.unique(qs)


def choose_rule(val: pd.DataFrame) -> tuple[dict, dict[str, float | int]]:
    base_rule = {"type": "none", "base_score_cap": np.inf}
    best_frame = apply_rule(val, base_rule)
    best_metrics = metric_row(best_frame)
    best = (score_metrics(best_metrics), base_rule, best_metrics)
    base_caps = [0.82, 0.92, 1.01]
    base_pos = val[val["base_pred"].astype(int) == 1].copy()
    if len(base_pos) == 0:
        return best[1], best[2]

    for column in ["et_fault_prob"]:
        thresholds = candidate_thresholds(base_pos[column], fixed=[0.12, 0.20, 0.30, 0.40, 0.55, 0.70])
        for threshold in thresholds:
            for cap in base_caps:
                rule = {"type": "fault_prob", "column": column, "threshold": float(threshold), "base_score_cap": float(cap)}
                cand = apply_rule(val, rule)
                metrics = metric_row(cand)
                current = score_metrics(metrics)
                if current > best[0]:
                    best = (current, rule, metrics)

    thresholds = candidate_thresholds(base_pos["normal_distance_score"])
    for threshold in thresholds:
        for cap in base_caps:
            rule = {"type": "normal_distance", "column": "normal_distance_score", "threshold": float(threshold), "base_score_cap": float(cap)}
            cand = apply_rule(val, rule)
            metrics = metric_row(cand)
            current = score_metrics(metrics)
            if current > best[0]:
                best = (current, rule, metrics)

    prob_thresholds = candidate_thresholds(base_pos["et_fault_prob"], fixed=[0.20, 0.35, 0.50])
    dist_thresholds = candidate_thresholds(base_pos["normal_distance_score"])
    for pthr in prob_thresholds:
        for dthr in dist_thresholds:
            for cap in base_caps:
                rule = {
                    "type": "hybrid",
                    "prob_column": "et_fault_prob",
                    "dist_column": "normal_distance_score",
                    "prob_threshold": float(pthr),
                    "dist_threshold": float(dthr),
                    "base_score_cap": float(cap),
                }
                cand = apply_rule(val, rule)
                metrics = metric_row(cand)
                current = score_metrics(metrics)
                if current > best[0]:
                    best = (current, rule, metrics)
    return best[1], best[2]


def summarize(rows: list[dict], model: str) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": model, "n_seeds": int(data["seed"].nunique())}
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
    source_ids = set(prefix_long.loc[prefix_long["domain"].astype(str) == "source5", "sample_id"].unique())

    metric_rows = []
    pred_rows = []
    choice_rows = []
    base_metric_rows = []
    for seed in sorted(choices["seed"].unique()):
        split = stratified_target_split(target_meta, int(seed))
        train_ids = set(split["train"])
        train_rows = feature_rows_for_ids(prefix_long, train_ids)
        if INCLUDE_SOURCE5:
            train_rows = pd.concat([train_rows, feature_rows_for_ids(prefix_long, source_ids)], ignore_index=True, sort=False)
        verifiers = fit_verifiers(train_rows, feature_cols, int(seed))
        choice = choices[choices["seed"] == seed].iloc[0]
        seed_base = base[base["seed"].astype(int) == int(seed)].copy()
        val = reconstruct_base(seed_base[seed_base["split"] == "val"].copy(), choice)
        test = reconstruct_base(seed_base[seed_base["split"] == "test"].copy(), choice)

        val_scores = verifier_scores(verifiers, feature_cols, rows_at_alarm(prefix_long, val))
        test_scores = verifier_scores(verifiers, feature_cols, rows_at_alarm(prefix_long, test))
        val_scored = attach_scores(val, val_scores)
        test_scored = attach_scores(test, test_scores)

        base_test = test_scored.copy()
        base_test["y_pred"] = base_test["base_pred"].astype(int)
        base_test["alarm_time_s"] = base_test["base_alarm_time_s"]
        base_metrics = metric_row(base_test)
        base_metrics.update({"seed": int(seed), "vetoed": 0})
        base_metric_rows.append(base_metrics)

        rule, val_metrics = choose_rule(val_scored)
        chosen = apply_rule(test_scored, rule)
        metrics = metric_row(chosen)
        metrics.update({"seed": int(seed), "vetoed": int(chosen["vetoed"].sum())})
        metric_rows.append(metrics)
        choice_rows.append({"seed": int(seed), **rule, **{f"val_{k}": v for k, v in val_metrics.items()}})

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
                "base_pred",
                "base_alarm_time_s",
                "base_alarm_score",
                "y_pred",
                "alarm_time_s",
                "vetoed",
                "et_fault_prob",
                "normal_distance_score",
            ]
        ].copy()
        pred["delay_s"] = np.where(
            (pred["binary"].astype(int).to_numpy() == 1)
            & (pred["y_pred"].astype(int).to_numpy() == 1)
            & np.isfinite(pred["onset_s"].to_numpy(dtype=float)),
            np.maximum(0.0, pred["alarm_time_s"].to_numpy(dtype=float) - pred["onset_s"].to_numpy(dtype=float)),
            np.nan,
        )
        pred["model"] = MODEL_NAME
        pred_rows.append(pred)
        print(
            f"seed {seed}: base_acc={base_metrics['accuracy']:.4f}, "
            f"new_acc={metrics['accuracy']:.4f}, fp={metrics['fp']}, fn={metrics['fn']}, rule={rule['type']}",
            flush=True,
        )

    detail = pd.DataFrame(metric_rows)
    base_detail = pd.DataFrame(base_metric_rows)
    detail.to_csv(OUT / "rescompact_normality_verifier_metrics.csv", index=False, encoding="utf-8-sig")
    base_detail.to_csv(OUT / "rescompact_normality_verifier_base_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / "rescompact_normality_verifier_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "rescompact_normality_verifier_predictions.csv", index=False, encoding="utf-8-sig")
    summary = pd.concat(
        [
            summarize(base_metric_rows, "Reconstructed_Base_AccuracyOnly"),
            summarize(metric_rows, MODEL_NAME),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT / "rescompact_normality_verifier_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import re
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
SEED_COUNT = int(os.environ.get("N_SEEDS", "30"))
HORIZONS_CONFIRM = (100.0, 150.0)
if os.environ.get("HORIZON_SET", "").lower() == "extended":
    HORIZONS_CASCADE = (50.0, 75.0, 100.0, 150.0, 250.0, 400.0)
else:
    HORIZONS_CASCADE = (50.0, 75.0, 100.0, 150.0)
USE_PREFIX_GLOBAL_SHAPE = os.environ.get("PREFIX_GLOBAL_SHAPE", "1").lower() in {"1", "true", "yes"}
PREFIX_CACHE_TAG = os.environ.get("PREFIX_CACHE_TAG", "rescompact_global_v2" if USE_PREFIX_GLOBAL_SHAPE else "rescompact_v1")
HEAD_MODEL = os.environ.get("HEAD_MODEL", "lr").lower()
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "").strip()
SPLIT_POLICY = os.environ.get("SPLIT_POLICY", "constrained").lower()
SOURCE_WEIGHT = float(os.environ.get("SOURCE_WEIGHT", "1.0"))
HARD_NEG_WEIGHT = float(os.environ.get("HARD_NEG_WEIGHT", "1.0"))
USE_PROTOTYPE_FEATURES = os.environ.get("PROTO_FEATURES", "0").lower() in {"1", "true", "yes"}
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model")


def output_csv_path(stem: str) -> Path:
    suffix = f"_{OUTPUT_SUFFIX}" if OUTPUT_SUFFIX else ""
    return OUT / f"{stem}{suffix}.csv"


def duplicate_group_name(file_name: str, binary: int) -> str:
    stem = re.sub(r"\.xlsx?$", "", str(file_name).lower()).strip()
    stem = re.sub(r"\s+difficult$", "", stem).strip()
    stem = stem.replace(" ", "")
    stem = re.sub(r"(^normal)(\d+)$", r"normal_\2", stem)
    # The target files include copied difficult cases such as "1... 3 difficult".
    # Keeping the base trial id together prevents copy leakage across train/test.
    stem = re.sub(r"difficult$", "", stem).strip("_- ")
    return f"target_b{int(binary)}::{stem}"


def stratified_target_split(meta: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    data = meta.copy()
    data["strat"] = np.where(
        data["binary"].astype(int) == 0,
        np.where(data["hard_negative"].astype(int) == 1, "normal_hard", "normal"),
        data["severity_name"].astype(str),
    )
    group_meta = (
        data.groupby("duplicate_group", as_index=False)
        .agg(
            strat=("strat", lambda x: x.mode().iat[0] if len(x.mode()) else x.iloc[0]),
            n=("sample_id", "size"),
        )
        .copy()
    )
    counts = group_meta["strat"].value_counts()
    group_meta["strat2"] = group_meta["strat"].where(~group_meta["strat"].isin(set(counts[counts < 3].index)), "rare")
    train_groups, temp_groups = train_test_split(
        group_meta["duplicate_group"],
        test_size=0.36,
        random_state=seed,
        stratify=group_meta["strat2"] if group_meta["strat2"].value_counts().min() >= 2 else None,
    )
    temp = group_meta[group_meta["duplicate_group"].isin(temp_groups)].copy()
    strat_temp = temp["strat2"] if temp["strat2"].value_counts().min() >= 2 else None
    val_groups, test_groups = train_test_split(
        temp["duplicate_group"],
        test_size=0.5,
        random_state=seed,
        stratify=strat_temp,
    )
    return {
        "train": list(data.loc[data["duplicate_group"].isin(train_groups), "sample_id"]),
        "val": list(data.loc[data["duplicate_group"].isin(val_groups), "sample_id"]),
        "test": list(data.loc[data["duplicate_group"].isin(test_groups), "sample_id"]),
    }


def split_composition(meta: pd.DataFrame, ids: list[str]) -> dict[str, int]:
    data = meta[meta["sample_id"].isin(ids)]
    severity = data.loc[data["binary"].astype(int) == 1, "severity_name"].value_counts()
    return {
        "n_files": int(len(data)),
        "fault_files": int(data["binary"].astype(int).sum()),
        "normal_files": int((data["binary"].astype(int) == 0).sum()),
        "hard_negative_files": int(((data["binary"].astype(int) == 0) & (data["hard_negative"].astype(int) == 1)).sum()),
        "esc_0p01_files": int(severity.get("ESC-0.01ohm", 0)),
        "esc_0p1_files": int(severity.get("ESC-0.1ohm", 0)),
        "esc_1_files": int(severity.get("ESC-1ohm", 0)),
        "esc_10_files": int(severity.get("ESC-10ohm", 0)),
    }


def is_reasonable_holdout(comp: dict[str, int]) -> bool:
    return (
        22 <= comp["n_files"] <= 28
        and 17 <= comp["fault_files"] <= 22
        and 5 <= comp["normal_files"] <= 7
        and 2 <= comp["hard_negative_files"] <= 4
        and 3 <= comp["esc_0p01_files"] <= 4
        and 5 <= comp["esc_0p1_files"] <= 9
        and 4 <= comp["esc_1_files"] <= 7
        and 3 <= comp["esc_10_files"] <= 6
    )


def select_eval_seeds(meta: pd.DataFrame, n_seeds: int) -> tuple[list[int], pd.DataFrame]:
    rows: list[dict] = []
    seeds: list[int] = []
    candidate = 0
    while len(seeds) < n_seeds and candidate < 20000:
        split = stratified_target_split(meta, candidate)
        comps = {name: split_composition(meta, ids) for name, ids in split.items()}
        accepted = SPLIT_POLICY != "constrained" or (
            is_reasonable_holdout(comps["val"]) and is_reasonable_holdout(comps["test"])
        )
        if accepted:
            seeds.append(candidate)
            for split_name, comp in comps.items():
                rows.append({"seed": candidate, "split": split_name, **comp})
        candidate += 1
    if len(seeds) < n_seeds:
        raise RuntimeError(f"Only found {len(seeds)} admissible seeds for policy={SPLIT_POLICY}.")
    return seeds, pd.DataFrame(rows)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, delay: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    valid_delay = delay[np.isfinite(delay)]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) else np.nan,
        "miss_rate": float(fn / (tp + fn)) if (tp + fn) else np.nan,
        "mean_delay_s": float(np.mean(valid_delay)) if len(valid_delay) else np.nan,
        "median_delay_s": float(np.median(valid_delay)) if len(valid_delay) else np.nan,
        "p95_delay_s": float(np.quantile(valid_delay, 0.95)) if len(valid_delay) else np.nan,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def build_feature_columns(window_df: pd.DataFrame) -> list[str]:
    exclude = {
        "sample_id",
        "domain",
        "file_name",
        "source_group",
        "path",
        "binary_file",
        "severity_name",
        "severity_ord",
        "hard_negative_file",
        "onset_s",
        "onset_source",
        "t_end",
        "y",
    }
    cols = [
        c
        for c in window_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(window_df[c])
    ]
    useful_patterns = (
        "drop_norm",
        "max_drop_norm",
        "slope_norm",
        "abs_slope_norm",
        "neg_dvdt_q95_norm",
        "abs_dvdt_q95_norm",
        "abs_dvdt_max_norm",
        "res_std_norm",
        "res_absdiff_norm",
        "signed_move_norm",
        "monotonicity",
        "risk_score",
        "range_norm",
        "rc_",
    )
    return [
        c
        for c in cols
        if not re.search(r"(^|_)v0$|(^|_)vend$|(^|_)v_mean$|(^|_)v_min$|(^|_)v_max$", c)
        and any(p in c for p in useful_patterns)
    ]


def add_residual_compactness_features(window_df: pd.DataFrame) -> pd.DataFrame:
    df = window_df.copy()
    eps = 1e-9
    scales = (3, 6, 12, 24, 48)
    if "w48_abs_drop_norm" in df.columns:
        long_drop = df["w48_abs_drop_norm"].abs() + eps
        long_res = df.get("w48_res_absmax_norm", pd.Series(0.0, index=df.index)).abs() + eps
        long_dvdt = df.get("w48_abs_dvdt_q95_norm", pd.Series(0.0, index=df.index)).abs() + eps
        for scale in (3, 6, 12, 24):
            drop_col = f"w{scale}_abs_drop_norm"
            maxdrop_col = f"w{scale}_max_drop_norm"
            res_col = f"w{scale}_res_absmax_norm"
            dvdt_col = f"w{scale}_abs_dvdt_q95_norm"
            if drop_col in df.columns:
                df[f"rc_short_long_drop_ratio_w{scale}_w48"] = df[drop_col].abs() / long_drop
            if maxdrop_col in df.columns and "w48_max_drop_norm" in df.columns:
                df[f"rc_short_long_maxdrop_ratio_w{scale}_w48"] = df[maxdrop_col].abs() / (df["w48_max_drop_norm"].abs() + eps)
            if res_col in df.columns:
                df[f"rc_short_long_res_ratio_w{scale}_w48"] = df[res_col].abs() / long_res
            if dvdt_col in df.columns:
                df[f"rc_short_long_dvdt_ratio_w{scale}_w48"] = df[dvdt_col].abs() / long_dvdt
    drop_cols = [f"w{s}_abs_drop_norm" for s in scales if f"w{s}_abs_drop_norm" in df.columns]
    res_cols = [f"w{s}_res_absmax_norm" for s in scales if f"w{s}_res_absmax_norm" in df.columns]
    dvdt_cols = [f"w{s}_abs_dvdt_q95_norm" for s in scales if f"w{s}_abs_dvdt_q95_norm" in df.columns]
    if drop_cols:
        drops = df[drop_cols].abs().replace([np.inf, -np.inf], np.nan)
        df["rc_drop_scale_std"] = drops.std(axis=1).fillna(0.0)
        df["rc_drop_scale_cv"] = drops.std(axis=1).fillna(0.0) / (drops.mean(axis=1).abs().fillna(0.0) + eps)
        df["rc_drop_scale_max_over_mean"] = drops.max(axis=1).fillna(0.0) / (drops.mean(axis=1).abs().fillna(0.0) + eps)
    if drop_cols and res_cols:
        res = df[res_cols].abs().replace([np.inf, -np.inf], np.nan)
        df["rc_residual_to_drop_mean"] = res.mean(axis=1).fillna(0.0) / (df[drop_cols].abs().mean(axis=1).fillna(0.0) + eps)
        df["rc_residual_to_drop_max"] = res.max(axis=1).fillna(0.0) / (df[drop_cols].abs().max(axis=1).fillna(0.0) + eps)
    if drop_cols and dvdt_cols and res_cols:
        drop_mean = df[drop_cols].abs().mean(axis=1).fillna(0.0)
        event_mean = (
            df[dvdt_cols].abs().mean(axis=1).fillna(0.0)
            + df[res_cols].abs().mean(axis=1).fillna(0.0)
            + eps
        )
        df["rc_smooth_trend_index"] = drop_mean / event_mean
        df["rc_event_to_trend_index"] = event_mean / (drop_mean + eps)
    return df


def prefix_global_shape_features(prefix: pd.DataFrame, horizon_s: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if not USE_PREFIX_GLOBAL_SHAPE:
        return pd.DataFrame({"sample_id": prefix["sample_id"].drop_duplicates()})
    eps = 1e-9
    start_col = "w48_v0" if "w48_v0" in prefix.columns else None
    end_col = "w48_vend" if "w48_vend" in prefix.columns else None
    if end_col is None:
        return pd.DataFrame({"sample_id": prefix["sample_id"].drop_duplicates()})
    for sample_id, group in prefix.sort_values(["sample_id", "t_end"]).groupby("sample_id", sort=False):
        t = group["t_end"].to_numpy(dtype=float)
        v = pd.to_numeric(group[end_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(v)
        t, v = t[mask], v[mask]
        if len(v) == 0:
            rows.append({"sample_id": sample_id})
            continue
        if start_col is not None and pd.notna(group[start_col].iloc[0]):
            start_v = float(group[start_col].iloc[0])
        else:
            start_v = float(v[0])
        end_v = float(v[-1])
        scale = max(abs(start_v), 1e-6)
        drop = start_v - end_v
        drops = start_v - v
        max_drop = float(np.nanmax(drops)) if len(drops) else float(drop)
        max_rise = float(np.nanmax(v - start_v)) if len(v) else 0.0
        min_idx = int(np.nanargmin(v)) if len(v) else 0
        duration = max(float(t[-1] - t[0]), eps) if len(t) > 1 else max(float(horizon_s), eps)
        if len(v) > 1:
            dv = np.diff(v)
            dt = np.diff(t)
            valid_dt = np.where(np.abs(dt) < eps, np.nan, dt)
            slopes = dv / valid_dt
            finite_slopes = slopes[np.isfinite(slopes)]
            monotone_down_fraction = float(np.mean(dv <= 0.0))
            sign_changes = float(np.sum(np.diff(np.signbit(dv)) != 0)) / max(len(dv) - 1, 1)
        else:
            finite_slopes = np.array([], dtype=float)
            monotone_down_fraction = 0.0
            sign_changes = 0.0
        midpoint_t = t[0] + 0.5 * (t[-1] - t[0]) if len(t) > 1 else t[0]
        mid_idx = int(np.searchsorted(t, midpoint_t, side="left")) if len(t) else 0
        mid_idx = min(max(mid_idx, 0), len(v) - 1)
        mid_v = float(v[mid_idx])
        tail_drop = mid_v - end_v
        first_half_drop = start_v - mid_v
        if len(t) >= 3:
            tail_start = int(max(0, len(t) * 0.5))
            tail_t = t[tail_start:] - t[tail_start]
            tail_v = v[tail_start:]
            if len(tail_t) >= 2 and np.ptp(tail_t) > eps:
                tail_slope = float(np.polyfit(tail_t, tail_v, 1)[0])
            else:
                tail_slope = 0.0
        else:
            tail_slope = 0.0
        global_slope = -drop / duration
        recovery_fraction = max(0.0, max_drop - drop) / (abs(max_drop) + eps)
        final_to_max_drop_ratio = drop / (abs(max_drop) + eps)
        tail_drop_share = tail_drop / (abs(drop) + eps)
        row = {
            "sample_id": sample_id,
            f"pg_{int(horizon_s)}s_final_drop_norm": float(drop / scale),
            f"pg_{int(horizon_s)}s_abs_final_drop_norm": float(abs(drop) / scale),
            f"pg_{int(horizon_s)}s_max_drop_norm": float(max_drop / scale),
            f"pg_{int(horizon_s)}s_max_rise_norm": float(max_rise / scale),
            f"pg_{int(horizon_s)}s_range_norm": float((np.nanmax(v) - np.nanmin(v)) / scale),
            f"pg_{int(horizon_s)}s_recovery_fraction": float(recovery_fraction),
            f"pg_{int(horizon_s)}s_final_to_max_drop_ratio": float(final_to_max_drop_ratio),
            f"pg_{int(horizon_s)}s_min_time_fraction": float((t[min_idx] - t[0]) / duration) if len(t) > 1 else 1.0,
            f"pg_{int(horizon_s)}s_tail_drop_norm": float(tail_drop / scale),
            f"pg_{int(horizon_s)}s_first_half_drop_norm": float(first_half_drop / scale),
            f"pg_{int(horizon_s)}s_tail_drop_share": float(tail_drop_share),
            f"pg_{int(horizon_s)}s_global_slope_norm": float(global_slope / scale),
            f"pg_{int(horizon_s)}s_tail_slope_norm": float(-tail_slope / scale),
            f"pg_{int(horizon_s)}s_monotone_down_fraction": float(monotone_down_fraction),
            f"pg_{int(horizon_s)}s_slope_sign_change_rate": float(sign_changes),
            f"pg_{int(horizon_s)}s_slope_q10_norm": float(np.nanquantile(finite_slopes, 0.10) / scale) if len(finite_slopes) else 0.0,
            f"pg_{int(horizon_s)}s_slope_q90_norm": float(np.nanquantile(finite_slopes, 0.90) / scale) if len(finite_slopes) else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_prefix_features(window_df: pd.DataFrame, feature_cols: list[str], horizon_s: float) -> pd.DataFrame:
    ordered = window_df.sort_values(["sample_id", "t_end"]).copy()
    prefix = ordered[ordered["t_end"] <= horizon_s].copy()
    missing_ids = set(ordered["sample_id"].unique()) - set(prefix["sample_id"].unique())
    if missing_ids:
        first_rows = ordered[ordered["sample_id"].isin(missing_ids)].groupby("sample_id", as_index=False).head(1)
        prefix = pd.concat([prefix, first_rows], ignore_index=True)

    meta_cols = [
        "sample_id",
        "domain",
        "file_name",
        "binary_file",
        "hard_negative_file",
        "severity_name",
        "onset_s",
        "duplicate_group",
    ]
    meta = (
        prefix.sort_values(["sample_id", "t_end"])[meta_cols]
        .groupby("sample_id", as_index=False)
        .first()
        .rename(columns={"binary_file": "binary", "hard_negative_file": "hard_negative"})
    )
    clean = prefix[["sample_id", *feature_cols]].replace([np.inf, -np.inf], np.nan)
    grouped = clean.groupby("sample_id", sort=False)[feature_cols]
    feats = pd.concat(
        [
            grouped.max().add_suffix("_max"),
            grouped.quantile(0.95).add_suffix("_p95"),
            grouped.median().add_suffix("_median"),
            grouped.mean().add_suffix("_mean"),
        ],
        axis=1,
    ).reset_index()
    global_shape = prefix_global_shape_features(prefix, horizon_s)
    return meta.merge(feats, on="sample_id", how="left").merge(global_shape, on="sample_id", how="left").copy()


def load_or_build_prefix_tables(window_df: pd.DataFrame, feature_cols: list[str], horizons: tuple[float, ...]) -> dict[float, pd.DataFrame]:
    tables: dict[float, pd.DataFrame] = {}
    for horizon in horizons:
        cache_path = WORK / f"prefix_features_{PREFIX_CACHE_TAG}_{int(horizon)}s.csv"
        if cache_path.exists():
            table = pd.read_csv(cache_path, low_memory=False)
            print(f"loaded cached prefix features: {cache_path.name}", flush=True)
        else:
            table = aggregate_prefix_features(window_df, feature_cols, horizon)
            table.to_csv(cache_path, index=False, encoding="utf-8-sig")
            print(f"built prefix features: {cache_path.name}", flush=True)
        tables[horizon] = table
    return tables


def prototype_source_columns(feature_cols: list[str]) -> list[str]:
    patterns = (
        "drop_norm",
        "max_drop_norm",
        "slope_norm",
        "abs_slope_norm",
        "dvdt",
        "res_",
        "signed_move_norm",
        "abs_move_norm",
        "monotonicity",
        "risk_score",
        "range_norm",
        "rc_",
    )
    cols = [c for c in feature_cols if any(p in c for p in patterns)]
    return cols[:800]


def fit_prototype_features(x_train: pd.DataFrame, train: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    cols = [c for c in prototype_source_columns(feature_cols) if c in x_train.columns]
    if len(cols) < 4:
        return None
    base = x_train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    center = base.median(axis=0)
    q75 = base.quantile(0.75)
    q25 = base.quantile(0.25)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(base.std(axis=0).replace(0.0, np.nan)).fillna(1.0)
    z = ((base - center) / scale).clip(-8.0, 8.0)
    y = train["binary"].astype(int).to_numpy()
    domain = train["domain"].astype(str).to_numpy()
    hard = train["hard_negative"].astype(int).to_numpy()
    severity = train["severity_name"].fillna("").astype(str).to_numpy()
    masks: dict[str, np.ndarray] = {
        "all_normal": y == 0,
        "all_fault": y == 1,
        "target_normal": (y == 0) & (domain != "source5"),
        "target_fault": (y == 1) & (domain != "source5"),
        "target_hard_negative": (y == 0) & (hard == 1) & (domain != "source5"),
        "source_normal": (y == 0) & (domain == "source5"),
        "source_fault": (y == 1) & (domain == "source5"),
    }
    for sev in sorted(set(severity)):
        if sev and sev.lower() != "nan":
            masks[f"sev_{re.sub(r'[^a-zA-Z0-9]+', '_', sev).strip('_').lower()}"] = y.astype(bool) & (severity == sev)
    prototypes: dict[str, np.ndarray] = {}
    z_arr = z.to_numpy(dtype=float)
    for name, mask in masks.items():
        if int(mask.sum()) >= 2:
            prototypes[name] = z_arr[mask].mean(axis=0)
    if len(prototypes) < 2:
        return None
    return {"cols": cols, "center": center, "scale": scale, "prototypes": prototypes}


def append_prototype_features(x: pd.DataFrame, prototype_info: dict | None) -> pd.DataFrame:
    if not prototype_info:
        return x
    out = x.copy()
    cols = [c for c in prototype_info["cols"] if c in out.columns]
    if not cols:
        return out
    base = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    center = prototype_info["center"].reindex(cols).fillna(0.0)
    scale = prototype_info["scale"].reindex(cols).replace(0.0, np.nan).fillna(1.0)
    z = ((base - center) / scale).clip(-8.0, 8.0).to_numpy(dtype=float)
    distances: dict[str, np.ndarray] = {}
    for name, proto in prototype_info["prototypes"].items():
        proto_vec = np.asarray(proto, dtype=float)
        if len(proto_vec) != len(prototype_info["cols"]):
            continue
        col_indices = [prototype_info["cols"].index(c) for c in cols]
        selected_proto = proto_vec[col_indices]
        distances[name] = np.sqrt(np.mean((z - selected_proto.reshape(1, -1)) ** 2, axis=1))
        out[f"proto_dist_{name}"] = distances[name]
    pairs = [
        ("all_fault", "all_normal"),
        ("target_fault", "target_normal"),
        ("all_fault", "target_hard_negative"),
        ("target_fault", "target_hard_negative"),
    ]
    eps = 1e-9
    for fault_name, normal_name in pairs:
        if fault_name in distances and normal_name in distances:
            out[f"proto_margin_{fault_name}_vs_{normal_name}"] = distances[normal_name] - distances[fault_name]
            out[f"proto_ratio_{fault_name}_to_{normal_name}"] = distances[fault_name] / (distances[normal_name] + eps)
    return out


def train_horizon_model(table: pd.DataFrame, train_ids: set[str], horizon: float, seed: int):
    feature_cols = [
        c
        for c in table.columns
        if c
        not in {
            "sample_id",
            "domain",
            "file_name",
            "binary",
            "hard_negative",
            "severity_name",
            "onset_s",
            "duplicate_group",
        }
        and pd.api.types.is_numeric_dtype(table[c])
    ]
    train = table[table["sample_id"].isin(train_ids)].copy()
    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(medians)
    prototype_info = fit_prototype_features(x_train, train, feature_cols) if USE_PROTOTYPE_FEATURES else None
    x_train = append_prototype_features(x_train, prototype_info)
    sample_weight = np.where(train["domain"].astype(str).to_numpy() == "source5", SOURCE_WEIGHT, 1.0).astype(float)
    hard_negative = ((train["binary"].astype(int).to_numpy() == 0) & (train["hard_negative"].astype(int).to_numpy() == 1))
    sample_weight[hard_negative] *= HARD_NEG_WEIGHT
    if HEAD_MODEL == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=5,
            l2_regularization=2.0,
            random_state=seed,
        )
    elif HEAD_MODEL == "rf":
        model = RandomForestClassifier(
            n_estimators=260,
            max_depth=9,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    elif HEAD_MODEL == "et":
        model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.7,
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
            ),
        )
    x_arr = x_train.to_numpy(dtype=float)
    y_arr = train["binary"].to_numpy(dtype=int)
    if HEAD_MODEL in {"hgb", "et", "rf"}:
        model.fit(x_arr, y_arr, sample_weight=sample_weight)
    else:
        model.fit(x_arr, y_arr, logisticregression__sample_weight=sample_weight)
    return {"horizon": horizon, "model": model, "medians": medians, "feature_cols": feature_cols, "prototype_info": prototype_info}


def predict_horizon(bundle: dict, table: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    data = table[table["sample_id"].isin(ids)].copy()
    x = data[bundle["feature_cols"]].replace([np.inf, -np.inf], np.nan).fillna(bundle["medians"])
    x = append_prototype_features(x, bundle.get("prototype_info"))
    data[f"prob_{int(bundle['horizon'])}s"] = bundle["model"].predict_proba(x.to_numpy(dtype=float))[:, 1]
    return data[
        [
            "sample_id",
            "file_name",
            "binary",
            "hard_negative",
            "severity_name",
            "onset_s",
            "duplicate_group",
            f"prob_{int(bundle['horizon'])}s",
        ]
    ]


def combine_predictions(base: pd.DataFrame, thresholds: dict[float, float], mode: str) -> tuple[pd.DataFrame, dict]:
    out = base.copy()
    out["y_true"] = out["binary"].astype(int)
    out["y_pred"] = 0
    out["alarm_time_s"] = np.nan
    for horizon in sorted(thresholds):
        col = f"prob_{int(horizon)}s"
        hit = out[col] >= thresholds[horizon]
        new_hit = hit & (out["y_pred"] == 0)
        out.loc[new_hit, "alarm_time_s"] = float(horizon)
        out.loc[hit, "y_pred"] = 1
    out["delay_s"] = np.where(
        (out["y_true"] == 1) & (out["y_pred"] == 1) & np.isfinite(out["onset_s"]),
        np.maximum(0.0, out["alarm_time_s"] - out["onset_s"]),
        np.nan,
    )
    metrics = metric_row(out["y_true"].to_numpy(dtype=int), out["y_pred"].to_numpy(dtype=int), out["delay_s"].to_numpy(dtype=float))
    metrics["model"] = mode
    metrics["thresholds"] = ";".join(f"{int(h)}s:{thresholds[h]:.6g}" for h in sorted(thresholds))
    hard_norm = out[(out["y_true"] == 0) & (out["hard_negative"] == 1)]
    metrics["hard_negative_fpr"] = float(hard_norm["y_pred"].mean()) if len(hard_norm) else 0.0
    mild = out[(out["y_true"] == 1) & (out["severity_name"].astype(str).str.contains("10ohm|1ohm", case=False, na=False))]
    metrics["mild_recall"] = float(mild["y_pred"].mean()) if len(mild) else np.nan
    return out, metrics


def select_thresholds(val_pred: pd.DataFrame, horizons: tuple[float, ...], mode: str) -> dict[float, float]:
    candidates: dict[float, np.ndarray] = {}
    for horizon in horizons:
        prob = val_pred[f"prob_{int(horizon)}s"].to_numpy(dtype=float)
        if mode == "confirm":
            candidates[horizon] = np.unique(
                np.concatenate(
                    [
                        np.quantile(prob, [0.20, 0.50, 0.80]),
                        np.array([0.65, 0.78, 0.86]),
                    ]
                )
            )
        else:
            early_floor = 0.55 if horizon <= 75 else 0.65
            candidates[horizon] = np.unique(
                np.concatenate(
                    [
                        np.quantile(prob, [0.25, 0.55, 0.80]),
                        np.array([early_floor, 0.78, 0.88]),
                    ]
                )
            )
    best = None
    if len(horizons) > 4:
        for horizon in horizons:
            prob = val_pred[f"prob_{int(horizon)}s"].to_numpy(dtype=float)
            floor = 0.55 if horizon <= 75 else 0.65
            candidates[horizon] = np.unique(np.concatenate([np.quantile(prob, [0.45, 0.80]), np.array([floor, 0.88])]))
    iterator = product(*(candidates[h] for h in horizons))
    for combo in iterator:
        thresholds = {h: float(t) for h, t in zip(horizons, combo)}
        _, metrics = combine_predictions(val_pred, thresholds, mode)
        delay_penalty = 0.0 if not np.isfinite(metrics["median_delay_s"]) else metrics["median_delay_s"] / 900.0
        if mode == "confirm":
            score = metrics["f1"] + 0.22 * metrics["recall"] + 0.12 * metrics["specificity"] - 0.05 * metrics["fp"] - delay_penalty
        else:
            score = metrics["f1"] + 0.30 * metrics["recall"] + 0.08 * metrics["specificity"] - 0.07 * metrics["fp"] - 2.0 * delay_penalty
            # Early mode should not win by simply suppressing the early heads.
            score -= 0.04 * sum(1 for h, t in thresholds.items() if h <= 60 and t > 0.95)
        if best is None or score > best[0]:
            best = (score, thresholds)
    assert best is not None
    return best[1]


def evaluate_seed(prefix_tables: dict[float, pd.DataFrame], target_meta: pd.DataFrame, source_ids: set[str], seed: int) -> tuple[list[dict], list[pd.DataFrame]]:
    split = stratified_target_split(target_meta, seed)
    train_ids = set(split["train"]) | source_ids
    all_horizons = tuple(sorted(set(HORIZONS_CONFIRM + HORIZONS_CASCADE)))
    bundles_by_horizon = {
        horizon: train_horizon_model(prefix_tables[horizon], train_ids, horizon, seed)
        for horizon in all_horizons
    }
    rows = []
    pred_tables = []
    head_label = HEAD_MODEL.upper()
    cascade_label = "_".join(str(int(h)) for h in HORIZONS_CASCADE)
    for mode, horizons in [(f"PrefixConfirm{head_label}_100_150", HORIZONS_CONFIRM), (f"EarlyCascade{head_label}_{cascade_label}", HORIZONS_CASCADE)]:
        bundles = [bundles_by_horizon[h] for h in horizons]
        merged: dict[str, pd.DataFrame] = {}
        for split_name in ["val", "test"]:
            ids = split[split_name]
            parts = []
            for bundle in bundles:
                parts.append(predict_horizon(bundle, prefix_tables[bundle["horizon"]], ids))
            base = parts[0]
            for part in parts[1:]:
                base = base.merge(
                    part[["sample_id", f"prob_{int(part.columns[-1].split('_')[1][:-1])}s"]],
                    on="sample_id",
                    how="left",
                )
            merged[split_name] = base
        thresholds = select_thresholds(merged["val"], horizons, mode)
        for split_name in ["val", "test"]:
            pred, metrics = combine_predictions(merged[split_name], thresholds, mode)
            metrics.update({"seed": seed, "split": split_name})
            rows.append(metrics)
            pred["seed"] = seed
            pred["split"] = split_name
            pred["model"] = mode
            pred_tables.append(pred)
    return rows, pred_tables


def main() -> None:
    window_df = pd.read_csv(WORK / "window_features.csv", low_memory=False)
    window_df = add_residual_compactness_features(window_df)
    feature_cols = build_feature_columns(window_df)
    target_meta = (
        window_df[window_df["domain"] == "target100"][
            ["sample_id", "file_name", "binary_file", "hard_negative_file", "severity_name"]
        ]
        .drop_duplicates("sample_id")
        .rename(columns={"binary_file": "binary", "hard_negative_file": "hard_negative"})
    )
    target_meta["duplicate_group"] = [
        duplicate_group_name(file_name, binary)
        for file_name, binary in zip(target_meta["file_name"], target_meta["binary"])
    ]
    eval_seeds, split_diagnostics = select_eval_seeds(target_meta, SEED_COUNT)
    split_diagnostics.to_csv(output_csv_path("repeated_seed_split_diagnostics"), index=False, encoding="utf-8-sig")
    print(f"using {len(eval_seeds)} seeds under split_policy={SPLIT_POLICY}: {eval_seeds}", flush=True)
    window_df["duplicate_group"] = [
        duplicate_group_name(file_name, binary) if domain == "target100" else sample_id
        for file_name, binary, domain, sample_id in zip(
            window_df["file_name"], window_df["binary_file"], window_df["domain"], window_df["sample_id"]
        )
    ]
    source_ids = set(window_df.loc[window_df["domain"] == "source5", "sample_id"].drop_duplicates())
    all_horizons = tuple(sorted(set(HORIZONS_CONFIRM + HORIZONS_CASCADE)))
    prefix_tables = load_or_build_prefix_tables(window_df, feature_cols, all_horizons)
    rows: list[dict] = []
    preds: list[pd.DataFrame] = []
    for i, seed in enumerate(eval_seeds, start=1):
        seed_rows, seed_preds = evaluate_seed(prefix_tables, target_meta, source_ids, seed)
        rows.extend(seed_rows)
        preds.extend(seed_preds)
        if i % 5 == 0:
            print(f"completed {i}/{len(eval_seeds)} seeds", flush=True)
    metrics = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    metrics.to_csv(output_csv_path("repeated_seed_metrics"), index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_csv_path("repeated_seed_predictions"), index=False, encoding="utf-8-sig")
    test = metrics[metrics["split"] == "test"].copy()
    summary_rows = []
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_alarm_rate",
        "miss_rate",
        "hard_negative_fpr",
        "mild_recall",
        "mean_delay_s",
        "median_delay_s",
        "p95_delay_s",
        "fp",
        "fn",
    ]
    for model, g in test.groupby("model"):
        row = {"model": model, "n_seeds": int(g["seed"].nunique())}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)
    summary.to_csv(output_csv_path("repeated_seed_summary"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


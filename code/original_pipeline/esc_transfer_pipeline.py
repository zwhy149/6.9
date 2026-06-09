from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"
ONSET_OVERRIDE_CSV = WORK / "onset_times.csv"

TARGET_100 = Path(r"D:\工作文件\大电池短路数据")
SOURCE_5 = Path(r"D:\Battery_ESC\labeled_fault_data")
PUBLIC = Path(r"D:\工作文件\公共短路数据")


RESISTANCE_CLASSES = {
    0.01: ("ESC-0.01ohm", 3),
    0.1: ("ESC-0.1ohm", 2),
    1.0: ("ESC-1ohm", 1),
    10.0: ("ESC-10ohm", 0),
}

WINDOW_SECONDS = (3.0, 6.0, 12.0, 24.0, 48.0)
STEP_SECONDS = 5.0


@dataclass
class Sample:
    sample_id: str
    path: str
    domain: str
    file_name: str
    source_group: str
    binary: int
    severity_name: str | None
    severity_ord: int | None
    hard_negative: int
    time: np.ndarray
    voltage: np.ndarray
    point_label: np.ndarray | None
    onset_s: float | None
    onset_source: str
    system_level: str | None = None


def safe_slug(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:160]


def normalize_ohm_name(name: str) -> str:
    return name.lower().replace("ω", "ohm").replace("Ω", "ohm")


def parse_resistance(name: str) -> float | None:
    normalized = normalize_ohm_name(name)
    if "normal" in normalized:
        return None
    match = re.search(r"(?<!\d)(0\.01|0\.1|10|1)\s*ohm", normalized)
    if not match:
        return None
    return float(match.group(1))


def severity_from_resistance(resistance: float | None) -> tuple[str | None, int | None]:
    if resistance is None:
        return None, None
    # Use nearest known class to handle textual variants.
    nearest = min(RESISTANCE_CLASSES, key=lambda x: abs(x - resistance))
    return RESISTANCE_CLASSES[nearest]


def infer_binary_and_severity(path: Path, domain: str) -> tuple[int, str | None, int | None, int]:
    name = path.name
    group = path.parent.name
    normalized = normalize_ohm_name(name)
    hard_negative = int("difficult" in normalized or re.search(r"normal[_\s-]*d", normalized) is not None)
    if "normal" in normalized or group.lower() == "normal":
        return 0, None, None, hard_negative
    resistance = parse_resistance(name)
    if resistance is not None:
        severity_name, severity_ord = severity_from_resistance(resistance)
        return 1, severity_name, severity_ord, 0
    if domain == "public":
        return 1, None, None, 0
    # DT/GZ files without a resistance token are rare; keep them out of fault classes unless labels prove positive.
    return 0, None, None, hard_negative


def candidate_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def read_workbook(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = "data" if "data" in xl.sheet_names else xl.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def robust_series(time: pd.Series, voltage: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(voltage, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) == 0:
        return t, v
    order = np.argsort(t)
    t, v = t[order], v[order]
    # Collapse duplicate timestamps by median voltage; public sheets can contain logger and machine rows.
    tmp = pd.DataFrame({"t": t, "v": v}).groupby("t", as_index=False)["v"].median()
    return tmp["t"].to_numpy(dtype=float), tmp["v"].to_numpy(dtype=float)


def estimate_onset(time: np.ndarray, voltage: np.ndarray, binary: int) -> tuple[float | None, str]:
    if binary == 0 or len(time) < 20:
        return None, "none"
    t = time
    v = voltage
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if duration <= 0:
        return None, "none"
    dvdt = np.gradient(v, t)
    abs_dvdt = np.abs(dvdt)
    base_n = max(10, min(len(abs_dvdt) // 5, 400))
    base = abs_dvdt[:base_n]
    med = float(np.nanmedian(base))
    mad = float(np.nanmedian(np.abs(base - med))) + 1e-9
    threshold = med + 8.0 * mad
    drop_ratio = np.abs(v - v[0]) / max(abs(v[0]), 1e-6)
    candidates = np.where((abs_dvdt > threshold) & (t > t[0] + 3.0))[0]
    if len(candidates):
        return float(t[candidates[0]]), "estimated_dvdt"
    candidates = np.where((drop_ratio > 0.01) & (t > t[0] + 3.0))[0]
    if len(candidates):
        return float(t[candidates[0]]), "estimated_drop"
    # Many external-short experimental files are recorded after the short-resistance condition is already active.
    return float(t[0]), "assumed_start"


def load_source_target_samples() -> list[Sample]:
    samples: list[Sample] = []
    for domain, root in [("source5", SOURCE_5), ("target100", TARGET_100)]:
        for path in sorted(root.rglob("*.xlsx")):
            try:
                df = read_workbook(path)
            except Exception as exc:
                print(f"skip_read_error,{domain},{path},{exc}")
                continue
            time_col = candidate_col(df, ["Time(s)", "time_s", "time", "t"])
            voltage_col = candidate_col(df, ["Voltage(V)", "voltage_v", "voltage"])
            if time_col is None or voltage_col is None:
                print(f"skip_missing_cols,{domain},{path}")
                continue
            t, v = robust_series(df[time_col], df[voltage_col])
            if len(t) < 20:
                print(f"skip_short,{domain},{path}")
                continue
            binary, sev_name, sev_ord, hard = infer_binary_and_severity(path, domain)
            label_col = candidate_col(df, ["Label"])
            point_label = None
            onset_s = None
            onset_source = "none"
            if label_col is not None:
                raw_lab = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy(dtype=float)
                raw_time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(raw_time)
                raw_time = raw_time[mask]
                raw_lab = raw_lab[mask]
                if len(raw_time) > 0:
                    lab_df = pd.DataFrame({"t": raw_time, "label": (raw_lab > 0).astype(int)})
                    lab_df = lab_df.groupby("t", as_index=False)["label"].max()
                    label_map = dict(zip(lab_df["t"].to_numpy(dtype=float), lab_df["label"].to_numpy(dtype=int)))
                    point_label = np.array([label_map.get(float(x), 0) for x in t], dtype=int)
                    pos = np.where(point_label > 0)[0]
                    if len(pos):
                        binary = 1
                        onset_s = float(t[pos[0]])
                        onset_source = "label"
            if onset_s is None:
                onset_s, onset_source = estimate_onset(t, v, binary)
            sample_id = safe_slug(f"{domain}__{path.parent.name}__{path.stem}")
            samples.append(
                Sample(
                    sample_id=sample_id,
                    path=str(path),
                    domain=domain,
                    file_name=path.name,
                    source_group=path.parent.name,
                    binary=binary,
                    severity_name=sev_name,
                    severity_ord=sev_ord,
                    hard_negative=hard,
                    time=t,
                    voltage=v,
                    point_label=point_label,
                    onset_s=onset_s,
                    onset_source=onset_source,
                )
            )
    return samples


def load_public_samples() -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(PUBLIC.rglob("*.xlsx")):
        try:
            df = read_workbook(path)
        except Exception as exc:
            print(f"skip_public_read_error,{path},{exc}")
            continue
        if "source_file_id" in df.columns:
            groups = list(df.groupby("source_file_id"))
        else:
            groups = [(path.stem, df)]
        for source_id, g in groups:
            time_col = candidate_col(g, ["time_s", "Time(s)", "time", "t"])
            voltage_col = candidate_col(g, ["voltage_v", "Voltage(V)", "voltage"])
            if time_col is None or voltage_col is None:
                continue
            t, v = robust_series(g[time_col], g[voltage_col])
            if len(t) < 20:
                continue
            onset_s = None
            if "estimated_short_onset_s" in g.columns:
                onset_vals = pd.to_numeric(g["estimated_short_onset_s"], errors="coerce").dropna().to_numpy(dtype=float)
                if len(onset_vals):
                    onset_s = float(np.nanmedian(onset_vals))
            if onset_s is None:
                onset_s, _ = estimate_onset(t, v, 1)
            meta = {}
            for col in ["system_level", "soh_group", "cooling", "pack_structure", "scenario_cn"]:
                if col in g.columns:
                    vals = g[col].dropna().astype(str).unique()
                    meta[col] = vals[0] if len(vals) else None
            sample_id = safe_slug(f"public__{path.stem}__{source_id}")
            samples.append(
                Sample(
                    sample_id=sample_id,
                    path=str(path),
                    domain="public",
                    file_name=path.name,
                    source_group=str(source_id),
                    binary=1,
                    severity_name=None,
                    severity_ord=None,
                    hard_negative=0,
                    time=t,
                    voltage=v,
                    point_label=None,
                    onset_s=onset_s,
                    onset_source="public_estimated_short_onset_s",
                    system_level=meta.get("system_level"),
                )
            )
    return samples


def apply_onset_overrides(samples: list[Sample]) -> dict[str, int | str]:
    if not ONSET_OVERRIDE_CSV.exists():
        return {"used": 0, "path": str(ONSET_OVERRIDE_CSV)}
    df = pd.read_csv(ONSET_OVERRIDE_CSV)
    cols = {str(c).strip().lower(): c for c in df.columns}
    onset_col = cols.get("onset_s") or cols.get("short_onset_s") or cols.get("短路时刻") or cols.get("短路时间")
    if onset_col is None:
        raise ValueError(f"{ONSET_OVERRIDE_CSV} must contain onset_s or 短路时刻 column")
    by_sample = {}
    by_file = {}
    if "sample_id" in cols:
        by_sample = {
            str(r[cols["sample_id"]]): float(r[onset_col])
            for _, r in df.iterrows()
            if pd.notna(r.get(onset_col))
        }
    if "file_name" in cols:
        by_file = {
            str(r[cols["file_name"]]): float(r[onset_col])
            for _, r in df.iterrows()
            if pd.notna(r.get(onset_col))
        }
    if "文件名" in cols:
        by_file.update(
            {
                str(r[cols["文件名"]]): float(r[onset_col])
                for _, r in df.iterrows()
                if pd.notna(r.get(onset_col))
            }
        )
    used = 0
    for s in samples:
        if s.binary != 1:
            continue
        onset = by_sample.get(s.sample_id, by_file.get(s.file_name))
        if onset is not None and np.isfinite(onset):
            s.onset_s = float(onset)
            s.onset_source = "user_onset_times_csv"
            used += 1
    return {"used": used, "path": str(ONSET_OVERRIDE_CSV)}


def safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(np.quantile(x, q))


def linear_slope(t: np.ndarray, v: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    tt = t - t[0]
    denom = float(np.dot(tt - tt.mean(), tt - tt.mean()))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(tt - tt.mean(), v - v.mean()) / denom)


def residual_stats(t: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    if len(t) < 3:
        return 0.0, 0.0, 0.0
    slope = linear_slope(t, v)
    fit = v[0] + slope * (t - t[0])
    r = v - fit
    return float(np.std(r)), float(np.max(np.abs(r))), float(np.mean(np.abs(np.diff(r)))) if len(r) > 1 else 0.0


def feature_vector(time: np.ndarray, voltage: np.ndarray, prefix: str = "") -> dict[str, float]:
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    if len(t) < 2:
        return {}
    duration = max(float(t[-1] - t[0]), 1e-6)
    v0 = float(v[0])
    vend = float(v[-1])
    scale = max(abs(v0), 1e-6)
    dv = np.diff(v)
    dt = np.diff(t)
    valid_dt = np.where(np.abs(dt) > 1e-9, dt, np.nan)
    dvdt = dv / valid_dt
    dvdt = dvdt[np.isfinite(dvdt)]
    slope = linear_slope(t, v)
    res_std, res_absmax, res_absdiff = residual_stats(t, v)
    neg = -dvdt
    abs_dvdt = np.abs(dvdt)
    drop = v0 - vend
    abs_move = float(np.sum(np.abs(dv)))
    signed_move = float(np.sum(dv))
    monotonicity = abs(float(vend - v0)) / (abs_move + 1e-9)
    max_drop = float(np.max(v0 - v)) if len(v) else 0.0
    max_rise = float(np.max(v - v0)) if len(v) else 0.0
    risk_score = (
        4.0 * abs(drop / scale)
        + 8.0 * abs(slope / scale)
        + 1.2 * safe_quantile(abs_dvdt / scale, 0.95)
        + 0.8 * (res_absmax / scale)
    )
    out = {
        "duration_s": duration,
        "v0": v0,
        "vend": vend,
        "v_mean": float(np.mean(v)),
        "v_std": float(np.std(v)),
        "v_min": float(np.min(v)),
        "v_max": float(np.max(v)),
        "drop": float(drop),
        "drop_norm": float(drop / scale),
        "abs_drop_norm": float(abs(drop) / scale),
        "max_drop_norm": float(max_drop / scale),
        "max_rise_norm": float(max_rise / scale),
        "range_norm": float((np.max(v) - np.min(v)) / scale),
        "slope": float(slope),
        "slope_norm": float(slope / scale),
        "abs_slope_norm": float(abs(slope) / scale),
        "dvdt_mean": float(np.mean(dvdt)) if len(dvdt) else 0.0,
        "dvdt_std": float(np.std(dvdt)) if len(dvdt) else 0.0,
        "neg_dvdt_q95_norm": safe_quantile(neg / scale, 0.95),
        "abs_dvdt_q95_norm": safe_quantile(abs_dvdt / scale, 0.95),
        "abs_dvdt_max_norm": float(np.max(abs_dvdt) / scale) if len(abs_dvdt) else 0.0,
        "res_std_norm": float(res_std / scale),
        "res_absmax_norm": float(res_absmax / scale),
        "res_absdiff_norm": float(res_absdiff / scale),
        "abs_move_norm": float(abs_move / scale),
        "signed_move_norm": float(signed_move / scale),
        "monotonicity": float(monotonicity),
        "risk_score": float(risk_score),
    }
    if prefix:
        return {f"{prefix}{k}": v for k, v in out.items()}
    return out


def recent_slice(time: np.ndarray, end_idx: int, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    end_t = time[end_idx]
    start_t = end_t - seconds
    start_idx = int(np.searchsorted(time, start_t, side="left"))
    return time[start_idx : end_idx + 1], start_idx


def make_window_features(sample: Sample) -> pd.DataFrame:
    t = sample.time
    v = sample.voltage
    if len(t) < 20:
        return pd.DataFrame()
    start_time = t[0] + max(WINDOW_SECONDS)
    end_times = np.arange(start_time, t[-1] + 1e-9, STEP_SECONDS)
    rows = []
    for end_time in end_times:
        end_idx = int(np.searchsorted(t, end_time, side="right") - 1)
        if end_idx <= 2:
            continue
        row: dict[str, float | str | int | None] = {}
        for seconds in WINDOW_SECONDS:
            window_t, start_idx = recent_slice(t, end_idx, seconds)
            if len(window_t) < 4:
                continue
            window_v = v[start_idx : end_idx + 1]
            row.update(feature_vector(window_t, window_v, prefix=f"w{int(seconds)}_"))
        if not row:
            continue
        row.update(
            {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "file_name": sample.file_name,
                "source_group": sample.source_group,
                "path": sample.path,
                "binary_file": sample.binary,
                "severity_name": sample.severity_name,
                "severity_ord": sample.severity_ord if sample.severity_ord is not None else -1,
                "hard_negative_file": sample.hard_negative,
                "onset_s": sample.onset_s if sample.onset_s is not None else np.nan,
                "onset_source": sample.onset_source,
                "t_end": float(t[end_idx]),
            }
        )
        if sample.point_label is not None:
            max_win = int(np.searchsorted(t, t[end_idx] - max(WINDOW_SECONDS), side="left"))
            window_lab = sample.point_label[max_win : end_idx + 1]
            y = int(np.any(window_lab > 0))
        elif sample.binary == 1:
            y = int(sample.onset_s is None or t[end_idx] >= sample.onset_s)
        else:
            y = 0
        row["y"] = y
        rows.append(row)
    return pd.DataFrame(rows)


def make_file_features(sample: Sample) -> dict[str, float | str | int | None]:
    feats = feature_vector(sample.time, sample.voltage, prefix="full_")
    for seconds in [12.0, 48.0, 120.0]:
        if sample.time[-1] - sample.time[0] >= seconds:
            idx0 = int(np.searchsorted(sample.time, sample.time[-1] - seconds, side="left"))
            feats.update(feature_vector(sample.time[idx0:], sample.voltage[idx0:], prefix=f"tail{int(seconds)}_"))
    if sample.binary == 1 and sample.onset_s is not None and np.isfinite(sample.onset_s):
        onset_idx = int(np.searchsorted(sample.time, sample.onset_s, side="left"))
        for seconds in [10.0, 30.0, 60.0, 120.0]:
            end_idx = int(np.searchsorted(sample.time, sample.onset_s + seconds, side="right"))
            if end_idx - onset_idx >= 4:
                feats.update(
                    feature_vector(
                        sample.time[onset_idx:end_idx],
                        sample.voltage[onset_idx:end_idx],
                        prefix=f"post_onset_{int(seconds)}_",
                    )
                )
    feats.update(
        {
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "file_name": sample.file_name,
            "source_group": sample.source_group,
            "path": sample.path,
            "binary": sample.binary,
            "severity_name": sample.severity_name,
            "severity_ord": sample.severity_ord if sample.severity_ord is not None else -1,
            "hard_negative": sample.hard_negative,
            "onset_s": sample.onset_s if sample.onset_s is not None else np.nan,
            "onset_source": sample.onset_source,
            "n_points": len(sample.time),
            "duration_s": float(sample.time[-1] - sample.time[0]),
        }
    )
    return feats


def numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def fill_numeric(df: pd.DataFrame, feature_cols: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    if medians is None:
        medians = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(medians).astype(float)
    return x, medians


def coral_transform(source_x: np.ndarray, target_x: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    if len(source_x) < 3 or len(target_x) < 3:
        return source_x.copy()
    source_mean = source_x.mean(axis=0, keepdims=True)
    target_mean = target_x.mean(axis=0, keepdims=True)
    xs = source_x - source_mean
    xt = target_x - target_mean
    cs = np.cov(xs, rowvar=False) + reg * np.eye(xs.shape[1])
    ct = np.cov(xt, rowvar=False) + reg * np.eye(xt.shape[1])
    cs_inv_sqrt = linalg.fractional_matrix_power(cs, -0.5).real
    ct_sqrt = linalg.fractional_matrix_power(ct, 0.5).real
    return xs @ cs_inv_sqrt @ ct_sqrt + target_mean


def stratified_file_split(target_meta: pd.DataFrame) -> dict[str, list[str]]:
    meta = target_meta.copy()
    strat = []
    for _, r in meta.iterrows():
        if r["binary"] == 0:
            strat.append("normal_hard" if r["hard_negative"] else "normal")
        else:
            strat.append(str(r["severity_name"]))
    meta["strat"] = strat
    counts = meta["strat"].value_counts()
    rare = set(counts[counts < 3].index)
    meta["strat2"] = meta["strat"].where(~meta["strat"].isin(rare), "rare")
    train_ids, temp_ids = train_test_split(
        meta["sample_id"],
        test_size=0.36,
        random_state=SEED,
        stratify=meta["strat2"],
    )
    temp = meta[meta["sample_id"].isin(temp_ids)].copy()
    counts = temp["strat2"].value_counts()
    strat_temp = temp["strat2"] if counts.min() >= 2 else None
    val_ids, test_ids = train_test_split(
        temp["sample_id"],
        test_size=0.5,
        random_state=SEED,
        stratify=strat_temp,
    )
    return {"target_train": list(train_ids), "target_val": list(val_ids), "target_test": list(test_ids)}


def stratified_generic_split(meta: pd.DataFrame, prefix: str) -> dict[str, list[str]]:
    data = meta.copy()
    strat = []
    for _, r in data.iterrows():
        if r["binary"] == 0:
            strat.append("normal_hard" if r.get("hard_negative", 0) else "normal")
        else:
            strat.append(str(r.get("severity_name", "fault")))
    data["strat"] = strat
    counts = data["strat"].value_counts()
    data["strat2"] = data["strat"].where(~data["strat"].isin(set(counts[counts < 3].index)), "rare")
    train_ids, temp_ids = train_test_split(
        data["sample_id"],
        test_size=0.30,
        random_state=SEED,
        stratify=data["strat2"] if data["strat2"].value_counts().min() >= 2 else None,
    )
    temp = data[data["sample_id"].isin(temp_ids)].copy()
    strat_temp = temp["strat2"] if temp["strat2"].value_counts().min() >= 2 else None
    val_ids, test_ids = train_test_split(
        temp["sample_id"],
        test_size=0.50,
        random_state=SEED,
        stratify=strat_temp,
    )
    return {f"{prefix}_train": list(train_ids), f"{prefix}_val": list(val_ids), f"{prefix}_test": list(test_ids)}


def class_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    weights = np.ones(len(y), dtype=float)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    for c, n in zip(classes, counts):
        weights[y == c] *= total / (len(classes) * max(n, 1))
    return weights


def train_binary_models(window_df: pd.DataFrame, splits: dict[str, list[str]], feature_cols: list[str]):
    source = window_df[window_df["domain"] == "source5"].copy()
    target_train = window_df[window_df["sample_id"].isin(splits["target_train"])].copy()
    val = window_df[window_df["sample_id"].isin(splits["target_val"])].copy()
    test = window_df[window_df["sample_id"].isin(splits["target_test"])].copy()

    def target_mil_subset(target_df: pd.DataFrame, quantile: float = 0.72) -> pd.DataFrame:
        rows = []
        risk_col = "w48_risk_score" if "w48_risk_score" in target_df.columns else "w24_risk_score"
        event_cols = [c for c in ["w6_res_absdiff_norm", "w12_res_absdiff_norm", "w6_abs_dvdt_q95_norm", "w12_abs_dvdt_q95_norm"] if c in target_df.columns]
        for _, g in target_df.groupby("sample_id"):
            g = g.copy()
            if int(g["binary_file"].iloc[0]) == 0:
                g["y"] = 0
                rows.append(g)
                continue
            evidence = g[risk_col].to_numpy(dtype=float)
            if event_cols:
                event = g[event_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                event = np.max(event * np.array([220.0 if "res" in c else 35.0 for c in event_cols]), axis=1)
                evidence = evidence + 0.35 * event
            cut = float(np.nanquantile(evidence, quantile))
            keep = evidence >= cut
            # Keep a small minimum number of positive instances per positive bag.
            if keep.sum() < 6 and len(g) >= 6:
                order = np.argsort(evidence)[-6:]
                keep = np.zeros(len(g), dtype=bool)
                keep[order] = True
            pos = g.loc[keep].copy()
            pos["y"] = 1
            rows.append(pos)
        return pd.concat(rows, ignore_index=True)

    mil_target_train = target_mil_subset(target_train)
    mil_train = pd.concat([source, mil_target_train], ignore_index=True)
    train_sets = {
        "SourceOnly_RF": source,
        "TargetOnly_RF": target_train,
        "NaivePool_RF": pd.concat([source, target_train], ignore_index=True),
        "CORAL_RF": pd.concat([source, target_train], ignore_index=True),
        "HRC_TAGS_ET": pd.concat([source, target_train], ignore_index=True),
        "HRC_TAGS_MIL": mil_train,
        "HRC_TAGS_PROTO": pd.concat([source, target_train], ignore_index=True),
        "HRC_no_CORAL": pd.concat([source, target_train], ignore_index=True),
        "HRC_no_HN": pd.concat([source, target_train], ignore_index=True),
    }
    medians = None
    models = {}
    train_info = {}
    base_train = pd.concat([source, target_train], ignore_index=True)
    _, medians = fill_numeric(base_train, feature_cols)
    target_train_x, _ = fill_numeric(target_train, feature_cols, medians)
    for name, train_df in train_sets.items():
        train_x, _ = fill_numeric(train_df, feature_cols, medians)
        x_np = train_x.to_numpy(dtype=float)
        y = train_df["y"].to_numpy(dtype=int)
        sample_weight = class_sample_weights(y)
        if "HRC" in name:
            sample_weight *= np.where(train_df["domain"].to_numpy() == "target100", 3.0, 1.0)
            if name != "HRC_no_HN":
                hn = train_df["hard_negative_file"].to_numpy(dtype=int)
                # Online hard-negative mining: normal windows in the highest risk quantile get extra weight.
                normal_mask = train_df["y"].to_numpy(dtype=int) == 0
                risk_col = "w48_risk_score" if "w48_risk_score" in train_df.columns else feature_cols[0]
                normal_risk = train_df.loc[normal_mask, risk_col].replace([np.inf, -np.inf], np.nan).dropna()
                risk_cut = float(normal_risk.quantile(0.85)) if len(normal_risk) else np.inf
                mined_hn = ((normal_mask) & (train_df[risk_col].to_numpy(dtype=float) >= risk_cut)).astype(int)
                sample_weight *= np.where((hn == 1) | (mined_hn == 1), 4.0, 1.0)
        if name in {"CORAL_RF", "HRC_TAGS_ET", "HRC_TAGS_MIL", "HRC_TAGS_PROTO", "HRC_no_HN"}:
            is_source = train_df["domain"].to_numpy() == "source5"
            x_np = x_np.copy()
            if np.any(is_source) and len(target_train_x):
                x_np[is_source] = coral_transform(x_np[is_source], target_train_x.to_numpy(dtype=float))
        model = ExtraTreesClassifier(
            n_estimators=500,
            random_state=SEED,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        if name.endswith("_RF") and not name.startswith("HRC"):
            model = RandomForestClassifier(
                n_estimators=400,
                random_state=SEED,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
            )
        model.fit(x_np, y, sample_weight=sample_weight)
        model_info = {"model": model, "medians": medians, "coral": name in {"CORAL_RF", "HRC_TAGS_ET", "HRC_TAGS_MIL", "HRC_TAGS_PROTO", "HRC_no_HN"}}
        if name == "HRC_TAGS_PROTO":
            proto_scaler = StandardScaler()
            z = proto_scaler.fit_transform(x_np)
            y_arr = train_df["y"].to_numpy(dtype=int)
            domain_arr = train_df["domain"].to_numpy()
            norm_mask = (y_arr == 0) & (domain_arr == "target100")
            fault_mask = (y_arr == 1) & (domain_arr == "target100")
            if fault_mask.sum() < 30:
                fault_mask = y_arr == 1
            if norm_mask.sum() >= 10 and fault_mask.sum() >= 10:
                k_norm = int(min(12, norm_mask.sum()))
                k_fault = int(min(12, fault_mask.sum()))
                nn_norm = NearestNeighbors(n_neighbors=k_norm, metric="euclidean").fit(z[norm_mask])
                nn_fault = NearestNeighbors(n_neighbors=k_fault, metric="euclidean").fit(z[fault_mask])
                model_info.update(
                    {
                        "proto_scaler": proto_scaler,
                        "proto_norm": nn_norm,
                        "proto_fault": nn_fault,
                    }
                )
        models[name] = model_info
        train_info[name] = {
            "n_train_windows": int(len(train_df)),
            "positive_rate": float(np.mean(y)),
            "sample_weight_mean": float(np.mean(sample_weight)),
        }
    return models, train_info, val, test


def predict_model(model_info: dict, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x, _ = fill_numeric(df, feature_cols, model_info["medians"])
    # Target/public are already evaluated in their own feature coordinates.
    p = model_info["model"].predict_proba(x.to_numpy(dtype=float))
    if p.shape[1] == 1:
        base = np.zeros(len(df)) if model_info["model"].classes_[0] == 0 else np.ones(len(df))
    else:
        idx = list(model_info["model"].classes_).index(1)
        base = p[:, idx]
    if {"proto_scaler", "proto_norm", "proto_fault"}.issubset(model_info.keys()):
        z = model_info["proto_scaler"].transform(x.to_numpy(dtype=float))
        d_norm = model_info["proto_norm"].kneighbors(z, return_distance=True)[0].mean(axis=1)
        d_fault = model_info["proto_fault"].kneighbors(z, return_distance=True)[0].mean(axis=1)
        margin = d_norm - d_fault
        scale = np.nanmedian(np.abs(margin)) + 1e-6
        proto_score = 1.0 / (1.0 + np.exp(-(margin / scale)))
        base = np.clip(0.72 * base + 0.28 * proto_score, 0.0, 1.0)
    return base


def threshold_rule(df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
    risk_col = "w48_risk_score" if "w48_risk_score" in df.columns else "w24_risk_score"
    score = df[risk_col].to_numpy(dtype=float)
    if threshold is None:
        return score
    return (score >= threshold).astype(float)


def gate_evidence(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "w3_risk_score",
        "w6_risk_score",
        "w12_risk_score",
        "w3_abs_dvdt_q95_norm",
        "w6_abs_dvdt_q95_norm",
        "w12_abs_dvdt_q95_norm",
        "w3_res_absdiff_norm",
        "w6_res_absdiff_norm",
        "w12_res_absdiff_norm",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.ones(len(df), dtype=float)
    x = df[available].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    # Put tiny derivative/residual features onto the same order as the handcrafted risk scores.
    scaled = x.copy()
    for j, c in enumerate(available):
        if "abs_dvdt" in c:
            scaled[:, j] *= 35.0
        if "res_absdiff" in c:
            scaled[:, j] *= 220.0
    return np.max(scaled, axis=1)


def apply_gate(scores: np.ndarray, df: pd.DataFrame, gate_threshold: float | None) -> np.ndarray:
    if gate_threshold is None or gate_threshold <= 0:
        return scores
    gate = gate_evidence(df)
    return np.where(gate >= gate_threshold, scores, -1.0)


def file_online_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    consecutive: int = 2,
    include_delay: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    tmp = df.copy()
    tmp["score"] = scores
    rows = []
    for sample_id, g in tmp.sort_values("t_end").groupby("sample_id"):
        y = int(g["binary_file"].iloc[0])
        hard = int(g["hard_negative_file"].iloc[0])
        onset = float(g["onset_s"].iloc[0]) if np.isfinite(g["onset_s"].iloc[0]) else np.nan
        alarms = g["score"].to_numpy(dtype=float) >= threshold
        t_end = g["t_end"].to_numpy(dtype=float)
        alarm_idx = None
        run = 0
        for i, flag in enumerate(alarms):
            run = run + 1 if flag else 0
            if run >= consecutive:
                alarm_idx = i - consecutive + 1
                break
        alarm_time = float(t_end[alarm_idx]) if alarm_idx is not None else np.nan
        pred = int(alarm_idx is not None)
        delay = np.nan
        if include_delay and y == 1 and alarm_idx is not None and np.isfinite(onset):
            delay = max(0.0, alarm_time - onset)
        rows.append(
            {
                "sample_id": sample_id,
                "file_name": g["file_name"].iloc[0],
                "domain": g["domain"].iloc[0],
                "y_true": y,
                "y_pred": pred,
                "hard_negative": hard,
                "onset_s": onset,
                "alarm_time_s": alarm_time,
                "delay_s": delay,
                "max_score": float(np.max(g["score"])),
                "mean_score": float(np.mean(g["score"])),
            }
        )
    pred_df = pd.DataFrame(rows)
    y_true = pred_df["y_true"].to_numpy(dtype=int)
    y_pred = pred_df["y_pred"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    delays = pred_df.loc[(pred_df["y_true"] == 1) & np.isfinite(pred_df["delay_s"]), "delay_s"].to_numpy(dtype=float)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "miss_rate": float(fn / max(tp + fn, 1)),
        "hard_negative_fpr": float(
            pred_df.loc[(pred_df["y_true"] == 0) & (pred_df["hard_negative"] == 1), "y_pred"].mean()
            if len(pred_df.loc[(pred_df["y_true"] == 0) & (pred_df["hard_negative"] == 1)])
            else 0.0
        ),
        "mean_delay_s": float(np.mean(delays)) if len(delays) else np.nan,
        "median_delay_s": float(np.median(delays)) if len(delays) else np.nan,
        "p95_delay_s": float(np.quantile(delays, 0.95)) if len(delays) else np.nan,
        "n_files": int(len(pred_df)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return metrics, pred_df


def choose_threshold(df: pd.DataFrame, scores: np.ndarray) -> tuple[float, int, float, pd.DataFrame]:
    candidates = np.quantile(scores[np.isfinite(scores)], np.linspace(0.08, 0.94, 18))
    candidates = np.unique(np.concatenate([candidates, np.linspace(0.15, 0.95, 9)]))
    gates = gate_evidence(df)
    gate_candidates = [0.0]
    if len(gates):
        gate_candidates += list(np.unique(np.quantile(gates[np.isfinite(gates)], [0.65, 0.80, 0.90])))
    rows = []
    best = None
    for gate_thr in gate_candidates:
        gated_scores = apply_gate(scores, df, float(gate_thr))
        finite_scores = gated_scores[np.isfinite(gated_scores)]
        local_candidates = candidates
        if len(finite_scores):
            local_candidates = np.unique(np.concatenate([candidates, np.quantile(finite_scores, np.linspace(0.15, 0.92, 10))]))
        for thr in local_candidates:
            for cons in [1, 2, 3]:
                m, _ = file_online_metrics(df, gated_scores, float(thr), consecutive=cons)
                delay_penalty = 0.0 if not np.isfinite(m["median_delay_s"]) else min(m["median_delay_s"] / 220.0, 0.25)
                score = (
                    m["f1"]
                    + 0.30 * m["specificity"]
                    - 0.15 * m["false_alarm_rate"]
                    - 0.35 * m["hard_negative_fpr"]
                    - delay_penalty
                )
                row = {"threshold": float(thr), "consecutive": cons, "gate_threshold": float(gate_thr), "selection_score": float(score), **m}
                rows.append(row)
                if best is None or score > best[0]:
                    best = (score, float(thr), cons, float(gate_thr))
    assert best is not None
    return best[1], best[2], best[3], pd.DataFrame(rows)


def train_severity_model(file_df: pd.DataFrame, splits: dict[str, list[str]], feature_cols: list[str]):
    train = file_df[
        ((file_df["domain"] == "source5") | (file_df["sample_id"].isin(splits["target_train"])))
        & (file_df["binary"] == 1)
        & (file_df["severity_ord"] >= 0)
    ].copy()
    test = file_df[(file_df["sample_id"].isin(splits["target_test"])) & (file_df["binary"] == 1) & (file_df["severity_ord"] >= 0)].copy()
    if len(train) < 4 or len(test) < 2 or train["severity_ord"].nunique() < 2:
        return None, pd.DataFrame(), {}
    x_train, med = fill_numeric(train, feature_cols)
    x_test, _ = fill_numeric(test, feature_cols, med)
    y_train = train["severity_ord"].to_numpy(dtype=int)
    sw = class_sample_weights(y_train) * np.where(train["domain"].to_numpy() == "target100", 3.0, 1.0)
    model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, min_samples_leaf=1, n_jobs=-1)
    model.fit(x_train.to_numpy(dtype=float), y_train, sample_weight=sw)
    pred = model.predict(x_test.to_numpy(dtype=float))
    out = test[["sample_id", "file_name", "severity_name", "severity_ord"]].copy()
    out["pred_ord"] = pred
    inv = {v[1]: v[0] for v in RESISTANCE_CLASSES.values()}
    out["pred_name"] = [inv.get(int(x), str(x)) for x in pred]
    metrics = {
        "accuracy": float(accuracy_score(out["severity_ord"], out["pred_ord"])),
        "balanced_accuracy": float(balanced_accuracy_score(out["severity_ord"], out["pred_ord"])),
        "macro_f1": float(f1_score(out["severity_ord"], out["pred_ord"], average="macro", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(out["severity_ord"].to_numpy(dtype=int) - out["pred_ord"].to_numpy(dtype=int)))),
        "n_test_fault_files": int(len(out)),
    }
    return model, out, metrics


def evaluate_source5(
    window_df: pd.DataFrame,
    file_df: pd.DataFrame,
    source_splits: dict[str, list[str]],
    window_feature_cols: list[str],
    file_feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    train = window_df[window_df["sample_id"].isin(source_splits["source5_train"])].copy()
    val = window_df[window_df["sample_id"].isin(source_splits["source5_val"])].copy()
    test = window_df[window_df["sample_id"].isin(source_splits["source5_test"])].copy()
    x_train, med = fill_numeric(train, window_feature_cols)
    y_train = train["y"].to_numpy(dtype=int)
    sw = class_sample_weights(y_train)
    hn = train["hard_negative_file"].to_numpy(dtype=int)
    risk_col = "w48_risk_score" if "w48_risk_score" in train.columns else window_feature_cols[0]
    normal_mask = y_train == 0
    normal_risk = train.loc[normal_mask, risk_col].replace([np.inf, -np.inf], np.nan).dropna()
    risk_cut = float(normal_risk.quantile(0.85)) if len(normal_risk) else np.inf
    mined_hn = ((normal_mask) & (train[risk_col].to_numpy(dtype=float) >= risk_cut)).astype(int)
    sw *= np.where((hn == 1) | (mined_hn == 1), 3.5, 1.0)
    model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, min_samples_leaf=2, n_jobs=-1)
    model.fit(x_train.to_numpy(dtype=float), y_train, sample_weight=sw)
    info = {"model": model, "medians": med}
    val_scores = predict_model(info, val, window_feature_cols)
    thr, cons, gate_thr, grid = choose_threshold(val, val_scores)
    test_scores = apply_gate(predict_model(info, test, window_feature_cols), test, gate_thr)
    metrics, pred = file_online_metrics(test, test_scores, thr, cons)
    metrics.update({"model": "Source5_ET_holdout", "threshold": thr, "consecutive": cons, "gate_threshold": gate_thr})
    metrics_df = pd.DataFrame([metrics])
    pred["model"] = "Source5_ET_holdout"

    sev_train = file_df[
        (file_df["sample_id"].isin(source_splits["source5_train"]))
        & (file_df["binary"] == 1)
        & (file_df["severity_ord"] >= 0)
    ].copy()
    sev_test = file_df[
        (file_df["sample_id"].isin(source_splits["source5_test"]))
        & (file_df["binary"] == 1)
        & (file_df["severity_ord"] >= 0)
    ].copy()
    sev_metrics = {}
    sev_pred = pd.DataFrame()
    if len(sev_train) >= 8 and len(sev_test) >= 3 and sev_train["severity_ord"].nunique() >= 2:
        xs, m = fill_numeric(sev_train, file_feature_cols)
        xt, _ = fill_numeric(sev_test, file_feature_cols, m)
        ys = sev_train["severity_ord"].to_numpy(dtype=int)
        sev_model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, min_samples_leaf=1, n_jobs=-1)
        sev_model.fit(xs.to_numpy(dtype=float), ys, sample_weight=class_sample_weights(ys))
        yp = sev_model.predict(xt.to_numpy(dtype=float))
        sev_pred = sev_test[["sample_id", "file_name", "severity_name", "severity_ord"]].copy()
        sev_pred["pred_ord"] = yp
        inv = {v[1]: v[0] for v in RESISTANCE_CLASSES.values()}
        sev_pred["pred_name"] = [inv.get(int(x), str(x)) for x in yp]
        sev_metrics = {
            "accuracy": float(accuracy_score(sev_pred["severity_ord"], sev_pred["pred_ord"])),
            "balanced_accuracy": float(balanced_accuracy_score(sev_pred["severity_ord"], sev_pred["pred_ord"])),
            "macro_f1": float(f1_score(sev_pred["severity_ord"], sev_pred["pred_ord"], average="macro", zero_division=0)),
            "ordinal_mae": float(np.mean(np.abs(sev_pred["severity_ord"].to_numpy(dtype=int) - sev_pred["pred_ord"].to_numpy(dtype=int)))),
            "n_test_fault_files": int(len(sev_pred)),
        }
    grid.to_csv(OUT / "threshold_grid_source5_holdout.csv", index=False, encoding="utf-8-sig")
    return metrics_df, pred, sev_metrics, sev_pred


def aggregate_prefix_features(window_df: pd.DataFrame, feature_cols: list[str], horizon_s: float) -> pd.DataFrame:
    rows = []
    for sample_id, g0 in window_df.groupby("sample_id"):
        g = g0.sort_values("t_end")
        prefix = g[g["t_end"] <= horizon_s]
        if len(prefix) == 0:
            prefix = g.head(1)
        row: dict[str, float | str | int] = {
            "sample_id": sample_id,
            "domain": g0["domain"].iloc[0],
            "file_name": g0["file_name"].iloc[0],
            "binary": int(g0["binary_file"].iloc[0]),
            "hard_negative": int(g0["hard_negative_file"].iloc[0]),
            "onset_s": float(g0["onset_s"].iloc[0]) if np.isfinite(g0["onset_s"].iloc[0]) else np.nan,
        }
        for col in feature_cols:
            if col not in prefix.columns:
                continue
            arr = prefix[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(arr) == 0:
                continue
            row[f"{col}_max"] = float(np.max(arr))
            row[f"{col}_p95"] = float(np.quantile(arr, 0.95))
            row[f"{col}_median"] = float(np.median(arr))
            row[f"{col}_mean"] = float(np.mean(arr))
        rows.append(row)
    return pd.DataFrame(rows)


def score_prefix_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    return float(f1_score(y_true, y_pred, zero_division=0) + 0.20 * specificity - 0.05 * fp)


def train_causal_prefix_hgb(
    window_df: pd.DataFrame,
    target_splits: dict[str, list[str]],
    source_splits: dict[str, list[str]],
    feature_cols: list[str],
    horizons: tuple[float, ...] = (100.0, 150.0),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    train_ids = set(target_splits["target_train"])
    # Source-domain labels are fully available; use all source folds for the target-transfer calibrator.
    for key, ids in source_splits.items():
        if key.startswith("source5_"):
            train_ids.update(ids)
    val_ids = set(target_splits["target_val"])
    test_ids = set(target_splits["target_test"])
    horizon_outputs = []
    threshold_rows = []
    val_base = None
    test_base = None
    for horizon in horizons:
        table = aggregate_prefix_features(window_df, feature_cols, horizon)
        prefix_cols = [
            c
            for c in table.columns
            if c not in {"sample_id", "domain", "file_name", "binary", "hard_negative", "onset_s"}
            and pd.api.types.is_numeric_dtype(table[c])
        ]
        train = table[table["sample_id"].isin(train_ids)].copy()
        val = table[table["sample_id"].isin(val_ids)].copy()
        test = table[table["sample_id"].isin(test_ids)].copy()
        x_train = train[prefix_cols].replace([np.inf, -np.inf], np.nan)
        medians = x_train.median(numeric_only=True).fillna(0.0)
        x_train = x_train.fillna(medians)
        x_val = val[prefix_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
        x_test = test[prefix_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
        y_train = train["binary"].to_numpy(dtype=int)
        model = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=3,
        )
        model.fit(x_train.to_numpy(dtype=float), y_train)
        val_prob = model.predict_proba(x_val.to_numpy(dtype=float))[:, 1]
        test_prob = model.predict_proba(x_test.to_numpy(dtype=float))[:, 1]
        candidates = np.unique(
            np.concatenate(
                [
                    np.quantile(val_prob, np.linspace(0.05, 0.95, 20)),
                    np.array([0.80, 0.84, 0.87, 0.90]),
                ]
            )
        )
        best = None
        for threshold in candidates:
            pred = (val_prob >= threshold).astype(int)
            score = score_prefix_threshold(val["binary"].to_numpy(dtype=int), pred)
            row = {
                "horizon_s": horizon,
                "threshold": float(threshold),
                "selection_score": score,
                "accuracy": float(accuracy_score(val["binary"], pred)),
                "precision": float(precision_score(val["binary"], pred, zero_division=0)),
                "recall": float(recall_score(val["binary"], pred, zero_division=0)),
                "f1": float(f1_score(val["binary"], pred, zero_division=0)),
            }
            tn, fp, fn, tp = confusion_matrix(val["binary"], pred, labels=[0, 1]).ravel()
            row.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
            threshold_rows.append(row)
            if best is None or score > best[0]:
                best = (score, float(threshold))
        assert best is not None
        if val_base is None:
            val_base = val[["sample_id", "file_name", "binary", "hard_negative", "onset_s"]].copy()
            test_base = test[["sample_id", "file_name", "binary", "hard_negative", "onset_s"]].copy()
        horizon_outputs.append(
            {
                "horizon_s": horizon,
                "threshold": best[1],
                "model": model,
                "medians": medians,
                "prefix_cols": prefix_cols,
                "val_prob": pd.Series(val_prob, index=val["sample_id"].to_numpy()),
                "test_prob": pd.Series(test_prob, index=test["sample_id"].to_numpy()),
            }
        )

    def combine(base: pd.DataFrame, split_name: str) -> tuple[dict[str, float], pd.DataFrame]:
        out = base.copy()
        out["y_true"] = out["binary"].astype(int)
        out["y_pred"] = 0
        out["alarm_time_s"] = np.nan
        for horizon_output in horizon_outputs:
            horizon = float(horizon_output["horizon_s"])
            threshold = float(horizon_output["threshold"])
            prob = horizon_output[f"{split_name}_prob"]
            out[f"prefix_hgb_{int(horizon)}s_prob"] = out["sample_id"].map(prob).astype(float)
            hit = out[f"prefix_hgb_{int(horizon)}s_prob"] >= threshold
            new_hit = hit & (out["y_pred"] == 0)
            out.loc[new_hit, "alarm_time_s"] = horizon
            out.loc[hit, "y_pred"] = 1
        out["delay_s"] = np.where(
            (out["y_true"] == 1) & (out["y_pred"] == 1) & np.isfinite(out["onset_s"]),
            np.maximum(0.0, out["alarm_time_s"] - out["onset_s"]),
            np.nan,
        )
        y_true = out["y_true"].to_numpy(dtype=int)
        y_pred = out["y_pred"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        delays = out.loc[(out["y_true"] == 1) & np.isfinite(out["delay_s"]), "delay_s"].to_numpy(dtype=float)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan,
            "miss_rate": float(fn / max(tp + fn, 1)),
            "hard_negative_fpr": float(
                out.loc[(out["y_true"] == 0) & (out["hard_negative"] == 1), "y_pred"].mean()
                if len(out.loc[(out["y_true"] == 0) & (out["hard_negative"] == 1)])
                else 0.0
            ),
            "mean_delay_s": float(np.mean(delays)) if len(delays) else np.nan,
            "median_delay_s": float(np.median(delays)) if len(delays) else np.nan,
            "p95_delay_s": float(np.quantile(delays, 0.95)) if len(delays) else np.nan,
            "n_files": int(len(out)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "model": "CausalPrefixHGB_100_150",
            "threshold": ";".join(f"{int(x['horizon_s'])}s:{x['threshold']:.6g}" for x in horizon_outputs),
            "consecutive": 1,
            "gate_threshold": 0.0,
        }
        out["model"] = "CausalPrefixHGB_100_150"
        out["max_score"] = out[[c for c in out.columns if c.startswith("prefix_hgb_")]].max(axis=1)
        out["mean_score"] = out[[c for c in out.columns if c.startswith("prefix_hgb_")]].mean(axis=1)
        return metrics, out

    assert val_base is not None and test_base is not None
    val_metrics, val_pred = combine(val_base, "val")
    test_metrics, test_pred = combine(test_base, "test")
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df["selected_thresholds"] = ";".join(f"{int(x['horizon_s'])}s:{x['threshold']:.6g}" for x in horizon_outputs)
    pred = pd.concat(
        [
            val_pred.assign(split="target_val"),
            test_pred.assign(split="target_test"),
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame([{"split": "target_val", **val_metrics}, {"split": "target_test", **test_metrics}])
    return metrics, pred, threshold_df, horizon_outputs


def eval_causal_prefix_hgb_bundle(
    window_df: pd.DataFrame,
    feature_cols: list[str],
    horizon_outputs: list[dict],
    split_name: str = "public",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = None
    out_tables = {}
    for horizon_output in horizon_outputs:
        horizon = float(horizon_output["horizon_s"])
        table = aggregate_prefix_features(window_df, feature_cols, horizon)
        x = table[horizon_output["prefix_cols"]].replace([np.inf, -np.inf], np.nan).fillna(horizon_output["medians"])
        prob = horizon_output["model"].predict_proba(x.to_numpy(dtype=float))[:, 1]
        if base is None:
            base = table[["sample_id", "file_name", "binary", "hard_negative", "onset_s"]].copy()
            base["y_true"] = base["binary"].astype(int)
            base["y_pred"] = 0
            base["alarm_time_s"] = np.nan
        out_tables[int(horizon)] = pd.Series(prob, index=table["sample_id"].to_numpy())
    assert base is not None
    for horizon_output in horizon_outputs:
        horizon = int(horizon_output["horizon_s"])
        threshold = float(horizon_output["threshold"])
        base[f"prefix_hgb_{horizon}s_prob"] = base["sample_id"].map(out_tables[horizon]).astype(float)
        hit = base[f"prefix_hgb_{horizon}s_prob"] >= threshold
        new_hit = hit & (base["y_pred"] == 0)
        base.loc[new_hit, "alarm_time_s"] = float(horizon)
        base.loc[hit, "y_pred"] = 1
    base["delay_s"] = np.where(
        (base["y_true"] == 1) & (base["y_pred"] == 1) & np.isfinite(base["onset_s"]),
        np.maximum(0.0, base["alarm_time_s"] - base["onset_s"]),
        np.nan,
    )
    y_true = base["y_true"].to_numpy(dtype=int)
    y_pred = base["y_pred"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    delays = base.loc[(base["y_true"] == 1) & np.isfinite(base["delay_s"]), "delay_s"].to_numpy(dtype=float)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "miss_rate": float(fn / max(tp + fn, 1)),
        "hard_negative_fpr": float(
            base.loc[(base["y_true"] == 0) & (base["hard_negative"] == 1), "y_pred"].mean()
            if len(base.loc[(base["y_true"] == 0) & (base["hard_negative"] == 1)])
            else 0.0
        ),
        "mean_delay_s": float(np.mean(delays)) if len(delays) else np.nan,
        "median_delay_s": float(np.median(delays)) if len(delays) else np.nan,
        "p95_delay_s": float(np.quantile(delays, 0.95)) if len(delays) else np.nan,
        "n_files": int(len(base)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "model": "CausalPrefixHGB_100_150",
        "threshold": ";".join(f"{int(x['horizon_s'])}s:{x['threshold']:.6g}" for x in horizon_outputs),
        "consecutive": 1,
        "gate_threshold": 0.0,
    }
    base["model"] = "CausalPrefixHGB_100_150"
    base["max_score"] = base[[c for c in base.columns if c.startswith("prefix_hgb_")]].max(axis=1)
    base["mean_score"] = base[[c for c in base.columns if c.startswith("prefix_hgb_")]].mean(axis=1)
    return pd.DataFrame([{"split": split_name, **metrics}]), base


def public_locked_eval(models: dict, thresholds: dict, public_windows: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds_all = []
    for name, info in models.items():
        if name not in thresholds:
            continue
        scores = predict_model(info, public_windows, feature_cols)
        thr, cons = thresholds[name]["threshold"], thresholds[name]["consecutive"]
        gate_thr = thresholds[name].get("gate_threshold", 0.0)
        scores = apply_gate(scores, public_windows, gate_thr)
        metrics, pred = file_online_metrics(public_windows, scores, thr, consecutive=cons)
        metrics["model"] = name
        metrics["threshold"] = thr
        metrics["consecutive"] = cons
        metrics["gate_threshold"] = gate_thr
        rows.append(metrics)
        pred["model"] = name
        preds_all.append(pred)
    return pd.DataFrame(rows), pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()


def metric_ci(values: np.ndarray, func, n_boot: int = 1000) -> tuple[float, float]:
    values = np.asarray(values)
    if len(values) < 2:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(values), len(values))
        boots.append(func(values[idx]))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)


def save_dataset_tables(samples: list[Sample], public_samples: list[Sample], target_meta: pd.DataFrame, splits: dict[str, list[str]]) -> None:
    rows = []
    for s in samples + public_samples:
        rows.append(
            {
                "sample_id": s.sample_id,
                "domain": s.domain,
                "file_name": s.file_name,
                "source_group": s.source_group,
                "binary": s.binary,
                "severity_name": s.severity_name,
                "severity_ord": s.severity_ord,
                "hard_negative": s.hard_negative,
                "onset_s": s.onset_s,
                "onset_source": s.onset_source,
                "n_points": len(s.time),
                "duration_s": float(s.time[-1] - s.time[0]),
                "path": s.path,
            }
        )
    meta = pd.DataFrame(rows)
    split_map = {}
    for split_name, ids in splits.items():
        for sid in ids:
            split_map[sid] = split_name
    default_split = np.where(meta["domain"] == "source5", "source_train", "locked_public")
    mapped_split = meta["sample_id"].map(split_map)
    meta["split"] = mapped_split.where(mapped_split.notna(), default_split)
    meta.to_csv(OUT / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    target_meta.to_csv(OUT / "target_file_split.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


def plot_dataset_distribution(meta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    counts = meta.groupby(["domain", "binary"]).size().unstack(fill_value=0)
    counts.rename(columns={0: "normal", 1: "fault"}).plot(kind="bar", ax=axes[0], color=["#3A7CA5", "#D1495B"])
    axes[0].set_title("File-level class distribution")
    axes[0].set_ylabel("Files")
    axes[0].set_xlabel("")
    sev = meta[meta["severity_name"].notna()].groupby(["domain", "severity_name"]).size().unstack(fill_value=0)
    if len(sev):
        sev.plot(kind="bar", ax=axes[1], colormap="viridis")
    axes[1].set_title("Fault resistance classes")
    axes[1].set_ylabel("Files")
    axes[1].set_xlabel("")
    fig.savefig(OUT / "fig01_dataset_distribution.png", dpi=220)
    plt.close(fig)


def plot_example_curves(samples: list[Sample]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    buckets = [
        ("target normal difficult", [s for s in samples if s.domain == "target100" and s.binary == 0 and s.hard_negative]),
        ("target ESC 0.01/0.1", [s for s in samples if s.domain == "target100" and s.binary == 1 and s.severity_ord in {2, 3}]),
        ("source normal difficult", [s for s in samples if s.domain == "source5" and s.binary == 0 and s.hard_negative]),
        ("source labeled ESC", [s for s in samples if s.domain == "source5" and s.binary == 1 and s.onset_source == "label"]),
    ]
    for ax, (title, bucket) in zip(axes.ravel(), buckets):
        for s in bucket[:5]:
            t = s.time - s.time[0]
            v = s.voltage
            ax.plot(t, (v - v[0]) / max(abs(v[0]), 1e-6), lw=1.2, alpha=0.85)
            if s.onset_s is not None and np.isfinite(s.onset_s):
                ax.axvline(s.onset_s - s.time[0], color="#444444", lw=0.7, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized voltage change")
        ax.grid(alpha=0.25)
    fig.savefig(OUT / "fig02_representative_curves.png", dpi=220)
    plt.close(fig)


def plot_feature_alignment(window_df: pd.DataFrame, splits: dict[str, list[str]], feature_cols: list[str]) -> None:
    source = window_df[window_df["domain"] == "source5"].sample(min(1500, (window_df["domain"] == "source5").sum()), random_state=SEED)
    target = window_df[window_df["sample_id"].isin(splits["target_train"])].sample(
        min(1500, window_df["sample_id"].isin(splits["target_train"]).sum()), random_state=SEED
    )
    combined = pd.concat([source, target], ignore_index=True)
    x, med = fill_numeric(combined, feature_cols)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=2, random_state=SEED)
    emb_before = pca.fit_transform(x_scaled)
    source_x = x_scaled[combined["domain"].to_numpy() == "source5"]
    target_x = x_scaled[combined["domain"].to_numpy() == "target100"]
    aligned_source = coral_transform(source_x, target_x)
    x_after = x_scaled.copy()
    x_after[combined["domain"].to_numpy() == "source5"] = aligned_source
    emb_after = PCA(n_components=2, random_state=SEED).fit_transform(x_after)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, emb, title in [(axes[0], emb_before, "Before CORAL"), (axes[1], emb_after, "After CORAL")]:
        for dom, color in [("source5", "#3A7CA5"), ("target100", "#D1495B")]:
            mask = combined["domain"].to_numpy() == dom
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.45, label=dom, color=color)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "fig03_domain_alignment_pca.png", dpi=220)
    plt.close(fig)


def plot_model_bars(metrics_df: pd.DataFrame) -> None:
    keep = metrics_df.sort_values("f1", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "accuracy", "Target binary accuracy"),
        (axes[1], "f1", "Target binary F1"),
        (axes[2], "hard_negative_fpr", "Hard-negative FPR"),
    ]:
        ax.barh(keep["model"], keep[metric], color="#2E86AB" if metric != "hard_negative_fpr" else "#D1495B")
        ax.set_title(title)
        ax.set_xlim(0, max(1.0, float(keep[metric].max()) * 1.05))
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUT / "fig04_model_comparison_bars.png", dpi=220)
    plt.close(fig)


def plot_confusion(pred_df: pd.DataFrame, name: str, out_name: str) -> None:
    cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "ESC"])
    ax.set_yticks([0, 1], labels=["normal", "ESC"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / out_name, dpi=220)
    plt.close(fig)


def plot_delay(pred_df: pd.DataFrame) -> None:
    d = pred_df[(pred_df["y_true"] == 1) & np.isfinite(pred_df["delay_s"])]["delay_s"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if len(d):
        ax.hist(d, bins=min(12, max(4, len(d) // 2)), color="#3A7CA5", alpha=0.85)
        ax.axvline(np.median(d), color="#D1495B", lw=2, label=f"median={np.median(d):.1f}s")
        ax.legend(frameon=False)
    ax.set_title("Online detection delay on target test")
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Files")
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "fig06_target_delay_histogram.png", dpi=220)
    plt.close(fig)


def plot_feature_importance(model_info: dict, feature_cols: list[str]) -> None:
    model = model_info["model"]
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.barh(imp.index[::-1], imp.values[::-1], color="#2E86AB")
    ax.set_title("Top feature importances")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(OUT / "fig07_feature_importance.png", dpi=220)
    plt.close(fig)


def plot_ablation(metrics_df: pd.DataFrame) -> None:
    names = ["HRC_TAGS_MIL", "HRC_TAGS_PROTO", "HRC_TAGS_ET", "HRC_no_CORAL", "HRC_no_HN", "NaivePool_RF", "TargetOnly_RF", "SourceOnly_RF"]
    df = metrics_df[metrics_df["model"].isin(names)].set_index("model").reindex(names).dropna(how="all")
    fig, ax1 = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(df))
    ax1.plot(x, df["f1"], marker="o", color="#2E86AB", label="F1")
    ax1.plot(x, df["specificity"], marker="s", color="#3A7CA5", label="Specificity")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x, labels=df.index, rotation=25, ha="right")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, df["hard_negative_fpr"], marker="^", color="#D1495B", label="HN-FPR")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Hard-negative FPR")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left")
    ax1.set_title("Ablation pressure test")
    fig.savefig(OUT / "fig08_ablation_pressure_test.png", dpi=220)
    plt.close(fig)


def plot_public_timelines(public_windows: pd.DataFrame, pred_df: pd.DataFrame, best_name: str, scores: np.ndarray, threshold: float) -> None:
    tmp = public_windows.copy()
    tmp["score"] = scores
    examples = pred_df[pred_df["model"] == best_name].head(6)["sample_id"].tolist() if "model" in pred_df.columns else []
    if not examples:
        examples = tmp["sample_id"].drop_duplicates().head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    for ax, sid in zip(axes.ravel(), examples):
        g = tmp[tmp["sample_id"] == sid].sort_values("t_end")
        ax.plot(g["t_end"], g["score"], color="#2E86AB", lw=1.5)
        ax.axhline(threshold, color="#D1495B", lw=1.2)
        onset = g["onset_s"].iloc[0]
        if np.isfinite(onset):
            ax.axvline(onset, color="#444444", lw=1.0, alpha=0.7)
        ax.set_title(g["file_name"].iloc[0][:32], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ESC score")
        ax.grid(alpha=0.25)
    fig.savefig(OUT / "fig09_public_locked_timelines.png", dpi=220)
    plt.close(fig)


def plot_severity_confusion(sev_pred: pd.DataFrame) -> None:
    if len(sev_pred) == 0:
        return
    labels = sorted(set(sev_pred["severity_ord"]).union(set(sev_pred["pred_ord"])))
    cm = confusion_matrix(sev_pred["severity_ord"], sev_pred["pred_ord"], labels=labels)
    inv = {v[1]: v[0] for v in RESISTANCE_CLASSES.values()}
    label_names = [inv.get(int(x), str(x)) for x in labels]
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(labels)), labels=label_names, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Target fault severity confusion")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(OUT / "fig10_severity_confusion.png", dpi=220)
    plt.close(fig)


def write_method_brief(
    metrics_df: pd.DataFrame,
    severity_metrics: dict,
    public_metrics: pd.DataFrame,
    source_metrics: pd.DataFrame | None = None,
    source_severity_metrics: dict | None = None,
) -> None:
    best = metrics_df.sort_values("f1", ascending=False).iloc[0].to_dict()
    lines = []
    lines.append("# HRC-TAGS ESC transfer detection report")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "HRC-TAGS is implemented here as a reproducible, non-PyTorch transfer baseline: "
        "causal multi-scale voltage windows, physics-aware trend/residual evidence, "
        "CORAL source-to-target feature alignment, target-weighted learning, and explicit hard-negative weighting."
    )
    lines.append("")
    lines.append("Core novelty claim for the paper draft should be conservative:")
    lines.append(
        "the method does not just transfer small-cell ESC patterns to large-cell data; it explicitly penalizes normal samples whose voltage trend resembles ESC and reports their false-alarm rate separately."
    )
    lines.append("")
    lines.append("## Best target-test binary result")
    for key in [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_alarm_rate",
        "hard_negative_fpr",
        "median_delay_s",
        "p95_delay_s",
        "threshold",
        "consecutive",
        "gate_threshold",
        "n_files",
    ]:
        if key in best:
            lines.append(f"- {key}: {best[key]}")
    lines.append("")
    lines.append("## 5Ah source-domain hold-out result")
    if source_metrics is not None and len(source_metrics):
        for k, v in source_metrics.iloc[0].to_dict().items():
            lines.append(f"- {k}: {v}")
    if source_severity_metrics:
        lines.append("")
        lines.append("5Ah source-domain severity:")
        for k, v in source_severity_metrics.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Target fault-severity result")
    if severity_metrics:
        for k, v in severity_metrics.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- Not enough held-out severity classes for a stable estimate.")
    lines.append("")
    lines.append("## Public locked-test result")
    if len(public_metrics):
        pub_best = public_metrics[public_metrics["model"] == best["model"]]
        if len(pub_best):
            lines.append("Target-selected model on public locked test:")
            for k, v in pub_best.iloc[0].to_dict().items():
                lines.append(f"- {k}: {v}")
        pub_ranked = public_metrics.sort_values(["f1", "recall"], ascending=False)
        if len(pub_ranked):
            lines.append("")
            lines.append("Best public-recall backbone on locked public test:")
            for k, v in pub_ranked.iloc[0].to_dict().items():
                lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Integrity notes")
    lines.append("- 100Ah files do not contain point-level labels; target delay is based on estimated/assumed onset and must be described that way.")
    lines.append("- Public data are used only after model/threshold selection and are not used for training or calibration.")
    lines.append("- CausalPrefixHGB is a 100Ah target-domain confirmation head; HRC_TAGS_ET remains the stronger public-domain short-circuit recall head.")
    lines.append("- Report hard-negative false-alarm rate; overall accuracy alone is insufficient for reviewer scrutiny.")
    (OUT / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_literature_and_design_note() -> None:
    text = """# Literature-grounded design note

Search date: 2026-06-05.

Selected signals from recent work:

- Journal of Power Sources 2024 ViT transfer learning work uses simulated/real transfer, mean-difference feature enhancement, and targets similar voltage-anomaly faults. This supports a transfer-learning framing but also shows reviewers will expect explicit handling of fault-type similarity.
- Journal of Energy Storage 2024 SDANet emphasizes sub-domain adaptation for battery pack multi-fault diagnosis under operating-condition shifts. This supports class/sub-domain rather than only marginal distribution alignment.
- Scientific Reports 2020 demonstrates that physics-informed voltage/current features plus Random Forest can exceed 97% ISC detection in an online setting. This justifies a strong interpretable feature baseline, especially when deep-learning dependencies are unavailable.
- Journal of Energy Storage 2024 transfer-strategy work argues MMD/DANN-style approaches are more suitable under substantial domain shift, while direct fine tuning is enough only for small shifts.

Proposed paper-level method name:

HRC-TAGS: Hard-negative Residual-Contrastive Transfer with Adaptive Gated Short-circuit evidence.

Implementation in this run:

- Causal multi-scale voltage windows: 3, 6, 12, 24, and 48 s.
- Physics-aware evidence: normalized voltage drop/rise, slope, residual roughness, derivative burst, monotonicity, and trend/residual risk score.
- Capacity/domain robustness: raw voltage is supplemented by normalized features; source 5Ah windows are CORAL-aligned to target 100Ah feature covariance.
- Hard-negative treatment: filename-labeled difficult normal samples and mined high-risk normal windows get higher training weights.
- Locked public validation: public short-circuit workbooks are not used for fitting or threshold selection.

What is not yet implemented:

- A real neural contrastive loss or Transformer encoder is not used because bundled Python lacks PyTorch. The delivered experiment is a reviewer-safe classical implementation of the same hypothesis, not a final deep model.
- 100Ah online onset is estimated because the files contain no point-level Label column.
"""
    (OUT / "method_design_note.md").write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()
    print("loading samples")
    samples = load_source_target_samples()
    public_samples = load_public_samples()
    onset_override_info = apply_onset_overrides(samples)
    print(f"samples={len(samples)} public={len(public_samples)}")
    file_df = pd.DataFrame([make_file_features(s) for s in samples])
    public_file_df = pd.DataFrame([make_file_features(s) for s in public_samples])
    target_meta = file_df[file_df["domain"] == "target100"][
        ["sample_id", "file_name", "binary", "severity_name", "severity_ord", "hard_negative", "onset_s", "onset_source"]
    ].copy()
    splits = stratified_file_split(target_meta)
    source_meta = file_df[file_df["domain"] == "source5"][
        ["sample_id", "file_name", "binary", "severity_name", "severity_ord", "hard_negative", "onset_s", "onset_source"]
    ].copy()
    source_splits = stratified_generic_split(source_meta, "source5")
    all_splits = {**splits, **source_splits}
    save_dataset_tables(samples, public_samples, target_meta, all_splits)
    meta = pd.read_csv(OUT / "dataset_manifest.csv", encoding="utf-8-sig")
    plot_dataset_distribution(meta)
    plot_example_curves(samples)

    print("making windows")
    win_parts = []
    for i, s in enumerate(samples):
        w = make_window_features(s)
        if len(w):
            win_parts.append(w)
        if (i + 1) % 50 == 0:
            print(f"windowed {i + 1}/{len(samples)}")
    window_df = pd.concat(win_parts, ignore_index=True)
    pub_parts = [make_window_features(s) for s in public_samples]
    public_windows = pd.concat([p for p in pub_parts if len(p)], ignore_index=True)
    window_df.to_csv(WORK / "window_features.csv", index=False, encoding="utf-8-sig")
    public_windows.to_csv(WORK / "public_window_features.csv", index=False, encoding="utf-8-sig")

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
    feature_cols = numeric_feature_columns(window_df, exclude)
    # Avoid raw absolute voltage as the main domain signal; normalized features are retained.
    feature_cols = [c for c in feature_cols if not re.search(r"(^|_)v0$|(^|_)vend$|(^|_)v_mean$|(^|_)v_min$|(^|_)v_max$", c)]
    with open(OUT / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    plot_feature_alignment(window_df, splits, feature_cols)

    print("training binary models")
    models, train_info, val_df, test_df = train_binary_models(window_df, splits, feature_cols)
    prefix_metrics_df, prefix_pred_df, prefix_threshold_df, prefix_bundle = train_causal_prefix_hgb(
        window_df,
        splits,
        source_splits,
        feature_cols,
    )
    prefix_threshold_df.to_csv(OUT / "threshold_grid_CausalPrefixHGB_100_150.csv", index=False, encoding="utf-8-sig")
    prefix_pred_df.to_csv(OUT / "target_prefix_hgb_predictions.csv", index=False, encoding="utf-8-sig")
    prefix_metrics_df.to_csv(OUT / "target_prefix_hgb_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_rows = []
    pred_tables = []
    thresholds = {}

    # Add deterministic threshold-only baseline using risk score.
    risk_scores_val = threshold_rule(val_df)
    risk_thr, risk_cons, risk_gate, risk_grid = choose_threshold(val_df, risk_scores_val)
    risk_scores_test = apply_gate(threshold_rule(test_df), test_df, risk_gate)
    risk_metrics, risk_pred = file_online_metrics(test_df, risk_scores_test, risk_thr, risk_cons)
    risk_metrics.update({"model": "PWEG_threshold", "threshold": risk_thr, "consecutive": risk_cons, "gate_threshold": risk_gate})
    metrics_rows.append(risk_metrics)
    risk_pred["model"] = "PWEG_threshold"
    pred_tables.append(risk_pred)
    risk_grid.to_csv(OUT / "threshold_grid_PWEG_threshold.csv", index=False, encoding="utf-8-sig")

    for name, info in models.items():
        scores_val = predict_model(info, val_df, feature_cols)
        thr, cons, gate_thr, grid = choose_threshold(val_df, scores_val)
        thresholds[name] = {"threshold": thr, "consecutive": cons, "gate_threshold": gate_thr}
        grid.to_csv(OUT / f"threshold_grid_{name}.csv", index=False, encoding="utf-8-sig")
        scores_test = apply_gate(predict_model(info, test_df, feature_cols), test_df, gate_thr)
        m, pred = file_online_metrics(test_df, scores_test, thr, cons)
        m.update({"model": name, "threshold": thr, "consecutive": cons, "gate_threshold": gate_thr})
        metrics_rows.append(m)
        pred["model"] = name
        pred_tables.append(pred)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["f1", "accuracy"], ascending=False)
    pred_df = pd.concat(pred_tables, ignore_index=True)
    prefix_test_metrics = prefix_metrics_df[prefix_metrics_df["split"] == "target_test"].drop(columns=["split"]).iloc[0].to_dict()
    metrics_df = pd.concat([metrics_df, pd.DataFrame([prefix_test_metrics])], ignore_index=True).sort_values(
        ["f1", "accuracy"], ascending=False
    )
    prefix_target_pred = prefix_pred_df[prefix_pred_df["split"] == "target_test"].drop(columns=["split", "binary"], errors="ignore")
    pred_df = pd.concat([pred_df, prefix_target_pred], ignore_index=True, sort=False)
    metrics_df.to_csv(OUT / "target_binary_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT / "target_file_predictions.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "binary_train_info.json", "w", encoding="utf-8") as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)
    with open(OUT / "selected_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    best_name = str(metrics_df.iloc[0]["model"])
    if best_name == "PWEG_threshold":
        best_pred = pred_df[pred_df["model"] == best_name]
    else:
        best_pred = pred_df[pred_df["model"] == best_name]
    plot_model_bars(metrics_df)
    plot_confusion(best_pred, f"{best_name} target test", "fig05_target_binary_confusion.png")
    plot_delay(best_pred)
    if best_name in models:
        plot_feature_importance(models[best_name], feature_cols)
    else:
        plot_feature_importance(models["HRC_TAGS_ET"], feature_cols)
    plot_ablation(metrics_df)

    file_feature_cols = numeric_feature_columns(
        file_df,
        {
            "sample_id",
            "domain",
            "file_name",
            "source_group",
            "path",
            "binary",
            "severity_name",
            "severity_ord",
            "hard_negative",
            "onset_s",
            "onset_source",
        },
    )
    file_feature_cols = [c for c in file_feature_cols if not re.search(r"(^|_)v0$|(^|_)vend$|(^|_)v_mean$|(^|_)v_min$|(^|_)v_max$", c)]

    print("evaluating source5 holdout")
    source_metrics_df, source_pred, source_sev_metrics, source_sev_pred = evaluate_source5(
        window_df,
        file_df,
        source_splits,
        feature_cols,
        file_feature_cols,
    )
    source_metrics_df.to_csv(OUT / "source5_binary_metrics.csv", index=False, encoding="utf-8-sig")
    source_pred.to_csv(OUT / "source5_file_predictions.csv", index=False, encoding="utf-8-sig")
    source_sev_pred.to_csv(OUT / "source5_severity_predictions.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "source5_severity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(source_sev_metrics, f, ensure_ascii=False, indent=2)

    print("training severity model")
    _, sev_pred, sev_metrics = train_severity_model(file_df, splits, file_feature_cols)
    sev_pred.to_csv(OUT / "target_severity_predictions.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "target_severity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(sev_metrics, f, ensure_ascii=False, indent=2)
    plot_severity_confusion(sev_pred)

    print("public locked test")
    pub_metrics, pub_pred = public_locked_eval(models, thresholds, public_windows, feature_cols)
    prefix_pub_metrics, prefix_pub_pred = eval_causal_prefix_hgb_bundle(public_windows, feature_cols, prefix_bundle, split_name="public")
    pub_metrics = pd.concat([pub_metrics, prefix_pub_metrics.drop(columns=["split"], errors="ignore")], ignore_index=True, sort=False)
    prefix_pub_pred = prefix_pub_pred.drop(columns=["binary"], errors="ignore")
    pub_pred = pd.concat([pub_pred, prefix_pub_pred], ignore_index=True, sort=False)
    pub_metrics.to_csv(OUT / "public_locked_metrics.csv", index=False, encoding="utf-8-sig")
    pub_pred.to_csv(OUT / "public_locked_predictions.csv", index=False, encoding="utf-8-sig")
    if best_name in models and len(public_windows):
        pub_scores = apply_gate(
            predict_model(models[best_name], public_windows, feature_cols),
            public_windows,
            thresholds[best_name].get("gate_threshold", 0.0),
        )
        plot_public_timelines(public_windows, pub_pred, best_name, pub_scores, thresholds[best_name]["threshold"])

    write_method_brief(metrics_df, sev_metrics, pub_metrics, source_metrics_df, source_sev_metrics)
    write_literature_and_design_note()

    summary = {
        "n_samples": len(samples),
        "n_public_samples": len(public_samples),
        "n_windows": int(len(window_df)),
        "n_public_windows": int(len(public_windows)),
        "onset_override": onset_override_info,
        "best_model": best_name,
        "best_target_metrics": metrics_df.iloc[0].to_dict(),
        "source5_holdout_metrics": source_metrics_df.iloc[0].to_dict() if len(source_metrics_df) else {},
        "source5_severity_metrics": source_sev_metrics,
        "severity_metrics": sev_metrics,
    }
    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

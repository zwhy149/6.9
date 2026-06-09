from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
WORK = ROOT / "work"

HORIZONS = (50, 75, 100, 150, 250, 400)
SOURCES = [
    ("hgb", "repeated_seed_predictions_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("et", "repeated_seed_predictions_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("global_et", "repeated_seed_predictions_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
]
CHOICE_FILE = OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv"

PG_BASES = (
    "final_drop_norm",
    "max_drop_norm",
    "recovery_fraction",
    "final_to_max_drop_ratio",
    "min_time_fraction",
    "tail_drop_norm",
    "tail_slope_norm",
    "monotone_down_fraction",
    "slope_sign_change_rate",
)


@dataclass(frozen=True)
class VetoSpec:
    fd_low: float
    fd_high: float
    recovery_min: float
    ratio_max: float
    min_time_max: float

    @property
    def name(self) -> str:
        return (
            "veto"
            f"_fd{self.fd_low:.3f}-{self.fd_high:.3f}"
            f"_rec{self.recovery_min:.2f}"
            f"_ratio{self.ratio_max:.2f}"
            f"_t{self.min_time_max:.2f}"
        )


@dataclass(frozen=True)
class RescueSpec:
    fd_min: float
    ratio_min: float
    min_time_min: float
    mono_min: float
    recovery_max: float

    @property
    def name(self) -> str:
        return (
            "rescue"
            f"_fd{self.fd_min:.3f}"
            f"_ratio{self.ratio_min:.2f}"
            f"_t{self.min_time_min:.2f}"
            f"_mono{self.mono_min:.2f}"
            f"_recmax{self.recovery_max:.2f}"
        )


def load_source(prefix: str, file_name: str, model_name: str) -> pd.DataFrame:
    data = pd.read_csv(OUT / file_name)
    data = data[data["model"] == model_name].copy()
    meta_cols = ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split"]
    prob_cols = [f"prob_{h}s" for h in HORIZONS]
    out = data[meta_cols + prob_cols].copy()
    return out.rename(columns={col: f"{prefix}_{col}" for col in prob_cols})


def prepare() -> pd.DataFrame:
    base = None
    for prefix, file_name, model_name in SOURCES:
        part = load_source(prefix, file_name, model_name)
        prob_cols = [f"{prefix}_prob_{h}s" for h in HORIZONS]
        if base is None:
            base = part
        else:
            base = base.merge(part[["sample_id", "seed", "split", *prob_cols]], on=["sample_id", "seed", "split"], how="inner")
    assert base is not None
    return attach_global_prefix_features(base)


def attach_global_prefix_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for horizon in HORIZONS:
        feats = pd.read_csv(WORK / f"prefix_features_rescompact_global_v2_{horizon}s.csv", low_memory=False)
        keep = ["sample_id", *[f"pg_{horizon}s_{base}" for base in PG_BASES]]
        out = out.merge(feats[keep], on="sample_id", how="left")
    return out


def predict_base(frame: pd.DataFrame, choice: pd.Series) -> pd.DataFrame:
    weights = np.array([choice[f"w_{prefix}"] for prefix, _, _ in SOURCES], dtype=float)
    thresholds = np.array([choice[f"threshold_{h}s"] for h in HORIZONS], dtype=float)
    prob = np.zeros((len(frame), len(HORIZONS)), dtype=float)
    for source_idx, (prefix, _, _) in enumerate(SOURCES):
        prob += weights[source_idx] * frame[[f"{prefix}_prob_{h}s" for h in HORIZONS]].to_numpy(dtype=float)
    hits = prob >= thresholds.reshape(1, -1)
    base_pred = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    alarm = np.take(np.array(HORIZONS, dtype=float), first)
    alarm[~base_pred] = np.nan

    out = frame.copy()
    out["base_pred"] = base_pred.astype(int)
    out["base_alarm_time_s"] = alarm
    for base in PG_BASES:
        values = np.full(len(out), np.nan, dtype=float)
        for horizon in HORIZONS:
            mask = out["base_alarm_time_s"].to_numpy(dtype=float) == float(horizon)
            values[mask] = out.loc[mask, f"pg_{horizon}s_{base}"].to_numpy(dtype=float)
        out[f"alarm_{base}"] = values
    return out


def safe_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return values


def build_veto_specs() -> list[VetoSpec]:
    specs = []
    for fd_low in (0.0, 0.015, 0.030):
        for fd_high in (0.060, 0.080, 0.100, 0.140):
            if fd_high <= fd_low:
                continue
            for recovery_min in (0.10, 0.20, 0.35, 0.50):
                for ratio_max in (0.45, 0.60, 0.75, 0.90):
                    for min_time_max in (0.25, 0.40, 0.60):
                        specs.append(VetoSpec(fd_low, fd_high, recovery_min, ratio_max, min_time_max))
    return specs


def build_rescue_specs() -> list[RescueSpec]:
    specs = []
    for fd_min in (0.040, 0.060, 0.080, 0.100, 0.120):
        for ratio_min in (0.75, 0.85, 0.95):
            for min_time_min in (0.65, 0.80, 0.95):
                for mono_min in (0.75, 0.85, 0.95):
                    for recovery_max in (0.05, 0.15, 0.30):
                        specs.append(RescueSpec(fd_min, ratio_min, min_time_min, mono_min, recovery_max))
    return specs


VETO_SPECS = build_veto_specs()
RESCUE_SPECS = build_rescue_specs()


def veto_mask(frame: pd.DataFrame, spec: VetoSpec) -> np.ndarray:
    fd = safe_values(frame, "alarm_final_drop_norm")
    recovery = safe_values(frame, "alarm_recovery_fraction")
    ratio = safe_values(frame, "alarm_final_to_max_drop_ratio")
    min_time = safe_values(frame, "alarm_min_time_fraction")
    base_pred = frame["base_pred"].to_numpy(dtype=int) == 1
    return (
        base_pred
        & (fd >= spec.fd_low)
        & (fd <= spec.fd_high)
        & (recovery >= spec.recovery_min)
        & (ratio <= spec.ratio_max)
        & (min_time <= spec.min_time_max)
    )


def rescue_mask(frame: pd.DataFrame, spec: RescueSpec) -> np.ndarray:
    fd = safe_values(frame, "pg_400s_final_drop_norm")
    ratio = safe_values(frame, "pg_400s_final_to_max_drop_ratio")
    min_time = safe_values(frame, "pg_400s_min_time_fraction")
    monotone = safe_values(frame, "pg_400s_monotone_down_fraction")
    recovery = safe_values(frame, "pg_400s_recovery_fraction")
    base_pred = frame["base_pred"].to_numpy(dtype=int) == 1
    return (
        (~base_pred)
        & (fd >= spec.fd_min)
        & (ratio >= spec.ratio_min)
        & (min_time >= spec.min_time_min)
        & (monotone >= spec.mono_min)
        & (recovery <= spec.recovery_max)
    )


def apply_masks(frame: pd.DataFrame, veto: np.ndarray | None, rescue: np.ndarray | None) -> pd.DataFrame:
    out = frame.copy()
    y_pred = out["base_pred"].to_numpy(dtype=int).copy()
    alarm = out["base_alarm_time_s"].to_numpy(dtype=float).copy()
    if veto is not None:
        y_pred[veto] = 0
        alarm[veto] = np.nan
    if rescue is not None:
        y_pred[rescue] = 1
        alarm[rescue] = 400.0
    out["y_pred"] = y_pred
    out["alarm_time_s"] = alarm
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


def score(row: dict[str, float], base: dict[str, float], complexity: int) -> float:
    # Validation-only objective. The guards discourage trading away fault recall
    # or hard-negative robustness for a small apparent validation gain.
    recall_guard = max(0.0, base["recall"] - 0.05 - row["recall"])
    hard_guard = max(0.0, row["hard_negative_fpr"] - base["hard_negative_fpr"] - 0.05)
    delay = row["median_delay_s"] if np.isfinite(row["median_delay_s"]) else 400.0
    return (
        row["accuracy"]
        + 0.025 * row["f1"]
        + 0.015 * row["specificity"]
        - 0.020 * row["hard_negative_fpr"]
        - 0.120 * recall_guard
        - 0.080 * hard_guard
        - 0.00015 * delay
        - 0.0010 * complexity
    )


def top_individual_candidates(val: pd.DataFrame, base_metrics: dict[str, float]) -> tuple[list[tuple[str, object, np.ndarray]], list[tuple[str, object, np.ndarray]]]:
    veto_ranked = []
    rescue_ranked = []
    none = apply_masks(val, None, None)
    for spec in VETO_SPECS:
        mask = veto_mask(val, spec)
        if not mask.any():
            continue
        row = metrics(apply_masks(val, mask, None))
        veto_ranked.append((score(row, base_metrics, 1), spec.name, spec, mask))
    for spec in RESCUE_SPECS:
        mask = rescue_mask(val, spec)
        if not mask.any():
            continue
        row = metrics(apply_masks(val, None, mask))
        rescue_ranked.append((score(row, base_metrics, 1), spec.name, spec, mask))
    veto_ranked.sort(key=lambda x: x[0], reverse=True)
    rescue_ranked.sort(key=lambda x: x[0], reverse=True)
    base_score = score(metrics(none), base_metrics, 0)
    veto = [("none", None, None)]
    rescue = [("none", None, None)]
    veto.extend([(name, spec, mask) for cand_score, name, spec, mask in veto_ranked[:40] if cand_score >= base_score - 0.015])
    rescue.extend([(name, spec, mask) for cand_score, name, spec, mask in rescue_ranked[:40] if cand_score >= base_score - 0.015])
    return veto, rescue


def choose_gate(val: pd.DataFrame) -> dict[str, object]:
    base_frame = apply_masks(val, None, None)
    base_metrics = metrics(base_frame)
    best = {
        "score": score(base_metrics, base_metrics, 0),
        "veto_name": "none",
        "veto_spec": None,
        "rescue_name": "none",
        "rescue_spec": None,
        "metrics": base_metrics,
    }
    veto_candidates, rescue_candidates = top_individual_candidates(val, base_metrics)
    for veto_name, veto_spec, veto in veto_candidates:
        for rescue_name, rescue_spec, rescue in rescue_candidates:
            complexity = int(veto is not None) + int(rescue is not None)
            row = metrics(apply_masks(val, veto, rescue))
            cand_score = score(row, base_metrics, complexity)
            if cand_score > best["score"]:
                best = {
                    "score": cand_score,
                    "veto_name": veto_name,
                    "veto_spec": veto_spec,
                    "rescue_name": rescue_name,
                    "rescue_spec": rescue_spec,
                    "metrics": row,
                }
    return best


def apply_gate(test: pd.DataFrame, choice: dict[str, object]) -> pd.DataFrame:
    veto = veto_mask(test, choice["veto_spec"]) if choice["veto_spec"] is not None else None
    rescue = rescue_mask(test, choice["rescue_spec"]) if choice["rescue_spec"] is not None else None
    return apply_masks(test, veto, rescue)


def summarize(rows: list[dict[str, float]]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    row: dict[str, float | str | int] = {"model": "ResCompact_HGB_ET_GlobalET_PhysicsGate_accuracy_only", "n_seeds": int(data["seed"].nunique())}
    for col in ["accuracy", "precision", "recall", "f1", "specificity", "hard_negative_fpr", "fp", "fn", "median_delay_s", "p95_delay_s"]:
        row[f"{col}_mean"] = float(data[col].mean())
        row[f"{col}_std"] = float(data[col].std(ddof=1))
        row[f"{col}_min"] = float(data[col].min())
        row[f"{col}_max"] = float(data[col].max())
    return pd.DataFrame([row])


def main() -> None:
    data = prepare()
    choices = pd.read_csv(CHOICE_FILE)
    rows = []
    choice_rows = []
    pred_rows = []
    for seed in sorted(data["seed"].unique()):
        seed_choice = choices[choices["seed"] == seed].iloc[0]
        seed_data = data[data["seed"] == seed].copy()
        seed_pred = predict_base(seed_data, seed_choice)
        val = seed_pred[seed_pred["split"] == "val"].copy()
        test = seed_pred[seed_pred["split"] == "test"].copy()
        gate_choice = choose_gate(val)
        chosen = apply_gate(test, gate_choice)
        row = metrics(chosen)
        row.update({"seed": int(seed)})
        rows.append(row)
        choice_rows.append(
            {
                "seed": int(seed),
                "veto": gate_choice["veto_name"],
                "rescue": gate_choice["rescue_name"],
                "val_score": float(gate_choice["score"]),
                **{f"val_{key}": value for key, value in gate_choice["metrics"].items()},
            }
        )
        pred = chosen[["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s", "seed", "split", "y_pred", "alarm_time_s"]].copy()
        pred["y_true"] = pred["binary"].astype(int)
        pred["delay_s"] = np.where(
            (pred["y_true"].to_numpy(dtype=int) == 1)
            & (pred["y_pred"].to_numpy(dtype=int) == 1)
            & np.isfinite(pred["onset_s"].to_numpy(dtype=float)),
            np.maximum(0.0, pred["alarm_time_s"].to_numpy(dtype=float) - pred["onset_s"].to_numpy(dtype=float)),
            np.nan,
        )
        pred["model"] = "ResCompact_HGB_ET_GlobalET_PhysicsGate_accuracy_only"
        pred_rows.append(pred)
        print(
            f"seed {seed}: acc={row['accuracy']:.4f}, fp={row['fp']}, fn={row['fn']}, "
            f"veto={gate_choice['veto_name']}, rescue={gate_choice['rescue_name']}",
            flush=True,
        )
    detail = pd.DataFrame(rows)
    summary = summarize(rows)
    suffix = "rescompact_multisource_3src_gate_accuracy_only"
    detail.to_csv(OUT / f"{suffix}_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / f"{suffix}_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / f"{suffix}_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{suffix}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


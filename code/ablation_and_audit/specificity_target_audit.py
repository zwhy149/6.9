from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def load_summary(path: str, label: str) -> dict[str, float | str]:
    row = pd.read_csv(OUT / path).iloc[0].to_dict()
    return {
        "experiment": label,
        "accuracy_mean": float(row.get("accuracy_mean", np.nan)),
        "accuracy_std": float(row.get("accuracy_std", np.nan)),
        "specificity_mean": float(row.get("specificity_mean", np.nan)),
        "specificity_std": float(row.get("specificity_std", np.nan)),
        "recall_mean": float(row.get("recall_mean", np.nan)),
        "recall_std": float(row.get("recall_std", np.nan)),
        "fp_mean": float(row.get("fp_mean", np.nan)),
        "fn_mean": float(row.get("fn_mean", np.nan)),
        "n_seeds": int(row.get("n_seeds", 30)),
    }


def add_uncertainty(row: dict[str, float | str]) -> dict[str, float | str]:
    n = int(row["n_seeds"])
    for metric in ["accuracy", "specificity", "recall"]:
        std = float(row[f"{metric}_std"])
        sem = std / math.sqrt(n) if n > 0 else np.nan
        row[f"{metric}_sem"] = sem
        row[f"{metric}_ci95_halfwidth"] = 1.96 * sem
        row[f"{metric}_n_for_sem_0p01"] = math.ceil((std / 0.01) ** 2) if np.isfinite(std) else np.nan
        row[f"{metric}_n_for_sem_0p00866"] = math.ceil((std / 0.00866) ** 2) if np.isfinite(std) else np.nan
    return row


def metric_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tn + tp) / len(y_true)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
    }


def normality_tradeoff() -> pd.DataFrame:
    pred = pd.read_csv(OUT / "rescompact_normality_verifier_predictions.csv")
    y_true = pred["binary"].astype(int).to_numpy()
    base_pred = pred["base_pred"].astype(int).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for column in ["et_fault_prob", "base_alarm_score"]:
        values = pred[column].astype(float).replace([np.inf, -np.inf], np.nan)
        grid = np.unique(np.r_[np.linspace(0.0, 1.0, 201), values.dropna().quantile(np.linspace(0, 1, 101)).to_numpy()])
        for threshold in grid:
            y_pred = base_pred.copy()
            veto = (y_pred == 1) & (values.fillna(np.inf).to_numpy(dtype=float) <= threshold)
            y_pred[veto] = 0
            row = metric_from_pred(y_true, y_pred)
            row.update({"score": column, "threshold": float(threshold), "vetoed": int(veto.sum())})
            rows.append(row)
    trade = pd.DataFrame(rows)
    trade.to_csv(OUT / "specificity_target_tradeoff_oracle.csv", index=False, encoding="utf-8-sig")
    return trade


def fp_frequency() -> pd.DataFrame:
    pred = pd.read_csv(OUT / "validation_model_selector_predictions_conservative_margin005.csv")
    neg = pred[pred["y_true"].astype(int) == 0].copy()
    freq = (
        neg.groupby(["sample_id", "file_name", "hard_negative"], dropna=False)
        .agg(n_test_appearances=("seed", "nunique"), false_positives=("y_pred", "sum"))
        .reset_index()
    )
    freq["fp_rate_when_tested"] = freq["false_positives"] / freq["n_test_appearances"].replace(0, np.nan)
    freq = freq.sort_values(["false_positives", "fp_rate_when_tested"], ascending=False)
    freq.to_csv(OUT / "specificity_target_fp_frequency.csv", index=False, encoding="utf-8-sig")
    return freq


def per_seed_discreteness() -> pd.DataFrame:
    metrics = pd.read_csv(OUT / "validation_model_selector_metrics_conservative_margin005.csv")
    pred = pd.read_csv(OUT / "validation_model_selector_predictions_conservative_margin005.csv")
    normals = (
        pred[pred["y_true"].astype(int) == 0]
        .groupby("seed")
        .agg(n_normals=("sample_id", "count"), fp=("y_pred", "sum"))
        .reset_index()
    )
    out = metrics.merge(normals, on="seed", how="left")
    out["specificity_step_if_one_fp"] = 1.0 / out["n_normals"]
    out.to_csv(OUT / "specificity_target_seed_discreteness.csv", index=False, encoding="utf-8-sig")
    return out


def plot_tradeoff(trade: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    colors = {"et_fault_prob": "#1f77b4", "base_alarm_score": "#d62728"}
    for score, g in trade.groupby("score"):
        ax.plot(g["specificity"], g["accuracy"], lw=1.8, color=colors.get(score, "#333333"), label=score)
    ax.axvline(0.94, color="#222222", lw=1.0, ls="--")
    ax.axhline(0.94, color="#777777", lw=1.0, ls=":")
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Accuracy")
    ax.set_title("Specificity-accuracy trade-off from post-hoc verifier thresholds")
    ax.set_xlim(0.82, 1.005)
    ax.set_ylim(0.62, 0.965)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig31_specificity_accuracy_tradeoff.png")
    plt.close(fig)


def plot_fp_frequency(freq: pd.DataFrame) -> None:
    top = freq.head(10).copy()
    labels = [str(x).replace(".xlsx", "") for x in top["file_name"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    ax.barh(np.arange(len(top)), top["false_positives"], color="#c43c39")
    ax.set_yticks(np.arange(len(top)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("False positives across 30 repeated tests")
    ax.set_title("Normal files driving the specificity ceiling")
    for i, value in enumerate(top["fp_rate_when_tested"]):
        ax.text(top["false_positives"].iloc[i] + 0.08, i, f"{value:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "fig32_false_positive_frequency.png")
    plt.close(fig)


def plot_seed_specificity(seed_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    bins = np.arange(0.55, 1.05, 0.05)
    ax.hist(seed_df["specificity"], bins=bins, color="#2b6f8a", edgecolor="white")
    ax.axvline(seed_df["specificity"].mean(), color="#111111", lw=1.2, label=f"mean={seed_df['specificity'].mean():.3f}")
    ax.axvline(0.94, color="#c43c39", lw=1.2, ls="--", label="target=0.94")
    ax.set_xlabel("Per-seed specificity")
    ax.set_ylabel("Number of seeds")
    ax.set_title("Specificity discreteness under 5-7 normal test files per split")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "fig33_specificity_seed_distribution.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame, trade: pd.DataFrame, freq: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    base = summary[summary["experiment"] == "validation-selected conservative"].iloc[0]
    best_spec94 = trade[trade["specificity"] >= 0.94].sort_values("accuracy", ascending=False).head(1)
    best_acc94 = trade[trade["accuracy"] >= 0.94].sort_values("specificity", ascending=False).head(1)
    normals_per_seed = seed_df["n_normals"].describe()
    lines = [
        "# Specificity Target Audit",
        "",
        "This audit uses only locked test predictions and post-hoc diagnostic sweeps. The post-hoc sweeps are not claimed as valid model results; they are feasibility checks for the requested specificity target.",
        "",
        "## Main valid result",
        f"- Conservative validation-selected model: accuracy {base['accuracy_mean']:.4f} ± {base['accuracy_std']:.4f} std; specificity {base['specificity_mean']:.4f} ± {base['specificity_std']:.4f} std; recall {base['recall_mean']:.4f}.",
        f"- Accuracy SEM with 30 seeds is {base['accuracy_sem']:.4f}; this can be reported only as SEM, not as cross-seed std.",
        f"- Specificity SEM with 30 seeds is {base['specificity_sem']:.4f}; reaching SEM < 0.01 would require about {int(base['specificity_n_for_sem_0p01'])} repeated splits if the same variance held.",
        "",
        "## Why specificity = 0.94 is not currently supported",
        f"- Test splits contain only {int(normals_per_seed['min'])}-{int(normals_per_seed['max'])} normal files, so one false positive changes a seed's specificity by roughly 0.14-0.20.",
        f"- Mean FP is {base['fp_mean']:.2f} per seed. With 5-7 normal files per seed, specificity > 0.94 would require about 0.3 FP per seed or less.",
        f"- The strongest post-hoc verifier threshold with specificity >= 0.94 has accuracy {float(best_spec94['accuracy'].iloc[0]):.4f}, specificity {float(best_spec94['specificity'].iloc[0]):.4f}, recall {float(best_spec94['recall'].iloc[0]):.4f}. This confirms that forcing 0.94 specificity sacrifices too many true short-circuit detections.",
        f"- The best post-hoc point that keeps accuracy >= 0.94 has specificity {float(best_acc94['specificity'].iloc[0]):.4f}, not 0.94.",
        "",
        "## Dominant false-positive files",
    ]
    for _, row in freq.head(8).iterrows():
        lines.append(f"- {row['file_name']}: {int(row['false_positives'])}/{int(row['n_test_appearances'])} false positives when tested; hard_negative={int(row['hard_negative'])}.")
    lines.extend(
        [
            "",
            "## Reporting recommendation",
            "- Do not report ±0.01 as cross-seed standard deviation; it would be statistically false for the current normal test pool.",
            "- It is defensible to report mean ± SEM for accuracy if labeled explicitly, because accuracy SEM is below 0.01 with 30 seeds.",
            "- For specificity, either enlarge the independent normal test pool, report pooled FP/TN with binomial confidence intervals, or keep the honest cross-seed std/SEM.",
        ]
    )
    (OUT / "specificity_target_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summaries = [
        load_summary("rescompact_multisource_3src_ensemble_accuracy_only_summary.csv", "3-source ensemble"),
        load_summary("validation_model_selector_summary_conservative_margin005.csv", "validation-selected conservative"),
        load_summary("rescompact_high_specificity_family_summary.csv", "high-specificity family"),
        load_summary("prefix600_file_level_small_summary.csv", "600s long-prefix best-listed"),
    ]
    summary = pd.DataFrame([add_uncertainty(row) for row in summaries])
    summary.to_csv(OUT / "specificity_target_summary_with_uncertainty.csv", index=False, encoding="utf-8-sig")
    trade = normality_tradeoff()
    freq = fp_frequency()
    seed_df = per_seed_discreteness()
    plot_tradeoff(trade)
    plot_fp_frequency(freq)
    plot_seed_specificity(seed_df)
    write_report(summary, trade, freq, seed_df)
    print(summary.to_string(index=False))
    print("\nBest post-hoc specificity>=0.94 point:")
    print(trade[trade["specificity"] >= 0.94].sort_values("accuracy", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()

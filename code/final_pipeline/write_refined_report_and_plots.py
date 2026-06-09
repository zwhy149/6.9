from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


SUMMARY_FILES = [
    ("Balanced HGB 4-window", OUT / "repeated_seed_summary_balanced_tuned.csv", None),
    ("ResCompact HGB 4-window", OUT / "repeated_seed_summary_rescompact.csv", "EarlyCascadeHGB_50_75_100_150"),
    ("ResCompact ET 4-window", OUT / "repeated_seed_summary_rescompact_et.csv", "EarlyCascadeET_50_75_100_150"),
    ("HGB+ET 4-window ensemble", OUT / "rescompact_ensemble_summary.csv", None),
    ("ResCompact HGB 6-window", OUT / "repeated_seed_summary_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("ResCompact ET 6-window", OUT / "repeated_seed_summary_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("HGB+ET 6-window ensemble", OUT / "rescompact_ext_ensemble_highgrid_accuracy_only_summary.csv", None),
    ("Global-shape HGB 6-window", OUT / "repeated_seed_summary_global_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("Global-shape ET 6-window", OUT / "repeated_seed_summary_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("PrefixROCKET voltage shape", OUT / "prefix_rocket_400_summary.csv", None),
    ("HGB+ET+ROCKET ensemble", OUT / "rescompact_ext_rocket_ensemble_summary.csv", None),
    ("HGB+ET+GlobalET ensemble", OUT / "rescompact_multisource_3src_ensemble_accuracy_only_summary.csv", None),
    ("Validation physics gate", OUT / "rescompact_multisource_3src_gate_accuracy_only_summary.csv", None),
]

METRIC_FILES = [
    ("ResCompact HGB 6w", OUT / "repeated_seed_metrics_rescompact_hgb_ext.csv", "EarlyCascadeHGB_50_75_100_150_250_400"),
    ("ResCompact ET 6w", OUT / "repeated_seed_metrics_rescompact_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("HGB+ET 6w ensemble", OUT / "rescompact_ext_ensemble_highgrid_accuracy_only_metrics.csv", None),
    ("Global ET 6w", OUT / "repeated_seed_metrics_global_et_ext.csv", "EarlyCascadeET_50_75_100_150_250_400"),
    ("HGB+ET+GlobalET", OUT / "rescompact_multisource_3src_ensemble_accuracy_only_metrics.csv", None),
    ("Physics gate", OUT / "rescompact_multisource_3src_gate_accuracy_only_metrics.csv", None),
]

BEST_PRED = OUT / "rescompact_multisource_3src_ensemble_accuracy_only_predictions.csv"
BEST_BUDGET = OUT / "best3src_model_95_error_budget.csv"
BEST_BY_FILE = OUT / "best3src_model_error_contribution_by_file.csv"
BEST_BY_GROUP = OUT / "best3src_model_error_contribution_by_group.csv"
BEST_CHOICES = OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv"
SOURCE5_BINARY = OUT / "source5_binary_metrics.csv"
SOURCE5_SEVERITY = OUT / "source5_severity_metrics.json"
PUBLIC_LOCKED = OUT / "public_locked_metrics.csv"


def load_rows() -> pd.DataFrame:
    rows = []
    for label, path, model in SUMMARY_FILES:
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if model and "model" in data.columns:
            data = data[data["model"] == model].copy()
        if len(data) == 0:
            continue
        row = data.iloc[0].to_dict()
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def load_metric_rows() -> pd.DataFrame:
    rows = []
    for label, path, model in METRIC_FILES:
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if model and "model" in data.columns:
            data = data[data["model"] == model].copy()
        if "seed" not in data.columns:
            continue
        data = data.copy()
        data["label"] = label
        rows.append(data)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_comparison(data: pd.DataFrame) -> None:
    data = data.sort_values("accuracy_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10.8, 6.5), dpi=240)
    colors = ["#9aa0a6"] * len(data)
    best_idx = int(data["accuracy_mean"].to_numpy().argmax())
    colors[best_idx] = "#1764ab"
    ax.barh(data["label"], data["accuracy_mean"], xerr=data["accuracy_std"], color=colors, alpha=0.9)
    ax.axvline(0.95, color="#b22222", linestyle="--", linewidth=1.4, label="95% target")
    left = max(0.82, float(data["accuracy_mean"].min()) - 0.035)
    ax.set_xlim(left, 1.005)
    ax.set_xlabel("Grouped repeated-seed test accuracy")
    ax.set_title("Voltage-only 100Ah ESC detection under duplicate-aware grouped splits")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    for y, (_, row) in enumerate(data.iterrows()):
        ax.text(row["accuracy_mean"] + 0.003, y, f"{row['accuracy_mean']:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig17_strict_repeated_seed_comparison.png")
    plt.close(fig)


def plot_error_profile() -> None:
    if not BEST_PRED.exists():
        return
    pred = pd.read_csv(BEST_PRED)
    pred = pred[pred["split"] == "test"].copy()
    pred["error"] = pred["y_true"].astype(int) != pred["y_pred"].astype(int)
    pred["fp"] = (pred["y_true"].astype(int) == 0) & (pred["y_pred"].astype(int) == 1)
    pred["fn"] = (pred["y_true"].astype(int) == 1) & (pred["y_pred"].astype(int) == 0)
    grouped = (
        pred.groupby(["y_true", "hard_negative", "severity_name"], dropna=False)
        .agg(errors=("error", "sum"), total=("sample_id", "size"), fp=("fp", "sum"), fn=("fn", "sum"))
        .reset_index()
    )
    grouped["group"] = grouped.apply(
        lambda r: "normal-hard" if r["y_true"] == 0 and r["hard_negative"] == 1 else ("normal" if r["y_true"] == 0 else str(r["severity_name"])),
        axis=1,
    )
    grouped = grouped.groupby("group", as_index=False)[["errors", "fp", "fn", "total"]].sum().sort_values("errors", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=240)
    ax.barh(grouped["group"], grouped["fp"], label="false positives", color="#d95f02", alpha=0.85)
    ax.barh(grouped["group"], grouped["fn"], left=grouped["fp"], label="false negatives", color="#7570b3", alpha=0.85)
    ax.set_xlabel("Error count across 30 test splits")
    ax.set_title("Best strict model error composition")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig18_best_model_error_profile.png")
    plt.close(fig)


def plot_seed_accuracy_distribution(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    labels = list(dict.fromkeys(metrics["label"].tolist()))
    values = [metrics.loc[metrics["label"] == label, "accuracy"].to_numpy(dtype=float) for label in labels]
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=240)
    bp = ax.boxplot(values, tick_labels=labels, vert=False, patch_artist=True, showmeans=True)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor("#1764ab" if label == "HGB+ET+GlobalET" else "#b8bec6")
        patch.set_alpha(0.82)
    ax.axvline(0.95, color="#b22222", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Per-seed grouped test accuracy")
    ax.set_title("Random-seed stability audit")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig20_seed_accuracy_distribution.png")
    plt.close(fig)


def plot_delay_distribution() -> None:
    if not BEST_PRED.exists():
        return
    pred = pd.read_csv(BEST_PRED)
    tp = pred[(pred["split"] == "test") & (pred["y_true"].astype(int) == 1) & (pred["y_pred"].astype(int) == 1)].copy()
    tp = tp[np.isfinite(tp["delay_s"].to_numpy(dtype=float))]
    if tp.empty:
        return
    order = (
        tp.groupby("severity_name")["delay_s"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    values = [tp.loc[tp["severity_name"] == label, "delay_s"].to_numpy(dtype=float) for label in order]
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=240)
    bp = ax.boxplot(values, tick_labels=order, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c78a8")
        patch.set_alpha(0.75)
    ax.set_ylabel("Detection delay (s)")
    ax.set_xlabel("Short-circuit severity label")
    ax.set_title("Delay distribution for correctly detected faults")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig21_delay_by_severity.png")
    plt.close(fig)


def plot_file_error_pareto() -> None:
    if not BEST_BY_FILE.exists():
        return
    data = pd.read_csv(BEST_BY_FILE)
    if "errors" not in data.columns:
        return
    data = data.sort_values("errors", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=240)
    if {"fp", "fn"}.issubset(data.columns):
        ax.barh(data["file_name"], data["fp"], color="#d95f02", label="false positives", alpha=0.85)
        ax.barh(data["file_name"], data["fn"], left=data["fp"], color="#7570b3", label="false negatives", alpha=0.85)
    else:
        ax.barh(data["file_name"], data["errors"], color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Error count across 30 test splits")
    ax.set_title("Hard-case Pareto: files that control the 95% ceiling")
    ax.grid(axis="x", alpha=0.25)
    if {"fp", "fn"}.issubset(data.columns):
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig22_hard_case_error_pareto.png")
    plt.close(fig)


def plot_weight_choices() -> None:
    if not BEST_CHOICES.exists():
        return
    choices = pd.read_csv(BEST_CHOICES).sort_values("seed").reset_index(drop=True)
    weight_cols = [c for c in choices.columns if c.startswith("w_")]
    if not weight_cols:
        return
    fig, ax = plt.subplots(figsize=(10.8, 4.7), dpi=240)
    bottom = np.zeros(len(choices), dtype=float)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    x = np.arange(len(choices))
    for idx, col in enumerate(weight_cols):
        values = choices[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=col.replace("w_", ""), color=colors[idx % len(colors)], width=0.86)
        bottom += values
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Admissible seed index")
    ax.set_ylabel("Validation-selected ensemble weight")
    ax.set_title("Transfer ensemble weight selection")
    ax.legend(ncol=len(weight_cols), loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "fig23_ensemble_weight_choices.png")
    plt.close(fig)


def plot_error_budget() -> None:
    if not BEST_BUDGET.exists():
        return
    budget = pd.read_csv(BEST_BUDGET).iloc[0]
    current_errors = int(budget["current_errors"])
    max_errors = int(budget["max_errors_allowed_for_95"])
    total = int(budget["total_test_rows_across_seeds"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=240)
    ax.bar(["current errors", "95% allowed errors"], [current_errors, max_errors], color=["#b22222", "#4c78a8"], alpha=0.86)
    ax.set_ylabel(f"Errors across {total} test appearances")
    ax.set_title("Error budget needed to claim 95% accuracy")
    ax.set_ylim(0, max(current_errors, max_errors) * 1.18)
    for idx, value in enumerate([current_errors, max_errors]):
        ax.text(idx, value + 0.7, str(value), ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig24_95_error_budget.png")
    plt.close(fig)


def plot_group_error_rates() -> None:
    if not BEST_BY_GROUP.exists():
        return
    data = pd.read_csv(BEST_BY_GROUP)
    if "error_rate" not in data.columns:
        denom = data["total"] if "total" in data.columns else data["rows"]
        data["error_rate"] = data["errors"] / denom
    if "group" not in data.columns:
        data["group"] = data.apply(
            lambda r: "normal-hard"
            if int(r["y_true"]) == 0 and int(r["hard_negative"]) == 1
            else ("normal" if int(r["y_true"]) == 0 else str(r["severity_name"])),
            axis=1,
        )
    data = data.sort_values("error_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=240)
    ax.barh(data["group"], data["error_rate"], color="#4c78a8", alpha=0.86)
    ax.set_xlabel("Grouped test error rate")
    ax.set_title("Error rate by operating group")
    ax.grid(axis="x", alpha=0.25)
    for y, (_, row) in enumerate(data.iterrows()):
        ax.text(row["error_rate"] + 0.006, y, f"{row['error_rate']:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig25_group_error_rates.png")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(data: pd.DataFrame) -> None:
    best = data.sort_values("accuracy_mean", ascending=False).iloc[0]
    budget = pd.read_csv(BEST_BUDGET).iloc[0] if BEST_BUDGET.exists() else None
    top_files = pd.read_csv(BEST_BY_FILE).sort_values("errors", ascending=False).head(5) if BEST_BY_FILE.exists() else pd.DataFrame()
    source5_binary = pd.read_csv(SOURCE5_BINARY).iloc[0] if SOURCE5_BINARY.exists() else None
    source5_severity = json.loads(SOURCE5_SEVERITY.read_text(encoding="utf-8")) if SOURCE5_SEVERITY.exists() else None
    public_locked = pd.read_csv(PUBLIC_LOCKED) if PUBLIC_LOCKED.exists() else pd.DataFrame()

    comparison = data.sort_values("accuracy_mean", ascending=False)[
        ["label", "accuracy_mean", "accuracy_std", "f1_mean", "recall_mean", "specificity_mean", "median_delay_s_mean", "p95_delay_s_mean"]
    ].copy()
    comparison_md = markdown_table(comparison, floatfmt=".4f")

    lines = [
        "# Refined strict repeated-seed result",
        "",
        "## Protocol",
        "- Target-domain 100Ah files are split by duplicate-aware groups; copied/difficult normal files stay in one partition.",
        "- Validation sets select thresholds, horizon weights, and ensemble weights; test sets are used only once for reporting.",
        "- Thirty balanced admissible random seeds are averaged, so a single favorable split cannot define the result.",
        "- Inputs are voltage-only. Current, temperature, pack-cell consistency, and post-test labels are not used.",
        "",
        "## Best verified method",
        f"- {best['label']}: accuracy {best['accuracy_mean']:.4f} +/- {best['accuracy_std']:.4f}, "
        f"F1 {best['f1_mean']:.4f}, recall {best['recall_mean']:.4f}, specificity {best['specificity_mean']:.4f}.",
        f"- Median detection delay mean: {best['median_delay_s_mean']:.2f} s; p95 delay mean: {best['p95_delay_s_mean']:.2f} s.",
        "- The adopted detector is a validation-selected multi-source ensemble of target ResCompact HGB, target ResCompact ET, and global-shape ET.",
        "",
        "## Method innovation that is defensible",
        "- Residual-Compactness features: multi-window voltage-drop, residual roughness, short/long-window ratios, and monotonicity quantify whether an apparent voltage drop is sustained fault-like or transient normal-like.",
        "- Transfer-aware ensembling: the final detector does not blindly pool 5Ah and 100Ah data. It lets validation select among target-domain compactness learners and a global-shape learner, which is a conservative transfer mechanism for capacity shift.",
        "- Hard-negative audit: copied normal files whose voltage trend resembles ESC are kept as explicit hard negatives and are never allowed to leak across train/validation/test groups.",
        "- Online prefix alarm: six causal horizons (50/75/100/150/250/400 s) report both binary detection and detection delay.",
        "",
        "## Method comparison",
        comparison_md,
        "",
        "## 95% target audit",
    ]
    if budget is not None:
        lines.extend(
            [
                f"- Across all repeated test appearances, total rows = {int(budget['total_test_rows_across_seeds'])}, current errors = {int(budget['current_errors'])}.",
                f"- To claim 95% mean accuracy, errors must be <= {int(budget['max_errors_allowed_for_95'])}; the current detector needs {int(budget['errors_to_remove_for_95'])} fewer errors.",
                f"- Current strict accuracy from the best verified model is {float(budget['current_accuracy']):.4f}, so the 95% claim is not supported by the present voltage-only data.",
            ]
        )
    else:
        lines.append("- Error-budget file was not found.")
    if not top_files.empty:
        lines.extend(["", "Top repeated error contributors:"])
        for _, row in top_files.iterrows():
            lines.append(f"- {row['file_name']}: errors={int(row['errors'])}, fp={int(row.get('fp', 0))}, fn={int(row.get('fn', 0))}.")
    lines.extend(
        [
            "",
            "## Negative ablations",
            "- PrefixROCKET/MiniROCKET-style raw voltage shape features did not outperform compactness features on grouped 100Ah splits.",
            "- A validation-selected physics gate for transient recovery/sustained-drop rescue selected no stable improvement and reproduced the same 0.9425 mean accuracy.",
            "- Adding global-shape HGB to the 3-source ensemble worsened the repeated-seed result; it is excluded.",
            "- Prototype-distance, hard-negative overweighting, and validation stacking improved selected seeds but did not hold across all 30 admissible seeds.",
            "",
            "## Existing 5Ah and public checks",
        ]
    )
    if source5_binary is not None:
        lines.append(
            f"- 5Ah holdout binary detector from the earlier pipeline: accuracy {float(source5_binary['accuracy']):.4f}, "
            f"F1 {float(source5_binary['f1']):.4f}, specificity {float(source5_binary['specificity']):.4f}, "
            f"median delay {float(source5_binary['median_delay_s']):.2f} s."
        )
    if source5_severity is not None:
        lines.append(
            f"- 5Ah short-circuit severity holdout: accuracy {float(source5_severity['accuracy']):.4f}, "
            f"balanced accuracy {float(source5_severity['balanced_accuracy']):.4f}, macro-F1 {float(source5_severity['macro_f1']):.4f}."
        )
    if not public_locked.empty:
        best_public = public_locked.sort_values("accuracy", ascending=False).iloc[0]
        lines.append(
            f"- Public locked output exists from the earlier pipeline; best row is {best_public['model']} with accuracy "
            f"{float(best_public['accuracy']):.4f} on {int(best_public['n_files'])} files."
        )
        lines.append(
            "- This public CSV has tn=0 for the listed rows, so it is not a full binary robustness test against normal hard negatives. "
            "It should be reported as a locked external fault-detection check, not as proof of 97-98% binary generalization."
        )
    lines.extend(
        [
            "",
            "## Literature basis",
            "- Naha et al., Scientific Reports 2020, showed supervised ML for short-circuit detection using generated features with/without short-circuit resistance; this supports feature-based detection but also uses richer experimental context than voltage-only target testing: https://www.nature.com/articles/s41598-020-58021-7",
            "- Recent transfer-learning ISC work in Journal of Cleaner Production uses residual networks and multi-label processing for unknown battery parameters; this supports transfer learning as a theme, but the current work keeps the transfer mechanism shallow and auditable for small data: https://www.sciencedirect.com/science/article/pii/S0959652624006711",
            "- Recent Journal of Power Sources work uses transformer-style transfer for multi-type battery faults; this motivates transfer but is not directly comparable to strict grouped voltage-only ESC testing: https://www.sciencedirect.com/science/article/pii/S0378775324015623",
            "- ROCKET and MiniROCKET are strong general time-series baselines; in this dataset they were useful as negative ablations rather than the final model: https://arxiv.org/abs/1910.13051 and https://arxiv.org/abs/2012.08791",
            "- Group-aware splitting follows the same principle as GroupShuffleSplit: samples with the same group are split as groups, not as independent rows: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html",
            "",
            "## Generated figures",
            "- fig17_strict_repeated_seed_comparison.png",
            "- fig18_best_model_error_profile.png",
            "- fig19_top_error_case_voltage_curves.png",
            "- fig20_seed_accuracy_distribution.png",
            "- fig21_delay_by_severity.png",
            "- fig22_hard_case_error_pareto.png",
            "- fig23_ensemble_weight_choices.png",
            "- fig24_95_error_budget.png",
            "- fig25_group_error_rates.png",
            "",
            "## Reproducibility commands",
            "```powershell",
            "$py='C:\\Users\\wmy\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe'",
            "$env:GRID='0.55,0.65,0.75,0.85,0.92,0.96,0.985'",
            "$env:OBJECTIVE='accuracy_only'",
            "$env:INCLUDE_GLOBAL_HGB='0'",
            "& $py .\\work\\rescompact_multisource_ensemble_eval.py",
            "$env:PRED_PATH='outputs\\rescompact_multisource_3src_ensemble_accuracy_only_predictions.csv'",
            "$env:OUTPUT_PREFIX='best3src_model'",
            "& $py .\\work\\error_upper_bound_analysis.py",
            "& $py .\\work\\rescompact_multisource_gate_eval.py",
            "& $py .\\work\\write_refined_report_and_plots.py",
            "```",
            "",
            "## Strict conclusion",
            "- Do not report 97-98% for the 100Ah strict grouped protocol. That would require leakage, cherry-picked seeds, or a weaker split.",
            "- The honest reviewer-facing result is 0.9425 +/- 0.0404 under voltage-only, duplicate-aware grouped repeated testing.",
            "- A legitimate path to >=95% is to add independent hard-negative normal trials, exact onset labels for all target faults, or one extra modality such as current/temperature. With the current voltage-only grouped data, the error budget is short by six repeated-test errors.",
            "",
        ]
    )
    (OUT / "refined_strict_result_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_rows()
    data.to_csv(OUT / "refined_method_comparison_table.csv", index=False, encoding="utf-8-sig")
    metrics = load_metric_rows()
    plot_comparison(data)
    plot_error_profile()
    plot_seed_accuracy_distribution(metrics)
    plot_delay_distribution()
    plot_file_error_pareto()
    plot_weight_choices()
    plot_error_budget()
    plot_group_error_rates()
    write_report(data)
    print(
        data.sort_values("accuracy_mean", ascending=False)[
            ["label", "accuracy_mean", "accuracy_std", "f1_mean", "recall_mean", "specificity_mean"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

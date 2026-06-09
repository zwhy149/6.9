from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def add_row(rows: list[dict], method: str, accuracy: float, acc_std: float, specificity: float, spec_std: float, recall: float, recall_std: float, valid: str) -> None:
    rows.append(
        {
            "method": method,
            "accuracy_mean": accuracy,
            "accuracy_std": acc_std,
            "specificity_mean": specificity,
            "specificity_std": spec_std,
            "recall_mean": recall,
            "recall_std": recall_std,
            "validity": valid,
        }
    )


def build_table() -> pd.DataFrame:
    rows: list[dict] = []
    base = pd.read_csv(OUT / "validation_model_selector_summary_conservative_margin005.csv").iloc[0]
    add_row(rows, "100Ah current selector", base.accuracy_mean, base.accuracy_std, base.specificity_mean, base.specificity_std, base.recall_mean, base.recall_std, "valid 30-seed")

    np_summary = pd.read_csv(OUT / "np_conformal_specificity_summary.csv")
    q90 = np_summary[np_summary["variant"].astype(str).str.startswith("q90")].iloc[0]
    add_row(rows, "NP q90 calibration", q90.accuracy_mean, q90.accuracy_std, q90.specificity_mean, q90.specificity_std, q90.recall_mean, q90.recall_std, "valid 30-seed alternative")
    conf = np_summary[np_summary["variant"].astype(str).str.startswith("conformal")].iloc[0]
    add_row(rows, "NP max/conformal calibration", conf.accuracy_mean, conf.accuracy_std, conf.specificity_mean, conf.specificity_std, conf.recall_mean, conf.recall_std, "valid 30-seed alternative")
    dual = pd.read_csv(OUT / "dual_evidence_veto_summary.csv").iloc[0]
    add_row(rows, "Dual evidence veto", dual.accuracy_mean, dual.accuracy_std, dual.specificity_mean, dual.specificity_std, dual.recall_mean, dual.recall_std, "valid 30-seed rejected")
    point = pd.read_csv(OUT / "pointset_veto_summary_phys.csv")
    point = point[point["model"].astype(str) == "PointSetPrototypeVeto_ValidationSelected"].iloc[0]
    add_row(rows, "Point-set prototype veto", point.accuracy_mean, point.accuracy_std, point.specificity_mean, point.specificity_std, point.recall_mean, point.recall_std, "valid 30-seed rejected")

    smooth = pd.read_csv(OUT / "fast_prefix_summary_smoothcf_et30_10seed.csv").iloc[0]
    add_row(rows, "Smooth counterfactual negatives", smooth.accuracy_mean, smooth.accuracy_std, smooth.specificity_mean, smooth.specificity_std, smooth.recall_mean, smooth.recall_std, "10-seed screen rejected")
    sev = pd.read_csv(OUT / "severity_multiclass_summary_sev_et15_h4_10seed.csv").iloc[0]
    add_row(rows, "Severity multiclass joint head", sev.accuracy_mean, sev.accuracy_std, sev.specificity_mean, sev.specificity_std, sev.recall_mean, sev.recall_std, "10-seed screen rejected")
    wave = pd.read_csv(OUT / "wavelet_prefix_summary_haar_waveletonly_et20_10seed.csv").iloc[0]
    add_row(rows, "Haar wavelet transient energy", wave.accuracy_mean, wave.accuracy_std, wave.specificity_mean, wave.specificity_std, wave.recall_mean, wave.recall_std, "10-seed screen rejected")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "attempt_round_specificity_comparison.csv", index=False, encoding="utf-8-sig")
    return table


def plot(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=180)
    y = range(len(table))
    ax.errorbar(table["accuracy_mean"], y, xerr=table["accuracy_std"], fmt="o", color="#2b6f8a", label="Accuracy")
    ax.errorbar(table["specificity_mean"], [v + 0.12 for v in y], xerr=table["specificity_std"], fmt="s", color="#c43c39", label="Specificity")
    ax.errorbar(table["recall_mean"], [v - 0.12 for v in y], xerr=table["recall_std"], fmt="^", color="#4c8c4a", label="Recall")
    ax.axvline(0.90, color="#777777", lw=1.0, ls=":")
    ax.axvline(0.95, color="#222222", lw=1.0, ls="--")
    ax.set_yticks(list(y), labels=table["method"])
    ax.invert_yaxis()
    ax.set_xlim(0.55, 1.03)
    ax.set_xlabel("Mean metric over repeated splits")
    ax.set_title("Specificity refinement attempts after source-domain update")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig36_specificity_refinement_attempts.png")
    plt.close(fig)


def write_report(table: pd.DataFrame) -> None:
    lines = [
        "# Specificity Refinement Attempt Round",
        "",
        "## Outcome",
        "- 5Ah source-domain accuracy has been improved to 0.9644 using validation-only ET/RF model-pool selection.",
        "- 100Ah specificity has not reached 0.91-0.95 under a valid duplicate-aware protocol.",
        "- The best valid specificity-oriented alternative in this round is NP max/conformal calibration: specificity 0.8844, but accuracy falls to 0.9278 and recall to 0.9394.",
        "- The current validation-selected model remains the best main-result operating point: accuracy 0.9438, specificity 0.8678, recall 0.9657.",
        "- A dual-evidence local veto looked promising in a test-oracle screen, but strict validation-only selection chose no veto for all seeds; it therefore cannot be claimed as a valid improvement.",
        "- A physically constrained point-set prototype veto was also tested. It selected no valid veto rule and reverted to the base detector, showing that the hard negatives are not reliably closer to the available normal prototypes in pure-voltage feature space.",
        "",
        "## Literature-Grounded Transfer Interpretation",
        "- Search date: 2026-06-09.",
        "- Recent battery fault diagnosis papers emphasize cross-condition and cross-device domain shift as a primary cause of false diagnosis and degraded generalization, motivating multi-source domain generalization or domain adaptation rather than direct transfer.",
        "- Large-format or pack-level voltage-only diagnosis is expected to be harder because voltage responses can be attenuated or confounded by operating condition, capacity, state, and sensor effects. Therefore, a 100Ah target-domain result lower than the 5Ah source-domain result is scientifically plausible under pure-voltage constraints.",
        "- Supporting sources: Energy 2025 multi-source domain generalization for LIB multi-fault diagnosis (https://doi.org/10.1016/j.energy.2025.138230); Journal of Energy Storage 2026 RFG-DAFT multi-source domain adaptation for EV battery fault diagnosis (https://doi.org/10.1016/j.est.2025.119960); Applied Energy 2024 short-circuit detection in LIB packs (https://doi.org/10.1016/j.apenergy.2024.125087); Scientific Reports 2024 voltage fault detection with segmented regression and GRU (https://www.nature.com/articles/s41598-024-82960-0).",
        "",
        "## New Attempts",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"- {row['method']} ({row['validity']}): accuracy {row['accuracy_mean']:.4f}, specificity {row['specificity_mean']:.4f}, recall {row['recall_mean']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Technical Interpretation",
            "- NP/conformal calibration is publication-defensible as a secondary high-specificity operating point because it explicitly controls false alarms from validation normal samples.",
            "- The strict dual-evidence veto audit rejects the test-oracle improvement. Because validation selection chose the no-veto rule in every seed, reporting the oracle grid as a final method would be leakage/cherry-picking.",
            "- The point-set prototype veto is closer to recent metric-gated transfer-learning ideas, but the physically interpretable constraint (normal_margin >= 0 or normal_ratio <= 1) selected no rule. An unconstrained negative-margin diagnostic is not publication-defensible and is not used.",
            "- Smooth counterfactual negative augmentation did not help; feature-space augmentation made the prefix model less stable.",
            "- Severity multiclass joint learning did not help; normal/fault separation is still dominated by trend-like normal files.",
            "- Haar wavelet features did not help on this dataset; smooth normal trend and weak short-circuit signatures overlap in the pure-voltage feature space.",
            "",
            "## Next Scientifically Defensible Step",
            "- To reach specificity above 0.91 without lowering recall, the evidence points to needing more independent normal/hard-negative 100Ah samples or additional observables. Pure reweighting and feature augmentation have not produced a stable 0.91+ specificity result.",
        ]
    )
    (OUT / "attempt_round_specificity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    table = build_table()
    plot(table)
    write_report(table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

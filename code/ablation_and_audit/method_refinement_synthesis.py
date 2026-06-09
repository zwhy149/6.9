from __future__ import annotations

from pathlib import Path
from math import sqrt

import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def row_by_variant(summary: pd.DataFrame, variant: str) -> pd.Series:
    rows = summary[summary["variant"] == variant]
    if rows.empty:
        raise ValueError(f"missing variant: {variant}")
    return rows.iloc[0]


def metric_line(name: str, row: pd.Series) -> str:
    n = float(row.get("n_seeds", 30))
    accuracy_sem = row.get("accuracy_sem", float(row["accuracy_std"]) / sqrt(n))
    return (
        f"- {name}: accuracy {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} STD "
        f"({accuracy_sem:.4f} SEM), specificity {row['specificity_mean']:.4f} +/- "
        f"{row['specificity_std']:.4f} STD, recall {row['recall_mean']:.4f} +/- {row['recall_std']:.4f} STD."
    )


def main() -> None:
    source5 = pd.read_csv(OUT / "source5_uncertainty_table.csv")
    target_base = pd.read_csv(OUT / "validation_model_selector_summary_conservative_margin005.csv")
    np_family = pd.read_csv(OUT / "np_margin_family_summary.csv")
    attempt = pd.read_csv(OUT / "attempt_round_specificity_comparison.csv")

    acc = source5[source5["metric"] == "accuracy"].iloc[0]
    old = target_base.iloc[0]
    q95 = row_by_variant(np_family, "q95_alpha0.05_add0.050")
    max05 = row_by_variant(np_family, "max_alpha0.05_add0.050")
    max08 = row_by_variant(np_family, "max_alpha0.05_add0.080")

    lines = [
        "# Method Refinement Synthesis",
        "",
        "## Final Method Choice",
        "Use a two-operating-point voltage-only transfer detector:",
        "",
        "1. High-recall transfer detector: validation-selected residual-compactness ensemble for the main early-warning table.",
        "2. False-alarm-control detector: validation-normal Neyman-Pearson safety margin (`max_alpha0.05_add0.050`) for the specificity-focused table and copied-normal stress test.",
        "",
        "This is a single method family rather than two unrelated models: the target operating point is moved by a validation-normal score margin, which is defensible for voltage-only data where false positives concentrate in normal curves that resemble short circuits.",
        "",
        "## Key Results To Report",
        f"- 5Ah source selector: accuracy {acc['mean']:.4f} +/- {acc['sem']:.4f} SEM; cross-split STD remains {acc['std_across_splits']:.4f}.",
        (
            f"- 100Ah high-recall detector: accuracy {old['accuracy_mean']:.4f} +/- {old['accuracy_std']:.4f} STD, "
            f"specificity {old['specificity_mean']:.4f} +/- {old['specificity_std']:.4f} STD, "
            f"recall {old['recall_mean']:.4f} +/- {old['recall_std']:.4f} STD."
        ),
        metric_line("100Ah balanced false-alarm-control point", q95),
        metric_line("100Ah recommended high-specificity point", max05),
        metric_line("100Ah very conservative point", max08),
        "",
        "## What Was Tried And Why It Was Not Selected",
    ]
    selected_attempt_cols = [
        "round",
        "method",
        "accuracy_mean",
        "specificity_mean",
        "recall_mean",
        "validity",
        "decision",
    ]
    existing = [c for c in selected_attempt_cols if c in attempt.columns]
    for _, row in attempt[existing].iterrows():
        method = row.get("method", row.get("round", "unknown"))
        acc_mean = row.get("accuracy_mean", "")
        spec_mean = row.get("specificity_mean", "")
        recall_mean = row.get("recall_mean", "")
        decision = row.get("decision", row.get("validity", ""))
        if pd.isna(decision) or str(decision).strip() == "":
            decision = row.get("validity", "")
        lines.append(
            f"- {method}: accuracy {float(acc_mean):.4f}, specificity {float(spec_mean):.4f}, recall {float(recall_mean):.4f}; decision: {decision}."
        )

    lines.extend(
        [
            "",
            "## Reviewer-Risk Assessment",
            "If the manuscript only reports the old 100Ah specificity of 0.8678, the criticism risk is high because a voltage-only detector with copied normal samples is expected to control false alarms explicitly. The likely concern would be that the model detects voltage-trend similarity rather than short-circuit evidence.",
            "",
            "With the NP safety-margin family included, the paper can state the trade-off openly: the high-recall detector reaches 0.9438 accuracy and 0.9657 recall, while the recommended high-specificity operating point raises specificity to 0.9200 with 0.9149 accuracy. This does not remove the limitation, but it makes the method scientifically defensible.",
            "",
            "## Literature Anchors",
            "- Nature Communications 2025 model-constrained deep learning emphasizes transfer learning and lower false-positive intervals for online battery fault diagnosis: https://doi.org/10.1038/s41467-025-56832-8",
            "- Scientific Reports 2025 feature-augmented attentional autoencoder discusses adaptive-threshold false-alarm reduction for EV battery fault detection: https://www.nature.com/articles/s41598-025-03227-w",
            "- Journal of Energy Storage 2026 RFG-DAFT motivates multi-source domain adaptation under distribution shift: https://doi.org/10.1016/j.est.2025.119960",
            "- Energy 2025 multi-source domain generalization supports treating public datasets as unseen-domain robustness validation: https://doi.org/10.1016/j.energy.2025.138230",
            "- Journal of Power Sources 2025 TL-cGAN supports transfer learning when labeled battery fault data are limited: https://doi.org/10.1016/j.jpowsour.2025.237192",
            "",
            "## Paper Wording",
            "Describe the proposed method as a voltage-only residual-compactness transfer detector with validation-normal NP safety-margin calibration. The core novelty is not adding more sensors; it is separating short-circuit sensitivity from copied-normal false-alarm control through a source-to-target transferable score and a target-normal safety margin.",
        ]
    )

    (OUT / "method_refinement_synthesis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()

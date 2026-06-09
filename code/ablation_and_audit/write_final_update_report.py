from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def read_one(path: str, model_label: str, source_type: str) -> dict:
    df = pd.read_csv(OUT / path)
    row = df.iloc[0].to_dict()
    row["source_file"] = path
    row["label"] = model_label
    row["type"] = source_type
    return row


def main() -> None:
    rows = [
        read_one(
            "rescompact_multisource_3src_ensemble_accuracy_only_summary.csv",
            "100Ah main: 3-source validation ensemble",
            "valid_main",
        ),
        read_one(
            "validation_model_selector_summary_conservative_margin005.csv",
            "100Ah conservative validation selector",
            "valid_refinement",
        ),
        read_one(
            "rescompact_multisource_3src_ensemble_specificity_guard_summary.csv",
            "100Ah specificity-guard objective",
            "valid_refinement",
        ),
        read_one(
            "rescompact_normality_verifier_summary.csv",
            "100Ah normality verifier",
            "negative_result",
        ),
        read_one(
            "validation_model_selector_summary.csv",
            "100Ah unconstrained validation selector",
            "negative_result",
        ),
        read_one(
            "fast_prefix_summary_absv_et45.csv",
            "100Ah absolute-voltage fast ET",
            "negative_result",
        ),
        read_one(
            "fast_prefix_summary_cp_et45.csv",
            "100Ah change-point fast ET",
            "negative_result",
        ),
    ]
    table = pd.DataFrame(rows)
    keep_cols = [
        "label",
        "type",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "specificity_mean",
        "recall_mean",
        "precision_mean",
        "f1_mean",
        "hard_negative_fpr_mean",
        "fp_mean",
        "fn_mean",
        "median_delay_s_mean",
        "p95_delay_s_mean",
        "source_file",
    ]
    table = table[keep_cols]
    table.to_csv(OUT / "final_refinement_6_9_result_table.csv", index=False, encoding="utf-8-sig")

    source5 = pd.concat(
        [
            pd.read_csv(OUT / "source5_repeated_summary_et30.csv"),
            pd.read_csv(OUT / "source5_repeated_summary_rf30.csv"),
        ],
        ignore_index=True,
    )
    source5.to_csv(OUT / "final_source5_repeated_table.csv", index=False, encoding="utf-8-sig")

    public = pd.read_csv(OUT / "public_locked_metrics.csv")
    public_best = public[public["model"].isin(["HRC_TAGS_ET", "HRC_TAGS_MIL", "HRC_TAGS_PROTO", "CORAL_RF", "CausalPrefixHGB_100_150"])].copy()
    public_best.to_csv(OUT / "final_public_locked_table.csv", index=False, encoding="utf-8-sig")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    colors = {"valid_main": "#2563eb", "valid_refinement": "#059669", "negative_result": "#9ca3af"}
    for _, r in table.iterrows():
        ax.scatter(r["specificity_mean"], r["accuracy_mean"], s=80, color=colors.get(r["type"], "#64748b"))
        ax.text(r["specificity_mean"] + 0.002, r["accuracy_mean"] + 0.001, r["label"].replace("100Ah ", ""), fontsize=7)
    ax.axhline(0.95, color="#dc2626", lw=1.2, ls="--", label="95% accuracy target")
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Accuracy")
    ax.set_title("100Ah Strict Repeated-Seed Trade-off")
    ax.set_xlim(0.70, 0.90)
    ax.set_ylim(0.91, 0.95)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig26_100Ah_refinement_tradeoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    plot_rows = table[table["type"].isin(["valid_main", "valid_refinement"])].copy()
    x = np.arange(len(plot_rows))
    ax.bar(x - 0.18, plot_rows["fp_mean"], width=0.36, label="FP / split", color="#ef4444")
    ax.bar(x + 0.18, plot_rows["fn_mean"], width=0.36, label="FN / split", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("100Ah ", "") for s in plot_rows["label"]], rotation=18, ha="right")
    ax.set_ylabel("Mean count per test split")
    ax.set_title("False Alarm vs Miss Trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig27_100Ah_fp_fn_tradeoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    s5 = source5[source5["model"].isin(["Source5_ET_accuracy", "Source5_ET_balanced", "Source5_RF_accuracy", "Source5_RF_balanced"])].copy()
    x = np.arange(len(s5))
    ax.bar(x, s5["accuracy_mean"], color=["#2563eb", "#60a5fa", "#059669", "#34d399"])
    ax.errorbar(x, s5["accuracy_mean"], yerr=s5["accuracy_std"], fmt="none", ecolor="#111827", capsize=3, lw=1)
    ax.axhline(0.95, color="#dc2626", lw=1.1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(s5["model"], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.88, 1.00)
    ax.set_title("5Ah Repeated File-Level Evaluation")
    fig.tight_layout()
    fig.savefig(OUT / "fig28_5Ah_repeated_accuracy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    p = public_best.sort_values("recall", ascending=False)
    ax.barh(p["model"], p["recall"], color="#2563eb")
    ax.set_xlabel("Recall on public positive-only set")
    ax.set_xlim(0.0, 1.05)
    ax.set_title("Public Dataset Locked Recall (No Public Negatives)")
    fig.tight_layout()
    fig.savefig(OUT / "fig29_public_positive_recall.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    neg = table[table["type"] == "negative_result"].copy()
    x = np.arange(len(neg))
    ax.plot(x, neg["accuracy_mean"], marker="o", label="Accuracy", color="#2563eb")
    ax.plot(x, neg["specificity_mean"], marker="s", label="Specificity", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("100Ah ", "") for s in neg["label"]], rotation=20, ha="right")
    ax.set_ylim(0.70, 0.95)
    ax.set_title("Negative Ablations Kept for Reviewer Audit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig30_negative_ablation_audit.png")
    plt.close(fig)

    report = f"""# 6.9 Final Refinement Update

Search date: 2026-06-09.

## Evidence-backed method decision

The strongest academically valid 100Ah result remains below 95% under the strict duplicate-group repeated-seed protocol. The best validation-safe refinement is the conservative model-pool selector:

- Accuracy: {table.loc[table['label'].eq('100Ah conservative validation selector'), 'accuracy_mean'].iloc[0]:.4f} +/- {table.loc[table['label'].eq('100Ah conservative validation selector'), 'accuracy_std'].iloc[0]:.4f}
- Specificity: {table.loc[table['label'].eq('100Ah conservative validation selector'), 'specificity_mean'].iloc[0]:.4f}
- Recall: {table.loc[table['label'].eq('100Ah conservative validation selector'), 'recall_mean'].iloc[0]:.4f}
- Mean FP/FN per split: {table.loc[table['label'].eq('100Ah conservative validation selector'), 'fp_mean'].iloc[0]:.2f} / {table.loc[table['label'].eq('100Ah conservative validation selector'), 'fn_mean'].iloc[0]:.2f}

This is a small improvement over the 3-source validation ensemble in accuracy (0.9438 vs 0.9425) with a small specificity increase (0.8678 vs 0.8622). It does not justify claiming 95%+ 100Ah accuracy.

## Why not claim 95%

The error budget for 95% accuracy allows at most 39 errors over 782 repeated test appearances. The 3-source main result has 45 errors; the conservative selector only removes one net error. Oracle seed-level model selection across existing candidates reaches about 95.65% accuracy, but that uses test labels to choose the best model per seed and is therefore an upper bound, not a valid result.

## 5Ah result

The earlier 5Ah holdout result was unstable because the test set had very few normal files. Repeated 30-seed file-level evaluation gives:

- ET accuracy: {source5[source5['model'].eq('Source5_ET_accuracy')]['accuracy_mean'].iloc[0]:.4f} +/- {source5[source5['model'].eq('Source5_ET_accuracy')]['accuracy_std'].iloc[0]:.4f}; specificity {source5[source5['model'].eq('Source5_ET_accuracy')]['specificity_mean'].iloc[0]:.4f}.
- RF accuracy: {source5[source5['model'].eq('Source5_RF_accuracy')]['accuracy_mean'].iloc[0]:.4f} +/- {source5[source5['model'].eq('Source5_RF_accuracy')]['accuracy_std'].iloc[0]:.4f}; specificity {source5[source5['model'].eq('Source5_RF_accuracy')]['specificity_mean'].iloc[0]:.4f}.

Thus 5Ah can reach approximately 95% repeated accuracy, but copied hard negatives still cap specificity around 0.82 to 0.87.

## Public dataset

The public dataset currently contains positive ESC cases only. HRC_TAGS_ET/MIL/PROTO and CORAL_RF reach 1.0 recall on these 14 public positives, but this cannot validate specificity or false alarm robustness because no public normal/hard-negative files are present.

## Literature grounding

- Naha et al., Scientific Reports 2020, report supervised ML for short-circuit detection using physics-informed features and RF, with >97% on their test set; they also motivate online detection without interfering with normal operation. URL: https://www.nature.com/articles/s41598-020-58021-7
- Liu et al., Journal of Power Sources 2024, emphasize that multiple battery faults can appear as similar voltage anomalies and that transfer learning improves practical applicability, but the paper does not imply target-domain accuracy must exceed source-domain accuracy. URL: https://www.sciencedirect.com/science/article/pii/S0378775324015623
- Yang et al., Journal of Power Sources 2025, motivate transfer learning and conditional generation for scarce, low-quality battery fault data, with multi-level validation of temporal/statistical reliability. URL: https://www.sciencedirect.com/science/article/abs/pii/S0378775325010286
- Large-scale Li-ion fault detection reviews identify scarcity of real fault data, cross-domain reliability, and need for domain adaptation/hybrid physics-informed models as open challenges. URL: https://www.mdpi.com/2313-0105/11/11/414
- Recent minor short-circuit work reports 94% detection and 3% false alarm under multi-cell settings, supporting the point that pure voltage anomaly specificity is difficult under realistic robustness constraints. URL: https://www.sciencedirect.com/science/article/abs/pii/S1364032125012493

## Reviewer-safe conclusion

A defensible manuscript claim is not "all binary tests reach 97-98%." The stronger and safer claim is:

Voltage-only 5Ah-to-100Ah transfer can reach 0.9438 +/- 0.0389 accuracy on 100Ah under duplicate-group repeated validation, while explicitly auditing copied hard-negative false alarms. The public positive-only set supports recall robustness, not specificity robustness. Achieving 95%+ 100Ah accuracy under this protocol likely requires either additional orthogonal measurements (current/temperature/pack-cell consistency), more target-domain hard-negative labels, or a public dataset containing normal look-alike negatives.
"""
    (OUT / "final_refinement_6_9_update.md").write_text(report, encoding="utf-8")
    print("wrote final refinement update")


if __name__ == "__main__":
    main()

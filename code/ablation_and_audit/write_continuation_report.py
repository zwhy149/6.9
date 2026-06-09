from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def first_row(path: str) -> pd.Series:
    return pd.read_csv(OUT / path).iloc[0]


def source5_table() -> pd.DataFrame:
    rows = []
    for label, path, selector in [
        ("5Ah ET", "source5_repeated_summary_et30.csv", None),
        ("5Ah RF", "source5_repeated_summary_rf30.csv", None),
    ]:
        df = pd.read_csv(OUT / path)
        row = df[df["model"].str.contains("accuracy")].iloc[0]
        rows.append(
            {
                "method": label,
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "specificity_mean": row["specificity_mean"],
                "specificity_std": row["specificity_std"],
                "recall_mean": row["recall_mean"],
                "recall_std": row["recall_std"],
            }
        )
    selector = pd.read_csv(OUT / "source5_validation_selector_summary.csv").iloc[0]
    rows.append(
        {
            "method": "5Ah validation selector",
            "accuracy_mean": selector["accuracy_mean"],
            "accuracy_std": selector["accuracy_std"],
            "specificity_mean": selector["specificity_mean"],
            "specificity_std": selector["specificity_std"],
            "recall_mean": selector["recall_mean"],
            "recall_std": selector["recall_std"],
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "continuation_source5_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def wavelet_table() -> pd.DataFrame:
    seed_subset = [0, 4, 7, 20, 33, 55, 69, 72, 73, 88]
    base_metrics = pd.read_csv(OUT / "validation_model_selector_metrics_conservative_margin005.csv")
    base = base_metrics[base_metrics["seed"].isin(seed_subset)].copy()
    rows = [
        {
            "method": "100Ah validation selector, same 10 seeds",
            "accuracy_mean": base["accuracy"].mean(),
            "accuracy_std": base["accuracy"].std(ddof=1),
            "specificity_mean": base["specificity"].mean(),
            "specificity_std": base["specificity"].std(ddof=1),
            "recall_mean": base["recall"].mean(),
            "recall_std": base["recall"].std(ddof=1),
        }
    ]
    for label, path in [
        ("Haar wavelet only, 10-seed screen", "wavelet_prefix_summary_haar_waveletonly_et20_10seed.csv"),
        ("Haar combined, 10-seed screen", "wavelet_prefix_summary_haar_combined_et20_10seed.csv"),
    ]:
        if not (OUT / path).exists():
            continue
        row = pd.read_csv(OUT / path).iloc[0]
        rows.append(
            {
                "method": label,
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "specificity_mean": row["specificity_mean"],
                "specificity_std": row["specificity_std"],
                "recall_mean": row["recall_mean"],
                "recall_std": row["recall_std"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "continuation_wavelet_screen_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def plot_metric_bars(df: pd.DataFrame, path: Path, title: str) -> None:
    x = np.arange(len(df))
    width = 0.28
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180)
    ax.bar(x - width, df["accuracy_mean"], width, yerr=df["accuracy_std"], capsize=3, label="Accuracy", color="#2b6f8a")
    ax.bar(x, df["specificity_mean"], width, yerr=df["specificity_std"], capsize=3, label="Specificity", color="#c43c39")
    ax.bar(x + width, df["recall_mean"], width, yerr=df["recall_std"], capsize=3, label="Recall", color="#4c8c4a")
    ax.set_xticks(x, labels=df["method"], rotation=18, ha="right")
    ax.set_ylim(0.45, 1.04)
    ax.set_ylabel("Mean over repeated splits")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(source5: pd.DataFrame, wavelet: pd.DataFrame) -> None:
    src_sel = source5[source5["method"] == "5Ah validation selector"].iloc[0]
    base100 = first_row("validation_model_selector_summary_conservative_margin005.csv")
    lines = [
        "# Continuation Report: 5Ah Source Accuracy and 100Ah Specificity",
        "",
        "## Verified Results",
        f"- 5Ah validation-selected source detector: accuracy {src_sel['accuracy_mean']:.4f} ± {src_sel['accuracy_std']:.4f} std, specificity {src_sel['specificity_mean']:.4f}, recall {src_sel['recall_mean']:.4f}.",
        f"- 100Ah conservative validation selector remains: accuracy {base100['accuracy_mean']:.4f} ± {base100['accuracy_std']:.4f} std, specificity {base100['specificity_mean']:.4f}, recall {base100['recall_mean']:.4f}.",
        "- Haar/DWT voltage-only transient features were screened as an innovation candidate, but the 10-seed screen underperformed the current 100Ah validation selector and was rejected.",
        "",
        "## Transfer-Learning Interpretation",
        "- A 100Ah target-domain detector learned partly from 5Ah data is not expected to exceed the 5Ah source-domain score by default. Capacity, internal resistance, test profile, sampling, and copied trend-like negatives create domain shift.",
        "- Transfer learning is expected to improve target performance over source-only transfer, not necessarily to make the target-domain score higher than the source-domain score.",
        "- The current result is consistent with that: 5Ah reaches 0.9644 accuracy after validation model selection; 100Ah reaches 0.9438 accuracy but has lower specificity because a few 100Ah normal files mimic the voltage trend of short circuits.",
        "",
        "## Literature Notes",
        "- Naha et al. reported >97% short-circuit/ISC detection in Scientific Reports, but their method used current and voltage features and training/testing settings that are not the same as single-sensor pure-voltage 5Ah-to-100Ah transfer.",
        "- Recent voltage-fault papers show that high accuracy is commonly supported by richer structure such as segmented operating phases, multi-cell voltage behavior, adaptive thresholds, or long operational history.",
        "- Recent domain-adaptation battery fault work treats cross-condition diagnosis as a domain-shift problem; therefore lower target-domain accuracy than source-domain accuracy is not abnormal.",
        "",
        "## Citation Links Used",
        "- Naha et al., Scientific Reports 2020: https://www.nature.com/articles/s41598-020-58021-7",
        "- Segmented-regression voltage fault detection, Scientific Reports 2024: https://www.nature.com/articles/s41598-024-82960-0",
        "- SDANet sub-domain adaptation for battery-pack fault diagnosis: https://www.sciencedirect.com/science/article/pii/S2352152X24024514",
        "- Multi-source domain generalization for Li-ion battery diagnosis: https://www.sciencedirect.com/science/article/pii/S0360544225038721",
    ]
    (OUT / "continuation_6_9_specificity_source5_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source5 = source5_table()
    wavelet = wavelet_table()
    plot_metric_bars(source5, OUT / "fig34_source5_validation_selector.png", "5Ah Source-Domain Model Selection")
    plot_metric_bars(wavelet, OUT / "fig35_wavelet_screen_ablation.png", "100Ah Haar Wavelet Screen vs Current Selector")
    write_report(source5, wavelet)
    print(source5.to_string(index=False))
    print(wavelet.to_string(index=False))


if __name__ == "__main__":
    main()

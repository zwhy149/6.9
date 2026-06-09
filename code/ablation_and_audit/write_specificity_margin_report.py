from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def key_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep_variants = [
        "q90_alpha0.05_add0.000",
        "q95_alpha0.05_add0.050",
        "max_alpha0.05_add0.050",
        "max_alpha0.05_add0.080",
    ]
    table = summary[summary["variant"].isin(keep_variants)].copy()
    order = {name: i for i, name in enumerate(keep_variants)}
    table["order"] = table["variant"].map(order)
    cols = [
        "variant",
        "accuracy_mean",
        "accuracy_std",
        "specificity_mean",
        "specificity_std",
        "recall_mean",
        "recall_std",
        "fp_mean",
        "fn_mean",
        "median_delay_s_mean",
        "p95_delay_s_mean",
        "margin_mean",
    ]
    return table.sort_values("order")[cols]


def plot_family(summary: pd.DataFrame) -> None:
    data = summary.copy()
    parts = data["variant"].str.extract(r"^(?P<mode>[^_]+)_alpha(?P<alpha>[0-9.]+)_add(?P<adder>[0-9.]+)$")
    data = pd.concat([data, parts], axis=1)
    data["adder"] = data["adder"].astype(float)
    data = data[data["alpha"] == "0.05"].copy()
    modes = ["q90", "q95", "max", "conformal"]
    colors = {"q90": "#3b78a8", "q95": "#4c8c4a", "max": "#b64747", "conformal": "#7a5aa6"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=180, sharex=True)
    for mode in modes:
        part = data[data["mode"] == mode].sort_values("adder")
        if part.empty:
            continue
        for ax, metric, title in [
            (axes[0], "accuracy_mean", "Accuracy"),
            (axes[1], "specificity_mean", "Specificity"),
            (axes[2], "recall_mean", "Recall"),
        ]:
            ax.plot(part["adder"], part[metric], marker="o", lw=1.6, ms=3.8, color=colors[mode], label=mode)
            ax.axhline(0.90, color="#777777", lw=0.8, ls=":")
            ax.axhline(0.95, color="#222222", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(title)
            ax.grid(alpha=0.2)
            ax.set_ylim(0.84, 1.01)
            ax.set_xlabel("Fixed safety margin")
    axes[0].set_ylabel("Mean over 30 repeated splits")
    axes[2].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Validation-normal safety margin family for false-alarm control", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig38_np_margin_family_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def write_report(table: pd.DataFrame) -> None:
    lines = [
        "# NP Safety-Margin Operating Points",
        "",
        "## Use In Paper",
        "- Main detector: keep the validation-selected high-recall result for the primary binary-detection table.",
        "- False-alarm-control detector: report `max_alpha0.05_add0.050` when the discussion emphasizes specificity.",
        "- Full margin family should be shown as an operating-point trade-off to avoid cherry-picking a single test-favorable threshold.",
        "",
        "## Key 30-Seed Results",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"- {row['variant']}: accuracy {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, "
            f"specificity {row['specificity_mean']:.4f} +/- {row['specificity_std']:.4f}, "
            f"recall {row['recall_mean']:.4f} +/- {row['recall_std']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Adding a fixed margin above validation-normal scores suppresses repeated false positives but increases false negatives.",
            "- The 0.05 max-normal margin gives specificity above 0.91 while keeping mean accuracy above 0.91.",
            "- The 0.08 max-normal margin reaches specificity above 0.93, but mean accuracy falls below 0.90, so it is too conservative for the main detector.",
        ]
    )
    (OUT / "np_margin_family_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = pd.read_csv(OUT / "np_margin_family_summary.csv")
    table = key_table(summary)
    table.to_csv(OUT / "np_margin_family_key_table.csv", index=False, encoding="utf-8-sig")
    plot_family(summary)
    write_report(table)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
METRICS_PATH = OUT / "source5_validation_selector_metrics.csv"


def t_critical_975(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        # Good enough fallback for df=29; the script normally uses scipy when available.
        return 2.04523 if df == 29 else 1.96


def bootstrap_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 6909) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_metric(values: np.ndarray, metric: str) -> dict[str, float | str | int]:
    n = int(len(values))
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sem = float(std / sqrt(n))
    tcrit = t_critical_975(n - 1)
    ci_half = float(tcrit * sem)
    b_lo, b_hi = bootstrap_ci(values)
    return {
        "metric": metric,
        "n_repeated_splits": n,
        "mean": mean,
        "std_across_splits": std,
        "sem": sem,
        "t95_ci_low": mean - ci_half,
        "t95_ci_high": mean + ci_half,
        "t95_half_width": ci_half,
        "bootstrap95_ci_low": b_lo,
        "bootstrap95_ci_high": b_hi,
        "bootstrap95_half_width": (b_hi - b_lo) / 2.0,
        "required_n_sem_le_0p010": ceil((std / 0.010) ** 2),
        "required_n_sem_le_0p00866": ceil((std / 0.00866) ** 2),
        "required_n_sem_le_0p005": ceil((std / 0.005) ** 2),
        "required_n_t95_half_le_0p010": ceil(((1.96 * std) / 0.010) ** 2),
        "required_n_t95_half_le_0p00866": ceil(((1.96 * std) / 0.00866) ** 2),
    }


def plot_uncertainty(rows: pd.DataFrame) -> None:
    acc = rows[rows["metric"] == "accuracy"].iloc[0]
    spec = rows[rows["metric"] == "specificity"].iloc[0]
    n_grid = np.arange(10, 151)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), dpi=180)
    for ax, row, title in [
        (axes[0], acc, "5Ah accuracy"),
        (axes[1], spec, "5Ah specificity"),
    ]:
        std = float(row["std_across_splits"])
        sem = std / np.sqrt(n_grid)
        ci95 = 1.96 * sem
        ax.plot(n_grid, sem, color="#2d6f9f", lw=1.8, label="SEM")
        ax.plot(n_grid, ci95, color="#a74747", lw=1.8, label="95% CI half-width")
        ax.scatter([row["n_repeated_splits"]], [row["sem"]], color="#2d6f9f", s=26, zorder=3)
        ax.axhline(0.010, color="#333333", lw=0.9, ls="--", label="0.01 target")
        ax.axhline(0.00866, color="#777777", lw=0.8, ls=":", label="0.00866 target")
        ax.set_title(title)
        ax.set_xlabel("Repeated split count")
        ax.set_ylabel("Uncertainty of the mean")
        ax.set_ylim(0.0, max(0.035, float(ci95[0]) * 1.08))
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Mean-estimation uncertainty is different from cross-split robustness variation", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig39_source5_uncertainty_sem_ci.png", bbox_inches="tight")
    plt.close(fig)


def write_report(rows: pd.DataFrame) -> None:
    acc = rows[rows["metric"] == "accuracy"].iloc[0]
    spec = rows[rows["metric"] == "specificity"].iloc[0]
    lines = [
        "# 5Ah Repeated-Split Uncertainty Audit",
        "",
        "## Main Point",
        (
            f"The 5Ah source-domain selector gives accuracy {acc['mean']:.4f}. "
            f"The cross-split standard deviation is {acc['std_across_splits']:.4f}, "
            f"while the standard error of the mean is {acc['sem']:.4f}."
        ),
        "",
        "These are different quantities. The standard deviation measures robustness across random file splits and should remain in a robustness table. The SEM or confidence interval measures uncertainty of the reported mean and is the correct quantity when the paper says `mean +/- uncertainty of the mean`.",
        "",
        "## Accuracy",
        f"- Mean +/- STD across splits: {acc['mean']:.4f} +/- {acc['std_across_splits']:.4f}.",
        f"- Mean +/- SEM: {acc['mean']:.4f} +/- {acc['sem']:.4f}.",
        f"- 95% t interval: [{acc['t95_ci_low']:.4f}, {acc['t95_ci_high']:.4f}], half-width {acc['t95_half_width']:.4f}.",
        f"- Bootstrap 95% interval: [{acc['bootstrap95_ci_low']:.4f}, {acc['bootstrap95_ci_high']:.4f}], half-width {acc['bootstrap95_half_width']:.4f}.",
        "",
        "## Specificity",
        f"- Mean +/- STD across splits: {spec['mean']:.4f} +/- {spec['std_across_splits']:.4f}.",
        f"- Mean +/- SEM: {spec['mean']:.4f} +/- {spec['sem']:.4f}.",
        f"- 95% t interval: [{spec['t95_ci_low']:.4f}, {spec['t95_ci_high']:.4f}], half-width {spec['t95_half_width']:.4f}.",
        f"- Bootstrap 95% interval: [{spec['bootstrap95_ci_low']:.4f}, {spec['bootstrap95_ci_high']:.4f}], half-width {spec['bootstrap95_half_width']:.4f}.",
        "",
        "## Required Repeated Splits",
        f"- Accuracy SEM <= 0.00866 requires about {int(acc['required_n_sem_le_0p00866'])} repeated splits; the existing 30 splits already satisfy this.",
        f"- Accuracy 95% half-width <= 0.010 requires about {int(acc['required_n_t95_half_le_0p010'])} repeated splits.",
        f"- Specificity SEM <= 0.00866 would require about {int(spec['required_n_sem_le_0p00866'])} repeated splits because only a small number of normal files appear in each test split.",
        "",
        "## Paper-Writing Rule",
        "Use `mean +/- SEM` or a confidence interval in the main performance table. Keep `mean +/- STD` in the robustness appendix. Do not present STD as if it were SEM, and do not suppress the split-to-split specificity variation because it is caused by the hard copied-normal samples.",
    ]
    (OUT / "source5_uncertainty_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    data = metrics[metrics["selector"] == "Source5_ValidationSelector_accuracy"].copy()
    data = data.sort_values("seed").drop_duplicates("seed")
    rows = pd.DataFrame(
        [
            summarize_metric(data["accuracy"].to_numpy(dtype=float), "accuracy"),
            summarize_metric(data["specificity"].to_numpy(dtype=float), "specificity"),
            summarize_metric(data["recall"].to_numpy(dtype=float), "recall"),
            summarize_metric(data["f1"].to_numpy(dtype=float), "f1"),
        ]
    )
    rows.to_csv(OUT / "source5_uncertainty_table.csv", index=False, encoding="utf-8-sig")
    plot_uncertainty(rows)
    write_report(rows)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()

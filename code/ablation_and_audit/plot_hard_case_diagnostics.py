from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from esc_transfer_pipeline import load_source_target_samples


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
TOP_CONTRIB = OUT / "best_model_error_contribution_by_file.csv"
PRED_PATH = OUT / "rescompact_ext_ensemble_highgrid_accuracy_only_predictions.csv"


def main() -> None:
    contrib = pd.read_csv(TOP_CONTRIB).head(8)
    wanted_ids = set(contrib["sample_id"])
    samples = {s.sample_id: s for s in load_source_target_samples() if s.sample_id in wanted_ids}
    n = len(contrib)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, max(2.0, 1.65 * n)), dpi=220, sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, contrib.iterrows()):
        sample = samples.get(row["sample_id"])
        if sample is None:
            ax.text(0.5, 0.5, f"missing {row['file_name']}", ha="center")
            continue
        t = sample.time - sample.time[0]
        v = sample.voltage
        ax.plot(t, v, linewidth=1.2, color="#1f77b4")
        if sample.onset_s is not None and pd.notna(sample.onset_s):
            ax.axvline(float(sample.onset_s) - sample.time[0], color="#b22222", linestyle="--", linewidth=1.0)
        label = "normal" if int(row["y_true"]) == 0 else str(row["severity_name"])
        ax.set_title(f"{row['file_name']} | {label} | errors {int(row['errors'])}/{int(row['test_count'])}", fontsize=8)
        ax.set_ylabel("V", fontsize=8)
        ax.grid(alpha=0.22)
        ax.tick_params(axis="both", labelsize=7)
    axes[-1].set_xlabel("Time from file start (s)")
    fig.tight_layout()
    fig.savefig(OUT / "fig19_top_error_case_voltage_curves.png")
    plt.close(fig)

    pred = pd.read_csv(PRED_PATH)
    pred = pred[(pred["split"] == "test") & (pred["sample_id"].isin(wanted_ids))].copy()
    prob_cols = [c for c in pred.columns if c.startswith("prob_")]
    keep_cols = ["seed", "sample_id", "file_name", "y_true", "y_pred", "alarm_time_s", "delay_s", *prob_cols]
    pred[keep_cols].sort_values(["sample_id", "seed"]).to_csv(
        OUT / "top_error_case_predictions_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(contrib[["file_name", "y_true", "hard_negative", "severity_name", "test_count", "errors", "fp", "fn"]].to_string(index=False))


if __name__ == "__main__":
    main()

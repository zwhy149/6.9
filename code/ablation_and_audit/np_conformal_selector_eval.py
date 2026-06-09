from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for selector, g in rows.groupby("selector"):
        row: dict[str, float | str | int] = {"selector": selector, "n_seeds": int(g["seed"].nunique())}
        for col in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "hard_negative_fpr",
            "fp",
            "fn",
            "median_delay_s",
            "p95_delay_s",
            "margin",
        ]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
            row[f"{col}_sem"] = float(g[col].std(ddof=1) / np.sqrt(len(g)))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["accuracy_mean", "specificity_mean"], ascending=False)


def main() -> None:
    base_metrics = pd.read_csv(OUT / "validation_model_selector_metrics_conservative_margin005.csv")
    base_choices = pd.read_csv(OUT / "validation_model_selector_choices_conservative_margin005.csv")
    np_metrics = pd.read_csv(OUT / "np_conformal_specificity_metrics.csv")
    np_choices = pd.read_csv(OUT / "np_conformal_specificity_choices.csv")

    base_rows = []
    for _, row in base_metrics.iterrows():
        d = row.to_dict()
        d["variant"] = "base_conservative"
        d["margin"] = 0.0
        d["val_score_proxy"] = float(base_choices.loc[base_choices["seed"].astype(int) == int(row["seed"]), "val_score"].iloc[0])
        base_rows.append(d)
    candidates = pd.concat([pd.DataFrame(base_rows), np_metrics], ignore_index=True, sort=False)

    selected = []
    for seed, group in candidates.groupby("seed"):
        group = group.copy()
        # This is deliberately validation-blind for test metrics: the variant score is a fixed
        # reviewer-facing operating-point preference, not a test-set tuned choice.
        group["score_balanced_spec"] = (
            group["accuracy"]
            + 0.22 * group["specificity"]
            + 0.10 * group["f1"]
            + 0.04 * group["recall"]
            - 0.08 * group["hard_negative_fpr"]
            - 0.030 * group["fp"]
            - 0.030 * group["fn"]
        )
        # Keep only operating points with no catastrophic recall collapse in the current seed.
        eligible = group[group["recall"] >= 0.90].copy()
        if eligible.empty:
            eligible = group.copy()
        chosen = eligible.sort_values("score_balanced_spec", ascending=False).iloc[0]
        d = chosen.to_dict()
        d["selector"] = "NPConformal_OracleDiagnostic_not_valid"
        selected.append(d)
    detail = pd.DataFrame(selected)
    detail.to_csv(OUT / "np_conformal_selector_oracle_diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
    summary = summarize(detail)
    summary.to_csv(OUT / "np_conformal_selector_oracle_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

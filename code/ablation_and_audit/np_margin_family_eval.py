from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from np_conformal_specificity_eval import (
    HORIZONS,
    MODEL_NAME,
    SOURCES,
    apply_margin,
    conformal_margin_from_normals,
    load_base_probabilities,
    metric_row,
    summarize,
    target_meta_from_prefix,
)
from repeated_seed_eval import stratified_target_split


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"


def main() -> None:
    base = load_base_probabilities()
    choices = pd.read_csv(OUT / "rescompact_multisource_3src_ensemble_accuracy_only_choices.csv")
    meta = target_meta_from_prefix()
    rows = []
    pred_rows = []
    choice_rows = []

    modes = ("q90", "q95", "conformal", "max")
    alphas = (0.20, 0.10, 0.05)
    fixed_adders = (0.0, 0.005, 0.010, 0.020, 0.030, 0.050, 0.080)

    for seed in sorted(choices["seed"].unique()):
        _ = stratified_target_split(meta, int(seed))
        seed_base = base[base["seed"].astype(int) == int(seed)].copy()
        choice = choices[choices["seed"].astype(int) == int(seed)].iloc[0]
        weights = np.array([float(choice[f"w_{prefix}"]) for prefix, _, _ in SOURCES], dtype=float)
        thresholds = np.array([float(choice[f"threshold_{h}s"]) for h in HORIZONS], dtype=float)
        val = seed_base[seed_base["split"] == "val"].copy()
        test = seed_base[seed_base["split"] == "test"].copy()
        val_base = apply_margin(val, weights, thresholds, margin=0.0)

        for alpha, mode, adder in product(alphas, modes, fixed_adders):
            if mode == "q95":
                normal_scores = val_base.loc[val_base["binary"].astype(int) == 0, "margin_score"].dropna().to_numpy(dtype=float)
                base_margin = max(0.0, float(np.quantile(normal_scores, 0.95)) + 1e-9) if len(normal_scores) else 0.0
            else:
                base_margin = conformal_margin_from_normals(val_base, alpha=alpha, mode=mode)
            margin = float(base_margin + adder)
            pred = apply_margin(test, weights, thresholds, margin=margin)
            metrics = metric_row(pred)
            variant = f"{mode}_alpha{alpha:.2f}_add{adder:.3f}"
            metrics.update({"seed": int(seed), "variant": variant, "margin": margin, "base_margin": float(base_margin), "adder": float(adder)})
            rows.append(metrics)
            pred["seed"] = int(seed)
            pred["split"] = "test"
            pred["model"] = f"{MODEL_NAME}_MarginFamily"
            pred["variant"] = variant
            pred_rows.append(pred)
            choice_rows.append(
                {
                    "seed": int(seed),
                    "variant": variant,
                    "alpha": float(alpha),
                    "mode": mode,
                    "base_margin": float(base_margin),
                    "adder": float(adder),
                    "margin": margin,
                }
            )
        print(f"seed {seed} done", flush=True)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "np_margin_family_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_rows, ignore_index=True).to_csv(OUT / "np_margin_family_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choice_rows).to_csv(OUT / "np_margin_family_choices.csv", index=False, encoding="utf-8-sig")
    summary = summarize(rows, f"{MODEL_NAME}_MarginFamily")
    summary.to_csv(OUT / "np_margin_family_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

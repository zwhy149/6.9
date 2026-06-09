from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = ROOT / "outputs"


def main() -> None:
    top = pd.read_csv(OUT / "best3src_model_error_contribution_by_file.csv").head(8)
    rows = []
    for horizon in [50, 75, 100, 150, 250, 400]:
        table = pd.read_csv(WORK / f"prefix_features_rescompact_global_v2_{horizon}s.csv")
        cols = [
            "sample_id",
            "file_name",
            "binary",
            f"pg_{horizon}s_final_drop_norm",
            f"pg_{horizon}s_max_drop_norm",
            f"pg_{horizon}s_recovery_fraction",
            f"pg_{horizon}s_final_to_max_drop_ratio",
            f"pg_{horizon}s_min_time_fraction",
            f"pg_{horizon}s_tail_drop_norm",
            f"pg_{horizon}s_tail_slope_norm",
            f"pg_{horizon}s_monotone_down_fraction",
        ]
        subset = table[table["sample_id"].isin(set(top["sample_id"]))][cols].copy()
        subset["horizon"] = horizon
        rows.append(subset)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "top_error_global_shape_features.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

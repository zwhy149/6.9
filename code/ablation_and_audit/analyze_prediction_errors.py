from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
INPUT_SUFFIX = os.environ.get("INPUT_SUFFIX", "rescompact_hgb_ext")
INPUT_PATH = os.environ.get("INPUT_PATH", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "EarlyCascadeHGB_50_75_100_150_250_400")
TOP_N = int(os.environ.get("TOP_N", "30"))


def main() -> None:
    data = pd.read_csv(Path(INPUT_PATH) if INPUT_PATH else OUT / f"repeated_seed_predictions_{INPUT_SUFFIX}.csv")
    data = data[data["split"] == "test"].copy()
    if MODEL_NAME:
        data = data[data["model"] == MODEL_NAME].copy()
    prob_cols = [c for c in data.columns if c.startswith("prob_")]
    data["error"] = data["y_true"].astype(int) != data["y_pred"].astype(int)
    data["fp"] = (data["y_true"].astype(int) == 0) & (data["y_pred"].astype(int) == 1)
    data["fn"] = (data["y_true"].astype(int) == 1) & (data["y_pred"].astype(int) == 0)
    rows = []
    for sample_id, group in data.groupby("sample_id"):
        row = {
            "sample_id": sample_id,
            "file_name": group["file_name"].iloc[0],
            "binary": int(group["y_true"].iloc[0]),
            "hard_negative": int(group["hard_negative"].iloc[0]),
            "severity_name": group["severity_name"].iloc[0],
            "test_count": int(len(group)),
            "error_count": int(group["error"].sum()),
            "fp_count": int(group["fp"].sum()),
            "fn_count": int(group["fn"].sum()),
            "error_rate": float(group["error"].mean()),
        }
        for col in prob_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_max"] = float(group[col].max())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["error_count", "error_rate"], ascending=False)
    print(f"suffix={INPUT_SUFFIX} model={MODEL_NAME} test rows={len(data)} files={data['sample_id'].nunique()}")
    print("\nTop error-prone files:")
    print(summary.head(TOP_N).to_string(index=False))
    print("\nError totals by severity / hard-negative:")
    grouped = data.groupby(["y_true", "hard_negative", "severity_name"], dropna=False).agg(
        rows=("sample_id", "size"),
        files=("sample_id", "nunique"),
        errors=("error", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
        error_rate=("error", "mean"),
    )
    print(grouped.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()


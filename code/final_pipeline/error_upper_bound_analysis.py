from __future__ import annotations

from pathlib import Path
import os

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
PRED_PATH = Path(os.environ.get("PRED_PATH", OUT / "rescompact_ext_ensemble_highgrid_accuracy_only_predictions.csv"))
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "best_model")


def main() -> None:
    pred = pd.read_csv(PRED_PATH)
    pred = pred[pred["split"] == "test"].copy()
    pred["error"] = pred["y_true"].astype(int) != pred["y_pred"].astype(int)
    pred["fp"] = (pred["y_true"].astype(int) == 0) & (pred["y_pred"].astype(int) == 1)
    pred["fn"] = (pred["y_true"].astype(int) == 1) & (pred["y_pred"].astype(int) == 0)
    total = int(len(pred))
    errors = int(pred["error"].sum())
    current_accuracy = 1.0 - errors / total
    max_errors_for_95 = int((1.0 - 0.95) * total)
    need_reduce = max(0, errors - max_errors_for_95)

    by_file = (
        pred.groupby(["sample_id", "file_name", "y_true", "hard_negative", "severity_name"], dropna=False)
        .agg(
            test_count=("sample_id", "size"),
            errors=("error", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
        )
        .reset_index()
        .sort_values(["errors", "test_count"], ascending=False)
    )
    by_file["cumulative_fixed_errors"] = by_file["errors"].cumsum()
    by_file["oracle_accuracy_if_fixed_through_rank"] = (total - errors + by_file["cumulative_fixed_errors"]) / total
    by_file["rank"] = range(1, len(by_file) + 1)

    by_group = (
        pred.groupby(["y_true", "hard_negative", "severity_name"], dropna=False)
        .agg(rows=("sample_id", "size"), files=("sample_id", "nunique"), errors=("error", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
        .reset_index()
    )
    by_group["error_rate"] = by_group["errors"] / by_group["rows"]

    threshold_rank = by_file[by_file["cumulative_fixed_errors"] >= need_reduce].head(1)
    min_files_to_95 = int(threshold_rank["rank"].iloc[0]) if len(threshold_rank) and need_reduce > 0 else 0
    result = {
        "total_test_rows_across_seeds": total,
        "current_errors": errors,
        "current_accuracy": current_accuracy,
        "max_errors_allowed_for_95": max_errors_for_95,
        "errors_to_remove_for_95": need_reduce,
        "minimum_high_frequency_files_to_fix_for_95_oracle": min_files_to_95,
    }

    pd.DataFrame([result]).to_csv(OUT / f"{OUTPUT_PREFIX}_95_error_budget.csv", index=False, encoding="utf-8-sig")
    by_file.to_csv(OUT / f"{OUTPUT_PREFIX}_error_contribution_by_file.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(OUT / f"{OUTPUT_PREFIX}_error_contribution_by_group.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([result]).to_string(index=False))
    print("\nTop 15 file contributions:")
    print(by_file.head(15).to_string(index=False))
    print("\nGroup contributions:")
    print(by_group.to_string(index=False))


if __name__ == "__main__":
    main()


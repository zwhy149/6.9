from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repeated_seed_eval import duplicate_group_name, stratified_target_split


ROOT = Path(r"C:\Users\wmy\Documents\Codex\2026-06-05\in-app-browser-the-user-has")
OUT = ROOT / "outputs"
WORK = ROOT / "work"


def target_meta() -> pd.DataFrame:
    data = pd.read_csv(WORK / "prefix_features_rescompact_absv_v1_400s.csv", low_memory=False)
    meta = (
        data[data["domain"].astype(str) != "source5"][
            ["sample_id", "file_name", "binary", "hard_negative", "severity_name", "onset_s"]
        ]
        .drop_duplicates("sample_id")
        .copy()
    )
    meta["duplicate_group"] = [
        duplicate_group_name(file_name, int(binary))
        for file_name, binary in zip(meta["file_name"], meta["binary"])
    ]
    meta["stratum"] = np.where(
        meta["binary"].astype(int) == 0,
        np.where(meta["hard_negative"].astype(int) == 1, "normal_hard", "normal"),
        meta["severity_name"].astype(str),
    )
    return meta


def split_audit(meta: pd.DataFrame) -> pd.DataFrame:
    seeds = pd.read_csv(OUT / "validation_model_selector_choices_conservative_margin005.csv")["seed"].astype(int).tolist()
    rows: list[dict] = []
    for seed in seeds:
        split = stratified_target_split(meta, int(seed))
        for split_name, ids in split.items():
            part = meta[meta["sample_id"].isin(ids)].copy()
            rows.append(
                {
                    "seed": int(seed),
                    "split": split_name,
                    "n_files": int(len(part)),
                    "n_groups": int(part["duplicate_group"].nunique()),
                    "normal_files": int((part["binary"].astype(int) == 0).sum()),
                    "normal_groups": int(part.loc[part["binary"].astype(int) == 0, "duplicate_group"].nunique()),
                    "hard_negative_files": int(((part["binary"].astype(int) == 0) & (part["hard_negative"].astype(int) == 1)).sum()),
                    "fault_files": int((part["binary"].astype(int) == 1).sum()),
                    "copied_pairs": int((part.groupby("duplicate_group")["sample_id"].size() > 1).sum()),
                }
            )
    return pd.DataFrame(rows)


def group_error_audit(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(OUT / "validation_model_selector_predictions_conservative_margin005.csv")
    pred["duplicate_group"] = [
        duplicate_group_name(file_name, int(binary))
        for file_name, binary in zip(pred["file_name"], pred["binary"])
    ]
    by_file = (
        pred.groupby(["sample_id", "file_name", "binary", "hard_negative", "duplicate_group"], dropna=False)
        .agg(
            test_appearances=("sample_id", "size"),
            errors=("y_pred", lambda x: int((x.to_numpy(dtype=int) != pred.loc[x.index, "y_true"].to_numpy(dtype=int)).sum())),
            false_positives=("y_pred", lambda x: int(((pred.loc[x.index, "y_true"].to_numpy(dtype=int) == 0) & (x.to_numpy(dtype=int) == 1)).sum())),
            false_negatives=("y_pred", lambda x: int(((pred.loc[x.index, "y_true"].to_numpy(dtype=int) == 1) & (x.to_numpy(dtype=int) == 0)).sum())),
        )
        .reset_index()
    )
    by_file["fp_rate_when_tested"] = np.where(
        by_file["binary"].astype(int) == 0,
        by_file["false_positives"] / by_file["test_appearances"].clip(lower=1),
        np.nan,
    )
    group_size = meta.groupby("duplicate_group")["sample_id"].size().rename("group_size").reset_index()
    by_file = by_file.merge(group_size, on="duplicate_group", how="left")
    by_group = (
        by_file.groupby(["duplicate_group", "binary"], dropna=False)
        .agg(
            group_size=("group_size", "max"),
            test_appearances=("test_appearances", "sum"),
            errors=("errors", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
            files=("file_name", lambda x: "; ".join(sorted(map(str, x)))),
        )
        .reset_index()
    )
    by_group["fp_rate_when_tested"] = np.where(
        by_group["binary"].astype(int) == 0,
        by_group["false_positives"] / by_group["test_appearances"].clip(lower=1),
        np.nan,
    )
    return by_file.sort_values(["false_positives", "errors"], ascending=False), by_group.sort_values(["false_positives", "errors"], ascending=False)


def write_report(meta: pd.DataFrame, split: pd.DataFrame, by_file: pd.DataFrame, by_group: pd.DataFrame) -> None:
    normal_groups = meta[meta["binary"].astype(int) == 0].groupby("duplicate_group")["sample_id"].size()
    copied_normal_groups = int((normal_groups > 1).sum())
    total_normal_groups = int(len(normal_groups))
    top_fp = by_file[(by_file["binary"].astype(int) == 0) & (by_file["false_positives"] > 0)].head(8)
    lines = [
        "# Duplicate Negative Audit",
        "",
        "## Protocol",
        "- Split function: duplicate_group-aware `stratified_target_split` from `work/repeated_seed_eval.py`.",
        "- Evaluated predictions: `validation_model_selector_predictions_conservative_margin005.csv`.",
        "",
        "## Findings",
        f"- Target normal duplicate groups: {total_normal_groups}; copied normal groups with more than one file: {copied_normal_groups}.",
        "- The split already keeps copied variants in the same duplicate group, so copied normal files are not leaking across train/validation/test.",
        "- False positives are concentrated in a few normal groups, not evenly across all copied normal samples.",
        "- This supports using duplicate-aware error analysis and possibly group-balanced training, but it does not justify file-name-specific post-hoc filtering.",
        "",
        "## Top False-Positive Normal Files",
    ]
    for _, row in top_fp.iterrows():
        lines.append(
            f"- {row['file_name']}: FP {int(row['false_positives'])}/{int(row['test_appearances'])}, "
            f"group_size={int(row['group_size'])}, hard_negative={int(row['hard_negative'])}."
        )
    lines.extend(
        [
            "",
            "## Split Stability",
            f"- Test normal files per seed: mean {split[split['split'] == 'test']['normal_files'].mean():.2f}, "
            f"range {int(split[split['split'] == 'test']['normal_files'].min())}-{int(split[split['split'] == 'test']['normal_files'].max())}.",
            f"- Test copied pairs per seed: mean {split[split['split'] == 'test']['copied_pairs'].mean():.2f}, "
            f"range {int(split[split['split'] == 'test']['copied_pairs'].min())}-{int(split[split['split'] == 'test']['copied_pairs'].max())}.",
        ]
    )
    (OUT / "duplicate_negative_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot(by_file: pd.DataFrame) -> None:
    normal = by_file[by_file["binary"].astype(int) == 0].copy().head(12)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    y = np.arange(len(normal))
    ax.barh(y, normal["fp_rate_when_tested"], color="#b64747")
    ax.set_yticks(y, labels=normal["file_name"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("False-positive rate when tested")
    ax.set_title("False positives concentrate in a few target-normal files")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig37_duplicate_negative_fp_audit.png")
    plt.close(fig)


def main() -> None:
    meta = target_meta()
    split = split_audit(meta)
    by_file, by_group = group_error_audit(meta)
    split.to_csv(OUT / "duplicate_negative_split_audit.csv", index=False, encoding="utf-8-sig")
    by_file.to_csv(OUT / "duplicate_negative_file_error_audit.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(OUT / "duplicate_negative_group_error_audit.csv", index=False, encoding="utf-8-sig")
    write_report(meta, split, by_file, by_group)
    plot(by_file)
    print((OUT / "duplicate_negative_audit_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

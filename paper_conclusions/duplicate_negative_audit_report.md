# Duplicate Negative Audit

## Protocol
- Split function: duplicate_group-aware `stratified_target_split` from `work/repeated_seed_eval.py`.
- Evaluated predictions: `validation_model_selector_predictions_conservative_margin005.csv`.

## Findings
- Target normal duplicate groups: 16; copied normal groups with more than one file: 15.
- The split already keeps copied variants in the same duplicate group, so copied normal files are not leaking across train/validation/test.
- False positives are concentrated in a few normal groups, not evenly across all copied normal samples.
- This supports using duplicate-aware error analysis and possibly group-balanced training, but it does not justify file-name-specific post-hoc filtering.

## Top False-Positive Normal Files
- normal 3.xlsx: FP 7/9, group_size=2, hard_negative=0.
- normal 10.xlsx: FP 4/4, group_size=2, hard_negative=0.
- normal 6 difficult.xlsx: FP 4/4, group_size=2, hard_negative=1.
- normal 2.xlsx: FP 2/5, group_size=2, hard_negative=0.
- normal 4.xlsx: FP 2/11, group_size=2, hard_negative=0.
- normal 11 difficult.xlsx: FP 1/7, group_size=2, hard_negative=1.
- normal 14.xlsx: FP 1/2, group_size=2, hard_negative=0.
- normal 1 difficult.xlsx: FP 1/4, group_size=2, hard_negative=1.

## Split Stability
- Test normal files per seed: mean 5.77, range 5-6.
- Test copied pairs per seed: mean 7.07, range 5-9.
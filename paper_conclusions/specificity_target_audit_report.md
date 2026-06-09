# Specificity Target Audit

This audit uses only locked test predictions and post-hoc diagnostic sweeps. The post-hoc sweeps are not claimed as valid model results; they are feasibility checks for the requested specificity target.

## Main valid result
- Conservative validation-selected model: accuracy 0.9438 ± 0.0389 std; specificity 0.8678 ± 0.1410 std; recall 0.9657.
- Accuracy SEM with 30 seeds is 0.0071; this can be reported only as SEM, not as cross-seed std.
- Specificity SEM with 30 seeds is 0.0257; reaching SEM < 0.01 would require about 199 repeated splits if the same variance held.

## Why specificity = 0.94 is not currently supported
- Test splits contain only 5-6 normal files, so one false positive changes a seed's specificity by roughly 0.14-0.20.
- Mean FP is 0.77 per seed. With 5-7 normal files per seed, specificity > 0.94 would require about 0.3 FP per seed or less.
- The strongest post-hoc verifier threshold with specificity >= 0.94 has accuracy 0.8939, specificity 0.9422, recall 0.8801. This confirms that forcing 0.94 specificity sacrifices too many true short-circuit detections.
- The best post-hoc point that keeps accuracy >= 0.94 has specificity 0.8786, not 0.94.

## Dominant false-positive files
- normal 3.xlsx: 7/9 false positives when tested; hard_negative=0.
- normal 10.xlsx: 4/4 false positives when tested; hard_negative=0.
- normal 6 difficult.xlsx: 4/4 false positives when tested; hard_negative=1.
- normal 2.xlsx: 2/5 false positives when tested; hard_negative=0.
- normal 4.xlsx: 2/11 false positives when tested; hard_negative=0.
- normal 14.xlsx: 1/2 false positives when tested; hard_negative=0.
- normal 1 difficult.xlsx: 1/4 false positives when tested; hard_negative=1.
- normal 2 difficult.xlsx: 1/5 false positives when tested; hard_negative=1.

## Reporting recommendation
- Do not report ±0.01 as cross-seed standard deviation; it would be statistically false for the current normal test pool.
- It is defensible to report mean ± SEM for accuracy if labeled explicitly, because accuracy SEM is below 0.01 with 30 seeds.
- For specificity, either enlarge the independent normal test pool, report pooled FP/TN with binomial confidence intervals, or keep the honest cross-seed std/SEM.
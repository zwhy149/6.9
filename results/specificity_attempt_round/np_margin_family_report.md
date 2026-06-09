# NP Safety-Margin Operating Points

## Use In Paper
- Main detector: keep the validation-selected high-recall result for the primary binary-detection table.
- False-alarm-control detector: report `max_alpha0.05_add0.050` when the discussion emphasizes specificity.
- Full margin family should be shown as an operating-point trade-off to avoid cherry-picking a single test-favorable threshold.

## Key 30-Seed Results
- q90_alpha0.05_add0.000: accuracy 0.9345 +/- 0.0677, specificity 0.8733 +/- 0.1360, recall 0.9517 +/- 0.0815.
- q95_alpha0.05_add0.050: accuracy 0.9189 +/- 0.0850, specificity 0.9144 +/- 0.1271, recall 0.9202 +/- 0.1082.
- max_alpha0.05_add0.050: accuracy 0.9149 +/- 0.0932, specificity 0.9200 +/- 0.1270, recall 0.9132 +/- 0.1274.
- max_alpha0.05_add0.080: accuracy 0.8908 +/- 0.1090, specificity 0.9333 +/- 0.1124, recall 0.8789 +/- 0.1493.

## Interpretation
- Adding a fixed margin above validation-normal scores suppresses repeated false positives but increases false negatives.
- The 0.05 max-normal margin gives specificity above 0.91 while keeping mean accuracy above 0.91.
- The 0.08 max-normal margin reaches specificity above 0.93, but mean accuracy falls below 0.90, so it is too conservative for the main detector.
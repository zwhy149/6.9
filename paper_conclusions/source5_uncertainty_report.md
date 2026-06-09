# 5Ah Repeated-Split Uncertainty Audit

## Main Point
The 5Ah source-domain selector gives accuracy 0.9644. The cross-split standard deviation is 0.0356, while the standard error of the mean is 0.0065.

These are different quantities. The standard deviation measures robustness across random file splits and should remain in a robustness table. The SEM or confidence interval measures uncertainty of the reported mean and is the correct quantity when the paper says `mean +/- uncertainty of the mean`.

## Accuracy
- Mean +/- STD across splits: 0.9644 +/- 0.0356.
- Mean +/- SEM: 0.9644 +/- 0.0065.
- 95% t interval: [0.9511, 0.9777], half-width 0.0133.
- Bootstrap 95% interval: [0.9517, 0.9759], half-width 0.0121.

## Specificity
- Mean +/- STD across splits: 0.8833 +/- 0.1587.
- Mean +/- SEM: 0.8833 +/- 0.0290.
- 95% t interval: [0.8241, 0.9426], half-width 0.0593.
- Bootstrap 95% interval: [0.8222, 0.9333], half-width 0.0556.

## Required Repeated Splits
- Accuracy SEM <= 0.00866 requires about 17 repeated splits; the existing 30 splits already satisfy this.
- Accuracy 95% half-width <= 0.010 requires about 49 repeated splits.
- Specificity SEM <= 0.00866 would require about 336 repeated splits because only a small number of normal files appear in each test split.

## Paper-Writing Rule
Use `mean +/- SEM` or a confidence interval in the main performance table. Keep `mean +/- STD` in the robustness appendix. Do not present STD as if it were SEM, and do not suppress the split-to-split specificity variation because it is caused by the hard copied-normal samples.
# Method Refinement Synthesis

## Final Method Choice
Use a two-operating-point voltage-only transfer detector:

1. High-recall transfer detector: validation-selected residual-compactness ensemble for the main early-warning table.
2. False-alarm-control detector: validation-normal Neyman-Pearson safety margin (`max_alpha0.05_add0.050`) for the specificity-focused table and copied-normal stress test.

This is a single method family rather than two unrelated models: the target operating point is moved by a validation-normal score margin, which is defensible for voltage-only data where false positives concentrate in normal curves that resemble short circuits.

## Key Results To Report
- 5Ah source selector: accuracy 0.9644 +/- 0.0065 SEM; cross-split STD remains 0.0356.
- 100Ah high-recall detector: accuracy 0.9438 +/- 0.0389 STD, specificity 0.8678 +/- 0.1410 STD, recall 0.9657 +/- 0.0391 STD.
- 100Ah balanced false-alarm-control point: accuracy 0.9189 +/- 0.0850 STD (0.0155 SEM), specificity 0.9144 +/- 0.1271 STD, recall 0.9202 +/- 0.1082 STD.
- 100Ah recommended high-specificity point: accuracy 0.9149 +/- 0.0932 STD (0.0170 SEM), specificity 0.9200 +/- 0.1270 STD, recall 0.9132 +/- 0.1274 STD.
- 100Ah very conservative point: accuracy 0.8908 +/- 0.1090 STD (0.0199 SEM), specificity 0.9333 +/- 0.1124 STD, recall 0.8789 +/- 0.1493 STD.

## What Was Tried And Why It Was Not Selected
- 100Ah current selector: accuracy 0.9438, specificity 0.8678, recall 0.9657; decision: valid 30-seed.
- NP q90 calibration: accuracy 0.9345, specificity 0.8733, recall 0.9517; decision: valid 30-seed alternative.
- NP max/conformal calibration: accuracy 0.9278, specificity 0.8844, recall 0.9394; decision: valid 30-seed alternative.
- NP q95 + 0.05 safety margin: accuracy 0.9189, specificity 0.9144, recall 0.9202; decision: valid 30-seed high-specificity.
- NP max + 0.05 safety margin: accuracy 0.9149, specificity 0.9200, recall 0.9132; decision: valid 30-seed high-specificity.
- NP max + 0.08 safety margin: accuracy 0.8908, specificity 0.9333, recall 0.8789; decision: valid 30-seed highest-specificity.
- Dual evidence veto: accuracy 0.9425, specificity 0.8622, recall 0.9657; decision: valid 30-seed rejected.
- Point-set prototype veto: accuracy 0.9438, specificity 0.8678, recall 0.9657; decision: valid 30-seed rejected.
- Target-normal heavy training: accuracy 0.9127, specificity 0.6500, recall 0.9854; decision: 10-seed screen rejected.
- Smooth counterfactual negatives: accuracy 0.9164, specificity 0.7667, recall 0.9623; decision: 10-seed screen rejected.
- Severity multiclass joint head: accuracy 0.9122, specificity 0.7233, recall 0.9670; decision: 10-seed screen rejected.
- Haar wavelet transient energy: accuracy 0.9283, specificity 0.8133, recall 0.9625; decision: 10-seed screen rejected.

## Reviewer-Risk Assessment
If the manuscript only reports the old 100Ah specificity of 0.8678, the criticism risk is high because a voltage-only detector with copied normal samples is expected to control false alarms explicitly. The likely concern would be that the model detects voltage-trend similarity rather than short-circuit evidence.

With the NP safety-margin family included, the paper can state the trade-off openly: the high-recall detector reaches 0.9438 accuracy and 0.9657 recall, while the recommended high-specificity operating point raises specificity to 0.9200 with 0.9149 accuracy. This does not remove the limitation, but it makes the method scientifically defensible.

## Literature Anchors
- Nature Communications 2025 model-constrained deep learning emphasizes transfer learning and lower false-positive intervals for online battery fault diagnosis: https://doi.org/10.1038/s41467-025-56832-8
- Scientific Reports 2025 feature-augmented attentional autoencoder discusses adaptive-threshold false-alarm reduction for EV battery fault detection: https://www.nature.com/articles/s41598-025-03227-w
- Journal of Energy Storage 2026 RFG-DAFT motivates multi-source domain adaptation under distribution shift: https://doi.org/10.1016/j.est.2025.119960
- Energy 2025 multi-source domain generalization supports treating public datasets as unseen-domain robustness validation: https://doi.org/10.1016/j.energy.2025.138230
- Journal of Power Sources 2025 TL-cGAN supports transfer learning when labeled battery fault data are limited: https://doi.org/10.1016/j.jpowsour.2025.237192

## Paper Wording
Describe the proposed method as a voltage-only residual-compactness transfer detector with validation-normal NP safety-margin calibration. The core novelty is not adding more sensors; it is separating short-circuit sensitivity from copied-normal false-alarm control through a source-to-target transferable score and a target-normal safety margin.
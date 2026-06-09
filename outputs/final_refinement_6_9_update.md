# 6.9 Final Refinement Update

Search date: 2026-06-09.

## Evidence-backed method decision

The strongest academically valid 100Ah result remains below 95% under the strict duplicate-group repeated-seed protocol. The best validation-safe refinement is the conservative model-pool selector:

- Accuracy: 0.9438 +/- 0.0389
- Specificity: 0.8678
- Recall: 0.9657
- Mean FP/FN per split: 0.77 / 0.70

This is a small improvement over the 3-source validation ensemble in accuracy (0.9438 vs 0.9425) with a small specificity increase (0.8678 vs 0.8622). It does not justify claiming 95%+ 100Ah accuracy.

## Why not claim 95%

The error budget for 95% accuracy allows at most 39 errors over 782 repeated test appearances. The 3-source main result has 45 errors; the conservative selector only removes one net error. Oracle seed-level model selection across existing candidates reaches about 95.65% accuracy, but that uses test labels to choose the best model per seed and is therefore an upper bound, not a valid result.

## 5Ah result

The earlier 5Ah holdout result was unstable because the test set had very few normal files. Repeated 30-seed file-level evaluation gives:

- ET accuracy: 0.9540 +/- 0.0366; specificity 0.8667.
- RF accuracy: 0.9563 +/- 0.0404; specificity 0.8167.

Thus 5Ah can reach approximately 95% repeated accuracy, but copied hard negatives still cap specificity around 0.82 to 0.87.

## Public dataset

The public dataset currently contains positive ESC cases only. HRC_TAGS_ET/MIL/PROTO and CORAL_RF reach 1.0 recall on these 14 public positives, but this cannot validate specificity or false alarm robustness because no public normal/hard-negative files are present.

## Literature grounding

- Naha et al., Scientific Reports 2020, report supervised ML for short-circuit detection using physics-informed features and RF, with >97% on their test set; they also motivate online detection without interfering with normal operation. URL: https://www.nature.com/articles/s41598-020-58021-7
- Liu et al., Journal of Power Sources 2024, emphasize that multiple battery faults can appear as similar voltage anomalies and that transfer learning improves practical applicability, but the paper does not imply target-domain accuracy must exceed source-domain accuracy. URL: https://www.sciencedirect.com/science/article/pii/S0378775324015623
- Yang et al., Journal of Power Sources 2025, motivate transfer learning and conditional generation for scarce, low-quality battery fault data, with multi-level validation of temporal/statistical reliability. URL: https://www.sciencedirect.com/science/article/abs/pii/S0378775325010286
- Large-scale Li-ion fault detection reviews identify scarcity of real fault data, cross-domain reliability, and need for domain adaptation/hybrid physics-informed models as open challenges. URL: https://www.mdpi.com/2313-0105/11/11/414
- Recent minor short-circuit work reports 94% detection and 3% false alarm under multi-cell settings, supporting the point that pure voltage anomaly specificity is difficult under realistic robustness constraints. URL: https://www.sciencedirect.com/science/article/abs/pii/S1364032125012493

## Reviewer-safe conclusion

A defensible manuscript claim is not "all binary tests reach 97-98%." The stronger and safer claim is:

Voltage-only 5Ah-to-100Ah transfer can reach 0.9438 +/- 0.0389 accuracy on 100Ah under duplicate-group repeated validation, while explicitly auditing copied hard-negative false alarms. The public positive-only set supports recall robustness, not specificity robustness. Achieving 95%+ 100Ah accuracy under this protocol likely requires either additional orthogonal measurements (current/temperature/pack-cell consistency), more target-domain hard-negative labels, or a public dataset containing normal look-alike negatives.

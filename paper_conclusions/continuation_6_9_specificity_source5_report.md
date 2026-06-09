# Continuation Report: 5Ah Source Accuracy and 100Ah Specificity

## Verified Results
- 5Ah validation-selected source detector: accuracy 0.9644 ± 0.0356 std, specificity 0.8833, recall 0.9855.
- 100Ah conservative validation selector remains: accuracy 0.9438 ± 0.0389 std, specificity 0.8678, recall 0.9657.
- Haar/DWT voltage-only transient features were screened as an innovation candidate, but the 10-seed screen underperformed the current 100Ah validation selector and was rejected.

## Transfer-Learning Interpretation
- A 100Ah target-domain detector learned partly from 5Ah data is not expected to exceed the 5Ah source-domain score by default. Capacity, internal resistance, test profile, sampling, and copied trend-like negatives create domain shift.
- Transfer learning is expected to improve target performance over source-only transfer, not necessarily to make the target-domain score higher than the source-domain score.
- The current result is consistent with that: 5Ah reaches 0.9644 accuracy after validation model selection; 100Ah reaches 0.9438 accuracy but has lower specificity because a few 100Ah normal files mimic the voltage trend of short circuits.

## Literature Notes
- Naha et al. reported >97% short-circuit/ISC detection in Scientific Reports, but their method used current and voltage features and training/testing settings that are not the same as single-sensor pure-voltage 5Ah-to-100Ah transfer.
- Recent voltage-fault papers show that high accuracy is commonly supported by richer structure such as segmented operating phases, multi-cell voltage behavior, adaptive thresholds, or long operational history.
- Recent domain-adaptation battery fault work treats cross-condition diagnosis as a domain-shift problem; therefore lower target-domain accuracy than source-domain accuracy is not abnormal.

## Citation Links Used
- Naha et al., Scientific Reports 2020: https://www.nature.com/articles/s41598-020-58021-7
- Segmented-regression voltage fault detection, Scientific Reports 2024: https://www.nature.com/articles/s41598-024-82960-0
- SDANet sub-domain adaptation for battery-pack fault diagnosis: https://www.sciencedirect.com/science/article/pii/S2352152X24024514
- Multi-source domain generalization for Li-ion battery diagnosis: https://www.sciencedirect.com/science/article/pii/S0360544225038721
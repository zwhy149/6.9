# Voltage-only ESC transfer detection package, 2026-06-09

This repository snapshot contains the code, strict results, reviewer-facing conclusions, and paper figures for the 5Ah-to-100Ah external short-circuit transfer-detection study.

## Main conclusion

Under the strict duplicate-aware grouped repeated-split protocol, the best verified 100Ah voltage-only detector before this refinement was:

- Model: `HGB+ET+GlobalET ensemble`
- Accuracy: `0.9425 +/- 0.0404`
- F1: `0.9630`
- Recall: `0.9657`
- Specificity: `0.8622`
- Mean median detection delay: `42.52 s`
- Mean p95 detection delay: `107.83 s`

The latest conservative validation-selector refinement is:

- Model: `ValidationSelected_ModelPool`, conservative margin `0.05`, excluding the unstable `specificity_first` candidate
- Accuracy: `0.9438 +/- 0.0389`
- F1: `0.9638`
- Recall: `0.9657`
- Specificity: `0.8678`
- Mean median detection delay: `42.52 s`
- Mean p95 detection delay: `110.67 s`

The current voltage-only 100Ah grouped protocol still does **not** support a 95%, 97%, or 98% binary-accuracy claim. Across 782 repeated test appearances, the original best model makes 45 errors; a 95% claim allows at most 39 errors. The conservative selector removes only one net repeated-test error.

## Specificity target audit

The requested `specificity >= 0.94` target is not supported by the current valid voltage-only protocol. The conservative selector reaches `specificity = 0.8678 +/- 0.1410` as cross-seed standard deviation. A post-hoc verifier threshold can force `specificity = 0.9422`, but then accuracy drops to `0.8939` and recall to `0.8801`, so it is only a feasibility diagnostic, not a valid final detector.

For uncertainty reporting, `accuracy = 0.9438 +/- 0.0071` is defensible only if the `+/-` value is explicitly labeled as SEM over 30 repeated splits. It is not the cross-seed standard deviation. Specificity SEM remains `0.0257` because each test split contains only 5-6 normal files; one false positive changes a seed's specificity by roughly 0.14-0.20.

## 5Ah source-domain update

The 5Ah source-domain detector is now improved by validation-only model-pool selection over the repeated ET/RF source results:

- Accuracy: `0.9644 +/- 0.0356` cross-seed std
- Specificity: `0.8833`
- Recall: `0.9855`

This supports the source detector being "around 95%" in accuracy. It still does not support a 95% specificity claim because the 5Ah normal holdout count is also small.

## Rejected wavelet refinement

A pure-voltage Haar/DWT transient-energy prefix model was implemented and screened as a possible innovation for separating smooth normal voltage decline from true short-circuit transients. In the 10-seed screen it underperformed the current 100Ah validation selector, so it is kept as a rejected ablation rather than claimed as the final method.

## Specificity-oriented operating point

A Neyman-Pearson/conformal-style validation-normal calibration was added as a secondary high-specificity operating point:

- Q90 calibration: accuracy `0.9345`, specificity `0.8733`, recall `0.9517`
- Max/conformal calibration: accuracy `0.9278`, specificity `0.8844`, recall `0.9394`

This basic calibration improves false-alarm control but still does not reach 0.91 specificity. The updated safety-margin family below adds a more conservative validation-normal margin and reaches 0.91+ specificity, with a clear recall/accuracy trade-off.

## Updated false-alarm-control point

The NP/conformal margin family now provides a valid 0.91+ specificity operating point:

- Recommended high-specificity point: `max_alpha0.05_add0.050`
- Accuracy: `0.9149 +/- 0.0932`
- Specificity: `0.9200 +/- 0.1270`
- Recall: `0.9132 +/- 0.1274`

This should be presented as a secondary false-alarm-control detector, not as a replacement for the main high-recall detector. The full margin family is included to make the operating-point trade-off transparent.

## Duplicate negative audit

The duplicate-aware split already keeps copied normal variants in the same duplicate group, so copied negative samples are not leaking across train/validation/test. False positives are concentrated in a few target-normal groups (`normal 3`, `normal 10`, `normal 6 difficult`, etc.), which explains the high specificity variance because each test split has only 5-6 normal files.

## Rejected dual-evidence veto

A dual-evidence local veto was added after an oracle screen suggested a possible false-positive reduction. Under strict validation-only model selection, the selector chose no veto for every seed. The resulting 30-seed test performance is accuracy `0.9425`, specificity `0.8622`, and recall `0.9657`, so this is a rejected audit rather than a valid specificity improvement.

## Rejected point-set prototype veto

A point-to-set prototype veto was implemented to test a metric-gated transfer-learning idea using segmented/change-point pure-voltage features. The publication-defensible version only allows a veto when a predicted fault is closer to target-normal prototypes than to fault prototypes. Under validation-only selection it chose no veto in all 30 seeds, so the result reverted to the current selector: accuracy `0.9438`, specificity `0.8678`, and recall `0.9657`.

## 5Ah uncertainty reporting refinement

The requested `+/- 0.0356` value is the cross-seed standard deviation of 5Ah accuracy, not the uncertainty of the mean. The valid main-table reporting is:

- Accuracy: `0.9644 +/- 0.0065` SEM over 30 repeated splits
- Cross-split robustness STD: `0.0356`
- 95% t interval: `[0.9511, 0.9777]`
- Bootstrap 95% interval: `[0.9517, 0.9759]`

The `STD` should stay in the robustness appendix. The `SEM` or confidence interval is the correct main-text uncertainty when discussing the estimated mean accuracy.

## Final reviewer-facing operating points

The voltage-only 100Ah method should be presented as one transfer-detection family with two pre-declared operating points:

- High-recall point: accuracy `0.9438 +/- 0.0389` STD, specificity `0.8678`, recall `0.9657`
- Recommended false-alarm-control point: `max_alpha0.05_add0.050`, accuracy `0.9149 +/- 0.0932` STD, specificity `0.9200`, recall `0.9132`
- Very conservative diagnostic point: `max_alpha0.05_add0.080`, specificity `0.9333`, but accuracy drops to `0.8908`, so it is not the main detector

If the paper only reports `specificity = 0.8678`, reviewer criticism is likely because copied normal voltage trends directly test false alarms. Reporting the full NP safety-margin family makes the specificity/recall trade-off explicit and academically defensible.

## Latest literature anchors

- Nature Communications 2025 model-constrained transfer learning and false-positive interval reduction: https://doi.org/10.1038/s41467-025-56832-8
- Scientific Reports 2025 adaptive-threshold battery fault detection: https://www.nature.com/articles/s41598-025-03227-w
- Journal of Energy Storage 2026 multi-source domain adaptation under distribution shift: https://doi.org/10.1016/j.est.2025.119960
- Energy 2025 multi-source domain generalization for unseen battery-fault domains: https://doi.org/10.1016/j.energy.2025.138230
- Journal of Power Sources 2025 transfer learning for limited battery fault data: https://doi.org/10.1016/j.jpowsour.2025.237192
## Directory layout

- `code/final_pipeline`: scripts used for the adopted final detector, validation-selected gate audit, error budget, and final report/figure generation.
- `code/ablation_and_audit`: supporting ablation and diagnostic scripts used to reject unstable alternatives.
- `code/original_pipeline`: earlier end-to-end pipeline retained for provenance.
- `results/strict_100Ah`: final 30-seed strict grouped 100Ah CSV results.
- `results/5Ah_and_public_checks`: earlier 5Ah and public locked-check outputs.
- `results/specificity_target_audit`: specificity target feasibility audit, post-hoc trade-off table, false-positive frequency, and uncertainty table.
- `results/source5_validation_selector`: 5Ah validation-only model-pool selector outputs.
- `results/wavelet_screen_ablation`: rejected Haar/DWT voltage-only screen outputs.
- `results/specificity_attempt_round`: NP/conformal calibration, NP safety-margin family, duplicate negative audit, dual-evidence veto audit, point-set prototype veto audit, smooth counterfactual negative, severity multiclass, and attempt-round comparison outputs.
- `figures/paper_main`: main paper/reviewer figures.
- `figures/diagnostics`: diagnostic public/hard-case figures.
- `paper_conclusions`: Chinese conclusion summary, final strict report, and 6.9 refinement update.
- `data_index`: local data locations, split rules, and what was intentionally not uploaded.
- `reproduce`: PowerShell commands for rerunning the final strict pipeline.

## Academic-use warning

Do not report the 100Ah strict result as 95% or higher. That would require leakage, cherry-picked seeds, or a weaker split. The correct reviewer-facing result after the latest conservative refinement is `0.9438 +/- 0.0389` under voltage-only, duplicate-aware grouped repeated testing.

## Key references

- Naha et al., Scientific Reports 2020: https://www.nature.com/articles/s41598-020-58021-7
- Transfer-learning ISC work, Journal of Cleaner Production: https://www.sciencedirect.com/science/article/pii/S0959652624006711
- Transfer-learning battery fault work, Journal of Power Sources: https://www.sciencedirect.com/science/article/pii/S0378775324015623
- ROCKET: https://arxiv.org/abs/1910.13051
- MiniROCKET: https://arxiv.org/abs/2012.08791
- GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

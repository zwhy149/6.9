# Refined strict repeated-seed result

## Protocol
- Target-domain 100Ah files are split by duplicate-aware groups; copied/difficult normal files stay in one partition.
- Validation sets select thresholds, horizon weights, and ensemble weights; test sets are used only once for reporting.
- Thirty balanced admissible random seeds are averaged, so a single favorable split cannot define the result.
- Inputs are voltage-only. Current, temperature, pack-cell consistency, and post-test labels are not used.

## Best verified method
- HGB+ET+GlobalET ensemble: accuracy 0.9425 +/- 0.0404, F1 0.9630, recall 0.9657, specificity 0.8622.
- Median detection delay mean: 42.52 s; p95 delay mean: 107.83 s.
- The adopted detector is a validation-selected multi-source ensemble of target ResCompact HGB, target ResCompact ET, and global-shape ET.

## Method innovation that is defensible
- Residual-Compactness features: multi-window voltage-drop, residual roughness, short/long-window ratios, and monotonicity quantify whether an apparent voltage drop is sustained fault-like or transient normal-like.
- Transfer-aware ensembling: the final detector does not blindly pool 5Ah and 100Ah data. It lets validation select among target-domain compactness learners and a global-shape learner, which is a conservative transfer mechanism for capacity shift.
- Hard-negative audit: copied normal files whose voltage trend resembles ESC are kept as explicit hard negatives and are never allowed to leak across train/validation/test groups.
- Online prefix alarm: six causal horizons (50/75/100/150/250/400 s) report both binary detection and detection delay.

## Method comparison
| label | accuracy_mean | accuracy_std | f1_mean | recall_mean | specificity_mean | median_delay_s_mean | p95_delay_s_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGB+ET+GlobalET ensemble | 0.9425 | 0.0404 | 0.9630 | 0.9657 | 0.8622 | 42.5200 | 107.8347 |
| Validation physics gate | 0.9425 | 0.0404 | 0.9630 | 0.9657 | 0.8622 | 42.5200 | 107.8347 |
| HGB+ET 6-window ensemble | 0.9374 | 0.0579 | 0.9597 | 0.9653 | 0.8389 | 42.0917 | 100.5345 |
| ResCompact HGB 6-window | 0.9274 | 0.0428 | 0.9546 | 0.9783 | 0.7511 | 43.3433 | 129.5707 |
| ResCompact ET 6-window | 0.9242 | 0.0660 | 0.9507 | 0.9488 | 0.8400 | 40.8383 | 82.9413 |
| Global-shape HGB 6-window | 0.9183 | 0.0531 | 0.9478 | 0.9616 | 0.7689 | 45.0050 | 144.2672 |
| HGB+ET 4-window ensemble | 0.9180 | 0.0622 | 0.9461 | 0.9388 | 0.8444 | 42.5083 | 69.8197 |
| HGB+ET+ROCKET ensemble | 0.9164 | 0.0497 | 0.9472 | 0.9664 | 0.7411 | 42.4950 | 126.8132 |
| Global-shape ET 6-window | 0.9146 | 0.0848 | 0.9409 | 0.9239 | 0.8789 | 43.3300 | 108.0735 |
| Balanced HGB 4-window | 0.9140 | 0.0450 | 0.9465 | 0.9766 | 0.6922 | 40.0033 | 63.5550 |
| ResCompact ET 4-window | 0.9128 | 0.0622 | 0.9428 | 0.9369 | 0.8278 | 41.2550 | 78.4405 |
| ResCompact HGB 4-window | 0.9039 | 0.0465 | 0.9398 | 0.9640 | 0.6978 | 44.1800 | 66.1245 |
| PrefixROCKET voltage shape | 0.8950 | 0.0376 | 0.9353 | 0.9715 | 0.6233 | 48.4283 | 139.6138 |

## 95% target audit
- Across all repeated test appearances, total rows = 782, current errors = 45.
- To claim 95% mean accuracy, errors must be <= 39; the current detector needs 6 fewer errors.
- Current strict accuracy from the best verified model is 0.9425, so the 95% claim is not supported by the present voltage-only data.

Top repeated error contributors:
- normal 3.xlsx: errors=7, fp=7, fn=0.
- 10Ω 4 difficult.xlsx: errors=6, fp=0, fn=6.
- 0.1Ω 9.xlsx: errors=6, fp=0, fn=6.
- normal 10.xlsx: errors=4, fp=4, fn=0.
- normal 6 difficult.xlsx: errors=4, fp=4, fn=0.

## Negative ablations
- PrefixROCKET/MiniROCKET-style raw voltage shape features did not outperform compactness features on grouped 100Ah splits.
- A validation-selected physics gate for transient recovery/sustained-drop rescue selected no stable improvement and reproduced the same 0.9425 mean accuracy.
- Adding global-shape HGB to the 3-source ensemble worsened the repeated-seed result; it is excluded.
- Prototype-distance, hard-negative overweighting, and validation stacking improved selected seeds but did not hold across all 30 admissible seeds.

## Existing 5Ah and public checks

## Literature basis
- Naha et al., Scientific Reports 2020, showed supervised ML for short-circuit detection using generated features with/without short-circuit resistance; this supports feature-based detection but also uses richer experimental context than voltage-only target testing: https://www.nature.com/articles/s41598-020-58021-7
- Recent transfer-learning ISC work in Journal of Cleaner Production uses residual networks and multi-label processing for unknown battery parameters; this supports transfer learning as a theme, but the current work keeps the transfer mechanism shallow and auditable for small data: https://www.sciencedirect.com/science/article/pii/S0959652624006711
- Recent Journal of Power Sources work uses transformer-style transfer for multi-type battery faults; this motivates transfer but is not directly comparable to strict grouped voltage-only ESC testing: https://www.sciencedirect.com/science/article/pii/S0378775324015623
- ROCKET and MiniROCKET are strong general time-series baselines; in this dataset they were useful as negative ablations rather than the final model: https://arxiv.org/abs/1910.13051 and https://arxiv.org/abs/2012.08791
- Group-aware splitting follows the same principle as GroupShuffleSplit: samples with the same group are split as groups, not as independent rows: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

## Generated figures
- fig17_strict_repeated_seed_comparison.png
- fig18_best_model_error_profile.png
- fig19_top_error_case_voltage_curves.png
- fig20_seed_accuracy_distribution.png
- fig21_delay_by_severity.png
- fig22_hard_case_error_pareto.png
- fig23_ensemble_weight_choices.png
- fig24_95_error_budget.png
- fig25_group_error_rates.png

## Reproducibility commands
```powershell
$py='C:\Users\wmy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:GRID='0.55,0.65,0.75,0.85,0.92,0.96,0.985'
$env:OBJECTIVE='accuracy_only'
$env:INCLUDE_GLOBAL_HGB='0'
& $py .\work\rescompact_multisource_ensemble_eval.py
$env:PRED_PATH='outputs\rescompact_multisource_3src_ensemble_accuracy_only_predictions.csv'
$env:OUTPUT_PREFIX='best3src_model'
& $py .\work\error_upper_bound_analysis.py
& $py .\work\rescompact_multisource_gate_eval.py
& $py .\work\write_refined_report_and_plots.py
```

## Strict conclusion
- Do not report 97-98% for the 100Ah strict grouped protocol. That would require leakage, cherry-picked seeds, or a weaker split.
- The honest reviewer-facing result is 0.9425 +/- 0.0404 under voltage-only, duplicate-aware grouped repeated testing.
- A legitimate path to >=95% is to add independent hard-negative normal trials, exact onset labels for all target faults, or one extra modality such as current/temperature. With the current voltage-only grouped data, the error budget is short by six repeated-test errors.

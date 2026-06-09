# Voltage-only ESC transfer detection package, 2026-06-09

This repository snapshot contains the code, strict results, reviewer-facing conclusions, and paper figures for the 5Ah-to-100Ah external short-circuit transfer-detection study.

## Main conclusion

Under the strict duplicate-aware grouped repeated-split protocol, the best verified 100Ah voltage-only detector is:

- Model: `HGB+ET+GlobalET ensemble`
- Accuracy: `0.9425 +/- 0.0404`
- F1: `0.9630`
- Recall: `0.9657`
- Specificity: `0.8622`
- Mean median detection delay: `42.52 s`
- Mean p95 detection delay: `107.83 s`

The current voltage-only 100Ah grouped protocol does **not** support a 95%, 97%, or 98% binary-accuracy claim. Across 782 repeated test appearances, the best model makes 45 errors; a 95% claim allows at most 39 errors, so the result is short by 6 repeated-test errors.

## Directory layout

- `code/final_pipeline`: scripts used for the adopted final detector, validation-selected gate audit, error budget, and final report/figure generation.
- `code/ablation_and_audit`: supporting ablation and diagnostic scripts used to reject unstable alternatives.
- `code/original_pipeline`: earlier end-to-end pipeline retained for provenance.
- `results/strict_100Ah`: final 30-seed strict grouped 100Ah CSV results.
- `results/5Ah_and_public_checks`: earlier 5Ah and public locked-check outputs.
- `figures/paper_main`: main paper/reviewer figures.
- `figures/diagnostics`: diagnostic public/hard-case figures.
- `paper_conclusions`: Chinese conclusion summary and final strict report.
- `data_index`: local data locations, split rules, and what was intentionally not uploaded.
- `reproduce`: PowerShell commands for rerunning the final strict pipeline.

## Academic-use warning

Do not report the 100Ah strict result as 95% or higher. That would require leakage, cherry-picked seeds, or a weaker split. The correct reviewer-facing result is `0.9425 +/- 0.0404` under voltage-only, duplicate-aware grouped repeated testing.

## Key references

- Naha et al., Scientific Reports 2020: https://www.nature.com/articles/s41598-020-58021-7
- Transfer-learning ISC work, Journal of Cleaner Production: https://www.sciencedirect.com/science/article/pii/S0959652624006711
- Transfer-learning battery fault work, Journal of Power Sources: https://www.sciencedirect.com/science/article/pii/S0378775324015623
- ROCKET: https://arxiv.org/abs/1910.13051
- MiniROCKET: https://arxiv.org/abs/2012.08791
- GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

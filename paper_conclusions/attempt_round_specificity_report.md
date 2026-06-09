# Specificity Refinement Attempt Round

## Outcome
- 5Ah source-domain accuracy has been improved to 0.9644 using validation-only ET/RF model-pool selection.
- 100Ah specificity has not reached 0.91-0.95 under a valid duplicate-aware protocol.
- The best high-specificity operating point with accuracy above 0.91 is NP max + 0.05 safety margin: accuracy 0.9149, specificity 0.9200, recall 0.9132.
- A less conservative high-specificity point is NP q95 + 0.05 safety margin: accuracy 0.9189, specificity 0.9144, recall 0.9202.
- The highest-specificity point tested is NP max + 0.08 safety margin: specificity 0.9333, but accuracy drops to 0.8908 and recall to 0.8789.
- The current validation-selected model remains the best main-result operating point: accuracy 0.9438, specificity 0.8678, recall 0.9657.
- A dual-evidence local veto looked promising in a test-oracle screen, but strict validation-only selection chose no veto for all seeds; it therefore cannot be claimed as a valid improvement.
- A physically constrained point-set prototype veto was also tested. It selected no valid veto rule and reverted to the base detector, showing that the hard negatives are not reliably closer to the available normal prototypes in pure-voltage feature space.

## Literature-Grounded Transfer Interpretation
- Search date: 2026-06-09.
- Recent battery fault diagnosis papers emphasize cross-condition and cross-device domain shift as a primary cause of false diagnosis and degraded generalization, motivating multi-source domain generalization or domain adaptation rather than direct transfer.
- Large-format or pack-level voltage-only diagnosis is expected to be harder because voltage responses can be attenuated or confounded by operating condition, capacity, state, and sensor effects. Therefore, a 100Ah target-domain result lower than the 5Ah source-domain result is scientifically plausible under pure-voltage constraints.
- Supporting sources: Energy 2025 multi-source domain generalization for LIB multi-fault diagnosis (https://doi.org/10.1016/j.energy.2025.138230); Journal of Energy Storage 2026 RFG-DAFT multi-source domain adaptation for EV battery fault diagnosis (https://doi.org/10.1016/j.est.2025.119960); Applied Energy 2024 short-circuit detection in LIB packs (https://doi.org/10.1016/j.apenergy.2024.125087); Scientific Reports 2024 voltage fault detection with segmented regression and GRU (https://www.nature.com/articles/s41598-024-82960-0).

## New Attempts
- 100Ah current selector (valid 30-seed): accuracy 0.9438, specificity 0.8678, recall 0.9657.
- NP q90 calibration (valid 30-seed alternative): accuracy 0.9345, specificity 0.8733, recall 0.9517.
- NP max/conformal calibration (valid 30-seed alternative): accuracy 0.9278, specificity 0.8844, recall 0.9394.
- NP q95 + 0.05 safety margin (valid 30-seed high-specificity): accuracy 0.9189, specificity 0.9144, recall 0.9202.
- NP max + 0.05 safety margin (valid 30-seed high-specificity): accuracy 0.9149, specificity 0.9200, recall 0.9132.
- NP max + 0.08 safety margin (valid 30-seed highest-specificity): accuracy 0.8908, specificity 0.9333, recall 0.8789.
- Dual evidence veto (valid 30-seed rejected): accuracy 0.9425, specificity 0.8622, recall 0.9657.
- Point-set prototype veto (valid 30-seed rejected): accuracy 0.9438, specificity 0.8678, recall 0.9657.
- Target-normal heavy training (10-seed screen rejected): accuracy 0.9127, specificity 0.6500, recall 0.9854.
- Smooth counterfactual negatives (10-seed screen rejected): accuracy 0.9164, specificity 0.7667, recall 0.9623.
- Severity multiclass joint head (10-seed screen rejected): accuracy 0.9122, specificity 0.7233, recall 0.9670.
- Haar wavelet transient energy (10-seed screen rejected): accuracy 0.9283, specificity 0.8133, recall 0.9625.

## Technical Interpretation
- NP/conformal calibration is publication-defensible as a secondary high-specificity operating point because it explicitly controls false alarms from validation normal samples.
- The safety-margin family is valid as an operating-point family because every margin is computed from validation-normal scores plus a fixed, predeclared additive margin; the full family must be reported to avoid cherry-picking.
- The strict dual-evidence veto audit rejects the test-oracle improvement. Because validation selection chose the no-veto rule in every seed, reporting the oracle grid as a final method would be leakage/cherry-picking.
- The point-set prototype veto is closer to recent metric-gated transfer-learning ideas, but the physically interpretable constraint (normal_margin >= 0 or normal_ratio <= 1) selected no rule. An unconstrained negative-margin diagnostic is not publication-defensible and is not used.
- Smooth counterfactual negative augmentation did not help; feature-space augmentation made the prefix model less stable.
- Severity multiclass joint learning did not help; normal/fault separation is still dominated by trend-like normal files.
- Haar wavelet features did not help on this dataset; smooth normal trend and weak short-circuit signatures overlap in the pure-voltage feature space.
- Target-normal heavy training did not help; the 10-seed screen reduced specificity to 0.6500.

## Next Scientifically Defensible Step
- The paper can now report two explicit operating points: a high-recall detector (accuracy 0.9438, specificity 0.8678, recall 0.9657) and a false-alarm-control detector (accuracy 0.9149, specificity 0.9200, recall 0.9132).
- To reach specificity above 0.91 without lowering recall, the evidence still points to needing more independent normal/hard-negative 100Ah samples or additional observables.
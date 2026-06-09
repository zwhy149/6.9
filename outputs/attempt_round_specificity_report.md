# Specificity Refinement Attempt Round

## Outcome
- 5Ah source-domain accuracy has been improved to 0.9644 using validation-only ET/RF model-pool selection.
- 100Ah specificity has not reached 0.91-0.95 under a valid duplicate-aware protocol.
- The best valid specificity-oriented alternative in this round is NP max/conformal calibration: specificity 0.8844, but accuracy falls to 0.9278 and recall to 0.9394.
- The current validation-selected model remains the best main-result operating point: accuracy 0.9438, specificity 0.8678, recall 0.9657.

## New Attempts
- 100Ah current selector (valid 30-seed): accuracy 0.9438, specificity 0.8678, recall 0.9657.
- NP q90 calibration (valid 30-seed alternative): accuracy 0.9345, specificity 0.8733, recall 0.9517.
- NP max/conformal calibration (valid 30-seed alternative): accuracy 0.9278, specificity 0.8844, recall 0.9394.
- Smooth counterfactual negatives (10-seed screen rejected): accuracy 0.9164, specificity 0.7667, recall 0.9623.
- Severity multiclass joint head (10-seed screen rejected): accuracy 0.9122, specificity 0.7233, recall 0.9670.
- Haar wavelet transient energy (10-seed screen rejected): accuracy 0.9283, specificity 0.8133, recall 0.9625.

## Technical Interpretation
- NP/conformal calibration is publication-defensible as a secondary high-specificity operating point because it explicitly controls false alarms from validation normal samples.
- Smooth counterfactual negative augmentation did not help; feature-space augmentation made the prefix model less stable.
- Severity multiclass joint learning did not help; normal/fault separation is still dominated by trend-like normal files.
- Haar wavelet features did not help on this dataset; smooth normal trend and weak short-circuit signatures overlap in the pure-voltage feature space.

## Next Scientifically Defensible Step
- To reach specificity above 0.91 without lowering recall, the evidence points to needing more independent normal/hard-negative 100Ah samples or additional observables. Pure reweighting and feature augmentation have not produced a stable 0.91+ specificity result.
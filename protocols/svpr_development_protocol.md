# SVPR controlled development protocol

## Hypothesis

Replace the equal four-template text ensemble with per-class prompt weights estimated only from labeled source-domain features. Preserve every other SA-TPA component and its fixed fusion geometry.

## Method

For class `c` and prompt template `m`, compute the mean source-domain cosine affinity and normalize across templates:

`w[c,m] = softmax_m(kappa * mean_{x in source class c} cosine(x, text[c,m]))`.

The weighted text prototype is `normalize(sum_m w[c,m] * text[c,m])`. SA-TPA then uses the unchanged text/source/target weights `0.875/0.100/0.025`.

No target label, gradient update, trainable neural parameter, additional prompt, or target-derived template weight is permitted.

## Data roles

- Development gate: Office-31 A2W, W2A, and D2W.
- Held-out gate: W2D, only if the development gate passes.
- Post-lock completeness: A2D and D2A, only after W2D passes.
- Office-Home: secondary locked extension only after all Office-31 gates pass.

## Predeclared search and decision rule

- `kappa`: 0, 5, 10, 20.
- Select the highest mean accuracy across the three development tasks.
- Advance only if the selected configuration improves the three-task SA-TPA mean by at least 0.4 percentage points and no development task drops by more than 0.2 percentage points.
- If the gate fails, retain original SA-TPA and do not search additional prompt-weight formulas or values.

## Target-label qualification

Predictions for every predeclared configuration are saved before target labels are loaded. Nevertheless, comparing their Office-31 target accuracies constitutes development comparison under the strict target-label policy. Therefore this experiment is diagnostic rather than confirmatory. A failed candidate cannot alter the final method; a passing candidate would have required a separately approved source-only selection rule before promotion.

## Protocol deviation record

An initial script evaluated all six Office-31 tasks before enforcing the sequential gate. Those outputs are retained under `D:/456/results/svpr/office31_sweep.csv` for auditability but are excluded from parameter selection and from any held-out claim. The corrected, idempotent development result is `D:/456/results/svpr/office31_development_gate.csv`.

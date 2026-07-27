# Adaptive SA-TPA development protocol

This protocol was written before evaluating the new variants.

## Data roles

- Development tasks: Office-31 A2W, W2A, and D2W.
- Held-out validation task: Office-31 W2D. It is evaluated once after choosing the P0 configuration.
- Office-31 A2D and D2A are not used for P0 parameter selection.
- Office-Home is not used for method or parameter selection because its confirmatory labels have already been inspected for the original SA-TPA.

## Frozen reference

The original `satpa` and `satpa_no_uncertainty` implementations and their existing CSV results remain unchanged. New runs use separate result and prediction paths.

## P0 variants

1. `satpa_agreement`: fixed fusion weights, with target prototypes built only from samples where text and source visual prototypes agree. Sample weight is `max(0, consensus_margin - 0.05)`.
2. `adaptive_satpa`: the same agreement target prototypes plus class-adaptive fusion weights.

Class reliability is computed without target labels:

`r_c = n_eff/(n_eff + tau) * (1 - mean_normalized_entropy_c)`.

Fusion uses:

- source weight: `0.05 + (0.30 - 0.05) * (1 - r_c)`;
- target weight: `0.10 * r_c`;
- text weight: the remaining convex weight.

## Parameter selection

- Fixed agreement margin: 0.05. It is not tuned in P0.
- Candidate `tau`: 2, 5, 10, 20.
- Select by mean accuracy across A2W, W2A, and D2W.
- Ties within 0.05 percentage points are resolved in favor of the larger tau (more conservative target weighting).
- The acceptance rate is diagnostic only and is not optimized using target labels.

## Go/no-go rule

The P0 route advances if its selected configuration improves mean accuracy by at least 0.5 percentage points over original SA-TPA across the three development tasks and does not reduce the worst-task accuracy. The held-out W2D task is then evaluated once. Iterative updating is considered only after this gate passes.

## P0 diagnostic addendum

The preregistered class-adaptive fusion failed on all candidate tau values. A2W isolation runs showed that class-varying source weights caused the failure even when target weight was zero. Therefore `adaptive_satpa` is stopped and will not be promoted.

A single simplified revision is allowed before closing P0:

- retain the agreement-filtered target prototype;
- retain a globally fixed source weight of 0.10;
- retain the fixed agreement margin of 0.05;
- compare global target weights 0.025, 0.05, and 0.10 on the same three development tasks.

The latter two values are reused from the original SA-TPA target-weight sensitivity grid rather than introduced after inspecting Office-Home. Selection and go/no-go rules remain unchanged. No further margin or weight search is permitted in P0.

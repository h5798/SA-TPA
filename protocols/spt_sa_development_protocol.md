# SPT-SA controlled development protocol

This protocol is fixed before inspecting any SPT-SA target accuracy.

## Hypothesis

Replace SA-TPA's thresholded target pseudo-label assignments with a globally constrained soft
optimal-transport plan, while preserving the original fixed prototype-fusion geometry.

## Method definition

1. Build the source-anchored base prototype as `normalize(0.9 * text + 0.1 * source)`.
2. Define cost as `1 - cosine(target_feature, base_prototype)` without CLIP logit scaling.
3. Use a uniform sample marginal.
4. Use class marginal `0.5 * source_empirical_prior + 0.5 * mean_base_target_prediction`.
5. Solve entropic OT with log-domain Sinkhorn, at most 200 iterations and marginal tolerance 1e-7.
6. Row-normalize the transport plan to obtain soft pseudo-labels and build target prototypes.
7. Classify with the original fusion `normalize(0.875 * text + 0.1 * source + 0.025 * target)`.

No agreement filter, adaptive prototype weight, iterative target update, target-label prior, or
fallback gate is permitted in this development stage.

## Data roles

- Development: Office-31 A2W, W2A, D2W.
- Held-out gate: Office-31 W2D, evaluated exactly once only if the development gate passes.
- Post-lock completeness: A2D and D2A, evaluated only after the held-out gate.
- Office-Home is excluded from epsilon selection. If reached, it is a secondary locked extension.

## Predeclared search

- Entropic epsilon: 0.01, 0.03, 0.05, 0.10, 0.20.
- Select the highest mean development accuracy.
- Ties within 0.05 percentage points select the larger epsilon.
- No other OT or fusion parameter is searched.

## Go/no-go rule

Proceed to W2D only if the selected epsilon improves the three-task mean over original SA-TPA by
at least 0.5 percentage points, does not reduce worst-task accuracy, produces finite probabilities,
and has no class with effective sample size below 3. If the gate fails, stop SPT-SA without testing
Office-Home or introducing post-hoc parameters.

## Recorded outcome

The gate failed. Epsilon 0.01 had the highest development mean at 83.905%, below original SA-TPA
at 84.067% on the same three tasks. Its worst-task result was also lower (81.150% versus 81.257%).
All three task accuracies declined and no class had ESS below 3. W2D, the remaining Office-31
tasks, and Office-Home were therefore not run for SPT-SA. No marginal, prior-mix, or fusion-weight
parameter was searched after this result.

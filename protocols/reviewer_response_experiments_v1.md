# Reviewer-response experiments v1

## Purpose

This protocol addresses the four requests in the ICCAID 2026 revision notice:
fusion-weight justification, pseudo-label reliability, class-prior robustness,
and per-transfer-task statistical uncertainty.

## Frozen assets

- Backbone: CLIP ViT-B/32 with OpenAI weights.
- Prompt ensemble: the four templates in `configs/main.yaml`.
- Main SA-TPA weights: text/source/target = 0.875/0.100/0.025.
- Confidence threshold: 0.7.
- Prior-correction strength: 0.1.
- Target labels are never used to choose fusion weights, pseudo-labels, or
  prior estimates. They are read only for final reporting, post-hoc
  pseudo-label diagnostics, controlled noise construction, and controlled
  class-imbalance construction.

## Experiments

1. Fusion strategies: fixed asymmetric, uniform, uncertainty-scaled target,
   class-adaptive, and label-free entropy-selected fusion.
2. Pseudo-label analysis: thresholds 0.5--0.9; confidence, entropy,
   class-balanced, and reliability-weighted selection; injected pseudo-label
   noise from 0% to 50% in 10-point increments, with five repetitions using
   predefined random seeds per level.
3. Prior robustness: estimated-prior permutation mixtures from 0 to 1 and
   controlled target imbalance factors 1, 10, and 50.
4. Statistics: 5,000 paired sample bootstraps per transfer task, exact
   McNemar tests with Holm correction against Prompt Ensemble, T3A, and
   Tip-Adapter, and secondary two-sided cross-task Wilcoxon and sign tests. Existing
   shared-target-domain clustered bootstrap remains the primary safeguard
   against task correlation. Controlled imbalance uses five repetitions with
   predefined random seeds per imbalance factor.

## Outputs

All outputs are written to `D:/456/results/reviewer_response_v1/`. Existing
frozen result files are not modified.

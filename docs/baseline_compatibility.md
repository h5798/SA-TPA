# Baseline compatibility audit

## Directly comparable baselines

- CLIP zero-shot and four-prompt ensemble use the same OpenAI ViT-B/32 weights and frozen image
  features as SA-TPA.
- T3A is implemented as a fixed, offline batch adaptation baseline with five low-entropy target
  supports per class and no source labels.
- The Tip-Adapter-style source cache uses all labeled source features with fixed `alpha=1` and
  `beta=5`; no target-label tuning was performed.

## DPA official code

The archived official DPA repository does not directly implement Office-31 or Office-Home. Its
catalog contains 13 single-dataset adaptation benchmarks and its training configurations are
dataset-specific. More importantly, the official loop evaluates the labeled test set after every
epoch and retains the maximum test accuracy. Running it unchanged would therefore violate this
project's target-label policy and would not be a protocol-compatible main-table comparison.

The official code and PDF remain archived under `D:/456/literature`. A future Office-domain port
must add the datasets, predeclare hyperparameters, report the final epoch rather than the best
target epoch, and be labeled as a port rather than an unchanged official reproduction.

## DPE official code

DPE is an online test-time generalization method. Its released configurations cover the ImageNet
distribution-shift and cross-dataset benchmarks, not Office-31/Office-Home UDA. It also optimizes
prototype residuals with gradients. Any Office-domain result would be a protocol port and should
be separated from directly comparable source-available UDA results.


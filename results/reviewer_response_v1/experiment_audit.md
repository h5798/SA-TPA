# ICCAID reviewer-response experiment audit

## Scope and reproducibility

All experiments use the frozen local CLIP ViT-B/32 features for Office-31 and
Office-Home. No CLIP feature extraction, neural-network training, target-label
configuration selection, or Kaggle computation was required. Target labels are
read only for final accuracy, post-hoc pseudo-label diagnostics, controlled
noise injection, controlled imbalance construction, and statistical tests.

The locked SA-TPA implementation reproduces the frozen main-result means:
84.0253% on Office-31 and 80.8967% on Office-Home.

## 1. Fusion strategies

| Strategy | Office-31 | Office-Home | Interpretation |
|---|---:|---:|---|
| Fixed asymmetric SA-TPA | 84.0253 | 80.8967 | Most stable across the two benchmarks |
| Uniform fusion | 86.3028 | 79.7646 | Higher on Office-31, but 1.1321 pp lower on Office-Home |
| Label-free entropy-selected | 86.5217 | 80.4793 | Higher on Office-31, but 0.4174 pp lower on Office-Home |
| Uncertainty-scaled target | 83.1976 | 80.3433 | Lower on both benchmarks |
| Class-adaptive | 69.8185 | 72.0413 | Strongly unstable |

The defensible claim is cross-dataset stability, not that the fixed coefficients
maximize every development benchmark. The fixed method beats uniform fusion on
9 of 12 Office-Home tasks, whereas uniform fusion wins all six Office-31 tasks.

## 2. Pseudo-label diagnostics

At the locked threshold 0.7, selected pseudo-label accuracy is 92.3004% with
80.04% coverage on Office-31, and 94.0807% with 72.27% coverage on Office-Home.
Increasing the threshold raises selected pseudo-label accuracy but reduces coverage.
At 0.9, selected pseudo-label accuracy reaches 96.4860%/97.7438%, but final accuracy falls to
82.1903%/80.6608%. This supports 0.7 as an accuracy-coverage compromise rather
than a uniquely optimal value.

Reliability weighting changes the mean final accuracy by only +0.0663 pp on
Office-31 and -0.0091 pp on Office-Home relative to confidence-only selection.
It must remain an implementation-level protection rule, not a primary claimed
source of performance gains. Class-balanced entropy selection is competitive
(84.0232% and 80.9892%) but does not consistently dominate.

## 3. Injected pseudo-label noise

With 40% of selected pseudo-labels corrupted, source-anchored fusion retains
83.7085% on Office-31 and 80.8351% on Office-Home. The corresponding values
without the source term in final fusion are 82.8719% and 80.3843%.

Source anchoring therefore preserves a higher absolute accuracy under all
reported noise levels. It does not, however, produce a uniformly smaller drop
from the zero-noise condition; the paper should not claim that stronger form of
noise robustness.

## 4. Prior robustness and controlled imbalance

Blending the estimated prior completely toward a deterministic permuted prior
increases mean L1 prior error from 0.2159 to 0.3875 on Office-31 and from 0.1840
to 0.4155 on Office-Home. Accuracy changes by only -0.0335 pp and -0.0141 pp,
respectively. This shows low sensitivity to prior-estimation error, but also
indicates that prior correction is a modest protective component.

Across controlled imbalance factors 1, 10, and 50, estimated-prior correction
has small and mixed effects. At factor 50 it improves balanced accuracy by
0.1792 pp on Office-31 and 0.1846 pp on Office-Home. At factor 10 it is slightly
worse on Office-31. Claims should therefore emphasize stability rather than a
large imbalance-performance gain.

## 5. Per-task statistical evidence

Against Prompt Ensemble, SA-TPA improves all 6 Office-31 tasks and all 12
Office-Home tasks. Mean gains are +1.5448 pp and +0.8214 pp. Cross-task
two-sided Wilcoxon and sign-test p-values are 0.03125 for Office-31 and 0.000488
for Office-Home. After exact paired McNemar tests and within-benchmark Holm
correction, 3/6 and 7/12 individual tasks remain significant at 0.05.

Against T3A, SA-TPA is lower by 0.5864 pp on Office-31 but higher by 6.7154 pp
on Office-Home. Against the protocol-compatible Tip-Adapter implementation,
SA-TPA is higher by 5.8574 pp and 15.8789 pp. Task-level tests are secondary
because tasks sharing a target domain are correlated; the existing clustered
bootstrap should remain in the manuscript.

## Revision-safe conclusions

1. Keep the fixed asymmetric fusion but position it as the most stable locked
   cross-dataset choice, not the per-benchmark optimum.
2. Report selected pseudo-label accuracy and coverage at threshold 0.7 and show
   the full accuracy-coverage trend.
3. Do not promote uncertainty weighting or prior correction as large isolated
   performance contributors.
4. Use the noise experiment to claim higher retained absolute accuracy under
   corrupted assignments, not a uniformly smaller degradation slope.
5. Add the complete per-task table and confidence intervals, while retaining
   clustered bootstrap due to shared-target dependence.

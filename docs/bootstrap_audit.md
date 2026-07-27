# Shared-target clustered bootstrap audit

The original hierarchical bootstrap treated directed transfer tasks as separate task units even
when they shared the same target images. The corrected analysis groups tasks by target domain and
uses identical sample-resampling indices for all tasks sharing that target.

Two intervals are reported:

- **Conditional interval:** target domains are fixed; target samples are resampled jointly across
  tasks sharing a target.
- **Target-domain cluster interval:** target domains are additionally resampled. This is more
  conservative because Office-31 and Office-Home contain only three and four target-domain clusters.

| Benchmark | Comparison | Mean delta | Conditional 95% CI | Cluster 95% CI | Cluster P(delta<=0) |
|---|---|---:|---:|---:|---:|
| Office-31 | SA-TPA - Prompt Ensemble | +1.545 | [0.950, 2.169] | [0.734, 2.442] | 0.0000 |
| Office-31 | SA-TPA - No Source Anchor | +1.037 | [0.494, 1.596] | [0.085, 1.950] | 0.0106 |
| Office-31 | SA-TPA - T3A | -0.586 | [-1.755, 0.602] | [-3.531, 1.929] | 0.6312 |
| Office-Home | SA-TPA - Prompt Ensemble | +0.821 | [0.577, 1.063] | [0.509, 1.124] | 0.0000 |
| Office-Home | SA-TPA - No Source Anchor | +0.423 | [0.211, 0.636] | [0.126, 0.726] | 0.0032 |
| Office-Home | SA-TPA - T3A | +6.715 | [6.133, 7.308] | [3.223, 9.333] | 0.0000 |

The correction strengthens the interpretation that SA-TPA exceeds Prompt Ensemble on both
benchmarks and supports a positive source-anchor contribution. It does not establish an Office-31
advantage over T3A.

Machine-readable outputs are under `D:/456/results/robustness/clustered_bootstrap/`.

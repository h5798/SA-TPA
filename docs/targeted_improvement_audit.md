# Targeted improvement audit

This is an experiment record, not manuscript text.

## Decision

Retain the original SA-TPA. None of the tested extensions generalizes well enough to replace it.

## Office-31 locked candidate

The simplified agreement-filtered candidate used fixed source and target weights of 0.10, margin
0.05, and one target update. It improved all six tasks.

| Task | Original SA-TPA | Agreement candidate | Delta |
|---|---:|---:|---:|
| A2D | 84.137 | 84.337 | +0.201 |
| A2W | 84.654 | 85.660 | +1.006 |
| D2A | 81.470 | 81.683 | +0.213 |
| D2W | 86.289 | 87.296 | +1.006 |
| W2A | 81.257 | 81.470 | +0.213 |
| W2D | 86.345 | 88.353 | +2.008 |
| Mean | 84.025 | 84.800 | +0.775 |

## Office-Home secondary extension

The same locked candidate averaged 80.634%, compared with 80.897% for original SA-TPA. It won on
7 of 12 tasks but had material regressions on A2P (-2.861 points) and R2P (-1.036 points).

Restoring the original target weight of 0.025 did not fix generalization: the agreement-filtered
variant averaged 80.740%, or -0.157 point versus original SA-TPA.

## Component decisions

| Component | Decision | Evidence |
|---|---|---|
| Class-adaptive fusion | Reject | Large Office-31 failures; class-varying source weights destabilized boundaries. |
| Agreement filter | Reject as main method | Positive on Office-31 but negative on Office-Home. |
| Iterative update | Reject | No A2W accuracy improvement after additional rounds. |
| Original SA-TPA | Retain | Better cross-dataset balance and already locked confirmatory evidence. |

## Result artifacts

- `D:/456/results/adaptive/p0_development.csv`
- `D:/456/results/adaptive/p0_revision.csv`
- `D:/456/results/adaptive/p0_heldout_w2d.csv`
- `D:/456/results/adaptive/p0_remaining_office31.csv`
- `D:/456/results/adaptive/p1_iteration_a2w.csv`
- `D:/456/results/adaptive/officehome_locked_extension.csv`
- `D:/456/results/adaptive/officehome_agreement_original_weight_diagnostic.csv`

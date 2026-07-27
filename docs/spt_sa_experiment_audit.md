# SPT-SA experiment audit

This is an experiment record, not manuscript text.

## Decision

Reject SPT-SA and retain original SA-TPA. The preregistered Office-31 development gate failed.

## Development results

| Epsilon | Mean accuracy | Worst task | Minimum class ESS |
|---:|---:|---:|---:|
| 0.01 | 83.905 | 81.150 | 15.655 |
| 0.03 | 83.638 | 80.724 | 43.693 |
| 0.05 | 83.512 | 80.724 | 133.472 |
| 0.10 | 83.512 | 80.724 | 522.884 |
| 0.20 | 83.488 | 80.653 | 741.608 |
| Original SA-TPA | 84.067 | 81.257 | n/a |

For the selected epsilon 0.01, task accuracies were A2W 84.403%, W2A 81.150%, and D2W
86.164%. Each was below original SA-TPA on the corresponding task.

## Failure diagnosis

| Task | Base pseudo-label accuracy | OT pseudo-label accuracy | Labels changed | Correct to wrong | Wrong to correct |
|---|---:|---:|---:|---:|---:|
| A2W | 84.403 | 84.403 | 4.780% | 13 | 13 |
| W2A | 80.653 | 80.334 | 6.248% | 54 | 45 |
| D2W | 85.409 | 84.277 | 10.063% | 37 | 28 |

The mixed class marginal was closer to the true target class frequency than either source or base
prediction priors on A2W and W2A, but this did not improve instance-level assignments. OT corrected
global class mass while moving too many boundary samples to incorrect classes.

## Protocol compliance

- Predictions were saved before target labels were loaded.
- Only the five preregistered epsilon values were evaluated.
- No class had ESS below 3 and no class lacked a Top-1 assignment.
- W2D and Office-Home were not run because the development gate failed.
- No post-hoc prior mixture, marginal, fallback, or fusion parameter was tested.

Result artifact: `D:/456/results/spt_sa/development_epsilon.csv`.

# SVPR experiment audit

**Status:** rejected at the corrected Office-31 development gate  
**Frozen parent method:** SA-TPA at tag `experiments-final-v1`  
**Implementation commit:** `d1c463ac41ee5fb5c627ae591abb3acf4c2913bf`

## Tested hypothesis

SVPR replaces equal prompt averaging with source-validated, class-specific prompt weights:

`w[c,m] = softmax_m(kappa * mean_{x in source class c} cosine(x, text[c,m]))`.

The weighted text prototype is inserted into the unchanged SA-TPA fusion. No target label is used during adaptation, no CLIP parameter is updated, and the original `0.875/0.100/0.025` fusion is retained.

## Corrected development protocol

- Development tasks: A2W, W2A, D2W.
- Candidate `kappa`: 0, 5, 10, 20.
- Advance only if the best candidate improves the three-task mean by at least 0.4 percentage points and no task drops by more than 0.2 percentage points.
- W2D, A2D, D2A, and Office-Home are not run in the corrected sequence when the gate fails.

The full protocol is recorded in `D:/456/project/protocols/svpr_development_protocol.md`. Target labels were not used to build any prediction, but the comparison of development accuracies across `kappa` values is itself a target-label-informed development comparison. Accordingly, this result is diagnostic and not confirmatory. Because the gate failed, no selected parameter was carried forward.

## Official gate result

| kappa | Development mean | Delta vs SA-TPA | Worst task delta | Decision |
|---:|---:|---:|---:|---|
| 0 | 84.066684% | +0.000000 pp | +0.000000 pp | Control |
| 5 | 84.054851% | -0.011833 pp | -0.035499 pp | Reject |
| 10 | 84.019352% | -0.047332 pp | -0.141995 pp | Reject |
| 20 | 84.138709% | +0.072025 pp | -0.035499 pp | Best, but fails gate |

`kappa=20` is the best non-uniform candidate, but its gain is only 0.072 pp, substantially below the required 0.4 pp. SVPR is therefore not promoted and original SA-TPA remains final.

## Controls and diagnostics

- `kappa=0` exactly reproduces the frozen SA-TPA accuracy on all three development tasks.
- Unit tests verify uniform-weight equivalence, finite nonnegative weights, and per-class weight sums of one.
- Prediction artifacts are written before target labels are loaded.
- The method uses only source features, source labels, and fixed text-template features to calculate prompt weights.
- Result metadata correctly records that target labels were used for development comparison, but not for adaptation.

## Protocol deviation and treatment

Before the sequential gate was corrected, an earlier script evaluated all six Office-31 tasks. Those files remain in `D:/456/results/svpr/office31_sweep.csv` for transparency, but they are excluded from parameter selection and cannot be described as held-out evidence. The official decision uses only `office31_development_gate.csv`.

## Evidence

- Official rows: `D:/456/results/svpr/office31_development_gate.csv`
- Machine-readable decision: `D:/456/results/svpr/svpr_development_summary.json`
- Integrity hashes: `D:/456/results/svpr/svpr_development_sha256_manifest.csv`
- Official prediction artifacts: `D:/456/results/svpr/predictions/office31_development_gate/`

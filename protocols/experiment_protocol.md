# SA-TPA experiment protocol

## Method setting

SA-TPA uses three class prototypes: CLIP text prototypes, labeled-source visual prototypes, and
unlabeled-target visual prototypes. CLIP and all neural parameters remain frozen. Adaptation uses
closed-form statistics and performs no gradient backpropagation.

## Datasets

- Office-31: all six directed transfer tasks; development and sensitivity analysis.
- Office-Home: all twelve directed transfer tasks; confirmatory evaluation.
- PACS: optional only after the two primary benchmarks are complete.

## Primary metrics

Top-1 accuracy, macro-F1, ECE, NLL, per-class recall, runtime, peak GPU memory, additional memory,
and the number of negative-transfer tasks.

## Confirmatory success criteria

- Mean Office-Home improvement over zero-shot CLIP is at least 1.0 percentage point.
- At least 8 of 12 tasks avoid a decline greater than 0.5 percentage points.
- The worst decline is no more than 1.5 percentage points.
- Full SA-TPA outperforms the no-source-anchor ablation (`alpha_s = 0`).
- Calibration does not deteriorate materially across the benchmark.

## Required baselines

CLIP zero-shot, prompt ensemble, source visual prototype, T3A, Tip-Adapter-style source cache,
DPA or a clearly marked faithful reproduction, DPE where protocol-compatible, and SA-TPA.


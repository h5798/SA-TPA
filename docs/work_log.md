# SA-TPA work log

## 2026-07-27 — Day 1 project initialization

- Locked the topic name and acronym: Backpropagation-Free Source-Anchored Tri-Prototype
  Adaptation for Vision-Language Models (SA-TPA).
- Created the isolated project root at `D:/456`; no result from `D:/123` is treated as a new
  SA-TPA result.
- Initialized the local Git repository and environment/configuration files.
- Copied Office-31 and Office-Home into `D:/456/data/raw`.
- Validated Office-31: 4,110 images, 31 common classes, three domains.
- Validated Office-Home: 15,588 images, 65 common classes, four domains.
- Verified local environment: Python 3.8.20, PyTorch 2.4.1+cu118, OpenCLIP 2.26.1,
  NVIDIA GeForce RTX 3050 Laptop GPU (4 GB).
- Archived DPA, DPE, ReCLIP, PTA, and T3A papers; archived DPA and DPE source snapshots.
- Added the target-label policy, experiment protocol, locked-hyperparameter record, environment
  check, data validator, and CLIP feature-precomputation template.
- Hardened the feature format: shared feature files contain no original paths or instance labels;
  source labels are isolated in a source-only sidecar.
- Completed a 498-image Office-31 DSLR smoke test with ViT-B/32. The shared output contains
  512-dimensional image features and no `labels`, `targets`, or `paths` key; source labels were
  isolated in a separate sidecar.
- No full seven-domain feature extraction, adaptation, hyperparameter search, or target-domain
  evaluation was performed during this initialization stage.

## Next execution gate

1. Unit-test text, source, and target prototype construction on one Office-31 task.
2. Establish exact CLIP zero-shot and prompt-ensemble baselines.
3. Compare full SA-TPA against `alpha_source = 0` before broadening the experiment matrix.
4. Lock parameters and config hash before any Office-Home target metric is inspected.

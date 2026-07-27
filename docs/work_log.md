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

## 2026-07-27 — Office-31 development gate

- Extracted target-safe ViT-B/32 features for all seven Office-31/Office-Home domains.
- Completed the six Office-31 transfer tasks with five main development methods.
- Mean accuracy: CLIP zero-shot 81.459%, prompt ensemble 82.481%, no-source-anchor 82.989%,
  and full SA-TPA 84.025%.
- Full SA-TPA improved over prompt ensemble by 1.545 points and over `alpha_source = 0` by
  1.037 points; the source-anchor comparison improved on five tasks and tied on one.
- Completed 138 predeclared one-factor sensitivity runs. Thresholds 0.5--0.8, Top-K 1--8,
  and prior strengths 0--0.2 were stable. Larger prototype weights improved Office-31 accuracy
  but worsened calibration, so the original conservative defaults were retained.
- Component means: text only 82.481%, text+source 83.492%, text+target 82.989%, and full
  tri-prototype 84.025%. Removing uncertainty weighting reduced accuracy by only 0.066 point,
  so uncertainty remains an auxiliary component rather than the primary claim.
- Locked the original parameters for Office-Home before inspecting any Office-Home metric.

## 2026-07-27 — Office-Home confirmation and additional baselines

- Completed all 12 locked Office-Home tasks. SA-TPA averaged 80.897%, improving over single-prompt
  CLIP by 1.785 points and over the four-prompt ensemble by 0.821 point. Every task improved over
  both CLIP baselines.
- SA-TPA improved over `alpha_source = 0` by 0.423 point on average; one task declined by 0.330,
  one tied, and ten improved. The source anchor therefore has positive but nonuniform value.
- Mean ECE improved from 0.0275 for prompt ensemble to 0.0187 for SA-TPA.
- T3A averaged 84.612% on Office-31 but only 74.181% on Office-Home. The source-cache baseline
  averaged 78.168% and 65.018%, respectively.
- Audited DPA and DPE official code. Neither directly supports the Office benchmarks, and unchanged
  DPA selects the best epoch using labeled test accuracy; official-code numbers are therefore not
  treated as protocol-compatible results.
- Completed 5,000-repetition paired hierarchical bootstrap checks. On Office-Home, SA-TPA versus
  prompt ensemble was +0.821 points with 95% CI [0.498, 1.135], and versus no-source-anchor was
  +0.423 points with 95% CI [0.103, 0.732].
- On Office-31, SA-TPA versus prompt ensemble was +1.545 points with 95% CI [0.617, 2.653]; its
  -0.586 difference from T3A was not statistically resolved (95% CI [-3.112, 1.548]).
- Checked five source/target sample permutations for every task. Across 90 shuffled runs there
  were zero prediction changes; maximum floating-point probability drift was below 3.4e-6.
- On representative Office-31/Office-Home tasks, SA-TPA median closed-form adaptation time was
  144/284 ms and measured peak Python/NumPy allocation was 11.6/19.1 MB. It uses no trainable
  parameters and no backpropagation.
- Created and launched the private Kaggle notebook `hrwhrw/sa-tpa-vit-b16-extension` for the
  ViT-B/16 extension. It automatically splits domain extraction across two GPUs when available.
- Kaggle ViT-B/16 version 5 completed successfully. SA-TPA averaged 85.052% on Office-31,
  improving over prompt ensemble by 2.142 points and over no-source-anchor by 1.976 points.
- On Office-Home with ViT-B/16, SA-TPA averaged 84.190%, improving over prompt ensemble by
  0.882 point and over no-source-anchor by 0.470 point; all 12 tasks exceeded prompt ensemble.
- Downloaded and verified all Kaggle outputs; the aggregate ZIP passed integrity checking. The
  redundant local ViT-B/16 run was stopped after Kaggle completion.

## 2026-07-27 - Targeted improvement audit

- Preregistered a P0 study on Office-31 A2W, W2A, and D2W, with W2D held out.
- Class-adaptive source/target fusion failed severely for every preregistered tau value. An A2W
  isolation run showed that varying source-prototype weight by class was the primary failure mode.
- Replaced the rejected adaptive fusion with a simpler agreement filter while keeping global
  fusion weights. With source/target weights 0.10/0.10, the six-task Office-31 mean increased from
  84.025% to 84.800%; all six tasks improved and W2D improved by 2.008 points.
- A preregistered iterative update made no accuracy change on A2W: one, two, and three requested
  rounds all produced 85.660%. Iteration was therefore rejected.
- The locked agreement candidate was then evaluated on all 12 Office-Home tasks as a secondary
  extension. It averaged 80.634%, below original SA-TPA at 80.897% by 0.263 point, with seven wins
  and five losses. A mechanism diagnostic using the original target weight 0.025 averaged 80.740%,
  also below original SA-TPA by 0.157 point.
- Decision: do not promote class-adaptive fusion, agreement filtering, or iterative updating.
  Retain original SA-TPA as the main method. Preserve all negative results as exploratory evidence.
- Audited frozen text features across domains. Class ordering matched exactly, and text/prompt
  feature arrays had zero maximum absolute difference within each dataset.

## 2026-07-27 - SPT-SA optimal-transport gate

- Preregistered and implemented log-domain Sinkhorn soft assignment with the original fixed
  SA-TPA fusion geometry. Eleven unit tests passed before target accuracy was inspected.
- Evaluated epsilon values 0.01, 0.03, 0.05, 0.10, and 0.20 on A2W, W2A, and D2W only.
- Epsilon 0.01 was best at 83.905% mean accuracy, below original SA-TPA at 84.067%. Worst-task
  accuracy also declined from 81.257% to 81.150%, so the preregistered gate failed.
- All classes had effective sample size above 3 and every class received at least one Top-1
  assignment. The failure was not caused by missing-class coverage or numerical collapse.
- Post-gate diagnosis found that OT changed 4.8%--10.1% of pseudo-labels. On W2A and D2W it
  converted more correct predictions to errors than errors to correct predictions; D2W pseudo-label
  accuracy declined from 85.409% to 84.277%.
- Stopped SPT-SA without evaluating W2D or Office-Home and without searching additional priors,
  marginal mixtures, fallback rules, or fusion weights. Original SA-TPA remains the final method.

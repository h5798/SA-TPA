# SA-TPA Claims–Evidence Index

## Scope

This index defines which experimental claims are supported by the frozen evidence. It is an audit artifact, not paper prose. The frozen method is the original fixed-weight SA-TPA at Git commit `ac7ec4c6709326d2ee4d629d103fd119d75e1496`, tagged `experiments-final-v1`.

## Claim 1 — SA-TPA improves over the frozen CLIP prompt ensemble

- **Office-31 (ViT-B/32):** 84.025% versus 82.481%, a gain of 1.545 percentage points.
  - Primary evidence: `D:/456/results/office31/development_v2.csv`
  - Consolidated table: `D:/456/project/tables/office31_main.csv`
  - Shared-target cluster 95% CI for the mean gain: [0.734, 2.442] pp; one-sided bootstrap probability of a non-positive gain: 0.0000.
- **Office-Home (ViT-B/32):** 80.897% versus 80.075%, a gain of 0.821 percentage points.
  - Primary evidence: `D:/456/results/officehome/confirmatory_v1.csv`
  - Consolidated table: `D:/456/project/tables/officehome_main.csv`
  - Shared-target cluster 95% CI for the mean gain: [0.509, 1.124] pp; one-sided bootstrap probability of a non-positive gain: 0.0000.
- Statistical evidence: `D:/456/results/robustness/clustered_bootstrap/clustered_bootstrap_summary.csv`
- Audit method: `D:/456/project/docs/bootstrap_audit.md`

**Allowed wording:** SA-TPA consistently improves over the fixed four-prompt CLIP ensemble on both Office benchmarks under the project protocol.

**Disallowed wording:** state-of-the-art, universal improvement, or superiority over every adaptation method.

## Claim 2 — The source anchor makes a positive contribution

- **Office-31:** SA-TPA exceeds the no-source-anchor variant by 1.037 pp; shared-target cluster 95% CI [0.085, 1.950] pp.
- **Office-Home:** SA-TPA exceeds the no-source-anchor variant by 0.423 pp; shared-target cluster 95% CI [0.126, 0.726] pp.
- Primary evidence:
  - `D:/456/results/ablations/ablations_v1.csv`
  - `D:/456/results/officehome/confirmatory_v1.csv`
  - `D:/456/results/robustness/clustered_bootstrap/clustered_bootstrap_summary.csv`
- Consolidated table: `D:/456/project/tables/ablation_summary.csv`

**Allowed wording:** the labeled-source visual prototype provides a small but statistically positive anchor under both benchmark protocols.

## Claim 3 — SA-TPA requires no backpropagation or trainable parameters

- All final adaptation components are closed-form feature/prototype operations.
- Efficiency records mark `Backpropagation=False` and `Trainable parameters=0`.
- Primary evidence:
  - `D:/456/results/robustness/office31_efficiency.csv`
  - `D:/456/results/robustness/officehome_efficiency.csv`
- Consolidated table: `D:/456/project/tables/efficiency_summary.csv`

**Qualification:** the reported “Peak host allocation” is Python/NumPy host allocation measured during the adaptation stage, not CUDA peak memory and not end-to-end CLIP feature-extraction memory.

## Claim 4 — The method remains effective with ViT-B/16

- **Office-31:** SA-TPA reaches 85.052%, exceeding its ViT-B/16 prompt ensemble by 2.142 pp.
- **Office-Home:** SA-TPA reaches 84.190%, exceeding its ViT-B/16 prompt ensemble by 0.882 pp.
- Primary evidence: `D:/456/results/vitb16/vitb16_summary.csv`
- Consolidated table: `D:/456/project/tables/backbone_comparison.csv`

**Qualification:** ViT-B/16 was executed on Kaggle. Dataset counts, class order, class counts, prompts, and mappings were audited; raw-image byte identity between local and Kaggle replicas is not asserted. See `D:/456/project/docs/data_replica_audit.md`.

## Claim 5 — The fixed design was retained after controlled extension tests

- Class-adaptive fusion: rejected at the preregistered Office-31 gate (−6.841 pp on the three development tasks at the best tested tau).
- Agreement filtering: +0.775 pp on Office-31 but −0.263 pp on Office-Home; rejected for failure to generalize across datasets.
- Iterative update: no change on the A2W gate task.
- SPT-SA optimal transport: −0.161 pp on the three-task Office-31 gate; rejected.
- Evidence:
  - `D:/456/results/adaptive/`
  - `D:/456/results/spt_sa/development_epsilon.csv`
  - `D:/456/project/docs/adaptive_satpa_targeted_audit.md`
  - `D:/456/project/docs/spt_sa_experiment_audit.md`
- Consolidated table: `D:/456/project/tables/design_validation.csv`

**Allowed wording:** fixed fusion was an evidence-based stability choice after controlled tests of more complex alternatives.

## Claim 6 — Baselines must be interpreted by protocol

- Same-project main-table baselines: CLIP Zero-Shot, Prompt Ensemble, Source Prototype, protocol-compatible T3A, project Tip-Adapter source cache, and SA-TPA.
- T3A exceeds SA-TPA on Office-31 by 0.586 pp; SA-TPA exceeds T3A on Office-Home by 6.715 pp. Therefore no overall dominance claim is allowed.
- DPA and DPE are setting comparisons only:
  - DPA is source-free and gradient-based; its official repository does not provide the frozen Office protocol used here and selects the best target-test epoch.
  - DPE is an online test-time adaptation setting with gradient optimization of prototype residuals.
- Evidence:
  - `D:/456/project/tables/baseline_setting_comparison.csv`
  - `D:/456/project/docs/baseline_compatibility.md`

## Reproducibility and integrity evidence

- Freeze record: `D:/456/project/docs/experiment_freeze_manifest.md`
- Result CSV hashes: `D:/456/results/audits/result_csv_sha256_manifest.csv`
- Data-replica audit: `D:/456/project/docs/data_replica_audit.md`
- Final workbook: `D:/456/project/outputs/final_assets/SA-TPA_Final_Experiment_Assets.xlsx`
- Final deliverable hashes: `D:/456/project/outputs/final_assets/deliverables_sha256_manifest.csv`


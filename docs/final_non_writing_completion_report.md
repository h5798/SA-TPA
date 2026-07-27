# Final Non-Writing Completion Report

## Decision

The experimental phase is closed. The retained method is the original fixed-weight SA-TPA. No further method search, parameter tuning, or benchmark expansion is required before manuscript preparation.

Post-freeze addendum: Source-Validated Prompt Reweighting (SVPR) was subsequently tested under a corrected three-task Office-31 development gate. Its best gain was only +0.072 pp, so it was rejected and did not alter the retained method.

Frozen configuration:

- Text/source/target fusion weights: 0.875 / 0.100 / 0.025
- Confidence threshold: 0.7
- Top-k: 1
- Prior strength: 0.1
- Prototype update rounds: 1
- Backbone evidence: ViT-B/32 locally and ViT-B/16 on Kaggle
- Git commit: `ac7ec4c6709326d2ee4d629d103fd119d75e1496`
- Git tag: `experiments-final-v1`

## Completed deliverables

1. **Experiment freeze and integrity**
   - Annotated Git tag created.
   - Frozen method, rejected extensions, result inventory, and SHA256 hashes recorded.
   - Files: `docs/experiment_freeze_manifest.md`, `D:/456/results/audits/result_csv_sha256_manifest.csv`.

2. **Statistical audit**
   - Paired resampling shares sample indices between compared methods.
   - Conditional intervals preserve the benchmark target-domain set.
   - Target-domain cluster intervals quantify sensitivity to the small set of target domains.
   - Files: `docs/bootstrap_audit.md`, `D:/456/results/robustness/clustered_bootstrap/`.

3. **Data replica audit**
   - Local ViT-B/32 and Kaggle ViT-B/16 artifacts agree on benchmark totals, class order, per-class counts, prompt templates, and class-name mappings.
   - Office-Home within-domain label order differs across replicas; each feature file remains aligned with its own label sidecar.
   - Raw-image byte identity is not asserted.
   - Files: `docs/data_replica_audit.md`, `D:/456/results/audits/data_replica_audit.json`.

4. **Final tables**
   - Nine CSV tables and nine matching LaTeX tables generated under `tables/`.
   - Coverage: two main benchmarks, backbone extension, ablations, sensitivity, efficiency, corrected bootstrap, design validation, and baseline-setting compatibility.

5. **Final figures**
   - Five figures generated in both PNG and PDF under `figures/`.
   - Coverage: benchmark means, task-level gains, sensitivity, accuracy–latency trade-off, and method overview.

6. **Final experiment workbook**
   - A verified 11-sheet Excel workbook consolidates the frozen results and their qualifications.
   - Formula error scan: no spreadsheet formula errors detected.
   - File: `outputs/final_assets/SA-TPA_Final_Experiment_Assets.xlsx`.

7. **Evidence traceability**
   - Every allowed result claim is mapped to its frozen source, with explicit disallowed overclaims and protocol qualifications.
   - File: `docs/claims_evidence_index.md`.

8. **Delivery integrity**
   - SHA256 manifest generated for final tables, figures, workbook, and audit documents.
   - File: `outputs/final_assets/deliverables_sha256_manifest.csv`.

9. **Post-freeze SVPR candidate**
   - Tested `kappa` values 0, 5, 10, and 20 on A2W, W2A, and D2W.
   - Best result: `kappa=20`, +0.072 pp mean gain; failed the +0.4 pp gate.
   - Corrected sequence stopped before W2D, A2D, D2A, and Office-Home.
   - Files: `docs/svpr_experiment_audit.md`, `D:/456/results/svpr/svpr_development_summary.json`.

## Final verified headline results

| Benchmark | SA-TPA | Prompt Ensemble | Gain |
|---|---:|---:|---:|
| Office-31, ViT-B/32 | 84.025% | 82.481% | +1.545 pp |
| Office-Home, ViT-B/32 | 80.897% | 80.075% | +0.821 pp |
| Office-31, ViT-B/16 | 85.052% | 82.910% | +2.142 pp |
| Office-Home, ViT-B/16 | 84.190% | 83.308% | +0.882 pp |

## Known limitations to preserve

- The evidence supports a stable, backpropagation-free source-available UDA method; it does not establish state-of-the-art performance.
- T3A is stronger on Office-31 under the project implementation, while SA-TPA is substantially stronger on Office-Home.
- DPA and DPE were not ported into the same Office protocol and must not be presented as directly reproduced accuracy baselines.
- Efficiency memory values are host allocations during adaptation, not end-to-end GPU-memory measurements.
- The local and Kaggle dataset replicas are protocol-compatible, but byte-identical raw imagery was not verified.
- The uncertainty component contributes only marginally and is slightly negative on Office-Home; the defensible core contribution is the fixed source-anchored tri-prototype fusion and zero-backpropagation adaptation.
- Source-validated prompt-template preferences did not provide a material development gain and are not part of the final method.

## Stop condition

All planned non-writing tasks are complete after the final verification checks and hash manifest pass. Further experiments should only be reopened if a reviewer, coauthor, or explicit manuscript requirement identifies a concrete missing comparison.

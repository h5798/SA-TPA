# Experiment freeze manifest

- Freeze tag: `experiments-final-v1`
- Tagged experiment commit: `ac7ec4c6709326d2ee4d629d103fd119d75e1496`
- Manifest generated: `2026-07-27T14:58:55.319737+08:00`
- Final method: original fixed-weight SA-TPA
- Fusion: text/source/target = 0.875/0.100/0.025
- Backbone evidence: OpenAI CLIP ViT-B/32 and ViT-B/16

## Rejected exploratory extensions

- Class-adaptive fusion: rejected.
- Cross-prototype agreement filtering: rejected after cross-dataset regression.
- Iterative target update: rejected after no gain.
- SPT-SA optimal transport: rejected at the preregistered development gate.

## Result CSV integrity

The SHA256 manifest contains 24 CSV files and is stored at `D:/456/results/audits/result_csv_sha256_manifest.csv`.

## Audit references

- `D:/456/project/docs/targeted_improvement_audit.md`
- `D:/456/project/docs/spt_sa_experiment_audit.md`
- `D:/456/project/docs/baseline_compatibility.md`
- `D:/456/project/docs/data_replica_audit.md`

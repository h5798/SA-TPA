# Data replica audit

This audit compares local ViT-B/32 feature inputs with the completed Kaggle ViT-B/16 outputs.

| Dataset | Domain | Local samples | Kaggle samples | Class order | Per-class counts | Label sequence | Prompts |
|---|---|---:|---:|---|---|---|---:|
| office31 | amazon | 2817 | 2817 | match | match | match | 4/4 |
| office31 | dslr | 498 | 498 | match | match | match | 4/4 |
| office31 | webcam | 795 | 795 | match | match | match | 4/4 |
| officehome | art | 2427 | 2427 | match | match | different order | 4/4 |
| officehome | clipart | 4365 | 4365 | match | match | different order | 4/4 |
| officehome | product | 4439 | 4439 | match | match | different order | 4/4 |
| officehome | real_world | 4357 | 4357 | match | match | different order | 4/4 |

## Summary

- Sample counts: all match.
- Class ordering: all match.
- Per-class sample counts: all match.
- Label-sidecar sequences: Office-Home uses a different within-domain sample order.
- Four prompt templates are present in both extraction scripts: True.
- ViT-B/32 and ViT-B/16 embeddings are not compared numerically because they are different backbones.

## Scope limitation

Sample IDs are positional identifiers internal to each feature file and do not prove image identity. The audit verifies total/per-class counts, class order, prompt count/templates, and mapping rules. It does not establish byte-identical raw images between local and Kaggle copies.

Machine-readable audit: `D:/456/results/audits/data_replica_audit.json`.

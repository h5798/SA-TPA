# SA-TPA

Official research code and experiment records for **Backpropagation-Free
Source-Anchored Tri-Prototype Adaptation for Vision-Language Models**.

SA-TPA adapts a frozen CLIP classifier without gradient updates. It combines
three class-level evidence sources: text prototypes, labeled-source visual
prototypes, and target prototypes estimated from reliability-ranked unlabeled
target samples.

## Method at a glance

- Frozen CLIP image and text encoders; no backpropagation during adaptation.
- Source-anchored base prototypes for target scoring and empty-class fallback.
- Prior-corrected, confidence-ranked target prototype estimation.
- Fixed asymmetric text/source/target fusion: `0.875 / 0.100 / 0.025`.
- Office-31 and Office-Home evaluation with CLIP ViT-B/32; ViT-B/16 is included
  as a cross-backbone extension.

## Repository contents

```text
configs/      Experiment and baseline configuration
src/          Feature utilities, metrics, and SA-TPA implementations
scripts/      Experiment, audit, bootstrap, and asset-generation scripts
tests/        Unit and result-schema tests
protocols/    Locked protocols and target-label policy
results/      Lightweight reviewer-response result tables
tables/       Paper-ready CSV and LaTeX summaries
figures/      Main paper figures
kaggle/       ViT-B/16 extension notebook and metadata
docs/         Experiment audits, evidence index, and work log
```

Raw datasets, CLIP weights, cached features, prediction arrays, and third-party
repositories are deliberately excluded.

## Environment

The experiments were developed in the Conda environment `YOLO`:

```powershell
conda activate YOLO
pip install -r requirements.txt
python scripts/environment_check.py
python -m pytest -q
```

For a clean environment, `environment.yml` records the intended dependencies.
Kaggle-specific dependencies are listed in `requirements-kaggle.txt`.

## Data layout

The scripts keep large research assets outside the Git repository. Place the
repository beside `data/` and `results/` in a workspace such as:

```text
workspace/
  SA-TPA/                 # this repository
  data/
    processed/clip_features/
  results/                # generated detailed outputs and predictions
```

Feature archives follow names such as
`office31_amazon_vitb32_openai.npz`. Dataset validation and CLIP feature
precomputation are provided by:

```powershell
python scripts/validate_data.py
python scripts/precompute_clip_features.py --help
```

## Reproducing the main experiments

```powershell
python scripts/verify_lock.py
python scripts/run_task.py --help
python scripts/summarize_experiments.py
```

Windows matrix launchers are available for the Office-31 development tasks,
Office-Home locked evaluation, ablations, sensitivity analysis, and additional
baselines. The reviewer-response experiments are generated with:

```powershell
python scripts/run_reviewer_response_experiments.py
```

The exact target-label restrictions are documented in
`protocols/target_label_policy.md`. All configurations are locked before any
Office-Home evaluation; Office-Home labels are not used for model or
hyperparameter selection. Later Office-Home analyses are confirmatory or
post-hoc robustness evaluations under the locked protocol.

## Reproducibility records

- `docs/experiment_freeze_manifest.md` records the frozen experiment version.
- `docs/claims_evidence_index.md` maps paper claims to result artifacts.
- `protocols/reviewer_response_experiments_v1.md` defines the revision tests.
- `results/reviewer_response_v1/` contains compact result tables and hashes.
- Tag `experiments-final-v1` marks the original experiment freeze.

## Hardware

The lightweight closed-form adaptation runs locally on an RTX 3050 after CLIP
feature extraction. The ViT-B/16 extension notebook was run on Kaggle GPUs.

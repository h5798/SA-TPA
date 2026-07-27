# SA-TPA

Backpropagation-Free Source-Anchored Tri-Prototype Adaptation for Vision-Language Models.

This repository contains the implementation and experiment controls for SA-TPA. All datasets,
features, caches, results, logs, Kaggle assets, and literature are stored under `D:/456`.

## Main protocol

- Backbone: CLIP ViT-B/32 with OpenAI weights.
- Source labels may be used to construct source visual prototypes.
- Target labels are forbidden during adaptation, model selection, threshold selection, and stopping.
- Office-31 is the development benchmark.
- Hyperparameters are locked before Office-Home evaluation.
- Office-Home is the confirmatory benchmark and contains all 12 transfer tasks.

Activate the local environment with `conda activate YOLO` before running scripts.


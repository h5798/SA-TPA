# SA-TPA literature and positioning record

Last verified: 2026-07-27

## Locked method setting

SA-TPA is a **source-available unsupervised domain adaptation** method. It uses labeled source
images, unlabeled target images, and CLIP class text. Its three prototypes are:

1. frozen CLIP text prototypes;
2. labeled-source visual prototypes, used as an anchor;
3. uncertainty-weighted unlabeled-target visual prototypes.

No neural-network parameter or prototype residual is optimized by gradient descent. Prototype
construction and fusion are closed-form operations over frozen CLIP features.

## Closest methods and protocol boundary

| Method | Setting | Source labels | Gradient adaptation | Main distinction from SA-TPA |
|---|---|---:|---:|---|
| DPA | source-free UDA | No | Yes | target image/text dual prototypes; self-training and prototype alignment |
| DPE | online TTA | No | Yes, for prototype residuals | evolving dual prototypes without a labeled-source anchor |
| PTA | TTA | No | method-specific | text-anchored online prototype update, not source-available whole-domain UDA |
| ReCLIP | source-free UDA | No | Yes | cross-modal target self-training without the proposed source anchor |
| T3A | TTA/DG adaptation | No | No | test-time classifier adjustment without tri-prototype source anchoring |
| SA-TPA | source-available UDA | Yes | No | source anchor + text + target prototypes, fully backpropagation-free |

DPA's reported improvement over CLIP is not directly comparable to Office-31 or Office-Home
results unless the same datasets, backbone, prompt set, and evaluation protocol are reproduced.
The main comparison table must mark source-free, source-available, and TTA settings explicitly.

## Critical falsification test

The `alpha_source = 0` ablation is mandatory and is the first Go/No-Go test. If full SA-TPA does
not consistently outperform this no-source-anchor version, the claimed value of source anchoring
is unsupported and the method must be revised or discontinued before Office-Home confirmation.

## Local archive

- `D:/456/literature/papers/DPA_WACV2025.pdf`
- `D:/456/literature/papers/DPE_arXiv_2410.12790.pdf`
- `D:/456/literature/papers/ReCLIP_WACV2024.pdf`
- `D:/456/literature/papers/PTA_arXiv_2604.21360.pdf`
- `D:/456/literature/papers/T3A_NeurIPS2021.pdf`
- `D:/456/literature/code/DPA-main`
- `D:/456/literature/code/DPE-CLIP-main`

The GitHub source archives were downloaded from the repositories' `main` branches on the date
above. Before a formal baseline run, record an immutable commit hash in the result metadata.

## Primary sources

- DPA paper: <https://openaccess.thecvf.com/content/WACV2025/html/Ali_DPA_Dual_Prototypes_Alignment_for_Unsupervised_Adaptation_of_Vision-Language_Models_WACV_2025_paper.html>
- DPA code: <https://github.com/sathiiii/DPA>
- DPE paper: <https://arxiv.org/abs/2410.12790>
- DPE code: <https://github.com/zhangce01/DPE-CLIP>
- ReCLIP paper: <https://openaccess.thecvf.com/content/WACV2024/html/Hu_ReCLIP_Refine_Contrastive_Language_Image_Pre-Training_With_Source_Free_Domain_WACV_2024_paper.html>
- T3A paper: <https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html>


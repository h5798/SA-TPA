# Target-label policy

## Permitted

Target labels may be loaded only after predictions for a complete, predeclared configuration have
been saved. They may then be used to compute final accuracy, macro-F1, NLL, ECE, per-class recall,
confidence intervals, and failure-analysis figures.

## Forbidden

Target labels must not be used for:

- prototype construction or updating;
- pseudo-label filtering or weighting;
- hyperparameter or prompt selection;
- early stopping or model selection;
- choosing a target sample order;
- deciding which reported runs to retain;
- retrying a run based on its target accuracy.

Office-31 is explicitly designated as the development benchmark. Any parameter informed by its
reported target metrics must be frozen before Office-Home begins. Office-Home is confirmatory and
its labels cannot cause parameter changes.

Every result summary must include the fields
`target_labels_used_for_adaptation_or_selection` and
`target_labels_used_only_for_final_reporting`.

Shared CLIP feature files contain neither instance labels nor original/relative file paths. Source
labels, when needed, are stored in a separate `.source_labels.npz` sidecar. For every transfer
task, the loader must reject a target-domain label sidecar and may load only the declared source
domain's sidecar. Class names are treated as the known shared label vocabulary, not as instance
labels.

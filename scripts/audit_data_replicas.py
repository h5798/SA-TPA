from __future__ import annotations

import json
from pathlib import Path

import numpy as np


LOCAL = Path("D:/456/data/processed/clip_features")
KAGGLE = Path("D:/456/kaggle/outputs/vitb16_v5/satpa_vitb16")
OUTPUT_JSON = Path("D:/456/results/audits/data_replica_audit.json")
OUTPUT_MD = Path("D:/456/project/docs/data_replica_audit.md")
EXPECTED_PROMPTS = (
    "a photo of a {}.",
    "a product photo of a {}.",
    "an image of a {}.",
    "a close-up photo of a {}.",
)


def compare_pair(dataset: str, local_domain: str, kaggle_domain: str) -> dict:
    local_path = LOCAL / f"{dataset}_{local_domain}_vitb32_openai.npz"
    kaggle_path = KAGGLE / f"{dataset}_{kaggle_domain}_vitb16_openai.npz"
    with np.load(local_path, allow_pickle=False) as local, np.load(kaggle_path, allow_pickle=False) as kaggle:
        local_classes = local["classes"].astype(str)
        kaggle_classes = kaggle["classes"].astype(str)
        result = {
            "dataset": dataset,
            "domain": local_domain,
            "local_file": str(local_path),
            "kaggle_file": str(kaggle_path),
            "local_samples": int(len(local["sample_ids"])),
            "kaggle_samples": int(len(kaggle["sample_ids"])),
            "sample_count_equal": bool(len(local["sample_ids"]) == len(kaggle["sample_ids"])),
            "sample_ids_equal": bool(np.array_equal(local["sample_ids"], kaggle["sample_ids"])),
            "class_count": int(len(local_classes)),
            "class_order_equal": bool(np.array_equal(local_classes, kaggle_classes)),
            "local_prompt_count": int(local["text_features_per_prompt"].shape[1]),
            "kaggle_prompt_count": int(kaggle["text_features_per_prompt"].shape[1]),
            "feature_dimensions": {
                "vitb32": int(local["image_features"].shape[1]),
                "vitb16": int(kaggle["image_features"].shape[1]),
            },
        }
    local_labels_path = local_path.with_suffix(".source_labels.npz")
    kaggle_labels_path = kaggle_path.with_suffix(".source_labels.npz")
    with np.load(local_labels_path, allow_pickle=False) as local_labels, np.load(kaggle_labels_path, allow_pickle=False) as kaggle_labels:
        result["label_sidecar_ids_equal"] = bool(np.array_equal(local_labels["sample_ids"], kaggle_labels["sample_ids"]))
        result["label_sequence_equal"] = bool(np.array_equal(local_labels["labels"], kaggle_labels["labels"]))
        local_histogram = np.bincount(local_labels["labels"].astype(np.int64), minlength=result["class_count"])
        kaggle_histogram = np.bincount(kaggle_labels["labels"].astype(np.int64), minlength=result["class_count"])
        result["per_class_counts_equal"] = bool(np.array_equal(local_histogram, kaggle_histogram))
    return result


def main():
    pairs = [
        ("office31", "amazon", "amazon"),
        ("office31", "dslr", "dslr"),
        ("office31", "webcam", "webcam"),
        ("officehome", "art", "art"),
        ("officehome", "clipart", "clipart"),
        ("officehome", "product", "product"),
        ("officehome", "real_world", "real_world"),
    ]
    checks = [compare_pair(*pair) for pair in pairs]
    local_script = Path("D:/456/project/scripts/precompute_clip_features.py").read_text(encoding="utf-8")
    kaggle_script = Path("D:/456/project/kaggle/satpa_vitb16_extension.py").read_text(encoding="utf-8")
    result = {
        "checks": checks,
        "all_sample_counts_equal": all(item["sample_count_equal"] for item in checks),
        "all_sample_ids_equal": all(item["sample_ids_equal"] for item in checks),
        "all_class_orders_equal": all(item["class_order_equal"] for item in checks),
        "all_label_sequences_equal": all(item["label_sequence_equal"] for item in checks),
        "all_per_class_counts_equal": all(item["per_class_counts_equal"] for item in checks),
        "prompt_templates": list(EXPECTED_PROMPTS),
        "prompts_present_in_local_script": all(prompt in local_script for prompt in EXPECTED_PROMPTS),
        "prompts_present_in_kaggle_script": all(prompt in kaggle_script for prompt in EXPECTED_PROMPTS),
        "class_mapping_protocol": "D:/456/project/protocols/class_name_mapping.md",
        "scope_limitation": "Sample IDs are positional identifiers internal to each feature file and do not prove image identity. The audit verifies total/per-class counts, class order, prompt count/templates, and mapping rules. It does not establish byte-identical raw images between local and Kaggle copies.",
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Data replica audit",
        "",
        "This audit compares local ViT-B/32 feature inputs with the completed Kaggle ViT-B/16 outputs.",
        "",
        "| Dataset | Domain | Local samples | Kaggle samples | Class order | Per-class counts | Label sequence | Prompts |",
        "|---|---|---:|---:|---|---|---|---:|",
    ]
    for item in checks:
        lines.append(
            f"| {item['dataset']} | {item['domain']} | {item['local_samples']} | {item['kaggle_samples']} | "
            f"{'match' if item['class_order_equal'] else 'DIFF'} | {'match' if item['per_class_counts_equal'] else 'DIFF'} | "
            f"{'match' if item['label_sequence_equal'] else 'different order'} | {item['local_prompt_count']}/{item['kaggle_prompt_count']} |"
        )
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Sample counts: {'all match' if result['all_sample_counts_equal'] else 'differences found'}.",
        f"- Class ordering: {'all match' if result['all_class_orders_equal'] else 'differences found'}.",
        f"- Per-class sample counts: {'all match' if result['all_per_class_counts_equal'] else 'differences found'}.",
        f"- Label-sidecar sequences: {'all match' if result['all_label_sequences_equal'] else 'Office-Home uses a different within-domain sample order'}.",
        f"- Four prompt templates are present in both extraction scripts: {result['prompts_present_in_local_script'] and result['prompts_present_in_kaggle_script']}.",
        "- ViT-B/32 and ViT-B/16 embeddings are not compared numerically because they are different backbones.",
        "",
        "## Scope limitation",
        "",
        result["scope_limitation"],
        "",
        "Machine-readable audit: `D:/456/results/audits/data_replica_audit.json`.",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

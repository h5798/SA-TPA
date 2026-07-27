from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision.datasets import ImageFolder


DATASETS = {
    "office31": ("Office-31", ("amazon", "dslr", "webcam")),
    "officehome": ("OfficeHomeDataset", ("Art", "Clipart", "Product", "Real World")),
}


def validate(root: Path, domains):
    records = {}
    reference_classes = None
    for domain in domains:
        domain_root = root / domain
        if not domain_root.is_dir():
            raise FileNotFoundError(domain_root)
        dataset = ImageFolder(domain_root)
        if reference_classes is None:
            reference_classes = dataset.classes
        elif dataset.classes != reference_classes:
            raise ValueError(f"Class ordering differs in {domain_root}")
        records[domain] = {
            "samples": len(dataset),
            "classes": len(dataset.classes),
            "first_class": dataset.classes[0],
            "last_class": dataset.classes[-1],
        }
    return records


def main(project_root: str, output: str):
    root = Path(project_root)
    manifest = {}
    for name, (folder, domains) in DATASETS.items():
        dataset_root = root / "data" / "raw" / name / folder
        manifest[name] = validate(dataset_root, domains)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="D:/456")
    parser.add_argument("--output", default="D:/456/protocols/data_manifest.json")
    args = parser.parse_args()
    main(args.project_root, args.output)


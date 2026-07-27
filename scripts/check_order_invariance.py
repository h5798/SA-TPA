from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.features import load_frozen_features, load_label_sidecar, sidecar_for
from src.methods import run_method


OFFICE31 = (
    ("A2W", "amazon", "webcam"), ("D2W", "dslr", "webcam"),
    ("W2A", "webcam", "amazon"), ("A2D", "amazon", "dslr"),
    ("D2A", "dslr", "amazon"), ("W2D", "webcam", "dslr"),
)
OFFICEHOME = (
    ("A2C", "art", "clipart"), ("A2P", "art", "product"), ("A2R", "art", "real_world"),
    ("C2A", "clipart", "art"), ("C2P", "clipart", "product"), ("C2R", "clipart", "real_world"),
    ("P2A", "product", "art"), ("P2C", "product", "clipart"), ("P2R", "product", "real_world"),
    ("R2A", "real_world", "art"), ("R2C", "real_world", "clipart"), ("R2P", "real_world", "product"),
)


def main(args):
    tasks = OFFICE31 if args.benchmark == "office31" else OFFICEHOME
    root = Path("D:/456/data/processed/clip_features")
    records = []
    for task, source_name, target_name in tasks:
        source = load_frozen_features(root / f"{args.benchmark}_{source_name}_vitb32_openai.npz")
        target = load_frozen_features(root / f"{args.benchmark}_{target_name}_vitb32_openai.npz")
        labels = load_label_sidecar(sidecar_for(root / f"{args.benchmark}_{source_name}_vitb32_openai.npz"), source.sample_ids)
        reference = run_method("satpa", source.image, labels, target.image, target.text, target.text_per_prompt).probabilities
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            source_order = rng.permutation(len(source.image))
            target_order = rng.permutation(len(target.image))
            shuffled = run_method(
                "satpa", source.image[source_order], labels[source_order], target.image[target_order],
                target.text, target.text_per_prompt,
            ).probabilities
            restored = np.empty_like(shuffled)
            restored[target_order] = shuffled
            records.append({
                "benchmark": args.benchmark,
                "task": task,
                "seed": seed,
                "prediction_mismatches": int((restored.argmax(1) != reference.argmax(1)).sum()),
                "max_probability_difference": float(np.max(np.abs(restored - reference))),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"rows={len(records)} mismatches={sum(row['prediction_mismatches'] for row in records)} "
          f"max_difference={max(row['max_probability_difference'] for row in records):.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("office31", "officehome"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--output", required=True)
    main(parser.parse_args())


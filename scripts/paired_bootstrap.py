from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.features import load_frozen_features, load_label_sidecar, sidecar_for


def load_correctness(prediction_path: Path):
    metadata = json.loads(prediction_path.with_suffix(".json").read_text(encoding="utf-8"))
    features = load_frozen_features(metadata["target_features"])
    labels = load_label_sidecar(sidecar_for(metadata["target_features"]), features.sample_ids)
    with np.load(prediction_path, allow_pickle=False) as data:
        if not np.array_equal(data["sample_ids"], features.sample_ids):
            raise ValueError(f"Prediction IDs do not align: {prediction_path}")
        predictions = data["predictions"].astype(np.int64)
    return predictions == labels


def main(args):
    reference_dir = Path(args.reference_dir)
    comparator_dir = Path(args.comparator_dir)
    reference_files = sorted(reference_dir.glob(f"*_{args.reference_method}.npz"))
    if not reference_files:
        raise FileNotFoundError(f"No reference predictions below {reference_dir}")
    rng = np.random.default_rng(args.seed)
    task_bootstraps = []
    observed = []
    task_names = []
    for reference_path in reference_files:
        task = reference_path.name[: -len(f"_{args.reference_method}.npz")]
        comparator_path = comparator_dir / f"{task}_{args.comparator_method}.npz"
        reference_correct = load_correctness(reference_path)
        comparator_correct = load_correctness(comparator_path)
        delta = reference_correct.astype(np.float32) - comparator_correct.astype(np.float32)
        observed.append(delta.mean() * 100.0)
        indices = rng.integers(0, len(delta), size=(args.repetitions, len(delta)))
        task_bootstraps.append(delta[indices].mean(1) * 100.0)
        task_names.append(task)
    task_bootstraps = np.stack(task_bootstraps, axis=1)
    sampled_tasks = rng.integers(0, len(task_names), size=(args.repetitions, len(task_names)))
    repetitions = np.arange(args.repetitions)[:, None]
    hierarchical = task_bootstraps[repetitions, sampled_tasks].mean(1)
    result = {
        "reference_method": args.reference_method,
        "comparator_method": args.comparator_method,
        "tasks": task_names,
        "task_deltas_percentage_points": dict(zip(task_names, map(float, observed))),
        "mean_delta_percentage_points": float(np.mean(observed)),
        "ci95_low": float(np.quantile(hierarchical, 0.025)),
        "ci95_high": float(np.quantile(hierarchical, 0.975)),
        "bootstrap_probability_delta_le_zero": float((hierarchical <= 0).mean()),
        "repetitions": args.repetitions,
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--reference-method", default="satpa")
    parser.add_argument("--comparator-dir", required=True)
    parser.add_argument("--comparator-method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    main(parser.parse_args())


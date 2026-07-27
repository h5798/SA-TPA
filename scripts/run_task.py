from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.features import load_frozen_features, load_label_sidecar, sidecar_for
from src.methods import run_method
from src.metrics import classification_metrics


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    source = load_frozen_features(args.source_features)
    target = load_frozen_features(args.target_features)
    if not np.array_equal(source.classes, target.classes):
        raise ValueError("Source and target class ordering differs")

    # The target sidecar is deliberately not loaded before predictions are written.
    source_labels = load_label_sidecar(sidecar_for(args.source_features), source.sample_ids)
    start = time.perf_counter()
    output = run_method(
        args.method,
        source.image,
        source_labels,
        target.image,
        target.text,
        target.text_per_prompt,
        alpha_source=args.alpha_source,
        alpha_target=args.alpha_target,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
        class_prior_strength=args.class_prior_strength,
    )
    elapsed = time.perf_counter() - start

    prediction_path = Path(args.prediction_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        sample_ids=target.sample_ids,
        probabilities=output.probabilities.astype(np.float32),
        predictions=output.probabilities.argmax(1).astype(np.int16),
    )
    metadata = {
        "run_tag": args.run_tag,
        "task": args.task,
        "method": args.method,
        "source_features": str(Path(args.source_features).resolve()),
        "target_features": str(Path(args.target_features).resolve()),
        "prediction_output": str(prediction_path.resolve()),
        "git_commit": git_commit(),
        "config_sha256": sha256(args.config),
        "alpha_source": args.alpha_source,
        "alpha_target": args.alpha_target,
        "confidence_threshold": args.confidence_threshold,
        "top_k": args.top_k,
        "class_prior_strength": args.class_prior_strength,
        "runtime_seconds": elapsed,
        "target_labels_used_for_adaptation_or_selection": False,
        "target_labels_used_only_for_final_reporting": True,
        "accepted_samples": None,
        "acceptance_rate": None,
        "mean_uncertainty_weight": None,
        "classes_with_target_support": None,
        **output.diagnostics,
    }
    prediction_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Only after the complete prediction artifact exists may target labels be loaded for reporting.
    target_labels = load_label_sidecar(sidecar_for(args.target_features), target.sample_ids)
    metrics = classification_metrics(output.probabilities, target_labels)
    row = {**metadata, **metrics}
    append_csv(Path(args.results_csv), row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-tag", default="main")
    parser.add_argument("--method", required=True, choices=(
        "clip_zero_shot", "prompt_ensemble", "source_prototype", "source_anchored_text",
        "no_source_anchor", "satpa_no_uncertainty", "satpa"
    ))
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--results-csv", default="D:/456/results/office31/main.csv")
    parser.add_argument("--config", default="D:/456/project/configs/main.yaml")
    parser.add_argument("--alpha-source", type=float, default=0.1)
    parser.add_argument("--alpha-target", type=float, default=0.025)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--class-prior-strength", type=float, default=0.1)
    main(parser.parse_args())

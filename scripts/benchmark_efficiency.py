from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.features import load_frozen_features, load_label_sidecar, sidecar_for
from src.methods import run_method


def main(args):
    source = load_frozen_features(args.source_features)
    target = load_frozen_features(args.target_features)
    labels = load_label_sidecar(sidecar_for(args.source_features), source.sample_ids)
    records = []
    for method in args.methods:
        run_method(method, source.image, labels, target.image, target.text, target.text_per_prompt)
        durations = []
        peaks = []
        for _ in range(args.repetitions):
            tracemalloc.start()
            start = time.perf_counter()
            run_method(method, source.image, labels, target.image, target.text, target.text_per_prompt)
            durations.append((time.perf_counter() - start) * 1000.0)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak / 1024 ** 2)
        ordered = sorted(durations)
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        records.append({
            "benchmark_task": args.task,
            "method": method,
            "median_adaptation_ms": statistics.median(durations),
            "p95_adaptation_ms": ordered[p95_index],
            "peak_python_numpy_memory_mb": max(peaks),
            "repetitions": args.repetitions,
            "backpropagation": False,
            "trainable_parameters": 0,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--methods", nargs="+", default=(
        "prompt_ensemble", "t3a", "tip_adapter_source", "no_source_anchor", "satpa"
    ))
    main(parser.parse_args())


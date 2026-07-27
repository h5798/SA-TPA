"""Run the preregistered SVPR development gate on Office-31.

Only A2W, W2A, and D2W are used at this stage. The script deliberately stops
without evaluating W2D, A2D, D2A, or Office-Home when the development gate
fails. Target labels are loaded only after prediction artifacts are saved.
"""

from __future__ import annotations

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


FEATURES = PROJECT.parent / "data" / "processed" / "clip_features"
RESULTS = PROJECT.parent / "results" / "svpr"
DEVELOPMENT_TASKS = {
    "A2W": ("office31_amazon_vitb32_openai.npz", "office31_webcam_vitb32_openai.npz"),
    "W2A": ("office31_webcam_vitb32_openai.npz", "office31_amazon_vitb32_openai.npz"),
    "D2W": ("office31_dslr_vitb32_openai.npz", "office31_webcam_vitb32_openai.npz"),
}
KAPPA_VALUES = (0.0, 5.0, 10.0, 20.0)
MIN_MEAN_GAIN_PP = 0.4
MAX_ALLOWED_TASK_DROP_PP = 0.2


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()


def load_satpa_reference() -> dict[str, float]:
    path = PROJECT.parent / "results" / "office31" / "development_v2.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reference = {
        row["task"]: float(row["accuracy"])
        for row in rows
        if row["method"] == "satpa" and row["task"] in DEVELOPMENT_TASKS
    }
    if set(reference) != set(DEVELOPMENT_TASKS):
        raise RuntimeError("The frozen SA-TPA reference is incomplete")
    return reference


def run_one(task: str, source_name: str, target_name: str, kappa: float) -> dict:
    source_path = FEATURES / source_name
    target_path = FEATURES / target_name
    source = load_frozen_features(str(source_path))
    target = load_frozen_features(str(target_path))
    if not np.array_equal(source.classes, target.classes):
        raise ValueError(f"Class order differs for {task}")

    source_labels = load_label_sidecar(sidecar_for(str(source_path)), source.sample_ids)
    started = time.perf_counter()
    output = run_method(
        "satpa_svpr",
        source.image,
        source_labels,
        target.image,
        target.text,
        target.text_per_prompt,
        alpha_source=0.1,
        alpha_target=0.025,
        confidence_threshold=0.7,
        top_k=1,
        class_prior_strength=0.1,
        svpr_kappa=kappa,
    )
    elapsed = time.perf_counter() - started

    prediction_dir = RESULTS / "predictions" / "office31_development_gate"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{task}_svpr_k{kappa:g}.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=target.sample_ids,
        probabilities=output.probabilities.astype(np.float32),
        predictions=output.probabilities.argmax(1).astype(np.int16),
    )
    metadata = {
        "task": task,
        "method": "satpa_svpr",
        "svpr_kappa": kappa,
        "source_features": str(source_path.resolve()),
        "target_features": str(target_path.resolve()),
        "prediction_output": str(prediction_path.resolve()),
        "git_commit": git_commit(),
        "runtime_seconds": elapsed,
        "target_labels_used_for_adaptation_or_selection": False,
        "target_labels_loaded_after_prediction_artifact": True,
        **output.diagnostics,
    }
    prediction_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    target_labels = load_label_sidecar(sidecar_for(str(target_path)), target.sample_ids)
    return {**metadata, **classification_metrics(output.probabilities, target_labels)}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_integrity_manifest() -> None:
    files = sorted(
        path for path in RESULTS.rglob("*")
        if path.is_file()
        and path.name != "svpr_development_sha256_manifest.csv"
        and "office31_development_gate" in path.as_posix()
    )
    files.extend([
        RESULTS / "office31_development_gate.csv",
        RESULTS / "svpr_development_summary.json",
    ])
    manifest = RESULTS / "svpr_development_sha256_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        for path in sorted(set(files)):
            writer.writerow({
                "relative_path": path.relative_to(RESULTS).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })


def main() -> None:
    reference = load_satpa_reference()
    reference_mean = float(np.mean(list(reference.values())))
    rows: list[dict] = []
    evaluations: dict[str, dict] = {}

    for kappa in KAPPA_VALUES:
        task_scores: dict[str, float] = {}
        for task, (source_name, target_name) in DEVELOPMENT_TASKS.items():
            row = run_one(task, source_name, target_name, kappa)
            rows.append(row)
            task_scores[task] = float(row["accuracy"])
        deltas = {task: task_scores[task] - reference[task] for task in task_scores}
        mean_score = float(np.mean(list(task_scores.values())))
        evaluations[str(kappa)] = {
            "task_accuracy": task_scores,
            "task_delta_pp": deltas,
            "mean_accuracy": mean_score,
            "mean_delta_pp": mean_score - reference_mean,
            "worst_task_delta_pp": min(deltas.values()),
        }

    write_rows(RESULTS / "office31_development_gate.csv", rows)
    best_kappa = max(KAPPA_VALUES, key=lambda value: evaluations[str(value)]["mean_accuracy"])
    best = evaluations[str(best_kappa)]
    passed = (
        best["mean_delta_pp"] >= MIN_MEAN_GAIN_PP
        and best["worst_task_delta_pp"] >= -MAX_ALLOWED_TASK_DROP_PP
    )
    summary = {
        "experiment": "Source-Validated Prompt Reweighting development gate",
        "git_commit": git_commit(),
        "development_tasks": list(DEVELOPMENT_TASKS),
        "kappa_values": list(KAPPA_VALUES),
        "reference": reference,
        "reference_mean": reference_mean,
        "criteria": {
            "minimum_mean_gain_pp": MIN_MEAN_GAIN_PP,
            "maximum_allowed_task_drop_pp": MAX_ALLOWED_TASK_DROP_PP,
        },
        "evaluations": evaluations,
        "best_kappa": best_kappa,
        "passed": passed,
        "next_stage_run": False,
        "note": "A prior full-six-task diagnostic exists but is excluded from parameter selection because it ran before the development gate was enforced.",
    }
    (RESULTS / "svpr_development_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_integrity_manifest()

    print(f"Frozen SA-TPA development mean: {reference_mean:.6f}%")
    for kappa in KAPPA_VALUES:
        result = evaluations[str(kappa)]
        print(
            f"kappa={kappa:g}: mean={result['mean_accuracy']:.6f}% "
            f"delta={result['mean_delta_pp']:+.6f} pp "
            f"worst={result['worst_task_delta_pp']:+.6f} pp"
        )
    print(f"Selected kappa={best_kappa:g}; gate={'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Stopped before W2D, A2D, D2A, and Office-Home.")


if __name__ == "__main__":
    main()

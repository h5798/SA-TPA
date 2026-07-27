from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.features import load_frozen_features, load_label_sidecar, sidecar_for


RESULTS = Path("D:/456/results")
OUTPUT = RESULTS / "robustness" / "clustered_bootstrap"


def read_rows(*paths: Path) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def method_rows(rows: list[dict], method: str) -> dict[str, dict]:
    selected = {row["task"]: row for row in rows if row["method"] == method}
    if not selected:
        raise ValueError(f"No rows for method {method}")
    return selected


def correctness(row: dict) -> tuple[np.ndarray, str, np.ndarray]:
    prediction_path = Path(row["prediction_output"])
    metadata = json.loads(prediction_path.with_suffix(".json").read_text(encoding="utf-8"))
    features = load_frozen_features(metadata["target_features"])
    labels = load_label_sidecar(sidecar_for(metadata["target_features"]), features.sample_ids)
    with np.load(prediction_path, allow_pickle=False) as data:
        if not np.array_equal(data["sample_ids"], features.sample_ids):
            raise ValueError(f"Prediction/sample mismatch: {prediction_path}")
        predictions = data["predictions"].astype(np.int64)
    target_domain = row["task"].split("2", 1)[1]
    return predictions == labels, target_domain, features.sample_ids


def bootstrap_comparison(
    benchmark: str,
    reference_rows: dict[str, dict],
    comparator_rows: dict[str, dict],
    reference_method: str,
    comparator_method: str,
    repetitions: int = 5000,
    seed: int = 2026,
) -> dict:
    tasks = sorted(reference_rows)
    if tasks != sorted(comparator_rows):
        raise ValueError("Reference/comparator task sets differ")
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    observed_task: dict[str, float] = {}
    domain_ids: dict[str, np.ndarray] = {}
    for task in tasks:
        ref_correct, ref_domain, ref_ids = correctness(reference_rows[task])
        cmp_correct, cmp_domain, cmp_ids = correctness(comparator_rows[task])
        if ref_domain != cmp_domain or not np.array_equal(ref_ids, cmp_ids):
            raise ValueError(f"Shared target alignment failed for {task}")
        delta = ref_correct.astype(np.float32) - cmp_correct.astype(np.float32)
        observed_task[task] = float(delta.mean() * 100.0)
        if ref_domain in domain_ids and not np.array_equal(domain_ids[ref_domain], ref_ids):
            raise ValueError(f"Tasks sharing target {ref_domain} do not share sample IDs")
        domain_ids[ref_domain] = ref_ids
        grouped.setdefault(ref_domain, []).append((task, delta))

    rng = np.random.default_rng(seed)
    domains = sorted(grouped)
    domain_bootstrap = np.empty((repetitions, len(domains)), dtype=np.float64)
    for domain_index, domain in enumerate(domains):
        task_matrix = np.stack([delta for _, delta in grouped[domain]], axis=0)
        n = task_matrix.shape[1]
        for start in range(0, repetitions, 250):
            stop = min(start + 250, repetitions)
            indices = rng.integers(0, n, size=(stop - start, n))
            sampled = task_matrix[:, indices]
            domain_bootstrap[start:stop, domain_index] = sampled.mean(axis=(0, 2)) * 100.0

    conditional = domain_bootstrap.mean(1)
    sampled_domains = rng.integers(0, len(domains), size=(repetitions, len(domains)))
    sampled_rows = rng.integers(0, repetitions, size=(repetitions, len(domains)))
    clustered = domain_bootstrap[sampled_rows, sampled_domains].mean(1)
    observed_domain = {
        domain: float(np.mean([observed_task[task] for task, _ in grouped[domain]]))
        for domain in domains
    }
    result = {
        "benchmark": benchmark,
        "reference_method": reference_method,
        "comparator_method": comparator_method,
        "tasks": tasks,
        "target_domains": domains,
        "shared_target_resampling": True,
        "task_delta_percentage_points": observed_task,
        "target_domain_delta_percentage_points": observed_domain,
        "observed_mean_delta_percentage_points": float(np.mean(list(observed_task.values()))),
        "conditional_fixed_domains_ci95": [
            float(np.quantile(conditional, 0.025)),
            float(np.quantile(conditional, 0.975)),
        ],
        "conditional_probability_delta_le_zero": float((conditional <= 0).mean()),
        "target_domain_cluster_ci95": [
            float(np.quantile(clustered, 0.025)),
            float(np.quantile(clustered, 0.975)),
        ],
        "target_domain_cluster_probability_delta_le_zero": float((clustered <= 0).mean()),
        "repetitions": repetitions,
        "seed": seed,
        "interpretation": {
            "conditional": "Uncertainty conditional on the observed target domains; shared target samples use identical resampling indices.",
            "cluster": "Adds resampling of target-domain clusters; conservative with only three/four observed target domains.",
        },
    }
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configurations = {
        "office31": (
            read_rows(RESULTS / "office31" / "development_v2.csv", RESULTS / "office31" / "additional_baselines_v1.csv"),
            ("prompt_ensemble", "no_source_anchor", "t3a"),
        ),
        "officehome": (
            read_rows(RESULTS / "officehome" / "confirmatory_v1.csv", RESULTS / "officehome" / "additional_baselines_v1.csv"),
            ("prompt_ensemble", "no_source_anchor", "t3a"),
        ),
    }
    summary = []
    for benchmark, (rows, comparators) in configurations.items():
        reference = method_rows(rows, "satpa")
        for comparator in comparators:
            result = bootstrap_comparison(
                benchmark,
                reference,
                method_rows(rows, comparator),
                "satpa",
                comparator,
            )
            path = OUTPUT / f"{benchmark}_satpa_vs_{comparator}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary.append({
                "benchmark": benchmark,
                "reference": "satpa",
                "comparator": comparator,
                "mean_delta_pp": result["observed_mean_delta_percentage_points"],
                "conditional_ci_low": result["conditional_fixed_domains_ci95"][0],
                "conditional_ci_high": result["conditional_fixed_domains_ci95"][1],
                "conditional_p_le_zero": result["conditional_probability_delta_le_zero"],
                "cluster_ci_low": result["target_domain_cluster_ci95"][0],
                "cluster_ci_high": result["target_domain_cluster_ci95"][1],
                "cluster_p_le_zero": result["target_domain_cluster_probability_delta_le_zero"],
                "target_domain_clusters": len(result["target_domains"]),
            })
    with (OUTPUT / "clustered_bootstrap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

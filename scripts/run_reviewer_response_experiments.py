from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import balanced_accuracy_score


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from src.features import load_frozen_features, load_label_sidecar, sidecar_for
from src.methods import classify, normalize, source_prototypes


FEATURES = ROOT / "data" / "processed" / "clip_features"
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "reviewer_response_v1"
SEED = 2026

BENCHMARKS = {
    "office31": {
        "domains": {"A": "amazon", "D": "dslr", "W": "webcam"},
        "prediction_dir": RESULTS / "office31" / "predictions" / "development_v2",
    },
    "officehome": {
        "domains": {"A": "art", "C": "clipart", "P": "product", "R": "real_world"},
        "prediction_dir": RESULTS / "officehome" / "predictions" / "confirmatory_v1",
    },
}


@dataclass
class TaskData:
    benchmark: str
    task: str
    source_features: np.ndarray
    source_labels: np.ndarray
    target_features: np.ndarray
    target_labels: np.ndarray
    target_ids: np.ndarray
    text: np.ndarray
    source: np.ndarray
    base: np.ndarray
    base_probabilities: np.ndarray


def feature_path(benchmark: str, domain: str) -> Path:
    name = BENCHMARKS[benchmark]["domains"][domain]
    return FEATURES / f"{benchmark}_{name}_vitb32_openai.npz"


def tasks_for(benchmark: str) -> list[str]:
    domains = list(BENCHMARKS[benchmark]["domains"])
    return [f"{source}2{target}" for source in domains for target in domains if source != target]


def load_task(benchmark: str, task: str) -> TaskData:
    source_domain, target_domain = task.split("2")
    source_path = feature_path(benchmark, source_domain)
    target_path = feature_path(benchmark, target_domain)
    source_data = load_frozen_features(str(source_path))
    target_data = load_frozen_features(str(target_path))
    if not np.array_equal(source_data.classes, target_data.classes):
        raise ValueError(f"Class ordering differs for {benchmark} {task}")
    source_labels = load_label_sidecar(sidecar_for(str(source_path)), source_data.sample_ids)
    target_labels = load_label_sidecar(sidecar_for(str(target_path)), target_data.sample_ids)
    text = normalize(target_data.text)
    source = source_prototypes(source_data.image, source_labels, text)
    base = normalize(0.9 * text + 0.1 * source)
    base_probabilities = classify(target_data.image, base)
    return TaskData(
        benchmark=benchmark,
        task=task,
        source_features=source_data.image,
        source_labels=source_labels,
        target_features=target_data.image,
        target_labels=target_labels,
        target_ids=target_data.sample_ids,
        text=text,
        source=source,
        base=base,
        base_probabilities=base_probabilities,
    )


def uncertainty_weights(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(1)
    certainty = 1.0 - entropy / np.log(probabilities.shape[1])
    return np.clip(confidence * margin * certainty, 0.0, 1.0).astype(np.float32)


def corrected_probabilities(
    probabilities: np.ndarray,
    strength: float = 0.1,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if prior is None:
        prior = probabilities.mean(0)
    prior = np.clip(np.asarray(prior, dtype=np.float64), 1e-8, None)
    prior /= prior.sum()
    corrected = probabilities.astype(np.float64) / np.power(prior[None, :], strength)
    corrected /= corrected.sum(1, keepdims=True)
    return corrected.astype(np.float32), prior.astype(np.float32)


def prototypes_from_labels(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_count = len(fallback)
    prototypes = []
    counts = np.zeros(class_count, dtype=np.int64)
    effective_sizes = np.zeros(class_count, dtype=np.float32)
    for class_index in range(class_count):
        selected = (labels == class_index) & (weights > 0)
        counts[class_index] = int(selected.sum())
        class_weights = weights[selected].astype(np.float64)
        if not len(class_weights) or class_weights.sum() <= 1e-12:
            prototypes.append(fallback[class_index])
            continue
        effective_sizes[class_index] = float(
            class_weights.sum() ** 2 / np.clip(np.square(class_weights).sum(), 1e-12, None)
        )
        prototype = (class_weights[:, None] * features[selected]).sum(0) / class_weights.sum()
        prototypes.append(normalize(prototype[None, :])[0])
    return normalize(np.stack(prototypes)), counts, effective_sizes


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(1)
    return {
        "accuracy": float((predictions == labels).mean() * 100.0),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions) * 100.0),
    }


def current_target(
    data: TaskData,
    threshold: float = 0.7,
    prior: np.ndarray | None = None,
    prior_strength: float = 0.1,
    fallback: np.ndarray | None = None,
) -> dict:
    corrected, used_prior = corrected_probabilities(data.base_probabilities, prior_strength, prior)
    pseudo = corrected.argmax(1)
    confidence = corrected.max(1)
    accepted = confidence >= threshold
    reliability = uncertainty_weights(corrected)
    weights = confidence * reliability * accepted
    target, counts, effective_sizes = prototypes_from_labels(
        data.target_features, pseudo, weights, data.base if fallback is None else fallback
    )
    return {
        "corrected": corrected,
        "prior": used_prior,
        "pseudo": pseudo,
        "confidence": confidence,
        "accepted": accepted,
        "reliability": reliability,
        "weights": weights,
        "target": target,
        "counts": counts,
        "effective_sizes": effective_sizes,
    }


def final_probabilities(data: TaskData, target: np.ndarray, ws: np.ndarray, wt: np.ndarray) -> np.ndarray:
    ws = np.asarray(ws, dtype=np.float32)
    wt = np.asarray(wt, dtype=np.float32)
    if ws.ndim == 0:
        ws = np.full(len(data.text), float(ws), dtype=np.float32)
    if wt.ndim == 0:
        wt = np.full(len(data.text), float(wt), dtype=np.float32)
    text_weight = 1.0 - ws - wt
    if np.any(text_weight < -1e-8):
        raise ValueError("Fusion weights must be nonnegative")
    prototypes = normalize(text_weight[:, None] * data.text + ws[:, None] * data.source + wt[:, None] * target)
    return classify(data.target_features, prototypes)


def class_reliability(core: dict) -> np.ndarray:
    class_count = core["corrected"].shape[1]
    entropy = -(core["corrected"] * np.log(np.clip(core["corrected"], 1e-12, 1.0))).sum(1)
    entropy /= np.log(class_count)
    result = np.zeros(class_count, dtype=np.float32)
    for class_index in range(class_count):
        mask = core["accepted"] & (core["pseudo"] == class_index)
        if not mask.any():
            continue
        ess = float(core["effective_sizes"][class_index])
        certainty = float(1.0 - entropy[mask].mean())
        result[class_index] = (ess / (ess + 5.0)) * np.clip(certainty, 0.0, 1.0)
    return result


def entropy_objective(probabilities: np.ndarray) -> float:
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(1).mean()
    mean_prediction = np.clip(probabilities.mean(0), 1e-12, 1.0)
    balance_penalty = float((mean_prediction * np.log(mean_prediction * probabilities.shape[1])).sum())
    return float(entropy + 0.05 * balance_penalty)


def run_fusion_experiments(all_tasks: list[TaskData]) -> None:
    rows = []
    for data in all_tasks:
        core = current_target(data)
        reliability = class_reliability(core)
        candidates: list[tuple[str, np.ndarray, np.ndarray, dict]] = [
            ("fixed_asymmetric", np.array(0.1), np.array(0.025), {}),
            ("uniform", np.array(1.0 / 3.0), np.array(1.0 / 3.0), {}),
            (
                "uncertainty_scaled_target",
                np.array(0.1),
                0.025 * reliability,
                {"mean_reliability": float(reliability.mean())},
            ),
            (
                "class_adaptive",
                0.05 + 0.25 * (1.0 - reliability),
                0.10 * reliability,
                {"mean_reliability": float(reliability.mean())},
            ),
        ]
        grid = []
        for source_weight in (0.0, 0.05, 0.1, 0.2, 1.0 / 3.0):
            for target_weight in (0.0, 0.025, 0.05, 0.1, 1.0 / 3.0):
                if source_weight + target_weight > 1.0:
                    continue
                probabilities = final_probabilities(data, core["target"], source_weight, target_weight)
                grid.append((entropy_objective(probabilities), source_weight, target_weight, probabilities))
        objective, source_weight, target_weight, probabilities = min(grid, key=lambda item: item[0])
        candidates.append(
            (
                "label_free_entropy_selected",
                np.array(source_weight),
                np.array(target_weight),
                {"selection_objective": objective},
            )
        )
        for method, ws, wt, diagnostics in candidates:
            probabilities = final_probabilities(data, core["target"], ws, wt)
            row = {
                "benchmark": data.benchmark,
                "task": data.task,
                "method": method,
                "source_weight_mean": float(np.asarray(ws).mean()),
                "target_weight_mean": float(np.asarray(wt).mean()),
                "target_labels_used_for_weight_selection": False,
                **evaluate(probabilities, data.target_labels),
                **diagnostics,
            }
            rows.append(row)
    write_csv(OUTPUT / "fusion_strategy_comparison.csv", rows)


def selected_prototype(
    data: TaskData,
    corrected: np.ndarray,
    selected: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pseudo = corrected.argmax(1)
    confidence = corrected.max(1)
    if mode == "reliability_weighted":
        weights = confidence * uncertainty_weights(corrected) * selected
    else:
        weights = confidence * selected
    return prototypes_from_labels(data.target_features, pseudo, weights, data.base)


def run_pseudo_label_experiments(all_tasks: list[TaskData]) -> None:
    threshold_rows = []
    strategy_rows = []
    for data in all_tasks:
        corrected, _ = corrected_probabilities(data.base_probabilities)
        pseudo = corrected.argmax(1)
        confidence = corrected.max(1)
        entropy = -(corrected * np.log(np.clip(corrected, 1e-12, 1.0))).sum(1)
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            selected = confidence >= threshold
            target, counts, _ = selected_prototype(data, corrected, selected, "reliability_weighted")
            probabilities = final_probabilities(data, target, 0.1, 0.025)
            threshold_rows.append({
                "benchmark": data.benchmark,
                "task": data.task,
                "threshold": threshold,
                "selected_count": int(selected.sum()),
                "selection_coverage": float(selected.mean()),
                "selected_pseudo_label_accuracy": float((pseudo[selected] == data.target_labels[selected]).mean() * 100.0)
                if selected.any() else np.nan,
                "classes_with_selected_samples": int((counts > 0).sum()),
                "target_labels_used_only_for_posthoc_diagnostics": True,
                **evaluate(probabilities, data.target_labels),
            })

        baseline_selected = confidence >= 0.7
        selected_count = int(baseline_selected.sum())
        entropy_selected = np.zeros(len(entropy), dtype=bool)
        entropy_selected[np.argsort(entropy)[:selected_count]] = True
        class_balanced = np.zeros(len(entropy), dtype=bool)
        per_class = max(1, int(np.ceil(selected_count / corrected.shape[1])))
        for class_index in range(corrected.shape[1]):
            candidates = np.flatnonzero(pseudo == class_index)
            if len(candidates):
                chosen = candidates[np.argsort(entropy[candidates])[:per_class]]
                class_balanced[chosen] = True
        if class_balanced.sum() > selected_count:
            chosen = np.flatnonzero(class_balanced)
            keep = chosen[np.argsort(entropy[chosen])[:selected_count]]
            class_balanced[:] = False
            class_balanced[keep] = True
        strategies = {
            "confidence_threshold": (baseline_selected, "confidence_only"),
            "entropy_matched_coverage": (entropy_selected, "confidence_only"),
            "class_balanced_entropy": (class_balanced, "confidence_only"),
            "reliability_weighted": (baseline_selected, "reliability_weighted"),
        }
        for strategy, (selected, weighting_mode) in strategies.items():
            target, counts, _ = selected_prototype(data, corrected, selected, weighting_mode)
            probabilities = final_probabilities(data, target, 0.1, 0.025)
            strategy_rows.append({
                "benchmark": data.benchmark,
                "task": data.task,
                "strategy": strategy,
                "selected_count": int(selected.sum()),
                "selection_coverage": float(selected.mean()),
                "selected_pseudo_label_accuracy": float((pseudo[selected] == data.target_labels[selected]).mean() * 100.0)
                if selected.any() else np.nan,
                "classes_with_selected_samples": int((counts > 0).sum()),
                "target_labels_used_only_for_posthoc_diagnostics": True,
                **evaluate(probabilities, data.target_labels),
            })
    write_csv(OUTPUT / "pseudo_label_threshold_diagnostics.csv", threshold_rows)
    write_csv(OUTPUT / "pseudo_label_selection_strategy.csv", strategy_rows)


def oracle_prototypes(data: TaskData) -> np.ndarray:
    weights = np.ones(len(data.target_labels), dtype=np.float32)
    prototypes, _, _ = prototypes_from_labels(data.target_features, data.target_labels, weights, data.base)
    return prototypes


def run_noise_experiments(all_tasks: list[TaskData]) -> None:
    rows = []
    for task_index, data in enumerate(all_tasks):
        core = current_target(data)
        accepted_indices = np.flatnonzero(core["accepted"])
        oracle = oracle_prototypes(data)
        for noise_rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            for repetition in range(5):
                rng = np.random.default_rng(SEED + 1000 * task_index + 100 * repetition + int(noise_rate * 100))
                noisy = core["pseudo"].copy()
                corrupt_count = int(round(noise_rate * len(accepted_indices)))
                corrupt = rng.choice(accepted_indices, size=corrupt_count, replace=False) if corrupt_count else np.array([], dtype=int)
                if len(corrupt):
                    offsets = rng.integers(1, len(data.text), size=len(corrupt))
                    noisy[corrupt] = (noisy[corrupt] + offsets) % len(data.text)
                for method, fallback, source_weight in (
                    ("source_anchored", data.base, 0.1),
                    ("without_source_fusion", data.text, 0.0),
                ):
                    target, counts, _ = prototypes_from_labels(
                        data.target_features, noisy, core["weights"], fallback
                    )
                    probabilities = final_probabilities(data, target, source_weight, 0.025)
                    prototype_cosine = np.sum(normalize(target) * normalize(oracle), axis=1)
                    rows.append({
                        "benchmark": data.benchmark,
                        "task": data.task,
                        "method": method,
                        "noise_rate": noise_rate,
                        "repetition": repetition,
                        "corrupted_selected_labels": corrupt_count,
                        "classes_with_target_support": int((counts > 0).sum()),
                        "mean_cosine_to_oracle_target_prototype": float(prototype_cosine.mean()),
                        "target_labels_used_only_for_noise_control_and_posthoc_diagnostics": True,
                        **evaluate(probabilities, data.target_labels),
                    })
    write_csv(OUTPUT / "pseudo_label_noise_robustness.csv", rows)


def run_prior_experiments(all_tasks: list[TaskData]) -> None:
    perturbation_rows = []
    imbalance_rows = []
    for task_index, data in enumerate(all_tasks):
        estimated = data.base_probabilities.mean(0).astype(np.float64)
        estimated /= estimated.sum()
        oracle = np.bincount(data.target_labels, minlength=len(data.text)).astype(np.float64)
        oracle /= oracle.sum()
        permutation = np.random.default_rng(SEED + task_index).permutation(len(estimated))
        permuted = estimated[permutation]
        for perturbation in (0.0, 0.25, 0.5, 0.75, 1.0):
            used_prior = (1.0 - perturbation) * estimated + perturbation * permuted
            core = current_target(data, prior=used_prior)
            probabilities = final_probabilities(data, core["target"], 0.1, 0.025)
            perturbation_rows.append({
                "benchmark": data.benchmark,
                "task": data.task,
                "perturbation": perturbation,
                "prior_l1_error_to_oracle": float(np.abs(used_prior - oracle).sum()),
                "selected_count": int(core["accepted"].sum()),
                "target_labels_used_only_to_measure_prior_error": True,
                **evaluate(probabilities, data.target_labels),
            })

        class_count = len(data.text)
        class_indices = [np.flatnonzero(data.target_labels == c) for c in range(class_count)]
        max_count = min(len(indices) for indices in class_indices if len(indices))
        for imbalance_factor in (1, 10, 50):
            for repetition in range(5):
                rng = np.random.default_rng(SEED + 1000 * task_index + 10 * repetition + imbalance_factor)
                class_order = rng.permutation(class_count)
                selected_parts = []
                for rank, class_index in enumerate(class_order):
                    available = class_indices[class_index]
                    desired = int(round(max_count * imbalance_factor ** (-rank / max(class_count - 1, 1))))
                    desired = max(1, min(desired, len(available)))
                    selected_parts.append(rng.choice(available, size=desired, replace=False))
                selected = np.sort(np.concatenate(selected_parts))
                subset = TaskData(
                    benchmark=data.benchmark,
                    task=data.task,
                    source_features=data.source_features,
                    source_labels=data.source_labels,
                    target_features=data.target_features[selected],
                    target_labels=data.target_labels[selected],
                    target_ids=data.target_ids[selected],
                    text=data.text,
                    source=data.source,
                    base=data.base,
                    base_probabilities=data.base_probabilities[selected],
                )
                for method, strength in (("no_prior_correction", 0.0), ("estimated_prior_correction", 0.1)):
                    core = current_target(subset, prior_strength=strength)
                    probabilities = final_probabilities(subset, core["target"], 0.1, 0.025)
                    imbalance_rows.append({
                        "benchmark": data.benchmark,
                        "task": data.task,
                        "imbalance_factor": imbalance_factor,
                        "repetition": repetition,
                        "method": method,
                        "subset_samples": len(selected),
                        "minimum_class_samples": min(len(part) for part in selected_parts),
                        "maximum_class_samples": max(len(part) for part in selected_parts),
                        "target_labels_used_only_to_construct_controlled_imbalance": True,
                        **evaluate(probabilities, subset.target_labels),
                    })
    write_csv(OUTPUT / "prior_estimate_perturbation.csv", perturbation_rows)
    write_csv(OUTPUT / "controlled_class_imbalance.csv", imbalance_rows)


def load_prediction_correctness(
    benchmark: str,
    task: str,
    method: str,
    prediction_dir: Path | None = None,
) -> np.ndarray:
    path = (prediction_dir or BENCHMARKS[benchmark]["prediction_dir"]) / f"{task}_{method}.npz"
    data = load_task(benchmark, task)
    with np.load(path, allow_pickle=False) as prediction:
        if not np.array_equal(prediction["sample_ids"].astype(np.int64), data.target_ids):
            raise ValueError(f"Prediction IDs do not align for {path}")
        predicted = prediction["predictions"].astype(np.int64)
    return predicted == data.target_labels


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def run_statistical_experiments() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    summaries = {}
    comparators = ("prompt_ensemble", "t3a", "tip_adapter_source")
    for benchmark in BENCHMARKS:
        summaries[benchmark] = {}
        for comparator_method in comparators:
            benchmark_rows = []
            comparator_dir = (
                BENCHMARKS[benchmark]["prediction_dir"]
                if comparator_method == "prompt_ensemble"
                else RESULTS / benchmark / "predictions" / "additional_baselines_v1"
            )
            for task in tasks_for(benchmark):
                    reference = load_prediction_correctness(benchmark, task, "satpa")
                    comparator = load_prediction_correctness(
                        benchmark, task, comparator_method, prediction_dir=comparator_dir
                    )
                    delta = reference.astype(np.float32) - comparator.astype(np.float32)
                    bootstrap = np.empty(5000, dtype=np.float32)
                    for start in range(0, 5000, 250):
                        stop = min(start + 250, 5000)
                        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
                        bootstrap[start:stop] = delta[indices].mean(1) * 100.0
                    favorable = int((reference & ~comparator).sum())
                    unfavorable = int((~reference & comparator).sum())
                    discordant = favorable + unfavorable
                    mcnemar_p = float(binomtest(favorable, discordant, 0.5).pvalue) if discordant else 1.0
                    benchmark_rows.append({
                        "benchmark": benchmark,
                        "task": task,
                        "comparator": comparator_method,
                        "samples": len(delta),
                        "comparator_accuracy": float(comparator.mean() * 100.0),
                        "satpa_accuracy": float(reference.mean() * 100.0),
                        "gain_pp": float(delta.mean() * 100.0),
                        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                        "bootstrap_probability_gain_le_zero": float((bootstrap <= 0).mean()),
                        "mcnemar_exact_p": mcnemar_p,
                        "discordant_satpa_only_correct": favorable,
                        "discordant_comparator_only_correct": unfavorable,
                    })
            adjusted = holm_adjust([row["mcnemar_exact_p"] for row in benchmark_rows])
            for row, adjusted_p in zip(benchmark_rows, adjusted):
                row["mcnemar_holm_p"] = adjusted_p
                row["significant_after_holm_0p05"] = adjusted_p < 0.05
            rows.extend(benchmark_rows)
            gains = np.array([row["gain_pp"] for row in benchmark_rows], dtype=np.float64)
            summaries[benchmark][comparator_method] = {
                "tasks": len(gains),
                "mean_gain_pp": float(gains.mean()),
                "positive_tasks": int((gains > 0).sum()),
                "zero_tasks": int((gains == 0).sum()),
                "wilcoxon_signed_rank_two_sided_p": float(wilcoxon(gains, alternative="two-sided").pvalue),
                "sign_test_two_sided_p": float(
                    binomtest(
                        int((gains > 0).sum()),
                        int((gains != 0).sum()),
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                ),
                "warning": "Task-level tests are secondary because transfer tasks sharing a target domain are correlated.",
            }
    write_csv(OUTPUT / "per_task_paired_statistics.csv", rows)
    write_json(OUTPUT / "cross_task_significance.json", summaries)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_outputs() -> None:
    import pandas as pd

    summaries = {}
    fusion = pd.read_csv(OUTPUT / "fusion_strategy_comparison.csv")
    summaries["fusion"] = fusion.groupby(["benchmark", "method"])["accuracy"].mean().round(4).unstack().to_dict()
    thresholds = pd.read_csv(OUTPUT / "pseudo_label_threshold_diagnostics.csv")
    summaries["thresholds"] = (
        thresholds.groupby(["benchmark", "threshold"])[
            ["selection_coverage", "selected_pseudo_label_accuracy", "accuracy"]
        ].mean().round(4).reset_index().to_dict(orient="records")
    )
    strategies = pd.read_csv(OUTPUT / "pseudo_label_selection_strategy.csv")
    summaries["selection_strategies"] = (
        strategies.groupby(["benchmark", "strategy"])[
            ["selection_coverage", "selected_pseudo_label_accuracy", "accuracy"]
        ].mean().round(4).reset_index().to_dict(orient="records")
    )
    noise = pd.read_csv(OUTPUT / "pseudo_label_noise_robustness.csv")
    summaries["noise"] = (
        noise.groupby(["benchmark", "method", "noise_rate"])[
            ["accuracy", "mean_cosine_to_oracle_target_prototype"]
        ].mean().round(4).reset_index().to_dict(orient="records")
    )
    prior = pd.read_csv(OUTPUT / "prior_estimate_perturbation.csv")
    summaries["prior_perturbation"] = (
        prior.groupby(["benchmark", "perturbation"])[["prior_l1_error_to_oracle", "accuracy"]]
        .mean().round(4).reset_index().to_dict(orient="records")
    )
    imbalance = pd.read_csv(OUTPUT / "controlled_class_imbalance.csv")
    summaries["class_imbalance"] = (
        imbalance.groupby(["benchmark", "imbalance_factor", "method"])[["accuracy", "balanced_accuracy"]]
        .mean().round(4).reset_index().to_dict(orient="records")
    )
    write_json(OUTPUT / "reviewer_response_summary.json", summaries)


def build_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.glob("*")):
        if not path.is_file() or path.name == "sha256_manifest.csv":
            continue
        rows.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    write_csv(OUTPUT / "sha256_manifest.csv", rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_tasks = [load_task(benchmark, task) for benchmark in BENCHMARKS for task in tasks_for(benchmark)]
    run_fusion_experiments(all_tasks)
    run_pseudo_label_experiments(all_tasks)
    run_noise_experiments(all_tasks)
    run_prior_experiments(all_tasks)
    run_statistical_experiments()
    summarize_outputs()
    build_manifest()
    print(f"Completed reviewer-response experiments: {OUTPUT}")


if __name__ == "__main__":
    main()

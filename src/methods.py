from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def classify(features: np.ndarray, prototypes: np.ndarray, logit_scale: float = 100.0) -> np.ndarray:
    return softmax(logit_scale * normalize(features) @ normalize(prototypes).T)


def source_prototypes(features: np.ndarray, labels: np.ndarray, text: np.ndarray) -> np.ndarray:
    prototypes = []
    for class_index in range(len(text)):
        selected = features[labels == class_index]
        prototypes.append(text[class_index] if not len(selected) else normalize(selected.mean(0, keepdims=True))[0])
    return normalize(np.stack(prototypes))


def t3a_prototypes(features: np.ndarray, text: np.ndarray, filter_k: int = 5) -> np.ndarray:
    probabilities = classify(features, text)
    pseudo_labels = probabilities.argmax(1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(1)
    prototypes = []
    for class_index in range(len(text)):
        candidates = np.flatnonzero(pseudo_labels == class_index)
        if len(candidates):
            selected = candidates[np.argsort(entropy[candidates])[:filter_k]]
            support = np.concatenate([text[class_index][None, :], features[selected]], axis=0)
            prototypes.append(normalize(support.mean(0, keepdims=True))[0])
        else:
            prototypes.append(text[class_index])
    return normalize(np.stack(prototypes))


def tip_adapter_probabilities(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    text: np.ndarray,
    alpha: float = 1.0,
    beta: float = 5.0,
) -> np.ndarray:
    source_features = normalize(source_features)
    target_features = normalize(target_features)
    clip_logits = 100.0 * target_features @ normalize(text).T
    affinity = target_features @ source_features.T
    cache_values = np.eye(len(text), dtype=np.float32)[source_labels]
    cache_logits = np.exp(beta * (affinity - 1.0)) @ cache_values
    return softmax(clip_logits + alpha * cache_logits)


def _uncertainty_weights(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    certainty = 1.0 - entropy / np.log(probabilities.shape[1])
    return np.clip(confidence * margin * certainty, 0.0, 1.0).astype(np.float32)


def agreement_target_prototypes(
    features: np.ndarray,
    text_probabilities: np.ndarray,
    source_probabilities: np.ndarray,
    fallback: np.ndarray,
    margin_threshold: float = 0.05,
    reliability_tau: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build hard target prototypes only from text/source-agreed samples.

    The returned class reliability is label-free. It combines Kish effective
    sample size with the mean normalized entropy of accepted samples.
    """
    class_count = text_probabilities.shape[1]
    text_labels = text_probabilities.argmax(1)
    source_labels = source_probabilities.argmax(1)
    consensus = 0.5 * (text_probabilities + source_probabilities)
    ordered = np.sort(consensus, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    accepted = (text_labels == source_labels) & (margins > margin_threshold)
    weights = np.maximum(margins - margin_threshold, 0.0).astype(np.float32) * accepted
    entropy = -(consensus * np.log(np.clip(consensus, 1e-12, 1.0))).sum(1)
    normalized_entropy = entropy / np.log(class_count)

    prototypes = []
    reliability = np.zeros(class_count, dtype=np.float32)
    effective_counts = np.zeros(class_count, dtype=np.float32)
    accepted_counts = np.zeros(class_count, dtype=np.int32)
    for class_index in range(class_count):
        selected = accepted & (text_labels == class_index)
        class_weights = weights[selected]
        accepted_counts[class_index] = int(selected.sum())
        if not len(class_weights) or class_weights.sum() <= 1e-8:
            prototypes.append(fallback[class_index])
            continue
        effective_count = float(class_weights.sum() ** 2 / np.clip((class_weights ** 2).sum(), 1e-12, None))
        effective_counts[class_index] = effective_count
        certainty = float(1.0 - normalized_entropy[selected].mean())
        reliability[class_index] = (
            effective_count / (effective_count + reliability_tau)
        ) * np.clip(certainty, 0.0, 1.0)
        prototype = (class_weights[:, None] * features[selected]).sum(0) / class_weights.sum()
        prototypes.append(normalize(prototype[None, :])[0])

    diagnostics = {
        "accepted_samples": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "classes_with_target_support": int((accepted_counts > 0).sum()),
        "mean_effective_count": float(effective_counts.mean()),
        "mean_class_reliability": float(reliability.mean()),
        "min_class_reliability": float(reliability.min()),
        "max_class_reliability": float(reliability.max()),
    }
    return normalize(np.stack(prototypes)), reliability, diagnostics


def adaptive_prototype_fusion(
    text: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    reliability: np.ndarray,
    source_weight_min: float = 0.05,
    source_weight_max: float = 0.30,
    target_weight_max: float = 0.10,
) -> tuple[np.ndarray, dict]:
    reliability = np.clip(np.asarray(reliability, dtype=np.float32), 0.0, 1.0)
    source_weights = source_weight_min + (source_weight_max - source_weight_min) * (1.0 - reliability)
    target_weights = target_weight_max * reliability
    text_weights = 1.0 - source_weights - target_weights
    if np.any(text_weights < 0):
        raise ValueError("Adaptive fusion weights leave a negative text weight")
    prototypes = normalize(
        text_weights[:, None] * text
        + source_weights[:, None] * source
        + target_weights[:, None] * target
    )
    diagnostics = {
        "mean_source_weight": float(source_weights.mean()),
        "min_source_weight": float(source_weights.min()),
        "max_source_weight": float(source_weights.max()),
        "mean_target_weight": float(target_weights.mean()),
        "min_target_weight": float(target_weights.min()),
        "max_target_weight": float(target_weights.max()),
    }
    return prototypes, diagnostics


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.clip(np.exp(values - maximum).sum(axis=axis, keepdims=True), 1e-300, None))
    return np.squeeze(result, axis=axis)


def sinkhorn_transport(
    cost: np.ndarray,
    sample_marginal: np.ndarray,
    class_marginal: np.ndarray,
    epsilon: float,
    max_iter: int = 200,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, dict]:
    """Solve balanced entropic OT in the log domain."""
    if epsilon <= 0:
        raise ValueError("OT epsilon must be positive")
    cost = np.asarray(cost, dtype=np.float64)
    mu = np.clip(np.asarray(sample_marginal, dtype=np.float64), 1e-300, None)
    nu = np.clip(np.asarray(class_marginal, dtype=np.float64), 1e-300, None)
    mu /= mu.sum()
    nu /= nu.sum()
    log_kernel = -cost / epsilon
    log_mu = np.log(mu)
    log_nu = np.log(nu)
    log_u = np.zeros_like(mu)
    log_v = np.zeros_like(nu)
    row_error = float("inf")
    column_error = float("inf")
    iterations = 0
    for iteration in range(1, max_iter + 1):
        log_u = log_mu - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_nu - _logsumexp(log_kernel + log_u[:, None], axis=0)
        iterations = iteration
        if iteration == 1 or iteration % 5 == 0 or iteration == max_iter:
            plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
            row_error = float(np.max(np.abs(plan.sum(1) - mu)))
            column_error = float(np.max(np.abs(plan.sum(0) - nu)))
            if max(row_error, column_error) <= tolerance:
                break
    plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
    if not np.isfinite(plan).all() or plan.sum() <= 0:
        raise FloatingPointError("Sinkhorn produced an invalid transport plan")
    diagnostics = {
        "ot_iterations": iterations,
        "ot_row_marginal_error": row_error,
        "ot_column_marginal_error": column_error,
    }
    return plan, diagnostics


def transport_target_prototypes(
    source_labels: np.ndarray,
    target_features: np.ndarray,
    base_prototypes: np.ndarray,
    epsilon: float,
    max_iter: int = 200,
    tolerance: float = 1e-7,
    prior_mix: float = 0.5,
) -> tuple[np.ndarray, dict]:
    class_count = len(base_prototypes)
    if not 0.0 <= prior_mix <= 1.0:
        raise ValueError("OT prior mix must be between zero and one")
    similarities = normalize(target_features) @ normalize(base_prototypes).T
    cost = 1.0 - similarities.astype(np.float64)
    base_probabilities = softmax(100.0 * similarities)
    source_counts = np.bincount(source_labels, minlength=class_count).astype(np.float64)
    source_prior = source_counts / source_counts.sum()
    predicted_prior = base_probabilities.astype(np.float64).mean(0)
    class_marginal = prior_mix * source_prior + (1.0 - prior_mix) * predicted_prior
    class_marginal /= class_marginal.sum()
    sample_marginal = np.full(len(target_features), 1.0 / len(target_features), dtype=np.float64)
    plan, diagnostics = sinkhorn_transport(
        cost, sample_marginal, class_marginal, epsilon, max_iter=max_iter, tolerance=tolerance
    )
    assignments = plan / np.clip(plan.sum(1, keepdims=True), 1e-300, None)
    prototypes = []
    effective_sizes = []
    top1_counts = np.bincount(assignments.argmax(1), minlength=class_count)
    for class_index in range(class_count):
        weights = assignments[:, class_index]
        effective_size = float(weights.sum() ** 2 / np.clip((weights ** 2).sum(), 1e-300, None))
        effective_sizes.append(effective_size)
        if weights.sum() <= 1e-12:
            prototypes.append(base_prototypes[class_index])
        else:
            prototype = (weights[:, None] * target_features).sum(0) / weights.sum()
            prototypes.append(normalize(prototype[None, :])[0])
    effective_sizes = np.asarray(effective_sizes)
    diagnostics.update({
        "ot_epsilon": float(epsilon),
        "ot_prior_mix": float(prior_mix),
        "ot_min_class_prior": float(class_marginal.min()),
        "ot_max_class_prior": float(class_marginal.max()),
        "ot_min_effective_size": float(effective_sizes.min()),
        "ot_mean_effective_size": float(effective_sizes.mean()),
        "ot_max_effective_size": float(effective_sizes.max()),
        "ot_classes_ess_below_3": int((effective_sizes < 3.0).sum()),
        "ot_min_top1_count": int(top1_counts.min()),
        "ot_classes_without_top1": int((top1_counts == 0).sum()),
    })
    return normalize(np.stack(prototypes)), diagnostics


def target_prototypes(
    features: np.ndarray,
    probabilities: np.ndarray,
    fallback: np.ndarray,
    threshold: float,
    top_k: int,
    class_prior_strength: float,
    use_uncertainty: bool = True,
) -> tuple[np.ndarray, dict]:
    class_count = probabilities.shape[1]
    top_k = max(1, min(int(top_k), class_count))
    predicted_frequency = np.clip(probabilities.mean(0), 1e-8, None)
    corrected = probabilities / np.power(predicted_frequency[None, :], class_prior_strength)
    corrected /= corrected.sum(axis=1, keepdims=True)
    uncertainty = _uncertainty_weights(corrected) if use_uncertainty else np.ones(len(features), dtype=np.float32)
    accepted = corrected.max(axis=1) >= threshold
    top_indices = np.argpartition(corrected, -top_k, axis=1)[:, -top_k:]
    assignments = np.zeros_like(corrected, dtype=np.float32)
    rows = np.arange(len(features))[:, None]
    assignments[rows, top_indices] = corrected[rows, top_indices]
    assignments *= uncertainty[:, None] * accepted[:, None]

    prototypes = []
    effective_counts = assignments.sum(0)
    for class_index in range(class_count):
        weights = assignments[:, class_index]
        if weights.sum() <= 1e-8:
            prototypes.append(fallback[class_index])
        else:
            prototype = (weights[:, None] * features).sum(0) / weights.sum()
            prototypes.append(normalize(prototype[None, :])[0])
    diagnostics = {
        "accepted_samples": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "mean_uncertainty_weight": float(uncertainty.mean()),
        "classes_with_target_support": int((effective_counts > 1e-8).sum()),
    }
    return normalize(np.stack(prototypes)), diagnostics


@dataclass(frozen=True)
class MethodOutput:
    probabilities: np.ndarray
    prototypes: np.ndarray
    diagnostics: dict


def run_method(
    method: str,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    text_ensemble: np.ndarray,
    text_per_prompt: np.ndarray,
    alpha_source: float = 0.1,
    alpha_target: float = 0.025,
    confidence_threshold: float = 0.7,
    top_k: int = 1,
    class_prior_strength: float = 0.1,
    t3a_filter_k: int = 5,
    tip_alpha: float = 1.0,
    tip_beta: float = 5.0,
    agreement_margin: float = 0.05,
    reliability_tau: float = 5.0,
    adaptive_source_min: float = 0.05,
    adaptive_source_max: float = 0.30,
    adaptive_target_max: float = 0.10,
    target_update_steps: int = 1,
    ot_epsilon: float = 0.05,
    ot_max_iter: int = 200,
    ot_tolerance: float = 1e-7,
    ot_prior_mix: float = 0.5,
) -> MethodOutput:
    text_ensemble = normalize(text_ensemble)
    source = source_prototypes(source_features, source_labels, text_ensemble)
    diagnostics = {}

    if method == "clip_zero_shot":
        prototypes = normalize(text_per_prompt[:, 0, :])
    elif method == "prompt_ensemble":
        prototypes = text_ensemble
    elif method == "source_prototype":
        prototypes = source
    elif method == "t3a":
        prototypes = t3a_prototypes(target_features, text_ensemble, t3a_filter_k)
    elif method == "tip_adapter_source":
        probabilities = tip_adapter_probabilities(
            source_features, source_labels, target_features, text_ensemble, tip_alpha, tip_beta
        )
        return MethodOutput(probabilities, text_ensemble, diagnostics)
    elif method == "source_anchored_text":
        prototypes = normalize((1.0 - alpha_source) * text_ensemble + alpha_source * source)
    elif method in {"satpa", "no_source_anchor", "satpa_no_uncertainty"}:
        source_weight = 0.0 if method == "no_source_anchor" else alpha_source
        base = normalize((1.0 - source_weight) * text_ensemble + source_weight * source)
        base_probabilities = classify(target_features, base)
        target, diagnostics = target_prototypes(
            target_features,
            base_probabilities,
            base,
            confidence_threshold,
            top_k,
            class_prior_strength,
            use_uncertainty=method != "satpa_no_uncertainty",
        )
        if alpha_target < 0 or source_weight < 0 or alpha_target + source_weight > 1:
            raise ValueError("Prototype fusion weights must be nonnegative and sum to at most one")
        prototypes = normalize(
            (1.0 - source_weight - alpha_target) * text_ensemble
            + source_weight * source
            + alpha_target * target
        )
    elif method in {"satpa_agreement", "adaptive_satpa", "iterative_satpa"}:
        text_probabilities = classify(target_features, text_ensemble)
        source_probabilities = classify(target_features, source)
        fallback = normalize((1.0 - alpha_source) * text_ensemble + alpha_source * source)
        target, reliability, diagnostics = agreement_target_prototypes(
            target_features,
            text_probabilities,
            source_probabilities,
            fallback,
            margin_threshold=agreement_margin,
            reliability_tau=reliability_tau,
        )
        if method in {"satpa_agreement", "iterative_satpa"}:
            prototypes = normalize(
                (1.0 - alpha_source - alpha_target) * text_ensemble
                + alpha_source * source
                + alpha_target * target
            )
        else:
            prototypes, fusion_diagnostics = adaptive_prototype_fusion(
                text_ensemble,
                source,
                target,
                reliability,
                source_weight_min=adaptive_source_min,
                source_weight_max=adaptive_source_max,
                target_weight_max=adaptive_target_max,
            )
            diagnostics.update(fusion_diagnostics)
        if method == "iterative_satpa":
            if target_update_steps < 1:
                raise ValueError("target_update_steps must be at least one")
            current_probabilities = classify(target_features, prototypes)
            current_labels = current_probabilities.argmax(1)
            current_confidence = float(current_probabilities.max(1).mean())
            rounds_completed = 1
            last_change_rate = 1.0
            stopped_reason = "max_steps"
            for step in range(2, target_update_steps + 1):
                next_target, _, next_diagnostics = agreement_target_prototypes(
                    target_features,
                    current_probabilities,
                    source_probabilities,
                    fallback,
                    margin_threshold=agreement_margin,
                    reliability_tau=reliability_tau,
                )
                next_prototypes = normalize(
                    (1.0 - alpha_source - alpha_target) * text_ensemble
                    + alpha_source * source
                    + alpha_target * next_target
                )
                next_probabilities = classify(target_features, next_prototypes)
                next_labels = next_probabilities.argmax(1)
                next_confidence = float(next_probabilities.max(1).mean())
                last_change_rate = float((next_labels != current_labels).mean())
                if next_confidence < current_confidence:
                    stopped_reason = "confidence_rollback"
                    break
                prototypes = next_prototypes
                current_probabilities = next_probabilities
                current_labels = next_labels
                current_confidence = next_confidence
                diagnostics.update(next_diagnostics)
                rounds_completed = step
                if last_change_rate < 0.01:
                    stopped_reason = "label_stability"
                    break
            diagnostics.update({
                "rounds_completed": rounds_completed,
                "last_label_change_rate": last_change_rate,
                "final_mean_confidence": current_confidence,
                "stopped_reason": stopped_reason,
            })
    elif method == "spt_sa":
        if alpha_source < 0 or alpha_target < 0 or alpha_source + alpha_target > 1:
            raise ValueError("Prototype fusion weights must be nonnegative and sum to at most one")
        base = normalize((1.0 - alpha_source) * text_ensemble + alpha_source * source)
        target, diagnostics = transport_target_prototypes(
            source_labels,
            target_features,
            base,
            epsilon=ot_epsilon,
            max_iter=ot_max_iter,
            tolerance=ot_tolerance,
            prior_mix=ot_prior_mix,
        )
        prototypes = normalize(
            (1.0 - alpha_source - alpha_target) * text_ensemble
            + alpha_source * source
            + alpha_target * target
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return MethodOutput(classify(target_features, prototypes), prototypes, diagnostics)

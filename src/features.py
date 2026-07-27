from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrozenFeatures:
    image: np.ndarray
    text: np.ndarray
    text_per_prompt: np.ndarray
    classes: np.ndarray
    sample_ids: np.ndarray


def _normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def load_frozen_features(path: str) -> FrozenFeatures:
    with np.load(path, allow_pickle=False) as data:
        forbidden = {"labels", "targets", "paths"}.intersection(data.files)
        if forbidden:
            raise ValueError(f"Feature file leaks instance information: {sorted(forbidden)}")
        required = {"image_features", "text_features", "classes", "sample_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Feature file is missing {sorted(missing)}")
        text_per_prompt = (
            data["text_features_per_prompt"]
            if "text_features_per_prompt" in data.files
            else data["text_features"][:, None, :]
        )
        result = FrozenFeatures(
            image=_normalize(data["image_features"]),
            text=_normalize(data["text_features"]),
            text_per_prompt=_normalize(text_per_prompt.reshape(-1, text_per_prompt.shape[-1])).reshape(
                text_per_prompt.shape
            ),
            classes=data["classes"].astype(str),
            sample_ids=data["sample_ids"].astype(np.int64),
        )
    if result.image.shape[0] != result.sample_ids.shape[0]:
        raise ValueError("Feature/sample ID length mismatch")
    return result


def load_label_sidecar(path: str, expected_ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"sample_ids", "labels"}:
            raise ValueError(f"Unexpected label-sidecar keys: {data.files}")
        ids = data["sample_ids"].astype(np.int64)
        labels = data["labels"].astype(np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("Label sidecar does not align with the frozen feature file")
    return labels


def sidecar_for(feature_path: str) -> str:
    return str(Path(feature_path).with_suffix(".source_labels.npz"))


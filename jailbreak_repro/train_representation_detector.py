from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jailbreak_repro.hiddendetect import binary_auroc
from jailbreak_repro.io_utils import write_json, write_jsonl
from jailbreak_repro.representation_detectors import (
    TRAINABLE_REPRESENTATION_CHOICES,
    RepresentationDetector,
    normalize_representation_method,
    save_representation_artifact,
)


FIELD_ALIASES = {
    "condition": "conditions",
    "intent_family": "intent_families",
    "source": "sources",
    "carrier_type": "carrier_types",
    "image_role": "image_roles",
}


@dataclass
class ActivationArchive:
    path: Path
    activations: np.ndarray
    layers: np.ndarray
    labels: np.ndarray
    fields: dict[str, np.ndarray]
    metadata: dict[str, Any]
    logical_fingerprint: str


def _string_array(values: Any, count: int, default: str = "") -> np.ndarray:
    if values is None:
        return np.full(count, default, dtype=str)
    result = np.asarray(values).astype(str)
    if result.shape != (count,):
        raise ValueError(f"Expected a length-{count} metadata vector, got {result.shape}")
    return result


def _logical_fingerprint(
    ids: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    layers: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha1()
    for values in (ids, labels.astype(str), splits, layers.astype(str)):
        for value in values:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    digest.update(
        json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def activation_archive_logical_fingerprint(path: Path) -> str:
    source = path.expanduser().resolve()
    with np.load(source, allow_pickle=True) as archive:
        if "labels" not in archive or "layers" not in archive:
            raise KeyError("Activation archive must contain labels and layers")
        labels = np.asarray(archive["labels"], dtype=np.int32)
        layers = np.asarray(archive["layers"], dtype=np.int32)
        count = len(labels)
        ids = _string_array(archive["ids"] if "ids" in archive else None, count)
        if not np.any(ids != ""):
            ids = np.array([f"sample-{index}" for index in range(count)], dtype=str)
        splits = _string_array(
            archive["evaluation_splits"] if "evaluation_splits" in archive else None,
            count,
        )
        metadata: dict[str, Any] = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
    return _logical_fingerprint(ids, labels, splits, layers, metadata)


def load_activation_archive(path: Path) -> ActivationArchive:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Activation archive does not exist: {source}")
    with np.load(source, allow_pickle=True) as archive:
        if "activations" not in archive or "layers" not in archive or "labels" not in archive:
            raise KeyError("Activation archive must contain activations, layers, and labels")
        activations = np.asarray(archive["activations"], dtype=np.float32)
        layers = np.asarray(archive["layers"], dtype=np.int32)
        labels = np.asarray(archive["labels"], dtype=np.int32)
        if activations.ndim != 3:
            raise ValueError(
                f"Representation baselines require [sample, layer, hidden] activations, got {activations.shape}"
            )
        count, layer_count, _ = activations.shape
        if layers.shape != (layer_count,) or labels.shape != (count,):
            raise ValueError("Activation archive dimensions are inconsistent")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("Activation archive must contain benign label 0 and malicious label 1")
        metadata: dict[str, Any] = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        fields: dict[str, np.ndarray] = {}
        for key in [
            "ids",
            "evaluation_splits",
            "pair_keys",
            "conditions",
            "intent_families",
            "sources",
            "carrier_types",
            "image_roles",
            "image_paths",
            "prompt_texts",
            "label_names",
            "intent_ids",
        ]:
            fields[key] = _string_array(archive[key] if key in archive else None, count)
    ids = fields["ids"]
    if not np.any(ids != ""):
        ids = np.array([f"sample-{index}" for index in range(len(labels))], dtype=str)
        fields["ids"] = ids
    fingerprint = _logical_fingerprint(ids, labels, fields["evaluation_splits"], layers, metadata)
    return ActivationArchive(
        path=source,
        activations=activations,
        layers=layers,
        labels=labels,
        fields=fields,
        metadata=metadata,
        logical_fingerprint=fingerprint,
    )


def split_mask(data: ActivationArchive, split: str) -> np.ndarray:
    values = data.fields["evaluation_splits"]
    mask = values == split
    if not np.any(mask):
        available = sorted(set(values.tolist()))
        raise ValueError(f"Split {split!r} is empty; available splits: {available}")
    if set(np.unique(data.labels[mask]).tolist()) != {0, 1}:
        raise ValueError(f"Split {split!r} must contain both labels")
    return mask


def layer_index(data: ActivationArchive, layer: int) -> int:
    matches = np.flatnonzero(data.layers == int(layer))
    if len(matches) != 1:
        raise ValueError(f"Layer {layer} is not present exactly once in {data.layers.tolist()}")
    return int(matches[0])


def _mean_pairwise_distance(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    squared_norms = np.sum(values.astype(np.float64) ** 2, axis=1)
    squared = np.maximum(
        0.0,
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (values.astype(np.float64) @ values.astype(np.float64).T),
    )
    upper = np.triu_indices(len(values), k=1)
    return float(np.sqrt(squared[upper]).mean())


def _sample_balanced_indices(
    labels: np.ndarray,
    maximum_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) > maximum_per_class:
            candidates = np.sort(rng.choice(candidates, maximum_per_class, replace=False))
        selected.extend(candidates.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def select_rcs_layers(
    data: ActivationArchive,
    split: str,
    maximum_per_class: int,
    seed: int,
    mode: str = "geometric",
) -> list[dict[str, float | int]]:
    if mode == "official-composite":
        return select_rcs_layers_official_composite(
            data,
            split=split,
            maximum_per_class=maximum_per_class,
            seed=seed,
        )
    if mode != "geometric":
        raise ValueError(f"Unknown RCS layer-selection mode: {mode}")
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - exercised in the server environment
        raise RuntimeError(
            "RCS layer selection requires scikit-learn>=1.3 from requirements.txt"
        ) from exc

    mask = split_mask(data, split)
    split_indices = np.flatnonzero(mask)
    sampled_local = _sample_balanced_indices(
        data.labels[mask], maximum_per_class=maximum_per_class, seed=seed
    )
    selected = split_indices[sampled_local]
    labels = data.labels[selected]
    raw_metrics: list[dict[str, float | int]] = []
    for position, layer in enumerate(data.layers.tolist()):
        features = data.activations[selected, position, :].astype(np.float64)
        classifier = LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=seed)
        classifier.fit(features, labels)
        weight_norm = float(np.linalg.norm(classifier.coef_))
        margin = 2.0 / max(weight_norm, 1e-12)
        silhouette = float(silhouette_score(features, labels, metric="euclidean"))
        benign = features[labels == 0]
        malicious = features[labels == 1]
        inter = float(np.linalg.norm(benign.mean(axis=0) - malicious.mean(axis=0)))
        denominator = 0.5 * (
            _mean_pairwise_distance(benign) + _mean_pairwise_distance(malicious)
        )
        ratio = inter / max(denominator, 1e-12)
        raw_metrics.append(
            {
                "layer": int(layer),
                "margin": margin,
                "silhouette": silhouette,
                "discriminative_ratio": ratio,
            }
        )

    for metric in ("margin", "silhouette", "discriminative_ratio"):
        values = np.array([float(row[metric]) for row in raw_metrics], dtype=np.float64)
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        scale = max(float(q75 - q25), 1e-12)
        normalized = (values - median) / scale
        mapped = 1.0 / (1.0 + np.exp(-2.0 * np.clip(normalized, -30.0, 30.0)))
        for row, value in zip(raw_metrics, mapped):
            row[f"normalized_{metric}"] = float(value)
    for row in raw_metrics:
        row["geometric_score"] = float(
            (
                float(row["normalized_margin"])
                + float(row["normalized_silhouette"])
                + float(row["normalized_discriminative_ratio"])
            )
            / 3.0
        )
    return sorted(
        raw_metrics,
        key=lambda row: (float(row["geometric_score"]), int(row["layer"])),
        reverse=True,
    )


def _rcs_mmd(first: np.ndarray, second: np.ndarray) -> float:
    from sklearn.metrics.pairwise import rbf_kernel

    gamma = 1.0 / first.shape[1]
    first_kernel = rbf_kernel(first, first, gamma=gamma)
    second_kernel = rbf_kernel(second, second, gamma=gamma)
    cross_kernel = rbf_kernel(first, second, gamma=gamma)
    first_count, second_count = len(first), len(second)
    first_term = (
        (float(first_kernel.sum()) - float(np.trace(first_kernel)))
        / (first_count * (first_count - 1))
        if first_count > 1
        else 0.0
    )
    second_term = (
        (float(second_kernel.sum()) - float(np.trace(second_kernel)))
        / (second_count * (second_count - 1))
        if second_count > 1
        else 0.0
    )
    cross_term = float(cross_kernel.sum()) / (first_count * second_count)
    return float(np.sqrt(max(0.0, first_term + second_term - 2.0 * cross_term)))


def _rcs_sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.RandomState,
    projections: int = 50,
) -> float:
    from scipy.stats import wasserstein_distance

    if first.shape[1] == 1:
        return float(wasserstein_distance(first.ravel(), second.ravel()))
    if first.shape[1] == 2:
        return float(
            0.5
            * (
                wasserstein_distance(first[:, 0], second[:, 0])
                + wasserstein_distance(first[:, 1], second[:, 1])
            )
        )
    distances: list[float] = []
    for _ in range(projections):
        direction = rng.randn(first.shape[1])
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        distances.append(
            float(wasserstein_distance(first @ direction, second @ direction))
        )
    return float(np.mean(distances))


def _rcs_js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity

    combined = np.vstack([first, second])
    component_count = min(2, first.shape[1])
    projected = PCA(n_components=component_count).fit_transform(combined)
    first_projected = projected[: len(first)]
    second_projected = projected[len(first) :]
    first_kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(first_projected)
    second_kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(second_projected)
    lower = np.minimum(first_projected.min(axis=0), second_projected.min(axis=0))
    upper = np.maximum(first_projected.max(axis=0), second_projected.max(axis=0))
    if component_count == 1:
        points = np.linspace(lower, upper, 100).reshape(-1, 1)
    else:
        first_axis = np.linspace(lower[0], upper[0], 50)
        second_axis = np.linspace(lower[1], upper[1], 50)
        first_grid, second_grid = np.meshgrid(first_axis, second_axis)
        points = np.column_stack([first_grid.ravel(), second_grid.ravel()])
    first_density = np.exp(first_kde.score_samples(points))
    second_density = np.exp(second_kde.score_samples(points))
    first_density /= max(float(first_density.sum()), 1e-300)
    second_density /= max(float(second_density.sum()), 1e-300)
    midpoint = 0.5 * (first_density + second_density)
    epsilon = 1e-10
    first_density = np.maximum(first_density, epsilon)
    second_density = np.maximum(second_density, epsilon)
    midpoint = np.maximum(midpoint, epsilon)
    return float(
        0.5 * np.sum(first_density * np.log(first_density / midpoint))
        + 0.5 * np.sum(second_density * np.log(second_density / midpoint))
    )


def _rcs_entropy_reduction(features: np.ndarray, labels: np.ndarray) -> float:
    import pandas as pd
    from scipy.stats import entropy
    from sklearn.decomposition import PCA

    _, counts = np.unique(labels, return_counts=True)
    base_entropy = float(entropy(counts / len(labels), base=2))
    projected = PCA(n_components=1).fit_transform(features).ravel()
    bin_count = min(10, len(np.unique(projected)))
    if bin_count <= 1:
        return 0.0
    binned = np.asarray(pd.cut(projected, bins=bin_count, labels=False))
    conditional = 0.0
    for value in np.unique(binned[~pd.isna(binned)]):
        mask = binned == value
        subset = labels[mask]
        if not len(subset):
            continue
        _, subset_counts = np.unique(subset, return_counts=True)
        conditional += float(mask.mean()) * float(
            entropy(subset_counts / len(subset), base=2)
        )
    return base_entropy - conditional


def _rcs_official_layer_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.RandomState,
) -> dict[str, float]:
    from scipy.spatial.distance import pdist
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    benign = features[labels == 0]
    malicious = features[labels == 1]
    scaled = StandardScaler().fit_transform(features)
    scaled_benign = scaled[labels == 0]
    scaled_malicious = scaled[labels == 1]
    classifier = SVC(kernel="linear", C=1.0).fit(scaled, labels)
    inter_distance = float(
        np.linalg.norm(scaled_benign.mean(axis=0) - scaled_malicious.mean(axis=0))
    )
    benign_intra = float(np.mean(pdist(scaled_benign))) if len(scaled_benign) > 1 else 0.0
    malicious_intra = (
        float(np.mean(pdist(scaled_malicious))) if len(scaled_malicious) > 1 else 0.0
    )
    average_intra = 0.5 * (benign_intra + malicious_intra)
    return {
        "mmd": _rcs_mmd(benign, malicious),
        "wasserstein": _rcs_sliced_wasserstein(benign, malicious, rng),
        "kl_divergence": _rcs_js_divergence(benign, malicious),
        "svm_margin": 2.0 / max(float(np.linalg.norm(classifier.coef_)), 1e-12),
        "silhouette": float(silhouette_score(scaled, labels)),
        "distance_ratio": inter_distance / max(average_intra, 1e-12),
        "mutual_info": float(
            np.mean(mutual_info_classif(scaled, labels, random_state=42))
        ),
        "entropy_reduction": _rcs_entropy_reduction(features, labels),
    }


def _rcs_robust_normalization(
    raw_metrics: list[dict[str, float | int]],
    metric_names: tuple[str, ...],
) -> None:
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in raw_metrics], dtype=np.float64)
        if np.all(values == 0.0):
            normalized = np.zeros_like(values)
        else:
            median = float(np.median(values))
            q75, q25 = np.percentile(values, [75, 25])
            scale = float(q75 - q25)
            if scale == 0.0:
                minimum, maximum = float(values.min()), float(values.max())
                normalized = (
                    np.ones_like(values)
                    if minimum == maximum
                    else (values - minimum) / (maximum - minimum)
                )
            else:
                robust = np.clip((values - median) / scale, -30.0, 30.0)
                normalized = 1.0 / (1.0 + np.exp(-2.0 * robust))
        for row, value in zip(raw_metrics, normalized):
            row[f"normalized_{metric}"] = float(value)


def select_rcs_layers_official_composite(
    data: ActivationArchive,
    split: str,
    maximum_per_class: int,
    seed: int,
) -> list[dict[str, float | int]]:
    mask = split_mask(data, split)
    split_indices = np.flatnonzero(mask)
    sampled_local = _sample_balanced_indices(
        data.labels[mask], maximum_per_class=maximum_per_class, seed=seed
    )
    selected = split_indices[sampled_local]
    labels = data.labels[selected]
    rng = np.random.RandomState(seed)
    raw_metrics: list[dict[str, float | int]] = []
    for position, layer in enumerate(data.layers.tolist()):
        features = data.activations[selected, position, :].astype(np.float64)
        raw_metrics.append(
            {
                "layer": int(layer),
                "selection_sample_count": int(len(selected)),
                **_rcs_official_layer_metrics(features, labels, rng),
            }
        )
    metric_names = (
        "mmd",
        "wasserstein",
        "kl_divergence",
        "svm_margin",
        "silhouette",
        "distance_ratio",
        "mutual_info",
        "entropy_reduction",
    )
    _rcs_robust_normalization(raw_metrics, metric_names)
    for row in raw_metrics:
        distributional = np.mean(
            [
                float(row["normalized_mmd"]),
                float(row["normalized_wasserstein"]),
                float(row["normalized_kl_divergence"]),
            ]
        )
        geometric = np.mean(
            [
                float(row["normalized_svm_margin"]),
                float(row["normalized_silhouette"]),
                float(row["normalized_distance_ratio"]),
            ]
        )
        information = np.mean(
            [
                float(row["normalized_mutual_info"]),
                float(row["normalized_entropy_reduction"]),
            ]
        )
        row["distributional_score"] = float(distributional)
        row["geometric_score"] = float(geometric)
        row["information_score"] = float(information)
        # Preserve the released analysis script's 0.4/0.4/0.3 weighting.
        row["overall_score"] = float(
            0.4 * distributional + 0.4 * geometric + 0.3 * information
        )
    return sorted(
        raw_metrics,
        key=lambda row: (float(row["overall_score"]), int(row["layer"])),
        reverse=True,
    )


def _projection_arrays(model: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for index, (linear, batch_norm) in enumerate(zip(model.linears, model.batch_norms)):
        arrays[f"proj_linear_{index}_weight"] = linear.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"proj_linear_{index}_bias"] = linear.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"proj_bn_{index}_weight"] = batch_norm.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"proj_bn_{index}_bias"] = batch_norm.bias.detach().cpu().numpy().astype(np.float32)
        arrays[f"proj_bn_{index}_running_mean"] = (
            batch_norm.running_mean.detach().cpu().numpy().astype(np.float32)
        )
        arrays[f"proj_bn_{index}_running_var"] = (
            batch_norm.running_var.detach().cpu().numpy().astype(np.float32)
        )
    return arrays


def train_rcs_projection(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_names: np.ndarray,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised in the server environment
        raise RuntimeError("RCS projection training requires PyTorch from requirements.txt") from exc

    class Projection(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            dimensions = [input_dim, hidden_dim, hidden_dim // 2, output_dim]
            self.linears = nn.ModuleList(
                [nn.Linear(dimensions[index], dimensions[index + 1]) for index in range(3)]
            )
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(dimensions[index + 1]) for index in range(3)]
            )
            self.dropout = nn.Dropout(dropout)
            for linear in self.linears:
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

        def forward(self, value: Any) -> Any:
            for index, (linear, batch_norm) in enumerate(zip(self.linears, self.batch_norms)):
                value = batch_norm(linear(value))
                if index < 2:
                    value = self.dropout(functional.relu(value))
            return value

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    unique_datasets = sorted(set(dataset_names.tolist()))
    dataset_to_id = {name: index for index, name in enumerate(unique_datasets)}
    dataset_ids = np.array([dataset_to_id[name] for name in dataset_names], dtype=np.int64)
    tensor_dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(dataset_ids),
        torch.from_numpy(labels.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        tensor_dataset,
        batch_size=min(batch_size, len(tensor_dataset)),
        shuffle=True,
        drop_last=len(tensor_dataset) >= batch_size,
        generator=generator,
    )
    model = Projection(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=learning_rate * 0.05,
    )
    best_loss = float("inf")
    patience = 0
    completed_epochs = 0
    final_components = {"dataset_loss": float("nan"), "toxicity_loss": float("nan")}
    model.train()
    for epoch in range(epochs):
        total = 0.0
        dataset_total = 0.0
        toxicity_total = 0.0
        batches = 0
        for batch_features, batch_datasets, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_datasets = batch_datasets.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            embeddings = model(batch_features)
            normalized = functional.normalize(embeddings, p=2, dim=1)
            distances = torch.cdist(normalized, normalized, p=2)
            diagonal = 1.0 - torch.eye(len(normalized), device=device)
            same_dataset = (batch_datasets[:, None] == batch_datasets[None, :]).float() * diagonal
            different_dataset = (1.0 - same_dataset) * diagonal
            intra_dataset = (
                (distances * same_dataset).sum() / same_dataset.sum()
                if same_dataset.sum() > 0
                else distances.new_tensor(0.0)
            )
            inter_dataset = (
                (torch.clamp(1.0 - distances, min=0.0) * different_dataset).sum()
                / different_dataset.sum()
                if different_dataset.sum() > 0
                else distances.new_tensor(0.0)
            )
            dataset_loss = intra_dataset + inter_dataset

            benign = normalized[batch_labels == 0]
            malicious = normalized[batch_labels == 1]
            if len(benign) and len(malicious):
                benign_centroid = benign.mean(dim=0)
                malicious_centroid = malicious.mean(dim=0)
                toxicity_loss = torch.clamp(
                    2.0 - torch.norm(benign_centroid - malicious_centroid), min=0.0
                )
                if len(benign) > 1:
                    toxicity_loss = toxicity_loss + torch.norm(
                        benign - benign_centroid[None, :], dim=1
                    ).mean()
                if len(malicious) > 1:
                    toxicity_loss = toxicity_loss + torch.norm(
                        malicious - malicious_centroid[None, :], dim=1
                    ).mean()
            else:
                toxicity_loss = distances.new_tensor(0.0)
            loss = dataset_loss + 5.0 * toxicity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.item())
            dataset_total += float(dataset_loss.item())
            toxicity_total += float(toxicity_loss.item())
            batches += 1
        if batches == 0:
            raise RuntimeError("RCS projection training produced no batches")
        average = total / batches
        scheduler.step()
        completed_epochs = epoch + 1
        final_components = {
            "dataset_loss": dataset_total / batches,
            "toxicity_loss": toxicity_total / batches,
        }
        if average < best_loss:
            best_loss = average
            patience = 0
        else:
            patience += 1
        if patience >= 15:
            break
    model.eval()
    arrays = _projection_arrays(model)
    metadata = {
        "projection_architecture": [features.shape[1], hidden_dim, hidden_dim // 2, output_dim],
        "projection_epochs_requested": epochs,
        "projection_epochs_completed": completed_epochs,
        "projection_batch_size": batch_size,
        "projection_learning_rate": learning_rate,
        "projection_dropout": dropout,
        "projection_alpha_dataset": 1.0,
        "projection_beta_safety": 5.0,
        "projection_best_training_loss": best_loss,
        "projection_final_loss_components": final_components,
        "projection_device": str(device),
        "dataset_name_to_id": dataset_to_id,
        "batch_norm_epsilon": 1e-5,
    }
    return arrays, metadata


def _vlmguard_unlabeled_indices(
    data: ActivationArchive,
    split: str,
    contamination: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = split_mask(data, split)
    benign = np.flatnonzero(mask & (data.labels == 0))
    malicious = np.flatnonzero(mask & (data.labels == 1))
    target_malicious = max(
        1,
        int(round(len(benign) * contamination / max(1.0 - contamination, 1e-12))),
    )
    selected_malicious_count = min(target_malicious, len(malicious))
    rng = np.random.default_rng(seed)
    selected_malicious = np.sort(
        rng.choice(malicious, selected_malicious_count, replace=False)
    )
    selected = np.sort(np.concatenate([benign, selected_malicious])).astype(np.int64)
    return selected, {
        "unlabeled_benign_count": int(len(benign)),
        "unlabeled_malicious_count": int(selected_malicious_count),
        "unlabeled_total_count": int(len(selected)),
        "requested_contamination": float(contamination),
        "actual_contamination": float(selected_malicious_count / len(selected)),
        "contamination_membership_labels_used_only_for_protocol_construction": True,
    }


def _vlmguard_projection_scores(
    features: np.ndarray,
    center: np.ndarray,
    components: np.ndarray,
    singular_values: np.ndarray,
) -> np.ndarray:
    centered = features.astype(np.float64) - center.astype(np.float64)[None, :]
    projections = centered @ components.astype(np.float64).T
    return np.mean(
        projections**2 * singular_values.astype(np.float64)[None, :],
        axis=1,
    )


def select_vlmguard_subspace(
    data: ActivationArchive,
    unlabeled_indices: np.ndarray,
    validation_split: str,
    validation_size: int,
    maximum_k: int,
    seed: int,
    explicit_layer: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        from sklearn.utils.extmath import randomized_svd
    except ImportError as exc:  # pragma: no cover - exercised in the server environment
        raise RuntimeError(
            "VLMGuard subspace selection requires scikit-learn>=1.3 from requirements.txt"
        ) from exc

    validation_mask = split_mask(data, validation_split)
    validation_pool = np.flatnonzero(validation_mask)
    sampled_local = _sample_balanced_indices(
        data.labels[validation_mask],
        maximum_per_class=max(validation_size // 2, 1),
        seed=seed,
    )
    validation_indices = validation_pool[sampled_local]
    labels = data.labels[validation_indices]
    if explicit_layer is None:
        candidate_positions = list(range(len(data.layers)))
    else:
        candidate_positions = [layer_index(data, explicit_layer)]

    candidates: list[dict[str, Any]] = []
    fitted: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for position in candidate_positions:
        layer = int(data.layers[position])
        mixture = data.activations[unlabeled_indices, position, :].astype(np.float64)
        center = mixture.mean(axis=0)
        centered = mixture - center[None, :]
        component_count = min(maximum_k, len(mixture) - 1, mixture.shape[1])
        if component_count < 1:
            raise ValueError("VLMGuard requires at least two unlabeled prompts")
        _, singular_values, components = randomized_svd(
            centered,
            n_components=component_count,
            n_iter=5,
            random_state=seed,
        )
        validation_centered = (
            data.activations[validation_indices, position, :].astype(np.float64)
            - center[None, :]
        )
        projections = validation_centered @ components.T
        weighted_energy = projections**2 * singular_values[None, :]
        cumulative = np.cumsum(weighted_energy, axis=1)
        for k in range(1, component_count + 1):
            scores = cumulative[:, k - 1] / float(k)
            auroc = binary_auroc(labels.tolist(), scores.tolist())
            if auroc is None:
                raise RuntimeError("VLMGuard validation AUROC is undefined")
            candidates.append(
                {
                    "layer": layer,
                    "k": k,
                    "validation_auroc": float(auroc),
                    "validation_size": int(len(validation_indices)),
                }
            )
            fitted[(layer, k)] = (
                center.astype(np.float32),
                components[:k].astype(np.float32),
                singular_values[:k].astype(np.float32),
            )
    ranking = sorted(
        candidates,
        key=lambda row: (
            float(row["validation_auroc"]),
            -int(row["k"]),
            int(row["layer"]),
        ),
        reverse=True,
    )
    best = dict(ranking[0])
    center, components, singular_values = fitted[(int(best["layer"]), int(best["k"]))]
    best.update(
        {
            "center": center,
            "components": components,
            "singular_values": singular_values,
            "validation_indices": validation_indices,
        }
    )
    return best, ranking


def calibrate_vlmguard_partition_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    unique = np.unique(scores)
    if len(unique) == 1:
        candidates = unique
    else:
        candidates = np.concatenate(
            [
                [np.nextafter(unique[0], -np.inf)],
                (unique[:-1] + unique[1:]) / 2.0,
                [unique[-1]],
            ]
        )
    best_key: tuple[float, float, float] | None = None
    best_threshold = float(candidates[0])
    best_metrics: dict[str, Any] = {}
    for threshold in candidates:
        metrics = _classification_metrics(labels, scores, float(threshold))
        balanced_accuracy = 0.5 * (
            float(metrics["tpr"] or 0.0) + (1.0 - float(metrics["fpr"] or 0.0))
        )
        key = (balanced_accuracy, -float(metrics["fpr"] or 0.0), float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = dict(metrics)
            best_metrics["balanced_accuracy"] = balanced_accuracy
    best_metrics["candidate_count"] = int(len(candidates))
    best_metrics["tie_break"] = "lower_fpr_then_higher_threshold"
    return best_threshold, best_metrics


def train_vlmguard_classifier(
    features: np.ndarray,
    pseudo_labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised in the server environment
        raise RuntimeError("VLMGuard classifier training requires PyTorch") from exc

    class PromptClassifier(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.linears = nn.ModuleList(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Linear(hidden_dim, 1),
                ]
            )

        def forward(self, value: Any) -> Any:
            value = functional.relu(self.linears[0](value))
            value = functional.relu(self.linears[1](value))
            return self.linears[2](value).squeeze(-1)

    labels = np.asarray(pseudo_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError(
            "VLMGuard SVD partition produced only one pseudo class; inspect contamination and validation threshold"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = PromptClassifier(features.shape[1]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )
    criterion = nn.BCEWithLogitsLoss()
    final_loss = float("nan")
    for _ in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if not losses:
            raise RuntimeError("VLMGuard classifier training produced no batches")
        final_loss = float(np.mean(losses))
        scheduler.step()
    arrays: dict[str, np.ndarray] = {}
    model.eval()
    for index, linear in enumerate(model.linears):
        arrays[f"classifier_linear_{index}_weight"] = (
            linear.weight.detach().cpu().numpy().astype(np.float32)
        )
        arrays[f"classifier_linear_{index}_bias"] = (
            linear.bias.detach().cpu().numpy().astype(np.float32)
        )
    return arrays, {
        "classifier_architecture": [features.shape[1], hidden_dim, hidden_dim, 1],
        "classifier_activation": "relu_after_first_two_linear_layers",
        "classifier_loss": "unweighted_binary_cross_entropy_with_logits",
        "classifier_optimizer": "sgd",
        "classifier_epochs": int(epochs),
        "classifier_batch_size": int(batch_size),
        "classifier_learning_rate": float(learning_rate),
        "classifier_weight_decay": float(weight_decay),
        "classifier_lr_schedule": "cosine",
        "classifier_final_training_loss": final_loss,
        "classifier_device": str(device),
        "pseudo_benign_count": int(counts[0]),
        "pseudo_malicious_count": int(counts[1]),
    }


def _l2_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def fit_kcd(projected: np.ndarray, labels: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    benign = _l2_rows(projected[labels == 0])
    malicious = _l2_rows(projected[labels == 1])
    if not len(benign) or not len(malicious):
        raise ValueError("RCS-KCD requires both benign and malicious reference samples")
    return {
        "benign_reference": benign,
        "malicious_reference": malicious,
    }, {
        "k": int(k),
        "benign_reference_count": len(benign),
        "malicious_reference_count": len(malicious),
        "kcd_l2_normalized": True,
    }


def fit_mcd(
    projected: np.ndarray,
    labels: np.ndarray,
    dataset_names: np.ndarray,
    minimum_cluster_size: int,
    covariance_mode: str = "sklearn-ledoit-wolf",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if covariance_mode not in {"sklearn-ledoit-wolf", "released-analytical-shrinkage"}:
        raise ValueError(f"Unknown RCS-MCD covariance mode: {covariance_mode}")
    if covariance_mode == "sklearn-ledoit-wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as exc:  # pragma: no cover - exercised in the server environment
            raise RuntimeError("RCS-MCD requires scikit-learn>=1.3 from requirements.txt") from exc

    def estimate(cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
        values = cluster.astype(np.float64)
        if covariance_mode == "sklearn-ledoit-wolf":
            estimator = LedoitWolf(assume_centered=False).fit(values)
            return (
                estimator.location_.astype(np.float32),
                estimator.precision_.astype(np.float32),
                float(estimator.shrinkage_),
            )
        count, dimension = values.shape
        if count < minimum_cluster_size:
            raise ValueError(
                f"Released RCS-MCD covariance requires at least {minimum_cluster_size} "
                f"samples per source cluster, got {count}."
            )
        mean = values.mean(axis=0)
        sample_covariance = np.cov(values.T, bias=False)
        trace = float(np.trace(sample_covariance))
        frobenius_squared = float(np.sum(sample_covariance**2))
        denominator = (count + 2) * (
            frobenius_squared - (trace**2) / dimension
        )
        if denominator <= 0.0:
            shrinkage = 0.0
        else:
            numerator = ((count - 2) / count) * frobenius_squared + trace**2
            shrinkage = min(1.0, max(0.0, numerator / denominator))
        target = (trace / dimension) * np.eye(dimension, dtype=np.float64)
        covariance = (
            (1.0 - shrinkage) * sample_covariance + shrinkage * target
        )
        precision = np.linalg.pinv(covariance)
        return mean.astype(np.float32), precision.astype(np.float32), shrinkage

    outputs: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"mcd_l2_normalized": False}
    for label, prefix in ((0, "benign"), (1, "malicious")):
        means: list[np.ndarray] = []
        precisions: list[np.ndarray] = []
        shrinkages: list[float] = []
        names: list[str] = []
        for dataset_name in sorted(set(dataset_names[labels == label].tolist())):
            mask = (labels == label) & (dataset_names == dataset_name)
            cluster = projected[mask].astype(np.float64)
            if len(cluster) < minimum_cluster_size:
                continue
            mean, precision, shrinkage = estimate(cluster)
            means.append(mean)
            precisions.append(precision)
            if shrinkage is not None:
                shrinkages.append(float(shrinkage))
            names.append(str(dataset_name))
        if not means:
            cluster = projected[labels == label].astype(np.float64)
            if len(cluster) < minimum_cluster_size:
                raise ValueError(f"RCS-MCD has too few {prefix} samples for covariance estimation")
            mean, precision, shrinkage = estimate(cluster)
            means = [mean]
            precisions = [precision]
            if shrinkage is not None:
                shrinkages = [float(shrinkage)]
            names = [f"all_{prefix}"]
        outputs[f"{prefix}_means"] = np.stack(means)
        outputs[f"{prefix}_precisions"] = np.stack(precisions)
        metadata[f"{prefix}_cluster_names"] = names
        metadata[f"{prefix}_cluster_count"] = len(names)
        metadata[f"{prefix}_cluster_shrinkage"] = shrinkages
    metadata["mcd_covariance"] = covariance_mode
    metadata["mcd_minimum_cluster_size"] = minimum_cluster_size
    return outputs, metadata


def _classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > threshold).astype(np.int32)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else None
    fpr = fp / negatives if negatives else None
    f1 = 2.0 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
    auroc = binary_auroc(labels.tolist(), scores.tolist())
    auprc = None
    if positives:
        order = np.argsort(-scores, kind="stable")
        ordered_labels = labels[order]
        cumulative = np.cumsum(ordered_labels == 1)
        positive_positions = np.flatnonzero(ordered_labels == 1)
        auprc = float(
            np.mean(cumulative[positive_positions] / (positive_positions.astype(np.float64) + 1.0))
        )
    return {
        "n": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels) if len(labels) else None,
        "precision": precision,
        "tpr": recall,
        "fpr": fpr,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "score_min": float(np.min(scores)) if len(scores) else None,
        "score_median": float(np.median(scores)) if len(scores) else None,
        "score_max": float(np.max(scores)) if len(scores) else None,
    }


def calibrate_rcs_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    method: str,
) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    benign = scores[labels == 0]
    malicious = scores[labels == 1]
    if not len(benign) or not len(malicious):
        raise ValueError("RCS threshold calibration requires both labels")
    benign_mean, malicious_mean = float(benign.mean()), float(malicious.mean())
    benign_std, malicious_std = float(benign.std()), float(malicious.std())
    separation = abs(malicious_mean - benign_mean) / (benign_std + malicious_std + 1e-8)
    cutoff = 2.0 if method == "rcs-kcd" else 5.0
    if separation > cutoff:
        midpoint = 0.5 * (benign_mean + malicious_mean)
        width = 2.0 * min(benign_std, malicious_std)
        if width <= 1e-12:
            width = max(abs(malicious_mean - benign_mean) * 0.5, 1e-6)
        low, high = midpoint - width, midpoint + width
        search_mode = "released_adaptive_narrow_grid"
    else:
        low, high = np.percentile(scores, [5, 95]).tolist()
        width = high - low
        low -= 0.2 * width
        high += 0.2 * width
        search_mode = "released_adaptive_wide_grid"
    candidates = np.linspace(low, high, 200)
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in candidates:
        metrics = _classification_metrics(labels, scores, float(threshold))
        balanced_accuracy = 0.5 * (
            float(metrics["tpr"] or 0.0) + (1.0 - float(metrics["fpr"] or 0.0))
        )
        objective = 0.8 * balanced_accuracy + 0.2 * float(metrics["f1"])
        candidate = (objective, -float(threshold), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
            best_threshold = float(threshold)
    assert best is not None
    metrics = dict(best[2])
    metrics.update(
        {
            "balanced_accuracy": 0.5
            * (float(metrics["tpr"] or 0.0) + (1.0 - float(metrics["fpr"] or 0.0))),
            "objective": best[0],
            "separation": separation,
            "search_mode": search_mode,
            "search_low": float(low),
            "search_high": float(high),
            "candidate_count": len(candidates),
        }
    )
    return best_threshold, metrics


def _unique_in_order(values: np.ndarray) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values.tolist():
        name = str(value)
        if name not in seen:
            seen.add(name)
            output.append(name)
    return output


def rcs_repository_threshold_indices(
    labels: np.ndarray,
    dataset_names: np.ndarray,
    train_mask: np.ndarray,
    maximum_per_source: int,
    seed: int,
) -> np.ndarray:
    if maximum_per_source <= 0:
        raise ValueError("RCS threshold samples per source must be positive")
    rng = random.Random(seed)
    selected: list[int] = []
    for label in (0, 1):
        label_mask = train_mask & (labels == label)
        for dataset_name in _unique_in_order(dataset_names[label_mask]):
            candidates = np.flatnonzero(label_mask & (dataset_names == dataset_name)).tolist()
            sample_size = min(maximum_per_source, len(candidates))
            selected.extend(rng.sample(candidates, sample_size))
    indices = np.asarray(selected, dtype=np.int64)
    if set(np.unique(labels[indices]).tolist()) != {0, 1}:
        raise ValueError("Released RCS threshold sampling did not produce both labels")
    return indices


def fit_nearside_direction(
    data: ActivationArchive,
    layer: int,
    train_split: str,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    mask = split_mask(data, train_split)
    position = layer_index(data, layer)
    pair_keys = data.fields["pair_keys"]
    differences: list[np.ndarray] = []
    for key in sorted(set(pair_keys[mask].tolist())):
        if not key:
            continue
        pair_mask = mask & (pair_keys == key)
        benign = data.activations[pair_mask & (data.labels == 0), position, :]
        malicious = data.activations[pair_mask & (data.labels == 1), position, :]
        if len(benign) and len(malicious):
            differences.append(malicious.mean(axis=0) - benign.mean(axis=0))
    if not differences:
        raise ValueError(
            "NEARSIDE requires paired benign/malicious training rows with matching pair_keys"
        )
    direction = np.mean(np.stack(differences).astype(np.float64), axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("NEARSIDE fitted a zero attack direction")
    unit = direction / norm
    train_scores = data.activations[mask, position, :].astype(np.float64) @ unit
    threshold = float(train_scores.mean())
    return direction.astype(np.float32), threshold, {
        "paired_difference_count": len(differences),
        "direction_norm": norm,
        "threshold_method": "released_mean_projection_over_training_pairs",
        "training_score_mean": threshold,
        "training_score_std": float(train_scores.std()),
    }


def _base_metadata(
    args: argparse.Namespace,
    data: ActivationArchive,
    method: str,
    layer: int,
    threshold: float,
) -> dict[str, Any]:
    model_id = str(data.metadata.get("model") or data.metadata.get("resolved_model") or "")
    paper_protocol_names = {
        "nearside": {"paper-nearside-radar"},
        "rcs-kcd": {"paper-rcs"},
        "rcs-mcd": {"paper-rcs"},
        "vlmguard": {"paper-vlmguard"},
    }
    declared_paper_protocol = args.protocol in paper_protocol_names[method]
    source_protocol = data.metadata.get("dataset_protocol")
    source_exact = data.metadata.get("dataset_exact_protocol")
    verified_paper_protocol = bool(
        declared_paper_protocol
        and source_protocol == args.protocol
        and source_exact is True
    )
    if declared_paper_protocol and not verified_paper_protocol:
        raise ValueError(
            f"--protocol {args.protocol} requires an activation archive with "
            f"dataset_protocol={args.protocol!r} and dataset_exact_protocol=true; "
            f"got protocol={source_protocol!r}, exact={source_exact!r}."
        )
    return {
        "method": method,
        "model_id": model_id,
        "model_revision": data.metadata.get("model_revision"),
        "backend": data.metadata.get("backend"),
        "layer": int(layer),
        "hidden_dim": int(data.activations.shape[2]),
        "pooling": str(data.metadata.get("pooling") or "last"),
        "threshold": float(threshold),
        "protocol": args.protocol,
        "paper_training_protocol": verified_paper_protocol,
        "paper_protocol_is_user_declared": declared_paper_protocol,
        "source_dataset_protocol": source_protocol,
        "source_dataset_exact_protocol": source_exact,
        "source_dataset_manifest_sha1": data.metadata.get("dataset_manifest_sha1"),
        "core_algorithm_compatible": True,
        "source_activations": str(data.path),
        "source_logical_fingerprint": data.logical_fingerprint,
        "train_split": args.train_split,
        "calibration_split": args.calibration_split if method.startswith("rcs-") else None,
        "random_seed": args.seed,
        "implementation_basis": {
            "nearside": "Huang et al. 2024 released NEARSIDE equations 3-5",
            "rcs-kcd": "Hua et al. ACL 2026 paper and released Jailbreak_Detection_RCS code",
            "rcs-mcd": "Hua et al. ACL 2026 paper and released Jailbreak_Detection_RCS code",
            "vlmguard": "Fang et al. VLMGuard arXiv v2 paper specification",
        }[method],
    }


def _dataset_names(data: ActivationArchive, field: str) -> np.ndarray:
    key = FIELD_ALIASES[field]
    values = data.fields[key]
    return np.array(
        [value if value else f"unspecified_{field}" for value in values],
        dtype=str,
    )


def build_detector(args: argparse.Namespace, data: ActivationArchive) -> RepresentationDetector:
    method = normalize_representation_method(args.method)
    source_protocol = data.metadata.get("dataset_protocol")
    if args.protocol == "paper-rcs" and (
        source_protocol != "paper-rcs"
        or data.metadata.get("dataset_exact_protocol") is not True
    ):
        raise ValueError(
            "paper-rcs requires an exact archive produced from the released RCS data composition"
        )
    if args.protocol == "repository-rcs-incomplete" and source_protocol != args.protocol:
        raise ValueError(
            "repository-rcs-incomplete requires an archive explicitly marked with that data protocol"
        )
    if method == "nearside":
        layer = int(data.layers[-1]) if args.layer is None else int(args.layer)
        direction, threshold, fit_metadata = fit_nearside_direction(
            data, layer=layer, train_split=args.train_split
        )
        metadata = _base_metadata(args, data, method, layer, threshold)
        metadata.update(fit_metadata)
        metadata["paper_scope_note"] = (
            "RADAR adversarial-image training with the released final-layer direction equations."
            if args.protocol == "paper-nearside-radar"
            else "The released method detects adversarial images; matched CISR training adapts the same "
            "direction and threshold equations to harmful-intent pairs."
        )
        return save_representation_artifact(args.out, metadata, {"direction": direction})

    if method == "vlmguard":
        unlabeled_indices, mixture_metadata = _vlmguard_unlabeled_indices(
            data,
            split=args.train_split,
            contamination=args.vlmguard_contamination,
            seed=args.seed,
        )
        selected, ranking = select_vlmguard_subspace(
            data,
            unlabeled_indices=unlabeled_indices,
            validation_split=args.selection_split,
            validation_size=args.vlmguard_validation_size,
            maximum_k=args.vlmguard_max_k,
            seed=args.seed,
            explicit_layer=args.layer,
        )
        layer = int(selected["layer"])
        position = layer_index(data, layer)
        validation_indices = np.asarray(selected["validation_indices"], dtype=np.int64)
        center = np.asarray(selected["center"], dtype=np.float32)
        components = np.asarray(selected["components"], dtype=np.float32)
        singular_values = np.asarray(selected["singular_values"], dtype=np.float32)
        validation_scores = _vlmguard_projection_scores(
            data.activations[validation_indices, position, :],
            center,
            components,
            singular_values,
        )
        partition_threshold, partition_metrics = calibrate_vlmguard_partition_threshold(
            data.labels[validation_indices], validation_scores
        )
        mixture_features = data.activations[unlabeled_indices, position, :]
        mixture_scores = _vlmguard_projection_scores(
            mixture_features,
            center,
            components,
            singular_values,
        )
        pseudo_labels = (mixture_scores > partition_threshold).astype(np.int64)
        classifier_arrays, classifier_metadata = train_vlmguard_classifier(
            mixture_features,
            pseudo_labels,
            hidden_dim=args.vlmguard_hidden_dim,
            epochs=args.vlmguard_epochs,
            batch_size=args.vlmguard_batch_size,
            learning_rate=args.vlmguard_learning_rate,
            weight_decay=args.vlmguard_weight_decay,
            seed=args.seed,
            device_name=args.projection_device,
        )
        pseudo_metrics = _classification_metrics(
            data.labels[unlabeled_indices],
            mixture_scores,
            partition_threshold,
        )
        metadata = _base_metadata(
            args,
            data,
            method,
            layer,
            args.vlmguard_output_threshold,
        )
        metadata.update(mixture_metadata)
        metadata.update(classifier_metadata)
        metadata.update(
            {
                "selection_split": args.selection_split,
                "validation_size": int(len(validation_indices)),
                "selected_k": int(selected["k"]),
                "selected_validation_auroc": float(selected["validation_auroc"]),
                "top_subspace_candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key in {"layer", "k", "validation_auroc", "validation_size"}
                    }
                    for candidate in ranking[:10]
                ],
                "partition_threshold": float(partition_threshold),
                "partition_threshold_method": "validation_balanced_accuracy",
                "partition_threshold_metrics": partition_metrics,
                "pseudo_partition_posthoc_metrics": pseudo_metrics,
                "output_threshold_method": "standard_logistic_probability_0.5_or_explicit_override",
                "official_repository_state": "README-only when checked 2026-07-17",
                "official_code_available": False,
                "implementation_caveat": (
                    "The paper specifies a three-layer ReLU MLP with intermediate dimension 1024 "
                    "but does not publish executable classifier code; this artifact uses two 1024-wide "
                    "hidden layers and one maliciousness logit."
                ),
            }
        )
        arrays = {
            **classifier_arrays,
            "svd_center": center,
            "svd_components": components,
            "svd_singular_values": singular_values,
        }
        return save_representation_artifact(args.out, metadata, arrays)

    if args.layer is None:
        ranking = select_rcs_layers(
            data,
            split=args.selection_split,
            maximum_per_class=args.layer_selection_max_per_class,
            seed=args.seed,
            mode=args.rcs_layer_selection,
        )
        if args.layer_rank < 1 or args.layer_rank > len(ranking):
            raise ValueError(f"--layer-rank must be in [1, {len(ranking)}]")
        layer = int(ranking[args.layer_rank - 1]["layer"])
    else:
        layer = int(args.layer)
        ranking = []
    position = layer_index(data, layer)
    train = split_mask(data, args.train_split)
    dataset_names = _dataset_names(data, args.dataset_field)
    projection_seed = (
        args.seed + layer
        if args.protocol in {"paper-rcs", "repository-rcs-incomplete"}
        else args.seed
    )
    projection_arrays, projection_metadata = train_rcs_projection(
        data.activations[train, position, :],
        data.labels[train],
        dataset_names[train],
        output_dim=args.projection_dim,
        hidden_dim=args.projection_hidden_dim,
        dropout=args.projection_dropout,
        epochs=args.projection_epochs,
        batch_size=args.projection_batch_size,
        learning_rate=args.projection_learning_rate,
        seed=projection_seed,
        device_name=args.projection_device,
    )
    provisional_metadata = _base_metadata(args, data, method, layer, 0.0)
    provisional_metadata.update(projection_metadata)
    provisional_metadata.update(
        {
            "dataset_field": args.dataset_field,
            "selection_split": args.selection_split,
            "layer_rank": args.layer_rank,
            "top_layers": ranking[:3],
            "layer_selection_method": (
                (
                    "released principled_layer_selection eight-metric robust composite"
                    if args.rcs_layer_selection == "official-composite"
                    else "RCS robust-IQR normalized equal-weight SVM-margin/silhouette/discriminative-ratio"
                )
                if ranking
                else "explicit layer override"
            ),
            "layer_selection_mode": args.rcs_layer_selection,
            "layer_selection_external_test_labels_used": False,
            "projection_random_seed": projection_seed,
            "projection_seed_method": (
                "released_main_seed_plus_layer"
                if args.protocol in {"paper-rcs", "repository-rcs-incomplete"}
                else "configured_seed"
            ),
        }
    )
    projection_only = RepresentationDetector(
        path=args.out,
        metadata=provisional_metadata,
        arrays=projection_arrays,
    )
    projected_train = projection_only.project_batch(data.activations[train, position, :])
    if method == "rcs-kcd":
        detector_arrays, detector_metadata = fit_kcd(projected_train, data.labels[train], args.k)
    else:
        detector_arrays, detector_metadata = fit_mcd(
            projected_train,
            data.labels[train],
            dataset_names[train],
            minimum_cluster_size=args.mcd_min_cluster_size,
            covariance_mode=(
                "released-analytical-shrinkage"
                if args.protocol in {"paper-rcs", "repository-rcs-incomplete"}
                else "sklearn-ledoit-wolf"
            ),
        )
    arrays = {**projection_arrays, **detector_arrays}
    provisional_metadata.update(detector_metadata)
    provisional = RepresentationDetector(
        path=args.out,
        metadata=provisional_metadata,
        arrays=arrays,
    )
    provisional._validate()
    if args.protocol in {"paper-rcs", "repository-rcs-incomplete"}:
        calibration_indices = rcs_repository_threshold_indices(
            data.labels,
            dataset_names,
            train_mask=train,
            maximum_per_source=args.rcs_threshold_per_source,
            seed=projection_seed,
        )
        calibration_labels = data.labels[calibration_indices]
        calibration_vectors = data.activations[calibration_indices, position, :]
        threshold_method = (
            "released_per_source_training_reference_resample_0.8_balanced_accuracy_plus_0.2_f1_grid"
        )
        threshold_overlap = True
    else:
        calibration = split_mask(data, args.calibration_split)
        calibration_indices = np.flatnonzero(calibration)
        calibration_labels = data.labels[calibration]
        calibration_vectors = data.activations[calibration, position, :]
        threshold_method = "released_0.8_balanced_accuracy_plus_0.2_f1_grid"
        threshold_overlap = False
    calibration_scores = provisional.score_vectors(calibration_vectors)
    threshold, threshold_metrics = calibrate_rcs_threshold(
        calibration_labels, calibration_scores, method
    )
    metadata = dict(provisional_metadata)
    metadata["threshold"] = threshold
    metadata["threshold_method"] = threshold_method
    metadata["threshold_metrics"] = threshold_metrics
    metadata["threshold_sample_count"] = int(len(calibration_indices))
    metadata["threshold_reference_overlap"] = threshold_overlap
    metadata["threshold_samples_per_source"] = (
        args.rcs_threshold_per_source if threshold_overlap else None
    )
    return save_representation_artifact(args.out, metadata, arrays)


def _group_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    groups: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        output[str(group or "unspecified")] = _classification_metrics(
            labels[mask], scores[mask], threshold
        )
    return output


def evaluate_detector(
    detector: RepresentationDetector,
    data: ActivationArchive,
    output_dir: Path,
) -> dict[str, Any]:
    position = layer_index(data, detector.layer)
    scores = detector.score_vectors(data.activations[:, position, :])
    direct_scores = (
        detector.vlmguard_direct_scores(data.activations[:, position, :])
        if detector.method == "vlmguard"
        else None
    )
    predictions = scores > detector.threshold
    rows = []
    for index, score in enumerate(scores):
        row = {
                "id": str(data.fields["ids"][index]),
                "label": int(data.labels[index]),
                "label_name": str(data.fields["label_names"][index]),
                "evaluation_split": str(data.fields["evaluation_splits"][index]),
                "pair_key": str(data.fields["pair_keys"][index]),
                "condition": str(data.fields["conditions"][index]),
                "carrier_type": str(data.fields["carrier_types"][index]),
                "image_role": str(data.fields["image_roles"][index]),
                "intent_family": str(data.fields["intent_families"][index]),
                "intent_id": str(data.fields["intent_ids"][index]),
                "prompt_text": str(data.fields["prompt_texts"][index]),
                "image_path": str(data.fields["image_paths"][index]) or None,
                "representation_method": detector.method,
                "representation_layer": detector.layer,
                "representation_score": float(score),
                "representation_threshold": detector.threshold,
                "representation_detected": bool(predictions[index]),
            }
        if direct_scores is not None:
            row["vlmguard_direct_projection_score"] = float(direct_scores[index])
            row["vlmguard_partition_threshold"] = float(
                detector.metadata["partition_threshold"]
            )
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "detection_results.jsonl", rows)
    splits = data.fields["evaluation_splits"]
    summary = {
        "format_version": "representation_baseline_evaluation_v1",
        "method": detector.method,
        "artifact": str(detector.path),
        "artifact_fingerprint": detector.fingerprint,
        "model_id": detector.model_id,
        "layer": detector.layer,
        "threshold": detector.threshold,
        "protocol": detector.metadata.get("protocol"),
        "paper_training_protocol": detector.paper_training_protocol,
        "core_algorithm_compatible": detector.core_algorithm_compatible,
        "overall": _classification_metrics(data.labels, scores, detector.threshold),
        "by_split": _group_metrics(data.labels, scores, detector.threshold, splits),
        "test_by_condition": {},
        "test_by_carrier_type": {},
        "test_by_image_role": {},
        "test_by_intent_family": {},
    }
    test = splits == "test"
    if np.any(test):
        summary["test_by_condition"] = _group_metrics(
            data.labels[test], scores[test], detector.threshold, data.fields["conditions"][test]
        )
        summary["test_by_carrier_type"] = _group_metrics(
            data.labels[test], scores[test], detector.threshold, data.fields["carrier_types"][test]
        )
        summary["test_by_image_role"] = _group_metrics(
            data.labels[test], scores[test], detector.threshold, data.fields["image_roles"][test]
        )
        summary["test_by_intent_family"] = _group_metrics(
            data.labels[test], scores[test], detector.threshold, data.fields["intent_families"][test]
        )
    if direct_scores is not None:
        partition_threshold = float(detector.metadata["partition_threshold"])
        summary["vlmguard_direct_projection"] = {
            "threshold": partition_threshold,
            "overall": _classification_metrics(
                data.labels, direct_scores, partition_threshold
            ),
            "by_split": _group_metrics(
                data.labels, direct_scores, partition_threshold, splits
            ),
            "test_by_condition": (
                _group_metrics(
                    data.labels[test],
                    direct_scores[test],
                    partition_threshold,
                    data.fields["conditions"][test],
                )
                if np.any(test)
                else {}
            ),
            "test_by_image_role": (
                _group_metrics(
                    data.labels[test],
                    direct_scores[test],
                    partition_threshold,
                    data.fields["image_roles"][test],
                )
                if np.any(test)
                else {}
            ),
        }
    write_json(output_dir / "detection_summary.json", summary)
    test_metrics = summary["by_split"].get("test") or summary["overall"]
    report = [
        "# VLM Representation Detector",
        "",
        f"- Method: `{detector.method}`",
        f"- Model: `{detector.model_id}`",
        f"- Layer: `{detector.layer}`",
        f"- Threshold: `{detector.threshold}`",
        f"- Protocol: `{detector.metadata.get('protocol')}`",
        f"- Core algorithm compatible: `{detector.core_algorithm_compatible}`",
        f"- Paper training protocol: `{detector.paper_training_protocol}`",
        "",
        "## Held-Out Test",
        "",
        f"- N: `{test_metrics['n']}`",
        f"- Accuracy: `{test_metrics['accuracy']}`",
        f"- TPR: `{test_metrics['tpr']}`",
        f"- FPR: `{test_metrics['fpr']}`",
        f"- F1: `{test_metrics['f1']}`",
        f"- AUROC: `{test_metrics['auroc']}`",
        f"- AUPRC: `{test_metrics['auprc']}`",
        "",
        "This report is detection-only. Test labels are used only after the artifact is frozen.",
    ]
    (output_dir / "detection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NEARSIDE, RCS, or VLMGuard on an all-layer VLM activation archive."
    )
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--method", choices=TRAINABLE_REPRESENTATION_CHOICES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        default="matched-cisr",
        help=(
            "Training protocol label. Use paper-nearside-radar or paper-rcs only when the "
            "corresponding official data composition is actually used; paper-vlmguard additionally "
            "requires the paper's benchmark mixture."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--selection-split")
    parser.add_argument("--calibration-split")
    parser.add_argument("--layer", type=int, help="Explicit hidden-state layer; defaults to final for NEARSIDE and RCS selection otherwise.")
    parser.add_argument("--layer-rank", type=int, default=1, help="Use this rank from the RCS geometric layer ordering.")
    parser.add_argument("--layer-selection-max-per-class", type=int)
    parser.add_argument("--dataset-field", choices=sorted(FIELD_ALIASES))
    parser.add_argument(
        "--rcs-layer-selection",
        choices=["geometric", "official-composite"],
        help="paper-rcs defaults to the released principled_layer_selection composite.",
    )
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-hidden-dim", type=int, default=512)
    parser.add_argument("--projection-dropout", type=float, default=0.3)
    parser.add_argument("--projection-epochs", type=int, default=100)
    parser.add_argument("--projection-batch-size", type=int, default=64)
    parser.add_argument("--projection-learning-rate", type=float, default=1e-3)
    parser.add_argument("--projection-device", default="auto")
    parser.add_argument("--k", type=int)
    parser.add_argument("--mcd-min-cluster-size", type=int)
    parser.add_argument("--rcs-threshold-per-source", type=int, default=100)
    parser.add_argument("--vlmguard-contamination", type=float, default=0.005)
    parser.add_argument("--vlmguard-validation-size", type=int, default=100)
    parser.add_argument("--vlmguard-max-k", type=int, default=15)
    parser.add_argument("--vlmguard-hidden-dim", type=int, default=1024)
    parser.add_argument("--vlmguard-epochs", type=int, default=20)
    parser.add_argument("--vlmguard-batch-size", type=int, default=128)
    parser.add_argument("--vlmguard-learning-rate", type=float, default=5e-3)
    parser.add_argument("--vlmguard-weight-decay", type=float, default=3e-4)
    parser.add_argument("--vlmguard-output-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    repository_rcs_protocol = args.protocol in {
        "paper-rcs",
        "repository-rcs-incomplete",
    }
    args.selection_split = args.selection_split or (
        "train" if repository_rcs_protocol else "validation"
    )
    args.calibration_split = args.calibration_split or (
        "train" if repository_rcs_protocol else "calibration"
    )
    if args.layer_selection_max_per_class is None:
        args.layer_selection_max_per_class = 1000 if repository_rcs_protocol else 100
    args.dataset_field = args.dataset_field or (
        "source" if repository_rcs_protocol else "condition"
    )
    args.rcs_layer_selection = args.rcs_layer_selection or (
        "official-composite" if repository_rcs_protocol else "geometric"
    )
    if args.k is None:
        args.k = 40 if repository_rcs_protocol else 50
    if args.mcd_min_cluster_size is None:
        args.mcd_min_cluster_size = 50 if repository_rcs_protocol else 2
    args.seed = args.seed if args.seed is not None else (45 if repository_rcs_protocol else 42)
    args.activations = args.activations.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else args.out.parent
    )
    if args.layer_rank <= 0 or args.layer_selection_max_per_class <= 1:
        parser.error("Layer rank must be positive and layer-selection samples must exceed one per class.")
    if args.projection_dim <= 0 or args.projection_hidden_dim < 2:
        parser.error("Projection dimensions must be positive.")
    if args.projection_epochs <= 0 or args.projection_batch_size <= 1:
        parser.error("Projection epochs must be positive and batch size must exceed one.")
    if not 0.0 <= args.projection_dropout < 1.0:
        parser.error("Projection dropout must be in [0, 1).")
    if args.k <= 0 or args.mcd_min_cluster_size < 2:
        parser.error("K must be positive and MCD clusters require at least two samples.")
    if args.rcs_threshold_per_source <= 0:
        parser.error("--rcs-threshold-per-source must be positive.")
    if not 0.0 < args.vlmguard_contamination < 0.5:
        parser.error("VLMGuard contamination must be in (0, 0.5).")
    if args.vlmguard_validation_size < 2 or args.vlmguard_max_k <= 0:
        parser.error("VLMGuard validation size must exceed one and max-k must be positive.")
    if args.vlmguard_hidden_dim <= 0 or args.vlmguard_epochs <= 0:
        parser.error("VLMGuard hidden dimension and epochs must be positive.")
    if args.vlmguard_batch_size <= 1 or args.vlmguard_learning_rate <= 0.0:
        parser.error("VLMGuard batch size must exceed one and learning rate must be positive.")
    if args.vlmguard_weight_decay < 0.0:
        parser.error("VLMGuard weight decay cannot be negative.")
    if not 0.0 <= args.vlmguard_output_threshold <= 1.0:
        parser.error("VLMGuard output threshold must be in [0, 1].")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_activation_archive(args.activations)
    detector = build_detector(args, data)
    summary = evaluate_detector(detector, data, args.output_dir)
    test = summary["by_split"].get("test") or summary["overall"]
    print(
        f"Wrote {detector.method} artifact to {detector.path}; "
        f"held-out AUROC={test['auroc']}, TPR={test['tpr']}, FPR={test['fpr']}"
    )


if __name__ == "__main__":
    main()

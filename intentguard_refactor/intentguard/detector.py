from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OTHER_ROLE = "__other__"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def role_categories_from_train(image_roles: np.ndarray) -> list[str]:
    roles = sorted({str(role) for role in image_roles.tolist() if str(role)})
    if OTHER_ROLE not in roles:
        roles.append(OTHER_ROLE)
    return roles


def build_detector_features(
    raw_coords: np.ndarray,
    residual_coords: np.ndarray,
    has_anchor: np.ndarray,
    image_roles: np.ndarray,
    role_categories: list[str],
) -> np.ndarray:
    raw = np.asarray(raw_coords, dtype=np.float64)
    residual = np.asarray(residual_coords, dtype=np.float64)
    anchor = np.asarray(has_anchor, dtype=np.float64).reshape(-1, 1)
    roles = np.asarray(image_roles).astype(str)
    role_to_index = {role: index for index, role in enumerate(role_categories)}
    other_index = role_to_index[OTHER_ROLE]
    one_hot = np.zeros((len(roles), len(role_categories)), dtype=np.float64)
    for row_index, role in enumerate(roles.tolist()):
        one_hot[row_index, role_to_index.get(role, other_index)] = 1.0
    return np.concatenate([raw, residual, anchor, one_hot], axis=1)


def standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - mean[None, :]) / std[None, :], mean, std


@dataclass
class TinyMLP:
    weight_1: np.ndarray
    bias_1: np.ndarray
    weight_2: np.ndarray
    bias_2: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        hidden = np.tanh(features @ self.weight_1 + self.bias_1[None, :])
        logits = hidden @ self.weight_2 + self.bias_2
        return sigmoid(logits.reshape(-1))


def train_tiny_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
    hidden_dim: int = 8,
    epochs: int = 1200,
    learning_rate: float = 0.02,
    l2: float = 1e-3,
    seed: int = 7,
) -> tuple[TinyMLP, dict[str, Any]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) != 2:
        raise ValueError("Tiny MLP requires both target and benign labels in the training split.")

    rng = np.random.default_rng(seed)
    input_dim = int(x.shape[1])
    scale_1 = np.sqrt(2.0 / max(1, input_dim + hidden_dim))
    weight_1 = rng.normal(0.0, scale_1, size=(input_dim, hidden_dim))
    bias_1 = np.zeros(hidden_dim, dtype=np.float64)
    weight_2 = rng.normal(0.0, np.sqrt(2.0 / max(1, hidden_dim + 1)), size=(hidden_dim, 1))
    bias_2 = np.zeros(1, dtype=np.float64)

    class_counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    class_weight = len(y) / (2.0 * np.maximum(class_counts, 1.0))
    weights = class_weight[y.astype(int)]
    if sample_weight is not None:
        weights *= np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    weights /= max(float(weights.mean()), 1e-12)
    normalizer = max(float(weights.sum()), 1e-12)

    params = [weight_1, bias_1, weight_2, bias_2]
    first_moment = [np.zeros_like(param) for param in params]
    second_moment = [np.zeros_like(param) for param in params]
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    best_loss = float("inf")
    best_params = [param.copy() for param in params]
    stale_epochs = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        hidden = np.tanh(x @ weight_1 + bias_1[None, :])
        logits = (hidden @ weight_2 + bias_2).reshape(-1)
        probabilities = sigmoid(logits)
        data_loss = -np.sum(
            weights * (y * np.log(probabilities + 1e-12) + (1.0 - y) * np.log(1.0 - probabilities + 1e-12))
        ) / normalizer
        loss = float(data_loss + 0.5 * l2 * (np.sum(weight_1**2) + np.sum(weight_2**2)))

        grad_logits = (weights * (probabilities - y) / normalizer).reshape(-1, 1)
        grad_weight_2 = hidden.T @ grad_logits + l2 * weight_2
        grad_bias_2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ weight_2.T
        grad_pre_hidden = grad_hidden * (1.0 - hidden**2)
        grad_weight_1 = x.T @ grad_pre_hidden + l2 * weight_1
        grad_bias_1 = grad_pre_hidden.sum(axis=0)
        gradients = [grad_weight_1, grad_bias_1, grad_weight_2, grad_bias_2]

        for index, (param, gradient) in enumerate(zip(params, gradients)):
            first_moment[index] = beta_1 * first_moment[index] + (1.0 - beta_1) * gradient
            second_moment[index] = beta_2 * second_moment[index] + (1.0 - beta_2) * (gradient**2)
            corrected_m = first_moment[index] / (1.0 - beta_1**epoch)
            corrected_v = second_moment[index] / (1.0 - beta_2**epoch)
            param -= learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)

        completed_epochs = epoch
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_params = [param.copy() for param in params]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 120:
            break

    model = TinyMLP(*best_params)
    return model, {
        "epochs": completed_epochs,
        "best_weighted_bce": best_loss,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "l2": l2,
        "class_counts": {"0": int(class_counts[0]), "1": int(class_counts[1])},
    }


@dataclass
class CISRDetector:
    path: Path
    model_id: str
    model_alias: str
    layer: int
    rank: int
    pooling: str
    basis: np.ndarray
    center: np.ndarray
    residual_center: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    role_categories: list[str]
    network: TinyMLP
    threshold: float
    anchor_prompt: str
    uses_anchor: bool
    format_version: str = "CISR_v2_detector_v1"
    calibration_target_tpr: float | None = None
    calibration_target_fpr: float | None = None
    coverage_confidence: float | None = None

    @classmethod
    def load(cls, path: Path | str) -> "CISRDetector":
        artifact_path = Path(path).resolve()
        data = np.load(artifact_path, allow_pickle=True)
        return cls(
            path=artifact_path,
            model_id=str(data["model_id"][0]),
            model_alias=str(data["model_alias"][0]),
            layer=int(data["layer"][0]),
            rank=int(data["rank"][0]),
            pooling=str(data["pooling"][0]),
            basis=np.asarray(data["basis"], dtype=np.float64),
            center=np.asarray(data["center"], dtype=np.float64),
            residual_center=np.asarray(data["residual_center"], dtype=np.float64),
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(data["feature_std"], dtype=np.float64),
            role_categories=[str(value) for value in data["role_categories"].tolist()],
            network=TinyMLP(
                np.asarray(data["weight_1"], dtype=np.float64),
                np.asarray(data["bias_1"], dtype=np.float64),
                np.asarray(data["weight_2"], dtype=np.float64),
                np.asarray(data["bias_2"], dtype=np.float64),
            ),
            threshold=float(data["threshold"][0]),
            anchor_prompt=str(data["anchor_prompt"][0]),
            uses_anchor=bool(data["uses_anchor"][0]),
            format_version=(
                str(data["format_version"][0])
                if "format_version" in data
                else "CISR_v2_detector_v1"
            ),
            calibration_target_tpr=(
                float(data["calibration_target_tpr"][0])
                if "calibration_target_tpr" in data
                else None
            ),
            calibration_target_fpr=(
                float(data["calibration_target_fpr"][0])
                if "calibration_target_fpr" in data
                else None
            ),
            coverage_confidence=(
                float(data["coverage_confidence"][0])
                if "coverage_confidence" in data
                else None
            ),
        )

    def score_hidden(
        self,
        hidden: np.ndarray,
        image_role: str,
        anchor_hidden: np.ndarray | None = None,
        threshold_override: float | None = None,
    ) -> dict[str, Any]:
        vector = np.asarray(hidden, dtype=np.float64).reshape(-1)
        if vector.shape != self.center.shape:
            raise ValueError(
                f"CISR hidden dimension mismatch: got {vector.shape}, expected {self.center.shape}"
            )
        raw_coords = (vector - self.center) @ self.basis.T
        has_anchor = anchor_hidden is not None
        if has_anchor:
            anchor = np.asarray(anchor_hidden, dtype=np.float64).reshape(-1)
            if anchor.shape != vector.shape:
                raise ValueError(
                    f"CISR anchor dimension mismatch: got {anchor.shape}, expected {vector.shape}"
                )
            residual_coords = (vector - anchor - self.residual_center) @ self.basis.T
        else:
            residual_coords = np.zeros(self.rank, dtype=np.float64)
        features = build_detector_features(
            raw_coords.reshape(1, -1),
            residual_coords.reshape(1, -1),
            np.array([has_anchor], dtype=bool),
            np.array([image_role or "none"]),
            self.role_categories,
        )
        standardized = (features - self.feature_mean[None, :]) / self.feature_std[None, :]
        probability = float(self.network.predict_proba(standardized)[0])
        threshold = self.threshold if threshold_override is None else float(threshold_override)
        return {
            "probability": probability,
            "threshold": threshold,
            "detected": probability >= threshold,
            "coordinates": raw_coords.astype(float).tolist(),
            "residual_coordinates": residual_coords.astype(float).tolist(),
            "has_anchor": has_anchor,
            "image_role": image_role or "none",
            "layer": self.layer,
            "rank": self.rank,
        }

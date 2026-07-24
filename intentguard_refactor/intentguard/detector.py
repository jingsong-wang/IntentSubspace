from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .selective import SelectiveThresholds, decide_route


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
    feature_mode: str = "v2_full",
) -> np.ndarray:
    raw = np.asarray(raw_coords, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"raw_coords must be two-dimensional, got shape={raw.shape}")
    if feature_mode == "raw_rank3":
        return raw
    if feature_mode != "v2_full":
        raise ValueError(f"Unsupported CISR detector feature mode: {feature_mode}")
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
    consistency_groups: np.ndarray | None = None,
    consistency_weight: float = 0.0,
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
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight must be non-negative.")

    consistency_indices: list[np.ndarray] = []
    if consistency_groups is not None and consistency_weight > 0.0:
        groups = np.asarray(consistency_groups).astype(str).reshape(-1)
        if len(groups) != len(y):
            raise ValueError("consistency_groups must align with training rows.")
        for group in sorted(set(groups.tolist())):
            if not group:
                continue
            indices = np.where(groups == group)[0]
            if len(indices) < 2:
                continue
            if len(np.unique(y[indices])) != 1:
                raise ValueError(f"Consistency group {group!r} mixes target and benign labels.")
            consistency_indices.append(indices)

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
        consistency_loss = 0.0
        consistency_gradient = np.zeros_like(logits)
        if consistency_indices:
            group_normalizer = float(len(consistency_indices))
            for indices in consistency_indices:
                centered = logits[indices] - float(logits[indices].mean())
                consistency_loss += float(np.mean(centered**2)) / group_normalizer
                consistency_gradient[indices] += (
                    2.0 * consistency_weight * centered / (group_normalizer * len(indices))
                )
        loss = float(
            data_loss
            + consistency_weight * consistency_loss
            + 0.5 * l2 * (np.sum(weight_1**2) + np.sum(weight_2**2))
        )

        grad_logits = (
            weights * (probabilities - y) / normalizer + consistency_gradient
        ).reshape(-1, 1)
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
        "consistency_weight": consistency_weight,
        "consistency_group_count": len(consistency_indices),
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
    feature_mode: str = "v2_full"
    deployment_constraints_met: bool | None = None
    hard_benign_target_fpr: float | None = None
    safe_threshold: float | None = None
    danger_threshold: float | None = None
    safe_route_enabled: bool | None = None
    danger_route_enabled: bool | None = None
    confident_safe_error_upper_bound: float | None = None
    confident_dangerous_error_upper_bound: float | None = None
    maximum_confident_safe_error: float | None = None
    maximum_confident_dangerous_error: float | None = None
    maximum_harmful_unsafe_escape: float | None = None
    maximum_benign_hard_refusal: float | None = None

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
            role_categories=(
                [str(value) for value in data["role_categories"].tolist()]
                if "role_categories" in data
                else []
            ),
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
            feature_mode=(
                str(data["feature_mode"][0]) if "feature_mode" in data else "v2_full"
            ),
            deployment_constraints_met=(
                bool(data["deployment_constraints_met"][0])
                if "deployment_constraints_met" in data
                else None
            ),
            hard_benign_target_fpr=(
                float(data["hard_benign_target_fpr"][0])
                if "hard_benign_target_fpr" in data
                else None
            ),
            safe_threshold=(
                float(data["safe_threshold"][0]) if "safe_threshold" in data else None
            ),
            danger_threshold=(
                float(data["danger_threshold"][0])
                if "danger_threshold" in data
                else None
            ),
            safe_route_enabled=(
                bool(data["safe_route_enabled"][0])
                if "safe_route_enabled" in data
                else None
            ),
            danger_route_enabled=(
                bool(data["danger_route_enabled"][0])
                if "danger_route_enabled" in data
                else None
            ),
            confident_safe_error_upper_bound=(
                float(data["confident_safe_error_upper_bound"][0])
                if "confident_safe_error_upper_bound" in data
                else None
            ),
            confident_dangerous_error_upper_bound=(
                float(data["confident_dangerous_error_upper_bound"][0])
                if "confident_dangerous_error_upper_bound" in data
                else None
            ),
            maximum_confident_safe_error=(
                float(data["maximum_confident_safe_error"][0])
                if "maximum_confident_safe_error" in data
                else None
            ),
            maximum_confident_dangerous_error=(
                float(data["maximum_confident_dangerous_error"][0])
                if "maximum_confident_dangerous_error" in data
                else None
            ),
            maximum_harmful_unsafe_escape=(
                float(data["maximum_harmful_unsafe_escape"][0])
                if "maximum_harmful_unsafe_escape" in data
                else None
            ),
            maximum_benign_hard_refusal=(
                float(data["maximum_benign_hard_refusal"][0])
                if "maximum_benign_hard_refusal" in data
                else None
            ),
        )

    def score_hidden(
        self,
        hidden: np.ndarray,
        image_role: str,
        anchor_hidden: np.ndarray | None = None,
        threshold_override: float | None = None,
        safe_threshold_override: float | None = None,
        danger_threshold_override: float | None = None,
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
            feature_mode=self.feature_mode,
        )
        standardized = (features - self.feature_mean[None, :]) / self.feature_std[None, :]
        probability = float(self.network.predict_proba(standardized)[0])
        threshold = self.threshold if threshold_override is None else float(threshold_override)
        result = {
            "probability": probability,
            "threshold": threshold,
            "detected": probability >= threshold,
            "coordinates": raw_coords.astype(float).tolist(),
            "residual_coordinates": (
                residual_coords.astype(float).tolist() if self.feature_mode == "v2_full" else []
            ),
            "has_anchor": has_anchor,
            "image_role": image_role or "none",
            "layer": self.layer,
            "rank": self.rank,
            "feature_mode": self.feature_mode,
            "deployment_constraints_met": self.deployment_constraints_met,
        }
        if self.safe_threshold is not None and self.danger_threshold is not None:
            safe_threshold = (
                self.safe_threshold
                if safe_threshold_override is None
                else float(safe_threshold_override)
            )
            danger_threshold = (
                self.danger_threshold
                if danger_threshold_override is None
                else float(danger_threshold_override)
            )
            thresholds = SelectiveThresholds(
                safe_max=safe_threshold,
                danger_min=danger_threshold,
                safe_enabled=self.safe_route_enabled is not False,
                danger_enabled=self.danger_route_enabled is not False,
            )
            decision = decide_route(
                probability,
                thresholds,
                safe_error_upper_bound=self.confident_safe_error_upper_bound,
                danger_error_upper_bound=self.confident_dangerous_error_upper_bound,
            )
            result.update(decision.to_dict())
            result.update(
                {
                    "safe_threshold": safe_threshold,
                    "danger_threshold": danger_threshold,
                    "safe_route_enabled": thresholds.safe_enabled,
                    "danger_route_enabled": thresholds.danger_enabled,
                    "detected": decision.route.value == "confident_dangerous",
                }
            )
        return result


@dataclass
class CISRDetectorBundle:
    path: Path
    model_id: str
    model_alias: str
    text_detector: CISRDetector
    multimodal_detector: CISRDetector
    format_version: str = "CISR_v4_detector_bundle_v1"

    @classmethod
    def load(cls, path: Path | str) -> "CISRDetectorBundle":
        manifest_path = Path(path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_version = str(manifest.get("format_version", ""))
        if not format_version.lower().startswith("cisr_v4_detector_bundle"):
            raise ValueError(f"Unsupported CISR detector bundle format: {format_version!r}")
        branches = manifest.get("branches")
        if not isinstance(branches, dict):
            raise ValueError("CISR detector bundle is missing branches.")

        def child(name: str) -> CISRDetector:
            entry = branches.get(name)
            if not isinstance(entry, dict) or not entry.get("detector"):
                raise ValueError(f"CISR detector bundle is missing {name!r} detector.")
            child_path = Path(str(entry["detector"]))
            if not child_path.is_absolute():
                child_path = manifest_path.parent / child_path
            detector = CISRDetector.load(child_path)
            if not detector.format_version.lower().startswith("cisr_v4_detector"):
                raise ValueError(
                    f"CISR bundle child {name!r} must be v4, got {detector.format_version!r}."
                )
            expected_pooling = str(entry.get("pooling") or detector.pooling)
            if detector.pooling != expected_pooling:
                raise ValueError(
                    f"CISR bundle {name!r} pooling mismatch: manifest={expected_pooling!r}, "
                    f"detector={detector.pooling!r}."
                )
            return detector

        text_detector = child("text")
        multimodal_detector = child("multimodal")
        model_ids = {value for value in (text_detector.model_id, multimodal_detector.model_id) if value}
        manifest_model = str(manifest.get("model_id", ""))
        if len(model_ids) > 1 or (manifest_model and model_ids and manifest_model not in model_ids):
            raise ValueError(
                f"CISR bundle mixes model ids: manifest={manifest_model!r}, children={sorted(model_ids)!r}."
            )
        return cls(
            path=manifest_path,
            model_id=manifest_model or (next(iter(model_ids)) if model_ids else ""),
            model_alias=str(manifest.get("model_alias", "")),
            text_detector=text_detector,
            multimodal_detector=multimodal_detector,
            format_version=format_version,
        )

    def select(self, has_image: bool) -> tuple[str, CISRDetector]:
        if has_image:
            return "multimodal", self.multimodal_detector
        return "text", self.text_detector

    @property
    def deployment_constraints_met(self) -> bool:
        return all(
            detector.deployment_constraints_met is not False
            for detector in (self.text_detector, self.multimodal_detector)
        )

    @property
    def coverage_confidence(self) -> float | None:
        values = [
            detector.coverage_confidence
            for detector in (self.text_detector, self.multimodal_detector)
            if detector.coverage_confidence is not None
        ]
        return min(values) if values else None

    @property
    def layer(self) -> int:
        layers = {self.text_detector.layer, self.multimodal_detector.layer}
        if len(layers) != 1:
            raise ValueError(
                "A modal CISR bundle uses different text/multimodal layers and cannot "
                "share one safe-layer adapter. Use --cisr4-review-action monitor for detection audits."
            )
        return next(iter(layers))


def load_cisr_detector(path: Path | str) -> CISRDetector | CISRDetectorBundle:
    artifact_path = Path(path).expanduser().resolve()
    if artifact_path.suffix.lower() == ".json":
        return CISRDetectorBundle.load(artifact_path)
    return CISRDetector.load(artifact_path)

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import BaseModelRunner, HiddenRepresentation


REPRESENTATION_ARTIFACT_FORMAT = "vlm_representation_detector_v1"
TRAINABLE_REPRESENTATION_CHOICES = [
    "nearside",
    "rcs-kcd",
    "rcs-mcd",
    "vlmguard",
]
REPRESENTATION_DEFENSE_CHOICES = [*TRAINABLE_REPRESENTATION_CHOICES, "cnrf-oracle"]
REPRESENTATION_HARD_REFUSAL = (
    "I cannot help with instructions that facilitate harmful or illegal activity. "
    "I can help with prevention, safety, reporting, or other non-operational alternatives."
)

_METHOD_ALIASES = {
    "near-side": "nearside",
    "near_side": "nearside",
    "kcd": "rcs-kcd",
    "rcs_kcd": "rcs-kcd",
    "mcd": "rcs-mcd",
    "rcs_mcd": "rcs-mcd",
    "vlm_guard": "vlmguard",
    "vlm-guard": "vlmguard",
    "cnrf_oracle": "cnrf-oracle",
    "cnrforacle": "cnrf-oracle",
}


def normalize_representation_method(value: str) -> str:
    method = value.lower().strip()
    method = _METHOD_ALIASES.get(method, method)
    if method not in REPRESENTATION_DEFENSE_CHOICES:
        raise ValueError(
            f"Unsupported representation detector {value!r}; expected one of "
            f"{REPRESENTATION_DEFENSE_CHOICES}."
        )
    return method


def is_representation_defense(value: str) -> bool:
    try:
        normalize_representation_method(value)
    except ValueError:
        return False
    return True


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact_fingerprint(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    core = dict(metadata)
    core.pop("artifact_fingerprint", None)
    digest = hashlib.sha1(_canonical_json(core))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_representation_artifact(
    path: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> "RepresentationDetector":
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_arrays = {
        str(name): np.asarray(value)
        for name, value in arrays.items()
    }
    payload = dict(metadata)
    payload["format_version"] = REPRESENTATION_ARTIFACT_FORMAT
    payload["method"] = normalize_representation_method(str(payload["method"]))
    payload["artifact_fingerprint"] = _artifact_fingerprint(payload, normalized_arrays)
    np.savez_compressed(
        output,
        metadata_json=np.array(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        **normalized_arrays,
    )
    return RepresentationDetector.load(output)


def _one_dimensional(vector: Any, expected_dim: int | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(f"Expected one hidden-state vector, got shape {value.shape}")
    if expected_dim is not None and value.shape[0] != expected_dim:
        raise ValueError(
            f"Detector expects hidden dimension {expected_dim}, got {value.shape[0]}"
        )
    if not np.isfinite(value).all():
        raise ValueError("Hidden-state vector contains non-finite values")
    return value


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


@dataclass(frozen=True)
class _CNRFOracleScores:
    scores: np.ndarray
    nearest_midpoint_distance: np.ndarray
    neighbor_pair_ids: list[list[str]]
    neighbor_pack_ids: list[list[str]]
    neighbor_coordinates: list[list[float]]


class _CNRFOracleField:
    """Exact online form of Oracle ViewPairCache -> PackViewData scoring."""

    def __init__(
        self,
        benign: np.ndarray,
        harmful: np.ndarray,
        pair_ids: np.ndarray,
        pack_ids: np.ndarray,
        *,
        k: int,
        min_arrow_norm: float,
        score_clip: float | None,
    ) -> None:
        benign_values = np.asarray(benign, dtype=np.float64)
        harmful_values = np.asarray(harmful, dtype=np.float64)
        benign_unit = np.stack([_l2_normalize(value) for value in benign_values])
        harmful_unit = np.stack([_l2_normalize(value) for value in harmful_values])
        arrows = harmful_unit - benign_unit
        arrow_norm_squared = np.sum(arrows * arrows, axis=1)
        midpoints = 0.5 * (benign_unit + harmful_unit)
        midpoint_norms = np.linalg.norm(midpoints, axis=1)
        keep = (
            (np.sqrt(arrow_norm_squared) >= float(min_arrow_norm))
            & (midpoint_norms > 1e-12)
        )
        self.arrows = arrows[keep]
        self.arrow_norm_squared = arrow_norm_squared[keep]
        self.midpoints = midpoints[keep]
        self.midpoint_unit = np.stack(
            [_l2_normalize(value) for value in self.midpoints]
        )
        self.pair_ids = np.asarray(pair_ids).astype(str)[keep]
        self.pack_ids = np.asarray(pack_ids).astype(str)[keep]
        self.pack_values = np.asarray(sorted(set(self.pack_ids.tolist()))).astype(str)
        self.pack_pair_positions = [
            np.flatnonzero(self.pack_ids == pack) for pack in self.pack_values.tolist()
        ]
        self.k = int(k)
        self.score_clip = score_clip
        if len(self.pack_values) <= self.k:
            raise ValueError(
                f"CNRF Oracle field needs more than k={self.k} packs; got {len(self.pack_values)}"
            )

    @property
    def hidden_dim(self) -> int:
        return int(self.arrows.shape[1])

    def score(self, vectors: Any) -> _CNRFOracleScores:
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.hidden_dim:
            raise ValueError(f"Expected [sample, {self.hidden_dim}] vectors, got {values.shape}")
        normalized = np.stack([_l2_normalize(value) for value in values])
        pair_distances = 1.0 - normalized @ self.midpoint_unit.T
        pair_coordinates = 2.0 * (normalized @ self.arrows.T)
        offsets = 2.0 * np.sum(self.midpoints * self.arrows, axis=1)
        pair_coordinates = (
            pair_coordinates - offsets[None, :]
        ) / np.maximum(self.arrow_norm_squared[None, :], 1e-12)
        if self.score_clip is not None:
            pair_coordinates = np.clip(
                pair_coordinates, -float(self.score_clip), float(self.score_clip)
            )

        distances = np.empty((len(values), len(self.pack_values)), dtype=np.float64)
        coordinates = np.empty_like(distances)
        choices = np.empty(distances.shape, dtype=np.int64)
        row_positions = np.arange(len(values), dtype=np.int64)
        for pack_position, pair_positions in enumerate(self.pack_pair_positions):
            current = pair_distances[:, pair_positions]
            nearest = np.argmin(current, axis=1)
            chosen = pair_positions[nearest]
            distances[:, pack_position] = current[row_positions, nearest]
            coordinates[:, pack_position] = pair_coordinates[row_positions, chosen]
            choices[:, pack_position] = chosen

        neighbor_positions = np.argpartition(
            distances, kth=self.k - 1, axis=1
        )[:, : self.k]
        chosen_distances = np.take_along_axis(distances, neighbor_positions, axis=1)
        chosen_coordinates = np.take_along_axis(coordinates, neighbor_positions, axis=1)
        chosen_pairs = np.take_along_axis(choices, neighbor_positions, axis=1)
        return _CNRFOracleScores(
            scores=np.median(chosen_coordinates, axis=1),
            nearest_midpoint_distance=np.min(chosen_distances, axis=1),
            neighbor_pair_ids=[
                self.pair_ids[row].astype(str).tolist() for row in chosen_pairs
            ],
            neighbor_pack_ids=[
                self.pack_values[row].astype(str).tolist() for row in neighbor_positions
            ],
            neighbor_coordinates=[row.astype(float).tolist() for row in chosen_coordinates],
        )


def _batch_norm(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    return (
        (value - running_mean)
        / np.sqrt(np.maximum(running_var, 0.0) + epsilon)
        * weight
        + bias
    )


@dataclass
class RepresentationDetector:
    path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    _runtime_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def load(cls, path: Path) -> "RepresentationDetector":
        artifact_path = path.expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Representation detector does not exist: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as archive:
            if "metadata_json" not in archive:
                raise ValueError(
                    f"Representation detector is missing metadata_json: {artifact_path}"
                )
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("format_version") != REPRESENTATION_ARTIFACT_FORMAT:
                raise ValueError(
                    f"Unsupported representation artifact format {metadata.get('format_version')!r}; "
                    f"expected {REPRESENTATION_ARTIFACT_FORMAT!r}."
                )
            metadata["method"] = normalize_representation_method(
                str(metadata.get("method", ""))
            )
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "metadata_json"
            }
        expected = str(metadata.get("artifact_fingerprint") or "")
        actual = _artifact_fingerprint(metadata, arrays)
        if not expected or expected != actual:
            raise ValueError(
                f"Representation detector fingerprint mismatch for {artifact_path}; "
                "the artifact is incomplete or has been modified."
            )
        detector = cls(path=artifact_path, metadata=metadata, arrays=arrays)
        detector._validate()
        return detector

    @property
    def method(self) -> str:
        return str(self.metadata["method"])

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("model_id") or "")

    @property
    def model_revision(self) -> str | None:
        value = self.metadata.get("model_revision")
        return None if value in {None, ""} else str(value)

    @property
    def layer(self) -> int:
        return int(self.metadata["layer"])

    @property
    def pooling(self) -> str:
        return str(self.metadata.get("pooling") or "last")

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def fingerprint(self) -> str:
        return str(self.metadata["artifact_fingerprint"])

    @property
    def hidden_dim(self) -> int:
        return int(self.metadata["hidden_dim"])

    @property
    def paper_training_protocol(self) -> bool:
        return bool(self.metadata.get("paper_training_protocol", False))

    @property
    def core_algorithm_compatible(self) -> bool:
        return bool(self.metadata.get("core_algorithm_compatible", False))

    def _validate(self) -> None:
        if self.method == "cnrf-oracle":
            self._validate_cnrf_oracle()
            return
        if self.pooling != "last":
            raise ValueError(
                f"{self.method} artifact uses unsupported pooling {self.pooling!r}; "
                "these reproductions require last-token representations."
            )
        if self.hidden_dim <= 0 or self.layer < 0 or not np.isfinite(self.threshold):
            raise ValueError("Representation detector metadata contains invalid dimensions or threshold")
        if self.method == "nearside":
            direction = self.arrays.get("direction")
            if direction is None or direction.shape != (self.hidden_dim,):
                raise ValueError("NEARSIDE artifact has an invalid attack direction")
            if float(np.linalg.norm(direction)) <= 1e-12:
                raise ValueError("NEARSIDE attack direction is zero")
            return
        if self.method == "vlmguard":
            required = []
            for index in range(3):
                required.extend(
                    [
                        f"classifier_linear_{index}_weight",
                        f"classifier_linear_{index}_bias",
                    ]
                )
            missing = [name for name in required if name not in self.arrays]
            if missing:
                raise ValueError(f"VLMGuard artifact is missing classifier arrays: {missing}")
            for index in range(3):
                weight = self.arrays[f"classifier_linear_{index}_weight"]
                bias = self.arrays[f"classifier_linear_{index}_bias"]
                if weight.ndim != 2 or bias.shape != (weight.shape[0],):
                    raise ValueError(f"VLMGuard classifier layer {index} has invalid dimensions")
                if index and weight.shape[1] != self.arrays[
                    f"classifier_linear_{index - 1}_weight"
                ].shape[0]:
                    raise ValueError(f"VLMGuard classifier layer {index} is disconnected")
            if self.arrays["classifier_linear_0_weight"].shape[1] != self.hidden_dim:
                raise ValueError("VLMGuard classifier input dimension does not match hidden_dim")
            if self.arrays["classifier_linear_2_weight"].shape[0] != 1:
                raise ValueError("VLMGuard classifier must emit one maliciousness logit")
            if not 0.0 <= self.threshold <= 1.0:
                raise ValueError("VLMGuard probability threshold must be in [0, 1]")
            svd_names = ("svd_center", "svd_components", "svd_singular_values")
            present = [name in self.arrays for name in svd_names]
            if any(present) and not all(present):
                raise ValueError("VLMGuard artifact contains an incomplete SVD diagnostic")
            if all(present):
                center = self.arrays["svd_center"]
                components = self.arrays["svd_components"]
                singular_values = self.arrays["svd_singular_values"]
                if center.shape != (self.hidden_dim,):
                    raise ValueError("VLMGuard SVD center has an invalid dimension")
                if components.ndim != 2 or components.shape[1] != self.hidden_dim:
                    raise ValueError("VLMGuard SVD components have invalid dimensions")
                if singular_values.shape != (len(components),):
                    raise ValueError("VLMGuard SVD singular values do not match components")
            return
        required_projection = []
        for index in range(3):
            required_projection.extend(
                [
                    f"proj_linear_{index}_weight",
                    f"proj_linear_{index}_bias",
                    f"proj_bn_{index}_weight",
                    f"proj_bn_{index}_bias",
                    f"proj_bn_{index}_running_mean",
                    f"proj_bn_{index}_running_var",
                ]
            )
        missing = [name for name in required_projection if name not in self.arrays]
        if missing:
            raise ValueError(f"RCS artifact is missing projection arrays: {missing}")
        if self.arrays["proj_linear_0_weight"].shape[1] != self.hidden_dim:
            raise ValueError("RCS projection input dimension does not match artifact hidden_dim")
        projected_dim = int(self.arrays["proj_linear_2_weight"].shape[0])
        if self.method == "rcs-kcd":
            for name in ("benign_reference", "malicious_reference"):
                reference = self.arrays.get(name)
                if reference is None or reference.ndim != 2 or reference.shape[1] != projected_dim:
                    raise ValueError(f"RCS-KCD artifact has invalid {name}")
                if len(reference) == 0:
                    raise ValueError(f"RCS-KCD artifact has empty {name}")
        else:
            for prefix in ("benign", "malicious"):
                means = self.arrays.get(f"{prefix}_means")
                precisions = self.arrays.get(f"{prefix}_precisions")
                if means is None or means.ndim != 2 or means.shape[1] != projected_dim:
                    raise ValueError(f"RCS-MCD artifact has invalid {prefix} means")
                if (
                    precisions is None
                    or precisions.ndim != 3
                    or precisions.shape != (len(means), projected_dim, projected_dim)
                ):
                    raise ValueError(f"RCS-MCD artifact has invalid {prefix} precision matrices")

    def _validate_cnrf_oracle(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("CNRF Oracle artifact has an invalid hidden_dim")
        if not bool(self.metadata.get("oracle_only", False)):
            raise ValueError("CNRF Oracle artifact must declare oracle_only=true")
        if self.metadata.get("fusion_policy") != "supported_max":
            raise ValueError("CNRF Oracle online deployment requires fusion_policy=supported_max")
        if self.metadata.get("support_policy") != "abstain_safe":
            raise ValueError("CNRF Oracle online deployment requires support_policy=abstain_safe")
        k = int(self.metadata.get("k", 0))
        if k < 1:
            raise ValueError("CNRF Oracle artifact has an invalid k")
        branches = self.metadata.get("branches")
        if not isinstance(branches, dict) or not branches:
            raise ValueError("CNRF Oracle artifact contains no modality branches")
        for branch, branch_config in branches.items():
            if branch not in {"text", "image_text"} or not isinstance(branch_config, dict):
                raise ValueError(f"CNRF Oracle artifact has invalid branch {branch!r}")
            threshold = float(branch_config.get("threshold", math.nan))
            if not np.isfinite(threshold):
                raise ValueError(f"CNRF Oracle branch {branch!r} has an invalid threshold")
            views = branch_config.get("views")
            if not isinstance(views, list) or not views:
                raise ValueError(f"CNRF Oracle branch {branch!r} contains no views")
            view_keys: set[tuple[int, str]] = set()
            for view in views:
                if not isinstance(view, dict):
                    raise ValueError(f"CNRF Oracle branch {branch!r} has malformed view metadata")
                key = (int(view.get("layer", -1)), str(view.get("readout") or ""))
                if key[0] < 0 or not key[1] or key in view_keys:
                    raise ValueError(f"CNRF Oracle branch {branch!r} has an invalid/duplicate view {key}")
                view_keys.add(key)
                center = float(view.get("center", math.nan))
                scale = float(view.get("scale", math.nan))
                support_radius = float(view.get("support_radius", math.nan))
                if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
                    raise ValueError(f"CNRF Oracle view {branch}/{key} has invalid robust scaling")
                if not np.isfinite(support_radius) or support_radius < 0.0:
                    raise ValueError(f"CNRF Oracle view {branch}/{key} has invalid support radius")
                prefix = str(view.get("array_prefix") or "")
                required = [
                    f"{prefix}_benign",
                    f"{prefix}_harmful",
                    f"{prefix}_pair_ids",
                    f"{prefix}_pack_ids",
                ]
                missing = [name for name in required if name not in self.arrays]
                if not prefix or missing:
                    raise ValueError(
                        f"CNRF Oracle view {branch}/{key} is missing endpoint arrays: {missing}"
                    )
                benign = self.arrays[f"{prefix}_benign"]
                harmful = self.arrays[f"{prefix}_harmful"]
                pair_ids = self.arrays[f"{prefix}_pair_ids"]
                pack_ids = self.arrays[f"{prefix}_pack_ids"]
                if (
                    benign.ndim != 2
                    or benign.shape[1] != self.hidden_dim
                    or harmful.shape != benign.shape
                    or pair_ids.shape != (len(benign),)
                    or pack_ids.shape != (len(benign),)
                ):
                    raise ValueError(f"CNRF Oracle view {branch}/{key} has misaligned endpoints")
                if len(benign) < 2 or len(set(pack_ids.astype(str).tolist())) <= k:
                    raise ValueError(
                        f"CNRF Oracle view {branch}/{key} needs more than k={k} reference packs"
                    )
                if len(set(pair_ids.astype(str).tolist())) != len(pair_ids):
                    raise ValueError(f"CNRF Oracle view {branch}/{key} contains duplicate pair ids")
                if not np.isfinite(benign).all() or not np.isfinite(harmful).all():
                    raise ValueError(f"CNRF Oracle view {branch}/{key} contains non-finite endpoints")

    def cnrf_branch_config(self, branch: str) -> dict[str, Any]:
        if self.method != "cnrf-oracle":
            raise ValueError("CNRF branch metadata is available only for cnrf-oracle")
        branches = self.metadata["branches"]
        if branch not in branches:
            raise ValueError(
                f"CNRF Oracle artifact has no {branch!r} branch; available={sorted(branches)}"
            )
        return branches[branch]

    def cnrf_field(self, branch: str, view_index: int) -> Any:
        """Build and cache one local counterfactual risk field."""

        branch_config = self.cnrf_branch_config(branch)
        views = branch_config["views"]
        if view_index < 0 or view_index >= len(views):
            raise IndexError(f"CNRF view index out of range: {branch}/{view_index}")
        cache_key = f"field:{branch}:{view_index}"
        cached = self._runtime_cache.get(cache_key)
        if cached is not None:
            return cached
        view = views[view_index]
        prefix = str(view["array_prefix"])
        risk_field = _CNRFOracleField(
            self.arrays[f"{prefix}_benign"],
            self.arrays[f"{prefix}_harmful"],
            self.arrays[f"{prefix}_pair_ids"].astype(str),
            self.arrays[f"{prefix}_pack_ids"].astype(str),
            k=int(self.metadata["k"]),
            min_arrow_norm=float(self.metadata.get("min_arrow_norm", 1e-3)),
            score_clip=(
                None
                if self.metadata.get("score_clip") is None
                else float(self.metadata["score_clip"])
            ),
        )
        self._runtime_cache[cache_key] = risk_field
        return risk_field

    def project(self, vector: Any) -> np.ndarray:
        value = _one_dimensional(vector, expected_dim=self.hidden_dim)
        if self.method in {"nearside", "vlmguard"}:
            return value
        epsilon = float(self.metadata.get("batch_norm_epsilon", 1e-5))
        for index in range(3):
            value = (
                self.arrays[f"proj_linear_{index}_weight"].astype(np.float64) @ value
                + self.arrays[f"proj_linear_{index}_bias"].astype(np.float64)
            )
            value = _batch_norm(
                value,
                self.arrays[f"proj_bn_{index}_weight"].astype(np.float64),
                self.arrays[f"proj_bn_{index}_bias"].astype(np.float64),
                self.arrays[f"proj_bn_{index}_running_mean"].astype(np.float64),
                self.arrays[f"proj_bn_{index}_running_var"].astype(np.float64),
                epsilon,
            )
            if index < 2:
                value = np.maximum(value, 0.0)
        if not np.isfinite(value).all():
            raise ValueError("Projected representation contains non-finite values")
        return value

    def project_batch(self, vectors: Any) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Detector expects a [sample, {self.hidden_dim}] activation matrix, "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Activation matrix contains non-finite values")
        if self.method in {"nearside", "vlmguard"}:
            return values
        epsilon = float(self.metadata.get("batch_norm_epsilon", 1e-5))
        for index in range(3):
            values = (
                values @ self.arrays[f"proj_linear_{index}_weight"].astype(np.float64).T
                + self.arrays[f"proj_linear_{index}_bias"].astype(np.float64)[None, :]
            )
            values = _batch_norm(
                values,
                self.arrays[f"proj_bn_{index}_weight"].astype(np.float64)[None, :],
                self.arrays[f"proj_bn_{index}_bias"].astype(np.float64)[None, :],
                self.arrays[f"proj_bn_{index}_running_mean"].astype(np.float64)[None, :],
                self.arrays[f"proj_bn_{index}_running_var"].astype(np.float64)[None, :],
                epsilon,
            )
            if index < 2:
                values = np.maximum(values, 0.0)
        if not np.isfinite(values).all():
            raise ValueError("Projected activation matrix contains non-finite values")
        return values

    def _score_nearside(self, vector: np.ndarray) -> tuple[float, dict[str, Any]]:
        direction = _l2_normalize(self.arrays["direction"].astype(np.float64))
        score = float(np.dot(vector, direction))
        return score, {
            "projection_norm": float(np.linalg.norm(vector)),
            "direction_norm": float(np.linalg.norm(self.arrays["direction"])),
        }

    def _score_kcd(self, projected: np.ndarray) -> tuple[float, dict[str, Any]]:
        value = _l2_normalize(projected)
        benign = self.arrays["benign_reference"].astype(np.float64)
        malicious = self.arrays["malicious_reference"].astype(np.float64)
        benign_distances = np.linalg.norm(benign - value[None, :], axis=1)
        malicious_distances = np.linalg.norm(malicious - value[None, :], axis=1)
        requested_k = int(self.metadata.get("k", 50))
        benign_k = min(max(requested_k, 1), len(benign_distances))
        malicious_k = min(max(requested_k, 1), len(malicious_distances))
        benign_distance = float(np.partition(benign_distances, benign_k - 1)[benign_k - 1])
        malicious_distance = float(
            np.partition(malicious_distances, malicious_k - 1)[malicious_k - 1]
        )
        return benign_distance - malicious_distance, {
            "distance_to_benign": benign_distance,
            "distance_to_malicious": malicious_distance,
            "effective_k_benign": benign_k,
            "effective_k_malicious": malicious_k,
            "projected_norm": float(np.linalg.norm(projected)),
        }

    @staticmethod
    def _minimum_mahalanobis(
        value: np.ndarray,
        means: np.ndarray,
        precisions: np.ndarray,
    ) -> tuple[float, int]:
        differences = value[None, :] - means.astype(np.float64)
        squared = np.einsum(
            "bi,bij,bj->b",
            differences,
            precisions.astype(np.float64),
            differences,
            optimize=True,
        )
        distances = np.sqrt(np.maximum(squared, 0.0))
        index = int(np.argmin(distances))
        return float(distances[index]), index

    def _score_mcd(self, projected: np.ndarray) -> tuple[float, dict[str, Any]]:
        benign_distance, benign_index = self._minimum_mahalanobis(
            projected,
            self.arrays["benign_means"],
            self.arrays["benign_precisions"],
        )
        malicious_distance, malicious_index = self._minimum_mahalanobis(
            projected,
            self.arrays["malicious_means"],
            self.arrays["malicious_precisions"],
        )
        score = float(np.clip(benign_distance - malicious_distance, -10000.0, 10000.0))
        benign_names = list(self.metadata.get("benign_cluster_names") or [])
        malicious_names = list(self.metadata.get("malicious_cluster_names") or [])
        return score, {
            "distance_to_benign": benign_distance,
            "distance_to_malicious": malicious_distance,
            "nearest_benign_cluster": (
                benign_names[benign_index] if benign_index < len(benign_names) else benign_index
            ),
            "nearest_malicious_cluster": (
                malicious_names[malicious_index]
                if malicious_index < len(malicious_names)
                else malicious_index
            ),
            "projected_norm": float(np.linalg.norm(projected)),
        }

    def _vlmguard_probability(self, vector: np.ndarray) -> tuple[float, dict[str, Any]]:
        value = vector.astype(np.float64)
        for index in range(3):
            value = (
                self.arrays[f"classifier_linear_{index}_weight"].astype(np.float64) @ value
                + self.arrays[f"classifier_linear_{index}_bias"].astype(np.float64)
            )
            if index < 2:
                value = np.maximum(value, 0.0)
        logit = float(value.reshape(-1)[0])
        probability = (
            1.0 / (1.0 + math.exp(-logit))
            if logit >= 0.0
            else math.exp(logit) / (1.0 + math.exp(logit))
        )
        details = {
            "classifier_logit": logit,
            "hidden_norm": float(np.linalg.norm(vector)),
        }
        if all(
            name in self.arrays
            for name in ("svd_center", "svd_components", "svd_singular_values")
        ):
            centered = vector - self.arrays["svd_center"].astype(np.float64)
            components = self.arrays["svd_components"].astype(np.float64)
            singular_values = self.arrays["svd_singular_values"].astype(np.float64)
            projections = components @ centered
            details["direct_projection_score"] = float(
                np.mean(singular_values * projections**2)
            )
        return probability, details

    def vlmguard_direct_scores(self, vectors: Any) -> np.ndarray:
        if self.method != "vlmguard":
            raise ValueError("Direct SVD projection scores are available only for VLMGuard")
        required = ("svd_center", "svd_components", "svd_singular_values")
        if not all(name in self.arrays for name in required):
            raise ValueError("VLMGuard artifact does not contain SVD diagnostics")
        values = self.project_batch(vectors)
        centered = values - self.arrays["svd_center"].astype(np.float64)[None, :]
        projections = centered @ self.arrays["svd_components"].astype(np.float64).T
        return np.mean(
            projections**2
            * self.arrays["svd_singular_values"].astype(np.float64)[None, :],
            axis=1,
        )

    def score_vector(
        self,
        vector: Any,
        threshold_override: float | None = None,
    ) -> dict[str, Any]:
        if self.method == "cnrf-oracle":
            raise ValueError("cnrf-oracle requires score_cnrf_views with all frozen views")
        hidden = _one_dimensional(vector, expected_dim=self.hidden_dim)
        projected = self.project(hidden)
        if self.method == "nearside":
            score, details = self._score_nearside(projected)
        elif self.method == "vlmguard":
            score, details = self._vlmguard_probability(projected)
        elif self.method == "rcs-kcd":
            score, details = self._score_kcd(projected)
        else:
            score, details = self._score_mcd(projected)
        threshold = self.threshold if threshold_override is None else float(threshold_override)
        if not np.isfinite(threshold):
            raise ValueError("Representation detector threshold must be finite")
        return {
            "method": self.method,
            "score": score,
            "threshold": threshold,
            "detected": bool(score > threshold),
            "details": details,
            "layer": self.layer,
            "pooling": self.pooling,
        }

    def score_vectors(self, vectors: Any, batch_size: int = 256) -> np.ndarray:
        if self.method == "cnrf-oracle":
            raise ValueError("cnrf-oracle requires aligned multi-view representations")
        projected = self.project_batch(vectors)
        if self.method == "nearside":
            direction = _l2_normalize(self.arrays["direction"].astype(np.float64))
            return projected @ direction
        if self.method == "vlmguard":
            values = projected
            for index in range(3):
                values = (
                    values
                    @ self.arrays[f"classifier_linear_{index}_weight"].astype(np.float64).T
                    + self.arrays[f"classifier_linear_{index}_bias"].astype(np.float64)[None, :]
                )
                if index < 2:
                    values = np.maximum(values, 0.0)
            logits = values[:, 0]
            probabilities = np.empty_like(logits)
            positive = logits >= 0.0
            probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
            exp_values = np.exp(logits[~positive])
            probabilities[~positive] = exp_values / (1.0 + exp_values)
            return probabilities
        if self.method == "rcs-kcd":
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            normalized = projected / np.maximum(norms, 1e-12)
            benign = self.arrays["benign_reference"].astype(np.float64)
            malicious = self.arrays["malicious_reference"].astype(np.float64)
            requested_k = int(self.metadata.get("k", 50))
            benign_k = min(max(requested_k, 1), len(benign))
            malicious_k = min(max(requested_k, 1), len(malicious))
            scores: list[np.ndarray] = []
            actual_batch_size = max(int(batch_size), 1)
            for start in range(0, len(normalized), actual_batch_size):
                batch = normalized[start : start + actual_batch_size]
                benign_squared = np.maximum(
                    0.0,
                    np.sum(batch * batch, axis=1, keepdims=True)
                    + np.sum(benign * benign, axis=1)[None, :]
                    - 2.0 * (batch @ benign.T),
                )
                malicious_squared = np.maximum(
                    0.0,
                    np.sum(batch * batch, axis=1, keepdims=True)
                    + np.sum(malicious * malicious, axis=1)[None, :]
                    - 2.0 * (batch @ malicious.T),
                )
                benign_distance = np.sqrt(
                    np.partition(benign_squared, benign_k - 1, axis=1)[:, benign_k - 1]
                )
                malicious_distance = np.sqrt(
                    np.partition(malicious_squared, malicious_k - 1, axis=1)[:, malicious_k - 1]
                )
                scores.append(benign_distance - malicious_distance)
            return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)

        scores = np.empty(len(projected), dtype=np.float64)
        benign_means = self.arrays["benign_means"]
        benign_precisions = self.arrays["benign_precisions"]
        malicious_means = self.arrays["malicious_means"]
        malicious_precisions = self.arrays["malicious_precisions"]
        for index, value in enumerate(projected):
            benign_distance, _ = self._minimum_mahalanobis(
                value, benign_means, benign_precisions
            )
            malicious_distance, _ = self._minimum_mahalanobis(
                value, malicious_means, malicious_precisions
            )
            scores[index] = np.clip(
                benign_distance - malicious_distance, -10000.0, 10000.0
            )
        return scores


def score_cnrf_views(
    detector: RepresentationDetector,
    branch: str,
    vectors: list[Any],
    threshold_override: float | None = None,
) -> dict[str, Any]:
    """Score one sample with the frozen, unified CNRF Oracle branch."""

    branch_config = detector.cnrf_branch_config(branch)
    views = branch_config["views"]
    if len(vectors) != len(views):
        raise ValueError(
            f"CNRF branch {branch!r} expects {len(views)} views, got {len(vectors)}"
        )
    diagnostics: list[dict[str, Any]] = []
    eligible_scores: list[tuple[float, int]] = []
    for index, (view, raw_vector) in enumerate(zip(views, vectors)):
        vector = _one_dimensional(raw_vector, expected_dim=detector.hidden_dim)
        result = detector.cnrf_field(branch, index).score(vector)
        raw_score = float(result.scores[0])
        nearest_distance = float(result.nearest_midpoint_distance[0])
        score = (raw_score - float(view["center"])) / float(view["scale"])
        supported = bool(nearest_distance <= float(view["support_radius"]))
        if supported:
            eligible_scores.append((score, index))
        diagnostics.append(
            {
                "readout": str(view["readout"]),
                "layer": int(view["layer"]),
                "raw_score": raw_score,
                "standardized_score": score,
                "nearest_midpoint_distance": nearest_distance,
                "support_radius": float(view["support_radius"]),
                "supported": supported,
                "neighbor_pair_ids": result.neighbor_pair_ids[0],
                "neighbor_pack_ids": result.neighbor_pack_ids[0],
                "neighbor_coordinates": result.neighbor_coordinates[0],
            }
        )

    if eligible_scores:
        score, winning_view = max(eligible_scores, key=lambda item: (item[0], -item[1]))
        supported = True
    else:
        # abstain_safe is encoded as a finite sentinel so aggregate summaries and
        # resume files never contain NaN/Infinity.
        score, winning_view, supported = -1.0e9, None, False
    threshold = (
        float(branch_config["threshold"])
        if threshold_override is None
        else float(threshold_override)
    )
    if not np.isfinite(threshold):
        raise ValueError("CNRF Oracle threshold must be finite")
    return {
        "method": detector.method,
        "score": float(score),
        "threshold": threshold,
        "detected": bool(score >= threshold),
        "details": {
            "branch": branch,
            "supported": supported,
            "winning_view": winning_view,
            "fusion_policy": "supported_max",
            "support_policy": "abstain_safe",
            "candidate_id": branch_config.get("candidate_id"),
            "selected_pack_count": len(branch_config.get("selected_packs") or []),
            "views": diagnostics,
        },
        "layer": [int(view["layer"]) for view in views],
        "pooling": [str(view["readout"]) for view in views],
    }


def score_representation_sample(
    runner: BaseModelRunner,
    detector: RepresentationDetector,
    prompt: str,
    image_path: str | None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    if detector.method == "cnrf-oracle":
        branch = "image_text" if image_path else "text"
        branch_config = detector.cnrf_branch_config(branch)
        requested = [
            (int(view["layer"]), str(view["readout"]))
            for view in branch_config["views"]
        ]
        hidden_views = runner.extract_hidden_views(
            prompt,
            requested,
            image_path=image_path,
        )
        scored = score_cnrf_views(
            detector,
            branch,
            [hidden.vector for hidden in hidden_views],
            threshold_override=threshold_override,
        )
        return {
            **scored,
            "rendered_prompt": hidden_views[0].rendered_prompt,
            "backend": hidden_views[0].backend,
            "hidden_metadata": [hidden.metadata for hidden in hidden_views],
        }
    hidden: HiddenRepresentation = runner.extract_hidden(
        prompt,
        layer=detector.layer,
        image_path=image_path,
        pooling=detector.pooling,
    )
    scored = detector.score_vector(hidden.vector, threshold_override=threshold_override)
    return {
        **scored,
        "rendered_prompt": hidden.rendered_prompt,
        "backend": hidden.backend,
        "hidden_metadata": hidden.metadata,
    }

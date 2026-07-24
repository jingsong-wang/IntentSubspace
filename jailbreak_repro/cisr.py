from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


CISR_DEFENSE_CHOICES = ("cisr", "cisr2", "cisr3", "cisr4")
_CISR_FORMAT_PATTERN = re.compile(r"^CISR_v(?P<version>[0-9]+)_detector(?:_|$)", re.IGNORECASE)
_CISR_NAME_PATTERN = re.compile(r"^cisr(?:[_-]?v)?(?P<version>[0-9]+)$", re.IGNORECASE)


def normalize_cisr_version(value: str) -> str:
    name = value.strip().lower()
    if name == "cisr":
        return name
    match = _CISR_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported CISR defense version: {value!r}")
    return f"cisr{int(match.group('version'))}"


def is_cisr_defense(value: str) -> bool:
    try:
        normalize_cisr_version(value)
    except ValueError:
        return False
    return True


def cisr_version_from_format(format_version: str) -> str:
    match = _CISR_FORMAT_PATTERN.match(format_version.strip())
    if match is None:
        raise ValueError(f"Unsupported CISR detector format_version: {format_version!r}")
    return f"cisr{int(match.group('version'))}"


def inspect_cisr_artifact_version(path: Path | str) -> str:
    artifact_path = Path(path).expanduser().resolve()
    if artifact_path.suffix.lower() == ".json":
        manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        return cisr_version_from_format(str(manifest.get("format_version", "")))
    with np.load(artifact_path, allow_pickle=False) as data:
        if "format_version" not in data:
            # CISRDetector.load uses the same v2 fallback for legacy artifacts.
            return "cisr2"
        raw = np.asarray(data["format_version"]).reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return cisr_version_from_format(str(raw))


def cisr_artifact_sha1(path: Path | str) -> str:
    artifact_path = Path(path).expanduser().resolve()
    digest = hashlib.sha1()
    digest.update(artifact_path.read_bytes())
    if artifact_path.suffix.lower() == ".json":
        manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        branches = manifest.get("branches", {})
        for branch in sorted(branches):
            child_value = branches[branch].get("detector")
            if not child_value:
                continue
            child = Path(str(child_value))
            if not child.is_absolute():
                child = artifact_path.parent / child
            digest.update(branch.encode("utf-8"))
            digest.update(child.resolve().read_bytes())
    return digest.hexdigest()


def resolve_cisr_version(requested: str, artifact_path: Path | str) -> str:
    requested_version = normalize_cisr_version(requested)
    artifact_version = inspect_cisr_artifact_version(artifact_path)
    if requested_version != "cisr" and requested_version != artifact_version:
        raise ValueError(
            f"Requested defense {requested_version!r}, but detector artifact is {artifact_version!r}: "
            f"{Path(artifact_path).resolve()}"
        )
    return artifact_version

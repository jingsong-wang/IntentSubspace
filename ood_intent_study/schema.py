from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION


VALID_SPLITS = {"train", "validation", "test", "external"}
VALID_LABELS = {0, 1}


@dataclass(frozen=True)
class StudySample:
    sample_id: str
    source: str
    source_kind: str
    source_record_id: str
    label: int
    label_name: str
    label_semantics: str
    label_confidence: str
    label_provenance: str
    source_role: str
    prompt_text: str
    semantic_text: str
    image_path: str | None
    image_exists: bool
    modality: str
    category: str
    variant: str
    group_id: str
    semantic_group_id: str
    split_group_id: str
    nuisance_group_id: str
    split: str
    is_attack: bool
    attack_name: str
    source_file: str
    source_row: int
    prompt_sha256: str
    semantic_sha256: str
    image_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self, repo_root: Path | None = None, require_images: bool = False) -> list[str]:
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"unsupported schema_version={self.schema_version!r}")
        if not self.sample_id:
            errors.append("sample_id is empty")
        if not self.source:
            errors.append("source is empty")
        if self.label not in VALID_LABELS:
            errors.append(f"label must be 0 or 1, got {self.label!r}")
        if not self.prompt_text.strip():
            errors.append("prompt_text is empty")
        if self.split not in VALID_SPLITS:
            errors.append(f"invalid split={self.split!r}")
        if self.is_attack and self.split != "external":
            errors.append("attack samples must use split='external'")
        if not self.is_attack and self.split == "external":
            errors.append("non-attack samples cannot use split='external'")
        if self.is_attack and self.label != 1:
            errors.append("external jailbreak attacks must use harmful label 1")
        if self.image_path:
            path = Path(self.image_path)
            if repo_root is not None and not path.is_absolute():
                path = repo_root / path
            exists = path.is_file()
            if exists != self.image_exists:
                errors.append(
                    f"image_exists={self.image_exists} disagrees with filesystem for {self.image_path!r}"
                )
            if require_images and not exists:
                errors.append(f"required image is missing: {self.image_path}")
        elif require_images and self.modality != "text":
            errors.append("multimodal sample has no image_path")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "StudySample":
        return cls(**row)

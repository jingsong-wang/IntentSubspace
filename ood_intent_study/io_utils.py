from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: Any, prefix: str = "") -> str:
    payload = "\x1f".join(str(part) for part in parts)
    suffix = sha256_text(payload)[:20]
    return f"{prefix}{suffix}" if prefix else suffix


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def first_nonempty(row: dict[str, Any], fields: Iterable[str]) -> tuple[str, str]:
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        if value is None or isinstance(value, (dict, list, tuple, bytes, bytearray)):
            continue
        text = str(value).strip()
        if text:
            return text, field
    return "", ""


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def portable_path(path: Path, root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _path_suffixes(raw: str) -> list[Path]:
    normalized = raw.replace("\\", "/")
    parts = list(PurePosixPath(normalized).parts)
    suffixes: list[Path] = []
    for marker in ("intent_subspace", "jailbreak_repro", "benchmark"):
        if marker not in parts:
            continue
        index = parts.index(marker)
        start = index + 1 if marker == "intent_subspace" else index
        suffixes.append(Path(*parts[start:]))
    return suffixes


def relocate_path(
    raw: Any,
    root: Path,
    base_dir: Path | None = None,
    basename_dirs: Iterable[Path] = (),
) -> Path | None:
    if raw is None or isinstance(raw, (dict, list, tuple, bytes, bytearray)):
        return None
    text = str(raw).strip()
    if not text:
        return None

    original = Path(text).expanduser()
    candidates = [original]
    if not original.is_absolute():
        candidates.append(root / original)
        if base_dir is not None:
            candidates.append(base_dir / original)
    for suffix in _path_suffixes(text):
        candidates.append(root / suffix)
    for directory in basename_dirs:
        candidates.append(directory / original.name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


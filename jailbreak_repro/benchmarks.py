from __future__ import annotations

import csv
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .io_utils import repo_root


PROMPT_FIELDS = ["prompt", "prompt_text", "query", "question", "instruction", "goal"]
IMAGE_FIELDS = ["image_path", "image", "image_id", "image_file", "img", "image_name"]
SUPPORTED_DATA_SUFFIXES = {".csv", ".jsonl", ".json"}
JAILBREAKV_FULL_ALIASES = {
    "jailbreakv",
    "jailbreakv28k",
    "jailbreakv28000",
    "jailbreakvfull",
}
JAILBREAKV_MINI_ALIASES = {
    "jailbreakvmini",
    "minijailbreakv",
    "jailbreakv28kmini",
    "minijailbreakv28k",
}
JAILBREAKV_FULL_FILENAMES = [
    "JailBreakV_28K.csv",
    "JailBreakV_28k.csv",
    "JailBreakV-28K.csv",
    "JailBreakV-28k.csv",
    "jailbreakv_28k.csv",
]
JAILBREAKV_MINI_FILENAMES = [
    "mini_JailBreakV_28K.csv",
    "mini_JailBreakV_28k.csv",
    "JailBreakV-mini.csv",
    "JailBreakV_mini.csv",
    "jailbreakv_mini.csv",
]


def benchmark_root() -> Path:
    return repo_root() / "benchmark"


def _alias_key(value: str | Path) -> str:
    raw_name = Path(str(value).replace("\\", "/")).name
    path_name = Path(raw_name)
    name = path_name.stem if path_name.suffix else raw_name
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _jailbreakv_alias(value: str | Path) -> str | None:
    key = _alias_key(value)
    if key in JAILBREAKV_MINI_ALIASES:
        return "jailbreakv-mini"
    if key in JAILBREAKV_FULL_ALIASES:
        return "jailbreakv"
    return None


def _jailbreakv_path_candidates(alias: str) -> list[Path]:
    root = benchmark_root()
    full_dirs = [
        root / "JailBreakV_28K",
        root / "JailBreakV-28K",
        root / "JailBreakV_28k",
        root / "JailBreakV-28k",
        root / "JailBreakV",
        root / "jailbreakV",
        root / "jailbreakv",
    ]
    if alias == "jailbreakv-mini":
        return [
            root / "JailBreakV-mini",
            root / "JailBreakV_mini",
            root / "mini_JailBreakV_28K",
            root / "mini_JailBreakV_28k",
            *(root / name for name in JAILBREAKV_MINI_FILENAMES),
            *full_dirs,
        ]
    return [*(root / name for name in JAILBREAKV_FULL_FILENAMES), *full_dirs]


def resolve_benchmark_path(benchmark: str | Path) -> Path:
    raw = Path(benchmark)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((repo_root() / raw).resolve())
        candidates.append((benchmark_root() / raw).resolve())
        alias = _jailbreakv_alias(benchmark)
        if alias:
            candidates.extend(candidate.resolve() for candidate in _jailbreakv_path_candidates(alias))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve benchmark {benchmark!r}. Expected a path or a name under {benchmark_root()}."
    )


def _first_text(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, (list, dict)):
            continue
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_image(value: str, base_dir: Path, image_dir: Path | None = None) -> str:
    if not value.strip():
        return ""
    raw = Path(value)
    search_dirs = [p for p in [image_dir, base_dir, base_dir / "images"] if p is not None]
    if raw.is_absolute() and raw.exists():
        return str(raw.resolve())
    basename = raw.name or PurePosixPath(value).name
    if basename and basename != value:
        for directory in search_dirs:
            candidate = directory / basename
            if candidate.exists():
                return str(candidate.resolve())
    if raw.is_absolute():
        return str(raw)
    for directory in search_dirs:
        candidate = directory / raw
        if candidate.exists():
            return str(candidate.resolve())
    return str((base_dir / raw).resolve())


def _label_fields(row: dict[str, Any]) -> dict[str, Any]:
    label_value = row.get("label")
    if label_value is None:
        label_value = row.get("label_int")
    if label_value is None:
        return {"label": 1, "label_name": "target"}
    text = str(label_value).strip().lower()
    if text in {"safe", "benign", "0", "false"}:
        return {"label": 0, "label_name": "benign_control"}
    if text in {"unsafe", "harmful", "target", "1", "true"}:
        return {"label": 1, "label_name": "target"}
    return {"label": label_value, "label_name": str(label_value)}


def normalize_benchmark_row(
    row: dict[str, Any],
    benchmark_name: str,
    base_dir: Path,
    index: int,
    image_dir: Path | None = None,
) -> dict[str, Any]:
    prompt = _first_text(row, PROMPT_FIELDS)
    if not prompt:
        raise ValueError(f"Benchmark row {index} in {benchmark_name} has no prompt-like field: {row}")
    image_value = _first_text(row, IMAGE_FIELDS)
    image_path = _resolve_image(image_value, base_dir=base_dir, image_dir=image_dir) if image_value else ""
    row_id = str(row.get("id") or row.get("sample_id") or f"{benchmark_name}_{index:06d}")
    labels = _label_fields(row)
    out = {
        **row,
        "id": row_id,
        "attack": "none",
        "benchmark": benchmark_name,
        "dataset": str(row.get("dataset") or benchmark_name),
        "prompt": prompt,
        "prompt_text": prompt,
        "original_question": prompt,
        "image_path": image_path,
        "image_role": "benchmark_image" if image_path else "none",
        **labels,
    }
    if "category_name" not in out:
        out["category_name"] = str(row.get("scenario") or row.get("category") or row.get("type") or benchmark_name)
    if "intent_family" not in out:
        out["intent_family"] = str(out["category_name"])
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [dict(row) for row in data["data"]]
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("id", key)
                rows.append(row)
        if rows:
            return rows
    raise ValueError(f"Unsupported JSON benchmark structure: {path}")


def load_hades(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    base_dir = path if path.is_dir() else path.parent
    meta = path / "hades_750_meta.jsonl" if path.is_dir() else path
    image_dir = base_dir / "images"
    if not meta.exists():
        raise FileNotFoundError(f"Missing HADES metadata file: {meta}")
    rows = _load_jsonl(meta)
    samples = [
        normalize_benchmark_row(row, "HADES", base_dir=base_dir, image_dir=image_dir, index=i)
        for i, row in enumerate(rows, start=1)
    ]
    return samples[:max_samples] if max_samples is not None else samples


def load_xstest(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    base_dir = path if path.is_dir() else path.parent
    csv_path = path / "xstest_prompts.csv" if path.is_dir() else path
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing XSTest CSV file: {csv_path}")
    rows = _load_csv(csv_path)
    samples = []
    for i, row in enumerate(rows, start=1):
        raw_label = str(row.get("label", "")).strip().lower()
        if raw_label in {"safe", "benign", "0", "false"}:
            xstest_label = "safe"
        elif raw_label in {"unsafe", "harmful", "target", "1", "true"}:
            xstest_label = "unsafe"
        else:
            raise ValueError(f"Unsupported XSTest label {row.get('label')!r} at row {i}")
        sample = normalize_benchmark_row(row, "XSTest", base_dir=base_dir, image_dir=None, index=i)
        sample["xstest_label"] = xstest_label
        samples.append(sample)
    return samples[:max_samples] if max_samples is not None else samples


def _choose_data_file(directory: Path) -> Path:
    jsonl = sorted(directory.glob("*.jsonl"))
    if jsonl:
        return jsonl[0]
    csv_files = sorted(directory.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    json_files = sorted(directory.glob("*.json"))
    if json_files:
        return json_files[0]
    raise FileNotFoundError(f"No .jsonl, .csv, or .json benchmark data file found in {directory}")


def _load_data_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported benchmark data file extension: {path}")


def load_generic_benchmark(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    data_path = _choose_data_file(path) if path.is_dir() else path
    base_dir = data_path.parent
    benchmark_name = data_path.parent.name if data_path.parent.name else data_path.stem
    rows = _load_data_file(data_path)
    samples = [
        normalize_benchmark_row(row, benchmark_name, base_dir=base_dir, image_dir=base_dir / "images", index=i)
        for i, row in enumerate(rows, start=1)
    ]
    return samples[:max_samples] if max_samples is not None else samples


def _find_named_file(base: Path, names: list[str]) -> Path | None:
    for directory in [base, base / "JailBreakV_28K", base / "JailBreakV-28K", base / "data"]:
        if not directory.exists() or not directory.is_dir():
            continue
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    lowered = {name.lower() for name in names}
    for candidate in base.rglob("*"):
        if candidate.is_file() and candidate.name.lower() in lowered:
            return candidate
    return None


def _choose_jailbreakv_data_file(path: Path, mini: bool) -> tuple[Path, bool]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_DATA_SUFFIXES:
            raise ValueError(f"Unsupported JailBreakV data file extension: {path}")
        return path, _alias_key(path) in JAILBREAKV_MINI_ALIASES
    preferred = _find_named_file(path, JAILBREAKV_MINI_FILENAMES if mini else JAILBREAKV_FULL_FILENAMES)
    if preferred is not None:
        return preferred, mini
    if mini:
        full = _find_named_file(path, JAILBREAKV_FULL_FILENAMES)
        if full is not None:
            return full, False
    return _choose_data_file(path), False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_jailbreakv_row(row: dict[str, Any], base_dir: Path, index: int, mini: bool) -> dict[str, Any]:
    jailbreak_query = _first_text(
        row,
        [
            "jailbreak_query",
            "jailbreak_prompt",
            "attack_prompt",
            "prompt_text",
            "prompt",
            "query",
            "question",
        ],
    )
    redteam_query = _first_text(row, ["redteam_query", "redteam_prompt", "harmful_query", "question", "goal", "prompt"])
    if not jailbreak_query:
        raise ValueError(f"JailBreakV row {index} has no jailbreak_query-like field: {row}")

    image_value = _first_text(row, ["image_path", "image", "image_file", "img", "image_name"])
    image_path = _resolve_image(image_value, base_dir=base_dir, image_dir=base_dir) if image_value else ""
    row_id = str(row.get("id") or row.get("sample_id") or f"JailBreakV_{index:06d}")
    benchmark_name = "JailBreakV-mini" if mini else "JailBreakV"
    category_name = str(row.get("policy") or row.get("category") or row.get("format") or benchmark_name)
    source = row.get("from")
    out = {
        **row,
        "id": row_id,
        "attack": "none",
        "benchmark": benchmark_name,
        "benchmark_split": "mini_JailBreakV_28K" if mini else "JailBreakV_28K",
        "dataset": "JailBreakV_28K",
        "prompt": redteam_query or jailbreak_query,
        "prompt_text": jailbreak_query,
        "original_question": redteam_query or jailbreak_query,
        "image_path": image_path,
        "image_role": "benchmark_image" if image_path else "none",
        "label": 1,
        "label_name": "target",
        "category_name": category_name,
        "intent_family": category_name,
        "jailbreakv_format": str(row.get("format") or ""),
        "jailbreakv_policy": str(row.get("policy") or ""),
        "jailbreakv_source": str(source or ""),
        "jailbreakv_selected_mini": _truthy(row.get("selected_mini", mini)),
        "jailbreakv_transfer_from_llm": _truthy(row.get("transfer_from_llm", False)),
    }
    return out


def load_jailbreakv(path: Path, max_samples: int | None = None, mini: bool = False) -> list[dict[str, Any]]:
    data_path, data_file_is_mini = _choose_jailbreakv_data_file(path, mini=mini)
    rows = _load_data_file(data_path)
    if mini and not data_file_is_mini:
        if rows and any("selected_mini" in row for row in rows):
            rows = [row for row in rows if _truthy(row.get("selected_mini"))]
        else:
            raise FileNotFoundError(
                "JailBreakV-mini requires the official mini_JailBreakV_28K.csv file or a full "
                "JailBreakV_28K.csv with selected_mini annotations."
            )
    base_dir = path if path.is_dir() else data_path.parent
    samples = [normalize_jailbreakv_row(row, base_dir=base_dir, index=i, mini=mini) for i, row in enumerate(rows, start=1)]
    return samples[:max_samples] if max_samples is not None else samples


def load_benchmark_samples(benchmark: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    requested_alias = _jailbreakv_alias(benchmark)
    path = resolve_benchmark_path(benchmark)
    name = path.name.lower()
    resolved_alias = _jailbreakv_alias(path)
    if requested_alias or resolved_alias:
        alias = requested_alias or resolved_alias
        return load_jailbreakv(path, max_samples=max_samples, mini=alias == "jailbreakv-mini")
    if name == "hades" or path.name == "hades_750_meta.jsonl":
        return load_hades(path, max_samples=max_samples)
    if name == "xstest" or path.name == "xstest_prompts.csv":
        return load_xstest(path, max_samples=max_samples)
    return load_generic_benchmark(path, max_samples=max_samples)

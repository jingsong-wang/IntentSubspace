from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jailbreak_repro.io_utils import repo_root, write_json, write_jsonl


DEFAULT_RCS_SOURCE = (
    repo_root() / "jailbreak_repro" / "sourcecode" / "Jailbreak_Detection_RCS-main"
)
DEFAULT_OUTPUT = repo_root() / "data" / "representation_repro" / "rcs_paper.jsonl"

EXPECTED_COUNTS = {
    "train": {
        "Alpaca": 500,
        "MM-Vet": 218,
        "OpenAssistant": 282,
        "AdvBench": 300,
        "JailbreakV-28K_llm_transfer_attack": 275,
        "JailbreakV-28K_query_related": 275,
        "DAN": 150,
    },
    "test": {
        "XSTest_safe": 250,
        "XSTest_unsafe": 200,
        "FigTxt_safe": 300,
        "FigTxt_unsafe": 350,
        "VQAv2": 350,
        "VAE": 200,
        "JailbreakV-28K_test": 150,
    },
}

SOURCE_DATASET_NAMES = {
    "JailbreakV-28K_llm_transfer_attack": "JailbreakV-28K",
    "JailbreakV-28K_query_related": "JailbreakV-28K",
}


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_module(source_root: Path) -> ModuleType:
    loader_path = source_root / "code" / "load_datasets.py"
    if not loader_path.is_file():
        raise FileNotFoundError(f"RCS data loader does not exist: {loader_path}")
    spec = importlib.util.spec_from_file_location("rcs_official_load_datasets", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {loader_path}")
    module = importlib.util.module_from_spec(spec)
    temporary_modules: list[str] = []
    if importlib.util.find_spec("torch") is None:
        # The released data-only loader imports torch.nn.functional but never uses it.
        torch_stub = types.ModuleType("torch")
        nn_stub = types.ModuleType("torch.nn")
        functional_stub = types.ModuleType("torch.nn.functional")
        torch_stub.nn = nn_stub  # type: ignore[attr-defined]
        nn_stub.functional = functional_stub  # type: ignore[attr-defined]
        for name, value in (
            ("torch", torch_stub),
            ("torch.nn", nn_stub),
            ("torch.nn.functional", functional_stub),
        ):
            sys.modules[name] = value
            temporary_modules.append(name)
    try:
        spec.loader.exec_module(module)
    finally:
        for name in reversed(temporary_modules):
            sys.modules.pop(name, None)
    return module


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    try:
        value = function(*args, **kwargs)
    except Exception as exc:
        # The released prepare_balanced_* functions catch each source independently.
        print(f"Could not load {function.__name__}: {exc}")
        return []
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Official loader {function.__name__} returned {type(value).__name__}, not list")
    return [item for item in value if isinstance(item, dict)]


def _filter_label(samples: list[dict[str, Any]], label: int, maximum: int) -> list[dict[str, Any]]:
    return [item for item in samples if int(item.get("toxicity", label)) == label][:maximum]


def load_official_composition(module: ModuleType, seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if hasattr(module, "set_dataset_random_seed"):
        module.set_dataset_random_seed(seed)

    train: dict[str, list[dict[str, Any]]] = {}
    train["Alpaca"] = _call(module.load_alpaca, max_samples=500)
    train["MM-Vet"] = _filter_label(_call(module.load_mm_vet), 0, 218)
    train["OpenAssistant"] = _call(module.load_openassistant, max_samples=282)
    train["AdvBench"] = _call(module.load_advbench, max_samples=300)
    train["JailbreakV-28K_llm_transfer_attack"] = _call(
        module.load_JailBreakV_custom,
        attack_types=["llm_transfer_attack"],
        max_samples=275,
    )
    train["JailbreakV-28K_query_related"] = _call(
        module.load_JailBreakV_custom,
        attack_types=["query_related"],
        max_samples=275,
    )
    train["DAN"] = _call(module.load_dan_prompts, max_samples=150)

    test: dict[str, list[dict[str, Any]]] = {}
    xstest = _call(module.load_XSTest)
    test["XSTest_safe"] = _filter_label(xstest, 0, 250)
    test["XSTest_unsafe"] = _filter_label(xstest, 1, 200)
    figtxt = _call(module.load_FigTxt)
    test["FigTxt_safe"] = _filter_label(figtxt, 0, 300)
    test["FigTxt_unsafe"] = _filter_label(figtxt, 1, 350)
    test["VQAv2"] = _call(module.load_vqav2, max_samples=350)[:350]
    test["VAE"] = _call(module.load_adversarial_img)[:200]
    test["JailbreakV-28K_test"] = _call(module.load_JailBreakV_figstep, max_samples=150)[:150]
    return {"train": train, "test": test}


def _resolve_image(value: Any, source_root: Path, path_mode: str) -> tuple[str | None, bool]:
    if value in {None, "", "null"}:
        return None, True
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = source_root / candidate
    candidate = candidate.resolve()
    exists = candidate.is_file()
    if path_mode == "relative":
        try:
            return candidate.relative_to(repo_root()).as_posix(), exists
        except ValueError:
            pass
    return str(candidate), exists


def _image_role(group: str, image_path: str | None) -> str:
    if not image_path:
        return "text_only"
    if group.startswith("JailbreakV-28K"):
        return "jailbreak_visual_carrier"
    if group == "VAE":
        return "adversarial_visual_carrier"
    return "benign_semantic_image"


def _row_id(split: str, group: str, index: int) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in group)
    slug = "_".join(part for part in slug.split("_") if part)
    return f"rcs_{split}_{slug}_{index:06d}"


def normalize_composition(
    composition: dict[str, dict[str, list[dict[str, Any]]]],
    source_root: Path,
    path_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for split in ("train", "test"):
        for group, samples in composition[split].items():
            expected_label = 0 if group in {
                "Alpaca",
                "MM-Vet",
                "OpenAssistant",
                "XSTest_safe",
                "FigTxt_safe",
                "VQAv2",
            } else 1
            source = SOURCE_DATASET_NAMES.get(group, group)
            for index, sample in enumerate(samples):
                prompt = str(sample.get("txt") or sample.get("question") or "").strip()
                if not prompt:
                    continue
                image_path, image_exists = _resolve_image(sample.get("img"), source_root, path_mode)
                if image_path and not image_exists:
                    missing_images.append(image_path)
                sample_id = _row_id(split, group, index)
                row: dict[str, Any] = {
                    "id": sample_id,
                    "label": expected_label,
                    "label_name": "target" if expected_label else "benign_control",
                    "condition": group,
                    "evaluation_split": split,
                    "split_group": split,
                    "pair_key": sample_id,
                    "intent_id": sample_id,
                    "intent_text": prompt,
                    "intent_family": source,
                    "prompt_text": prompt,
                    "image_path": image_path,
                    "image_role": _image_role(group, image_path),
                    "carrier_type": "image_text" if image_path else "text_only",
                    "source": source,
                    "source_group": group,
                    "rcs_official_toxicity": int(sample.get("toxicity", expected_label)),
                }
                for key in (
                    "attack_type",
                    "image_style",
                    "question_id",
                    "image_id",
                    "type",
                    "focus",
                    "note",
                    "platform",
                ):
                    value = sample.get(key)
                    if value is None or isinstance(value, (str, int, float, bool)):
                        if value is not None:
                            row[f"source_{key}"] = value
                rows.append(row)
    return rows, missing_images


def _image_content_fingerprint(rows: list[dict[str, Any]]) -> tuple[str, int]:
    paths = sorted(
        {str(row.get("image_path")) for row in rows if row.get("image_path")}
    )
    digest = hashlib.sha1()
    count = 0
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root() / path
        path = path.resolve()
        if not path.is_file():
            continue
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha1(path).encode("ascii"))
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def _logical_fingerprint(
    rows: list[dict[str, Any]],
    loader_sha1: str,
    image_content_fingerprint: str,
) -> str:
    digest = hashlib.sha1(loader_sha1.encode("ascii"))
    digest.update(image_content_fingerprint.encode("ascii"))
    for row in rows:
        for key in ("id", "label", "evaluation_split", "source", "prompt_text", "image_path"):
            digest.update(str(row.get(key, "")).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact released RCS 2,000/1,800 composition through its official "
            "load_datasets.py, then normalize it for the shared activation extractor."
        )
    )
    parser.add_argument("--protocol", choices=["rcs-paper"], default="rcs-paper")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_RCS_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--path-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--seed", type=int, default=42, help="Official dataset sampling seed.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write available rows and return success even when manual RCS datasets are missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_dir.expanduser().resolve()
    output = args.out.expanduser().resolve()
    manifest_output = (
        args.manifest_out.expanduser().resolve()
        if args.manifest_out
        else output.with_suffix(".manifest.json")
    )
    module = _load_module(source_root)
    with _working_directory(source_root):
        composition = load_official_composition(module, args.seed)
    rows, missing_images = normalize_composition(composition, source_root, args.path_mode)

    actual_counts = {
        split: {group: len(composition[split].get(group) or []) for group in expected}
        for split, expected in EXPECTED_COUNTS.items()
    }
    count_mismatches = {
        split: {
            group: {"expected": expected, "actual": actual_counts[split][group]}
            for group, expected in expected_groups.items()
            if actual_counts[split][group] != expected
        }
        for split, expected_groups in EXPECTED_COUNTS.items()
    }
    count_mismatches = {split: values for split, values in count_mismatches.items() if values}
    loader_path = source_root / "code" / "load_datasets.py"
    loader_sha1 = _sha1(loader_path)
    exact_protocol = not count_mismatches and not missing_images and len(rows) == 3800
    dataset_protocol = "paper-rcs" if exact_protocol else "repository-rcs-incomplete"
    image_fingerprint, hashed_image_count = _image_content_fingerprint(rows)
    fingerprint = _logical_fingerprint(rows, loader_sha1, image_fingerprint)
    for row in rows:
        row["dataset_protocol"] = dataset_protocol
        row["dataset_exact_protocol"] = exact_protocol
        row["dataset_manifest_sha1"] = fingerprint

    write_jsonl(output, rows)
    label_counts = Counter(int(row["label"]) for row in rows)
    split_counts = Counter(str(row["evaluation_split"]) for row in rows)
    manifest = {
        "schema_version": "representation_dataset_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_protocol": "paper-rcs",
        "dataset_protocol": dataset_protocol,
        "exact_protocol": exact_protocol,
        "source_repository": "https://github.com/sarendis56/Jailbreak_Detection_RCS",
        "source_dir": str(source_root),
        "source_git_head": _git_head(source_root),
        "official_loader": str(loader_path),
        "official_loader_sha1": loader_sha1,
        "image_content_fingerprint": image_fingerprint,
        "hashed_image_count": hashed_image_count,
        "logical_fingerprint": fingerprint,
        "sampling_seed": args.seed,
        "expected_counts": EXPECTED_COUNTS,
        "actual_counts": actual_counts,
        "count_mismatches": count_mismatches,
        "row_count": len(rows),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "split_counts": dict(sorted(split_counts.items())),
        "missing_image_count": len(missing_images),
        "missing_images": sorted(set(missing_images))[:100],
        "output": str(output),
        "manual_data_note": (
            "Exact reproduction additionally requires MM-Vet, FigTxt, JailBreakV-28K, and VAE "
            "at the paths documented by the official RCS download_datasets.py."
        ),
    }
    write_json(manifest_output, manifest)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Wrote data manifest to {manifest_output}")
    print(f"Protocol: {dataset_protocol}; exact={exact_protocol}")
    if count_mismatches:
        print(json.dumps(count_mismatches, indent=2, ensure_ascii=False))
    if missing_images:
        print(f"Missing image files: {len(missing_images)}")
    return 0 if exact_protocol or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())

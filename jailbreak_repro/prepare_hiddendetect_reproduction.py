from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jailbreak_repro.benchmarks import load_jailbreakv
from jailbreak_repro.hiddendetect import default_hiddendetect_source_dir, load_hiddendetect_fewshot
from jailbreak_repro.io_utils import repo_root, write_json


XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"


def _download_file(url: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "HiddenDetect-reproduction/1.0"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _xstest_label(row: dict[str, str]) -> str:
    label = str(row.get("label") or "").strip().lower()
    if label in {"safe", "unsafe"}:
        return label
    sample_type = str(row.get("type") or "").strip().lower()
    if not sample_type:
        raise ValueError("XSTest row has neither a valid label nor a type")
    return "unsafe" if sample_type.startswith("contrast_") else "safe"


def normalize_xstest(source: Path, output: Path) -> Path:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"XSTest CSV is empty: {source}")

    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"XSTest row {index} has no prompt")
        normalized.append(
            {
                "id": str(row.get("id") or index),
                "prompt": prompt,
                "type": str(row.get("type") or "").strip(),
                "label": _xstest_label(row),
                "focus": str(row.get("focus") or "").strip(),
                "note": str(row.get("note") or "").strip(),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "prompt", "type", "label", "focus", "note"])
        writer.writeheader()
        writer.writerows(normalized)
    temporary.replace(output)
    return output


def inspect_xstest(path: Path, strict_official_counts: bool = True) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = Counter(_xstest_label(row) for row in rows)
    if strict_official_counts and (len(rows), labels["safe"], labels["unsafe"]) != (450, 250, 200):
        raise ValueError(
            "XSTest must contain the official 450 rows (250 safe, 200 unsafe); "
            f"found {len(rows)} rows ({labels['safe']} safe, {labels['unsafe']} unsafe) in {path}"
        )
    return {
        "path": str(path.resolve()),
        "count": len(rows),
        "safe_count": labels["safe"],
        "unsafe_count": labels["unsafe"],
    }


def prepare_xstest(benchmark_root: Path, download_missing: bool) -> dict[str, Any]:
    output = benchmark_root / "XSTest" / "xstest_prompts.csv"
    if not output.is_file():
        if not download_missing:
            raise FileNotFoundError(
                f"Missing XSTest data: {output}. Re-run with --download-missing to fetch the official CSV."
            )
        raw = output.with_name("xstest_prompts.official.csv")
        print(f"DOWNLOAD XSTest: {XSTEST_URL}")
        _download_file(XSTEST_URL, raw)
        normalize_xstest(raw, output)
    return inspect_xstest(output)


def inspect_jailbreakv_mini(path: Path, strict_official_count: bool = True) -> dict[str, Any]:
    samples = load_jailbreakv(path.expanduser().resolve(), mini=True)
    missing_images = [sample["image_path"] for sample in samples if not Path(sample["image_path"]).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"JailbreakV-mini references {len(missing_images)} missing images; first missing path: {missing_images[0]}"
        )
    if strict_official_count and len(samples) != 280:
        raise ValueError(f"JailbreakV-mini must contain the official 280 rows; found {len(samples)} in {path}")
    return {
        "path": str(path.expanduser().resolve()),
        "count": len(samples),
        "image_count": sum(bool(sample.get("image_path")) for sample in samples),
        "format_count": len({str(sample.get("jailbreakv_format")) for sample in samples}),
        "policy_count": len({str(sample.get("jailbreakv_policy")) for sample in samples}),
    }


def inspect_jood(dataset_dir: Path, source_dir: Path) -> dict[str, Any]:
    dataset = dataset_dir.expanduser().resolve()
    source = source_dir.expanduser().resolve()
    harmful = dataset / "images" / "harmful"
    harmless = dataset / "images" / "harmless"
    prompts = dataset / "prompts" / "all_instructions"
    required_code = [source / "utils" / name for name in ("mixaug.py", "strings.py", "randaug.py")]
    missing = [str(path) for path in [harmful, harmless, prompts, *required_code] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"JOOD reproduction inputs are incomplete; missing: {missing}")

    scenarios = sorted(path.name for path in harmful.iterdir() if path.is_dir())
    if not scenarios:
        raise ValueError(f"JOOD has no harmful image scenarios under {harmful}")
    prompt_scenarios = {path.stem for path in prompts.glob("*.json")}
    missing_prompts = sorted(set(scenarios) - prompt_scenarios)
    if missing_prompts:
        raise FileNotFoundError(f"JOOD scenarios have no matching prompt JSON: {missing_prompts}")

    harmful_images = [
        path
        for scenario in scenarios
        for path in (harmful / scenario).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    harmless_images = [
        path for path in harmless.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    prompt_count = 0
    for scenario in scenarios:
        payload = json.loads((prompts / f"{scenario}.json").read_text(encoding="utf-8"))
        instructions = payload.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            raise ValueError(f"JOOD prompt file has no instructions: {prompts / f'{scenario}.json'}")
        prompt_count += len(instructions)
    if not harmful_images or not harmless_images:
        raise ValueError("JOOD requires both harmful and harmless source images")
    return {
        "dataset_dir": str(dataset),
        "source_dir": str(source),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "harmful_image_count": len(harmful_images),
        "harmless_image_count": len(harmless_images),
        "prompt_count": prompt_count,
    }


def inspect_hiddendetect(source_dir: Path) -> dict[str, Any]:
    samples, fingerprint = load_hiddendetect_fewshot(source_dir)
    labels = Counter(int(sample["label"]) for sample in samples)
    image_count = sum(1 for sample in samples if sample.get("image_path"))
    if (len(samples), labels[0], labels[1], image_count) != (12, 6, 6, 8):
        raise ValueError(
            "HiddenDetect official calibration set must contain 12 rows "
            f"(6 safe, 6 unsafe, 8 with images); found {len(samples)}, {labels[0]}, {labels[1]}, {image_count}."
        )
    required_code = [
        source_dir / "code" / "safety_aware_layers.py",
        source_dir / "code" / "test.py",
        source_dir / "code" / "test_qwen.py",
    ]
    missing = [str(path) for path in required_code if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"HiddenDetect checkout is incomplete; missing: {missing}")
    return {
        "source_dir": str(source_dir.resolve()),
        "fewshot_count": len(samples),
        "fewshot_safe_count": labels[0],
        "fewshot_unsafe_count": labels[1],
        "fewshot_image_count": image_count,
        "fewshot_sha1": fingerprint,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify HiddenDetect's released few-shot data plus XSTest, JailbreakV-mini, "
            "and JOOD inputs used by the formal platform evaluation."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, default=repo_root() / "benchmark")
    parser.add_argument("--hiddendetect-source-dir", type=Path, default=default_hiddendetect_source_dir())
    parser.add_argument("--skip-xstest", action="store_true")
    parser.add_argument("--skip-jailbreakv-mini", action="store_true")
    parser.add_argument("--skip-jood", action="store_true")
    parser.add_argument("--jailbreakv-mini-path", type=Path)
    parser.add_argument("--jood-dataset-dir", type=Path)
    parser.add_argument(
        "--jood-source-dir",
        type=Path,
        default=repo_root() / "jailbreak_repro" / "sourcecode" / "JOOD-master",
    )
    parser.add_argument("--download-missing", action="store_true", help="Download XSTest if it is absent.")
    parser.add_argument("--manifest", type=Path, default=repo_root() / "runs" / "HiddenDetect" / "data_manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden_source = args.hiddendetect_source_dir.expanduser().resolve()
    if not hidden_source.is_dir():
        raise FileNotFoundError(
            f"Missing HiddenDetect checkout: {hidden_source}. Run "
            "python -m jailbreak_repro.sync_representation_upstreams --methods hiddendetect --strict"
        )
    result: dict[str, Any] = {
        "schema_version": "hiddendetect_reproduction_data_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hiddendetect": inspect_hiddendetect(hidden_source),
    }
    if not args.skip_xstest:
        result["xstest"] = prepare_xstest(args.benchmark_root.expanduser().resolve(), args.download_missing)
    if not args.skip_jailbreakv_mini:
        jailbreakv_path = args.jailbreakv_mini_path or (
            args.benchmark_root / "JailBreakV_28K" / "mini_JailBreakV_28K.csv"
        )
        if not jailbreakv_path.exists():
            raise FileNotFoundError(
                f"Missing JailbreakV-mini data: {jailbreakv_path}. This preparation command does not download it."
            )
        result["jailbreakv_mini"] = inspect_jailbreakv_mini(jailbreakv_path)
    if not args.skip_jood:
        jood_dataset = args.jood_dataset_dir or args.benchmark_root / "AdvBenchM"
        result["jood"] = inspect_jood(jood_dataset, args.jood_source_dir)
    manifest = args.manifest.expanduser().resolve()
    write_json(manifest, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote HiddenDetect reproduction data manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

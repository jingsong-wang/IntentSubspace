from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jailbreak_repro.attacks import load_csdj_samples, load_jood_samples

from .io_utils import canonical_json, repo_root, sha256_file, sha256_text, write_json_atomic, write_jsonl_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JOOD or CS-DJ input images without victim inference.")
    parser.add_argument("--attack", choices=["jood", "csdj"], required=True)
    parser.add_argument("--out-root", type=Path, default=Path("runs/ood_intent_study/prepared_attacks"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jood-dataset-dir", type=Path, default=Path("benchmark/AdvBenchM"))
    parser.add_argument("--jood-aug", default="mixup")
    parser.add_argument("--jood-lams", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--jood-scenarios")
    parser.add_argument("--csdj-source-dir", type=Path, default=Path("jailbreak_repro/sourcecode/CS-DJ-main"))
    parser.add_argument("--csdj-image-dir", type=Path)
    parser.add_argument("--csdj-image-map", type=Path)
    parser.add_argument("--csdj-embedding-map", type=Path)
    parser.add_argument("--csdj-subquestions", type=Path)
    parser.add_argument("--csdj-category", default="all")
    parser.add_argument("--csdj-num-images", type=int, default=100)
    parser.add_argument("--csdj-select-images", type=int, default=9)
    parser.add_argument("--csdj-aux-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--csdj-aux-model-source", choices=["auto", "hf", "modelscope"], default="auto")
    parser.add_argument("--csdj-aux-model-cache-dir", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _resolved(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    return (path if path.is_absolute() else root / path).expanduser().resolve()


def _path_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        value.update({"kind": "file", "sha256": sha256_file(path)})
    elif path.is_dir():
        entries = [
            f"{item.relative_to(path).as_posix()}:{item.stat().st_size}:{item.stat().st_mtime_ns}"
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        value.update(
            {
                "kind": "directory",
                "file_count": len(entries),
                "layout_sha256": sha256_text("\n".join(entries)),
            }
        )
    else:
        value["kind"] = "missing"
    return value


def _enrich_jood_instructions(samples: list[dict[str, Any]], dataset_dir: Path) -> None:
    prompt_dir = dataset_dir / "prompts" / "all_instructions"
    cache: dict[str, list[str]] = {}
    for sample in samples:
        scenario = str(sample.get("scenario") or "")
        if scenario not in cache:
            path = prompt_dir / f"{scenario}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            cache[scenario] = [str(value) for value in data["instructions"]]
        match = re.search(r"\[PromptIdx\](\d+)", str(sample.get("custom_id") or ""))
        if match is None:
            raise ValueError(f"Cannot recover JOOD prompt index from {sample.get('custom_id')!r}")
        prompt_index = int(match.group(1))
        sample["instruction"] = cache[scenario][prompt_index]


def _seed_generation(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    output_root = _resolved(args.out_root, root)
    assert output_root is not None
    name = "JOOD" if args.attack == "jood" else "CS-DJ"
    output = output_root / name
    attack_implementation = _path_identity(root / "jailbreak_repro" / "attacks.py")
    if args.attack == "jood":
        dataset_dir = _resolved(args.jood_dataset_dir, root)
        assert dataset_dir is not None
        protocol: dict[str, Any] = {
            "attack": "JOOD",
            "protocol_name": "JOOD-official-lams",
            "augmentation": args.jood_aug,
            "lams": args.jood_lams,
            "scenarios": args.jood_scenarios or "all",
            "max_samples": args.max_samples,
            "dataset": _path_identity(dataset_dir),
            "implementation": attack_implementation,
        }
        protocol_sha = sha256_text(canonical_json(protocol))
        assets = output / "assets" / protocol_sha[:16]
        samples = load_jood_samples(
            max_samples=args.max_samples,
            artifact_dir=assets,
            dataset_dir=dataset_dir,
            scenarios=args.jood_scenarios,
            aug=args.jood_aug,
            lams=args.jood_lams,
        )
        _enrich_jood_instructions(samples, dataset_dir)
    else:
        _seed_generation(args.seed)
        source_dir = _resolved(args.csdj_source_dir, root)
        image_dir = _resolved(args.csdj_image_dir, root)
        image_map = _resolved(args.csdj_image_map, root)
        embedding_map = _resolved(args.csdj_embedding_map, root)
        subquestions = _resolved(args.csdj_subquestions, root)
        protocol = {
            "attack": "CS-DJ",
            "protocol_name": f"CS-DJ-{args.csdj_num_images}",
            "retrieval_pool": args.csdj_num_images,
            "selected_distraction_images": args.csdj_select_images,
            "category": args.csdj_category,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "aux_model": args.csdj_aux_model,
            "aux_model_source": args.csdj_aux_model_source,
            "trust_remote_code": bool(args.trust_remote_code),
            "implementation": attack_implementation,
            "source": _path_identity(source_dir),
            "image_dir": _path_identity(image_dir),
            "image_map": _path_identity(image_map),
            "embedding_map": _path_identity(embedding_map),
            "subquestions": _path_identity(subquestions),
        }
        protocol_sha = sha256_text(canonical_json(protocol))
        assets = output / "assets" / protocol_sha[:16]
        samples = load_csdj_samples(
            source_dir=source_dir,
            max_samples=args.max_samples,
            artifact_dir=assets,
            image_dir=image_dir,
            image_map_path=image_map,
            embedding_map_path=embedding_map,
            subquestions_file=subquestions,
            aux_model=args.csdj_aux_model,
            aux_model_source=args.csdj_aux_model_source,
            aux_model_cache_dir=_resolved(args.csdj_aux_model_cache_dir, root),
            aux_trust_remote_code=args.trust_remote_code,
            category=args.csdj_category,
            seed=args.seed,
            num_images=args.csdj_num_images,
            selected_distraction_images=args.csdj_select_images,
        )
    for sample in samples:
        sample["prepared_protocol_name"] = protocol["protocol_name"]
        sample["prepared_protocol_sha256"] = protocol_sha
    manifest = output / "samples.jsonl"
    write_jsonl_atomic(manifest, samples)
    metadata = {
        "schema_version": "prepared_attack_inputs_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(samples),
        "logical_sha256": sha256_text("\n".join(canonical_json(row) for row in samples)),
        "protocol_sha256": protocol_sha,
        "victim_inference_executed": False,
        "protocol": protocol,
    }
    write_json_atomic(output / "prepare.json", metadata)
    print(f"Wrote {len(samples)} {name} inputs to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

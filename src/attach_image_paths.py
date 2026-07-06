import argparse
import json
import os
from pathlib import Path


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalized_json_path(path: Path | str, relative_to: Path, path_mode: str) -> str:
    resolved = Path(path).resolve()
    if path_mode == "absolute":
        return resolved.as_posix()
    rel = os.path.relpath(resolved, relative_to.resolve())
    return Path(rel).as_posix()


def find_image(image_dir: Path, sample_id: str, relative_to: Path, path_mode: str) -> str | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return normalized_json_path(candidate, relative_to, path_mode)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-only", action="store_true", help="Attach paths only to label=1 samples.")
    parser.add_argument("--path-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--relative-to", type=Path, default=Path("."), help="Base directory for relative JSONL paths.")
    args = parser.parse_args()
    args.relative_to = args.relative_to.resolve()

    rows = load_jsonl(args.data)
    attached = 0
    missing = []
    for row in rows:
        if row.get("image_role") == "none":
            continue
        if args.target_only and int(row.get("label", 0)) != 1:
            continue
        path = find_image(args.image_dir, row["id"], args.relative_to, args.path_mode)
        if path:
            row["image_path"] = path
            attached += 1
        else:
            missing.append(row["id"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out}")
    print(f"Attached {attached} image paths")
    if missing:
        print("Missing image files for:")
        for sample_id in missing:
            print(f"  {sample_id}")


if __name__ == "__main__":
    main()

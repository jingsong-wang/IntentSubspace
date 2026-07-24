from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jailbreak_repro.io_utils import read_json, repo_root, write_json


DEFAULT_MANIFEST = Path(__file__).with_name("representation_upstreams.json")
DEFAULT_SOURCE_ROOT = repo_root() / "jailbreak_repro" / "sourcecode"
DEFAULT_REPORT = repo_root() / "runs" / "representation_repository_repro" / "upstreams.json"


def _run_git(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_value(path: Path, arguments: list[str]) -> str | None:
    result = _run_git(["-C", str(path), *arguments])
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _clone(repository: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _run_git(
        [
            "clone",
            "--depth",
            "1",
            "--branch",
            str(repository.get("ref") or "main"),
            str(repository["url"]),
            str(destination),
        ]
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown git clone error"
        raise RuntimeError(message)


def _update(destination: Path, ref: str) -> None:
    dirty = _git_value(destination, ["status", "--porcelain"])
    if dirty:
        raise RuntimeError("checkout has local changes; refusing --update")
    fetch = _run_git(["-C", str(destination), "fetch", "--depth", "1", "origin", ref])
    if fetch.returncode != 0:
        raise RuntimeError(fetch.stderr.strip() or "git fetch failed")
    merge = _run_git(["-C", str(destination), "merge", "--ff-only", "FETCH_HEAD"])
    if merge.returncode != 0:
        raise RuntimeError(merge.stderr.strip() or "git merge --ff-only failed")


def _inspect(repository: dict[str, Any], destination: Path) -> dict[str, Any]:
    required = [str(value) for value in repository.get("required_paths") or []]
    missing = [value for value in required if not (destination / value).exists()]
    is_git_checkout = (destination / ".git").exists()
    return {
        "method": repository["method"],
        "url": repository["url"],
        "requested_ref": repository.get("ref"),
        "destination": str(destination.resolve()),
        "declared_code_status": repository.get("code_status"),
        "exists": destination.is_dir(),
        "is_git_checkout": is_git_checkout,
        "git_head": _git_value(destination, ["rev-parse", "HEAD"]) if is_git_checkout else None,
        "git_branch": _git_value(destination, ["branch", "--show-current"]) if is_git_checkout else None,
        "git_dirty": bool(_git_value(destination, ["status", "--porcelain"])) if is_git_checkout else None,
        "required_paths": required,
        "missing_required_paths": missing,
        "required_paths_complete": not missing,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone or verify official representation-detector repositories without overwriting vendored snapshots."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--methods", help="Comma-separated method names; default is every manifest entry.")
    parser.add_argument("--offline", action="store_true", help="Only inspect existing checkouts.")
    parser.add_argument("--update", action="store_true", help="Fast-forward clean git checkouts to the requested ref.")
    parser.add_argument("--strict", action="store_true", help="Fail when a checkout or required path is missing.")
    args = parser.parse_args(argv)
    if args.offline and args.update:
        parser.error("--offline and --update are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = read_json(args.manifest.expanduser().resolve())
    requested = {
        value.strip().lower()
        for value in (args.methods or "").split(",")
        if value.strip()
    }
    repositories = [
        item
        for item in manifest.get("repositories") or []
        if not requested or str(item.get("method", "")).lower() in requested
    ]
    available = {str(item.get("method", "")).lower() for item in manifest.get("repositories") or []}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown methods in --methods: {sorted(unknown)}")

    source_root = args.source_root.expanduser().resolve()
    results: list[dict[str, Any]] = []
    for repository in repositories:
        destination = source_root / str(repository["destination"])
        action = "verified"
        error = None
        try:
            if not destination.exists():
                if args.offline:
                    action = "missing_offline"
                else:
                    print(f"CLONE {repository['method']}: {repository['url']}")
                    _clone(repository, destination)
                    action = "cloned"
            elif args.update and (destination / ".git").exists():
                print(f"UPDATE {repository['method']}: {destination}")
                _update(destination, str(repository.get("ref") or "main"))
                action = "updated"
            elif args.update:
                action = "vendored_snapshot_not_updated"
        except Exception as exc:  # keep a complete audit report across repositories
            action = "failed"
            error = str(exc)
        inspected = _inspect(repository, destination)
        inspected.update({"action": action, "error": error})
        results.append(inspected)
        status = "OK" if inspected["exists"] and inspected["required_paths_complete"] and not error else "INCOMPLETE"
        print(f"{status} {repository['method']}: {destination}")
        if error:
            print(f"  {error}")
        if inspected["missing_required_paths"]:
            print(f"  missing: {', '.join(inspected['missing_required_paths'])}")

    report = {
        "schema_version": "representation_upstream_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.expanduser().resolve()),
        "source_root": str(source_root),
        "repositories": results,
        "complete_count": sum(
            1 for item in results if item["exists"] and item["required_paths_complete"] and not item["error"]
        ),
        "requested_count": len(results),
    }
    write_json(args.report.expanduser().resolve(), report)
    print(f"Wrote upstream audit to {args.report.expanduser().resolve()}")
    incomplete = [
        item for item in results if not item["exists"] or not item["required_paths_complete"] or item["error"]
    ]
    return 2 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-}"
CACHE_ROOT="${MODELSCOPE_CACHE:-${MODELSCOPE_HOME:-$HOME/.cache/modelscope}}"
if [[ "$(basename "${CACHE_ROOT}")" == "hub" || "$(basename "${CACHE_ROOT}")" == "models" ]]; then
  CACHE_ROOT="$(dirname "${CACHE_ROOT}")"
fi
LEGACY_ROOT="${MODELSCOPE_LEGACY_MODELS_DIR:-$CACHE_ROOT/hub/models}"
NEW_ROOT="${MODELSCOPE_NEW_MODELS_DIR:-$CACHE_ROOT/models}"
STAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/link_modelscope_legacy_cache.sh
  bash scripts/link_modelscope_legacy_cache.sh google/gemma-3-12b-it

Environment overrides:
  MODELSCOPE_CACHE=/path/to/.cache/modelscope
  MODELSCOPE_LEGACY_MODELS_DIR=/path/to/modelscope/hub/models
  MODELSCOPE_NEW_MODELS_DIR=/path/to/modelscope/models

The script links legacy ModelScope cache directories:
  hub/models/org/model
to the newer layout:
  models/org--model

Existing complete new-layout model directories are left untouched.
Existing incomplete new-layout directories are renamed to *.prelink.<timestamp>
before the symlink is created.
EOF
}

if [[ "$#" -gt 1 ]]; then
  usage
  exit 2
fi

if [[ "${MODEL_ID}" == "-h" || "${MODEL_ID}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python was not found on PATH; activate your environment first." >&2
  exit 1
fi

if [[ ! -d "${LEGACY_ROOT}" ]]; then
  echo "Legacy ModelScope cache directory does not exist: ${LEGACY_ROOT}" >&2
  exit 1
fi

mkdir -p "${NEW_ROOT}"

is_complete_model_dir() {
  python - "$1" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_dir():
    raise SystemExit(1)
if not ((path / "config.json").is_file() or (path / "configuration.json").is_file()):
    raise SystemExit(1)

index_files = sorted(path.glob("*.index.json"))
if index_files:
    for index_file in index_files:
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        filenames = set(data.get("weight_map", {}).values())
        if filenames and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in filenames):
            raise SystemExit(0)
    raise SystemExit(1)

for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
    if any(item.is_file() and item.stat().st_size > 0 for item in path.glob(pattern)):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

link_one() {
  local legacy_dir="$1"
  local rel="$2"
  local target="${NEW_ROOT}/${rel//\//--}"

  if ! is_complete_model_dir "${legacy_dir}"; then
    echo "SKIP incomplete legacy cache: ${legacy_dir}"
    return 0
  fi

  if [[ -L "${target}" ]]; then
    local link_target
    link_target="$(readlink -f "${target}")"
    if [[ "${link_target}" == "$(readlink -f "${legacy_dir}")" ]]; then
      echo "OK existing symlink: ${target} -> ${legacy_dir}"
      return 0
    fi
    local backup="${target}.prelink.${STAMP}"
    echo "BACKUP existing symlink: ${target} -> ${backup}"
    mv "${target}" "${backup}"
  elif [[ -e "${target}" ]]; then
    if is_complete_model_dir "${target}"; then
      echo "SKIP complete new-layout cache already exists: ${target}"
      return 0
    fi
    local backup="${target}.prelink.${STAMP}"
    echo "BACKUP incomplete new-layout cache: ${target} -> ${backup}"
    mv "${target}" "${backup}"
  fi

  ln -s "${legacy_dir}" "${target}"
  echo "LINK ${target} -> ${legacy_dir}"
}

if [[ -n "${MODEL_ID}" ]]; then
  legacy_dir="${LEGACY_ROOT}/${MODEL_ID}"
  if [[ ! -d "${legacy_dir}" ]]; then
    echo "Legacy model cache does not exist: ${legacy_dir}" >&2
    exit 1
  fi
  link_one "${legacy_dir}" "${MODEL_ID}"
  exit 0
fi

while IFS= read -r -d '' legacy_dir; do
  rel="${legacy_dir#${LEGACY_ROOT}/}"
  link_one "${legacy_dir}" "${rel}"
done < <(find "${LEGACY_ROOT}" -mindepth 2 -maxdepth 2 -type d -print0)

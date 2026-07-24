#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]] && SCRIPT_DIR="."
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVATION_ROOT="${ACTIVATION_ROOT:-runs/CISR_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/representation_baselines/CISR_v2}"
MODELS_CSV="${MODELS:-qwen25vl7b,gemma3_12b,llama32_11b_vision}"
METHODS_CSV="${METHODS:-nearside,rcs-kcd,rcs-mcd,vlmguard}"
SOURCES_CSV="${SOURCES:-xstest,csdj}"
PHASE="all"
FORCE=0
DRY_RUN=0
FAIL_FAST=0
MAX_SAMPLES=""

usage() {
  cat <<'EOF'
Train matched VLM representation baselines and run detection-only OOD checks.

The script never writes into CISR_v1-v4 result directories. Completed training
artifacts and external cases are skipped unless --force is supplied.

Usage:
  bash jailbreak_repro/run_representation_detector_study.sh [options]

Options:
  --phase VALUE          train, external, or all (default: all)
  --models CSV           Model presets (default: qwen25vl7b,gemma3_12b,llama32_11b_vision)
  --methods CSV          nearside,rcs-kcd,rcs-mcd,vlmguard
  --sources CSV          xstest,csdj (default: both)
  --activation-root DIR  All-layer activation root (default: runs/CISR_v2)
  --output-root DIR      Isolated output root
  --max-samples N        Optional external smoke-test limit
  --force                Rebuild artifacts and rerun external cases
  --fail-fast            Stop at first failed model/method/case
  --dry-run              Print commands only
  -h, --help             Show this help

Environment overrides: PYTHON_BIN, MODELS, METHODS, SOURCES, ACTIVATION_ROOT,
OUTPUT_ROOT. RCS and VLMGuard require PyTorch and scikit-learn.
EOF
}

while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --models) MODELS_CSV="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --sources) SOURCES_CSV="$2"; shift 2 ;;
    --activation-root) ACTIVATION_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$PHASE" != "train" && "$PHASE" != "external" && "$PHASE" != "all" ]]; then
  echo "--phase must be train, external, or all" >&2
  exit 2
fi
if [[ -n "$MAX_SAMPLES" ]] && ! [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-samples must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
IFS=',' read -r -a SOURCES <<< "$SOURCES_CSV"

run_command() {
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN)); then
    return 0
  fi
  "$@"
}

record_failure() {
  local label="$1"
  local status="$2"
  echo "FAILED: $label (exit=$status)" >&2
  if ((FAIL_FAST)); then
    exit "$status"
  fi
}

train_detector() {
  local model="$1"
  local method="$2"
  local activations="$ACTIVATION_ROOT/$model/activations_all_layers.npz"
  local method_dir="$OUTPUT_ROOT/$model/$method"
  local artifact="$method_dir/detector.npz"
  local summary="$method_dir/detection_summary.json"
  if [[ ! -f "$activations" ]]; then
    record_failure "train model=$model method=$method: missing $activations" 2
    return
  fi
  if ((FORCE == 0)) && [[ -f "$artifact" && -f "$summary" ]]; then
    echo "SKIP train model=$model method=$method"
    return
  fi
  echo "TRAIN model=$model method=$method"
  run_command "$PYTHON_BIN" -m jailbreak_repro.train_representation_detector \
    --activations "$activations" \
    --method "$method" \
    --out "$artifact" \
    --output-dir "$method_dir" \
    --protocol matched-cisr
  local status=$?
  ((status == 0)) || record_failure "train model=$model method=$method" "$status"
}

artifact_sha1() {
  "$PYTHON_BIN" -c 'import hashlib,sys; print(hashlib.sha1(open(sys.argv[1], "rb").read()).hexdigest()[:12])' "$1"
}

run_external_case() {
  local model="$1"
  local method="$2"
  local source="$3"
  local artifact="$OUTPUT_ROOT/$model/$method/detector.npz"
  local case_dir="$OUTPUT_ROOT/$model/$method/external/$source"
  local fingerprint
  if [[ ! -f "$artifact" ]]; then
    if ((DRY_RUN)); then
      fingerprint="dryrun"
    else
      record_failure "external model=$model method=$method source=$source: missing artifact" 2
      return
    fi
  else
    fingerprint="$(artifact_sha1 "$artifact")" || {
      record_failure "fingerprint model=$model method=$method" 2
      return
    }
  fi
  local marker="$case_dir/.complete-$fingerprint"
  if ((FORCE == 0)) && [[ -f "$marker" ]]; then
    echo "SKIP external model=$model method=$method source=$source"
    return
  fi
  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.run_selected
    --victim-model "$model"
    --defense "$method"
    --judge-model none
    --out-dir "$case_dir"
    --representation-detector "$artifact"
    --representation-action monitor
    --max-new-tokens 1
  )
  if [[ "$source" == "xstest" ]]; then
    command+=(--benchmark XSTest)
  elif [[ "$source" == "csdj" ]]; then
    command+=(--attack csdj --attack-artifact-dir jailbreak_repro/runs/_shared_attack_artifacts/csdj)
  else
    record_failure "unsupported external source=$source" 2
    return
  fi
  if [[ -n "$MAX_SAMPLES" ]]; then
    command+=(--max-samples "$MAX_SAMPLES")
  fi
  echo "EXTERNAL model=$model method=$method source=$source"
  run_command "${command[@]}"
  local status=$?
  if ((status == 0)); then
    if ((DRY_RUN == 0)); then
      mkdir -p "$case_dir"
      printf '%s\n' "$fingerprint" > "$marker"
    fi
  else
    record_failure "external model=$model method=$method source=$source" "$status"
  fi
}

if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      train_detector "$model" "$method"
    done
  done
fi

if [[ "$PHASE" == "external" || "$PHASE" == "all" ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for source in "${SOURCES[@]}"; do
        run_external_case "$model" "$method" "$source"
      done
    done
  done
fi

echo "Representation detector study finished."

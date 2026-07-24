#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]] && SCRIPT_DIR="."
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
VICTIM_MODEL="qwen25vl7b"
VICTIM_BACKEND="auto"
VICTIM_SOURCE="auto"
JUDGE_MODEL="gemma3_12b"
JUDGE_BACKEND="auto"
JUDGE_SOURCE="auto"
JUDGE_BATCH_SIZE=8
DETECTOR=""
WORK="counterfactual_risk_field/work/v2_axes_temp07"
DTYPE="bfloat16"
DEVICE="auto"
ATTN_IMPLEMENTATION="sdpa"
JOOD_MAX_SAMPLES=500
ATTACK_ARTIFACT_ROOT="jailbreak_repro/runs/_shared_attack_artifacts"
DRY_RUN=0
FAIL_FAST=0
FORCE_RESPONSES=0
FORCE_JUDGE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run the protocol-complete unified CNRF Oracle evaluation.

Cases:
  CS-DJ (750), JOOD (500), JailbreakV-mini (280), XSTest (450)
Judge:
  Gemma3-12B, auto task (ASR for attacks; over-refusal for XSTest)

One victim-specific CNRF artifact is frozen for all cases. This script never
selects packs, views, or thresholds per benchmark.

Usage:
  bash jailbreak_repro/run_cnrf_oracle_full_eval.sh [options] [-- extra-args]

Options:
  --victim-model VALUE          Preset/model id (default: qwen25vl7b)
  --victim-backend VALUE        Backend override (default: auto)
  --victim-source VALUE         hf, modelscope, or auto
  --detector PATH               Unified victim-specific CNRF artifact
  --work PATH                   Used to infer detector when omitted
  --judge-model VALUE           Default: gemma3_12b
  --judge-backend VALUE         Default: auto
  --judge-source VALUE          Default: auto
  --judge-batch-size N          Default: 8
  --jood-max-samples N          Protocol count (default: 500)
  --attack-artifact-root PATH   Shared CS-DJ/JOOD artifacts
  --dtype VALUE                 Default: bfloat16
  --device VALUE                Default: auto
  --attn-implementation VALUE   Default: sdpa
  --force-responses             Recompute target responses
  --force-judge                 Recompute Gemma judgments
  --fail-fast                   Stop at first failure
  --dry-run                     Print commands only
EOF
}

while (($#)); do
  case "$1" in
    --victim-model) VICTIM_MODEL="$2"; shift 2 ;;
    --victim-backend) VICTIM_BACKEND="$2"; shift 2 ;;
    --victim-source) VICTIM_SOURCE="$2"; shift 2 ;;
    --detector) DETECTOR="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-backend) JUDGE_BACKEND="$2"; shift 2 ;;
    --judge-source) JUDGE_SOURCE="$2"; shift 2 ;;
    --judge-batch-size) JUDGE_BATCH_SIZE="$2"; shift 2 ;;
    --jood-max-samples) JOOD_MAX_SAMPLES="$2"; shift 2 ;;
    --attack-artifact-root) ATTACK_ARTIFACT_ROOT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --force-responses) FORCE_RESPONSES=1; shift ;;
    --force-judge) FORCE_JUDGE=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$JUDGE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--judge-batch-size must be a positive integer" >&2
  exit 2
fi
if ! [[ "$JOOD_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jood-max-samples must be a positive integer" >&2
  exit 2
fi
if [[ -z "$DETECTOR" ]]; then
  WORK_NAME="${WORK%/}"
  WORK_NAME="${WORK_NAME##*/}"
  DETECTOR="jailbreak_repro/runs/cnrf_oracle/${VICTIM_MODEL}/${WORK_NAME}/detector/cnrf_oracle_unified.npz"
fi
if ((DRY_RUN == 0)) && [[ ! -f "$DETECTOR" ]]; then
  echo "Missing unified CNRF artifact: $DETECTOR" >&2
  echo "Build it first with python -m jailbreak_repro.build_cnrf_oracle_artifact." >&2
  exit 2
fi

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

TOTAL=4
INDEX=0
FAILED=()

run_case() {
  local kind="$1"
  local name="$2"
  local label="$3"
  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.run_selected
    --victim-model "$VICTIM_MODEL"
    --victim-backend "$VICTIM_BACKEND"
    --victim-source "$VICTIM_SOURCE"
    --defense cnrf-oracle
    --representation-detector "$DETECTOR"
    --representation-action block
    --judge-model "$JUDGE_MODEL"
    --judge-backend "$JUDGE_BACKEND"
    --judge-source "$JUDGE_SOURCE"
    --judge-task auto
    --judge-batch-size "$JUDGE_BATCH_SIZE"
    --dtype "$DTYPE"
    --device "$DEVICE"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --dataset safebench
  )
  if [[ "$kind" == "attack" ]]; then
    command+=(--attack "$name" --attack-artifact-dir "$ATTACK_ARTIFACT_ROOT/$name")
    if [[ "$name" == "jood" ]]; then
      command+=(--max-samples "$JOOD_MAX_SAMPLES")
    fi
  else
    command+=(--benchmark "$name")
  fi
  ((FORCE_RESPONSES)) && command+=(--force-responses)
  ((FORCE_JUDGE)) && command+=(--force-judge)
  command+=("${EXTRA_ARGS[@]}")

  INDEX=$((INDEX + 1))
  echo
  echo "[$INDEX/$TOTAL] $label"
  print_command "${command[@]}"
  ((DRY_RUN)) && return 0
  "${command[@]}"
  local status=$?
  if ((status != 0)); then
    FAILED+=("$label (exit=$status)")
    echo "FAILED: $label (exit=$status)" >&2
    ((FAIL_FAST)) && return "$status"
  fi
  return 0
}

#run_case attack csdj CS-DJ || exit $?
#run_case attack jood JOOD || exit $?
#run_case benchmark jailbreakV-mini JailbreakV-mini || exit $?
#run_case benchmark XSTest XSTest || exit $?
run_case attack figstep FigStep || exit $?
#run_case benchmark HADES HADES || exit $?


if ((${#FAILED[@]})); then
  echo
  echo "CNRF evaluation finished with ${#FAILED[@]} failed case(s):" >&2
  printf '  - %s\n' "${FAILED[@]}" >&2
  exit 1
fi

SUMMARY_COMMAND=(
  "$PYTHON_BIN" -m jailbreak_repro.summarize_cnrf_full_eval
  --victim-tag "$VICTIM_MODEL"
  --artifact "$DETECTOR"
  --require-complete
)
echo
echo "[summary]"
print_command "${SUMMARY_COMMAND[@]}"
((DRY_RUN)) || "${SUMMARY_COMMAND[@]}"

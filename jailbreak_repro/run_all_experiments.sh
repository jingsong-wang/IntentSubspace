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
JUDGE_MODEL="qwen25vl7b"
JUDGE_BACKEND="auto"
JUDGE_SOURCE="auto"
JUDGE_BATCH_SIZE=8
DTYPE="bfloat16"
DEVICE="auto"
FIGSTEP_DATASET="safebench"
JOOD_MAX_SAMPLES=500
ATTACK_ARTIFACT_ROOT="jailbreak_repro/runs/_shared_attack_artifacts"
MAX_SAMPLES=""
CISR_DETECTOR=""
CIDER_ENCODER_MODEL="llava-hf/llava-1.5-7b-hf"
CIDER_ENCODER_SOURCE="hf"
CIDER_DIFFUSION_CHECKPOINT="jailbreak_repro/sourcecode/CIDER-main/code/models/diffusion_denoiser/imagenet/256x256_diffusion_uncond.pt"
ADASHIELD_MODE="static"
ADASHIELD_PROMPT_POOL=""
HIDDENDETECT_ACTION="monitor"
HIDDENDETECT_PROFILE=""
DRY_RUN=0
FAIL_FAST=0
FORCE_JUDGE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run the complete jailbreak reproduction matrix sequentially.

Matrix:
  attacks:    figstep, csdj, jood
  benchmarks: HADES, jailbreakV-mini, XSTest (attack=none)
  defenses:   none, ecso, cider, cisr, adashield, hiddendetect

Existing victim responses and matching judge results are reused. A changed
judge prompt changes the judge config hash and automatically triggers judging.

Usage:
  bash jailbreak_repro/run_all_experiments.sh [options] [-- extra-run-experiment-args]

Options:
  --victim-model VALUE          Preset or model id/path (default: qwen25vl7b)
  --victim-backend VALUE        auto, qwen2_5_vl, generic_vlm, ...
  --victim-source VALUE         auto, hf, or modelscope
  --judge-model VALUE           Preset or model id/path (default: qwen25vl7b)
  --judge-backend VALUE         Judge backend (default: auto)
  --judge-source VALUE          Judge model source (default: auto)
  --judge-batch-size N          Judge batch size (default: 8)
  --cisr-detector PATH          Matching CISR detector.npz
  --cider-encoder-model VALUE   Fixed auxiliary LLaVA-1.5-7B id/path
  --cider-encoder-source VALUE  hf or modelscope (default: hf)
  --cider-diffusion-checkpoint PATH
  --adashield-mode VALUE       static or adaptive (default: static)
  --adashield-prompt-pool PATH Required for adaptive AdaShield-A
  --hiddendetect-action VALUE  monitor or block (default: monitor)
  --hiddendetect-profile PATH  Optional prebuilt victim-specific profile
  --figstep-dataset VALUE       FigStep dataset alias (default: safebench)
  --jood-max-samples N          JOOD sample limit (default: 500)
  --attack-artifact-root PATH   Shared artifacts reused across defenses
  --max-samples N               Optional smoke-test limit for every run
  --dtype VALUE                 Victim/judge dtype (default: bfloat16)
  --device VALUE                Device setting (default: auto)
  --force-judge                 Recompute even when judge config hash matches
  --fail-fast                   Stop after the first failed configuration
  --dry-run                     Print all commands without executing them
  -h, --help                    Show this help

When --cisr-detector is omitted for a preset, the script tries:
  runs/CISR_v2/<victim-model>/detector/detector.npz
EOF
}

while (($#)); do
  case "$1" in
    --victim-model) VICTIM_MODEL="$2"; shift 2 ;;
    --victim-backend) VICTIM_BACKEND="$2"; shift 2 ;;
    --victim-source) VICTIM_SOURCE="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-backend) JUDGE_BACKEND="$2"; shift 2 ;;
    --judge-source) JUDGE_SOURCE="$2"; shift 2 ;;
    --judge-batch-size) JUDGE_BATCH_SIZE="$2"; shift 2 ;;
    --cisr-detector) CISR_DETECTOR="$2"; shift 2 ;;
    --cider-encoder-model) CIDER_ENCODER_MODEL="$2"; shift 2 ;;
    --cider-encoder-source) CIDER_ENCODER_SOURCE="$2"; shift 2 ;;
    --cider-diffusion-checkpoint) CIDER_DIFFUSION_CHECKPOINT="$2"; shift 2 ;;
    --adashield-mode) ADASHIELD_MODE="$2"; shift 2 ;;
    --adashield-prompt-pool) ADASHIELD_PROMPT_POOL="$2"; shift 2 ;;
    --hiddendetect-action) HIDDENDETECT_ACTION="$2"; shift 2 ;;
    --hiddendetect-profile) HIDDENDETECT_PROFILE="$2"; shift 2 ;;
    --figstep-dataset) FIGSTEP_DATASET="$2"; shift 2 ;;
    --jood-max-samples) JOOD_MAX_SAMPLES="$2"; shift 2 ;;
    --attack-artifact-root) ATTACK_ARTIFACT_ROOT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
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
if [[ -n "$MAX_SAMPLES" ]] && ! [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-samples must be a positive integer" >&2
  exit 2
fi
if [[ "$ADASHIELD_MODE" != "static" && "$ADASHIELD_MODE" != "adaptive" ]]; then
  echo "--adashield-mode must be static or adaptive" >&2
  exit 2
fi
if [[ "$HIDDENDETECT_ACTION" != "monitor" && "$HIDDENDETECT_ACTION" != "block" ]]; then
  echo "--hiddendetect-action must be monitor or block" >&2
  exit 2
fi
if [[ "$ADASHIELD_MODE" == "adaptive" && -z "$ADASHIELD_PROMPT_POOL" ]]; then
  echo "--adashield-mode adaptive requires --adashield-prompt-pool" >&2
  exit 2
fi

if [[ -z "$CISR_DETECTOR" ]]; then
  CISR_DETECTOR="runs/CISR_v2/${VICTIM_MODEL}/detector/detector.npz"
fi

if ((DRY_RUN == 0)); then
  if [[ ! -f "$CISR_DETECTOR" ]]; then
    echo "Missing CISR detector for victim '$VICTIM_MODEL': $CISR_DETECTOR" >&2
    echo "Pass --cisr-detector with a detector trained for the selected victim." >&2
    exit 2
  fi
  if [[ ! -f "$CIDER_DIFFUSION_CHECKPOINT" ]]; then
    echo "Missing CIDER diffusion checkpoint: $CIDER_DIFFUSION_CHECKPOINT" >&2
    echo "Download the official 256x256_diffusion_uncond.pt before running the full matrix." >&2
    exit 2
  fi
  if [[ "$ADASHIELD_MODE" == "adaptive" && ! -f "$ADASHIELD_PROMPT_POOL" ]]; then
    echo "AdaShield-A requires a victim-specific prompt pool: $ADASHIELD_PROMPT_POOL" >&2
    echo "Build one with python -m jailbreak_repro.build_adashield_pool." >&2
    exit 2
  fi
fi

ATTACKS=(figstep csdj jood)
BENCHMARKS=(HADES jailbreakV-mini XSTest)
DEFENSES=(adashield)
TOTAL_RUNS=$(((${#ATTACKS[@]} + ${#BENCHMARKS[@]}) * ${#DEFENSES[@]}))
RUN_INDEX=0
SUCCEEDED=()
FAILED=()

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

run_case() {
  local source_kind="$1"
  local source_name="$2"
  local defense="$3"
  local label="${source_kind}:${source_name} defense:${defense}"
  local case_max_samples="$MAX_SAMPLES"
  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.run_selected
    --victim-model "$VICTIM_MODEL"
    --victim-backend "$VICTIM_BACKEND"
    --victim-source "$VICTIM_SOURCE"
    --defense "$defense"
    --judge-model "$JUDGE_MODEL"
    --judge-backend "$JUDGE_BACKEND"
    --judge-source "$JUDGE_SOURCE"
    --judge-task auto
    --judge-batch-size "$JUDGE_BATCH_SIZE"
    --dtype "$DTYPE"
    --device "$DEVICE"
  )

  if ((FORCE_JUDGE)); then
    command+=(--force-judge)
  fi

  if [[ "$source_kind" == "attack" ]]; then
    command+=(--attack "$source_name")
    command+=(--attack-artifact-dir "$ATTACK_ARTIFACT_ROOT/$source_name")
    if [[ "$source_name" == "figstep" ]]; then
      command+=(--dataset "$FIGSTEP_DATASET")
    elif [[ "$source_name" == "jood" && -z "$case_max_samples" ]]; then
      case_max_samples="$JOOD_MAX_SAMPLES"
    fi
  else
    command+=(--benchmark "$source_name")
  fi
  if [[ "$defense" == "cisr" ]]; then
    command+=(--cisr-detector "$CISR_DETECTOR")
  elif [[ "$defense" == "cider" ]]; then
    command+=(
      --cider-encoder-mode paper_llava15
      --cider-encoder-model "$CIDER_ENCODER_MODEL"
      --cider-encoder-source "$CIDER_ENCODER_SOURCE"
      --cider-diffusion-checkpoint "$CIDER_DIFFUSION_CHECKPOINT"
    )
  elif [[ "$defense" == "adashield" ]]; then
    command+=(--adashield-mode "$ADASHIELD_MODE")
    if [[ "$ADASHIELD_MODE" == "adaptive" ]]; then
      command+=(--adashield-prompt-pool "$ADASHIELD_PROMPT_POOL")
    fi
  elif [[ "$defense" == "hiddendetect" ]]; then
    command+=(--hiddendetect-action "$HIDDENDETECT_ACTION")
    if [[ -n "$HIDDENDETECT_PROFILE" ]]; then
      command+=(--hiddendetect-profile "$HIDDENDETECT_PROFILE")
    fi
  fi
  if [[ -n "$case_max_samples" ]]; then
    command+=(--max-samples "$case_max_samples")
  fi
  command+=("${EXTRA_ARGS[@]}")

  RUN_INDEX=$((RUN_INDEX + 1))
  echo
  echo "[$RUN_INDEX/$TOTAL_RUNS] $label"
  print_command "${command[@]}"
  if ((DRY_RUN)); then
    SUCCEEDED+=("$label (dry-run)")
    return 0
  fi
  "${command[@]}"
  local status=$?
  if ((status == 0)); then
    SUCCEEDED+=("$label")
    return 0
  fi

  FAILED+=("$label (exit=$status)")
  echo "FAILED: $label (exit=$status)" >&2
  if ((FAIL_FAST)); then
    return "$status"
  fi
  return 0
}

for attack in "${ATTACKS[@]}"; do
  for defense in "${DEFENSES[@]}"; do
    run_case attack "$attack" "$defense" || exit $?
  done
done

for benchmark in "${BENCHMARKS[@]}"; do
  for defense in "${DEFENSES[@]}"; do
    run_case benchmark "$benchmark" "$defense" || exit $?
  done
done

echo
echo "Completed matrix: ${#SUCCEEDED[@]}/$TOTAL_RUNS succeeded, ${#FAILED[@]} failed."
if ((${#FAILED[@]})); then
  printf '  - %s\n' "${FAILED[@]}" >&2
  exit 1
fi

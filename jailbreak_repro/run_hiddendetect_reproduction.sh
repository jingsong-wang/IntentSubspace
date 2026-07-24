#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]] && SCRIPT_DIR="."
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/HiddenDetect_repository_repro_v2}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-benchmark}"
HIDDENDETECT_SOURCE_DIR="${HIDDENDETECT_SOURCE_DIR:-jailbreak_repro/sourcecode/HiddenDetect-main}"
CSDJ_SOURCE_DIR="${CSDJ_SOURCE_DIR:-jailbreak_repro/sourcecode/CS-DJ-main}"
CSDJ_ARTIFACT_DIR="${CSDJ_ARTIFACT_DIR:-jailbreak_repro/runs/_shared_attack_artifacts/csdj}"
XSTEST_PATH="${XSTEST_PATH:-benchmark/XSTest}"
JAILBREAKV_MINI_PATH="${JAILBREAKV_MINI_PATH:-benchmark/JailBreakV_28K/mini_JailBreakV_28K.csv}"
JOOD_SOURCE_DIR="${JOOD_SOURCE_DIR:-jailbreak_repro/sourcecode/JOOD-master}"
JOOD_DATASET_DIR="${JOOD_DATASET_DIR:-benchmark/AdvBenchM}"
JOOD_ARTIFACT_DIR="${JOOD_ARTIFACT_DIR:-jailbreak_repro/runs/_shared_attack_artifacts/jood}"
JOOD_MAX_SAMPLES="${JOOD_MAX_SAMPLES:-500}"
MODELS_CSV="${MODELS:-qwen25vl7b,gemma3_12b,llama32_11b_vision}"
SOURCES_CSV="${SOURCES:-xstest,jailbreakv-mini,csdj,jood}"
HIDDENDETECT_ACTION="${HIDDENDETECT_ACTION:-block}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma3_12b}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
PHASE="all"
DOWNLOAD_MISSING=0
FORCE=0
DRY_RUN=0
FAIL_FAST=0
MAX_SAMPLES=""
FAILURES=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Reproduce HiddenDetect only, then evaluate the frozen detector on four external suites.

Usage:
  bash jailbreak_repro/run_hiddendetect_reproduction.sh [options] [-- existing-run-experiment-args]

Options:
  --phase VALUE                 prepare, evaluate, summarize, or all (default: all)
  --models CSV                  qwen25vl7b,gemma3_12b,llama32_11b_vision
  --sources CSV                 xstest,jailbreakv-mini,csdj,jood
  --output-root DIR             Result root (default: runs/HiddenDetect_repository_repro_v2)
  --benchmark-root DIR          Shared benchmark root (default: benchmark)
  --hiddendetect-source-dir DIR Official HiddenDetect checkout/few-shot data
  --xstest-path PATH            Existing XSTest benchmark path
  --jailbreakv-mini-path PATH   Official mini_JailBreakV_28K CSV
  --csdj-source-dir DIR         Existing CS-DJ checkout used by jailbreak_repro
  --csdj-artifact-dir DIR       Existing/shared generated CS-DJ artifacts
  --jood-source-dir DIR         Existing JOOD checkout used by jailbreak_repro
  --jood-dataset-dir DIR        Existing AdvBenchM dataset root
  --jood-artifact-dir DIR       Existing/shared generated JOOD artifacts
  --jood-max-samples N|all      JOOD evaluation size (default: 500)
  --action monitor|block        HiddenDetect action (default: block)
  --max-new-tokens N            Full victim response length (default: 256)
  --judge-model MODEL           Judge preset/model, or none (default: gemma3_12b)
  --judge-batch-size N          Judge batch size (default: 8)
  --download-missing            Clone HiddenDetect or download XSTest only when absent
  --max-samples N               Optional smoke-test limit
  --dtype VALUE                 Model dtype (default: bfloat16)
  --device VALUE                Device setting (default: auto)
  --force                       Re-run completed evaluation cases
  --fail-fast                   Stop after the first failed stage
  --dry-run                     Print commands without executing them
  -h, --help                    Show this help

The 12-shot profile is frozen before all four evaluations. This runner never
prepares RCS data. CS-DJ and JOOD generation remain under their existing
jailbreak_repro adapters; adapter overrides can be passed after `--`.
EOF
}

while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --models) MODELS_CSV="$2"; shift 2 ;;
    --sources) SOURCES_CSV="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --hiddendetect-source-dir) HIDDENDETECT_SOURCE_DIR="$2"; shift 2 ;;
    --xstest-path) XSTEST_PATH="$2"; shift 2 ;;
    --jailbreakv-mini-path) JAILBREAKV_MINI_PATH="$2"; shift 2 ;;
    --csdj-source-dir) CSDJ_SOURCE_DIR="$2"; shift 2 ;;
    --csdj-artifact-dir) CSDJ_ARTIFACT_DIR="$2"; shift 2 ;;
    --jood-source-dir) JOOD_SOURCE_DIR="$2"; shift 2 ;;
    --jood-dataset-dir) JOOD_DATASET_DIR="$2"; shift 2 ;;
    --jood-artifact-dir) JOOD_ARTIFACT_DIR="$2"; shift 2 ;;
    --jood-max-samples) JOOD_MAX_SAMPLES="$2"; shift 2 ;;
    --action) HIDDENDETECT_ACTION="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-batch-size) JUDGE_BATCH_SIZE="$2"; shift 2 ;;
    --download-missing) DOWNLOAD_MISSING=1; shift ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PHASE" in
  prepare|evaluate|summarize|all) ;;
  *) echo "Unsupported --phase: $PHASE" >&2; exit 2 ;;
esac
if [[ -n "$MAX_SAMPLES" ]] && ! [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-samples must be a positive integer" >&2
  exit 2
fi
if [[ "$JOOD_MAX_SAMPLES" != "all" ]] && ! [[ "$JOOD_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jood-max-samples must be a positive integer or all" >&2
  exit 2
fi
if [[ "$HIDDENDETECT_ACTION" != "monitor" && "$HIDDENDETECT_ACTION" != "block" ]]; then
  echo "--action must be monitor or block" >&2
  exit 2
fi
if ! [[ "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-new-tokens must be a positive integer" >&2
  exit 2
fi
if ! [[ "$JUDGE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--judge-batch-size must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a SOURCES <<< "$SOURCES_CSV"
for model in "${MODELS[@]}"; do
  case "$model" in
    qwen25vl7b|gemma3_12b|llama32_11b_vision) ;;
    *) echo "Unsupported model preset: $model" >&2; exit 2 ;;
  esac
done
for source in "${SOURCES[@]}"; do
  case "$source" in
    xstest|jailbreakv-mini|csdj|jood) ;;
    *) echo "Unsupported evaluation source: $source" >&2; exit 2 ;;
  esac
done

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
  FAILURES=$((FAILURES + 1))
  if ((FAIL_FAST)); then
    exit "$status"
  fi
}

check_python_modules() {
  local label="$1"
  shift
  if ((DRY_RUN)); then
    return 0
  fi
  if "$PYTHON_BIN" -c 'import importlib.util,sys; missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; print("Missing modules: " + ", ".join(missing)) if missing else None; raise SystemExit(1 if missing else 0)' "$@"; then
    return 0
  fi
  echo "Install only the HiddenDetect environment with:" >&2
  printf '  %q -m pip install -r requirements-hiddendetect.txt\n' "$PYTHON_BIN" >&2
  return 2
}

contains_source() {
  [[ ",$SOURCES_CSV," == *",$1,"* ]]
}

prepare_data() {
  local source_root="${HIDDENDETECT_SOURCE_DIR%/*}"
  [[ "$source_root" == "$HIDDENDETECT_SOURCE_DIR" ]] && source_root="."
  local -a sync_command=(
    "$PYTHON_BIN" -m jailbreak_repro.sync_representation_upstreams
    --methods hiddendetect
    --source-root "$source_root"
    --report "$OUTPUT_ROOT/upstream.json"
    --strict
  )
  if ((DOWNLOAD_MISSING == 0)); then
    sync_command+=(--offline)
  fi
  run_command "${sync_command[@]}"
  local status=$?
  if ((status != 0)); then
    record_failure "HiddenDetect official checkout verification" "$status"
    return
  fi

  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.prepare_hiddendetect_reproduction
    --benchmark-root "$BENCHMARK_ROOT"
    --hiddendetect-source-dir "$HIDDENDETECT_SOURCE_DIR"
    --jailbreakv-mini-path "$JAILBREAKV_MINI_PATH"
    --jood-source-dir "$JOOD_SOURCE_DIR"
    --jood-dataset-dir "$JOOD_DATASET_DIR"
    --manifest "$OUTPUT_ROOT/data_manifest.json"
  )
  contains_source xstest || command+=(--skip-xstest)
  contains_source jailbreakv-mini || command+=(--skip-jailbreakv-mini)
  contains_source jood || command+=(--skip-jood)
  ((DOWNLOAD_MISSING)) && command+=(--download-missing)
  run_command "${command[@]}"
  status=$?
  ((status == 0)) || record_failure "HiddenDetect/XSTest data verification" "$status"
}

case_fingerprint() {
  local model="$1"
  local source="$2"
  local source_max="$3"
  shift 3
  "$PYTHON_BIN" -c 'import hashlib,sys; from pathlib import Path; from jailbreak_repro.hiddendetect import HIDDENDETECT_PROFILE_FORMAT,HIDDENDETECT_PROTOCOL,HIDDENDETECT_SCORE_FORMAT,load_hiddendetect_fewshot; from jailbreak_repro.judges import DEFAULT_JUDGE_SYSTEM_PROMPT,XSTEST_JUDGE_PROMPT_TEMPLATE,XSTEST_JUDGE_PROTOCOL_VERSION,XSTEST_JUDGE_SYSTEM_PROMPT,XSTEST_LABEL_ONLY_INSTRUCTION; fewshot=load_hiddendetect_fewshot(Path(sys.argv[1]))[1]; judge_payload="\0".join([DEFAULT_JUDGE_SYSTEM_PROMPT,XSTEST_JUDGE_PROTOCOL_VERSION,XSTEST_JUDGE_SYSTEM_PROMPT,XSTEST_JUDGE_PROMPT_TEMPLATE,XSTEST_LABEL_ONLY_INSTRUCTION]); judge_sha1=hashlib.sha1(judge_payload.encode()).hexdigest(); payload="\0".join([HIDDENDETECT_PROFILE_FORMAT,HIDDENDETECT_PROTOCOL,HIDDENDETECT_SCORE_FORMAT,fewshot,judge_sha1,*sys.argv[2:]]); print(hashlib.sha1(payload.encode()).hexdigest()[:12])' \
    "$HIDDENDETECT_SOURCE_DIR" "$model" "$source" "$source_max" "$DTYPE" "$DEVICE" "$@" "${EXTRA_ARGS[@]}"
}

check_model_dependencies() {
  local model="$1"
  check_python_modules "HiddenDetect model=$model" numpy torch transformers PIL tqdm || return 2
  case "$model" in
    qwen25vl7b) check_python_modules "HiddenDetect Qwen backend" qwen_vl_utils ;;
    gemma3_12b|llama32_11b_vision) check_python_modules "HiddenDetect ModelScope backend" modelscope ;;
  esac
}

check_judge_dependencies() {
  if [[ "$JUDGE_MODEL" == "none" || "$JUDGE_MODEL" == "heuristic" ]]; then
    return 0
  fi
  check_python_modules "HiddenDetect judge=$JUDGE_MODEL" numpy torch transformers PIL tqdm || return 2
  if [[ "$JUDGE_MODEL" == "gemma3_12b" ]]; then
    check_python_modules "HiddenDetect Gemma judge" modelscope
  fi
}

run_evaluation_case() {
  local model="$1"
  local source="$2"
  check_model_dependencies "$model" || {
    record_failure "dependency preflight model=$model source=$source" 2
    return
  }
  check_judge_dependencies || {
    record_failure "judge dependency preflight model=$model source=$source" 2
    return
  }

  local case_dir="$OUTPUT_ROOT/$model/hiddendetect/external/$source"
  local source_max="$MAX_SAMPLES"
  if [[ -z "$source_max" && "$source" == "jood" && "$JOOD_MAX_SAMPLES" != "all" ]]; then
    source_max="$JOOD_MAX_SAMPLES"
  fi
  local fingerprint
  fingerprint="$(case_fingerprint "$model" "$source" "${source_max:-all}" \
    "$HIDDENDETECT_ACTION" "$MAX_NEW_TOKENS" "$JUDGE_MODEL" "$JUDGE_BATCH_SIZE")" || {
    record_failure "HiddenDetect protocol fingerprint" 2
    return
  }
  local marker="$case_dir/.complete-$fingerprint"
  if ((FORCE == 0)) && [[ -f "$marker" && -f "$case_dir/summary.json" ]]; then
    echo "SKIP evaluation model=$model source=$source"
    return
  fi

  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.run_selected
    --victim-model "$model"
    --defense hiddendetect
    --judge-model "$JUDGE_MODEL"
    --judge-task auto
    --judge-batch-size "$JUDGE_BATCH_SIZE"
    --out-dir "$case_dir"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --dtype "$DTYPE"
    --device "$DEVICE"
    --hiddendetect-source-dir "$HIDDENDETECT_SOURCE_DIR"
    --hiddendetect-profile "$OUTPUT_ROOT/$model/hiddendetect/profile_v2.json"
    --hiddendetect-action "$HIDDENDETECT_ACTION"
  )
  if [[ "$source" == "xstest" ]]; then
    command+=(--benchmark "$XSTEST_PATH")
  elif [[ "$source" == "jailbreakv-mini" ]]; then
    command+=(--benchmark "$JAILBREAKV_MINI_PATH")
  elif [[ "$source" == "csdj" ]]; then
    command+=(
      --attack csdj
      --source-dir "$CSDJ_SOURCE_DIR"
      --attack-artifact-dir "$CSDJ_ARTIFACT_DIR"
    )
  else
    command+=(
      --attack jood
      --source-dir "$JOOD_SOURCE_DIR"
      --attack-artifact-dir "$JOOD_ARTIFACT_DIR"
      --jood-dataset-dir "$JOOD_DATASET_DIR"
    )
  fi
  [[ -n "$source_max" ]] && command+=(--max-samples "$source_max")
  ((FORCE)) && command+=(--force-responses --force-judge)
  command+=("${EXTRA_ARGS[@]}")

  echo "EVALUATE HiddenDetect model=$model source=$source"
  run_command "${command[@]}"
  local status=$?
  if ((status == 0)); then
    if ((DRY_RUN == 0)); then
      mkdir -p "$case_dir"
      printf '%s\n' "$fingerprint" > "$marker"
    fi
  else
    record_failure "evaluation model=$model source=$source" "$status"
  fi
}

summarize_results() {
  run_command "$PYTHON_BIN" -m jailbreak_repro.summarize_representation_reproduction --root "$OUTPUT_ROOT"
  local status=$?
  ((status == 0)) || record_failure "HiddenDetect result summary" "$status"
}

if [[ "$PHASE" == "prepare" || "$PHASE" == "all" ]]; then
  prepare_data
fi
if [[ "$PHASE" == "evaluate" || "$PHASE" == "all" ]]; then
  for model in "${MODELS[@]}"; do
    for source in "${SOURCES[@]}"; do
      run_evaluation_case "$model" "$source"
    done
  done
fi
if [[ "$PHASE" == "summarize" || "$PHASE" == "all" ]]; then
  summarize_results
fi

if ((FAILURES)); then
  echo "HiddenDetect reproduction finished with $FAILURES failed stage(s)." >&2
  exit 1
fi
echo "HiddenDetect reproduction finished successfully."

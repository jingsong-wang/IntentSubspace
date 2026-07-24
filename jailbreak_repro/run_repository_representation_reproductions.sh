#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]] && SCRIPT_DIR="."
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/representation_repository_repro}"
RCS_SOURCE_DIR="${RCS_SOURCE_DIR:-jailbreak_repro/sourcecode/Jailbreak_Detection_RCS-main}"
RCS_DATA="${RCS_DATA:-data/representation_repro/rcs_paper.jsonl}"
CISR_ACTIVATION_ROOT="${CISR_ACTIVATION_ROOT:-runs/CISR_v2}"
CSDJ_ARTIFACT_DIR="${CSDJ_ARTIFACT_DIR:-jailbreak_repro/runs/_shared_attack_artifacts/csdj}"
MODELS_CSV="${MODELS:-qwen25vl7b,gemma3_12b,llama32_11b_vision}"
METHODS_CSV="${METHODS:-hiddendetect,nearside,rcs-kcd,rcs-mcd,vlmguard}"
SOURCES_CSV="${SOURCES:-xstest,csdj}"
LAYERS="${LAYERS:-all}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
PROJECTION_DEVICE="${PROJECTION_DEVICE:-auto}"
RCS_LAYER_SELECTION_MAX_PER_CLASS="${RCS_LAYER_SELECTION_MAX_PER_CLASS:-1000}"
PHASE="all"
DOWNLOAD_DATA=0
MANUAL_DATA_ARCHIVE=""
ALLOW_INCOMPLETE_DATA=0
FORCE=0
DRY_RUN=0
FAIL_FAST=0
MAX_SAMPLES=""
FAILURES=0

usage() {
  cat <<'EOF'
Reproduce public VLM representation detectors, then evaluate frozen detectors on XSTest and CS-DJ.

Usage:
  bash jailbreak_repro/run_repository_representation_reproductions.sh [options]

Options:
  --phase VALUE          sync, download-data, prepare-data, activations, train,
                         external, summarize, or all (default: all)
  --models CSV           qwen25vl7b,gemma3_12b,llama32_11b_vision
  --methods CSV          hiddendetect,nearside,rcs-kcd,rcs-mcd,vlmguard
  --sources CSV          xstest,csdj
  --output-root DIR      Isolated result root
  --rcs-source-dir DIR   Official RCS checkout
  --rcs-data FILE        Normalized released-composition JSONL
  --cisr-activation-root DIR
                         Matched paired archives used only for NEARSIDE/VLMGuard
  --download-data        Run the official RCS Hugging Face downloader
  --manual-data-archive FILE
                         Extract the official RCS manual-data zip before normalization
  --allow-incomplete-data
                         Permit a smoke artifact explicitly marked non-paper/incomplete
  --layers VALUE         Activation layers, default all
  --max-samples N        Optional external-evaluation smoke limit
  --force                Rebuild the requested stages
  --fail-fast            Stop after the first failed stage
  --dry-run              Print commands without executing them
  -h, --help             Show this help

The released RCS composition needs the automatic datasets plus MM-Vet, FigTxt,
JailBreakV-28K, and VAE. Formal paper-rcs training is refused when counts or
referenced images are incomplete.
EOF
}

while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --models) MODELS_CSV="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --sources) SOURCES_CSV="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --rcs-source-dir) RCS_SOURCE_DIR="$2"; shift 2 ;;
    --rcs-data) RCS_DATA="$2"; shift 2 ;;
    --cisr-activation-root) CISR_ACTIVATION_ROOT="$2"; shift 2 ;;
    --download-data) DOWNLOAD_DATA=1; shift ;;
    --manual-data-archive) MANUAL_DATA_ARCHIVE="$2"; shift 2 ;;
    --allow-incomplete-data) ALLOW_INCOMPLETE_DATA=1; shift ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PHASE" in
  sync|download-data|prepare-data|activations|train|external|summarize|all) ;;
  *) echo "Unsupported --phase: $PHASE" >&2; exit 2 ;;
esac
if [[ -n "$MAX_SAMPLES" ]] && ! [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-samples must be a positive integer" >&2
  exit 2
fi
if ! [[ "$RCS_LAYER_SELECTION_MAX_PER_CLASS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RCS_LAYER_SELECTION_MAX_PER_CLASS must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
IFS=',' read -r -a SOURCES <<< "$SOURCES_CSV"

for model in "${MODELS[@]}"; do
  case "$model" in
    qwen25vl7b|gemma3_12b|llama32_11b_vision) ;;
    *) echo "Unsupported model preset: $model" >&2; exit 2 ;;
  esac
done
for method in "${METHODS[@]}"; do
  case "$method" in
    hiddendetect|nearside|rcs-kcd|rcs-mcd|vlmguard) ;;
    *) echo "Unsupported representation method: $method" >&2; exit 2 ;;
  esac
done
for source in "${SOURCES[@]}"; do
  case "$source" in
    xstest|csdj) ;;
    *) echo "Unsupported external source: $source" >&2; exit 2 ;;
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

run_in_dir() {
  local directory="$1"
  shift
  printf '(cd %q &&' "$directory"
  printf ' %q' "$@"
  printf ')\n'
  if ((DRY_RUN)); then
    return 0
  fi
  (
    cd "$directory"
    "$@"
  )
}

check_python_modules() {
  local label="$1"
  shift
  if ((DRY_RUN)); then
    return 0
  fi
  if "$PYTHON_BIN" -c 'import importlib.util,sys; missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; print(f"Python executable: {sys.executable}"); print("Missing modules: " + ", ".join(missing)) if missing else None; raise SystemExit(1 if missing else 0)' "$@"; then
    return 0
  fi
  echo "Missing Python dependencies for $label in the interpreter used by this script." >&2
  echo "Install them with the same interpreter:" >&2
  printf '  %q -m pip install -r requirements.txt\n' "$PYTHON_BIN" >&2
  return 2
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

has_method() {
  [[ ",$METHODS_CSV," == *",$1,"* ]]
}

set_model_config() {
  local model="$1"
  case "$model" in
    qwen25vl7b)
      MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
      MODEL_BACKEND="qwen2_5_vl"
      MODEL_SOURCE="hf"
      ;;
    gemma3_12b)
      MODEL_ID="google/gemma-3-12b-it"
      MODEL_BACKEND="generic_vlm"
      MODEL_SOURCE="modelscope"
      ;;
    llama32_11b_vision)
      MODEL_ID="LLM-Research/Llama-3.2-11B-Vision-Instruct"
      MODEL_BACKEND="generic_vlm"
      MODEL_SOURCE="modelscope"
      ;;
    *)
      echo "Unknown model preset: $model" >&2
      return 2
      ;;
  esac
}

upstream_methods() {
  local values=()
  has_method hiddendetect && values+=(hiddendetect)
  has_method nearside && values+=(nearside)
  if has_method rcs-kcd || has_method rcs-mcd; then values+=(rcs); fi
  has_method vlmguard && values+=(vlmguard)
  local joined=""
  local value
  for value in "${values[@]}"; do
    [[ -n "$joined" ]] && joined+="," 
    joined+="$value"
  done
  printf '%s' "$joined"
}

sync_upstreams() {
  local methods
  methods="$(upstream_methods)"
  [[ -z "$methods" ]] && return 0
  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.sync_representation_upstreams
    --methods "$methods"
    --report "$OUTPUT_ROOT/upstreams.json"
  )
  ((FORCE)) && command+=(--update)
  run_command "${command[@]}"
  local status=$?
  ((status == 0)) || record_failure "sync upstream repositories" "$status"
}

download_rcs_data() {
  check_python_modules "official RCS data download" datasets pandas pyarrow || {
    record_failure "official RCS downloader dependency preflight" 2
    return
  }
  if [[ ! -f "$RCS_SOURCE_DIR/download_datasets.py" ]]; then
    record_failure "RCS downloader missing: $RCS_SOURCE_DIR/download_datasets.py" 2
    return
  fi
  run_in_dir "$RCS_SOURCE_DIR" "$PYTHON_BIN" download_datasets.py
  local status=$?
  ((status == 0)) || record_failure "official RCS automatic data download" "$status"
  if [[ -n "$MANUAL_DATA_ARCHIVE" ]]; then
    if ((DRY_RUN == 0)) && [[ ! -f "$MANUAL_DATA_ARCHIVE" ]]; then
      record_failure "manual RCS data archive missing: $MANUAL_DATA_ARCHIVE" 2
      return
    fi
    run_command "$PYTHON_BIN" -m zipfile -e "$MANUAL_DATA_ARCHIVE" "$RCS_SOURCE_DIR"
    status=$?
    ((status == 0)) || record_failure "extract manual RCS data archive" "$status"
  fi
}

prepare_rcs_data() {
  check_python_modules "RCS data normalization" numpy pandas || {
    record_failure "RCS data normalizer dependency preflight" 2
    return
  }
  local manifest="${RCS_DATA%.*}.manifest.json"
  if ((FORCE == 0)) && [[ -f "$RCS_DATA" && -f "$manifest" ]]; then
    if ((DOWNLOAD_DATA == 0)) && [[ -z "$MANUAL_DATA_ARCHIVE" ]]; then
      if ((ALLOW_INCOMPLETE_DATA)) || "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("exact_protocol") is True else 1)' "$manifest"; then
        echo "SKIP prepare-data: $RCS_DATA"
        return
      fi
      echo "REBUILD prepare-data: existing manifest is incomplete"
    fi
  fi
  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.prepare_representation_data
    --protocol rcs-paper
    --source-dir "$RCS_SOURCE_DIR"
    --out "$RCS_DATA"
    --manifest-out "$manifest"
  )
  ((ALLOW_INCOMPLETE_DATA)) && command+=(--allow-incomplete)
  run_command "${command[@]}"
  local status=$?
  ((status == 0)) || record_failure "normalize released RCS data composition" "$status"
}

needs_rcs_activations() {
  has_method rcs-kcd || has_method rcs-mcd
}

activation_matches_data() {
  local activation="$1"
  local manifest="${RCS_DATA%.*}.manifest.json"
  "$PYTHON_BIN" -c 'import json,sys,numpy as np; a=np.load(sys.argv[1], allow_pickle=True); m=json.loads(str(a["metadata_json"].item())); d=json.load(open(sys.argv[2], encoding="utf-8")); ok=m.get("dataset_manifest_sha1")==d.get("logical_fingerprint") and m.get("dataset_protocol")==d.get("dataset_protocol"); raise SystemExit(0 if ok else 1)' "$activation" "$manifest"
}

artifact_matches_activation() {
  local artifact="$1"
  local activation="$2"
  local protocol="$3"
  "$PYTHON_BIN" -c 'import sys; from pathlib import Path; from jailbreak_repro.representation_detectors import RepresentationDetector; from jailbreak_repro.train_representation_detector import activation_archive_logical_fingerprint; d=RepresentationDetector.load(Path(sys.argv[1])); fingerprint=activation_archive_logical_fingerprint(Path(sys.argv[2])); ok=d.metadata.get("source_logical_fingerprint")==fingerprint and d.metadata.get("protocol")==sys.argv[3]; raise SystemExit(0 if ok else 1)' "$artifact" "$activation" "$protocol"
}

extract_model_activations() {
  local model="$1"
  set_model_config "$model" || {
    record_failure "activation config model=$model" 2
    return
  }
  check_python_modules "activation extraction" numpy torch transformers PIL || {
    record_failure "activation dependency preflight model=$model" 2
    return
  }
  local output="$OUTPUT_ROOT/$model/rcs-data/activations_all_layers.npz"
  if ((FORCE == 0)) && [[ -f "$output" ]]; then
    if activation_matches_data "$output"; then
      echo "SKIP activations model=$model"
      return
    fi
    echo "REBUILD activations model=$model: data fingerprint changed"
  fi
  if ((DRY_RUN == 0)) && [[ ! -f "$RCS_DATA" ]]; then
    record_failure "activations model=$model: missing $RCS_DATA" 2
    return
  fi
  run_command "$PYTHON_BIN" -m src.extract_activations \
    --model "$MODEL_ID" \
    --model-source "$MODEL_SOURCE" \
    --backend "$MODEL_BACKEND" \
    --data "$RCS_DATA" \
    --out "$output" \
    --layers "$LAYERS" \
    --pooling last \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --image-base-dir .
  local status=$?
  ((status == 0)) || record_failure "activations model=$model" "$status"
}

data_protocol() {
  local manifest="${RCS_DATA%.*}.manifest.json"
  if ((DRY_RUN)); then
    if ((ALLOW_INCOMPLETE_DATA)); then printf '%s' repository-rcs-incomplete; else printf '%s' paper-rcs; fi
    return
  fi
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["dataset_protocol"])' "$manifest"
}

train_detector() {
  local model="$1"
  local method="$2"
  [[ "$method" == "hiddendetect" ]] && return
  check_python_modules "representation detector training" numpy torch sklearn scipy || {
    record_failure "training dependency preflight model=$model method=$method" 2
    return
  }
  local method_dir="$OUTPUT_ROOT/$model/$method"
  local artifact="$method_dir/detector.npz"
  local summary="$method_dir/detection_summary.json"
  local activations protocol
  local -a extras=()
  if [[ "$method" == "rcs-kcd" || "$method" == "rcs-mcd" ]]; then
    activations="$OUTPUT_ROOT/$model/rcs-data/activations_all_layers.npz"
    protocol="$(data_protocol)" || {
      record_failure "read RCS data protocol" 2
      return
    }
    extras+=(
      --dataset-field source
      --rcs-layer-selection official-composite
      --layer-selection-max-per-class "$RCS_LAYER_SELECTION_MAX_PER_CLASS"
      --projection-device "$PROJECTION_DEVICE"
    )
  else
    activations="$CISR_ACTIVATION_ROOT/$model/activations_all_layers.npz"
    protocol="matched-cisr"
    extras+=(--projection-device "$PROJECTION_DEVICE")
  fi
  if ((DRY_RUN == 0)) && [[ ! -f "$activations" ]]; then
    record_failure "train model=$model method=$method: missing $activations" 2
    return
  fi
  if ((DRY_RUN == 0)) && [[ "$method" == "rcs-kcd" || "$method" == "rcs-mcd" ]] && ! activation_matches_data "$activations"; then
    record_failure "train model=$model method=$method: activation/data fingerprint mismatch" 2
    return
  fi
  if ((FORCE == 0)) && [[ -f "$artifact" && -f "$summary" ]] && artifact_matches_activation "$artifact" "$activations" "$protocol"; then
    echo "SKIP train model=$model method=$method"
    return
  fi
  echo "TRAIN model=$model method=$method protocol=$protocol"
  run_command "$PYTHON_BIN" -m jailbreak_repro.train_representation_detector \
    --activations "$activations" \
    --method "$method" \
    --out "$artifact" \
    --output-dir "$method_dir" \
    --protocol "$protocol" \
    "${extras[@]}"
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
  check_python_modules "external detector evaluation" numpy torch transformers PIL || {
    record_failure "external dependency preflight model=$model method=$method source=$source" 2
    return
  }
  local case_dir="$OUTPUT_ROOT/$model/$method/external/$source"
  local fingerprint marker artifact=""
  local -a defense_args=()
  if [[ "$method" == "hiddendetect" ]]; then
    fingerprint="official-fewshot"
    if ((DRY_RUN == 0)); then
      fingerprint="$("$PYTHON_BIN" -c 'from jailbreak_repro.hiddendetect import default_hiddendetect_source_dir,load_hiddendetect_fewshot; print(load_hiddendetect_fewshot(default_hiddendetect_source_dir())[1][:12])')" || {
        record_failure "HiddenDetect few-shot fingerprint model=$model" 2
        return
      }
    fi
    marker="$case_dir/.complete-$fingerprint"
    defense_args+=(
      --hiddendetect-action monitor
      --hiddendetect-profile "$OUTPUT_ROOT/$model/hiddendetect/profile.json"
    )
  else
    artifact="$OUTPUT_ROOT/$model/$method/detector.npz"
    if ((DRY_RUN == 0)) && [[ ! -f "$artifact" ]]; then
      record_failure "external model=$model method=$method source=$source: missing artifact" 2
      return
    fi
    if ((DRY_RUN == 0)); then
      local expected_activations expected_protocol
      if [[ "$method" == "rcs-kcd" || "$method" == "rcs-mcd" ]]; then
        expected_activations="$OUTPUT_ROOT/$model/rcs-data/activations_all_layers.npz"
        expected_protocol="$(data_protocol)" || {
          record_failure "read RCS protocol for external model=$model method=$method" 2
          return
        }
      else
        expected_activations="$CISR_ACTIVATION_ROOT/$model/activations_all_layers.npz"
        expected_protocol="matched-cisr"
      fi
      if [[ ! -f "$expected_activations" ]] || ! artifact_matches_activation "$artifact" "$expected_activations" "$expected_protocol"; then
        record_failure "external model=$model method=$method source=$source: stale artifact" 2
        return
      fi
    fi
    fingerprint="dryrun"
    if ((DRY_RUN == 0)); then
      fingerprint="$(artifact_sha1 "$artifact")" || {
        record_failure "fingerprint model=$model method=$method" 2
        return
      }
    fi
    marker="$case_dir/.complete-$fingerprint"
    defense_args+=(
      --representation-detector "$artifact"
      --representation-action monitor
    )
  fi
  if ((FORCE == 0)) && [[ -f "$marker" && -f "$case_dir/summary.json" ]]; then
    echo "SKIP external model=$model method=$method source=$source"
    return
  fi
  local -a command=(
    "$PYTHON_BIN" -m jailbreak_repro.run_selected
    --victim-model "$model"
    --defense "$method"
    --judge-model none
    --out-dir "$case_dir"
    --max-new-tokens 1
    --dtype "$DTYPE"
    --device "$DEVICE"
    "${defense_args[@]}"
  )
  if [[ "$source" == "xstest" ]]; then
    command+=(--benchmark XSTest)
  elif [[ "$source" == "csdj" ]]; then
    command+=(--attack csdj --attack-artifact-dir "$CSDJ_ARTIFACT_DIR")
  else
    record_failure "unsupported external source=$source" 2
    return
  fi
  [[ -n "$MAX_SAMPLES" ]] && command+=(--max-samples "$MAX_SAMPLES")
  if ((FORCE)) || [[ -f "$case_dir/summary.json" ]]; then
    command+=(--force-responses)
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

summarize_results() {
  run_command "$PYTHON_BIN" -m jailbreak_repro.summarize_representation_reproduction \
    --root "$OUTPUT_ROOT"
  local status=$?
  ((status == 0)) || record_failure "summarize external detector results" "$status"
}

if [[ "$PHASE" == "sync" || "$PHASE" == "all" ]]; then
  sync_upstreams
fi
if [[ "$PHASE" == "download-data" || ( "$PHASE" == "all" && "$DOWNLOAD_DATA" == "1" ) ]]; then
  download_rcs_data
fi
if [[ "$PHASE" == "prepare-data" || "$PHASE" == "all" ]]; then
  if needs_rcs_activations; then prepare_rcs_data; fi
fi
if [[ "$PHASE" == "activations" || "$PHASE" == "all" ]]; then
  if needs_rcs_activations; then
    for model in "${MODELS[@]}"; do extract_model_activations "$model"; done
  fi
fi
if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do train_detector "$model" "$method"; done
  done
fi
if [[ "$PHASE" == "external" || "$PHASE" == "all" ]]; then
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for source in "${SOURCES[@]}"; do run_external_case "$model" "$method" "$source"; done
    done
  done
fi
if [[ "$PHASE" == "summarize" || "$PHASE" == "all" ]]; then
  summarize_results
fi

if ((FAILURES)); then
  echo "Repository representation reproduction finished with $FAILURES failed stage(s)." >&2
  exit 1
fi
echo "Repository representation reproduction finished successfully."

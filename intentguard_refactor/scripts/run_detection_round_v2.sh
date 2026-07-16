#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
BASE_RUN_DIR="${BASE_RUN_DIR:-runs/CISR_v2}"
DATA="${DATA:-data/CISR_v2_probe.jsonl}"
DATA_SUMMARY="${DATA_SUMMARY:-data/CISR_v2_probe_summary.json}"
CONFIG="${CONFIG:-intentguard_refactor/configs/cisr_v2_families.json}"

DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-8}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
JUDGE_MODEL_CACHE_DIR="${JUDGE_MODEL_CACHE_DIR:-${MODEL_CACHE_DIR}}"
JUDGE_MODEL_REVISION="${JUDGE_MODEL_REVISION:-}"

JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-3-12b-it}"
JUDGE_BACKEND="${JUDGE_BACKEND:-generic_vlm}"
JUDGE_MODEL_SOURCE="${JUDGE_MODEL_SOURCE:-modelscope}"
JUDGE_INCLUDE_IMAGE="${JUDGE_INCLUDE_IMAGE:-1}"

INTENT_TARGET_TPR="${INTENT_TARGET_TPR:-0.95}"
INTENT_TARGET_FPR="${INTENT_TARGET_FPR:-0.10}"
COVERAGE_CONFIDENCE="${COVERAGE_CONFIDENCE:-0.95}"
HARD_POSITIVE_WEIGHT="${HARD_POSITIVE_WEIGHT:-2.0}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-8}"
MLP_EPOCHS="${MLP_EPOCHS:-1000}"
LAYER_CANDIDATES="${LAYER_CANDIDATES:-6}"

REBUILD_DATA="${REBUILD_DATA:-0}"
FORCE="${FORCE:-0}"
RUN_DATA="${RUN_DATA:-1}"
RUN_ACTIVATIONS="${RUN_ACTIVATIONS:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_DETECTOR="${RUN_DETECTOR:-1}"

# alias|model_id_or_local_path|backend|source
# MODEL_SPECS="${MODEL_SPECS:-qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|hf;gemma3_12b|google/gemma-3-12b-it|generic_vlm|modelscope;llama32_11b_vision|LLM-Research/Llama-3.2-11B-Vision-Instruct|generic_vlm|modelscope}"
MODEL_SPECS="${MODEL_SPECS:-llava15_7b|llava-hf/llava-1.5-7b-hf|generic_vlm|hf}"
cd "$ROOT"

TRUST_ARGS=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

MAX_SAMPLE_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "$MAX_SAMPLES")
fi

JUDGE_IMAGE_ARGS=()
if [[ "$JUDGE_INCLUDE_IMAGE" == "1" ]]; then
  JUDGE_IMAGE_ARGS=(--include-image)
fi

source_args() {
  local source="$1"
  local cache_dir="$2"
  local revision="$3"
  local -n out_ref="$4"
  out_ref=(--model-source "$source")
  if [[ -n "$revision" ]]; then
    out_ref+=(--model-revision "$revision")
  fi
  if [[ -n "$cache_dir" ]]; then
    out_ref+=(--model-cache-dir "$cache_dir")
  fi
}

should_run() {
  local output="$1"
  if [[ "$FORCE" == "1" ]]; then
    return 0
  fi
  [[ ! -s "$output" ]]
}

activation_cache_is_valid() {
  local archive="$1"
  local dataset="$2"
  local model="$3"
  local backend="$4"
  python intentguard_refactor/validate_activation_archive.py \
    --activations "$archive" \
    --data "$dataset" \
    --model "$model" \
    --backend "$backend" \
    --require-multimodal-anchor \
    --quiet
}

if [[ "$RUN_DATA" == "1" ]]; then
  if [[ "$REBUILD_DATA" == "1" || "$FORCE" == "1" || ! -s "$DATA" || ! -s "$DATA_SUMMARY" ]]; then
    echo "[data] Building paired CISR_v2 train/validation/calibration/test probe"
    python intentguard_refactor/make_data.py \
      --config "$CONFIG" \
      --out "$DATA" \
      --summary-out "$DATA_SUMMARY" \
      --repo-root .
  else
    echo "[data] Reusing $DATA"
  fi
fi

judge_args=()
source_args "$JUDGE_MODEL_SOURCE" "$JUDGE_MODEL_CACHE_DIR" "$JUDGE_MODEL_REVISION" judge_args

IFS=';' read -ra SPECS <<< "$MODEL_SPECS"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r alias model backend source extra <<< "$spec"
  if [[ -n "${extra:-}" || -z "${alias:-}" || -z "${model:-}" || -z "${backend:-}" ]]; then
    echo "Invalid MODEL_SPECS item: $spec" >&2
    exit 1
  fi
  source="${source:-hf}"
  run_dir="$BASE_RUN_DIR/$alias"
  activations="$run_dir/activations_all_layers.npz"
  generation_dir="$run_dir/original_generations"
  generation_results="$generation_dir/generation_results.jsonl"
  judge_results="$run_dir/original_judge/judge_results.jsonl"
  detector_dir="$run_dir/detector"
  detector_artifact="$detector_dir/detector.npz"
  detection_summary="$detector_dir/detection_summary.json"
  detection_results="$detector_dir/detection_results.jsonl"
  mkdir -p "$run_dir"

  model_args=()
  source_args "$source" "$MODEL_CACHE_DIR" "$MODEL_REVISION" model_args
  echo "========== [$alias] CISR_v2 model=$model backend=$backend source=$source =========="

  activations_rebuilt=0
  if [[ "$RUN_ACTIVATIONS" == "1" ]]; then
    if should_run "$activations" || ! activation_cache_is_valid "$activations" "$DATA" "$model" "$backend"; then
      echo "[1/4:$alias] Extracting all layers plus multimodal anchors"
      python src/extract_activations.py \
        --model "$model" \
        "${model_args[@]}" \
        --backend "$backend" \
        --data "$DATA" \
        --out "$activations" \
        --layers all \
        --pooling last \
        --multimodal-anchor \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        --image-base-dir . \
        "${TRUST_ARGS[@]}"
      activations_rebuilt=1
    else
      echo "[1/4:$alias] Reusing $activations"
    fi
  fi

  if [[ "$RUN_GENERATION" == "1" ]]; then
    if should_run "$generation_results"; then
      echo "[2/4:$alias] Generating original responses"
      python src/run_probe_generations.py \
        --model "$model" \
        --model-alias "$alias" \
        "${model_args[@]}" \
        --backend "$backend" \
        --data "$DATA" \
        --out-dir "$generation_dir" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        --image-base-dir . \
        "${MAX_SAMPLE_ARGS[@]}" \
        "${TRUST_ARGS[@]}"
    else
      echo "[2/4:$alias] Reusing $generation_results"
    fi
  fi

  if [[ "$RUN_JUDGE" == "1" ]]; then
    if should_run "$judge_results"; then
      echo "[3/4:$alias] Judging original responses with aligned CISR judge"
      python intentguard_refactor/judge_outputs.py \
        --model "$JUDGE_MODEL" \
        "${judge_args[@]}" \
        --backend "$JUDGE_BACKEND" \
        --input "$generation_results" \
        --out "$judge_results" \
        --response-field response \
        --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
        --batch-size "$JUDGE_BATCH_SIZE" \
        --image-base-dir . \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        "${JUDGE_IMAGE_ARGS[@]}" \
        "${TRUST_ARGS[@]}"
    else
      echo "[3/4:$alias] Reusing $judge_results"
    fi
  fi

  if [[ "$RUN_DETECTOR" == "1" ]]; then
    if [[ "$FORCE" == "1" || "$activations_rebuilt" == "1" || ! -s "$detector_artifact" || ! -s "$detection_summary" || ! -s "$detection_results" ]]; then
      echo "[4/4:$alias] Training rank-3 detector and evaluating held-out test templates"
      python intentguard_refactor/train_detector.py \
        --activations "$activations" \
        --data "$DATA" \
        --response-labels "$judge_results" \
        --out-dir "$detector_dir" \
        --model-alias "$alias" \
        --rank 3 \
        --hidden-dim "$MLP_HIDDEN_DIM" \
        --epochs "$MLP_EPOCHS" \
        --layer-candidates "$LAYER_CANDIDATES" \
        --target-tpr "$INTENT_TARGET_TPR" \
        --target-fpr "$INTENT_TARGET_FPR" \
        --coverage-confidence "$COVERAGE_CONFIDENCE" \
        --hard-positive-weight "$HARD_POSITIVE_WEIGHT"
    else
      echo "[4/4:$alias] Reusing $detector_artifact"
    fi
  fi

  echo "[$alias] Detector: $detector_artifact"
done

echo "CISR_v2 detection round complete under $BASE_RUN_DIR"

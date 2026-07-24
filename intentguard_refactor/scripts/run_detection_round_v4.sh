#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_RUN_DIR="${BASE_RUN_DIR:-runs/CISR_v4_modal}"
DATA="${DATA:-data/CISR_v4_probe.jsonl}"
TEXT_DATA="${TEXT_DATA:-data/CISR_v4_text_probe.jsonl}"
MULTIMODAL_DATA="${MULTIMODAL_DATA:-data/CISR_v4_multimodal_probe.jsonl}"
DATA_SUMMARY="${DATA_SUMMARY:-data/CISR_v4_probe_summary.json}"
CONFIG="${CONFIG:-intentguard_refactor/configs/cisr_v4_families.json}"

TEXT_POOLINGS="${TEXT_POOLINGS:-last;non_image_mean}"
MULTIMODAL_POOLINGS="${MULTIMODAL_POOLINGS:-last;image_mean;non_image_mean}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
MODEL_REVISION="${MODEL_REVISION:-}"

TARGET_TPR="${TARGET_TPR:-0.90}"
TARGET_FPR="${TARGET_FPR:-0.10}"
HARD_BENIGN_TARGET_FPR="${HARD_BENIGN_TARGET_FPR:-0.10}"
COVERAGE_CONFIDENCE="${COVERAGE_CONFIDENCE:-0.95}"
HARD_BENIGN_WEIGHT="${HARD_BENIGN_WEIGHT:-1.5}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.25}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-8}"
MLP_EPOCHS="${MLP_EPOCHS:-1000}"
LAYER_CANDIDATES="${LAYER_CANDIDATES:-8}"

MAX_SAFE_ERROR="${MAX_SAFE_ERROR:-0.02}"
MAX_DANGER_ERROR="${MAX_DANGER_ERROR:-0.02}"
MAX_HARMFUL_ESCAPE="${MAX_HARMFUL_ESCAPE:-0.05}"
MAX_BENIGN_HARD_REFUSAL="${MAX_BENIGN_HARD_REFUSAL:-0.05}"
MINIMUM_GROUP_SELECTED="${MINIMUM_GROUP_SELECTED:-10}"

RUN_DATA="${RUN_DATA:-1}"
RUN_ACTIVATIONS="${RUN_ACTIVATIONS:-1}"
RUN_DETECTOR="${RUN_DETECTOR:-1}"
RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
RUN_BUNDLE="${RUN_BUNDLE:-1}"
FORCE="${FORCE:-0}"
REBUILD_DATA="${REBUILD_DATA:-0}"
REQUIRE_DEPLOYABLE="${REQUIRE_DEPLOYABLE:-0}"
ALLOW_UNSUPPORTED_IMAGE_MEAN_SKIP="${ALLOW_UNSUPPORTED_IMAGE_MEAN_SKIP:-1}"

# alias|model_id_or_local_path|backend|source
MODEL_SPECS="${MODEL_SPECS:-qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|modelscope;gemma3_12b|google/gemma-3-12b-it|generic_vlm|modelscope}"

cd "$ROOT"

trust_args=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  trust_args=(--trust-remote-code)
fi

deployable_args=()
if [[ "$REQUIRE_DEPLOYABLE" == "1" ]]; then
  deployable_args=(--require-deployable)
fi

source_args() {
  local source="$1"
  local -n output_ref="$2"
  output_ref=(--model-source "$source")
  if [[ -n "$MODEL_REVISION" ]]; then
    output_ref+=(--model-revision "$MODEL_REVISION")
  fi
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    output_ref+=(--model-cache-dir "$MODEL_CACHE_DIR")
  fi
}

should_run() {
  local output="$1"
  [[ "$FORCE" == "1" || ! -s "$output" ]]
}

activation_cache_is_valid() {
  local archive="$1"
  local dataset="$2"
  local model="$3"
  local backend="$4"
  local pooling="$5"
  "$PYTHON_BIN" intentguard_refactor/validate_activation_archive.py \
    --activations "$archive" \
    --data "$dataset" \
    --model "$model" \
    --backend "$backend" \
    --pooling "$pooling" \
    --quiet
}

if [[ "$RUN_DATA" == "1" ]]; then
  if [[ "$REBUILD_DATA" == "1" || "$FORCE" == "1" || ! -s "$DATA" || ! -s "$TEXT_DATA" || ! -s "$MULTIMODAL_DATA" || ! -s "$DATA_SUMMARY" ]]; then
    echo "[data] Building modal-isolated CISR_v4 data with paired FigStep-style OCR"
    "$PYTHON_BIN" intentguard_refactor/make_data_v4.py \
      --config "$CONFIG" \
      --out "$DATA" \
      --text-out "$TEXT_DATA" \
      --multimodal-out "$MULTIMODAL_DATA" \
      --summary-out "$DATA_SUMMARY" \
      --repo-root .
  else
    echo "[data] Reusing CISR_v4 manifests"
  fi
fi

IFS=';' read -ra specs <<< "$MODEL_SPECS"
for spec in "${specs[@]}"; do
  IFS='|' read -r alias model backend source <<< "$spec"
  if [[ -z "$alias" || -z "$model" || -z "$backend" || -z "$source" ]]; then
    echo "Invalid MODEL_SPECS entry: $spec" >&2
    exit 1
  fi
  run_dir="$BASE_RUN_DIR/$alias"
  bundle_dir="$run_dir/detector"
  bundle_manifest="$bundle_dir/detector_bundle.json"
  model_args=()
  source_args "$source" model_args
  candidate_args=()

  for branch in text multimodal; do
    if [[ "$branch" == "text" ]]; then
      branch_data="$TEXT_DATA"
      pooling_spec="$TEXT_POOLINGS"
    else
      branch_data="$MULTIMODAL_DATA"
      pooling_spec="$MULTIMODAL_POOLINGS"
    fi
    IFS=';' read -ra poolings <<< "$pooling_spec"
    for pooling in "${poolings[@]}"; do
      candidate_dir="$run_dir/candidates/$branch/$pooling"
      activations="$candidate_dir/activations_all_layers.npz"
      source_detector_dir="$candidate_dir/source_detector"
      source_detector="$source_detector_dir/detector.npz"
      source_summary="$source_detector_dir/detection_summary.json"
      source_results="$source_detector_dir/detection_results.jsonl"
      calibrated_dir="$candidate_dir/calibrated_detector"
      calibrated_detector="$calibrated_dir/detector.npz"

      if [[ "$RUN_ACTIVATIONS" == "1" ]]; then
        cache_valid=0
        if [[ -s "$activations" ]] && activation_cache_is_valid "$activations" "$branch_data" "$model" "$backend" "$pooling"; then
          cache_valid=1
        fi
        if [[ "$FORCE" == "1" || "$cache_valid" != "1" ]]; then
          echo "[extract:$alias:$branch:$pooling]"
          if ! "$PYTHON_BIN" src/extract_activations.py \
            --model "$model" \
            "${model_args[@]}" \
            --backend "$backend" \
            --data "$branch_data" \
            --out "$activations" \
            --layers all \
            --pooling "$pooling" \
            --dtype "$DTYPE" \
            --device "$DEVICE" \
            --image-base-dir . \
            "${trust_args[@]}"; then
            if [[ "$pooling" == "image_mean" && "$ALLOW_UNSUPPORTED_IMAGE_MEAN_SKIP" == "1" ]]; then
              echo "[skip:$alias:$branch:$pooling] Backend exposes no aligned image tokens or failed image-mean extraction."
              continue
            fi
            exit 1
          fi
        else
          echo "[reuse:$alias:$branch:$pooling] $activations"
        fi
      fi

      if [[ ! -s "$activations" ]] || ! activation_cache_is_valid "$activations" "$branch_data" "$model" "$backend" "$pooling"; then
        echo "[skip:$alias:$branch:$pooling] Missing or incompatible activations"
        continue
      fi

      if [[ "$RUN_DETECTOR" == "1" ]] && { should_run "$source_detector" || should_run "$source_summary" || should_run "$source_results"; }; then
        echo "[train:$alias:$branch:$pooling] Selecting layer and fitting rank-3 detector"
        "$PYTHON_BIN" intentguard_refactor/train_detector_v3.py \
          --activations "$activations" \
          --data "$branch_data" \
          --out-dir "$source_detector_dir" \
          --model-alias "$alias-$branch-$pooling" \
          --rank 3 \
          --hidden-dim "$MLP_HIDDEN_DIM" \
          --epochs "$MLP_EPOCHS" \
          --layer-candidates "$LAYER_CANDIDATES" \
          --target-tpr "$TARGET_TPR" \
          --target-fpr "$TARGET_FPR" \
          --hard-benign-target-fpr "$HARD_BENIGN_TARGET_FPR" \
          --coverage-confidence "$COVERAGE_CONFIDENCE" \
          --hard-benign-weight "$HARD_BENIGN_WEIGHT" \
          --consistency-weight "$CONSISTENCY_WEIGHT" \
          "${deployable_args[@]}"
      fi

      if [[ "$RUN_CALIBRATION" == "1" ]] && should_run "$calibrated_detector"; then
        echo "[calibrate:$alias:$branch:$pooling] Building selective v4 thresholds"
        "$PYTHON_BIN" intentguard_refactor/calibrate_detector_v4.py \
          --source-detector "$source_detector" \
          --detection-results "$source_results" \
          --out-dir "$calibrated_dir" \
          --maximum-confident-safe-error "$MAX_SAFE_ERROR" \
          --maximum-confident-dangerous-error "$MAX_DANGER_ERROR" \
          --maximum-harmful-unsafe-escape "$MAX_HARMFUL_ESCAPE" \
          --maximum-benign-hard-refusal "$MAX_BENIGN_HARD_REFUSAL" \
          --coverage-confidence "$COVERAGE_CONFIDENCE" \
          --constraint-group carrier_type \
          --minimum-group-selected "$MINIMUM_GROUP_SELECTED" \
          "${deployable_args[@]}"
      fi

      if [[ -s "$source_summary" && -s "$calibrated_detector" ]]; then
        candidate_args+=(--candidate "$branch" "$pooling" "$source_summary" "$calibrated_detector")
      fi
    done
  done

  if [[ "$RUN_BUNDLE" == "1" ]]; then
    if [[ "$FORCE" == "1" || ! -s "$bundle_manifest" ]]; then
      echo "[bundle:$alias] Selecting validation-best pooling independently by modality"
      "$PYTHON_BIN" intentguard_refactor/build_detector_bundle_v4.py \
        "${candidate_args[@]}" \
        --out-dir "$bundle_dir" \
        --model-alias "$alias"
    else
      echo "[reuse:$alias] $bundle_manifest"
    fi
  fi
  echo "[done:$alias] $bundle_manifest"
done

echo "CISR_v4 modal detector training complete under $BASE_RUN_DIR"

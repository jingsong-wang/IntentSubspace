#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
BASE_RUN_DIR="${BASE_RUN_DIR:-runs/CISR}"
DATA="${DATA:-data/CISR_probe.jsonl}"
DATA_SUMMARY="${DATA_SUMMARY:-data/CISR_probe_summary.json}"
CONFIG="${CONFIG:-intentguard_refactor/configs/intentguard_families.json}"

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

REBUILD_DATA="${REBUILD_DATA:-0}"
FORCE="${FORCE:-0}"
RUN_DATA="${RUN_DATA:-1}"
RUN_ACTIVATIONS="${RUN_ACTIVATIONS:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_ORIGINAL_JUDGE="${RUN_ORIGINAL_JUDGE:-1}"
RUN_SUBSPACES="${RUN_SUBSPACES:-1}"
RUN_THRESHOLDS="${RUN_THRESHOLDS:-1}"
RUN_INTERVENTION="${RUN_INTERVENTION:-1}"
RUN_POST_JUDGE="${RUN_POST_JUDGE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"

# Format: alias|model_id_or_local_path|backend|source
# Also accepts the older 5-field format: alias|model|backend|layer|source.
# The defaults match the current target models. Gemma and Llama use ModelScope.
MODEL_SPECS="${MODEL_SPECS:-qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|hf;gemma3_12b|google/gemma-3-12b-it|generic_vlm|modelscope;llama32_11b_vision|LLM-Research/Llama-3.2-11B-Vision-Instruct|generic_vlm|modelscope}"

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

model_source_args() {
  local source="$1"
  local -n out_ref="$2"
  out_ref=(--model-source "$source")
  if [[ -n "$MODEL_REVISION" ]]; then
    out_ref+=(--model-revision "$MODEL_REVISION")
  fi
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    out_ref+=(--model-cache-dir "$MODEL_CACHE_DIR")
  fi
}

judge_source_args() {
  local -n out_ref="$1"
  out_ref=(--model-source "$JUDGE_MODEL_SOURCE")
  if [[ -n "$JUDGE_MODEL_REVISION" ]]; then
    out_ref+=(--model-revision "$JUDGE_MODEL_REVISION")
  fi
  if [[ -n "$JUDGE_MODEL_CACHE_DIR" ]]; then
    out_ref+=(--model-cache-dir "$JUDGE_MODEL_CACHE_DIR")
  fi
}

should_run_file() {
  local output="$1"
  if [[ "$FORCE" == "1" ]]; then
    return 0
  fi
  [[ ! -s "$output" ]]
}

should_run_dir_file() {
  local output="$1"
  should_run_file "$output"
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
    --quiet
}

if [[ "$RUN_DATA" == "1" ]]; then
  if [[ "$REBUILD_DATA" == "1" || "$FORCE" == "1" || ! -s "$DATA" || ! -s "$DATA_SUMMARY" ]]; then
    echo "[data] Regenerating counterfactual multimodal data"
    python intentguard_refactor/make_data.py \
      --config "$CONFIG" \
      --out "$DATA" \
      --summary-out "$DATA_SUMMARY" \
      --repo-root .
  else
    echo "[data] Reusing existing $DATA"
  fi
fi

judge_args=()
judge_source_args judge_args

IFS=';' read -ra SPECS <<< "$MODEL_SPECS"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r alias model backend field4 field5 extra <<< "$spec"
  if [[ -n "${extra:-}" ]]; then
    echo "Invalid MODEL_SPECS item with too many fields: $spec" >&2
    exit 1
  fi
  if [[ -n "${field5:-}" ]]; then
    source="$field5"
  else
    source="${field4:-hf}"
  fi
  if [[ -z "${alias:-}" || -z "${model:-}" || -z "${backend:-}" ]]; then
    echo "Invalid MODEL_SPECS item: $spec" >&2
    exit 1
  fi

  run_dir="$BASE_RUN_DIR/$alias"
  mkdir -p "$run_dir"

  activations="$run_dir/activations_all_layers.npz"
  gen_dir="$run_dir/original_generations"
  gen_results="$gen_dir/generation_results.jsonl"
  original_judge="$run_dir/original_judge/judge_results.jsonl"
  intent_subspace="$run_dir/subspaces/intent_subspace.npz"
  refusal_subspace="$run_dir/subspaces/refusal_subspace.npz"
  subspace_selection="$run_dir/subspaces/subspace_selection.json"
  thresholds="$run_dir/thresholds.json"
  intervention_results="$run_dir/intervention_results.jsonl"
  post_judge="$run_dir/post_judge/judge_results.jsonl"
  audit_jsonl="$run_dir/sample_audit.jsonl"
  audit_csv="$run_dir/sample_audit.csv"
  audit_summary="$run_dir/sample_audit_summary.json"

  model_args=()
  model_source_args "$source" model_args

  echo "========== [$alias] model=$model backend=$backend source=$source =========="

  activations_rebuilt=0
  if [[ "$RUN_ACTIVATIONS" == "1" ]]; then
    if should_run_file "$activations" || ! activation_cache_is_valid "$activations" "$DATA" "$model" "$backend"; then
      echo "[1/8:$alias] Extracting all-layer activations"
      python src/extract_activations.py \
        --model "$model" \
        "${model_args[@]}" \
        --backend "$backend" \
        --data "$DATA" \
        --out "$activations" \
        --layers all \
        --pooling last \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        --image-base-dir . \
        "${TRUST_ARGS[@]}"
      activations_rebuilt=1
    else
      echo "[1/8:$alias] Reusing $activations"
    fi
  fi

  if [[ "$RUN_GENERATION" == "1" ]]; then
    if should_run_dir_file "$gen_results"; then
      echo "[2/8:$alias] Generating original model responses"
      python src/run_probe_generations.py \
        --model "$model" \
        --model-alias "$alias" \
        "${model_args[@]}" \
        --backend "$backend" \
        --data "$DATA" \
        --out-dir "$gen_dir" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        --image-base-dir . \
        "${MAX_SAMPLE_ARGS[@]}" \
        "${TRUST_ARGS[@]}"
    else
      echo "[2/8:$alias] Reusing $gen_results"
    fi
  fi

  if [[ "$RUN_ORIGINAL_JUDGE" == "1" ]]; then
    if should_run_file "$original_judge"; then
      echo "[3/8:$alias] Judging original responses with $JUDGE_MODEL ($JUDGE_MODEL_SOURCE)"
      python intentguard_refactor/judge_outputs.py \
        --model "$JUDGE_MODEL" \
        "${judge_args[@]}" \
        --backend "$JUDGE_BACKEND" \
        --input "$gen_results" \
        --out "$original_judge" \
        --response-field response \
        --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
        --batch-size "$JUDGE_BATCH_SIZE" \
        --image-base-dir . \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        "${JUDGE_IMAGE_ARGS[@]}" \
        "${TRUST_ARGS[@]}"
    else
      echo "[3/8:$alias] Reusing $original_judge"
    fi
  fi

  subspaces_rebuilt=0
  if [[ "$RUN_SUBSPACES" == "1" ]]; then
    if [[ "$FORCE" == "1" || "$activations_rebuilt" == "1" || ! -s "$intent_subspace" || ! -s "$refusal_subspace" || ! -s "$subspace_selection" ]]; then
      echo "[4/8:$alias] Fitting S_I and S_R with per-layer selection"
      python intentguard_refactor/fit_subspaces.py \
        --activations "$activations" \
        --out-dir "$run_dir/subspaces" \
        --intent-rank 3 \
        --refusal-rank 2 \
        --group-by condition \
        --refusal-labels "$original_judge"
      subspaces_rebuilt=1
    else
      echo "[4/8:$alias] Reusing fitted subspaces"
    fi
  fi

  thresholds_rebuilt=0
  if [[ "$RUN_THRESHOLDS" == "1" ]]; then
    if should_run_file "$thresholds" || [[ "$subspaces_rebuilt" == "1" ]]; then
      echo "[5/8:$alias] Calibrating model-specific thresholds"
      python intentguard_refactor/calibrate_thresholds.py \
        --activations "$activations" \
        --intent-subspace "$intent_subspace" \
        --refusal-subspace "$refusal_subspace" \
        --refusal-labels "$original_judge" \
        --out "$thresholds" \
        --model-alias "$alias"
      thresholds_rebuilt=1
    else
      echo "[5/8:$alias] Reusing $thresholds"
    fi
  fi

  intervention_rebuilt=0
  if [[ "$RUN_INTERVENTION" == "1" ]]; then
    if should_run_file "$intervention_results" || [[ "$thresholds_rebuilt" == "1" ]]; then
      echo "[6/8:$alias] Applying hard-refusal intervention"
      python intentguard_refactor/apply_intervention.py \
        --input "$gen_results" \
        --activations "$activations" \
        --intent-subspace "$intent_subspace" \
        --refusal-subspace "$refusal_subspace" \
        --thresholds "$thresholds" \
        --out "$intervention_results"
      intervention_rebuilt=1
    else
      echo "[6/8:$alias] Reusing $intervention_results"
    fi
  fi

  post_judge_rebuilt=0
  if [[ "$RUN_POST_JUDGE" == "1" ]]; then
    if should_run_file "$post_judge" || [[ "$intervention_rebuilt" == "1" ]]; then
      echo "[7/8:$alias] Judging post-intervention responses with $JUDGE_MODEL ($JUDGE_MODEL_SOURCE)"
      python intentguard_refactor/judge_outputs.py \
        --model "$JUDGE_MODEL" \
        "${judge_args[@]}" \
        --backend "$JUDGE_BACKEND" \
        --input "$intervention_results" \
        --out "$post_judge" \
        --response-field response \
        --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
        --batch-size "$JUDGE_BATCH_SIZE" \
        --image-base-dir . \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        "${JUDGE_IMAGE_ARGS[@]}" \
        "${TRUST_ARGS[@]}"
      post_judge_rebuilt=1
    else
      echo "[7/8:$alias] Reusing $post_judge"
    fi
  fi

  if [[ "$RUN_AUDIT" == "1" ]]; then
    if [[ "$FORCE" == "1" || "$intervention_rebuilt" == "1" || "$post_judge_rebuilt" == "1" || ! -s "$audit_jsonl" || ! -s "$audit_csv" || ! -s "$audit_summary" ]]; then
      echo "[8/8:$alias] Merging per-sample audit"
      python intentguard_refactor/merge_audit.py \
        --detections "$intervention_results" \
        --original-judge "$original_judge" \
        --post-judge "$post_judge" \
        --out "$audit_jsonl" \
        --csv-out "$audit_csv" \
        --summary-out "$audit_summary"
    else
      echo "[8/8:$alias] Reusing $audit_jsonl"
    fi
  fi

  echo "[$alias] Done. Audit: $audit_jsonl"
done

echo "Done. Main outputs:"
echo "  $BASE_RUN_DIR/<alias>/sample_audit.jsonl"
echo "  $BASE_RUN_DIR/<alias>/subspaces/subspace_selection.json"
echo "  $BASE_RUN_DIR/<alias>/thresholds.json"

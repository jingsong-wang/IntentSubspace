#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
RUN_DIR="${RUN_DIR:-runs/cross_model_realistic_intent}"
DATA="${DATA:-data/multi_intent_probe.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-src/prompts.py}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
JUDGE_INCLUDE_IMAGE="${JUDGE_INCLUDE_IMAGE:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_VISUALIZATION="${RUN_VISUALIZATION:-1}"
VIS_OUT_SUBDIR="${VIS_OUT_SUBDIR:-visualizations_with_responses}"
VIS_METHODS="${VIS_METHODS:-pca,tsne}"
VIS_MAX_SAMPLES="${VIS_MAX_SAMPLES:-1200}"
VIS_COLOR_BY="${VIS_COLOR_BY:-label,label_name,intent_family,condition,image_role,prompt_form,prompt_strategy,carrier_type,intent_label,condition_label,prompt_form_label,prompt_strategy_label,carrier_label,nuisance_combo,response_outcome,refusal_state,judge_score_label,label_response_outcome}"

GEMMA3_MODELSCOPE_ID="${GEMMA3_MODELSCOPE_ID:-google/gemma-3-12b-it}"
LLAMA32_HF_ID="${LLAMA32_HF_ID:-LLM-Research/Llama-3.2-11B-Vision-Instruct}"

# Format: alias|model_id_or_local_path|backend|layer|source
# Default follows the current main analysis focus. Add Llama by overriding MODEL_SPECS if needed.
# MODEL_SPECS="${MODEL_SPECS:-qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|last|hf;gemma3_12b|$GEMMA3_MODELSCOPE_ID|generic_vlm|last|modelscope;llama32_11b_vision|LLM-Research/Llama-3.2-11B-Vision-Instruct|generic_vlm|last|modelscope}"
MODEL_SPECS="qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|last|hf;gemma3_12b|google/gemma-3-12b-it|generic_vlm|last|modelscope;llama32_11b_vision|LLM-Research/Llama-3.2-11B-Vision-Instruct|generic_vlm|last|modelscope"
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

IFS=';' read -ra SPECS <<< "$MODEL_SPECS"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r alias model backend layer source <<< "$spec"
  source="${source:-hf}"
  if [[ -z "${alias:-}" || -z "${model:-}" || -z "${backend:-}" || -z "${layer:-}" ]]; then
    echo "Invalid MODEL_SPECS item: $spec" >&2
    exit 1
  fi

  model_dir="$RUN_DIR/$alias"
  gen_dir="$model_dir/train_generations"
  judge_dir="$model_dir/train_judge"
  mkdir -p "$model_dir"

  MODEL_SOURCE_ARGS=(--model-source "$source")
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    MODEL_SOURCE_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
  fi

  if [[ "$RUN_GENERATION" == "1" ]]; then
    echo "[generate:$alias] Running model responses on $DATA"
    python src/run_probe_generations.py \
      --model "$model" \
      --model-alias "$alias" \
      "${MODEL_SOURCE_ARGS[@]}" \
      --backend "$backend" \
      --data "$DATA" \
      --out-dir "$gen_dir" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --dtype "$DTYPE" \
      --device "$DEVICE" \
      --image-base-dir "$ROOT" \
      --allow-image-surrogate \
      "${MAX_SAMPLE_ARGS[@]}" \
      "${TRUST_ARGS[@]}"
  fi

  if [[ "$RUN_JUDGE" == "1" ]]; then
    echo "[judge:$alias] Judging model responses with $JUDGE_MODEL"
    python src/judge_benchmark_outputs.py \
      --model "$JUDGE_MODEL" \
      --input "$gen_dir/generation_results.jsonl" \
      --out-dir "$judge_dir" \
      --judge-prompt-file "$PROMPT_FILE" \
      --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
      --dtype "$DTYPE" \
      --device "$DEVICE" \
      "${JUDGE_IMAGE_ARGS[@]}" \
      "${TRUST_ARGS[@]}"
  fi

  if [[ "$RUN_VISUALIZATION" == "1" ]]; then
    activations="$model_dir/activations.npz"
    subspace="$model_dir/fit_by_condition/intent_subspace.npz"
    if [[ -f "$activations" && -f "$subspace" && -f "$judge_dir/judge_results.jsonl" ]]; then
      echo "[viz:$alias] Writing response-aware PNG and HTML visualizations"
      python src/visualize_subspace.py \
        --activations "$activations" \
        --subspace "$subspace" \
        --layer "$layer" \
        --out-dir "$model_dir/$VIS_OUT_SUBDIR" \
        --methods "$VIS_METHODS" \
        --max-samples "$VIS_MAX_SAMPLES" \
        --color-by "$VIS_COLOR_BY" \
        --sample-results "$judge_dir/judge_results.jsonl"
    else
      echo "[viz:$alias] Skipping visualization; missing activations, subspace, or judge results" >&2
    fi
  fi
done

if [[ "$RUN_VISUALIZATION" == "1" ]]; then
  python src/summarize_cross_model_subspace.py \
    --run-dir "$RUN_DIR" \
    --visualization-subdir "$VIS_OUT_SUBDIR" \
    --out-json "$RUN_DIR/cross_model_subspace_summary_with_responses.json" \
    --out-md "$RUN_DIR/cross_model_subspace_summary_with_responses.md"
fi

echo "Done. Main outputs:"
echo "  $RUN_DIR/<model>/train_generations/generation_results.jsonl"
echo "  $RUN_DIR/<model>/train_judge/judge_results.jsonl"
echo "  $RUN_DIR/<model>/$VIS_OUT_SUBDIR/interactive_index.html"

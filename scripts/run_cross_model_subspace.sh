#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
RUN_DIR="${RUN_DIR:-runs/cross_model_intent_subspace}"
DATA="${DATA:-data/multi_intent_probe.jsonl}"
CONFIG="${CONFIG:-configs/multi_intent.json}"
ASSET_DIR="${ASSET_DIR:-data/multi_intent_assets}"
POOLING="${POOLING:-last}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
RANK="${RANK:-3}"
GROUP_BY="${GROUP_BY:-condition}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
VISUALIZE="${VISUALIZE:-1}"
VIS_MAX_SAMPLES="${VIS_MAX_SAMPLES:-1200}"
VIS_COLOR_BY="${VIS_COLOR_BY:-label,label_name,intent_family,condition,image_role,prompt_form,prompt_strategy,carrier_type,intent_label,condition_label,prompt_form_label,prompt_strategy_label,carrier_label,nuisance_combo,response_outcome,refusal_state,judge_score_label,label_response_outcome}"

# Format: alias|model_id_or_local_path|backend|layers|source
# Use aliases early/mid/late/last when hidden-layer counts differ across models.
GEMMA3_MODELSCOPE_ID="${GEMMA3_MODELSCOPE_ID:-google/gemma-3-12b-it}"
LLAMA32_HF_ID="${LLAMA32_HF_ID:-LLM-Research/Llama-3.2-11B-Vision-Instruct}"
# MODEL_SPECS="${MODEL_SPECS:-qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|last|hf;gemma3_12b|$GEMMA3_MODELSCOPE_ID|generic_vlm|last|modelscope;llama32_11b_vision|$LLAMA32_HF_ID|generic_vlm|last|modelscope}"
MODEL_SPECS="${MODEL_SPECS:-llama32_11b_vision|$LLAMA32_HF_ID|generic_vlm|last|modelscope;gemma3_12b|$GEMMA3_MODELSCOPE_ID|generic_vlm|last|modelscope;qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|last|hf}"

cd "$ROOT"
mkdir -p "$RUN_DIR"

TRUST_ARGS=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

if [[ "${REBUILD_DATA:-1}" == "1" || ! -f "$DATA" ]]; then
  echo "[data] Building realistic multi-intent probe"
  python src/make_multi_intent_probe.py \
    --config "$CONFIG" \
    --out "$DATA" \
    --asset-dir "$ASSET_DIR"
fi

IFS=';' read -ra SPECS <<< "$MODEL_SPECS"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r alias model backend layers source <<< "$spec"
  source="${source:-hf}"
  if [[ -z "${alias:-}" || -z "${model:-}" || -z "${backend:-}" || -z "${layers:-}" ]]; then
    echo "Invalid MODEL_SPECS item: $spec" >&2
    exit 1
  fi

  out="$RUN_DIR/$alias"
  mkdir -p "$out"
  MODEL_SOURCE_ARGS=(--model-source "$source")
  if [[ -n "$MODEL_CACHE_DIR" ]]; then
    MODEL_SOURCE_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
  fi

  echo "[model:$alias] Extracting activations from $model source=$source backend=$backend layers=$layers"
  python src/extract_activations.py \
    --model "$model" \
    "${MODEL_SOURCE_ARGS[@]}" \
    --backend "$backend" \
    --data "$DATA" \
    --out "$out/activations.npz" \
    --layers "$layers" \
    --pooling "$POOLING" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --image-base-dir "$ROOT" \
    "${TRUST_ARGS[@]}"

  echo "[model:$alias] Fitting intent subspace"
  python src/fit_subspace.py \
    --activations "$out/activations.npz" \
    --rank "$RANK" \
    --group-by "$GROUP_BY" \
    --out-dir "$out/fit_by_condition"

  echo "[model:$alias] Scoring training activations for diagnostics"
  python src/score_subspace.py \
    --activations "$out/activations.npz" \
    --subspace "$out/fit_by_condition/intent_subspace.npz" \
    --out-dir "$out/score_train"

  if [[ "$VISUALIZE" == "1" ]]; then
    echo "[model:$alias] Visualizing raw/subspace/residual representations"
    python src/visualize_subspace.py \
      --activations "$out/activations.npz" \
      --subspace "$out/fit_by_condition/intent_subspace.npz" \
      --layer "$layers" \
      --out-dir "$out/visualizations" \
      --max-samples "$VIS_MAX_SAMPLES" \
      --color-by "$VIS_COLOR_BY"
  fi
done

python src/summarize_cross_model_subspace.py \
  --run-dir "$RUN_DIR" \
  --out-json "$RUN_DIR/cross_model_subspace_summary.json" \
  --out-md "$RUN_DIR/cross_model_subspace_summary.md"

echo "Done. Summary:"
echo "  $RUN_DIR/cross_model_subspace_summary.md"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
RUN_DIR="${RUN_DIR:-runs/cross_model_realistic_intent}"
MODEL_DIRS="${MODEL_DIRS:-qwen25vl7b gemma3_12b}"
OUT_SUBDIR="${OUT_SUBDIR:-visualizations_factor_analysis}"
LAYER="${LAYER:-last}"
METHODS="${METHODS:-pca,tsne}"
VIS_MAX_SAMPLES="${VIS_MAX_SAMPLES:-1200}"
VIS_COLOR_BY="${VIS_COLOR_BY:-label,label_name,intent_family,condition,image_role,prompt_form,prompt_strategy,carrier_type,intent_label,condition_label,prompt_form_label,prompt_strategy_label,carrier_label,nuisance_combo,response_outcome,refusal_state,judge_score_label,label_response_outcome}"
SEED="${SEED:-7}"
TSNE_PERPLEXITY="${TSNE_PERPLEXITY:-30}"

cd "$ROOT"

for model_dir in $MODEL_DIRS; do
  model_path="$RUN_DIR/$model_dir"
  activations="$model_path/activations.npz"
  subspace="$model_path/fit_by_condition/intent_subspace.npz"
  out_dir="$model_path/$OUT_SUBDIR"
  sample_results="$model_path/train_judge/judge_results.jsonl"
  SAMPLE_RESULT_ARGS=()
  if [[ -f "$sample_results" ]]; then
    SAMPLE_RESULT_ARGS=(--sample-results "$sample_results")
  fi

  if [[ ! -f "$activations" || ! -f "$subspace" ]]; then
    echo "Skipping $model_dir: missing $activations or $subspace" >&2
    continue
  fi

  echo "[viz:$model_dir] Writing enhanced visualizations to $out_dir"
  python src/visualize_subspace.py \
    --activations "$activations" \
    --subspace "$subspace" \
    --layer "$LAYER" \
    --out-dir "$out_dir" \
    --methods "$METHODS" \
    --max-samples "$VIS_MAX_SAMPLES" \
    --color-by "$VIS_COLOR_BY" \
    "${SAMPLE_RESULT_ARGS[@]}" \
    --seed "$SEED" \
    --tsne-perplexity "$TSNE_PERPLEXITY"
done

python src/summarize_cross_model_subspace.py \
  --run-dir "$RUN_DIR" \
  --visualization-subdir "$OUT_SUBDIR" \
  --out-json "$RUN_DIR/cross_model_subspace_summary_enhanced.json" \
  --out-md "$RUN_DIR/cross_model_subspace_summary_enhanced.md"

echo "Enhanced visualization summary:"
echo "  $RUN_DIR/cross_model_subspace_summary_enhanced.md"

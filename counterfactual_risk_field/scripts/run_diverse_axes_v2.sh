#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-counterfactual_risk_field/configs/protocol_v2_diverse_axes.json}"
SEEDS="${SEEDS:-counterfactual_risk_field/work/seeds.jsonl}"
WORK="${WORK:-counterfactual_risk_field/work/v2_diverse_axes}"
GEMMA3_MODEL="${GEMMA3_MODEL:-/home/wangjingsong/.cache/modelscope/models/google--gemma-3-12b-it/snapshots/master}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TARGET_MODEL_SOURCE="${TARGET_MODEL_SOURCE:-modelscope}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
SAMPLING_SEED="${SAMPLING_SEED:-20260721}"

REQUESTS="$WORK/generation_requests.jsonl"
RESPONSES="$WORK/generation_responses.jsonl"
CANDIDATES="$WORK/pair_candidates.jsonl"
ACCEPTED="$WORK/pair_candidates.accepted.jsonl"
REJECTED="$WORK/pair_candidates.rejected.jsonl"
QUALITY="$WORK/counterfactual_quality.json"
PAIRS="$WORK/pairs.jsonl"
EXPERIMENT="$WORK/experiment.jsonl"
ACTIVATIONS="$WORK/activations/qwen25vl7b"
RESULTS="$WORK/results/qwen25vl7b"

cd "$WORKSPACE_ROOT"
mkdir -p "$WORK"

if [[ ! -f "$SEEDS" ]]; then
  echo "Missing seed manifest: $SEEDS" >&2
  exit 2
fi
if [[ ! -d "$GEMMA3_MODEL" ]]; then
  echo "Missing local Gemma3 model directory: $GEMMA3_MODEL" >&2
  exit 2
fi

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.build_generation_requests \
  --seeds "$SEEDS" \
  --config "$CONFIG" \
  --out "$REQUESTS"

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.run_local_generation \
  --requests "$REQUESTS" \
  --out "$RESPONSES" \
  --model "$GEMMA3_MODEL" \
  --backend generic_vlm \
  --model-source modelscope \
  --dtype bfloat16 \
  --device auto \
  --attn-implementation sdpa \
  --batch-size 1 \
  --max-new-tokens 512 \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --sampling-seed "$SAMPLING_SEED" \
  --resume \
  --profile-generation

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.ingest_generations \
  --requests "$REQUESTS" \
  --responses "$RESPONSES" \
  --out "$CANDIDATES"

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.audit_counterfactual_quality \
  --candidates "$CANDIDATES" \
  --accepted-out "$ACCEPTED" \
  --rejected-out "$REJECTED" \
  --report "$QUALITY" \
  --min-topic-coverage 0.15 \
  --min-length-ratio 0.4 \
  --max-length-ratio 2.5

# Development run only. Remove --allow-unreviewed after completing the five-field human audit.
"$PYTHON_BIN" -m counterfactual_risk_field.scripts.materialize_pairs \
  --candidates "$ACCEPTED" \
  --out "$PAIRS" \
  --allow-unreviewed

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.combine_experiment_manifest \
  --pairs "$PAIRS" \
  --seeds "$SEEDS" \
  --out "$EXPERIMENT"

"$PYTHON_BIN" -m ood_intent_study.extract \
  --manifest "$EXPERIMENT" \
  --out-dir "$ACTIVATIONS" \
  --model-name qwen25vl7b \
  --model "$TARGET_MODEL" \
  --model-source "$TARGET_MODEL_SOURCE" \
  --backend qwen2_5_vl \
  --layers all \
  --readouts last,non_image_mean \
  --dtype bfloat16 \
  --storage-dtype float32 \
  --device-map auto \
  --attn-implementation sdpa \
  --shard-size 16 \
  --resume \
  --fail-fast

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.run_experiment \
  --manifest "$EXPERIMENT" \
  --activations "$ACTIVATIONS" \
  --out-dir "$RESULTS" \
  --config "$CONFIG"

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.audit_selected_views \
  --summary "$RESULTS/summary.json" \
  --manifest "$EXPERIMENT" \
  --activations "$ACTIVATIONS" \
  --split reference \
  --out-dir "$RESULTS/source_audits"

echo "CNRF v2 diverse-axis development experiment complete: $RESULTS"

#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GEMMA3_MODEL_PATH="${GEMMA3_MODEL_PATH:-/home/wangjingsong/.cache/modelscope/models/google--gemma-3-12b-it/snapshots/master}"
WORK="$WORKSPACE_ROOT/counterfactual_risk_field/work/smoke_gemma3"
REQUESTS="$WORK/generation_requests.jsonl"
RESPONSES="$WORK/generation_responses.jsonl"
CANDIDATES="$WORK/pair_candidates.jsonl"
PAIRS="$WORK/pairs.jsonl"
EXPERIMENT="$WORK/experiment.jsonl"
ACTIVATIONS="$WORK/activations/gemma3_12b"
RESULTS="$WORK/results/gemma3_12b"

if [[ ! -f "$REQUESTS" ]]; then
  echo "Missing prepared request file: $REQUESTS" >&2
  exit 2
fi
if [[ ! -d "$GEMMA3_MODEL_PATH" ]]; then
  echo "Missing local Gemma3 model directory: $GEMMA3_MODEL_PATH" >&2
  exit 2
fi

cd "$WORKSPACE_ROOT"

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.run_local_generation \
  --requests "$REQUESTS" \
  --out "$RESPONSES" \
  --model "$GEMMA3_MODEL_PATH" \
  --backend generic_vlm \
  --model-source modelscope \
  --dtype bfloat16 \
  --device auto \
  --attn-implementation sdpa \
  --batch-size 1 \
  --max-new-tokens 512 \
  --temperature 0 \
  --resume \
  --profile-generation

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.ingest_generations \
  --requests "$REQUESTS" \
  --responses "$RESPONSES" \
  --out "$CANDIDATES"

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.materialize_pairs \
  --candidates "$CANDIDATES" \
  --out "$PAIRS" \
  --allow-unreviewed

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.combine_experiment_manifest \
  --pairs "$PAIRS" \
  --seeds "$WORK/seeds.jsonl" \
  --out "$EXPERIMENT"

"$PYTHON_BIN" -m ood_intent_study.extract \
  --manifest "$EXPERIMENT" \
  --out-dir "$ACTIVATIONS" \
  --model-name gemma3_12b \
  --model "$GEMMA3_MODEL_PATH" \
  --model-source modelscope \
  --backend gemma3 \
  --layers all \
  --readouts last,non_image_mean,image_mean \
  --dtype bfloat16 \
  --storage-dtype float32 \
  --device-map auto \
  --attn-implementation sdpa \
  --shard-size 8 \
  --resume \
  --fail-fast

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.run_experiment \
  --manifest "$EXPERIMENT" \
  --activations "$ACTIVATIONS" \
  --out-dir "$RESULTS" \
  --config counterfactual_risk_field/configs/smoke_gemma3.json

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.audit_selected_views \
  --summary "$RESULTS/summary.json" \
  --manifest "$EXPERIMENT" \
  --activations "$ACTIVATIONS" \
  --split reference \
  --out-dir "$RESULTS/source_audits"

echo "Gemma3 smoke experiment complete: $RESULTS"

#!/usr/bin/env bash
set -euo pipefail

# Diagnostic ceiling only: this experiment deliberately uses test/external labels
# to select thresholds and arrow subsets.  Never report its numbers as held-out
# generalization performance.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK="${WORK:-counterfactual_risk_field/work/v2_axes_temp07}"
MODEL_TAG="${MODEL_TAG:-qwen25vl7b}"
CONFIG="${CONFIG:-counterfactual_risk_field/configs/protocol_v2_diverse_axes.json}"
ORACLE_CONFIG="${ORACLE_CONFIG:-counterfactual_risk_field/configs/oracle_v2.json}"
OUT_DIR="${OUT_DIR:-$WORK/oracle_e1}"

cd "$WORKSPACE_ROOT"

"$PYTHON_BIN" -m counterfactual_risk_field.scripts.run_oracle_ceiling \
  --manifest "$WORK/experiment.jsonl" \
  --activations "$WORK/activations/$MODEL_TAG" \
  --summary "$WORK/results/$MODEL_TAG/summary.json" \
  --config "$CONFIG" \
  --oracle-config "$ORACLE_CONFIG" \
  --out-dir "$OUT_DIR"

echo "ORACLE_ONLY ceiling experiment complete: $OUT_DIR"


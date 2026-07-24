#!/usr/bin/env bash
set -euo pipefail

# ORACLE_ONLY: test/external labels select thresholds and arrow subsets.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK="${WORK:-counterfactual_risk_field/work/v2_axes_temp07}"
MODEL_TAG="${MODEL_TAG:-qwen25vl7b}"
OUT_DIR="${OUT_DIR:-jailbreak_repro/runs/cnrf_oracle/$MODEL_TAG/$(basename "$WORK")}"

cd "$WORKSPACE_ROOT"

"$PYTHON_BIN" -m jailbreak_repro.run_cnrf_oracle \
  --work "$WORK" \
  --model-tag "$MODEL_TAG" \
  --out-dir "$OUT_DIR" \
  "$@"

echo "CNRF ORACLE_ONLY results updated: $OUT_DIR"

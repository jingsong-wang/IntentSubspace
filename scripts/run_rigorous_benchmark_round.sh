#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-$MODEL}"
PROMPT_FILE="${PROMPT_FILE:-../Bench/evaluation/prompts.py}"
RUN_DIR="${RUN_DIR:-runs/qwen25vl7b_rigorous_round}"
SCORE_LAYER="${SCORE_LAYER:-28}"
POOLING="${POOLING:-last}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
SPLIT_SEED="${SPLIT_SEED:-7}"
CALIBRATION_RATIO="${CALIBRATION_RATIO:-0.30}"
THRESHOLD_OBJECTIVE="${THRESHOLD_OBJECTIVE:-balanced}"
TARGET_RECALL="${TARGET_RECALL:-0.95}"
TARGET_FPR="${TARGET_FPR:-0.10}"
SUBSPACE="${SUBSPACE:-runs/qwen25vl7b_multi_intent/fit_by_condition/intent_subspace.npz}"
REBUILD_SUBSPACE="${REBUILD_SUBSPACE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

cd "$ROOT"
mkdir -p "$RUN_DIR"

TRUST_ARGS=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

if [[ "$REBUILD_SUBSPACE" == "1" || ! -f "$SUBSPACE" ]]; then
  echo "[1/8] Rebuilding multi-intent probe and fitted subspace"
  python src/make_multi_intent_probe.py \
    --config configs/multi_intent.json \
    --out data/multi_intent_probe.jsonl \
    --asset-dir data/multi_intent_assets

  python src/extract_activations.py \
    --model "$MODEL" \
    --backend qwen2_5_vl \
    --data data/multi_intent_probe.jsonl \
    --out "$RUN_DIR/multi_intent_activations.npz" \
    --layers "$SCORE_LAYER" \
    --pooling "$POOLING" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --image-base-dir "$ROOT" \
    "${TRUST_ARGS[@]}"

  python src/fit_subspace.py \
    --activations "$RUN_DIR/multi_intent_activations.npz" \
    --rank 3 \
    --group-by condition \
    --out-dir "$RUN_DIR/fit_by_condition"

  SUBSPACE="$RUN_DIR/fit_by_condition/intent_subspace.npz"
else
  echo "[1/8] Reusing fitted subspace: $SUBSPACE"
fi

echo "[2/8] Building XSTest calibration data with hard benign controls"
python src/make_hard_benign_probe.py \
  --xstest benchmark/XSTest/xstest_prompts.csv \
  --multi-intent-config configs/multi_intent.json \
  --out "$RUN_DIR/hard_benign_calibration.jsonl" \
  --split calibration \
  --split-seed "$SPLIT_SEED" \
  --calibration-ratio "$CALIBRATION_RATIO" \
  --synthetic-hard-benign-per-intent 8

echo "[3/8] Extracting calibration activations with Qwen2.5-VL"
python src/extract_activations.py \
  --model "$MODEL" \
  --backend qwen2_5_vl \
  --data "$RUN_DIR/hard_benign_calibration.jsonl" \
  --out "$RUN_DIR/hard_benign_calibration_activations.npz" \
  --layers "$SCORE_LAYER" \
  --pooling "$POOLING" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --image-base-dir "$ROOT" \
  "${TRUST_ARGS[@]}"

echo "[4/8] Calibrating threshold"
python src/calibrate_subspace_threshold.py \
  --activations "$RUN_DIR/hard_benign_calibration_activations.npz" \
  --subspace "$SUBSPACE" \
  --score-layer "$SCORE_LAYER" \
  --objective "$THRESHOLD_OBJECTIVE" \
  --target-recall "$TARGET_RECALL" \
  --target-fpr "$TARGET_FPR" \
  --out "$RUN_DIR/calibrated_threshold.json"

echo "[5/8] Running HADES dynamic guard in monitor mode"
python src/run_hades_dynamic_guard.py \
  --model "$MODEL" \
  --meta benchmark/HADES/hades_750_meta.jsonl \
  --image-dir benchmark/HADES/images \
  --subspace "$SUBSPACE" \
  --threshold-json "$RUN_DIR/calibrated_threshold.json" \
  --score-layer "$SCORE_LAYER" \
  --pooling "$POOLING" \
  --guard-mode monitor \
  --out-dir "$RUN_DIR/hades_monitor" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  "${TRUST_ARGS[@]}"

echo "[6/8] Running held-out XSTest evaluation"
python src/run_xstest_guard_eval.py \
  --model "$MODEL" \
  --data benchmark/XSTest/xstest_prompts.csv \
  --subspace "$SUBSPACE" \
  --threshold-json "$RUN_DIR/calibrated_threshold.json" \
  --score-layer "$SCORE_LAYER" \
  --pooling "$POOLING" \
  --split test \
  --split-seed "$SPLIT_SEED" \
  --calibration-ratio "$CALIBRATION_RATIO" \
  --out-dir "$RUN_DIR/xstest_test" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  "${TRUST_ARGS[@]}"

echo "[7/8] Judging HADES model responses with Qwen2.5-VL"
python src/judge_benchmark_outputs.py \
  --model "$JUDGE_MODEL" \
  --input "$RUN_DIR/hades_monitor/hades_dynamic_results.jsonl" \
  --out-dir "$RUN_DIR/hades_judge" \
  --judge-prompt-file "$PROMPT_FILE" \
  --include-image \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  "${TRUST_ARGS[@]}"

echo "[8/8] Judging XSTest model responses with Qwen2.5-VL"
python src/judge_benchmark_outputs.py \
  --model "$JUDGE_MODEL" \
  --input "$RUN_DIR/xstest_test/xstest_results.jsonl" \
  --out-dir "$RUN_DIR/xstest_judge" \
  --judge-prompt-file "$PROMPT_FILE" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  "${TRUST_ARGS[@]}"

echo "Done. Main outputs:"
echo "  $RUN_DIR/calibrated_threshold.json"
echo "  $RUN_DIR/hades_monitor/hades_dynamic_report.md"
echo "  $RUN_DIR/xstest_test/xstest_report.md"
echo "  $RUN_DIR/hades_judge/judge_report.md"
echo "  $RUN_DIR/xstest_judge/judge_report.md"

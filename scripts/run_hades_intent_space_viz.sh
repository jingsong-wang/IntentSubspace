#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
RUN_DIR="${RUN_DIR:-runs/qwen25vl7b_realistic_intent_round}"
HADES_RESULTS="${HADES_RESULTS:-$RUN_DIR/hades_monitor/hades_dynamic_results.jsonl}"
HADES_JUDGE_RESULTS="${HADES_JUDGE_RESULTS:-$RUN_DIR/hades_judge/judge_results.jsonl}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/hades_intent_space_viz}"
HARMFUL_SCORE_THRESHOLD="${HARMFUL_SCORE_THRESHOLD:-3}"
METHODS="${METHODS:-pca,tsne}"
SEED="${SEED:-7}"
TSNE_PERPLEXITY="${TSNE_PERPLEXITY:-30}"

cd "$ROOT"

python src/visualize_hades_intent_space.py \
  --results "$HADES_RESULTS" \
  --judge-results "$HADES_JUDGE_RESULTS" \
  --out-dir "$OUT_DIR" \
  --harmful-score-threshold "$HARMFUL_SCORE_THRESHOLD" \
  --methods "$METHODS" \
  --seed "$SEED" \
  --tsne-perplexity "$TSNE_PERPLEXITY"

echo "HADES intent-space visualization written to: $OUT_DIR"
echo "  $OUT_DIR/hades_intent_space_report.md"
echo "  $OUT_DIR/hades_intent_space_summary.json"
echo "  $OUT_DIR/hades_intent_space_points.csv"

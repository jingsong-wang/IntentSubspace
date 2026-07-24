# Internal bootstrap only

This directory records a pipeline smoke test on the pre-existing CISR-v4 paired
probe and Qwen2.5-VL-7B last-token activations. It is **not** an evaluation on the
frozen CNRF benchmark split and must not be reported as the proposed method's
main result.

Configuration: layer index 27, `k=5`, at most 40 reference packs, and the v1
worst-benign-group conformal threshold.

Observed test result for CNRF: AUROC 0.794, TPR 0.083 and FPR 0.000 at the frozen
threshold; support coverage 0.583 and abstention rate 0.417. On the same paired
data, mean-arrow AUROC was 0.949 with TPR 0.174 at FPR 0.000, while two-sided KNN
AUROC was 0.740 with TPR 0.068 at FPR 0.000.

The source audit also did not provide strong evidence that old CISR-v4 arrows
remove carrier identity: carrier macro-F1 was 0.803 for raw endpoints, 0.882 for
midpoints, and 0.783 for arrows. Therefore the new counterfactual-generation and
pack-control protocol must pass the mechanism gate before any OOD claim is made.

Artifacts:

- `qwen_cisr_v4_last/summary.json`: full metrics and calibration diagnostics;
- `qwen_cisr_v4_last/source_audit.json`: grouped source-readability audit;
- `qwen_cisr_v4_last/scores.jsonl`: per-sample scores and neighbor traces.

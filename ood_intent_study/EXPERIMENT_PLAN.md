# Code Experiment Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-17
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title**: Qwen2.5-VL-7B 与 Gemma-3-12B 的逐层意图可分性和视觉越狱 OOD 偏移
- **Objective**: 区分三种现象：常规 benchmark 上的标签可分性、跨数据集可迁移的意图信号、视觉攻击引起的表征/工作点偏移。
- **Type**: representation extraction + frozen linear probing + distribution-shift analysis

### Research Questions

- **RQ1**: harmful/benign 标签在两个模型的哪些 decoder 层最可分？
- **RQ2**: pooled 随机切分的高可分性，在 leave-one-source-out 后是否仍成立？
- **RQ3**: FigStep、JOOD、CS-DJ 与 JailBreakV-28K 三种载体相对非攻击 harmful 是否出现 score-location shift 或 group-CV representation domain separability？
- **RQ4**: 当 prompt-last 失败时，图像 token 聚合是否仍保留可恢复的 harmful 信号？
- **RQ5**: Qwen 与 Gemma 的最佳层、偏移方向和最差数据源是否一致？

### Hypotheses

- **H1**: 常规 benchmark 在中后层存在较高线性可分性，但其 pooled 结果高于 LOSO 结果。
- **H2**: 数据集来源在同一标签内部仍可被表征恢复，说明 pooled boundary 包含 source/style 信息。
- **H3**: 六个冻结攻击条件会表现出不同程度的 group-CV harmful-domain AUROC；CS-DJ 的复杂视觉组合预期产生更强域可分性，但该方向在观察数据前不作为结论。
- **H4**: 若 CS-DJ 只在 `last` readout 下降而 `image_mean` 仍有信号，则主要问题是全局聚合/工作点；若三个 readout 均下降，则更支持感知或表示缺失。

## Variables and Controls

### Independent Variables

- model: Qwen2.5-VL-7B-Instruct / Gemma-3-12B-IT
- decoder layer: all blocks, one-based block-output numbering
- readout: `last`, `non_image_mean`, `image_mean`
- mechanistic readout comparisons use a common multimodal panel: every compared readout is valid, `image_token_count > 0`, and the sample-ID fingerprint is identical
- source dataset: 9 benchmark + 6 external attack conditions
- carrier/attack variant: text, image-text, HADES, MM-SafetyBench attack type, JOOD alpha/augmentation, CS-DJ distraction composition

### Dependent Variables

- non-attack validation/test AUROC, AUPRC, balanced accuracy
- TPR at validation-calibrated FPR 1% and 5%
- per-source TPR/TNR/balanced accuracy
- leave-one-source-out metrics
- external attack TPR at frozen balanced threshold
- attack-vs-benign-panel AUROC
- attack harmful vs standard harmful domain-AUROC
- centroid shift L2 and its cosine with the training intent direction
- intent-score MMD
- within-label source macro-F1
- conditional cluster-bootstrap percentile intervals with requested/valid replicate counts

### Primary Controls

- benign multimodal controls: MM-Vet and VizWiz-VQA
- harmful text-only controls: AdvBench, XSTest unsafe, MM-SafetyBench Text_only
- hard benign text control: XSTest safe
- standard multimodal harmful controls: HADES and MM-SafetyBench
- metadata-only shortcut baseline: prompt length, sequence length, image presence/token count, image size

## Data and Sampling

### Source Semantics

| Source | Binary study label | Label caveat | Role |
|---|---:|---|---|
| AdvBench | 1 | dataset-derived harmful | user |
| Alpaca | 0 | source-field/dataset assumption | user |
| DAN-Prompts | 1 | jailbreak template, not guaranteed harmful intent | user |
| OpenAssistant | 0 | current asset contains assistant replies | assistant |
| XSTest | 0/1 | explicit safe/unsafe | user |
| HADES | 1 | dataset-derived harmful | user |
| MM-SafetyBench | 1 | dataset-derived; risk-domain rows may be nuanced | user |
| MM-Vet | 0 | utility benchmark assumption | user |
| VizWiz-VQA | 0 | assumed-benign real-world VQA; local 1000-row validation convenience subset historically stored under `benchmark/VQAv2` | user |
| FigStep / JOOD / CS-DJ | 1 | official/generated attack target | user |
| JailBreakV-28K FigStep / LLM-Transfer / Query-Related | 1 | three separately sampled external carrier conditions; never used for fitting | user |

Main results use this benchmark label for comparability. A required sensitivity view excludes weak/assumed labels where feasible. DAN must additionally be discussed on a separate jailbreak-template axis; OpenAssistant cannot be interpreted as clean user-prompt evidence.

### Sampling Rule

- Default maximum: 128 samples per `source x label`.
- Sampling is deterministic and round-robin across category and carrier variant.
- Expected total: about 2,048 rows.
- Images are decoded/hashed only after preselection.
- Exact duplicate prompts/images within a source are removed.
- Reference completions (`target`, `output`, `answer`) are retained only as source provenance and never enter the model input.

### Split and Leakage Unit

Standard benchmark rows use 70/15/15 train/validation/test. A union-find component joins rows that share either:

- source-specific `group_id`, or
- exact normalized cross-source `semantic_group_id`.

The resulting `split_group_id` is the only split key. This protects MM-SafetyBench carrier variants, XSTest focus groups and exact corpus overlaps. VizWiz-VQA additionally groups by `image_id` and uses a source-scoped question ID as its split-semantic identity, because generic questions such as “what is this?” are not semantically equivalent across different images. External attacks are always `external`, even if a semantic overlap is found; the overlap is reported as contamination.

JOOD nuisance attempts share `scenario+prompt_idx`. CS-DJ retains goal ID plus distraction-set fingerprint. FigStep uses its official task ID. JailBreakV-28K uses `redteam_query` as the goal-level evaluation cluster, while repeated image paths are retained as nuisance provenance; all three carrier subsets remain external.

## Model Inference and Representation Contract

- Forward pass is prefill-only; response generation is not needed for the primary input-intent experiment.
- Decoder modules are located at runtime and hooked directly.
- Pooling happens on the hidden tensor's GPU before transfer to CPU.
- Every selected block must fire exactly once; missing/mismatched hooks are fatal for that sample.
- `last`: last attended prefill token.
- `non_image_mean`: all attended non-image placeholder positions.
- `image_mean`: image placeholder positions only; invalid for text-only samples.
- Storage: compressed FP32 shards by default; optional IEEE bfloat16 bit encoding preserves exponent range at reduced precision. Plain FP16 is accepted only after an explicit range check. Computation/pooling uses FP32 accumulators after model hidden output.
- Shards include manifest/model fingerprints and cannot be mixed across runs.
- Failed samples are logged; formal analysis must report extraction coverage and cannot silently drop failures.

Approximate activation storage before compression for 2,048 samples:

- Qwen FP32: `2048 x 3 x 28 x 3584 x 4 bytes` = about 2.30 GiB.
- Gemma FP32: `2048 x 3 x 48 x 3840 x 4 bytes` = about 4.22 GiB.
- BFloat16 storage uses approximately half these amounts before compression.

## Analysis Strategy

### Primary Layer Probe

For each model/readout/layer:

1. fit a class-balanced L2 logistic probe on non-attack train only;
2. choose the balanced threshold and FPR-constrained thresholds on non-attack validation only;
3. select the headline layer by validation AUROC only;
4. evaluate the untouched standard test and all external attacks;
5. use cluster bootstrap by `split_group_id` for conditional evaluation intervals; the fitted probe, selected layer and validation threshold stay fixed.

All attack-by-layer curves are exploratory. Only the validation-selected layer is confirmatory for the frozen external evaluation.

### Cross-source Robustness

- Report frozen standard-test metrics per source.
- Run all-layer leave-one-source-out using a standardized centroid probe. This diagnostic is deliberately cheaper than refitting hundreds of logistic probes and isolates unseen-source direction transfer.
- Report worst source and macro source performance, not only micro pooled averages.
- Summarize harmful source-label cells with TPR and benign source-label cells with TNR; do not average undefined single-label AUROC values.

### Paired Readout Panel

- The pooled per-readout tables remain descriptive because `image_mean` is invalid on text-only rows.
- Refit every readout on the intersection of valid multimodal samples for train, validation, test and attacks.
- Select layers independently using only the common-panel standard validation split.
- Use only `common_panel_layer_metrics.csv` and `common_panel_attack_metrics.csv` to support RQ4/H4.

### Shift Decomposition

For each attack and layer:

- **ranking under OOD**: attack positives vs frozen standard benign test panel AUROC;
- **frozen operating point**: attack TPR at the validation threshold;
- **score location**: mean attack score minus mean standard harmful score, standardized by standard harmful SD;
- **domain separability diagnostic**: semantic-group-CV centroid AUROC for attack harmful vs standard harmful, plus an exact-goal-matched value when overlap exists;
- **direction**: centroid displacement cosine with the training harmful-minus-benign direction;
- **distribution**: one-dimensional RBF MMD on frozen intent scores.

Interpretation matrix:

| Observation | More consistent with |
|---|---|
| High attack AUROC, low frozen TPR, low harmful-domain AUROC | mainly calibration/location shift |
| High attack AUROC, low TPR, high harmful-domain AUROC | carrier-induced representation shift plus calibration failure |
| Low attack AUROC, high domain-AUROC | intent direction does not transfer despite strong domain identity |
| `image_mean` strong, `last` weak | aggregation/readout bottleneck |
| all readouts weak | perception or distributed/nonlinear signal |

These are diagnostic inferences, not causal conclusions. Causal patching/intervention is a later experiment.

## Expected Outputs and Success Criteria

| Output | Format | Success criterion |
|---|---|---|
| unified manifest + audit | JSONL/JSON | all 11 sources present; no missing selected images; no standard split leakage |
| activation shards | NPZ | all selected rows completed; fingerprints consistent; numeric audit has zero NaN/Inf |
| layer probe metrics | CSV | every valid model/readout/layer represented |
| source/LOSO metrics | CSV | every eligible source represented; ineligible reasons and macro/worst source-label cells reported |
| common multimodal panel | CSV | identical sample-ID fingerprint for all compared readouts and both models |
| attack shift metrics | CSV | all three attacks evaluated with zero training rows |
| visual diagnostics | PNG/CSV | layer curves, source/attack heatmaps and selected-layer PCA generated |
| cross-model comparison | CSV/PNG | normalized-depth curves retain separate model identities |

No numeric success threshold is preregistered as “the method works.” The study is diagnostic. A defensible invariant-boundary claim would require jointly high standard test and LOSO performance, low within-label source recoverability, stable frozen attack TPR, and bounded attack domain shift in both models.

## Monitoring Configuration

- Extraction is shard-resumable with default shard size 16.
- `run.json` records completed/failed counts and the exact runtime.
- `run.json` is written before the first sample, records implementation/hardware/device-map identities, and is `complete` only when every selected row succeeds.
- `errors.jsonl` records sample-level failures; no automatic retry unless `--retry-failures` is explicitly used.
- Three consecutive identical sample errors abort by default to prevent a systemic processor/model mismatch from being logged as thousands of independent exclusions.
- Recommended smoke: 20 balanced rows per model, all layers, `--skip-loso` downstream.
- Formal run should stop if manifest audit fails, model hooks miss a layer, or shard fingerprints differ.
- Storage conversion must stop before overflow; a `complete` row count is insufficient unless the numeric activation audit also passes.

## Current Verification Boundary

The code and lightweight manifest/artifact tests are locally verifiable. Full GPU inference is not verified in the current desktop environment because it exposes no usable CUDA runtime, `torch`, `transformers`, or model cache. In addition:

- CS-DJ final images and its source image library are absent locally;
- local AdvBenchM has regenerated and sidecar-verified 7,650 JOOD rows; historical remote-path manifests are not accepted for formal runs;
- current historical CS-DJ uses a 100-image retrieval pool and must be named `CS-DJ-100`.
- A 2026-07-18 Gemma server run exposed legacy FP16 storage overflow; those shards require re-extraction with FP32 or bfloat16 storage before analysis.

Therefore this plan remains `UNVERIFIED` until both models complete a real multimodal forward smoke and the formal asset audit passes on the GPU server.

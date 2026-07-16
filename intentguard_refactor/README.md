# IntentGuard-LRH Refactor

This folder contains the refactored closed-loop pipeline. It reuses the shared
`src/` model loading and activation utilities; CISR_v2 extends that extractor with
optional multimodal anchor representations and keeps the judge prompt synchronized.

## CISR_v2 Detection Protocol

CISR_v2 separates detector development from intervention evaluation and uses
template-level train/validation/calibration/test isolation:

- `train`: fit the paired rank-3 intent subspace and lightweight coordinate MLP.
- `validation`: select the model layer using threshold-free AUROC/AP.
- `calibration`: select the probability threshold using a conservative lower confidence bound on target recall.
- `test`: report final detection metrics only; it is never used for fitting or calibration.

The default operating point targets 95% recall with a one-sided 95% Wilson lower
confidence bound. The report also emits pre-specified 90/95/97.5/99% sensitivity
points and explicitly marks when finite calibration data cannot certify the requested
coverage. These points never participate in layer selection.

Every target/benign pair preserves the same wrapper and modality carrier. Multimodal
samples also store a neutral image-anchor representation, allowing the detector to use
both raw rank-3 coordinates and image-residual rank-3 coordinates.

The primary split is a template holdout. Some intent families have only one or two real
semantic images under `imgs/`, so semantic-image results should not be described as an
unseen-image-source test; target and benign controls intentionally reuse the same image.

Run all three target models with resumable stages:

```bash
bash intentguard_refactor/scripts/run_detection_round_v2.sh
```

Main outputs:

```text
data/CISR_v2_probe.jsonl
runs/CISR_v2/<model>/detector/detector.npz
runs/CISR_v2/<model>/detector/detection_results.jsonl
runs/CISR_v2/<model>/detector/detection_summary.json
runs/CISR_v2/<model>/detector/detection_report.md
```

The v2 script deliberately stops after held-out detection evaluation. It does not fit a
new refusal gate or alter the existing hard-refusal intervention.

## Layout

```text
intentguard_refactor/
  configs/intentguard_families.json      # broader intent families + wrapper counts
  intentguard/dataset.py                 # counterfactual multimodal data builder
  intentguard/subspace.py                # S_I / S_R fitting, layer selection, scoring
  make_data.py                           # regenerate JSONL data under data/
  fit_subspaces.py                       # fit S_I and S_R over every extracted layer
  calibrate_thresholds.py                # dynamic model-specific thresholds
  apply_intervention.py                  # hard refusal intervention
  judge_outputs.py                       # Gemma3-12B default judge
  merge_audit.py                         # per-sample original/detection/post audit
  scripts/run_closed_loop_round.ps1      # end-to-end Windows pipeline
  scripts/run_detection_round_v2.sh      # held-out CISR_v2 detector pipeline
```

## Data

The generator expands the old setup to 13 intent families:

`animal_abuse`, `weapon`, `drug`, `fraud`, `financial_crime`, `privacy`,
`cyber`, `self_harm`, `violence`, `harassment`, `evasion`, `bio_chemical`, and
`sexual_exploitation`.

Each family gets paired target/control samples across a broader jailbreak
distribution:

- direct harmful requests;
- novice/practical-detail requests;
- fictional role-play;
- third-person advice to another actor;
- explicit policy-bypass requests;
- authority/research/audit framing;
- structured completion/checklist scaffolds;
- translation/paraphrase jailbreaks;
- format-constrained answers;
- confusing benign-prefix contexts;
- multi-turn continuation after an implied refusal;
- semantic images, OCR images, image-only OCR, text+OCR, semantic+OCR stitches,
  and irrelevant-image distractors.

All JSONL `image_path` values are rooted at `imgs/`. Semantic-image samples must
come from existing family-specific folders such as `imgs/weapon/`,
`imgs/cyber/`, or `imgs/bio/`; the generator now raises if an intent family has
no existing semantic image. If a family has fewer semantic images than requested,
the existing images are cycled instead of creating fallback icon art.

OCR prompt images and semantic+OCR stitched images are generated only when
needed and are written inside the corresponding family directory, for example
`imgs/weapon/_intentguard_generated/ocr/...`. They are not used as standalone
semantic images.

The harmful side uses realistic user-query forms such as direct "how do I ...",
role-play jailbreak, policy-bypass, research/audit framing, structured
completion, translation jailbreak, confusing-prefix, OCR, and multimodal
carriers. The benign side is hard benign: prevention, rejection, moderation,
reporting, de-escalation, or translate/paraphrase-without-answering controls.

```powershell
python intentguard_refactor/make_data.py `
  --config intentguard_refactor/configs/intentguard_families.json `
  --out data/intentguard_refactor_probe.jsonl `
  --summary-out data/intentguard_refactor_probe_summary.json `
  --repo-root .
```

## Closed Loop

The intended flow is:

1. Regenerate paired multimodal counterfactual data.
2. Extract activations with `--layers all`.
3. Generate original model responses.
4. Judge original responses with Gemma3-12B.
5. Fit the risk subspace `S_I` from target/control pairs.
6. Fit the refusal subspace `S_R` from refusal/non-refusal labels.
7. Select the best layer independently for `S_I` and `S_R`.
8. Calibrate model-specific thresholds for both scores.
9. Trigger hard refusal when `S_I >= threshold_I` and `S_R < threshold_R`.
10. Judge post-intervention responses and merge per-sample audit records.

Run the default pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File intentguard_refactor/scripts/run_closed_loop_round.ps1
```

On Linux/server shells, use the matching Bash entry:

```bash
bash intentguard_refactor/scripts/run_closed_loop_round.sh
```

The Bash entry runs all three target models by default:

```text
qwen25vl7b|Qwen/Qwen2.5-VL-7B-Instruct|qwen2_5_vl|hf
gemma3_12b|google/gemma-3-12b-it|generic_vlm|modelscope
llama32_11b_vision|LLM-Research/Llama-3.2-11B-Vision-Instruct|generic_vlm|modelscope
```

`MODEL_SPECS` also accepts the older five-field form
`alias|model|backend|layer|source`; the layer field is ignored because this
pipeline performs per-layer selection dynamically.

Gemma3-12B is also the default judge and is loaded from ModelScope:

```bash
JUDGE_MODEL=google/gemma-3-12b-it
JUDGE_MODEL_SOURCE=modelscope
JUDGE_BACKEND=generic_vlm
JUDGE_BATCH_SIZE=8
```

The judge uses the same prompt/parser/model runner as `jailbreak_repro`. Both
text-only and image-conditioned judging use batched `model.generate` calls. A
mixed logical batch is split once by image presence and then restored to input
order. Lower `JUDGE_BATCH_SIZE` if the judge GPU cannot fit the default batch.

Each stage is resumable. If the expected output file already exists and is
non-empty, the script reuses it. Set `FORCE=1` to rerun everything, or disable
individual stages with `RUN_ACTIVATIONS=0`, `RUN_GENERATION=0`,
`RUN_ORIGINAL_JUDGE=0`, `RUN_SUBSPACES=0`, `RUN_THRESHOLDS=0`,
`RUN_INTERVENTION=0`, `RUN_POST_JUDGE=0`, or `RUN_AUDIT=0`.

To run only one model, override `MODEL_SPECS`:

```bash
MODEL_SPECS="gemma3_12b|google/gemma-3-12b-it|generic_vlm|modelscope" \
bash intentguard_refactor/scripts/run_closed_loop_round.sh
```

Useful environment overrides:

```powershell
$env:MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
$env:MODEL_ALIAS="qwen25vl7b"
$env:MODEL_BACKEND="qwen2_5_vl"
$env:JUDGE_MODEL="google/gemma-3-12b-it"
$env:JUDGE_BACKEND="generic_vlm"
$env:JUDGE_MODEL_SOURCE="modelscope"
$env:JUDGE_BATCH_SIZE="8"
$env:RUN_DIR="runs/intentguard_refactor/qwen25vl7b"
$env:MAX_SAMPLES="40"   # smoke test
```

Main outputs:

```text
runs/intentguard_refactor/<model_alias>/
  activations_all_layers.npz
  original_generations/generation_results.jsonl
  original_judge/judge_results.jsonl
  subspaces/intent_subspace.npz
  subspaces/refusal_subspace.npz
  subspaces/subspace_selection.json
  thresholds.json
  intervention_results.jsonl
  post_judge/judge_results.jsonl
  sample_audit.jsonl
  sample_audit.csv
```

## Learned Safe-Layer Routing Scaffold

`apply_intervention.py` is intentionally retained as the hard-template
baseline. It does not edit a Transformer activation. The learned CSRL path is
implemented by these components:

```text
intentguard/routing.py              calibrated risk/refusal regions and oracle bypass
intentguard/intervention_data.py    train/teacher/retain/evaluation role assignment
intentguard/intervention.py         low-rank adapter, selected-layer hook, composite loss
build_intervention_manifest.py      joins detector and response-judge records
prepare_intervention_cache.py       aligns base/teacher hidden states by id and layer
train_intervention.py               cached-hidden representation pretraining
evaluate_intervention.py            strict oracle-bypass effect evaluation
configs/safe_layer_routing.example.json
```

Build the sample manifest after detection and response judging:

```powershell
python intentguard_refactor/build_intervention_manifest.py `
  --detections runs/CISR_v2/qwen25vl7b/detector/detection_results.jsonl `
  --judge runs/CISR_v2/qwen25vl7b/original_judge/judge_results.jsonl `
  --out runs/CISR_v2/qwen25vl7b/intervention/intervention_manifest.jsonl
```

The manifest enforces the following roles:

| Role | Definition | Use |
|---|---|---|
| `route_positive` | ground-truth risk with harmful output, including leaky refusals | safe-route training and efficacy evaluation |
| `safe_refusal_teacher` | ground-truth risk with a safe lexical refusal | teacher activation/completion source |
| `safe_target_control` | ground-truth risk with a safe non-refusal/pivot | non-regression teacher/control |
| `retain_benign` | benign and safely answered | KL/hidden-state retention |
| `over_refusal_control` | benign but safely refused | utility audit and retention, never reverse-steer automatically |
| `benign_judge_failure` | benign-labelled input judged harmful | manual label/judge audit before training |

Only `train` rows are fit-eligible. `validation` selects the adapter,
`calibration` selects routing thresholds, and `test` remains evaluation-only.
The detector data can therefore be reused, but detector labels alone are not
enough: each route-positive input also needs a safe target completion or safe
teacher hidden state.

After extracting prompt-side base activations and same-input, teacher-forced
safe-completion activations with matching IDs, build the training cache:

```powershell
python intentguard_refactor/prepare_intervention_cache.py `
  --manifest runs/CISR_v2/qwen25vl7b/intervention/intervention_manifest.jsonl `
  --base-activations runs/CISR_v2/qwen25vl7b/activations_all_layers.npz `
  --teacher-activations runs/CISR_v2/qwen25vl7b/intervention/safe_teacher_activations.npz `
  --layer 20 `
  --out runs/CISR_v2/qwen25vl7b/intervention/training_cache.npz
```

Teacher activations must use the same sample ID and selected-layer convention.
The cache builder fails closed when any train/validation `route_positive` lacks
a teacher row; calibration/test rows are never fitted and do not require one.

### Cached-Hidden Pretraining Contract

`train_intervention.py` expects an NPZ cache with:

```text
hidden                     float [n, d] or [n, tokens, d]
splits                     str   [n]
route_mask                  bool  [n]
retain_mask                 bool  [n]
teacher_hidden              float, optional, same shape as hidden
refusal_direction           float, optional, [d]
preserve_basis              float, optional, [rank, d]
```

At least `teacher_hidden` or `refusal_direction` is required. This entry point
trains the representation objectives. A model-aware second phase should feed
teacher-forced safe completions through the frozen VLM and additionally supply
`routed_logits`, `base_logits`, and `safe_labels` to
`compute_intervention_loss` for token CE and retention KL.

```powershell
python intentguard_refactor/train_intervention.py `
  --cache runs/CISR_v2/qwen25vl7b/intervention/training_cache.npz `
  --out runs/CISR_v2/qwen25vl7b/intervention/safe_layer_adapter.pt
```

### Runtime Hook

Attach `SafeLayerRouter` to the exact Transformer block selected in the model
adapter. Route decisions are request-local and the adapter is dormant outside
the context:

`detector.layer` is an `output_hidden_states` stack index, not automatically a
`ModuleList` index. In standard decoder stacks, hidden-state index `l` often
corresponds to block `l - 1`, because index 0 is the embedding output. Each model
adapter must capture the proposed module output once and verify numerical/shape
equivalence with the detector's cached hidden state before training or routing.

```python
from intentguard.intervention import (
    SafeLayerRouter,
    assert_hidden_equivalent,
    decisions_to_mask,
    load_adapter_artifact,
    resolve_module_path,
)
from intentguard.routing import RoutingThresholds, SynchronizedRoutingPolicy

selected_transformer_block = resolve_module_path(model, configured_module_path)
block_parameter = next(selected_transformer_block.parameters())
adapter, metadata = load_adapter_artifact(
    adapter_path, map_location=block_parameter.device
)
# Before deployment, compare one captured module output against the activation
# extractor's same-sample, same-token reference:
assert_hidden_equivalent(captured_hidden, cached_reference_hidden)
router = SafeLayerRouter(
    adapter.to(device=block_parameter.device, dtype=block_parameter.dtype)
)
policy = SynchronizedRoutingPolicy(thresholds)
decisions = [policy.decide(risk_score, safe_refusal_score)]

router.attach(selected_transformer_block)
with router.routing(
    decisions_to_mask(decisions, device=block_parameter.device),
    token_scope="last",
    token_mask=batch.get("attention_mask"),
    max_route_calls=1 + first_k_decode_steps,
):
    output = model.generate(**batch)
router.detach()
```

`safe_refusal_score` is specifically the probability of a safe refusal/pivot.
A lexical refusal that still leaks harmful content is a negative RSS example.
For causal intervention testing, pass `oracle_risk=True`; this bypasses both
detector gates and routes every ground-truth risk sample, including samples the
base model already handled safely. Report harmful-output reduction on base
failures and non-regression on base-safe risk samples separately.

After judging the forced-route outputs, enforce complete oracle coverage and
compute conversion/non-regression separately:

```powershell
python intentguard_refactor/evaluate_intervention.py `
  --manifest runs/CISR_v2/qwen25vl7b/intervention/intervention_manifest.jsonl `
  --base-judge runs/CISR_v2/qwen25vl7b/original_judge/judge_results.jsonl `
  --post-judge runs/CISR_v2/qwen25vl7b/oracle_intervention/judge_results.jsonl `
  --out runs/CISR_v2/qwen25vl7b/oracle_intervention/evaluation.json
```

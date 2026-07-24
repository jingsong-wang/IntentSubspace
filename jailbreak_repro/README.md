# Jailbreak Reproduction Framework

This folder is the unified place for reproducing multimodal jailbreak attacks,
defenses, model responses, and judge labels.

Implemented attacks/defenses:

- Attack: `FigStep`, loaded from `jailbreak_repro/sourcecode/FigStep-main`.
- Attack: `CS-DJ`, adapted from the official `image_embeding.py`,
  `Visual-Enhanced_Distraction.py`, and `main.py` construction flow.
- Attack: `JOOD`, adapted from the official `main.py` sample construction and
  `utils/mixaug.py`, `utils/strings.py`, `utils/randaug.py`.
- Attack: `UMK`, guarded as a white-box optimization attack. The official
  MiniGPT-4 artifact is available only through an explicit transfer-evaluation
  mode and is not treated as a target-model reproduction for Qwen/Gemma.
- Defense: `ECSO`, implemented as the official direct answer -> harm detect ->
  query-aware image caption -> text-only safe generation flow from
  `jailbreak_repro/sourcecode/ECSO-main`.
- Defense: `CIDER`, implemented as the official diffusion-denoising similarity
  detector with the fixed LLaVA-1.5 cross-modal encoder and hard refusal.
- Defense: `CISR`, a white-box prefill detector using a model-specific CISR_v2
  rank-3 subspace artifact and a lightweight coordinate network. Detected samples use
  the current fixed hard-refusal response.
- Defense: `AdaShield`, with separate released static (`AdaShield-S`) and
  victim-specific adaptive prompt-pool (`AdaShield-A`) modes.
- Defense: `HiddenDetect`, using the current victim's vocabulary-projected
  hidden states and victim-specific safety-aware layers.
- Detection baselines: `NEARSIDE`, `RCS-KCD`, `RCS-MCD`, and `VLMGuard`,
  exposed through a common model-specific representation artifact and
  `monitor`/`block` interface.

See `METHOD_FIDELITY.md` before reporting results. It records which adapters are
paper-grade reproductions and which modes are only transfer/artifact baselines.

## Quick Smoke Test

Run the full framework without downloading a VLM:

```bash
python -m jailbreak_repro.run_experiment \
  --model mock \
  --model-backend mock \
  --attack figstep \
  --defense ecso \
  --dataset tiny \
  --max-samples 3 \
  --out-dir jailbreak_repro/runs/smoke_figstep_ecso
```

Outputs are written under the selected run directory:

- `samples.jsonl`: normalized FigStep samples.
- `responses.jsonl`: direct response, ECSO intermediates, final response.
- `judge_results.jsonl`: judge labels.
- `summary.json`: aggregate metrics.
- `report.md`: readable report.

Default run directories are deterministic and do not include timestamps:

```text
jailbreak_repro/runs/<victim-model>/<source>/<defense>/n_<N-or-all>__cfg_<response-hash>/
```

Changing only the judge model keeps the same response directory and writes a new
judge-specific result under:

```text
<run-dir>/judges/<judge-key>/judge_results.jsonl
```

The root `judge_results.jsonl` is kept as the latest judge result for backward
compatibility.

## Real VLM Run

Example with Qwen2.5-VL:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --attack figstep \
  --defense ecso \
  --dataset tiny \
  --max-samples 10 \
  --dtype bfloat16 \
  --device auto \
  --judge-mode model \
  --judge-preset qwen25vl7b \
  --out-dir jailbreak_repro/runs/qwen25vl_figstep_ecso_smoke
```

Use `--defense none` for the direct FigStep baseline. Use `--dataset safebench`
for the full FigStep SafeBench split.

## AdaShield Defense

The adapter follows the [AdaShield paper](https://arxiv.org/abs/2403.09513)
and the released repository separately for its two variants.

AdaShield-S loads the exact released static prompt and uses the official target
wrapper construction `query + defense_prompt + query`:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --attack figstep \
  --dataset safebench \
  --defense adashield \
  --adashield-mode static \
  --judge-mode model \
  --judge-preset qwen25vl7b
```

AdaShield-A is an optimization method. Its prompt pool must be trained against
the selected victim; a pool learned with LLaVA is not silently reused for Qwen
or Gemma. Train the core released auto-refinement flow using the official
FigStep five-training/two-validation split, four refinement rounds, and
`alpha=0.8`:

```bash
python -m jailbreak_repro.train_adashield \
  --model-preset qwen25vl7b \
  --defender-model lmsys/vicuna-13b-v1.5 \
  --device cuda:0 \
  --defender-device cuda:1 \
  --out runs/AdaShield/qwen25vl7b/pool.json
```

The paper uses Vicuna-v1.5-13B as the defender. Victim and defender are loaded
together during refinement, so assign separate GPUs or place the defender on
CPU when memory is limited. The resulting pool records the exact victim id and
can be evaluated with the paper's CLIP ViT-B/32 concatenated image-text
retrieval and `beta=0.7` gate:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --attack figstep \
  --defense adashield \
  --adashield-mode adaptive \
  --adashield-prompt-pool runs/AdaShield/qwen25vl7b/pool.json \
  --judge-mode model \
  --judge-preset qwen25vl7b
```

Existing outputs from the official training scripts can be converted without
changing their prompts:

```bash
python -m jailbreak_repro.build_adashield_pool \
  --table-dir /path/to/victim-specific/adashield/wandb \
  --victim-model Qwen/Qwen2.5-VL-7B-Instruct \
  --out runs/AdaShield/qwen25vl7b/pool.json
```

The repository does not ship any trained `final_table.csv`. The paper also
describes a GPT-4 rephrasing stage, but neither released training entrypoint
calls the rephrase code. This unresolvable paper/code gap is recorded as
`paper_training_complete=false`; the framework does not invent rephrased
prompts or mark such pools as complete paper-training reproductions.

## HiddenDetect Defense

[HiddenDetect](https://arxiv.org/abs/2502.14744) projects every language
layer's final-token state through the current victim's final normalization and
LM head, then measures cosine alignment with the released refusal token set.
The LLaVA 16-29 and Qwen-VL 21-24 layer ranges are not reused. On first use,
the framework runs the official 12 safe/unsafe few-shot examples through the
selected Qwen, Gemma, or LLaVA victim and selects layers satisfying
`FDV_l > FDV_last`.

Paper-faithful monitoring mode:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset gemma3_12b \
  --benchmark XSTest \
  --defense hiddendetect \
  --hiddendetect-action monitor \
  --judge-mode model \
  --judge-preset qwen25vl7b
```

Profiles are cached at `runs/HiddenDetect/<model-preset>/profile.json`. Each
response stores all layer scores, selected scores, trapezoidal safety score,
threshold, and detection. `summary.json` reports AUPRC and AUROC whenever both
safe and unsafe labels are available, such as XSTest.

For a repository-isolated reproduction on all three supported victims, use the
dedicated launcher. It verifies the released HiddenDetect checkout and 12-shot
data in `jailbreak_repro/sourcecode/HiddenDetect-main`, reads XSTest from
`benchmark/XSTest`, and delegates CS-DJ unchanged to the existing attack
adapter:

```bash
python -m pip install -r requirements-hiddendetect.txt

bash jailbreak_repro/run_hiddendetect_reproduction.sh \
  --phase all \
  --models qwen25vl7b,gemma3_12b,llama32_11b_vision \
  --sources xstest,csdj
```

Use `--download-missing` only to clone a missing HiddenDetect checkout or fetch
a missing XSTest CSV. It never downloads or relocates CS-DJ data. CS-DJ keeps
using its current `sourcecode/CS-DJ-main` data and shared generated artifacts;
existing overrides such as `--csdj-image-dir` can be passed after `--`.

The launcher has no RCS preparation, activation, training, or dependency
preflight. Completed model/source cases are skipped using the official few-shot
fingerprint, while `--force` explicitly recomputes them. Final tables are
written as `external_detection_summary.{json,csv,md}` under the output root.

The paper evaluates threshold-free AUROC and explicitly states that
HiddenDetect does not alter response generation. It does not release a decision
threshold. The cached profile therefore labels its default balanced-accuracy
threshold on the official 12-shot set as a platform calibration. To turn the
detector into an active gate, use the explicit non-paper extension:

```text
--hiddendetect-action block [--hiddendetect-threshold VALUE]
```

`block` skips victim generation for detected inputs and returns a fixed refusal,
while retaining `paper_claim_compatible=false` for the intervention.

## Representation Detector Study

The representation study compares four distinct assumptions against CISR using
the same all-layer activation archives:

- `nearside`: final-layer paired mean direction and released mean-projection
  threshold. Its original scope is adversarial-image detection; training it on
  CISR intent pairs is an explicit core-equation adaptation.
- `rcs-kcd` and `rcs-mcd`: RCS geometric layer selection, learned projection,
  and contrastive K-nearest or Mahalanobis scoring. The released RCS source is
  vendored under `sourcecode/Jailbreak_Detection_RCS-main`.
- `vlmguard`: low-contamination SVD pseudo-partition followed by the paper's
  three-layer ReLU prompt classifier. The official repository contained no
  executable implementation when checked on 2026-07-17, so this is marked as a
  paper-spec implementation rather than an official-code reproduction.

Train one artifact without touching CISR result directories:

```bash
python -m jailbreak_repro.train_representation_detector \
  --activations runs/CISR_v2/qwen25vl7b/activations_all_layers.npz \
  --method rcs-kcd \
  --out runs/representation_baselines/CISR_v2/qwen25vl7b/rcs-kcd/detector.npz
```

Run it through the unified platform in detection-only monitoring mode:

```bash
python -m jailbreak_repro.run_selected \
  --victim-model qwen25vl7b \
  --benchmark XSTest \
  --defense rcs-kcd \
  --judge-model none \
  --representation-detector runs/representation_baselines/CISR_v2/qwen25vl7b/rcs-kcd/detector.npz \
  --representation-action monitor \
  --max-new-tokens 1
```

The resumable matrix launcher covers Qwen2.5-VL-7B, Gemma-3-12B, and
Llama-3.2-11B-Vision on held-out CISR data, XSTest, and CS-DJ:

```bash
bash jailbreak_repro/run_representation_detector_study.sh --phase all
```

Each artifact records whether the score follows released code, a paper
specification, or only a matched CISR adaptation. External test examples are
never added to detector training.

### Repository-grade representation reproduction

`run_representation_detector_study.sh` above is the controlled matched-CISR
comparison. The repository-grade path is separate:

```bash
python -m jailbreak_repro.sync_representation_upstreams
bash jailbreak_repro/run_repository_representation_reproductions.sh --help
```

The upstream audit records the exact git `HEAD` when a real checkout is
available and verifies the files used by each adapter. It also makes public
availability limits explicit:

| Method | Public repository state | Training source used by the launcher |
| --- | --- | --- |
| HiddenDetect | executable code and official 12-shot data | official 12-shot profile, victim-specific |
| NEARSIDE | executable code; full RADAR raw image corpus is not released in the repository | matched CISR pairs unless a separately obtained RADAR archive is supplied |
| RCS-KCD/MCD | executable code and downloaders; four datasets are manual/one-link downloads | exact released 2,000-example training composition |
| VLMGuard | repository is README-only | paper-spec implementation on the declared matched archive |
| SAHs | repository is README-only | not presented as an executable reproduction |
| JailNeurons | code has missing/hard-coded helpers | audited but not presented as a completed reproduction |

Prepare the RCS data with the released downloader and its recommended manual
archive, then run the complete resumable matrix:

```bash
python -m pip install -r requirements.txt
python -m gdown 1V09sherPVm6M0E_J_xz3uJ6IBrZ66cRV -O /tmp/rcs_manual_data.zip

bash jailbreak_repro/run_repository_representation_reproductions.sh \
  --phase all \
  --download-data \
  --manual-data-archive /tmp/rcs_manual_data.zip \
  --models qwen25vl7b,gemma3_12b,llama32_11b_vision \
  --methods hiddendetect,nearside,rcs-kcd,rcs-mcd,vlmguard \
  --sources xstest,csdj
```

Formal RCS training fails closed unless all released source counts and image
paths are present. `--allow-incomplete-data` is available only for pipeline
smoke tests; its artifact is marked `repository-rcs-incomplete` and cannot set
`paper_training_protocol=true`.

The launcher skips completed model/stage combinations and writes only under
`runs/representation_repository_repro/`. Its final files are:

```text
external_detection_summary.json
external_detection_summary.csv
external_detection_summary.md
```

The table reports frozen-threshold AUROC/TPR/FPR. On XSTest, FPR is the safe
prompt over-detection rate; on CS-DJ, `1-TPR` is the attack miss rate. Neither
benchmark is used to select a layer, fit a projection, or calibrate a threshold.

### CNRF Oracle ceiling diagnostic

The CNRF Oracle adapter is intentionally separate from the frozen-threshold
representation reproductions above. It reuses an existing CNRF work directory,
then lets test/external labels select operating thresholds, counterfactual-axis
subsets, and candidate arrow-pack subsets. Its outputs are diagnostic ceilings,
not OOD generalization estimates, and always retain `oracle_only=true` and
`paper_claim_compatible=false`.

```bash
WORK=counterfactual_risk_field/work/v2_axes_temp07 \
MODEL_TAG=qwen25vl7b \
bash jailbreak_repro/run_cnrf_oracle.sh
```

Results are written under
`jailbreak_repro/runs/cnrf_oracle/<model>/<work-name>/`. The `raw/` directory
contains the complete CNRF axis/pack search, while `summary.json`,
`summary.csv`, and `summary.md` provide the reproduction-platform view. The
axis search is exhaustive; the pack search covers LOO-ranked and random
candidates and is not a global combinatorial optimum.

## CIDER Defense

CIDER is a target-model-independent front-end detector. Qwen2.5-VL-7B or
Gemma-3-12B remains the victim model, while the paper's detector uses a fixed
LLaVA-1.5-7B cross-modal encoder. Replacing that encoder with the victim model
is an ablation, not a reproduction of the released detector.

This is explicit in the paper rather than an inference: Section 2.2 states that
the LLaVA-v1.5-7B image and text encoders capture the semantic meanings, and
Section 4.1 states that CIDER is an MLLM-independent auxiliary model that uses
LLaVA for both modalities. Therefore the LLaVA, InstructBLIP, MiniGPT4,
Qwen-VL, and GPT-4V results share the same CIDER semantic encoder; only the
response-generating victim changes.

The current official `utils.py` also contains a `QwenEncoder` branch, but this
is not the configuration reported by the paper: `settings/settings.yaml` does
not define the branch's required `Embed_model_path`, the README requires
LLaVA-1.5-7B for CIDER, and lists Qwen/InstructBLIP/MiniGPT4 separately as
response models. We retain this distinction instead of interpreting that
incomplete branch as a per-victim detector.

The official repository does not include its diffusion checkpoint. Download it
to the path expected by the adapter:

```bash
wget -c -O jailbreak_repro/sourcecode/CIDER-main/code/models/diffusion_denoiser/imagenet/256x256_diffusion_uncond.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

The LLaVA encoder can be a Hugging Face id or an already-downloaded local path.
Its precision defaults to the official `float16`, independently of the victim's
`--dtype`. Default `--cider-encoder-mode paper_llava15` validates the loaded
model class and LLaVA-1.5-7B text/vision dimensions. A renamed local directory
is accepted when its actual architecture matches.

Qwen2.5-VL-7B example:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --attack figstep \
  --defense cider \
  --dtype bfloat16 \
  --cider-encoder-mode paper_llava15 \
  --cider-encoder-model llava-hf/llava-1.5-7b-hf \
  --cider-dtype float16 \
  --judge-mode model \
  --judge-preset qwen25vl7b
```

Gemma-3-12B example; the victim weights still resolve through ModelScope:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset gemma3_12b \
  --attack figstep \
  --defense cider \
  --dtype bfloat16 \
  --cider-encoder-mode paper_llava15 \
  --cider-encoder-model /path/to/llava-1.5-7b-hf \
  --cider-dtype float16 \
  --judge-mode model \
  --judge-preset qwen25vl7b
```

The default is the released threshold `-0.003936767578125`. To reproduce the
paper's 95% clean-pass calibration, add:

```bash
--cider-calibration-image-dir jailbreak_repro/sourcecode/CIDER-main/data/img/clean \
--cider-calibration-text-file jailbreak_repro/sourcecode/CIDER-main/data/text/valset.csv \
--cider-calibration-pass-rate 0.95
```

The console exposes separate `CIDER denoising`, `CIDER threshold calibration`,
and `CIDER cross-modal detection` progress bars. Denoised checkpoints and the
`detections.jsonl` resume cache are stored in `<run-dir>/cider_artifacts/`.
CIDER batches diffusion denoising and the eight checkpoint embeddings without
changing the official computations. The defaults are
`--cider-denoise-batch-size 50` and `--cider-encoder-batch-size 8`; reduce the
former first if preprocessing runs out of GPU memory.
CIDER models are released before the victim is loaded, and the victim is
released before the judge is loaded. Detected samples receive the exact paper
hard refusal without running the victim; passed samples use resized checkpoint
0, matching the official flow.

`--cider-denoiser dncnn` and
`--cider-encoder-mode custom_llava_ablation` are supported for ablation only.
There is intentionally no per-victim encoder mode. CIDER was proposed for
optimization-based adversarial attacks; results on typographic or compositional
attacks should be labeled as out-of-scope generalization evaluations.

## CISR Defense

Train model-specific detector artifacts first:

```bash
bash intentguard_refactor/scripts/run_detection_round_v2.sh
```

Then call CISR from the unified reproduction runner:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --benchmark HADES \
  --defense cisr2 \
  --cisr-detector runs/CISR_v2/qwen25vl7b/detector/detector.npz \
  --judge-mode model \
  --judge-preset gemma3_12b
```

Use `--defense cisr3` with a detector from `runs/CISR_v3/`. The legacy
`--defense cisr` spelling inspects the artifact and is normalized to `cisr2` or
`cisr3` before the output directory is selected. Explicitly naming the wrong
version fails before victim-model loading. Results are stored under distinct
`defense_cisr2/` and `defense_cisr3/` paths, and every response and summary records
the resolved version.

The detector artifact must match the victim model. A cross-model transfer ablation must
be made explicit with `--cisr-allow-model-mismatch`. `--cisr-threshold` overrides the
validation-calibrated threshold for threshold sweeps. Each response row records raw and
multimodal-residual coordinates, probability, threshold, layer, and trigger state.

`run_experiment` does not keep the victim model and judge model in GPU memory at
the same time. It writes `responses.jsonl`, releases the victim runner and CUDA
cache, then loads the judge model. If memory is still tight, run with
`--judge-mode none` first and judge the saved `responses.jsonl` in a separate
process.

Runs resume by default. If `response_config.json` matches the current CLI,
existing `responses.jsonl` is reused and the victim model is not loaded. If the
same `judge_config.json` already exists under `judges/<judge-key>/`, judge
labels are reused too. Use `--force-responses` or `--force-judge` to recompute a
stage.

Judge tasks are selected with `--judge-task auto|asr|xstest`. `auto` keeps the
ASR harmfulness judge for jailbreak attacks and safety benchmarks, but switches
to the original XSTest refusal/compliance classifier when `--benchmark XSTest`
is used. The XSTest task returns `1_full_compliance`, `2_full_refusal`, or
`3_partial_refusal`, and reports over-refusal/failure metrics instead of ASR.
The original XSTest `safe`/`unsafe` label is preserved as `xstest_label` and
reported separately under `summary.json -> judge -> by_safety_label`:

- `safe`: reports compliance and `safe_overrefusal_rate`.
- `unsafe`: reports refusal and `unsafe_compliance_rate`.

Each group also includes its own sample count, valid-judgment count, parse-error
count, judge coverage, and three-class distribution. Rates use valid judgments
within that label group as the denominator.

Convenience launcher:

```bash
python -m jailbreak_repro.run_selected \
  --victim-model qwen25vl7b \
  --attack figstep \
  --defense ecso \
  --judge-model qwen25vl7b \
  --max-samples 10

python -m jailbreak_repro.run_selected \
  --victim-model qwen25vl7b \
  --benchmark jailbreakV-mini \
  --defense ecso \
  --judge-model qwen25vl7b
```

Unknown arguments are forwarded to `run_experiment`, so method-specific options
such as `--csdj-image-dir`, `--jood-scenarios`, or `--umk-mode` still work.

## Unified CNRF Oracle Deployment Evaluation

The deployable Oracle evaluation freezes one artifact per target model. Each
artifact contains exactly one `image_text` candidate and one `text` candidate,
both selected with `abstain_safe`, empirical FPR <= 5%, the cross-attack
`macro_harmful` objective, and a 25-pack budget. The same modality candidate and
threshold are used for CS-DJ, JOOD, JailbreakV-mini, and XSTest; there is no
per-benchmark Oracle. This remains an `ORACLE_ONLY` ceiling because test/external
labels participated in the unified candidate and threshold selection.

Build the Qwen artifact after `run_cnrf_oracle.sh` has completed:

```bash
python -m jailbreak_repro.build_cnrf_oracle_artifact \
  --work counterfactual_risk_field/work/v2_axes_temp07 \
  --model-tag qwen25vl7b \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct
```

The builder re-scores all Oracle evaluation rows and fails unless every stored
group TPR/FPR is reproduced exactly. Then run the aligned full-report matrix and
judge all responses with Gemma3-12B:

```bash
bash jailbreak_repro/run_cnrf_oracle_full_eval.sh \
  --victim-model qwen25vl7b \
  --judge-model gemma3_12b \
  --judge-batch-size 8
```

This evaluates CS-DJ (750), the existing report-aligned JOOD protocol (500),
JailbreakV-mini (280), and XSTest (450). It writes `summary.json`, `summary.csv`,
and `summary.md` under
`jailbreak_repro/runs/cnrf_oracle/<victim>/full_eval/`. Attack rows report
Gemma-judged ASR. XSTest reports the safe-subset over-refusal rate as well as the
detector FPR. Pass a different `--jood-max-samples` only when intentionally
changing the JOOD protocol; `9350` evaluates the repository's entire generated
JOOD configuration rather than the report-aligned 500-row setting.

For a Gemma3-12B target, first extract Gemma representations and repeat CNRF
view fitting, Oracle selection, artifact freezing, and the same response/judge
matrix:

```bash
python -m ood_intent_study.extract \
  --manifest counterfactual_risk_field/work/v2_axes_temp07/experiment.jsonl \
  --out-dir counterfactual_risk_field/work/v2_axes_temp07/activations/gemma3_12b \
  --model-name gemma3_12b \
  --model google/gemma-3-12b-it \
  --model-source modelscope \
  --backend gemma3 \
  --layers all \
  --readouts last,non_image_mean \
  --dtype bfloat16 \
  --storage-dtype float32 \
  --device-map auto \
  --attn-implementation sdpa \
  --shard-size 16 \
  --resume \
  --fail-fast

python -m counterfactual_risk_field.scripts.run_experiment \
  --manifest counterfactual_risk_field/work/v2_axes_temp07/experiment.jsonl \
  --activations counterfactual_risk_field/work/v2_axes_temp07/activations/gemma3_12b \
  --out-dir counterfactual_risk_field/work/v2_axes_temp07/results/gemma3_12b \
  --config counterfactual_risk_field/configs/protocol_v2_diverse_axes.json

WORK=counterfactual_risk_field/work/v2_axes_temp07 \
MODEL_TAG=gemma3_12b \
bash jailbreak_repro/run_cnrf_oracle.sh

python -m jailbreak_repro.build_cnrf_oracle_artifact \
  --work counterfactual_risk_field/work/v2_axes_temp07 \
  --model-tag gemma3_12b \
  --model-id google/gemma-3-12b-it

bash jailbreak_repro/run_cnrf_oracle_full_eval.sh \
  --victim-model gemma3_12b \
  --victim-source modelscope \
  --judge-model gemma3_12b \
  --judge-batch-size 8
```

## Complete Experiment Matrix

`run_all_experiments.sh` runs every requested attack/benchmark against every
defense. It produces 36 sequential configurations:

- `figstep`, `csdj`, `jood` x `none`, `ecso`, `cider`, selected `cisr2`/`cisr3`, `adashield`, `hiddendetect`.
- `HADES`, `jailbreakV-mini`, `XSTest` x the same six defenses. Benchmark
  cases are the framework's `attack=none` mode.

The matrix defaults to `AdaShield-S` and HiddenDetect `monitor`, because those
require no unshipped prompt pool and preserve the paper's no-intervention
HiddenDetect behavior. Select adaptive/block modes with
`--adashield-mode adaptive --adashield-prompt-pool ...` and
`--hiddendetect-action block`.

Matching `responses.jsonl` and judge results are reused. The actual ASR or
XSTest prompt content is part of the judge configuration hash, so a prompt
change automatically selects a new judge directory and recomputes labels;
unchanged configurations skip judge inference. Use the script's
`--force-judge` only when intentional recomputation is required. Model judging
uses real text-only batch inference with batch size 8 by default. XSTest retries
only the failed-to-parse members of a batch. `--judge-include-image` remains
supported but falls back to one sample per generation call.
Attack preprocessing artifacts are shared under
`jailbreak_repro/runs/_shared_attack_artifacts/<attack>/`, so CS-DJ image
embedding, selection, question splitting, and composition are not repeated for
each defense.
JOOD defaults to `--max-samples 500` in the matrix script because its full
configuration contains 9,350 samples. Change it with `--jood-max-samples`; the
global `--max-samples` option overrides this per-method default for smoke tests.

Qwen example:

```bash
bash jailbreak_repro/run_all_experiments.sh \
  --victim-model qwen25vl7b \
  --judge-model qwen25vl7b \
  --judge-batch-size 8 \
  --cisr-version cisr2 \
  --cisr-detector runs/CISR_v2/qwen25vl7b/detector/detector.npz
```

Gemma victim with ModelScope weights:

```bash
bash jailbreak_repro/run_all_experiments.sh \
  --victim-model gemma3_12b \
  --victim-source modelscope \
  --judge-model qwen25vl7b \
  --judge-batch-size 8 \
  --cisr-version cisr2 \
  --cisr-detector runs/CISR_v2/gemma3_12b/detector/detector.npz \
  --cider-encoder-model /path/to/llava-1.5-7b-hf
```

LLaVA-1.5-7B victim:

```bash
bash jailbreak_repro/run_all_experiments.sh \
  --victim-model llava15_7b \
  --judge-model qwen25vl7b \
  --dtype float16 \
  --cisr-version cisr2 \
  --cisr-detector runs/CISR_v2/llava15_7b/detector/detector.npz
```

Train the model-specific CISR detector before enabling `--defense cisr2` or
`--defense cisr3`:

```bash
MODEL_SPECS='llava15_7b|llava-hf/llava-1.5-7b-hf|generic_vlm|hf' \
  bash intentguard_refactor/scripts/run_detection_round_v2.sh
```

For CIDER runs, the fixed paper encoder and the LLaVA victim resolve to the same
weights, but they are loaded in separate stages and never remain in GPU memory
together.

The script checks the model-specific CISR artifact and official CIDER diffusion
checkpoint before starting. It continues after an individual configuration
fails and reports all failures at the end; add `--fail-fast` to stop
immediately. Validate paths and all generated commands without inference using:

```bash
bash jailbreak_repro/run_all_experiments.sh --dry-run --max-samples 1
```

## Qwen, Gemma, And LLaVA

The reproduction runner reuses the model loading helpers from the repository
root under `src/extract_activations.py`.

Available presets:

```text
--model-preset qwen25vl7b    -> Qwen/Qwen2.5-VL-7B-Instruct, qwen2_5_vl, hf
--model-preset gemma3_12b    -> google/gemma-3-12b-it, generic_vlm, modelscope
--model-preset llava15_7b    -> llava-hf/llava-1.5-7b-hf, generic_vlm, hf
--model-preset llama32_11b_vision -> LLM-Research/Llama-3.2-11B-Vision-Instruct, generic_vlm, modelscope
```

LLaVA victim example:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset llava15_7b \
  --attack figstep \
  --defense ecso \
  --dtype float16 \
  --device auto \
  --judge-mode model \
  --judge-preset qwen25vl7b
```

The direct equivalent is:

```text
--model llava-hf/llava-1.5-7b-hf --model-backend llava --model-source hf
```

LLaVA uses the generic multimodal runner with its own chat template and image
processor. CISR requires a detector trained specifically from this LLaVA
victim; Qwen or Gemma detector artifacts are not interchangeable.

Gemma weights are expected in ModelScope, matching the main project scripts. If
you pass a Gemma model name directly and leave `--model-source auto`, the runner
also resolves it to `modelscope`:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset gemma3_12b \
  --attack umk \
  --defense none \
  --umk-corpus advbench \
  --max-samples 10
```

Equivalent explicit form:

```bash
python -m jailbreak_repro.run_experiment \
  --model google/gemma-3-12b-it \
  --model-backend gemma \
  --model-source modelscope \
  --attack umk \
  --defense none
```

Judge models support the same presets:

```text
--judge-mode model --judge-preset qwen25vl7b
--judge-mode model --judge-preset gemma3_12b
--judge-mode model --judge-preset llava15_7b
```

## Attack Adapters

All generated attack artifacts are written under the run directory:

```text
<out-dir>/attack_artifacts/<attack>/
```

CS-DJ can either consume precomputed official artifacts or generate them. The
default source layout used by this framework is:

```text
jailbreak_repro/sourcecode/CS-DJ-main/
  data/images/                 # real LLaVA-CC3M image library
  instructions/*.json
  Super Moods.ttf
```

The legacy official `CS-DJ-main/llava_images/` layout is also detected. When
neither default exists, use `--csdj-image-dir`. A fully reproduced run executes
the official CLIP image embedding and distant-image selection, Qwen2.5-3B
three-question split, and 9 distraction + 3 text-image composition:

```bash
python -m jailbreak_repro.run_experiment \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-backend qwen2_5_vl \
  --attack csdj \
  --defense ecso \
  --csdj-category all \
  --csdj-num-images 10000 \
  --max-samples 10
```

The generated embedding map, distraction-image map, sub-questions, and final
12-panel images are cached under `<out-dir>/attack_artifacts/csdj/`. Rerunning
the same configuration resumes missing map entries and sub-questions instead
of repeating completed preprocessing.

Long CS-DJ preprocessing stages expose separate progress bars:

```text
CS-DJ sub-question split: ... cached=<count>, attempts=<1..6>
CS-DJ 12-panel images: ... built=<count>, reused=<count>
```

The first bar begins after the auxiliary Qwen weights and tokenizer are loaded.
Each completed split is persisted immediately. The splitter is released before
the image-composition stage, and existing final images are reused on resume.

The official Qwen2.5-3B auxiliary splitter supports the same model-source
resolution as victim models. `--csdj-aux-model-source auto` (the default) uses
ModelScope for `Qwen/*`, avoiding Hugging Face Xet authentication failures. It
can also be selected explicitly:

```text
--csdj-aux-model-source modelscope
--csdj-aux-model-revision master
--csdj-aux-model-cache-dir /path/to/modelscope/cache
```

If you already ran the official CS-DJ preprocessing, reuse the exact artifacts
and avoid loading the auxiliary Qwen splitter:

```bash
python -m jailbreak_repro.run_experiment \
  --model mock \
  --model-backend mock \
  --attack csdj \
  --defense none \
  --csdj-image-dir /path/to/data/images \
  --csdj-image-map /path/to/distraction_image_map_seed_0_num_100.json \
  --csdj-subquestions-file /path/to/subquestions.json \
  --csdj-aux-model none
```

JOOD expects the official AdvBenchM layout:

```text
AdvBenchM/
  images/harmful/<scenario>/*.png
  images/harmless/*.png
  prompts/all_instructions/<scenario>.json
```

Example:

```bash
python -m jailbreak_repro.run_experiment \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-backend qwen2_5_vl \
  --attack jood \
  --defense none \
  --jood-dataset-dir /path/to/AdvBenchM \
  --jood-scenarios Illegal_Activity \
  --jood-aug mixup \
  --jood-lams 0.5
```

UMK is a white-box optimization attack. For paper-grade results on Qwen/Gemma,
the adversarial image/text must be optimized against the same victim model used
by `--model`. The official `bad_vlm_prompt.bmp` bundled with UMK was optimized
for the paper's MiniGPT-4 setup, so evaluating it on Qwen/Gemma is only a
transfer-artifact baseline.

By default, `--attack umk` refuses to run until a target-model optimizer is
available:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --attack umk \
  --defense none
```

To evaluate an artifact that you have optimized for the exact current victim
model, provide both the image and the model id it was optimized against:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --attack umk \
  --umk-mode target_optimized_artifact \
  --umk-image-path /path/to/qwen25vl7b_optimized_bad_vlm_prompt.bmp \
  --umk-optimized-for-model Qwen/Qwen2.5-VL-7B-Instruct \
  --defense none \
  --umk-corpus advbench \
  --max-samples 10
```

To explicitly run the official MiniGPT-4 artifact as a transfer baseline:

```bash
python -m jailbreak_repro.run_experiment \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-backend qwen2_5_vl \
  --attack umk \
  --umk-mode transfer_eval \
  --defense none \
  --umk-corpus advbench \
  --max-samples 10
```

The adapters fail fast when required official datasets or artifacts are missing.
They do not substitute synthetic samples or transfer artifacts for formal
target-model reproduction runs.

## Benchmark Mode

Benchmark mode reads data from the repository-level `benchmark/` directory and
does not apply any attack adapter. Because benchmark selection and attack
selection are mutually exclusive, use `--benchmark` with `--attack none` or omit
`--attack`:

```bash
python -m jailbreak_repro.run_experiment \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-backend qwen2_5_vl \
  --benchmark HADES \
  --defense ecso \
  --max-samples 10 \
  --dtype bfloat16 \
  --device auto \
  --out-dir jailbreak_repro/runs/qwen25vl_hades_ecso_smoke
```

Text-only benchmark example:

```bash
python -m jailbreak_repro.run_experiment \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-backend text \
  --benchmark XSTest \
  --defense none \
  --judge-task xstest \
  --max-samples 20 \
  --out-dir jailbreak_repro/runs/qwen25_xstest_direct_smoke
```

`--benchmark` can be a name under `benchmark/`, a benchmark directory, or a
specific `.jsonl`, `.csv`, or `.json` file. Directory loading first uses known
adapters for `HADES`, `XSTest`, and `JailBreakV`; otherwise it picks the first
data file and normalizes common fields such as `prompt`, `query`, `question`,
`image`, `image_path`, and `label`.

JailBreakV aliases:

```bash
python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --benchmark jailbreakV \
  --defense ecso \
  --max-samples 10

python -m jailbreak_repro.run_experiment \
  --model-preset qwen25vl7b \
  --benchmark jailbreakV-mini \
  --defense ecso
```

`jailbreakV` resolves to `benchmark/JailBreakV_28K` when present.
`jailbreakV-mini` first looks for the official `mini_JailBreakV_28K.csv`;
if only `JailBreakV_28K.csv` is available, it uses the official
`selected_mini` annotations. Missing files fail fast instead of substituting
synthetic samples.

## Interface

The CLI selects the three key axes explicitly:

```text
--model <model-or-local-path>
--model-preset mock|qwen25vl7b|gemma3_12b
--model-backend auto|text|qwen|qwen2_5_vl|gemma|generic_vlm
--model-source auto|hf|modelscope
--attack none|figstep|csdj|jood|umk
--benchmark <name-or-path>
--defense none|ecso|cider|cisr2|cisr3|adashield|hiddendetect|nearside|rcs-kcd|rcs-mcd|vlmguard|cnrf-oracle
--cisr-detector <runs/CISR_v2|CISR_v3/<model>/detector/detector.npz>
--representation-detector <runs/representation_baselines/.../detector.npz>
--representation-action monitor|block
--adashield-mode static|adaptive
--hiddendetect-action monitor|block
```

Use either `--attack <method>` or `--benchmark ...`, not both. `--attack none`
means the samples come directly from a benchmark rather than from an attack
adapter, so it must be paired with `--benchmark`.

Internally the framework normalizes every sample to a JSONL row with:

- `prompt`: harmful source request used by the judge.
- `prompt_text`: actual user-visible attack prompt sent to the model.
- `image_path`: attack image sent to the VLM.
- `response`: final answer after defense.

This keeps the judge compatible with the existing project flow while preserving
the real multimodal prompt that the victim model saw.

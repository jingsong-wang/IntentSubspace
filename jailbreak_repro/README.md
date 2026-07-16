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
threshold, and detection. `summary.json` reports AUROC whenever both safe and
unsafe labels are available, such as XSTest.

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
  --defense cisr \
  --cisr-detector runs/CISR_v2/qwen25vl7b/detector/detector.npz \
  --judge-mode model \
  --judge-preset gemma3_12b
```

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

## Complete Experiment Matrix

`run_all_experiments.sh` runs every requested attack/benchmark against every
defense. It produces 36 sequential configurations:

- `figstep`, `csdj`, `jood` x `none`, `ecso`, `cider`, `cisr`, `adashield`, `hiddendetect`.
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
  --cisr-detector runs/CISR_v2/qwen25vl7b/detector/detector.npz
```

Gemma victim with ModelScope weights:

```bash
bash jailbreak_repro/run_all_experiments.sh \
  --victim-model gemma3_12b \
  --victim-source modelscope \
  --judge-model qwen25vl7b \
  --judge-batch-size 8 \
  --cisr-detector runs/CISR_v2/gemma3_12b/detector/detector.npz \
  --cider-encoder-model /path/to/llava-1.5-7b-hf
```

LLaVA-1.5-7B victim:

```bash
bash jailbreak_repro/run_all_experiments.sh \
  --victim-model llava15_7b \
  --judge-model qwen25vl7b \
  --dtype float16 \
  --cisr-detector runs/CISR_v2/llava15_7b/detector/detector.npz
```

Train the model-specific CISR detector before enabling `--defense cisr`:

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
--defense none|ecso|cider|cisr|adashield|hiddendetect
--cisr-detector <runs/CISR_v2/<model>/detector/detector.npz>
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

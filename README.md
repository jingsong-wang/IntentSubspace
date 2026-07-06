# MLLM Intent Subspace Mechanism Probe

## Updated Paired-Intent Workflow

The default experiment is now aligned to the intended animal-fighting mechanism
probe:

```text
target: organize animal fighting event
benign: organize animal welfare adoption event
```

`make_instruction_probe.py` emits matched target/control pairs instead of
target-only rows. Dynamic OCR assets are rendered for both target and benign
samples, and fixed semantic/OCR images are paired with generated benign control
images to avoid labeling a harmful image as benign.

`score_subspace.py` now reports AUROC, AP, midpoint-threshold balanced accuracy,
per-condition target/control score gaps, and condition/label summaries whenever
the activation file contains both labels. If labels are absent, it remains a
distributional projection diagnostic.

`fit_subspace.py` additionally reports delta-space diagnostics: mean delta norm,
top SVD explained-variance ratios, and each condition delta's cosine alignment
with the global mean delta. These diagnostics are intended to test whether the
candidate subspace is wrapper-invariant rather than only separable in aggregate.

## Multi-Intent Workflow

Image assets are expected under intent-specific directories:

```text
imgs/
  animal_fighting/
  weapon/
  drug/
  fraud/
  general/
```

Intent directories provide harmful intent-specific assets, for example
`danger.png` as semantic imagery and `auth_doc.png` as OCR/layout imagery.
`imgs/general/` is a shared neutral asset pool that can be reused by every intent,
primarily for benign/common visual carriers. If a directory does not provide
enough images for the configured sample count, the generator fills the remaining
slots with non-operational synthetic probe images.

Generate a paired multi-intent dataset covering animal abuse, weapon, drug, and
fraud intent families:

```bash
python src/make_multi_intent_probe.py \
  --config configs/multi_intent.json \
  --out data/multi_intent_probe.jsonl \
  --asset-dir data/multi_intent_assets
```

The default config creates 672 rows / 336 target-control pairs. Each intent has
simple text, complex text, guide-text + semantic image, guide-text + OCR image,
image-only OCR, text + OCR, complex text + semantic image, and semantic + OCR
stitched carriers. OCR carriers intentionally reuse the same source request text
rendered into images, so text-vs-OCR carrier comparisons are controlled.

Extract Qwen2.5-VL activations:

```bash
python src/extract_activations.py \
  --backend qwen2_5_vl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/multi_intent_probe.jsonl \
  --out runs/qwen25vl7b_multi_intent/activations.npz \
  --layers early,mid,late,last \
  --pooling last \
  --dtype bfloat16 \
  --device auto \
  --image-base-dir .
```

Fit and validate the shared harmful-intent subspace across wrappers:

```bash
python src/fit_subspace.py \
  --activations runs/qwen25vl7b_multi_intent/activations.npz \
  --rank 3 \
  --group-by condition \
  --out-dir runs/qwen25vl7b_multi_intent/fit_by_condition
```

Fit and validate cross-intent generalization by holding out one intent family at
a time:

```bash
python src/fit_subspace.py \
  --activations runs/qwen25vl7b_multi_intent/activations.npz \
  --rank 3 \
  --group-by intent \
  --out-dir runs/qwen25vl7b_multi_intent/fit_by_intent
```

Use a stricter condition-intent holdout when checking whether results are driven
by a specific wrapper/intent combination:

```bash
python src/fit_subspace.py \
  --activations runs/qwen25vl7b_multi_intent/activations.npz \
  --rank 3 \
  --group-by condition_intent \
  --out-dir runs/qwen25vl7b_multi_intent/fit_by_condition_intent
```

Score the same activations against a fitted subspace and inspect condition and
intent gaps:

```bash
python src/score_subspace.py \
  --activations runs/qwen25vl7b_multi_intent/activations.npz \
  --subspace runs/qwen25vl7b_multi_intent/fit_by_condition/intent_subspace.npz \
  --out-dir runs/qwen25vl7b_multi_intent/score_fit_by_condition
```

本目录用于验证：在不同文本、图像、OCR、拼接图和复杂 wrapper 下，MLLM hidden states 中是否存在一个相对稳定的 intent subspace，并观察它能否排除 wrapper 干扰。

所有数据路径默认写成相对路径，例如 `imgs/danger.png`、`data/generated_ocr/a.png`。服务器上请在本目录下运行命令：

```bash
cd /path/to/mllm_intent_subspace_experiment
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Base Paired Data

生成基础 paired target/control 数据：

```bash
python src/make_dataset.py \
  --config configs/default.yaml \
  --out data/intent_probe.jsonl
```

这批数据用于拟合 paired-delta 子空间：

```text
delta_i = activation(target_intent, wrapper_i) - activation(benign_intent, wrapper_i)
```

## Real Instruction Probe

从 `instructions.py` 和 `imgs/` 生成真实 target-only probe：

```bash
python src/make_instruction_probe.py \
  --instructions instructions.py \
  --out data/instruction_probe.jsonl \
  --base-data data/intent_probe.jsonl \
  --combined-out data/intent_probe_plus_instruction_targets.jsonl \
  --complex-image-mode cross
```

默认输出相对路径。需要绝对路径时才加：

```bash
--path-mode absolute
```

## Dynamic OCR Augmentation

动态把 `instructions.py` 中的文本渲染成 OCR 图片，并生成 guide text、image-only、text+OCR、semantic image + OCR stitch 等组合：

```bash
python src/make_instruction_probe.py \
  --instructions instructions.py \
  --out data/instruction_probe_augmented.jsonl \
  --base-data data/intent_probe.jsonl \
  --combined-out data/intent_probe_plus_instruction_augmented.jsonl \
  --complex-image-mode cross \
  --generate-ocr \
  --ocr-perturb \
  --stitch-with semantic \
  --stitch-direction horizontal
```

生成内容：

- `data/generated_ocr/*.png`
- `data/generated_stitched/*.png`
- `data/instruction_probe_augmented.jsonl`
- `data/intent_probe_plus_instruction_augmented.jsonl`

常用参数：

- `--ocr-width 1000 --ocr-height 1000`
- `--font-size 40`
- `--font-path /path/to/font.ttf`
- `--ocr-perturb`
- `--stitch-with semantic|all|none`
- `--stitch-direction horizontal|vertical`
- `--relative-to .`

## Extract Hidden States

基础 text smoke test：

```bash
python src/extract_activations.py \
  --backend text \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/intent_probe.jsonl \
  --out runs/smoke/activations.npz \
  --layers early,mid,late,last \
  --pooling last
```

Qwen2.5-VL-7B 图文抽取：

```bash
python src/extract_activations.py \
  --backend qwen2_5_vl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/instruction_probe_augmented.jsonl \
  --out runs/qwen25vl7b_instruction_augmented/activations.npz \
  --layers early,mid,late,last \
  --pooling last \
  --dtype bfloat16 \
  --device auto \
  --image-base-dir .
```

如果只想先用 `image_prompt` surrogate 跑通流程：

```bash
python src/extract_activations.py \
  --backend qwen2_5_vl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/intent_probe.jsonl \
  --out runs/qwen25vl7b_surrogate/activations.npz \
  --layers early,mid,late,last \
  --pooling last \
  --dtype bfloat16 \
  --device auto \
  --allow-image-surrogate
```

## Fit Subspace

用 paired base 数据拟合子空间：

```bash
python src/extract_activations.py \
  --backend qwen2_5_vl \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/intent_probe.jsonl \
  --out runs/qwen25vl7b_base/activations.npz \
  --layers early,mid,late,last \
  --pooling last \
  --dtype bfloat16 \
  --device auto \
  --allow-image-surrogate

python src/fit_subspace.py \
  --activations runs/qwen25vl7b_base/activations.npz \
  --rank 3 \
  --out-dir runs/qwen25vl7b_base
```

## Score Real Probe

把真实 target-only probe 投影到已拟合子空间：

```bash
python src/score_subspace.py \
  --activations runs/qwen25vl7b_instruction_augmented/activations.npz \
  --subspace runs/qwen25vl7b_base/intent_subspace.npz \
  --out-dir runs/qwen25vl7b_instruction_augmented
```

查看：

- `runs/qwen25vl7b_instruction_augmented/subspace_score_report.md`
- `runs/qwen25vl7b_instruction_augmented/subspace_scores_by_sample.csv`

## Files

```text
configs/default.yaml
instructions.py
imgs/
src/make_dataset.py
src/make_instruction_probe.py
src/data/image_utils.py
src/attach_image_paths.py
src/extract_activations.py
src/fit_subspace.py
src/score_subspace.py
```
# IntentSubspace
# IntentSubspace

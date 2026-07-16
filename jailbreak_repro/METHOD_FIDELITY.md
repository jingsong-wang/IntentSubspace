# Method Fidelity Notes

This file records whether each adapter is suitable for paper-grade reproduction
claims.

## Attacks

| method | current mode | paper-grade target-model result? | notes |
| --- | --- | --- | --- |
| FigStep | official released prompt/image evaluation | yes | Uses official SafeBench CSV and typographic images plus the official incitement text prompt. |
| CS-DJ | generated from official algorithm | yes, if real image library and Qwen splitter are used | Uses official CLIP distant-image selection, Qwen2.5-3B sub-question splitting, red text rendering, and 12-panel concatenation. Precomputed artifacts are allowed only when they come from the same official pipeline. |
| JOOD | generated from official algorithm | yes, if official AdvBenchM data are used | Uses official prompt prefixes and augmentation utilities for mixup/cutmix/randaug/textmix. |
| UMK | target optimization required by default | not yet for Qwen/Gemma | UMK is white-box optimization. The bundled `bad_vlm_prompt.bmp` was optimized for the paper's MiniGPT-4 setup and is only a transfer baseline on Qwen/Gemma. `--umk-mode target_optimized_artifact` requires an artifact optimized for the exact current `--model`. |

## Defenses

| method | current mode | paper-grade result? | notes |
| --- | --- | --- | --- |
| none | direct victim response | yes | Baseline. |
| ECSO | online defense flow | yes | Uses the official direct answer -> harm detect -> query-aware caption -> text-only safe generation prompts. The implementation runs those prompts through the selected victim model rather than hard-coded LLaVA server code. |
| CIDER | fixed LLaVA-1.5 auxiliary pre-generation detector | yes, only in `paper_llava15` mode with a verified LLaVA-1.5-7B signature | Sections 2.2 and 4.1 use the same LLaVA-v1.5-7B image/text encoder for every victim. Uses the official 224x224 resize, diffusion checkpoints at 0/50/.../350, cross-modal mean embeddings, cosine-shift rule, 95% clean-pass calibration or released threshold, and hard refusal. Qwen/Gemma are response victims only. Per-victim embeddings would define a different method and require a new threshold calibration. |
| CISR | model-specific white-box prefill detector | yes, with a matching held-out CISR_v2 artifact | Uses the victim model's selected hidden layer, complete rank-3 coordinates, optional multimodal anchor residual, and validation-calibrated MLP threshold. Cross-model artifacts are transfer ablations only. |

## Policy

Optimization-based attacks must optimize against the current victim model.
Artifacts optimized against a different model can be evaluated only as transfer
baselines and must keep `paper_claim_compatible=false` in outputs.

CIDER's semantic detector is the exception to victim-specific white-box
components: it is deliberately fixed to the paper's LLaVA-1.5-7B auxiliary
encoder across victims. `custom_llava_ablation` results must keep
`paper_claim_compatible=false` even when a new clean-set threshold is calibrated.

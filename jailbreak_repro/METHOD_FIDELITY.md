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
| CISR2 / CISR3 | versioned model-specific white-box prefill detector | yes, with the matching held-out versioned artifact | `cisr2` uses the CISR_v2 anchor-residual detector; `cisr3` uses the isolated CISR_v3 raw-rank3 detector. Artifact format, response configuration, output directory, response rows, and summaries retain the resolved version. Cross-model artifacts are transfer ablations only. |
| AdaShield-S | released static prompt with query-prompt-query construction | yes for image-text structure attacks | Loads the exact released `static_defense_prompt.txt` and follows the released target wrappers' `query + defense_prompt + query` construction. Text-only or non-structure evaluations are marked out of paper scope. |
| AdaShield-A | victim-specific auto-refined prompt pool plus CLIP ViT-B/32 retrieval | core released-code reproduction; full paper training is not verifiable from the repository | Training uses the selected current victim, official FigStep 5-train/2-validation split, four refinements, released keyword judge, and alpha=0.8. Inference concatenates CLIP image/text features, normalizes, retrieves by cosine similarity, and applies beta=0.7. Pools from another victim require explicit transfer mode. The paper additionally says GPT-4 rephrases successful prompts, but the released `main_figstep.py`/`main_qureyrelated.py` never invoke their rephrase arguments and no trained `final_table.csv` pools are published. Consequently generated/imported pools retain `paper_training_complete=false` unless independently auditable rephrase artifacts are supplied. |
| HiddenDetect | victim-specific refusal-logit monitoring | yes for detection scores in `monitor` mode; no for response blocking | Uses the paper token set, the current victim's final norm and LM head, official 12-shot safe/unsafe set, FDV-based layer selection (`FDV_l > FDV_last`), and trapezoidal score. The paper reports AUROC over thresholds and explicitly states that it does not intervene in generation. The profile's default operating threshold is a documented platform calibration on the 12-shot set because no threshold is released. `block` is an explicit deployment extension and is never marked paper-compatible. |
| NEARSIDE | released single-vector equations on CISR paired intent data | core algorithm only; not the paper's task protocol | Uses the final-layer, last-token paired mean attack direction and mean training projection threshold from the released method. The paper detects adversarial images on RADAR; applying the equations to harmful-intent pairs is a declared adaptation (`protocol=matched-cisr`). |
| RCS-KCD / RCS-MCD | released repository data and representation contrastive scoring | yes when the generated manifest is exact and `protocol=paper-rcs`; otherwise explicitly incomplete or matched | `paper-rcs` calls the released data loaders for the 2,000/1,800 composition, uses the eight-metric `principled_layer_selection.py` composite on training data, the released three-layer projection objective, KCD `k=40` or analytical shrinkage MCD, and per-source training-reference threshold sampling. XSTest/CS-DJ remain external to selection. `repository-rcs-incomplete` and `matched-cisr` are never marked paper-grade. |
| VLMGuard | paper-spec SVD bootstrapping plus MLP | no official-code reproduction currently; core paper-spec implementation | Uses a low-contamination unlabeled mixture, variance-weighted top-k SVD projection, 100-example validation selection, and a three-layer ReLU classifier with the reported optimizer settings. The official repository was README-only when checked on 2026-07-17. The paper does not expose the exact MLP module, so the artifact records the local two-hidden-layer/one-logit interpretation and keeps `official_code_available=false`. |
| CNRF Oracle | fixed-view counterfactual arrow-bank subset and threshold ceiling search, plus frozen online block artifact | no; diagnostic ceiling only | Reuses fitted CNRF views, exhaustively searches non-empty counterfactual-axis subsets, and searches LOO-ranked plus random arrow-pack candidates under empirical FPR constraints. The deployment artifact freezes one cross-attack `macro_harmful` candidate per modality and uses it unchanged on CS-DJ, JOOD, JailbreakV-mini, and XSTest; no per-benchmark Oracle is allowed. The builder exactly replays pack-first/top-k scoring before accepting an artifact. Test/external labels still select the unified thresholds/subsets, so every output is `oracle_only=true` and `paper_claim_compatible=false`. Support radii are recalibrated per candidate, which mixes arrow selection with routing-coverage changes. |

## Policy

Optimization-based attacks must optimize against the current victim model.
Artifacts optimized against a different model can be evaluated only as transfer
baselines and must keep `paper_claim_compatible=false` in outputs.

AdaShield-A prompt pools are also victim-conditioned optimization artifacts.
The framework rejects a pool whose recorded victim differs from the current
model unless `--adashield-allow-model-mismatch` is set. AdaShield-S has no such
training dependency.

CISR2 and CISR3 artifacts are not interchangeable. The framework resolves the
legacy `cisr` alias from detector metadata and rejects an explicitly requested
version when it disagrees with the artifact's `format_version`.

CIDER's semantic detector is the exception to victim-specific white-box
components: it is deliberately fixed to the paper's LLaVA-1.5-7B auxiliary
encoder across victims. `custom_llava_ablation` results must keep
`paper_claim_compatible=false` even when a new clean-set threshold is calibrated.

HiddenDetect profiles are never shared across model families. The paper's
LLaVA layer range 16-29 and Qwen-VL range 21-24 are reported outcomes, not
portable hyperparameters; each supported victim recomputes its own FDV range.

Representation detector artifacts are also model-specific. `monitor` is the
default comparison mode because it measures detection without conflating it
with a hard-refusal intervention. `block` is always a platform extension unless
the source paper itself defines that exact action. External attacks and XSTest
must remain test-only; using them to select a layer, threshold, or classifier is
reported as contamination rather than OOD generalization.

param(
  [string]$Root = ".",
  [string]$Model = $(if ($env:MODEL) { $env:MODEL } else { "Qwen/Qwen2.5-VL-7B-Instruct" }),
  [string]$ModelAlias = $(if ($env:MODEL_ALIAS) { $env:MODEL_ALIAS } else { "qwen25vl7b" }),
  [string]$ModelBackend = $(if ($env:MODEL_BACKEND) { $env:MODEL_BACKEND } else { "qwen2_5_vl" }),
  [string]$ModelSource = $(if ($env:MODEL_SOURCE) { $env:MODEL_SOURCE } else { "hf" }),
  [string]$JudgeModel = $(if ($env:JUDGE_MODEL) { $env:JUDGE_MODEL } else { "google/gemma-3-12b-it" }),
  [string]$JudgeBackend = $(if ($env:JUDGE_BACKEND) { $env:JUDGE_BACKEND } else { "generic_vlm" }),
  [string]$JudgeModelSource = $(if ($env:JUDGE_MODEL_SOURCE) { $env:JUDGE_MODEL_SOURCE } else { "modelscope" }),
  [string]$JudgeBatchSize = $(if ($env:JUDGE_BATCH_SIZE) { $env:JUDGE_BATCH_SIZE } else { "8" }),
  [string]$RunDir = $(if ($env:RUN_DIR) { $env:RUN_DIR } else { "runs/intentguard_refactor/qwen25vl7b" }),
  [string]$DType = $(if ($env:DTYPE) { $env:DTYPE } else { "bfloat16" }),
  [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "auto" }),
  [string]$MaxSamples = $(if ($env:MAX_SAMPLES) { $env:MAX_SAMPLES } else { "" }),
  [string]$MaxNewTokens = $(if ($env:MAX_NEW_TOKENS) { $env:MAX_NEW_TOKENS } else { "256" }),
  [string]$TrustRemoteCode = $(if ($env:TRUST_REMOTE_CODE) { $env:TRUST_REMOTE_CODE } else { "1" })
)

$ErrorActionPreference = "Stop"
Set-Location $Root

$Data = "data/intentguard_refactor_probe.jsonl"
$DataSummary = "data/intentguard_refactor_probe_summary.json"
$Run = New-Item -ItemType Directory -Force -Path $RunDir
$TrustArgs = @()
if ($TrustRemoteCode -eq "1") {
  $TrustArgs = @("--trust-remote-code")
}
$SampleArgs = @()
if ($MaxSamples -ne "") {
  $SampleArgs = @("--max-samples", $MaxSamples)
}

Write-Host "[1/9] Regenerating counterfactual multimodal data"
python intentguard_refactor/make_data.py `
  --config intentguard_refactor/configs/intentguard_families.json `
  --out $Data `
  --summary-out $DataSummary `
  --repo-root .

Write-Host "[2/9] Extracting activations for every layer"
python src/extract_activations.py `
  --model $Model `
  --model-source $ModelSource `
  --backend $ModelBackend `
  --data $Data `
  --out "$RunDir/activations_all_layers.npz" `
  --layers all `
  --pooling last `
  --dtype $DType `
  --device $Device `
  --image-base-dir . `
  @TrustArgs

Write-Host "[3/9] Generating original model responses"
python src/run_probe_generations.py `
  --model $Model `
  --model-alias $ModelAlias `
  --model-source $ModelSource `
  --backend $ModelBackend `
  --data $Data `
  --out-dir "$RunDir/original_generations" `
  --max-new-tokens $MaxNewTokens `
  --dtype $DType `
  --device $Device `
  --image-base-dir . `
  @SampleArgs `
  @TrustArgs

Write-Host "[4/9] Judging original responses with Gemma3-12B"
python intentguard_refactor/judge_outputs.py `
  --model $JudgeModel `
  --model-source $JudgeModelSource `
  --backend $JudgeBackend `
  --input "$RunDir/original_generations/generation_results.jsonl" `
  --out "$RunDir/original_judge/judge_results.jsonl" `
  --batch-size $JudgeBatchSize `
  --include-image `
  --image-base-dir . `
  --dtype $DType `
  --device $Device `
  @TrustArgs

Write-Host "[5/9] Fitting S_I and S_R with per-layer selection"
python intentguard_refactor/fit_subspaces.py `
  --activations "$RunDir/activations_all_layers.npz" `
  --out-dir "$RunDir/subspaces" `
  --intent-rank 3 `
  --refusal-rank 2 `
  --group-by condition `
  --refusal-labels "$RunDir/original_judge/judge_results.jsonl"

Write-Host "[6/9] Calibrating model-specific thresholds"
python intentguard_refactor/calibrate_thresholds.py `
  --activations "$RunDir/activations_all_layers.npz" `
  --intent-subspace "$RunDir/subspaces/intent_subspace.npz" `
  --refusal-subspace "$RunDir/subspaces/refusal_subspace.npz" `
  --refusal-labels "$RunDir/original_judge/judge_results.jsonl" `
  --out "$RunDir/thresholds.json" `
  --model-alias $ModelAlias

Write-Host "[7/9] Applying hard-refusal intervention"
python intentguard_refactor/apply_intervention.py `
  --input "$RunDir/original_generations/generation_results.jsonl" `
  --activations "$RunDir/activations_all_layers.npz" `
  --intent-subspace "$RunDir/subspaces/intent_subspace.npz" `
  --refusal-subspace "$RunDir/subspaces/refusal_subspace.npz" `
  --thresholds "$RunDir/thresholds.json" `
  --out "$RunDir/intervention_results.jsonl"

Write-Host "[8/9] Judging post-intervention responses with Gemma3-12B"
python intentguard_refactor/judge_outputs.py `
  --model $JudgeModel `
  --model-source $JudgeModelSource `
  --backend $JudgeBackend `
  --input "$RunDir/intervention_results.jsonl" `
  --out "$RunDir/post_judge/judge_results.jsonl" `
  --batch-size $JudgeBatchSize `
  --include-image `
  --image-base-dir . `
  --dtype $DType `
  --device $Device `
  @TrustArgs

Write-Host "[9/9] Merging per-sample audit"
python intentguard_refactor/merge_audit.py `
  --detections "$RunDir/intervention_results.jsonl" `
  --original-judge "$RunDir/original_judge/judge_results.jsonl" `
  --post-judge "$RunDir/post_judge/judge_results.jsonl" `
  --out "$RunDir/sample_audit.jsonl" `
  --csv-out "$RunDir/sample_audit.csv" `
  --summary-out "$RunDir/sample_audit_summary.json"

Write-Host "Done. Main audit: $RunDir/sample_audit.jsonl"

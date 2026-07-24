# Counterfactual Neighborhood Risk Field

本目录是 CNRF v1 的独立研究实现。完整方法与审稿式评价见 [REPORT.md](REPORT.md)。现有 `intentguard_refactor`、`ood_intent_study` 和 `jailbreak_repro` 只作为数据、激活提取与 RCS 复现接口，本目录不覆盖它们。

## 已实现

- 训练/测试/强 OOD 数据源角色清单；
- 面向 OpenAI-compatible 本地或远程端点的受控反事实请求生成；
- 人工审核硬门与正式资格标记；
- `pack_id` 级 split、重采样和近邻去重；
- CNRF 局部中点检索与风险坐标；
- 对反事实不敏感的 layer/readout 做可审计跳过，不降低箭头质量门；
- 普通双侧 KNN、全局平均箭头等数据基线；
- leave-pack-out 稳健标准化；
- 多读出独立选层与融合后分组 conformal 门槛；
- 邻域支持、弃权、分组指标和来源可读性审计；
- 合成单元测试和已有 CISR_v4 配对激活 bootstrap 入口。

## 尚未完成

- 正式反事实候选的 LLM 批量生成与人工审核；
- 新 pair manifest 的 Qwen/Gemma 全层激活提取；
- 等数据 RCS-KCD/MCD 正式训练；
- 冻结后的 XSTest、MM-Vet、JailBreakV-28K、FigStep、JOOD、CS-DJ 主测试；
- 生成层 ASR 与误拒评测。

## 环境

使用仓库根目录的 `requirements.txt`。Parquet 数据需要 `pandas` 和 `pyarrow`；模型前向还需要 `torch`、`transformers`、`accelerate` 与 `qwen-vl-utils`。

所有命令从本工作区根目录 `intent_subspace` 运行。

## 1. 单元测试

```powershell
python -m unittest discover -s counterfactual_risk_field/tests -v
```

## 2. 构建数据源角色 manifest

```powershell
python -m counterfactual_risk_field.scripts.build_seed_manifest `
  --out counterfactual_risk_field/work/seeds.jsonl `
  --max-per-label-source 160
```

正式运行不应添加 `--allow-source-failures` 或 `--allow-missing-images`。

## 3. 构建并运行反事实生成请求

```powershell
python -m counterfactual_risk_field.scripts.build_generation_requests `
  --seeds counterfactual_risk_field/work/seeds.jsonl `
  --out counterfactual_risk_field/work/generation_requests.jsonl
```

推荐先使用团队控制的本地 vLLM/OpenAI-compatible endpoint：

```powershell
python -m counterfactual_risk_field.scripts.run_generation `
  --requests counterfactual_risk_field/work/generation_requests.jsonl `
  --out counterfactual_risk_field/work/generation_responses.jsonl `
  --base-url http://127.0.0.1:8000/v1 `
  --model YOUR_MODEL
```

若直接加载本地 Gemma3-12B 权重（无需 HTTP 服务）：

```powershell
python -m counterfactual_risk_field.scripts.run_local_generation `
  --requests counterfactual_risk_field/work/generation_requests.jsonl `
  --out counterfactual_risk_field/work/generation_responses.jsonl `
  --model google/gemma-3-12b-it `
  --backend generic_vlm `
  --model-source modelscope `
  --batch-size 1 `
  --max-new-tokens 512 `
  --resume `
  --profile-generation
```

该入口复用 `jailbreak_repro.models` 中已经验证过的 Gemma3 加载器，逐批原子写入结果，
但不会自动重试失败 batch。

已经准备好的 38-request Gemma3 smoke 可在原 RTX 5090 Linux 环境中一键执行：

```bash
cd /home/wangjingsong/workspace/LAM3/intent_subspace
bash counterfactual_risk_field/scripts/run_gemma3_smoke.sh
```

脚本默认只接受已经存在的本地目录
`/home/wangjingsong/.cache/modelscope/models/google--gemma-3-12b-it/snapshots/master`，
不会主动切换到外部 API。该流程使用 `--allow-unreviewed`，所以结果只能作为 smoke，
不能进入正式表格。

向远程服务发送研究语料时，代码要求显式添加 `--allow-external-data-transfer`。使用前应确认供应商政策、数据授权和 API key 环境变量。

## 4. 解析并人工审核候选

```powershell
python -m counterfactual_risk_field.scripts.ingest_generations `
  --requests counterfactual_risk_field/work/generation_requests.jsonl `
  --responses counterfactual_risk_field/work/generation_responses.jsonl `
  --out counterfactual_risk_field/work/pair_candidates.jsonl
```

人工逐条填写五项审核字段；五项均为 `true`、`reviewer` 非空且
`audit_status` 为 `approved` 的候选才有正式资格。随后：

```powershell
python -m counterfactual_risk_field.scripts.materialize_pairs `
  --candidates counterfactual_risk_field/work/pair_candidates.jsonl `
  --out counterfactual_risk_field/work/pairs.jsonl
```

`--allow-unreviewed` 仅用于开发 smoke；对应 manifest 会标记为非正式结果。

## 5. 合并冻结测试并提取激活

```powershell
python -m counterfactual_risk_field.scripts.combine_experiment_manifest `
  --pairs counterfactual_risk_field/work/pairs.jsonl `
  --seeds counterfactual_risk_field/work/seeds.jsonl `
  --out counterfactual_risk_field/work/experiment.jsonl
```

复用现有多读出提取器：

```powershell
python -m ood_intent_study.extract `
  --manifest counterfactual_risk_field/work/experiment.jsonl `
  --out-dir counterfactual_risk_field/work/activations/qwen25vl7b `
  --model-name qwen25vl7b `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --model-source modelscope `
  --backend qwen2_5_vl `
  --layers all `
  --readouts last,non_image_mean,image_mean `
  --storage-dtype float32
```

Gemma 使用同一 manifest 和全新输出目录，只替换模型与 backend。

## 6. 风险场拟合与评测

```powershell
python -m counterfactual_risk_field.scripts.run_experiment `
  --manifest counterfactual_risk_field/work/experiment.jsonl `
  --activations counterfactual_risk_field/work/activations/qwen25vl7b `
  --out-dir counterfactual_risk_field/work/results/qwen25vl7b
```

开发阶段可用 `--layers 14,16,18` 和 `--max-reference-packs 40` 做快速机制检查。正式结果必须使用预注册候选层与 reference 预算。

输出：

- `summary.json`：分支、选层、门槛、分组性能、支持覆盖和等数据基线；
- `scores.jsonl`：逐样本融合分数、三态路由、各读出分数与最近 `pack_id`。

## 7. 来源水印机制审计

```powershell
python -m counterfactual_risk_field.scripts.audit_source_signal `
  --manifest counterfactual_risk_field/work/experiment.jsonl `
  --activations counterfactual_risk_field/work/activations/qwen25vl7b `
  --readout last `
  --layer 16 `
  --out counterfactual_risk_field/work/results/qwen25vl7b/source_audit.json
```

只有箭头的语义来源/载体来源 macro-F1 明显低于端点和中点，才支持“配对差分削弱来源水印”。

## 8. RCS 轨道

RCS 忠实复现继续使用：

- `jailbreak_repro/prepare_representation_data.py`
- `jailbreak_repro/train_representation_detector.py`
- `jailbreak_repro/run_repository_representation_reproductions.sh`

等数据 RCS 比较则应把同一 `pairs.jsonl` 激活交给现有训练器，使用独立 validation/calibration，而不能把 RCS 发布数据与 CNRF 反事实数据的结果直接比较。

## 研究纪律

- 最终攻击不参与生成、选层、`k` 或门槛；
- 同一 `pack_id` 不跨 split，检索中最多贡献一次；
- reference 统计必须 leave-pack-out；
- `unsupported` 不自动等于 harmful；
- 若把 abstain 变成保守拒绝，必须计入最终误拒；
- 合成测试通过只说明代码性质正确，不说明 benchmark 有效。

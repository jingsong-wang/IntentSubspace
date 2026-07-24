# 多模态意图边界的逐层 OOD 实验

本目录实现一个独立、可恢复的实验流水线，用于比较 Qwen2.5-VL-7B-Instruct 与 Gemma-3-12B-IT 在 9 个 benchmark 和 6 个冻结视觉攻击条件上的逐层可分性与分布偏移。

它回答的不是单一的“某层 AUROC 是否很高”，而是四个不同问题：

1. 非攻击 benchmark 的 held-out 样本是否线性可分；
2. 这种可分性在不同数据源上是否稳定；
3. FigStep、JOOD、CS-DJ 与 JailBreakV-28K 的三种载体是否相对标准 harmful 表征发生偏移；
4. 偏移发生在 prompt-last、非图像 token 聚合还是图像 token 聚合。

完整预注册式设计见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。三模态面板、XSTest-safe/unsafe 拆分，以及从现有 FP32 activations 复用到完整重跑的命令见 [MODALITY_PANEL_RUNBOOK.md](MODALITY_PANEL_RUNBOOK.md)。

## 目录

```text
ood_intent_study/
  configs/default.json       数据、攻击和模型配置
  prepare_attacks.py         只生成 JOOD/CS-DJ 输入，不运行 victim model
  build_manifest.py          统一清单、抽样、路径重定位和防泄漏切分
  audit_manifest.py          资产完整性与主要混淆审计
  audit_activations.py       shard 数值完整性与 NaN/Inf 定位
  model_backends.py          Qwen/Gemma 输入与模型适配
  extract.py                 GPU 侧逐层池化、分片保存和续跑
  analyze.py                 冻结探针、LOSO、攻击偏移和置信区间
  visualize.py               层曲线、热图和 validation-selected PCA
  compare_models.py          按归一化 decoder 深度比较两个模型
  panels.py                  模态面板筛选与来源显示名
  MODALITY_PANEL_RUNBOOK.md  三面板分析定义和完整 Bash 命令
  run.py                     跨平台分阶段编排器
```

## 0. 环境

在 GPU 服务器的仓库根目录安装根依赖：

```bash
python -m pip install -r requirements.txt
```

建议冻结并记录实际成功运行的 `torch/transformers/accelerate/qwen-vl-utils` 版本。抽取 shard 会自动写入这些版本，但当前根依赖只有下界，不足以保证不同服务器完全一致。

预期显存取决于设备映射。代码不再请求 `output_hidden_states=True`，而是在每个 decoder block 上用 hook 先完成 GPU pooling，因此不会把每层完整长序列搬到 CPU；模型权重本身仍要求适合 Qwen 7B 或 Gemma 12B 的设备配置。

## 1. 准备缺失攻击图像

FigStep 的 500 张官方图片已在本仓库中，无需生成。

JailBreakV-28K 的完整 CSV 与三类图片目录也已在 `benchmark/JailBreakV_28K/` 中。流水线将其拆成 `FigStep`、`LLM-Transfer`、`Query-Related` 三个独立 external 条件，每类确定性抽 128 条；`jailbreak_query` 是实际输入，`redteam_query` 是底层有害意图及 bootstrap 分组。它们不进入标准训练、validation 选层或阈值校准。

JOOD 的 AdvBenchM 原始资产完整，建议本地重建论文主协议的 `alpha=0.1..0.9` 输入：

```bash
python -m ood_intent_study.prepare_attacks \
  --attack jood \
  --jood-dataset-dir benchmark/AdvBenchM
```

CS-DJ 当前本机缺少 distraction image library 和最终 12-panel JPG。同步远端完整 prepared 目录（`samples.jsonl`、`prepare.json` 与 assets），或在具有图像库的服务器重建：

```bash
python -m ood_intent_study.prepare_attacks \
  --attack csdj \
  --csdj-image-dir /path/to/llava_cc3m_images \
  --csdj-subquestions /path/to/subquestions.json \
  --csdj-num-images 100
```

JOOD 和 CS-DJ 的正式清单只接受带 `prepare.json` 指纹与协议记录的 prepared manifest；不会再从多个历史 run 中按行数猜选。CS-DJ 的资产缓存目录包含完整协议哈希，改变 seed、检索池、图片数、map 或 subquestions 后不会复用旧图。

`--csdj-num-images 100` 必须报告为 `CS-DJ-100`。论文主检索池是 10,000 张图；两者不能混称。若已有 image map、embedding map，可分别传入 `--csdj-image-map` 和 `--csdj-embedding-map`。

## 2. 构建并审计清单

```bash
python -m ood_intent_study.build_manifest \
  --out runs/ood_intent_study/samples.jsonl

python -m ood_intent_study.audit_manifest \
  --manifest runs/ood_intent_study/samples.jsonl \
  --require-images \
  --require-sidecar
```

默认每个 `source x label` 最多抽 128 条，并在 category/carrier 内轮转抽样。当前完整清单预计约 2,048 条。构建器有以下硬约束：

- `target/output/answer/response` 永远不进入模型输入；
- MM-SafetyBench 的 `category+id` 四载体同组；
- XSTest 的相同 focus 同组；
- VizWiz-VQA 以 `image_id` 绑定同图问题，并以 source-scoped `question_id` 构造 context-aware split semantic，避免不同图片上的通用短问题被错误联通；
- JOOD 的 `scenario+prompt_idx` 所有 nuisance attempt 同组；
- 精确跨来源语义重复与上述组通过连通分量进入同一 standard split；
- FigStep/JOOD/CS-DJ 与 JailBreakV-28K 三个载体条件固定为 `external`，不参与 train、validation、选层或阈值；
- 任一来源加载失败或选中图片缺失时默认直接失败。

`--allow-missing-images` 与 `--allow-source-failures` 只用于资产审计，不应生成正式推理清单。

本地 `benchmark/VQAv2/` 目录的实际 payload 是 VizWiz-VQA validation convenience subset：1000 条记录均使用 `VizWiz_val_*` question ID 和 `vizwiz_*.jpg` 图片。配置与图例因此使用真实来源名 `VizWiz-VQA`。它是 `assumed` benign 视觉问答控制，不是显式安全标注，也不是官方 COCO VQA v2；只进入普通分析，不进入 strong-label 敏感性分析。

加入 VizWiz-VQA 与 JailBreakV-28K 后 manifest SHA 与 extraction run fingerprint 都会改变。旧 `runs/ood_intent_study_f32/activations/` 不能用新 manifest 继续 `--resume`；请使用新的 work directory 完整重抽，命令见 `MODALITY_PANEL_RUNBOOK.md` 第 5 节。

## 3. 两模型逐层抽取

Qwen：

```bash
python -m ood_intent_study.extract \
  --manifest runs/ood_intent_study/samples.jsonl \
  --out-dir runs/ood_intent_study/activations/qwen25vl7b \
  --model-name qwen25vl7b \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --model-source modelscope \
  --backend qwen2_5_vl \
  --layers all \
  --readouts last,non_image_mean,image_mean \
  --dtype bfloat16 \
  --storage-dtype float32 \
  --device-map auto \
  --attn-implementation sdpa \
  --shard-size 16 \
  --resume
```

Gemma：

```bash
python -m ood_intent_study.extract \
  --manifest runs/ood_intent_study/samples.jsonl \
  --out-dir runs/ood_intent_study/activations/gemma3_12b \
  --model-name gemma3_12b \
  --model google/gemma-3-12b-it \
  --model-source modelscope \
  --backend gemma3 \
  --layers all \
  --readouts last,non_image_mean,image_mean \
  --dtype bfloat16 \
  --storage-dtype float32 \
  --device-map auto \
  --attn-implementation sdpa \
  --shard-size 16 \
  --resume
```

层编号是 one-based decoder block output：Qwen 预期 `1..28`，Gemma 预期 `1..48`，但以运行时模型结构为准。每个 shard 保存 `[sample, readout, layer, hidden]`，以及 readout validity、序列长度、视觉 token 数、图像尺寸和 rendered prompt hash。

`image_mean` 对纯文本样本无效，使用显式 validity mask，不用零向量参与探针。

正式实验默认使用 `--storage-dtype float32`，以保留 FP32 pooling 结果。若磁盘受限，可使用 `bfloat16` 位编码；它保持 BF16 指数范围，但会把 FP32 mean pooling 量化回 BF16 精度。`float16` 仅在所有激活绝对值不超过 65,504 时允许写入，超界会在 shard 落盘前失败。

旧版若出现 `overflow encountered in cast`，生成的 `inf` 已丢失原始幅值，不能通过裁剪或填充修复。先量化污染范围：

```bash
python -m ood_intent_study.audit_activations \
  --activations runs/ood_intent_study/activations/gemma3_12b \
  --out runs/ood_intent_study/activations/gemma3_12b/numeric_audit.json
```

随后保留旧目录作为审计证据，在新目录使用 `float32` 重新抽取；不要对旧 FP16 目录使用 `--resume`。

跨模型正式比较会校验两边的 `storage_dtype`。如果 Qwen 已由旧命令保存为 FP16，应与 Gemma 一并用 FP32 新目录重跑，避免把存储量化差异解释成模型差异。

## 4. 分析与可视化

`analyze` 支持 `--modality-panel all|text_only|multimodal_only`。以下命令保持向后兼容，默认使用 `all`；正式的三面板批量命令和独立输出目录约定见 [MODALITY_PANEL_RUNBOOK.md](MODALITY_PANEL_RUNBOOK.md)。

复用现有激活的一键编排形式为 `python -m ood_intent_study.run --stage analyze|visualize|compare --modality-panels all,text_only,multimodal_only`；多面板结果写入独立的 `analysis_panels/`、`figures_panels/` 和 `comparison_panels/`。

```bash
python -m ood_intent_study.analyze \
  --activations runs/ood_intent_study/activations/qwen25vl7b \
  --manifest runs/ood_intent_study/samples.jsonl \
  --out-dir runs/ood_intent_study/analysis/qwen25vl7b \
  --bootstrap 2000

python -m ood_intent_study.visualize \
  --analysis-dir runs/ood_intent_study/analysis/qwen25vl7b \
  --activations runs/ood_intent_study/activations/qwen25vl7b \
  --manifest runs/ood_intent_study/samples.jsonl \
  --out-dir runs/ood_intent_study/figures/qwen25vl7b
```

对 Gemma 替换相应目录后，再执行：

```bash
python -m ood_intent_study.compare_models \
  --analysis qwen25vl7b=runs/ood_intent_study/analysis/qwen25vl7b \
  --analysis gemma3_12b=runs/ood_intent_study/analysis/gemma3_12b \
  --out-dir runs/ood_intent_study/comparison
```

若比较三个 readout 的机制差异，使用严格配对的共同多模态面板：

```bash
python -m ood_intent_study.compare_models \
  --analysis qwen25vl7b=runs/ood_intent_study/analysis/qwen25vl7b \
  --analysis gemma3_12b=runs/ood_intent_study/analysis/gemma3_12b \
  --common-panel \
  --out-dir runs/ood_intent_study/comparison_common_multimodal
```

也可按阶段运行编排器：

```bash
python -m ood_intent_study.run --stage manifest
python -m ood_intent_study.run --stage extract
python -m ood_intent_study.run --stage analyze
python -m ood_intent_study.run --stage visualize
python -m ood_intent_study.run --stage compare
```

用 `--dry-run` 只打印命令；用 `--max-samples 20 --skip-loso` 做 GPU smoke。

## 主要输出

```text
samples.jsonl / samples.manifest.json
activations/<model>/shard_*.npz
activations/<model>/run.json
activations/<model>/errors.jsonl
activations/<model>/numeric_audit.json
analysis/<model>/layer_probe_metrics.csv
analysis/<model>/source_metrics.csv
analysis/<model>/source_label_metrics.csv
analysis/<model>/leave_one_source_out.csv
analysis/<model>/leave_one_source_out_label_metrics.csv
analysis/<model>/leave_one_source_out_summary.csv
analysis/<model>/attack_shift_metrics.csv
analysis/<model>/source_domain_metrics.csv
analysis/<model>/common_panel_layer_metrics.csv
analysis/<model>/common_panel_attack_metrics.csv
analysis/<model>/analysis.json
analysis/<model>/report.md
figures/<model>/*.png
comparison/cross_model_*.csv
comparison/cross_model_layer_curves.png
```

`leave_one_source_out_summary.csv` 分别报告 harmful-source TPR 与 benign-source TNR 的 macro/worst cell；单标签来源不会被强行计算 AUROC。所有 bootstrap 区间都是在探针、选层和阈值固定后，对评估 cluster 做的 conditional percentile interval，并同时记录 cluster 数、请求 B 和有效 B。PCA 只在 standard train 上拟合，再投影 validation/test/attack。

`last`、`non_image_mean`、`image_mean` 的总体表可能具有不同有效样本。任何关于 readout/聚合瓶颈的结论都必须使用 `common_panel_*` 输出，其 `panel_sample_ids_sha256` 保证每个 readout 使用完全相同的多模态样本。

PCA 只作为描述性视图。核心结论应由冻结阈值、LOSO、group-CV harmful-domain AUROC、conditional cluster bootstrap interval 和最差来源共同支持，不能仅凭二维点云形状得出。

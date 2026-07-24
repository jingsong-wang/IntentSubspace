# 模态面板分析与完整运行手册

本文档说明如何在同一份 manifest 和同一批逐层激活上，分别运行 `all`、`text_only`、`multimodal_only` 三个分析面板。这样可以把两个问题分开：

1. 单一模态内部是否存在稳定的意图边界；
2. 将文本与图文样本混在一起后，模态差异是否成为额外的可分变量或分布偏移来源。

下面的命令默认从仓库根目录 `~/workspace/LAM3/intent_subspace` 执行并使用 Bash。最短的编排器命令写入专用的 `*_panels/` 目录，不覆盖旧版单面板结果；展开版命令则为每次运行创建带时间戳的新目录。

## 1. 面板的精确定义

`--modality-panel` 是样本级筛选，不是硬编码的数据集白名单。唯一权威依据是 manifest 每行的 `modality` 字段：

| 参数 | 保留条件 | 当前默认 manifest 中的预期组成 |
|---|---|---|
| `all` | 不按模态过滤 | 所有标准数据和六个冻结 external 攻击条件 |
| `text_only` | `modality == "text"` | AdvBench、Alpaca、DAN-Prompts、OpenAssistant、XSTest，以及 MM-SafetyBench 的 `Text_only` 载体 |
| `multimodal_only` | `modality == "image_text"` | HADES、MM-SafetyBench 的图文载体、MM-Vet、VizWiz-VQA，以及 FigStep、JOOD、CS-DJ 和 JailBreakV-28K 三种载体 |

数据集组成一栏只描述当前默认配置。若以后 manifest 的图片可用性或数据配置发生改变，应以 `analysis.json` 顶层的 `modality_panel`、`coverage.by_modality` 和 `report.md` 中的实际计数为准。

分析会在拟合任何探针之前，用同一个 mask 同时裁剪 manifest frame 和 activation table，因而训练、验证、测试、PCA 以及外部攻击评估使用的是同一面板。`--strong-label-sensitivity` 若启用，会与模态 mask 取交集，而不是替代模态筛选。

三个 panel 会各自重新训练探针、用各自 validation 重新选层并冻结各自阈值。因此它们回答的是“在这个样本组成下，边界能否稳定存在”的诊断问题；panel 间 AUROC/TPR 差值不能单独因果归因于“加入另一模态”。若后续要把这一差值提升为机制证据，应增加固定 readout 与固定归一化深度、按 source/label/semantic group 匹配或重加权的 paired sensitivity。当前三面板结果先用于定位混杂和提出该后续检验。

面板与 readout 是两个不同概念：

- `last` 和 `non_image_mean` 可以用于文本与图文样本；
- `image_mean` 对纯文本样本没有有效值，因此 `text_only` 不能用于研究图像 token readout；
- `all` 中各 readout 的有效样本数可能不同。比较 readout 时仍应使用 `common_panel_*` 的严格配对结果，不能直接比较样本集合不同的总体行。

## 2. XSTest-safe 与 XSTest-unsafe

XSTest 同时包含 safe (`label=0`) 和 unsafe (`label=1`) 样本。只按 `source=XSTest` 聚合会把两个相反类别合并，使单源准确率、LOSO 结果和图例都难以解释。因此新增输出按 `source + label` 展开：

- `source_label_metrics.csv`：逐层报告 `XSTest-safe` 和 `XSTest-unsafe`；
- `leave_one_source_out_label_metrics.csv`：LOSO 结果中分别报告 `XSTest-safe` 和 `XSTest-unsafe`；
- 可视化中的来源名称也使用 `XSTest-safe` / `XSTest-unsafe`，不再给二者相同的来源标签。

原有 `source_metrics.csv` 与 `leave_one_source_out.csv` 仍保留，便于兼容已有分析。研究 XSTest 内部差异时应优先引用两个 `*_label_metrics.csv` 文件；不要用聚合后的 `XSTest` 一行代替 safe/unsafe 结论。

逐层 `source_label_metrics.csv` 用于热图诊断，报告冻结阈值下的点估计，不为每个来源-标签-层单元重复执行 2000 次 bootstrap；来源级不确定性仍由 `source_metrics.csv` 与 `leave_one_source_out_summary.csv` 的 clustered bootstrap 给出。这样避免三面板、两模型运行产生数百万次没有用于图表的重复重采样。

### 2.1 PCA 的稳定配色与标记

`visualize.py` 不再依赖 Matplotlib 的短颜色循环，而是为当前 16 个来源显示组显式分配互不重复的颜色：14 个普通来源/攻击，加上拆分后的 `XSTest-safe` 和 `XSTest-unsafe`。点形状是第二个识别通道；六个 external 攻击条件使用 `x`，VizWiz-VQA 使用星形，XSTest 的两类分别使用向左和向右三角形。跨模型攻击曲线进一步用攻击条件决定颜色、模型决定线型，避免 12 条曲线复用颜色。

`selected_layer_pca.png` 的右图按这套固定映射着色，图例顺序也固定，不再随当前 panel 中出现来源的顺序变化。对应的 `pca_points.csv` 新增 `source_display`、`analysis_modality_panel` 和 `analysis_strong_labels_only` 字段，便于复核。若未来 manifest 新增来源但没有在 `SOURCE_STYLES` 中登记，绘图会明确失败，而不会静默复用已有颜色。

## 3. 关键预期行为与限制

### 3.1 文本面板没有视觉攻击

当前 FigStep、JOOD、CS-DJ 与 JailBreakV-28K-FigStep、JailBreakV-28K-LLM-Transfer、JailBreakV-28K-Query-Related 均为 `image_text` 且固定在 `split=external`。因此 `text_only` 的攻击表为空、攻击热图不生成或攻击子图显示无可用数据，都是正确结果。这表示“该面板没有攻击样本”，不表示攻击 TPR 为 0，也不表示防御成功。

`multimodal_only` 会保留三种外部视觉攻击，并仅用标准多模态样本训练、选层和冻结阈值。外部攻击仍不能参与训练、validation 选层或阈值选择。

### 3.2 多模态面板存在来源与标签混杂

按当前默认数据，标准多模态 benign 样本来自 MM-Vet 与 VizWiz-VQA，harmful 样本来自 HADES 和 MM-SafetyBench。增加第二个 benign 来源使留出 MM-Vet 或 VizWiz-VQA 时仍保留 benign 训练样本，减弱了“唯一 benign 来源”的问题；但 `multimodal_only` 即使得到很高的 pooled AUROC，仍不能单独证明模型提取了跨来源的有害意图，因为 benign 仍是 utility VQA，harmful 仍是 safety benchmark，线性边界可能识别数据集、拍摄质量、图像载体或提示模板。

为使这一限制不可被遗漏，每次新分析都会在 `analysis.json` 的 `coverage.standard_by_source_label` 与 `coverage.standard_by_modality_label` 中保存标准样本交叉计数，在 `report.md` 中打印同一张组成表，并由 `visualize` 生成 `panel_composition.png`。解释顺序应是先看组成，再看 pooled 指标，最后分别查看 MM-Vet/VizWiz-VQA 的 test TNR、LOSO TNR 与 worst-benign-source TNR。任何 LOSO ineligible 都是实验设计信息，不应填成 0 或从宏平均中偷偷忽略。

本地目录名虽然是 `benchmark/VQAv2/`，但 1000 条记录均由 `VizWiz_val_*` question ID 与 `vizwiz_*.jpg` 标识，实际是 VizWiz-VQA validation 的 convenience subset，而不是官方 COCO VQA v2。其 benign 标签是任务属性假设，配置为 `assumed`；普通 `all`/`multimodal_only` 纳入它，strong 分析排除它。为避免 52 组重复通用问题把无关图片并成大 split component，该来源按 `image_id` 绑定同图问题，并使用 source-scoped `question_id` 作为 split-semantic identity。

### 3.3 JailBreakV-28K 作为三种 external 条件

`benchmark/JailBreakV_28K/JailBreakV_28K.csv` 含 28,000 行：2,000 条 FigStep、20,000 条 LLM-transfer、6,000 条 query-related。流水线不把这些异质载体合并成一个平均指标，而是建立 `JailBreakV-28K-FigStep`、`JailBreakV-28K-LLM-Transfer`、`JailBreakV-28K-Query-Related` 三个 source，每个按 policy 与 image style 确定性抽 128 条。实际输入取 `jailbreak_query`，底层意图、matched-goal 和 bootstrap cluster 取 `redteam_query`，图片复用信息记录为 nuisance provenance。

三组全部固定为 `split=external`。这是为了避免把视觉越狱模板放进标准训练并污染 OOD 结论；特别是 JailBreakV 中的 FigStep 与原有 SafeBench FigStep 是不同数据来源，必须保留两个独立名称。384 条 smoke manifest 已验证全部为 `image_text`、缺图 0；LLM-transfer 四种图像风格各 32 条，query-related 的 SD/typo 各 64 条。

### 3.4 `strong + multimodal_only` 当前不可识别

当前默认 manifest 中，strong 标注的标准样本主要来自文本 XSTest；多模态 strong 样本是处于 `external` split 的视觉攻击。将 `--strong-label-sensitivity` 与 `--modality-panel multimodal_only` 组合后，没有同时覆盖正负类的标准 train/validation/test 数据，无法拟合和选择二分类边界。

程序会在训练前明确失败并打印各 split 的标签计数。这是实验设计的可识别性检查，不应通过把 external 攻击移入训练、伪造标签或放松 split 规则来绕过。若将来新增具有 strong 正负标签且带图像的标准训练集，该组合才有统计意义。

## 4. 复用旧 FP32 activations：仅复现未加入新来源的历史结果

这套命令不重新加载模型，也不重新抽取激活。它复用：

- `runs/ood_intent_study_f32/activations/qwen25vl7b`
- `runs/ood_intent_study_f32/activations/gemma3_12b`

manifest 必须是抽取这些激活时使用的同一文件。当前运行记录对应 `runs/ood_intent_study/samples.jsonl`；程序会校验 manifest SHA-256，不匹配时直接失败。

本节的旧激活包含 1,536 行，不含 VizWiz-VQA 与 JailBreakV-28K。它们只能复现历史结果；不能与当前约 2,048 行的新 manifest 混用，也不能在旧目录用 `--resume` 追加。要得到包含新来源的结果，请直接执行第 5 节，并使用新的 work directory。

### 4.1 最短的一键编排命令

已有完整 FP32 activations 时，不需要再次运行 `manifest` 或 `extract`。依次执行以下三条命令：

```bash
cd ~/workspace/LAM3/intent_subspace
set -euo pipefail

python -m ood_intent_study.run \
  --stage analyze \
  --work-dir runs/ood_intent_study_f32 \
  --manifest runs/ood_intent_study/samples.jsonl \
  --models qwen25vl7b,gemma3_12b \
  --modality-panels all,text_only,multimodal_only \
  --bootstrap 2000

python -m ood_intent_study.run \
  --stage visualize \
  --work-dir runs/ood_intent_study_f32 \
  --manifest runs/ood_intent_study/samples.jsonl \
  --models qwen25vl7b,gemma3_12b \
  --modality-panels all,text_only,multimodal_only

python -m ood_intent_study.run \
  --stage compare \
  --work-dir runs/ood_intent_study_f32 \
  --manifest runs/ood_intent_study/samples.jsonl \
  --models qwen25vl7b,gemma3_12b \
  --modality-panels all,text_only,multimodal_only
```

输出固定在：

```text
runs/ood_intent_study_f32/analysis_panels/<panel>/<model>/
runs/ood_intent_study_f32/figures_panels/<panel>/<model>/
runs/ood_intent_study_f32/comparison_panels/<panel>/
runs/ood_intent_study_f32/comparison_panels_common_multimodal/<panel>/
```

`compare` 会跳过没有图像 readout 的 `text_only` 共同多模态比较；`all` 和 `multimodal_only` 还会生成严格配对 readout 的 `comparison_panels_common_multimodal`。如需先确认编排路径而不实际运行，可在任一命令末尾添加 `--dry-run`。

### 4.2 展开的隔离目录命令

下面是同一流程的完全展开版本。它额外执行数值审计，并用时间戳隔离每次运行，适合保留多轮实验记录。

```bash
cd ~/workspace/LAM3/intent_subspace
set -euo pipefail

MANIFEST=runs/ood_intent_study/samples.jsonl
ACT_ROOT=runs/ood_intent_study_f32/activations
RUN_ID="modality_panels_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="runs/ood_intent_study_f32/${RUN_ID}"
BOOTSTRAP=2000
MODELS=(qwen25vl7b gemma3_12b)
PANELS=(all text_only multimodal_only)

test -f "$MANIFEST"
for model in "${MODELS[@]}"; do
  test -f "$ACT_ROOT/$model/run.json"
done

mkdir -p "$OUT_ROOT/audits"
for model in "${MODELS[@]}"; do
  python -m ood_intent_study.audit_activations \
    --activations "$ACT_ROOT/$model" \
    --out "$OUT_ROOT/audits/${model}.json"
done

for panel in "${PANELS[@]}"; do
  for model in "${MODELS[@]}"; do
    python -m ood_intent_study.analyze \
      --activations "$ACT_ROOT/$model" \
      --manifest "$MANIFEST" \
      --out-dir "$OUT_ROOT/analysis/$panel/$model" \
      --modality-panel "$panel" \
      --bootstrap "$BOOTSTRAP"

    python -m ood_intent_study.visualize \
      --analysis-dir "$OUT_ROOT/analysis/$panel/$model" \
      --activations "$ACT_ROOT/$model" \
      --manifest "$MANIFEST" \
      --out-dir "$OUT_ROOT/figures/$panel/$model"
  done

  python -m ood_intent_study.compare_models \
    --analysis "qwen25vl7b=$OUT_ROOT/analysis/$panel/qwen25vl7b" \
    --analysis "gemma3_12b=$OUT_ROOT/analysis/$panel/gemma3_12b" \
    --out-dir "$OUT_ROOT/comparison/$panel"

  if [[ "$panel" != text_only ]]; then
    python -m ood_intent_study.compare_models \
      --analysis "qwen25vl7b=$OUT_ROOT/analysis/$panel/qwen25vl7b" \
      --analysis "gemma3_12b=$OUT_ROOT/analysis/$panel/gemma3_12b" \
      --common-panel \
      --out-dir "$OUT_ROOT/comparison_common_multimodal/$panel"
  fi
done

printf 'All modality-panel outputs: %s\n' "$OUT_ROOT"
```

`visualize` 不需要再次传 `--modality-panel`：它从 analysis 目录的 `analysis.json` 自动读取并复现面板筛选，同时校验样本 ID 指纹。`compare_models` 同样读取两个 analysis 目录，并拒绝比较 panel、manifest、标签敏感性或存储精度不一致的结果。

### 可选：strong-label 敏感性分析

当前只运行可识别的 `all` 与 `text_only`；有意排除 `multimodal_only`：

```bash
cd ~/workspace/LAM3/intent_subspace
set -euo pipefail

MANIFEST=runs/ood_intent_study/samples.jsonl
ACT_ROOT=runs/ood_intent_study_f32/activations
RUN_ID="strong_modality_panels_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="runs/ood_intent_study_f32/${RUN_ID}"
BOOTSTRAP=2000
MODELS=(qwen25vl7b gemma3_12b)
PANELS=(all text_only)

for panel in "${PANELS[@]}"; do
  for model in "${MODELS[@]}"; do
    python -m ood_intent_study.analyze \
      --activations "$ACT_ROOT/$model" \
      --manifest "$MANIFEST" \
      --out-dir "$OUT_ROOT/analysis/$panel/$model" \
      --modality-panel "$panel" \
      --strong-label-sensitivity \
      --bootstrap "$BOOTSTRAP"

    python -m ood_intent_study.visualize \
      --analysis-dir "$OUT_ROOT/analysis/$panel/$model" \
      --activations "$ACT_ROOT/$model" \
      --manifest "$MANIFEST" \
      --out-dir "$OUT_ROOT/figures/$panel/$model"
  done

  python -m ood_intent_study.compare_models \
    --analysis "qwen25vl7b=$OUT_ROOT/analysis/$panel/qwen25vl7b" \
    --analysis "gemma3_12b=$OUT_ROOT/analysis/$panel/gemma3_12b" \
    --out-dir "$OUT_ROOT/comparison/$panel"
done

printf 'Strong-label outputs: %s\n' "$OUT_ROOT"
```

## 5. 从 manifest 与 FP32 抽取开始的全流程

以下命令假定 FigStep 资产已经存在，JOOD 与 CS-DJ 的 prepared manifest 和图片已经按 `configs/default.json` 所指路径准备好。若这些资产尚未准备，请先执行 README 中的“准备缺失攻击图像”。

包含 VizWiz-VQA 与 JailBreakV-28K 的最短完整命令如下；新目录避免与旧 1,536 行激活混用：

```bash
cd ~/workspace/LAM3/intent_subspace

python -m ood_intent_study.run \
  --stage all \
  --work-dir runs/ood_intent_study_vizwiz_jbv28k_f32 \
  --models qwen25vl7b,gemma3_12b \
  --modality-panels all,text_only,multimodal_only \
  --bootstrap 2000
```

该命令依次重建 manifest、严格审计图片、为两个模型重新抽取 FP32 激活、运行三个面板、生成图和跨模型比较；中断后可用同一命令继续，抽取阶段会按相同 run fingerprint `--resume`。

```bash
cd ~/workspace/LAM3/intent_subspace
set -euo pipefail

RUN_ID="ood_f32_vizwiz_jbv28k_panels_$(date +%Y%m%d_%H%M%S)"
WORK="runs/${RUN_ID}"
MANIFEST="$WORK/samples.jsonl"
ACT_ROOT="$WORK/activations"
BOOTSTRAP=2000
MODELS=(qwen25vl7b gemma3_12b)
PANELS=(all text_only multimodal_only)

mkdir -p "$WORK"

python -m ood_intent_study.build_manifest \
  --out "$MANIFEST"

python -m ood_intent_study.audit_manifest \
  --manifest "$MANIFEST" \
  --require-images \
  --require-sidecar \
  --out "$WORK/manifest_audit.json"

python -m ood_intent_study.extract \
  --manifest "$MANIFEST" \
  --out-dir "$ACT_ROOT/qwen25vl7b" \
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

python -m ood_intent_study.extract \
  --manifest "$MANIFEST" \
  --out-dir "$ACT_ROOT/gemma3_12b" \
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

mkdir -p "$WORK/audits"
for model in "${MODELS[@]}"; do
  python -m ood_intent_study.audit_activations \
    --activations "$ACT_ROOT/$model" \
    --out "$WORK/audits/${model}.json"
done

for panel in "${PANELS[@]}"; do
  for model in "${MODELS[@]}"; do
    python -m ood_intent_study.analyze \
      --activations "$ACT_ROOT/$model" \
      --manifest "$MANIFEST" \
      --out-dir "$WORK/analysis/$panel/$model" \
      --modality-panel "$panel" \
      --bootstrap "$BOOTSTRAP"

    python -m ood_intent_study.visualize \
      --analysis-dir "$WORK/analysis/$panel/$model" \
      --activations "$ACT_ROOT/$model" \
      --manifest "$MANIFEST" \
      --out-dir "$WORK/figures/$panel/$model"
  done

  python -m ood_intent_study.compare_models \
    --analysis "qwen25vl7b=$WORK/analysis/$panel/qwen25vl7b" \
    --analysis "gemma3_12b=$WORK/analysis/$panel/gemma3_12b" \
    --out-dir "$WORK/comparison/$panel"

  if [[ "$panel" != text_only ]]; then
    python -m ood_intent_study.compare_models \
      --analysis "qwen25vl7b=$WORK/analysis/$panel/qwen25vl7b" \
      --analysis "gemma3_12b=$WORK/analysis/$panel/gemma3_12b" \
      --common-panel \
      --out-dir "$WORK/comparison_common_multimodal/$panel"
  fi
done

printf 'Complete FP32 run: %s\n' "$WORK"
```

不要把 `--storage-dtype float32` 改回 `float16`。Gemma 激活曾出现超过 FP16 最大有限值 65,504 的幅值，FP16 落盘会产生 `inf`，后续分析无法恢复原始值。

## 6. 输出检查与比较顺序

每个 `$WORK/analysis/<panel>/<model>/` 或 `$OUT_ROOT/analysis/<panel>/<model>/` 至少应检查：

```text
analysis.json
report.md
layer_probe_metrics.csv
source_metrics.csv
source_label_metrics.csv
leave_one_source_out.csv
leave_one_source_out_label_metrics.csv
leave_one_source_out_summary.csv
attack_shift_metrics.csv
source_domain_metrics.csv
```

每个 figures 目录通常包含：

```text
panel_composition.png
layer_curves.png
common_multimodal_panel_layer_curves.png   # text_only 不生成
source_generalization_heatmap.png
attack_score_shift_heatmap.png             # text_only 不生成
attack_domain_auroc_heatmap.png            # text_only 不生成
selected_layer_pca.png
pca_points.csv
```

`panel_composition.png` 先检查标准数据的来源/标签混杂；`layer_curves.png` 检查逐层标准可分性与冻结攻击召回；`source_generalization_heatmap.png` 优先显示 LOSO 的来源-标签召回，XSTest 拆为 safe/unsafe；两个 attack heatmap 分别显示攻击分数位移与攻击/标准 harmful 的域可分性；`selected_layer_pca.png` 只用于描述验证集所选层的二维投影。共同多模态曲线用于 readout 间严格同样本比较，不能由普通总体曲线替代。

建议按以下顺序解释结果：

1. 先读 `analysis.json` 和 `report.md`，确认 panel、模态计数、标签计数、选中层和样本指纹；
2. 在每个 panel 内查看 `layer_probe_metrics.csv`、LOSO 和最差来源，而不是只看 pooled AUROC；
3. 用 `source_label_metrics.csv` 与 `leave_one_source_out_label_metrics.csv` 单独检查 XSTest-safe/unsafe；
4. 在 `multimodal_only` 查看六个攻击条件的冻结 TPR、score shift 与 harmful-domain AUROC；
5. 最后比较 `all` 与两个单模态 panel。如果 `all` 明显更易分，但两个单模态 panel 下降，且来源/模态 domain 指标较高，则 pooled 边界可能利用了模态或数据源差异；反之，只有当单模态、LOSO、最差来源和外部攻击均保持稳定时，才更支持跨分布的意图子空间。

PCA 是描述性图，不是子空间理论的单独证据。来源簇分开可能来自模态、长度、角色、模板或图像载体；必须与冻结阈值、LOSO、来源 domain 可分性和攻击迁移结果共同解读。

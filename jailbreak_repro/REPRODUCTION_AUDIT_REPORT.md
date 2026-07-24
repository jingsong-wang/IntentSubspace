# 多模态越狱交叉攻防平台复现完整性与公平对比审计报告

审计日期：2026-07-14  
审计范围：`jailbreak_repro/` 平台代码、`sourcecode/` 中的官方源码快照、相邻 `benchmark/` 数据、现有运行产物与单元测试。  
目标模型：Qwen2.5-VL-7B、Gemma3-12B。  
审计方式：静态源码逐项比对、官方论文与随附源码交叉验证、资产完整性检查、现有单元测试和实验矩阵 dry-run。未修改平台源码，未执行需要加载大模型的完整 GPU 实验。团队补充确认：CS-DJ 的完整图像库和 CIDER 扩散权重存放在远端服务器；本报告将其视为“部署端已提供、当前本地审计环境不可独立验证”，不再判定为方法缺失。

## 一、总评

### 1.1 结论

**当前平台尚不能把完整矩阵的结果直接作为论文中的“原方法公平对比结果”。**

问题并非都出在攻击/防御的核心代码。FigStep、CS-DJ、JOOD、ECSO 和 CIDER 的若干核心步骤已经较好地对应了官方实现；Qwen 和 Gemma 的推理适配器也确实传入了图像，没有发现所有方法统一退化为纯文本攻击的情况。但以下阻断项或可审计性缺口会使结果不再等价于原论文方法，或不足以支撑论文中的公平性声明：

1. **AdaShield 未接入平台**：只有 588 MB 的源码压缩包，没有 CLI、训练/检索流程或防御适配器。
2. **UMK 未在 Qwen/Gemma 上实现白盒优化**：默认模式主动报错；固定 MiniGPT-4 对抗图像只能算迁移实验，不能算目标模型上的 UMK 复现。
3. **CS-DJ 远端图像库未形成可审计资产证明**：团队确认远端保存了完整图像库，因此不构成缺失实现；但平台默认只取 100 张图，而论文主配置为 10,000 张，且当前 manifest 没有证明远端运行实际使用了论文规定的数据版本与规模。
4. **JOOD 的主协议被改变**：平台默认混合系数包含 0 和 1，论文主配置为 0.1 到 0.9；平台按每个变换样本求平均，没有实现论文的“同一有害指令多次尝试后取最有害响应”的聚合指标；全量脚本的前 500 条截断还只覆盖第一个场景。
5. **CIDER 远端权重未形成可审计资产证明**：团队确认扩散权重已部署在远端，因此不构成缺失实现；但当前本地审计无法核对权重 SHA-256、加载日志、阈值对应关系和真实扩散数值回归。
6. **JailBreakV 数据缺失**：相邻 `benchmark/` 下没有 JailBreakV；残留样本使用失效的 Linux 绝对路径，不能构成有效结果。
7. **统一生成配置不等于原论文配置**：平台默认 `max_new_tokens=256, temperature=0`，而各论文的响应长度、温度、采样次数和聚合方式不同。统一配置可以用于新的标准化重评估，但必须这样命名，不能宣称复现原论文数值。
8. **通用 ASR 评测存在分母偏差**：普通任务的 judge 解析失败被赋为分数 1，并继续进入全体分母，系统性压低 ASR；只有 XSTest 使用了有效判定数作为分母。

### 1.2 方法级结论

| 方法 | 当前状态 | 多模态链路 | 可否直接用于论文公平对比 |
|---|---|---|---|
| FigStep | 核心攻击忠实，数据齐全 | 是 | **有条件可用**：提高生成预算，并说明是新目标模型上的复现评估 |
| CS-DJ | 核心拼图流程接近官方；远端资产由团队确认，但默认检索规模、拆分重试、缓存和评测协议有问题 | 是 | **当前不可直接作为公平结果** |
| JOOD | 图像混合实现接近官方，但默认超参数、截断和指标协议不符 | 是 | **当前不可用** |
| UMK | 仅有 MiniGPT-4 官方产物迁移；Qwen/Gemma 白盒优化缺失 | 是，但非目标模型优化产物 | **不可作为目标模型 UMK 结果** |
| ECSO | 四阶段核心流程基本完整 | 是；第 4 阶段按原文主动去图像 | **有条件可用**：仅限真实图像输入，需统一或披露解码差异 |
| CIDER | 核心检测器较完整；远端权重由团队确认，但缺少可审计 manifest，且矩阵大量超出原文攻击范围 | 是 | **有条件可运行；不能把范围外实验称为原文复现** |
| AdaShield | 未实现/未接入 | 不适用 | **不可用** |
| CISR | 内部检测器接入完整，但无外部原论文可核验，且误报率偏高 | 是 | **只能作为自研方法，不能列为已复现外部基线** |

## 二、审计口径

本报告区分三种容易混淆的声明：

1. **数值复现**：目标模型、数据划分、攻击预算、生成参数、随机重复、评测器和聚合方式均与论文一致，目标是重现原表数值。
2. **方法复现/模型适配**：保留原算法核心机制，但将目标模型换成 Qwen2.5-VL-7B 或 Gemma3-12B。此类结果可用于同一平台内的横向比较，但不能称为重现原论文数值。
3. **迁移或范围外评估**：使用在其他模型上生成的对抗产物，或把防御用于原论文没有声称覆盖的攻击类型。此类结果有研究价值，但必须单独命名。

当前平台主要属于第 2 类，并混入部分第 3 类。只有在补齐资产和修正协议后，部分方法适合进入统一公平比较表。

## 三、攻击方法审计

### 3.1 FigStep

**原文方法。** FigStep 是黑盒、单轮、基于排版图像的攻击。它先把有害问题改写为不完整的编号步骤，把这些文本渲染为图像，再用一个表面无害的文本指令要求模型补全图中的三个步骤。原文同时发布 SafeBench（500 条、10 类）和 SafeBench-Tiny（50 条、10 类）。参见 [FigStep 论文](https://arxiv.org/abs/2311.05608)。

**平台实现。** `attacks.py:42-46` 使用了官方 incitement prompt；`attacks.py:112-167` 直接加载随附 SafeBench CSV 和图像，并把图像交给受害模型。评测时使用原始有害目标而不是只使用表面文本提示，方向正确。随附 SafeBench 有 500 条，Tiny 有 50 条，图像资源实际存在。

**完整性判断。** 核心方法和官方发布数据基本完整，没有发现以占位图或纯文本替代攻击图像的情况。它复现的是基础 FigStep，而不是后续增强变体。

**风险。**

- 平台默认只生成 256 tokens，而攻击明确要求三个约 100 词的段落。截断很可能把本应成功的响应判为低危，形成不公平的假阴性。
- 原论文的目标模型和评测流程不同，因此在 Qwen/Gemma 上应称为“方法适配后的重评估”，不能直接与原论文表格数值对齐。
- 现有测试没有覆盖 SafeBench 行数、图像存在性、prompt 字面一致性和端到端图像传递。

**结论：有条件通过。** 提高响应预算、固定模型版本并验证完整输出后，可进入统一平台比较。

### 3.2 CS-DJ

**原文方法。** CS-DJ 先用 Qwen2.5-3B 将有害指令拆成 3 个子问题；用 CLIP ViT-L/14 从 10,000 张 LLaVA-CC3M 图像中选择对比性干扰图；构造 3 张红字子问题图和 9 张干扰图，拼成 3 列、共 12 个面板的图像；最后配合 benign teacher prompt 攻击目标 MLLM。论文报告多组随机实验和 ASR/EASR。参见 [CVPR 2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Distraction_is_All_You_Need_for_Multimodal_Large_Language_Model_CVPR_2025_paper.html)。

**平台实现。** CLIP 检索、3 个子问题、9 张干扰图、12 面板拼图、字体和最终提示均与官方源码高度接近；最终图像确实传给受害模型。因此核心攻击没有退化为纯文本。

**影响公平性的关键问题。**

- 本地 `sourcecode/CS-DJ-main/data/images/` 为 0 个文件，但团队确认真实图像库存放在远端服务器。本报告因此不把它判为缺失实现；远端正式实验仍需记录数据来源、文件数、目录快照/校验和与实际 `--csdj-num-images`，否则审稿人无法确认使用的是论文规定的 10,000 张 LLaVA-CC3M 检索池，而不是替代图集或子集。
- `run_experiment.py:701` 的默认 `--csdj-num-images=100`，比论文主配置 10,000 少两个数量级。README 中虽提到 10,000，但全量脚本没有覆盖默认值。
- `attacks.py:490-493` 先检查输出不含模板占位词，下一分支却对任意 3 个编号项直接返回，导致模板回显也会被接受；声称的六次重试实质上被绕过。
- `subquestions.json` 只按原指令缓存，没有记录辅助模型版本、温度、prompt 版本等；最终拼图只按类别和序号复用（`attacks.py:750-754`），没有校验图像池、随机种子、检索规模、CLIP 版本或字体。改变配置后可能静默复用旧攻击图。
- 平台没有实现论文的 EASR 与多组随机实验协议，只计算通用响应级 ASR。
- 原官方受害模型生成预算约 1000 tokens、温度 0.1；平台统一使用 256 tokens、贪心解码，可能改变攻击强度。
- 字体缺失时会静默退回默认字体。当前字体存在，但论文级运行应失败退出或记录替代，而不是静默更换。

**结论：当前不通过公平性门槛，但不是因为远端图像缺失。** 算法骨架较忠实；在远端图像库真实存在的前提下，仍需修正默认检索规模、模板回显重试、缓存指纹、输出预算和 EASR/重复实验协议，之后才能作为 CS-DJ 的公平对比。

### 3.3 JOOD

**原文方法。** JOOD 将有害图像与无关的分布外辅助图像进行 Mixup、CutMix 或 ResizeMix 等组合，用 OOD 内容分散安全对齐注意力。多模态主设置使用 5 张辅助图、9 个混合系数 `0.1,...,0.9`，图像缩放/填充到 320×320，生成温度和 `top_p` 均为 1。对每条有害指令会产生多个候选响应，论文在候选集中选择最有害响应后再计算指令级 ASR。参见 [CVPR 2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Playing_the_Fool_Jailbreaking_LLMs_and_Multimodal_LLMs_with_Out-of-Distribution_CVPR_2025_paper.html)。

**平台实现。** 平台直接使用/移植了官方 `mixaug.py`、`randaug.py` 和 prompt 前缀，AdvBenchM 的 7 个场景、场景有害图和 5 张无害辅助图也存在。图像在运行时真实混合并送入模型，不是纯文本替代。

**严重协议偏差。**

- `attacks.py:76` 默认混合系数是 `0.0,0.1,...,1.0`，`run_experiment.py:710` 还把它描述为官方默认。论文主设置是 `0.1,...,0.9`。`alpha=0` 等于原图，`alpha=1` 会丢失有害图语义，两端点改变攻击预算和样本权重。
- 平台对所有变换样本直接求 harmful rate；没有按原始有害指令分组，也没有实现“候选集中取最有害响应”的论文级聚合。这不是同一个 ASR 定义。
- `run_all_experiments.sh` 将 JOOD 限为前 500 条。当前生成顺序按场景、图像、混合系数排列，因此前 500 条全部来自第一个 `bomb_explosive` 场景，并不覆盖 7 类；这个截断会造成严重选择偏差。
- 全量矩阵只运行默认 Mixup，未覆盖论文的其他主要变换。若只报告 Mixup，必须明确标为 JOOD-Mixup，而不是笼统的 JOOD 全方法结果。
- 统一的 `temperature=0, max_new_tokens=256` 与论文的 `temperature=1, top_p=1` 和官方脚本的长输出预算不符。

**结论：当前不通过。** 核心图像变换实现可信，但现有采样和指标足以改变结论；修正前不能作为论文公平基线。

### 3.4 UMK

**原文方法。** UMK 是白盒双模态联合优化攻击。第一阶段从噪声开始，用 PGD 优化通用对抗图像，使 MiniGPT-4 在无文本输入时倾向有害语义；第二阶段用 PGD 与 GCG 联合优化图像前缀和文本后缀，使模型对多种有害目标产生肯定响应。论文使用 66 条有害语句和从 AdvBench 抽取的 66 个目标-响应对，并在 MiniGPT-4 上报告 96% ASR；论文明确指出跨架构、参数和 tokenizer 的迁移能力有限。参见 [UMK 论文](https://arxiv.org/abs/2405.17894)。

**平台实现。**

- 默认 `target_optimized` 在 `attacks.py:1158-1164` 主动抛出 `NotImplementedError`，明确说明尚未实现 Qwen/Gemma 内部梯度优化。
- `transfer_eval` 使用官方 MiniGPT-4 对抗图并标记 `paper_claim_compatible=False`，这个边界处理是诚实的。
- `target_optimized_artifact` 只要求用户给出图像并填写匹配的模型字符串，然后直接标记兼容；没有验证优化日志、模型 revision、损失、步数、图像校验和或共同优化的文本后缀。

**不完整性。** Qwen2.5-VL-7B 和 Gemma3-12B 上没有原文要求的白盒双阶段优化，因此不存在这两个目标模型上的 UMK 主结果。更重要的是，目标模型图像若仍与官方 MiniGPT-4 文本后缀搭配，也不能证明它是联合优化得到的 UMK。

**结论：目标模型复现不通过。** 官方产物只能作为“MiniGPT-4 UMK 对 Qwen/Gemma 的迁移评估”单独报告。全量脚本实际上也没有运行 UMK。

## 四、防御方法审计

### 4.1 ECSO

**原文方法。** ECSO 是训练免费的四阶段防御：先让同一 MLLM 直接回答；再结合原图让同一模型判断初始响应是否有害；若有害，则结合原问题生成 query-aware 图像描述；最后移除图像，把描述和原问题送给同一 MLLM 的语言部分生成安全响应。最后一步主动退化为 text-only 是方法设计本身，而不是实现错误。参见 [ECCV 2024 论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02634.pdf)。

**平台实现。** `defenses.py:12-24` 的三个 prompt 与官方版本对应；`defenses.py:108-168` 完整实现 direct → tell → caption → text-only safe generation，并且前 3 个涉及图像的阶段继续传图，第 4 阶段 `image_path=None` 与论文一致。没有用外部文本模型替换同一 MLLM。

**差异和边界。**

- 官方判断通常只接受 `yes`/`yes.`；平台 `defenses.py:55-57` 使用 `startswith("yes")`，会接受更宽的输出集合。影响可能较小，但属于行为变更。
- 原官方调用通常给各阶段 1024 tokens、温度约 0.2；平台 tell 阶段默认 32 tokens，其余阶段继承 256 tokens、温度 0。caption 或最终安全响应可能被截断。
- XSTest 没有图像。平台仍会运行 ECSO，使其变成纯文本自检/重写流程。这不是 ECSO 原论文的多模态 I2T 防御，应标为范围外的文本扩展，不能与图像输入结果混合解释。
- 样本缺图时没有失败退出或显式“不适用”标记，可能把数据错误静默变成纯文本实验。

**结论：有条件通过。** 对真实图像样本，核心流程完整；需调整或完整披露阶段生成配置，并把无图像场景单独标记为 out-of-scope。

### 4.2 CIDER

**原文方法。** CIDER 面向优化型对抗图像越狱。它使用固定的 LLaVA-1.5-7B 跨模态编码器，将原图和逐步扩散去噪图分别与查询求余弦相似度，在 0、50、…、350 去噪步上检测相似度变化；阈值按 95% 干净图通过率校准；检测到攻击后输出固定拒绝，否则将处理后的图像交给目标 MLLM。参见 [Findings of EMNLP 2024 论文](https://aclanthology.org/2024.findings-emnlp.803/)。

**平台实现。** `cider.py` 对固定 LLaVA-1.5-7B 架构做了维度/类签名检查，默认去噪步和阈值对应官方设置，扩散调用和相似度差判定与源码一致；通过样本继续使用 0 步处理后的 224×224 图像。该实现没有改成受害模型隐藏层或纯文本过滤。

**资产可验证性与适用范围问题。**

- 本地审计环境没有官方扩散权重 `sourcecode/CIDER-main/code/models/diffusion_denoiser/imagenet/256x256_diffusion_uncond.pt`，但团队确认权重存放在远端服务器。本报告不再据此判定 CIDER 无法运行；正式结果仍应保存权重 SHA-256、成功加载日志、去噪步列表和少量中间图像/相似度轨迹，证明远端确实执行了官方扩散流程而不是替代或占位去噪器。
- 平台用按图像/步数固定的确定性噪声，而官方使用运行时随机噪声。确定性改动有利于重现，但已发布阈值未必仍对应 95% 干净通过率；必须用当前实现重新验证或校准阈值。
- PIL 双线性缩放与官方 torchvision 预处理可能产生小的数值差异，同样需要数值回归验证。
- CIDER 原论文声称覆盖的是 optimization-based adversarial image attacks。矩阵把它用于 FigStep、CS-DJ、JOOD、HADES 和 XSTest。尤其 XSTest 没有图像，`cider.py:604` 直接记为 `no_image` 并跳过检测，实际是 no-op。此类结果只能作为跨攻击泛化/不适用对照，不能称为原论文配置。
- 现有单测使用 fake 组件验证流程，真实 PyTorch 扩散批处理测试在本次环境中被跳过，没有验证官方权重上的逐步图像或相似度数值等价性。

**结论：实现本身有条件通过，远端资产存在性按团队声明接受但尚未独立验证。** 完成远端真实数值回归后，只能在原文适用的优化型图像攻击上标记为 paper-scope；其他攻击需明确为 out-of-domain generalization。

### 4.3 AdaShield

**原文方法。** AdaShield-S 使用人工静态防御 prompt；AdaShield-A 则让 defender LLM 根据目标 MLLM 的失败响应迭代生成场景相关防御 prompt，经过训练/验证形成 prompt pool，并在推理时按图文查询相似度检索合适 prompt。论文阈值为 `alpha=0.8, beta=0.7`，检索本身是关键组成，而不是随机选择 prompt。参见 [AdaShield 论文](https://arxiv.org/abs/2403.09513)。

**平台状态。** `sourcecode/AdaShield-main.zip` 存在，但平台的攻击/防御选项、运行脚本和 README 均没有 AdaShield 适配器。没有静态 prompt 基线、自动迭代、目标模型相关 prompt pool、验证筛选或检索流程。

**结论：完全未实现。** 保存官方源码压缩包不等于复现。AdaShield 不能出现在当前平台的方法对比表中，除非明确写为“未纳入”。

### 4.4 CISR

**方法定位。** CISR 代码来自相邻的 `intentguard_refactor`，使用目标模型隐藏状态构建低秩配对意图子空间，以 raw/residual anchor 坐标、图像角色和 TinyMLP 进行检测，触发后硬拒绝。平台会检查 detector artifact 与目标模型是否匹配，并在含图样本上实际提取多模态隐藏状态。

**可核验范围。** `sourcecode/` 中没有与 CISR 对应的外部论文或官方第三方实现，因此本审计只能检查平台内部一致性，不能证明“完整复现原文”。它应在论文中列为自研方法，而非外部基线。

**现有报告暴露的风险。**

- Qwen held-out：AUROC 约 0.822，TPR 约 0.819，FPR 约 0.235；成功攻击召回约 0.500。
- Gemma held-out：AUROC 约 0.894，TPR 约 0.919，FPR 约 0.416；成功攻击召回约 0.731。
- 图像角色分组更差：Qwen OCR-layout FPR 为 1.0、semantic-OCR-stitch FPR 约 0.923；Gemma 对应 FPR 约 0.718 和 1.0。
- 平台只要模型字符串匹配就可令 `paper_claim_compatible=True`（`defenses.py:226` 等），这个标志没有检查 held-out FPR、攻击覆盖范围或角色分布，不能作为论文级有效性证明。
- 未见真实模型端到端回归测试；现有 CISR 测试主要验证 artifact 读取、模型匹配和触发/不触发分支。

**结论：内部接入完整但统计风险较高。** 可作为自研防御评估，但必须同时报告 benign utility、分角色 FPR、成功攻击召回和置信区间，不能只报告 ASR 降幅。

## 五、基准数据审计

### 5.1 HADES

相邻目录中有约 981.6 MB、752 个文件，元数据 750 条，覆盖 5 类、每类约 150 条，与 HADES 发布的最终多模态攻击数据规模一致。元数据中的旧绝对路径可通过文件名在本地图像目录中重定位，当前样本图像实际存在。

平台评估的是 **HADES 官方发布攻击样本**，没有复现 HADES 的完整数据生成/优化流程。因此论文表述应为“在 HADES benchmark 上评估”，而不是“复现 HADES 攻击算法”。参见 [HADES 论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9265_ECCV_2024_paper.php)。

### 5.2 JailBreakV / JailBreakV-mini

当前 `../benchmark/` 下没有 JailBreakV 数据目录。残留运行样本含 `/home/...` 形式的失效 Linux 绝对路径，且没有完整 responses/config 结果，说明先前任务没有形成有效实验。

`benchmarks.py:128-129` 对无法重定位的绝对路径仍原样返回，错误会延迟到模型打开图像时才暴露。这不会静默转为文本输入，但会产生中途失败和不完整缓存。参见 [JailBreakV 论文](https://arxiv.org/abs/2404.03027)。

**结论：当前不可用。** 必须补齐发布数据并在预处理阶段验证每一张图像存在。

### 5.3 XSTest

当前数据有 450 条：250 条 safe、200 条 unsafe，平台保留原标签，并使用独立的三分类评测和有效判定分母。数据规模和标签结构与 XSTest 一致。参见 [XSTest 论文](https://aclanthology.org/2024.naacl-long.301/)。

XSTest 是纯文本过度拒答基准，不是多模态越狱基准。它适合衡量防御造成的 utility/safety trade-off，但 CIDER 在此处是 no-op，ECSO 在此处是范围外文本变体；两者都不应被解释为原论文多模态防御效果。

## 六、多模态链路与退化风险

### 6.1 未发现的系统性问题

- Qwen 适配器在 `models.py:336-346` 将本地图像作为 Qwen-VL image content 传入处理器。
- Gemma 通用 VLM 适配器在 `models.py:348-359` 用 PIL 读取 RGB 图像并把 images 交给处理器。
- FigStep、CS-DJ、JOOD、UMK transfer、HADES 和 JailBreakV loader 都生成/读取 `image_path`，有效路径下会走真实视觉分支。
- ECSO 最终安全生成阶段移除图像是原算法定义，不应被误判为退化。

### 6.2 仍需阻止的退化/失真

- ECSO 对无图样本仍执行，形成非原文的 text-only 防御变体。
- CIDER 对无图样本静默跳过，形成 no-op 防御。
- JailBreakV 失效绝对路径没有在 loader 层失败退出。
- CS-DJ 必须使用团队所述的远端完整图像库；任何临时小图集、不同 CC3M 切片或不透明替代都会改变检索分布。
- UMK 的 MiniGPT-4 对抗图在 Qwen/Gemma 上是跨模型迁移产物，不是为当前视觉编码器优化的攻击。
- 缺少统一的预运行校验，无法保证每条样本在送入模型前确实存在且可解码。

## 七、共享实验与评测层问题

### 7.1 生成参数和攻击预算

`run_experiment.py:745-748` 的公共默认是 256 tokens、温度 0、`top_p=0.9`。这可作为统一平台协议，但与原方法配置不同：CS-DJ 官方实现约 1000 tokens/温度 0.1，JOOD 官方脚本为温度与 `top_p` 均 1、最长 4096，UMK 官方评估采用采样，ECSO 阶段预算约 1024。FigStep 自身也需要足以生成三个长步骤的预算。

公平做法必须二选一：

1. 按每篇论文的配置复现，并分别报告；或
2. 预先定义统一预算、统一解码和统一 judge，明确称为 standardized re-evaluation，并确保预算不会截断任何方法的基本输出格式。

当前实现介于两者之间，容易把统一设置误写成原文设置。

### 7.2 Judge 与 ASR

平台用统一 Qwen2.5-VL judge 对回答给出 1-5 分并以阈值 3 计算 harmful rate。统一 judge 有利于平台内部一致性，但原论文分别使用人工、关键词、Beaver-Dam、GPT-4、LlamaGuard/HarmBench 类评测或特殊聚合，数值不可直接横比。

更严重的是，`judges.py:128` 在 JSON 解析失败时返回 `harmfulness_score=1`；`judges.py:459-470` 的普通 ASR 汇总仍以全部行 `n` 为分母。解析失败因此被当作无害响应，系统性压低 ASR。XSTest 汇总已经使用 `valid_n`，普通 ASR 应采用同样的 coverage 逻辑或重试至成功。论文至少应同时报告 valid-only ASR、judge coverage 和 parse error 数。

通用 judge 还缺少以下证据：

- 与人工标注的校准集和一致性指标；
- 对 Qwen 与 Gemma 回答风格的分模型偏差检查；
- 对拒答但包含复述、对攻击文本照抄、部分可执行内容等边界案例的标注协议；
- 随机抽样人工复核及置信区间。

### 7.3 样本聚合和独立性

JOOD、CS-DJ 等方法对同一原始目标产生多次尝试。直接把每个尝试当作独立样本求平均，会过度加权尝试数较多的方法，也与论文的目标级/最优候选 ASR 不同。平台应保留 `original_goal_id`、`attempt_id`、`seed`，先按目标聚合再统计置信区间。

### 7.4 缓存、版本与可追溯性

- CS-DJ 攻击产物的缓存键不包含核心生成配置，存在跨实验污染。
- 数据集快照、源码快照、模型 revision、CLIP/辅助模型 revision 和 CIDER 权重内容没有统一写入不可变 manifest。
- 模型名可能解析到可变的仓库主分支，未记录 commit SHA。
- 当前目录未见能锁定所有官方方法相互冲突依赖的统一环境锁文件。
- `paper_claim_compatible` 多为调用方提供的布尔元数据，并没有验证完整实验协议，不能作为审计证明。

### 7.5 全量矩阵本身不完整

`run_all_experiments.sh` 生成 24 个 Qwen 任务：3 个攻击 × 4 个防御，加 3 个 benchmark × 4 个防御。它没有运行 UMK，也没有 AdaShield；JOOD 只跑默认变换；CS-DJ 没有传 10,000 图配置。dry-run 虽显示 `24/24 succeeded`，但不加载数据或权重，所以不能验证远端 CS-DJ 图像库、CIDER 权重或缺失的 JailBreakV 数据是否在实际执行环境中可用。

目前也没有一套 Qwen/Gemma 的完整已完成矩阵产物可供反向核验。现存结果主要是 smoke run 或单一 XSTest 运行，不能证明平台端到端可重复。

## 八、测试覆盖评价

本次运行：31 项单元测试中 30 项通过，1 项跳过。跳过项是依赖 PyTorch 的 CIDER 扩散批处理测试，原因是轻量测试运行时没有 PyTorch。

已有测试覆盖：

- CIDER 的阈值方向、无图像跳过、fake 去噪流程、组件签名；
- CISR artifact 读写、模型不匹配、触发/放行分支；
- CS-DJ 路径重定位和 12 面板图像结构；
- judge 批处理、XSTest 解析与有效分母。

关键缺口：

- FigStep 数据/图像/prompt 的端到端一致性；
- CS-DJ 真实 10,000 图检索、辅助模型输出约束和缓存失效；
- JOOD 官方 9 个 alpha、7 场景分层采样及目标级聚合；
- UMK 双阶段优化或 artifact provenance；
- ECSO 四阶段在真实 Qwen/Gemma 上的图像存在性和 channel 检查；
- CIDER 官方权重上的数值回归；
- HADES/JailBreakV 全图像存在性；
- 两个真实目标模型上的小规模端到端 golden run。

## 九、论文使用建议与整改优先级

### P0：进入主结果表前必须完成

1. 在远端实验环境对 CS-DJ 的 10,000 张 LLaVA-CC3M 检索图生成可审计 manifest；默认和全量脚本显式使用论文规模，修复子问题模板回显检查，并为缓存加入完整配置指纹。
2. JOOD 改为论文主设置 `alpha=0.1,...,0.9`；按 7 场景分层采样；以原始目标为单位实现论文的候选聚合；若只跑 Mixup，方法名写为 JOOD-Mixup。
3. 若要报告 Qwen/Gemma 上的 UMK，必须实现并运行针对每个目标模型的 PGD+GCG 双阶段优化，同时保存图像、后缀、损失曲线和模型 revision。否则仅报告 UMK transfer baseline。
4. 对远端 CIDER 官方扩散权重记录 SHA-256 和加载日志，并在当前确定性噪声实现上重新验证 95% clean pass threshold；限定 paper-scope 结果为优化型对抗图像攻击。
5. 要么完整接入 AdaShield-S/A 的训练、验证、prompt pool 和检索，要么从“已复现方法”列表中删除。
6. 补齐 JailBreakV 数据，预运行时逐项校验图像存在和可解码，清理失效绝对路径缓存。
7. 普通 ASR 排除 judge parse error 或重试成功，报告 valid-only ASR 与 coverage；用人工标注子集验证统一 judge。

### P1：保证公平性和可重复性

1. 明确选择“逐论文配置复现”或“统一平台重评估”，不要混合表述；统一协议至少给足 FigStep/CS-DJ/JOOD 所需输出长度。
2. 以攻击目标而非变换样本为统计单位，记录 seed/attempt，并给出 bootstrap 置信区间。
3. 固定 Qwen、Gemma、辅助 Qwen、CLIP、LLaVA encoder 的 commit revision；对所有数据和权重写 SHA-256 manifest。
4. 对无图像样本显式标记 defense not applicable：CIDER 不应计作有效防御，ECSO 需标为 text-only extension。
5. 同时报告攻击 ASR 和 benign utility/over-refusal。CISR 尤其要披露分角色 FPR 和成功攻击召回。
6. 建立每种方法至少 1 个真实图像、真实模型的端到端回归样本，并在全矩阵前执行严格 preflight。

### P2：建议补充

1. 在主表之外单列 paper-scope、cross-model adaptation、out-of-domain generalization 三组结果。
2. 对 CIDER 确定性去噪与官方随机去噪做数值敏感性分析。
3. 对 Qwen judge 的自家族偏差做交叉 judge/人工复核。
4. 保留原始模型输出，不只保留判定分数，以支持盲审复核。

## 十、最终可发表性判断

在当前状态下：

- **可以作为工程原型或初步实验**：FigStep；真实图像上的 ECSO；CISR 自研方法的内部对比。
- **验证远端资产并修正协议后可能进入主表**：CS-DJ、JOOD、CIDER；JailBreakV 仍需确认实际执行环境的数据可用性。
- **不能宣称已在 Qwen/Gemma 上复现**：UMK。
- **完全未纳入**：AdaShield。

因此，现有完整矩阵不应直接用于论文中的公平对比表。推荐先完成 P0 项，再冻结一次不可变的实验 manifest，并对每个方法分别给出“原文一致项、平台统一改动项、范围外项”。完成这些条件后，FigStep、CS-DJ、JOOD、ECSO 和 CIDER 才能形成可辩护的跨模型方法复现；UMK 仍需要目标模型白盒优化，AdaShield 仍需要完整接入。

## 参考论文

1. Gong et al. [FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts](https://arxiv.org/abs/2311.05608).
2. Yang et al. [Distraction is All You Need for Multimodal Large Language Model Jailbreaking (CS-DJ)](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Distraction_is_All_You_Need_for_Multimodal_Large_Language_Model_CVPR_2025_paper.html).
3. Jeong et al. [Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy (JOOD)](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Playing_the_Fool_Jailbreaking_LLMs_and_Multimodal_LLMs_with_Out-of-Distribution_CVPR_2025_paper.html).
4. Wang et al. [White-box Multimodal Jailbreaks Against Large Vision-Language Models (UMK)](https://arxiv.org/abs/2405.17894).
5. Gou et al. [Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation (ECSO)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02634.pdf).
6. Wang et al. [CIDER: Detecting and Mitigating Multimodal Jailbreak Attacks](https://aclanthology.org/2024.findings-emnlp.803/).
7. Wang et al. [AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](https://arxiv.org/abs/2403.09513).
8. Li et al. [HADES: An Image-based Jailbreak Attack against Large Vision-Language Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9265_ECCV_2024_paper.php).
9. Luo et al. [JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks](https://arxiv.org/abs/2404.03027).
10. Röttger et al. [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://aclanthology.org/2024.naacl-long.301/).

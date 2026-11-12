# JEV、循环 Transformer 与潜在推理调研

调研日期：2026-11-11。依据论文原文、arXiv 摘要、会议论文页和作者项目页；未复现实验。以下性能均为作者报告，不代表当前项目的预期收益。

## 当前方案与结论

当前 `src/loop_jev/training/answer_slot_distill.py` 的学生输入为题目、选项及 `Answer: (`；冻结教师额外看到训练 rationale。CE 在有效答案字母上计算，温度 KL 在答案位置的全词表上计算。README 中 hidden loss 权重为 0。代码虽然支持答案位置多层 hidden 对齐，但没有连续思考步骤或循环结构。

已经有直接研究“JEV-style 决策接口＋循环隐状态”的论文：SanSi、LEV、RELOAD。不能将这个组合本身作为新颖性。对当前 Qwen3-1.7B 改造，RELOAD 最值得优先读；SanSi/LEV 是直接对照；CODI/LOTUS 是蒸馏与过程监督参考。

这里的 JEV-style 指固定候选集上的决策分布接口。这些独立工作没有公开 TypeSafe Jev 的内部结构，也不是直接修改其闭源权重。单次返回一个选择结果与内部重复计算并不冲突。

## 直接相关工作

### RELOAD：普通预训练模型改造，优先级最高

[Extending Reasoning in Typed Decision Models via Looping](https://arxiv.org/abs/2611.07736)，2026-11-05，预印本。HTML 内标题含 Adaptive Looping，arXiv ID 相同。

将模型分为前部 P、中间块 G、后部 S。针对预算 K，更新为：

`h₀ = P(embedding(x))`

`hₖ₊₁ = hₖ + (G(hₖ) − hₖ) / K`

`pK = softmax(head(S(hK)))`

训练 K=1、2、3、4 的不同预算端点，采用 CE＋0.5×Brier；不是一条轨迹上简单读四次。实验含 Qwen3-0.6B、1.7B、4B，以及三个双向编码器。Qwen3-1.7B 循环第 9–18 层，编号从 0 开始。解码器使用 Kev 风格的选项隔离与 pointer head，和本项目答案字母 LM head 有区别。

收益主要体现在超过训练推理长度的任务；编码器最高提升 23 个百分点，不能将该最大收益套用到 Qwen 或 MathQA。控制器以约四分之三的计算保留大部分收益。原文包含训练计算量、更多数据、参数增加等对照。

对本项目的价值：给出了同底模的中间块循环与联合训练先例，可保留当前答案读出先单独检验结构，再比较 CE 与 CE＋教师 KL。

### SanSi：直接的循环决策模型先例

[SanSi: A Looped Typed Decision Model for System 1.5 Thinking](https://arxiv.org/abs/2610.07730)，2026-10-06，预印本。

使用已接受循环预训练的 Ouro；共享层反复更新状态，每轮均读取选项概率，以 CE/Brier 监督。1.4B 主模型在 10,027 个决策上达到 72.0%，比同形状非循环对照高 13.5 个百分点。循环的准确率收益与概率校准收益并不同步，更多循环也会增加错误答案的自信。

[作者项目页](https://minnesotanlp.github.io/Sansi/)；[代码](https://github.com/minnesotanlp/Sansi)；[权重](https://huggingface.co/minnesotanlp/SanSi)。

### LEV：循环决策与按需停止

[LEV: Thinking Fast and Deep with Looped Decision Models](https://arxiv.org/abs/2610.20951)，2026-10-14，预印本。

建立在循环预训练底模上，每轮读出并联合训练各深度，再由 stopper 决定是否继续。2.6B 在 JevBench 从深度 1 的 63.2% 升至深度 4 的 76.2%；按需深度节省 15%–24% 计算，准确率损失不足一个百分点。其 stopper 学习与最深层答案是否一致，与 RELOAD 事前选择预算的控制器不同。

[代码](https://github.com/OpenLAIR/lev)；[作者项目页](https://openlair.github.io/lev/)。

## 循环与隐式推理的重要参考

| 论文 | 核心价值 | 对当前项目的意义 |
| --- | --- | --- |
| [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://aclanthology.org/2025.emnlp-main.36/)，EMNLP 2025 | 连续 hidden feedback，加教师/学生答案前位置的多层特征对齐；共享模型同时学习显式 CoT | 最接近现有教师蒸馏思路。但原方法含 CoT 辅助训练和 L1 特征对齐，当前冻结教师＋logits KL 不是 CODI 复现 |
| [Training Large Language Models to Reason in a Continuous Latent Space（Coconut）](https://arxiv.org/abs/2412.06769)，2024 首发、COLM 2025 | 将最后 hidden state 作为下一个连续思考输入；课程式替换显式 CoT | 连续思考入门与基线。仍有串行 latent 步骤，不等于固定工作区的深度循环 |
| [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers（LOTUS）](https://arxiv.org/abs/2606.31779)，2026-06，预印本 | 固定潜在块循环更新，训练时并行监督中间 CoT token | “循环＋过程监督”重点参考。3B GSM8K 设置接近显式 CoT；报告的是思考阶段 2.5–6.9 倍加速，不是相对当前直接答题模型的端到端收益 |
| [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384)，2025-11，预印本 | 预训练权重初始化循环结构，使用循环次数课程与继续训练 | 结构改造与稳定性参考；论文数学实验约 50B token 继续训练，不能视为少样本 SFT 即可复现 |
| [Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts（ETD）](https://arxiv.org/abs/2510.07358)，2025-10，预印本 | 从预训练模型选取推理相关层，形成 encoder／thinking block／decoder | 比循环整个 Qwen 更有针对性的结构参考；收益依赖 mid-training |
| [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)，ICLR 2025 | 实证与理论研究共享层循环和推理能力、有效深度的关系 | 理解为何循环值得做，以及参数量与计算量必须分开比较 |
| [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach（Huginn）](https://arxiv.org/abs/2502.05171)，2025，NeurIPS 2025 | 大规模深度循环语言模型，3.5B 参数、800B 训练 token | 原生循环架构的重要先例，成本与当前后训练场景差别较大 |
| [Scaling Latent Reasoning via Looped Language Models（Ouro）](https://arxiv.org/abs/2510.25741)，2025-10 首发 | 开放循环预训练模型，学习计算深度分配 | 可直接作为循环底模对照；换底模的收益需与结构改造收益区分 |
| [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451)，ICLR 2026 | 变长度轨迹、时间及步长条件、shortcut consistency | 在固定循环有效后，参考如何训练可控计算预算 |
| [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822)，2026-04，预印本 | 组合泛化、推理深度外推与过度循环退化的受控研究 | 指导评估是否获得多步能力，避免只看同分布选择题准确率 |

补充起点：[Universal Transformers](https://arxiv.org/abs/1807.03819)；其他递归架构：[HRM](https://arxiv.org/abs/2506.21734)、[TRM](https://arxiv.org/abs/2510.04871)。后两者的核心证据来自谜题等任务，不能直接推出对预训练语言模型数学选择题有效。

较新相邻工作：[LLoCoT](https://arxiv.org/abs/2610.11472) 迭代更新潜在工作区，主要在代码生成上评估；[Selective Reasoning for Typed Decision Models](https://arxiv.org/abs/2610.19644) 保留 typed 接口并按需生成、读取和中断显式推理，适合作为效率对照，但其内部仍生成文本。

## 建议实验与可能的研究问题

建议优先验证“RELOAD 式中间块循环＋当前答案字母读出＋完整 CoT 教师 KL”。这是后续实验建议，不是已验证的新方法。优先级高于从零训练 Huginn 或直接切换整个底模。

最小对照：

1. 当前单次计算 CE。
2. 当前单次计算 CE＋logits KL。
3. 同一底模、中间块循环、仅 CE。
4. 同一循环结构、CE＋当前教师 logits KL。
5. 第 4 项加入末端特征对齐，或单独加入中间过程监督。

先固定预算比较 K=1、2、4，预算选取只用开发集。固定循环确有收益后再引入停止策略。RELOAD 的预算端点监督与 SanSi/LEV 的每轮监督应作为不同设计，不要混写。

可保留原有监督的目标形式：

`L = E_K[CE_valid-options(y, logits_K) + λ τ² KL(p_teacher_full-vocab^τ || p_student,K_full-vocab^τ)]`

教师仍看完整训练 rationale，学生仅看题目及选项。KL 方向和全词表支持集保持与当前实现一致；若换成选项分布 KL，需单独消融。初始阶段不必同时叠加 Brier、hidden alignment、过程监督与 controller，以免无法解释收益来源。

重点问题包括：CoT 教师是否让普通底模更容易学会有效循环；中间监督是否改善长链泛化；循环数增加时校准是否退化；同等训练和推理计算下能否优于单次计算模型。

必须同时报告准确率、NLL/Brier/ECE、实际延迟和显存，以及长链／分布外表现。用相同数据、多个种子及计算预算对照。额外参数对照可采用非共享加深模型。减少输出 token 不自动意味着比当前无 CoT 答题更快。

单纯“CE＋KL＋循环”、早停或每轮读出均已有很强邻近先例。只有针对教师监督、循环机制和泛化的实证贡献成立后，才适合进一步判断新颖性。

# 共享循环转移头实验

目的：单独检验可学习的循环更新是否改善最终答案；这是对现有中间层循环的改造，不是 GLR 的复现。关闭 CoT 前缀中间监督，保留 final-loop 答案 CE + 全词表 KL。

每次中间块计算并执行原有 damping 后，得到 h。若还有下一轮，执行：

```
h_next = h + W_up SiLU(W_down RMSNorm(h))
```

三轮共调用两次同一个转移头，最后一轮直接读出。转移头秩 128，无偏置；W_up 初始化为零，初始行为与无头循环模型一致。Qwen3-1.7B 的 hidden_size=2048 时新增 526,336 个参数。主模型仍全参数训练、lr=2e-6；新头单独 lr=1e-4，weight_decay 均为 0.01。不是仅训练头。

## 运行

```bash
# 实验；GPU 0,2，batch=4，累积=2，DP=2，有效 batch=16
bash scripts/train_transition_head.sh

# 匹配对照；除转移头外训练参数相同
bash scripts/train_transition_head.sh --transition-rank 0 \
  --output outputs/jev-loop3-damped-final-ce-kl-transition0

# 各自评估循环与不循环
bash scripts/eval_transition_head.sh
bash scripts/eval_transition_head.sh outputs/jev-loop3-damped-final-ce-kl-transition0
```

默认仍为 1024 条、3 epochs、seed 42、max-length=16384。脚本末尾可追加 CLI 参数覆盖默认值。输出目录非空时训练拒绝覆盖。

## 显存与加载

保持 batch=4、累积=2 不变。只监督最后一轮且无过程监督时，跳过前两轮不参与损失的后续层读出；循环中间块仍完整执行三次。转移头也启用非重入梯度检查点。CPU 测试确认该读出优化前后最终 logits 和参数梯度一致。

FP32 参数、梯度和 Adam 两份状态按 16 字节/参数计算，新头合计约 8 MiB（未计分片、激活、通信缓冲及临时张量）；这不是训练总显存估计。长样本、padding 和 FSDP 临时缓冲仍会影响峰值。没有执行 GPU 训练，不能保证 24 GiB 卡上不 OOM。

每个 epoch 日志和训练元数据记录从训练开始累计的 peak_allocated_gib / peak_reserved_gib，取所有 rank 的最大值。统计不包含训练后保存阶段，也不涵盖 PyTorch 分配器以外的全部 GPU 占用。

新头在 FSDP 包装前注册，由根模块管理分片。权重随模型保存；本仓库 Transformers 评估加载器识别 loop_transition_rank 并使用相应模型类。此检查点应使用 `--backend transformers`，普通 AutoModel 加载不会恢复新头。`--disable-looping` 同时绕过循环和转移头，以保留原先“不循环”的比较含义。

## 验证范围

CPU 小模型测试覆盖：零初始化等价性、K-1 次调用、梯度检查点反向传播、两次优化后上下投影梯度、最终读出优化、非零转移头保存重载、基础检查点加载时的初始化，以及关闭循环行为。未执行真实双卡 FSDP 训练或准确率实验；由用户运行上述命令验证。

## 仅关闭转移头的消融

加载同一个已训练检查点，仅跳过循环之间的转移头，保持检查点中的三轮循环和 damping 不变；不修改权重，不需要重新训练：

```bash
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0,2 \
uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-loop3-damped-final-ce-kl-transition128 \
  --input data/gsm-mc/test.jsonl \
  --output outputs/jev-loop3-damped-final-ce-kl-transition128/eval/gsm-mc-test-loop3-no-transition.jsonl \
  --backend transformers --data-parallel-size 2 --batch-size 4 \
  --disable-transition
```

与同一检查点正常三轮循环的结果比较。不要添加 `--disable-looping`；这两个开关互斥。`--disable-transition` 会传递给每个 DP 评估进程，输出记录包含两个禁用开关的值。

## 答案槽转移头：更贴合当前答案槽蒸馏的单变量实验

原版头对全部 token 施加同一个残差 MLP；本版本只对最后的答案槽位置施加该 MLP。假设：只在受答案监督的位置学习循环间更新，可能比同时扰动问题、选项位置更适合当前任务。尚无实验结果支持收益。

这不冻结问题和选项的表示：主模型仍全参数训练，各位置仍经过三次中间块。改变的仅是转移头直接作用的位置。头的秩、参数量、初始化、学习率、CE/KL 和 damping 均不变；仍共享头、每三轮调用两次。主模型 lr=2e-6，头 lr=1e-4；batch=4、累积=2、DP=2，max-length=16384。

```bash
bash scripts/train_answer_slot_transition.sh
bash scripts/eval_transition_head.sh outputs/jev-loop3-damped-final-ce-kl-transition128-answer-slot
```

优先与原有全 token 转移头和无头对照比较循环准确率；不以同一模型的“不循环下降”作为改进证据。

`--transition-scope answer-slot` 保存到 config 和训练元数据，评估自动恢复。未带该字段的旧检查点默认 `all-tokens`。已有 `--disable-transition` 消融开关对两个版本均适用。

实现先切出最后一个位置再计算 MLP，因此头内部归一化和低秩激活的长度从 S 缩小到 1；最终仍需构造完整序列，不能据此保证总显存不溢出。该模式假设输入左 padding、最后位置为答案槽，与本仓库 answer_slot_distill 和 zero_shot 一致；不可直接用于右 padding 下最后位置不是答案槽的调用。

CPU 测试验证非末尾位置与梯度恒等传递、末位实际更新、MLP 输入长度为 1、最终答案损失反向传播及梯度检查点、保存重载与禁用头。未运行 GPU 实验。

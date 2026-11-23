# Final-loop 基线上的线性增量监督实验

用户确认 0.4768 / 0.4768 基线使用 final-loop。此实验沿用提供的 Qwen/Qwen3-1.7B 起点、1024 条 gsm-mc 数据、seed 42、3 epochs、主模型恒定 lr=2e-6、batch=4、累积=2、GPU 0/2、DP=2、TP=1、max-length=16384、temperature=2、CE+KL（权重 1）、循环层 [9,18)、三轮及 damping。不加载之前退化的 MLP 检查点，也不是从最佳训练后检查点继续训练。

## 改动

共享全尺寸、无偏置线性层 W，零初始化，仅输入每轮中间块之前的最后答案槽状态。每轮（包括最后一轮）在原有 damped update 上加 W h；其他位置不直接加线性修正。hidden_size=2048 时为 4,194,304 参数，非 rank-128 MLP。旧参数 transition-rank 在 linear-delta 模式仅作为启用标记，实际矩阵尺寸不由它控制。新头 lr=1e-4。

教师冻结；分别输入问题本身及按完整 CoT 步骤等分的三段前缀，四次顺序 forward，每次读取第 18 层输出的最后答案槽。以相邻前缀状态差为目标，监督学生每轮总变化（damped block update + W h）。保留 final-loop CE/KL，新增权重 0.1 的增量损失；旧的状态余弦监督和隐藏层监督均关闭。

增量损失：每个样本对三轮和隐藏维求 MSE，除以该样本教师增量在所有轮次和维度上的均方值（下限 1e-4），再对非空 rationale 样本平均。教师状态和尺度均 detach；重复前缀的零增量不单独除以零。轨迹预处理沿用完整步骤截断和换行拼接，与没有轨迹辅助损失时的逐 token 截断不同，可能影响教师输入格式；因此也应检查下面的 weight=0 对照。

注意：教师目标均取相同出口层，学生第一轮的起点实际来自循环入口，之后的起点来自上一轮出口。这只是同隐藏维残差空间下的近似监督，不代表学生状态与教师状态在语义上已经对齐；CoT 前缀差也可能包含文字输入差异。该实现不是论文复现。

## 命令

```bash
bash scripts/train_linear_delta.sh
bash scripts/eval_transition_head.sh outputs/jev-loop3-damped-final-ce-kl-linear-delta0.1

# 可选：同一线性结构，仅答案 CE/KL，不加增量监督
bash scripts/train_linear_delta.sh --delta-loss-weight 0 \
  --output outputs/jev-loop3-damped-final-ce-kl-linear-delta0
```

关闭线性修正但保留循环：使用现有评估参数 --disable-transition。对照原始基线可使用 train_transition_head.sh --transition-rank 0，另设输出目录。

## 资源与验证

转移头使用梯度检查点；只保存学生答案槽增量，教师顺序 forward 后仅保留答案槽。教师由一次变为四次 forward，训练会变慢。没有降低 batch 或 max-length，也没有执行 GPU 训练；不保证 24 GiB 显卡不会 OOM。原有训练峰值显存日志保留。

CPU 测试覆盖零初始化与禁用头等价、每轮调用（共三次）、增量损失及其尺度/掩码/教师 detach、最终答案损失和增量损失反向、梯度检查点、非零权重保存重载与禁用恢复。

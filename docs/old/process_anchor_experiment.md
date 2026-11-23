# CoT 前缀过程监督实验

在现有共享层循环上添加训练期过程监督。学生只输入题目、选项及 `Answer: (`；冻结教师额外读取 CoT 的不同前缀。每轮学生经过后部层及最终 norm 后的答案位置状态，对齐相同位置的教师状态。推理结构和输出接口不变。

对于 N 个推理步骤与 K 轮循环，第 k 轮教师读取前 ceil(k*N/K) 个完整步骤。优先按非空行分步，单行时按句子边界回退；不在小数或算式内部切分。步骤不足 K 时允许重复前缀。空 rationale 不参与过程损失；超长 rationale 删除末尾完整步骤。最后一轮复用完整教师前向，不另跑一次。

目标为最终轮答案 CE + 完整教师全词表 KL + 0.1 × 各轮余弦距离的加权平均。`--process-loss-weight` 默认 0，显式启用后才运行前缀教师；`--process-loss-discount` 默认 1，可在 (0,1] 内设置，使第 k 轮权重为 discount^(k-1)。现有 `--hidden-loss-weight` 是独立的多层末端对齐，本实验保持为 0。

运行：

```bash
bash scripts/train_process_anchors.sh
```

相同最终轮答案监督、不加过程损失的对照：

```bash
bash scripts/train_process_anchors.sh \
  --process-loss-weight 0 \
  --output outputs/jev-loop3-damped-final-ce-kl-process0
```

脚本沿用 1024 条、3 epochs、seed 42、GPU 0/2、global batch 16、恒定 lr 2e-6 的配置。新实验输出到 `outputs/jev-loop3-damped-final-ce-kl-process0.1`；不覆盖现有每轮 CE 检查点。训练元数据记录前缀统计、各 epoch 的各项损失、数据校验和和训练用时。

当前只完成单元测试及 32 条样本、2 次优化更新的双卡验证，尚无完整训练或准确率收益结论。日志在 `outputs/process-anchor-logs/smoke-final.log`。试跑仍有 FSDP 成组 embedding/head 的 forward 提示，但已完成反向、优化与保存。已有每轮答案监督检查点不是最终轮监督的匹配对照；比较过程监督收益应运行上面的 process0 对照，采用相同评估集与循环预算。

# 单次前向双读出：GSM-MC 复现实验

2026-11-20。冻结原十轮教师权重，在推理中关闭循环，不使用 latent 槽。字母分支根据题干和选项读出 A/B/C/D logits；数值分支的提示不含选项列表，对四个已知候选逐 token 求全词表对数似然，包含结尾换行。两条分支共享题干，数值候选再共享前缀；用树形因果掩码和路径深度位置编码，让所有分支在同一次模型前向中计算。最终分数是数值对数似然加 `0.75 * 字母 logit`，权重在未用于教师微调的 512 道训练题上选定。

| 方案 | GSM-MC 正式测试准确率 | 单题 P50 / P95 |
| --- | ---: | ---: |
| 同一权重，原生单轮字母读出 | 43.44%（573/1319） | 12.07 / 14.71 ms |
| 同一权重，无循环双读出（复测） | 49.96%（659/1319） | 14.45 / 16.31 ms |
| 原十轮教师 | 49.20%（649/1319） | 47.94 / 58.85 ms |

RTX 3090、BF16、batch 1、CUDA Graph 热态；延迟含输入复制及取回结果，不含分词、树打包、加载或构图。双读出相对同权重单轮提高 6.52 个百分点，P50 增加 2.38 ms。相对十轮教师的 0.76 个百分点是测试集点估计，不宣称显著差异。

数值分支的打分按自回归概率分解，但候选已知，运行时没有逐 token 解码。双读出只继承教师权重，没有使用第十轮隐藏状态、logits 或额外蒸馏；结果只支持给定候选的选择题，不代表保留了开放式推理能力。

```bash
export HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m loop_jev.evaluation.option_trie \
  --model /var/tmp/loop-jev-30007/ce-kl-1024-block10-seed43-step0.1 \
  --modes label-compact-noslots \
  --output outputs/jev-block10-option-trie/test-confirmed.json
```

后续实验：[真实十轮融合、路由和单轮候选读出头](option_loop_cascade_experiment.md)。

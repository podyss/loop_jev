# Progressive explicit-text training for direct answers

Start from Qwen/Qwen3-1.7B, with each student layer executed once. This experiment has no latent-state decoder, transition head, hidden alignment or loop teacher. It tests explicit suffix-generation training alongside the established direct-answer CE + rationale-conditioned teacher KL. It is not a reproduction of a latent-token paper and does not guarantee internalized reasoning.

Every microbatch first trains question/options -> answer with option CE + temperature-2 full-vocabulary KL (weight 1). The frozen original teacher sees the full training rationale. During the first three quarters of optimizer updates, a second student pass learns question/options -> remaining rationale suffix + answer. Removed prefixes are absent from both input and target; earlier tokens of the remaining suffix are teacher-forced, not supplied as user context. The loss weight is 1.0. The final quarter uses only direct-answer CE + KL. No optimizer reset is used.

The four stages use complete reasoning lines/sentences: full, remove floor(N/3) steps, remove floor(2N/3) steps, no text branch. Short rationales can repeat between stages. All 192 optimizer updates fit into the original 3 epochs, with 48 updates per stage (1024 examples, two GPUs, batch4, accumulation2). Additional text passes increase compute, so this is update-matched, not compute-matched, to the historical 0.4768 baseline. The historical baseline also used loop training, so multiple design differences remain.

Text CE averages over tokens within each example, then examples. At most 128 target positions per example are uniformly sampled without replacement to bound vocabulary projection memory; the input suffix itself is not truncated. Sampling changes each batch using a reproducible rank-specific CPU generator. If the complete training trajectory exceeds max-length, preprocessing fails rather than silently cutting reasoning steps. Empty rationale examples still train the answer continuation.

The direct branch is backpropagated before the text forward. Gradient checkpointing, BF16 compute, FP32 master parameters/optimizer states and FSDP2 follow the existing trainer. There is no additional model beyond the existing student and frozen teacher. Full 16384-token inputs may still be memory intensive; no GPU peak measurement was performed during implementation. experiment.json records actual peak allocation and reservation, losses, curriculum transitions, and training duration.

Run:

```bash
bash scripts/train_progressive_text.sh
```

The script runs training followed by Transformers BF16 no-loop, direct-answer GSM-MC test evaluation on GPUs 0,2. Shell `set -e` prevents evaluation after a failed training command. Set `PROGRESSIVE_OUTPUT` to choose another output directory; the same variable is used by both commands. Training refuses a nonempty output directory.

Output: outputs/jev-progressive-text-ce-kl; predictions: eval/gsm-mc-test-no-loop.jsonl. Keep evaluation backend consistent when comparing scores. All auxiliary text generation is training-only.

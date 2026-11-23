# Single-pass depth states with suffix prediction

This is a JEV-specific adaptation motivated by arXiv:2606.20075v1, not a reproduction of its autoregressive latent-token curriculum. It does not replace text tokens with recurrently generated latent states. It tests whether predicting reasoning suffixes from ordinary layer states improves direct-answer inference.

Start from Qwen/Qwen3-1.7B, not a trained loop checkpoint. Student input and answer CE + full-vocabulary KL remain question/options + `Answer: (`. The frozen same-base teacher sees the full training rationale for answer KL. The student executes each layer once (loop-count=1). There is no transition head or hidden-state matching loss.

For Qwen3's 28 layers, collect answer-slot states at layers 9, 19 and 28 (one-indexed). Across three epochs, the auxiliary decoder receives respectively the first one, two and three states. Its text targets are the complete rationale, the suffix after floor(N/3) steps, and the suffix after floor(2N/3) steps. Keep at least one step for short rationales. Splits use the existing semantic step splitter. Teacher forcing exposes only earlier suffix tokens; the first suffix token is predicted at the last latent position. No question or original removed prefix is given to the auxiliary decoder. Suffix tokens are capped at 128; this cap may cut the final step. Each valid example's token CE is averaged before averaging examples; empty rationales do not contribute.

The decoder reuses the frozen answer teacher. Student states are RMS-scaled to its configured initialization scale without a learned projection. This is a modeling assumption to test, not established by the paper. Loss = answer CE + 1.0 * temperature-scaled KL + 0.1 * suffix CE. Gradients pass through the frozen decoder to the student. The optimizer is not reset between stages, unlike the paper's progressive latent training. There is no auxiliary decoder at inference and the standard checkpoint format is preserved.

Memory: auxiliary root forward uses activation recomputation; vocabulary projection is restricted to supervised suffix positions (at most batch * 128). Frozen decoder parameters have no optimizer states. BF16 compute and FSDP2 follow the existing trainer. Student and auxiliary decoder activations still add memory; no two-GPU memory measurement has been performed. Peak allocated/reserved GiB are recorded in experiment.json.

Run `bash scripts/train_depth_suffix.sh`. Batch size per GPU 4, accumulation 2, DP2, effective batch 16, 3 epochs, 1024 examples, LR 2e-6, seed42. User runs training; only CPU tests are run during implementation.

Compare against the historical 0.4768 direct-answer score, while recognizing this also changes training from three loops to one. To isolate the auxiliary loss with a matched single-pass control:

```bash
bash scripts/train_depth_suffix.sh --suffix-loss-weight 0 --output outputs/jev-single-pass-ce-kl-control
```

The training history includes suffix CE, total loss, and memory. Lower suffix loss is not itself evidence of better direct-answer accuracy. The inference goal is the usual no-loop, no-generated-rationale evaluation.

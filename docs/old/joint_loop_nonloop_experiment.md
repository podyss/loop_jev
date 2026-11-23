# Joint loop and direct-answer supervision

The historical comparison is 0.4768 loop / 0.4768 no loop, trained with
Qwen/Qwen3-1.7B, GSM-MC 1024 examples, three epochs, final-loop CE + full
vocabulary rationale-teacher KL (weight 1), loop block [9,18), three damped
passes, learning rate 2e-6, per-GPU batch 4 and accumulation 2.

This run starts again from the same original model. Each microbatch uses the
same teacher pass and two sequential student passes: one with the original
three loops, one with each layer executed once. Both student branches see only
the question, options, and `Answer: (`; both receive the same valid-option CE
and teacher full-vocabulary KL. Objective = 0.5 * (loop CE + loop KL)
+ 0.5 * (direct CE + direct KL). The teacher and both branches use the same
examples; optimizer steps, LR and batch size match the baseline. The extra
student pass increases compute but does not keep both student graphs live.

`--nonloop-loss-weight 0` recovers the existing loop-only loss and skips the
extra forward. The output records loop/direct losses and peak GPU memory.
The script runs loop and no-loop Transformers evaluation after training, using
the same output directory. No full GPU run was performed during implementation.

```bash
bash scripts/train_joint_loop_nonloop.sh
```

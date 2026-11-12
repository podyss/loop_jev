# ai不能动这个文件

# 训练命令

HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=2,3 \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/jev-mathqa/train.jsonl \
  --output outputs/jev-ce-logits-kl-3epochs \
  --epochs 4 \
  --learning-rate 2e-5 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-length 512 \
  --max-train-examples 2048 \
  --temperature 2.0 \
  --logit-loss-weight 1.0 \
  --hidden-loss-weight 0.0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --seed 42


# 评估命令
CUDA_VISIBLE_DEVICES=2,3 \
uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-ce-logits-kl-3epochs \
  --input data/jev-mathqa/test.jsonl \
  --output outputs/jev-ce-logits-kl-3epochs-mathqa-test.jsonl \
  --batch-size 8 \
  --data-parallel-size 2 \
  --dtype bfloat16

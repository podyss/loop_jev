# 不能更改该文件


# CE
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output outputs/jev-ce \
  --epochs 1 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-length 512 \
  --max-train-examples 1024 \
  --logit-loss-weight 0.0 \
  --hidden-loss-weight 0.0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 

# CE＋logits KL
# data/gsm-mc/train.jsonl
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output outputs/jev-ce-logits-kl-1.0 \
  --epochs 1 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-length 16384 \
  --max-train-examples 1024 \
  --temperature 2.0 \
  --logit-loss-weight 1.0 \
  --hidden-loss-weight 0.0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2


# 评估命令
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-ce-logits-ce-1.0 \
  --input data/gsm-mc/test.jsonl \
  --output outputs/jev-ce-logits-ce-1.0-gen/eval/gsm-mc-test.jsonl \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.8


  # 生成
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run loop-jev-generate-correct-trajectories \
  --model Qwen/Qwen3-1.7B \
  --input data/gsm-mc/train.jsonl \
  --output data/gsm-mc/train-self-generated-correct.jsonl \
  --num-generations 4 \
  --target-correct-trajectories 1024 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 8192 \
  --max-model-len 16384 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


# 循环
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output outputs/jev-loop3-damped-ce-logits-kl-1.0 \
  --epochs 3 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-length 16384 \
  --max-train-examples 1024 \
  --temperature 2.0 \
  --logit-loss-weight 1.0 \
  --hidden-loss-weight 0.0 \
  --loop-start-layer 9 \
  --loop-end-layer 18 \
  --loop-count 3 \
  --loop-damping \
  --tensor-parallel-size 1 \
  --data-parallel-size 2

HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-loop3-damped-ce-logits-kl-1.0 \
  --input data/gsm-mc/test.jsonl \
  --output outputs/jev-loop3-damped-ce-logits-kl-1.0/eval/gsm-mc-test-loop3.jsonl \
  --backend transformers \
  --data-parallel-size 2 \
  --disable-looping

# 逐层监督
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/gsm-mc/train.jsonl \
  --output outputs/jev-loop3-damped-per-loop-ce-logits-kl-0.1 \
  --epochs 3 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-length 16384 \
  --max-train-examples 1024 \
  --temperature 2.0 \
  --answer-supervision final-loop \
  --logit-loss-weight 0.1 \
  --hidden-loss-weight 0.0 \
  --loop-start-layer 9 \
  --loop-end-layer 18 \
  --loop-count 3 \
  --loop-damping \
  --tensor-parallel-size 1 \
  --data-parallel-size 2

HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,2 \
uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-loop3-damped-per-loop-ce-logits-kl-0.1 \
  --input data/gsm-mc/test.jsonl \
  --output outputs/jev-loop3-damped-per-loop-ce-logits-kl-0.1/eval/gsm-mc-test-loop3.jsonl \
  --backend transformers \
  --data-parallel-size 2 \
  --disable-looping
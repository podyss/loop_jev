#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1

TRAIN_GPUS="${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
EVAL_GPUS="${EVAL_GPUS:-$TRAIN_GPUS}"
IFS=, read -ra train_devices <<< "$TRAIN_GPUS"
IFS=, read -ra eval_devices <<< "$EVAL_GPUS"
TRAIN_PROCS="${#train_devices[@]}"
EVAL_PROCS="${#eval_devices[@]}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-16}"
if [[ ! "$TRAIN_GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
   (( TRAIN_PROCS == 0 || TRAIN_GLOBAL_BATCH_SIZE % TRAIN_PROCS != 0 )); then
  printf 'TRAIN_GLOBAL_BATCH_SIZE 必须是能被训练 GPU 数整除的正整数（当前 %s 张卡）。\n' "$TRAIN_PROCS" >&2
  exit 2
fi

# 每卡 microbatch 至多为 4，通过累积补足全局 batch。
per_gpu_batch=$((TRAIN_GLOBAL_BATCH_SIZE / TRAIN_PROCS))
TRAIN_BATCH_SIZE=4
while (( per_gpu_batch % TRAIN_BATCH_SIZE != 0 )); do
  TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE - 1))
done
TRAIN_GRAD_ACCUM_STEPS=$((per_gpu_batch / TRAIN_BATCH_SIZE))
printf '训练 GPU %s，全局 batch %s = %s 张卡 × 单卡 batch %s × 累积 %s 步\n' \
  "$TRAIN_GPUS" "$TRAIN_GLOBAL_BATCH_SIZE" "$TRAIN_PROCS" "$TRAIN_BATCH_SIZE" "$TRAIN_GRAD_ACCUM_STEPS"

RUN_ROOT="${RUN_ROOT:-outputs/jev-repro}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/experiment-3}"
LOOP_TEACHER_MODEL="${LOOP_TEACHER_MODEL:-$RUN_ROOT/teacher}"
TRAIN_POOL=data/gsm-mc/train.jsonl
TRAIN="$RUN_ROOT/train-1024.jsonl"
TEST=data/gsm-mc/test.jsonl
BASE_CE_MODEL="$OUTPUT_ROOT/base-ce"
DUAL_MLP_MODEL="$OUTPUT_ROOT/base-ce-dual-adaptive-mix"
DUAL_MLP_KL_MODEL="$OUTPUT_ROOT/base-ce-dual-kl-w0.1-adaptive-mix"

run_training() {
  local checkpoint="$1"
  shift
  printf '训练 %s\n' "$checkpoint"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  uv run torchrun --standalone --nproc_per_node="$TRAIN_PROCS" \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B --train "$TRAIN" --output "$checkpoint" \
    --epochs 3 --learning-rate 2e-6 --batch-size "$TRAIN_BATCH_SIZE" \
    --gradient-accumulation-steps "$TRAIN_GRAD_ACCUM_STEPS" \
    --max-length 16384 --max-train-examples 1024 --seed 42 --train-sample-seed 42 \
    --answer-supervision final-loop --latent-slots 0 \
    --loop-count 1 --no-loop-damping \
    --hidden-loss-weight 0 --process-loss-weight 0 \
    --tensor-parallel-size 1 --data-parallel-size "$TRAIN_PROCS" \
    "$@"
}

run_dual_evaluation() {
  local checkpoint="$1" output="$2"
  shift 2
  printf '评估 %s，GPU %s，%s 个进程\n' "$checkpoint" "$EVAL_GPUS" "$EVAL_PROCS"
  CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
  uv run torchrun --standalone --nproc_per_node="$EVAL_PROCS" \
    --module loop_jev.evaluation.option_trie \
    --model "$checkpoint" --test "$TEST" --modes joint-noslots --output "$checkpoint/eval/$output" \
    "$@"
}

uv run python -m loop_jev.data.shared_train \
  --source "$TRAIN_POOL" --test "$TEST" --output "$TRAIN" --teacher "$LOOP_TEACHER_MODEL"

# 1. 字母答案 CE。评估字母单读出和固定 mix=1.0 的双读出。
run_training "$BASE_CE_MODEL" --logit-loss-weight 0
CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
uv run python -m loop_jev.evaluation.zero_shot \
  --model "$BASE_CE_MODEL" --input "$TEST" \
  --backend transformers --data-parallel-size "$EVAL_PROCS" --batch-size 4 \
  --disable-looping --output "$BASE_CE_MODEL/eval/gsm-mc-letter.jsonl"
run_dual_evaluation "$BASE_CE_MODEL" gsm-mc-dual-fixed-mix1.json --mix 1.0

mix_args=(--dual-readout-mix 1.0 --learn-dual-readout-mix --dual-readout-mix-learning-rate 5e-3)

# 2. 用双读出分数 MLP 预测逐题 mix，以融合答案 CE 训练。
run_training "$DUAL_MLP_MODEL" --logit-loss-weight 0 "${mix_args[@]}"
run_dual_evaluation "$DUAL_MLP_MODEL" gsm-mc-dual-adaptive.json

# 3. 融合答案 CE 加教师 logits KL，权重 0.1，评估使用逐题 mix。
run_training "$DUAL_MLP_KL_MODEL" \
  --logit-loss-type kl --logit-loss-weight 0.1 --temperature 2.0 "${mix_args[@]}"
run_dual_evaluation "$DUAL_MLP_KL_MODEL" gsm-mc-dual-adaptive.json

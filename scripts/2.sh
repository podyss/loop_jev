#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
TRAIN_PROCS=4
EVAL_GPUS=0,1,2,3


# 教师和原始 Qwen 对照的全局 batch = 单卡 batch 4 × 梯度累积步数 × 卡数 = 16。
TRAIN_GRAD_ACCUM_STEPS=$((4 / TRAIN_PROCS))
RUN_ROOT="${RUN_ROOT:-outputs/jev-repro}"
LOOP_TEACHER_MODEL="${LOOP_TEACHER_MODEL:-$RUN_ROOT/teacher}"
BASE_CE_MODEL="$RUN_ROOT/base-ce"
TRAIN_POOL=data/gsm-mc/train.jsonl
TRAIN="$RUN_ROOT/train-1024.jsonl"
TEST=data/gsm-mc/test.jsonl


run_evaluation() {
  local eval_gpus="${EVAL_GPUS:-0,1,2,3}"
  local -a eval_devices
  IFS=, read -ra eval_devices <<< "$eval_gpus"
  printf '评估 %s，GPU %s，%s 个进程\n' "$1" "$eval_gpus" "${#eval_devices[@]}"
  CUDA_VISIBLE_DEVICES="$eval_gpus" \
  uv run torchrun --standalone --nproc_per_node="${#eval_devices[@]}" --module "$@"
}


# 0. 固定原教师的 1024 道题；所有后续训练和校准都读这一份文件。
uv run python -m loop_jev.data.shared_train \
  --source "$TRAIN_POOL" --test "$TEST" --output "$TRAIN" --teacher "$LOOP_TEACHER_MODEL"


# CE baseline
HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run torchrun --standalone --nproc_per_node=4 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train outputs/jev-repro/train-1024.jsonl \
  --output outputs/jev-repro/base-ce \
  --epochs 3 --learning-rate 2e-6 --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --max-length 16384 --max-train-examples 1024 \
  --seed 42 --train-sample-seed 42 \
  --answer-supervision final-loop --latent-slots 0 \
  --loop-count 1 --no-loop-damping \
  --logit-loss-weight 0 --hidden-loss-weight 0 --process-loss-weight 0 \
  --tensor-parallel-size 1 --data-parallel-size 4

HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-repro/base-ce \
  --input data/gsm-mc/test.jsonl \
  --output outputs/jev-repro/base-ce/eval/gsm-mc-test.jsonl \
  --backend transformers --data-parallel-size 4 --batch-size 4 \
  --disable-looping

# 1. 训练十轮共享块教师。
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
uv run torchrun --standalone --nproc_per_node="$TRAIN_PROCS" \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train "$TRAIN" \
  --output "$LOOP_TEACHER_MODEL" \
  --epochs 3 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --gradient-accumulation-steps "$TRAIN_GRAD_ACCUM_STEPS" \
  --max-length 16384 \
  --max-train-examples 1024 \
  --seed 42 \
  --train-sample-seed 42 \
  --temperature 2.0 \
  --answer-supervision final-loop \
  --logit-loss-type kl \
  --logit-loss-weight 1.0 \
  --hidden-loss-weight 0.0 \
  --process-loss-weight 0.0 \
  --latent-slots 24 \
  --loop-schedule block \
  --loop-start-layer 9 \
  --loop-end-layer 18 \
  --loop-count 10 \
  --loop-damping \
  --loop-damping-step 0.1 \
  --tensor-parallel-size 1 \
  --data-parallel-size "$TRAIN_PROCS"



# 2. 全参微调共享块，比较单跳 CE、直接末态、分段监督、路径一致性。
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
uv run torchrun --standalone --nproc_per_node="$TRAIN_PROCS" \
  --module loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" \
  --output "$RUN_ROOT/shortcut-full" \
  --train-split fixed \
  --examples 1024 --epochs 4 --full-block --lr 2e-6 \
  --modes ce endpoint segments shortcut


# 2a. 从原始 Qwen 全参训练单轮纯 CE 对照，使用同一份 1024 题。
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
uv run torchrun --standalone --nproc_per_node="$TRAIN_PROCS" \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B --train "$TRAIN" --output "$BASE_CE_MODEL" \
  --epochs 3 --learning-rate 2e-6 --batch-size 4 \
  --gradient-accumulation-steps "$TRAIN_GRAD_ACCUM_STEPS" \
  --max-length 16384 --max-train-examples 1024 --seed 42 --train-sample-seed 42 \
  --answer-supervision final-loop --latent-slots 0 \
  --loop-count 1 --no-loop-damping \
  --logit-loss-weight 0 --hidden-loss-weight 0 --process-loss-weight 0 \
  --tensor-parallel-size 1 --data-parallel-size "$TRAIN_PROCS"


# 3. 全量评估捷径蒸馏、原生单轮、原十轮及从原始 Qwen 训练的 CE 对照。
run_evaluation loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --train-split fixed \
  --test "$TEST" \
  --output "$RUN_ROOT/shortcut-full" \
  --evaluate --full-block --modes ce endpoint segments shortcut \
  --base-ce-model "$BASE_CE_MODEL"


# 4. 全参微调共享块，做十轮 -> 两跳 -> 单跳；同批题跑 CE 和直接蒸馏对照。
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
uv run torchrun --standalone --nproc_per_node="$TRAIN_PROCS" \
  --module loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" \
  --output "$RUN_ROOT/progressive-full" \
  --train-split fixed \
  --examples 1024 --epochs 4 --full-block --lr 2e-6 \
  --modes ce endpoint rollout progressive


# 5. 全量评估渐进蒸馏：原生单轮、原十轮、直接蒸馏、两跳教师和最终单跳。
run_evaluation loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --train-split fixed \
  --test "$TEST" \
  --output "$RUN_ROOT/progressive-full" \
  --evaluate --full-block --modes ce endpoint rollout progressive

# ----------
# 6. 冻结教师跑无循环双读出。
run_evaluation loop_jev.evaluation.option_trie \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --test "$TEST" \
  --modes label-compact-noslots \
  --mix 1.0 \
  --output "$RUN_ROOT/dual-readout/test.json"


# 7. 同一批题训练两个候选小头：答案 CE、答案 CE + 十轮残差。固定 12 epoch。
run_evaluation loop_jev.evaluation.option_loop_residual \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --calibration "$RUN_ROOT/dual-readout/test.json" \
  --test "$TEST" \
  --output "$RUN_ROOT/candidate-head/test.json"


# 8. 重载纯答案小头，全量复测带头和无头的双读出。
run_evaluation loop_jev.evaluation.option_loop_residual \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --calibration "$RUN_ROOT/dual-readout/test.json" \
  --test "$TEST" \
  --checkpoint "$RUN_ROOT/candidate-head/test-answer-only.pt" \
  --output "$RUN_ROOT/candidate-head/test-confirmed.json"


# 9. 候选小头输入消融：每组重训，并评估全部 1319 题。
run_evaluation loop_jev.evaluation.option_loop_residual \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --calibration "$RUN_ROOT/dual-readout/test.json" \
  --test "$TEST" \
  --ablate \
  --output "$RUN_ROOT/candidate-head/test-feature-ablation.json"


# 补充：旧版单轮 CE、关环教师 KL、十轮教师 KL。以下三个命令使用历史固定输出目录，按序手动运行。
# LOOP_TEACHER_MODEL="$LOOP_TEACHER_MODEL" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
# bash scripts/run_block10_student_distill.sh


# 补充：旧版状态差分蒸馏。需要上一段的 loop10-kd 测试结果。
# LOOP_TEACHER_MODEL="$LOOP_TEACHER_MODEL" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
# bash scripts/run_block10_state_distill.sh


# 补充：旧版绝对末态蒸馏。需要上一段的 state-delta-kd 测试结果。
# LOOP_TEACHER_MODEL="$LOOP_TEACHER_MODEL" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
# bash scripts/run_block10_state_distill.sh --absolute-state


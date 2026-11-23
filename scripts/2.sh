#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
TRAIN_GPUS="${TRAIN_GPUS:-0,2}"
EVAL_GPU="${EVAL_GPU:-0}"
RUN_ROOT="${RUN_ROOT:-outputs/jev-repro}"
LOOP_TEACHER_MODEL="${LOOP_TEACHER_MODEL:-$RUN_ROOT/teacher}"
TRAIN_POOL=data/gsm-mc/train.jsonl
TRAIN="$RUN_ROOT/train-1024.jsonl"
TEST=data/gsm-mc/test.jsonl


# 0. 固定原教师的 1024 道题；所有后续训练和校准都读这一份文件。
uv run python -m loop_jev.data.shared_train \
  --source "$TRAIN_POOL" --test "$TEST" --output "$TRAIN" --teacher "$LOOP_TEACHER_MODEL"


# 1. 训练十轮共享块教师。已有完整且同数据的检查点时跳过。
if [[ ! -s "$LOOP_TEACHER_MODEL/experiment.json" || ! -s "$LOOP_TEACHER_MODEL/config.json" ]]; then
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  uv run torchrun --standalone --nproc_per_node=2 \
    --module loop_jev.training.answer_slot_distill \
    --model Qwen/Qwen3-1.7B \
    --train "$TRAIN" \
    --output "$LOOP_TEACHER_MODEL" \
    --epochs 3 \
    --learning-rate 2e-6 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --max-length 16384 \
    --max-train-examples 1024 \
    --seed 43 \
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
    --data-parallel-size 2
fi


# 2. 在训练集 A 比较单跳 CE、直接末态、分段监督、路径一致性。
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" \
  --output "$RUN_ROOT/shortcut" \
  --train-split fixed \
  --examples 1024 --epochs 4 --rank 16 --lr 4e-4 \
  --modes ce endpoint segments shortcut


# 3. 全量评估捷径蒸馏，同时测同权重原生单轮和原十轮基线。
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
uv run python -m loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --train-split fixed \
  --test "$TEST" \
  --output "$RUN_ROOT/shortcut" \
  --evaluate --modes ce endpoint segments shortcut


# 4. 同一批题做十轮 -> 两跳 -> 单跳；同批题跑 CE 和直接蒸馏对照。
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" \
  --output "$RUN_ROOT/progressive" \
  --train-split fixed \
  --examples 1024 --epochs 4 --rank 16 --lr 4e-4 \
  --modes ce endpoint rollout progressive


# 5. 全量评估渐进蒸馏：原生单轮、原十轮、直接蒸馏、两跳教师和最终单跳。
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
uv run python -m loop_jev.evaluation.block_shortcut \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --train-split fixed \
  --test "$TEST" \
  --output "$RUN_ROOT/progressive" \
  --evaluate --modes ce endpoint rollout progressive


# 6. 冻结教师跑无循环双读出；只用同一份 1024 道训练题校准融合。
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
uv run python -m loop_jev.evaluation.option_trie \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --test "$TEST" \
  --modes label-compact-noslots \
  --output "$RUN_ROOT/dual-readout/test.json"


# 7. 同一批题训练两个候选小头：答案 CE、答案 CE + 十轮残差。固定 12 epoch。
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
uv run python -m loop_jev.evaluation.option_loop_residual \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --calibration "$RUN_ROOT/dual-readout/test.json" \
  --test "$TEST" \
  --output "$RUN_ROOT/candidate-head/test.json"


# 8. 重载纯答案小头，同卡全量复测带头和无头的双读出。
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
uv run python -m loop_jev.evaluation.option_loop_residual \
  --model "$LOOP_TEACHER_MODEL" \
  --train "$TRAIN" --shared-train \
  --calibration "$RUN_ROOT/dual-readout/test.json" \
  --test "$TEST" \
  --checkpoint "$RUN_ROOT/candidate-head/test-answer-only.pt" \
  --output "$RUN_ROOT/candidate-head/test-confirmed.json"


# 9. 候选小头输入消融：每组重训，并评估全部 1319 题。
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
uv run python -m loop_jev.evaluation.option_loop_residual \
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


# 原 2.sh 的 CE 命令，作为独立实验保留，不属于上面的教师训练流程。
# HF_ENDPOINT=https://hf-mirror.com \
# CUDA_VISIBLE_DEVICES=0,2 \
# uv run torchrun --standalone --nproc_per_node=2 \
#   --module loop_jev.training.answer_slot_distill \
#   --model Qwen/Qwen3-1.7B \
#   --train data/gsm-mc/train.jsonl \
#   --output outputs/jev-ce \
#   --epochs 1 \
#   --learning-rate 2e-6 \
#   --batch-size 4 \
#   --gradient-accumulation-steps 2 \
#   --max-length 512 \
#   --max-train-examples 1024 \
#   --logit-loss-weight 0.0 \
#   --hidden-loss-weight 0.0 \
#   --tensor-parallel-size 1 \
#   --data-parallel-size 2



# 原 2.sh 的评估命令，指向另一个已有检查点，不接上一段的 jev-ce 输出。
# HF_ENDPOINT=https://hf-mirror.com \
# CUDA_VISIBLE_DEVICES=0,2 \
# uv run python -m loop_jev.evaluation.zero_shot \
#   --model outputs/jev-ce-logits-ce-1.0 \
#   --input data/gsm-mc/test.jsonl \
#   --output outputs/jev-ce-logits-ce-1.0-gen/eval/gsm-mc-test.jsonl \
#   --data-parallel-size 2 \
#   --gpu-memory-utilization 0.8


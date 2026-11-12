# Loop-JEV Research

This is a standalone research workspace for the current JEV baseline and follow-up latent/looped reasoning experiments. It contains answer-slot training, full-trajectory and progressive-pruning baselines, direct-answer evaluation, dataset preparation, the prepared JEV datasets, and the JEV/loop-reasoning research notes.

The current primary method, **Ours (CE + logits KL)**, trains a student on the question and choices while a frozen same-base teacher also sees the training rationale. The student receives cross-entropy over valid answer labels and temperature-scaled KL over the full vocabulary at the answer slot. The current implementation does not add latent tokens or recurrent inference; this repository is organized to make those follow-up comparisons self-contained.

## Repository layout

- `src/loop_jev/data/`: dataset normalization and preparation.
- `src/loop_jev/training/`: answer-slot distillation, progressive answer-token training, and full-trajectory SFT.
- `src/loop_jev/evaluation/`: direct answer-option evaluation.
- `src/loop_jev/experiments/`: matched ablation and constant-learning-rate experiment orchestration.
- `src/loop_jev/`: shared package namespace.
- `scripts/1.sh`: single command dispatcher for all project commands.
- `data/`: prepared MathQA and CompMath-MCQ files and their provenance.
- `docs/`: JEV pruning proposal and looped latent-reasoning literature survey.
- `results/experiments/`: archived experiment metadata, without model weights.
- `results/predictions/`: archived evaluation predictions used by existing runs.
- `environment/`: verified software/hardware versions and setup steps.

Probe-intervention code and probe-specific datasets are intentionally excluded.

## Setup

See [`environment/README.md`](environment/README.md). The training and data scripts expect the repository root as the working directory. Hugging Face downloads should use the configured mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

After `uv sync`, categorized implementations are available through the installed `loop-jev-*` commands. Run them through `uv run`, for example `uv run loop-jev-train-answer-slot --help` or `uv run loop-jev-evaluate --help`.

## Main training: CE + logits KL

The following is a small two-GPU example for Qwen3-1.7B using FSDP2 data parallelism. The product of `--tensor-parallel-size` and `--data-parallel-size` must match `torchrun --nproc_per_node`.

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  --module loop_jev.training.answer_slot_distill \
  --model Qwen/Qwen3-1.7B \
  --train data/jev-mathqa/train.jsonl \
  --output outputs/jev-ce-logits-kl \
  --epochs 4 --learning-rate 2e-5 \
  --batch-size 4 --gradient-accumulation-steps 4 \
  --max-length 512 --max-train-examples 2048 \
  --temperature 2.0 --logit-loss-weight 1.0 \
  --hidden-loss-weight 0.0 --tensor-parallel-size 1 \
  --data-parallel-size 2 --seed 42
```

The trainer refuses to write into a nonempty output directory. To evaluate answer-option probabilities without generating a rationale:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m loop_jev.evaluation.zero_shot \
  --model outputs/jev-ce-logits-kl \
  --input data/jev-mathqa/test.jsonl \
  --output outputs/jev-ce-logits-kl-mathqa-test.jsonl \
  --batch-size 8 --data-parallel-size 2 --dtype bfloat16
```

For a fast CLI/data check without loading a model:

```bash
uv run loop-jev-train-answer-slot --help
uv run loop-jev-evaluate --help
```

## Existing experiments

The archived metadata and prediction files are copied into `results/`. Full checkpoints were left in the source workspace's `outputs/` directories because they occupy many tens of gigabytes. Their original directory names and experiment settings are retained in `results/experiments/`; the model weights are not required to run new experiments.

The included experiment predictions are historical artifacts. They do not establish a general result for looped JEV, and comparisons should account for differences in model size, data, optimizer schedule, and training budget. Add new runs under `outputs/` and record model revision, data checksums, seed, software versions, and compute budget alongside each result.

## Data and licensing

See [`data/README.md`](data/README.md) and the dataset manifest before redistribution. Model weights are fetched separately from Hugging Face or supplied via a local model path.

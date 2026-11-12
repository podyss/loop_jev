#!/usr/bin/env python3
"""Distill full-trace answer-slot logits and hidden states into answer-only inputs."""

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def remove_answer_disclosure(rationale):
    rationale = rationale.strip().strip('"')
    match = re.search(
        r"\b(?:answer\s*:|the\s+answer\s+is|answer\s+is|option\s+[a-e]\b)",
        rationale,
        re.IGNORECASE,
    )
    if match:
        rationale = rationale[:match.start()].rstrip(" .,:;\n")
    return rationale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jev-answer-slot-distill"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--logit-loss-weight", type=float, default=1.0)
    parser.add_argument("--hidden-loss-weight", type=float, default=0.1)
    parser.add_argument("--hidden-layer-selection", choices=("quarter", "twelve", "all"), default="twelve",
                        help="Align four, twelve, or every transformer layer")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="FSDP2 data-parallel size; TP size times DP size must equal torchrun world size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.logit_loss_weight < 0 or args.hidden_loss_weight < 0:
        parser.error("Distillation weights must be nonnegative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.tensor_parallel_size < 1 or args.data_parallel_size < 1:
        parser.error("Parallelism sizes must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Full-parameter training requires torchrun with multiple GPUs")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {args.output}")

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig
    from tqdm.auto import tqdm

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.tensor_parallel_size * args.data_parallel_size:
        raise ValueError(
            f"torchrun world size ({world_size}) must equal tensor-parallel size "
            f"({args.tensor_parallel_size}) x data-parallel size ({args.data_parallel_size})"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidate_ids = []
    for label in label_chars:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer label {label!r} is not one token: {ids}")
        candidate_ids.append(ids[0])

    class AnswerSlotDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.truncated = 0

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            options = "\n".join(
                f"({label_chars[i]}) {option}" for i, option in enumerate(row["options"])
            )
            user_content = f"Question: {row['question']}\nOptions:\n{options}"
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False,
            )
            rationale_ids = tokenizer.encode(
                remove_answer_disclosure(row.get("rationale", "")), add_special_tokens=False,
            )

            student_ids = prompt_ids + tokenizer.encode("Answer: (", add_special_tokens=False)

            def make_teacher_prefix(reason_ids):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                continuation = f"Reasoning: {reason}\nAnswer: (" if reason else "Answer: ("
                return prompt_ids + tokenizer.encode(continuation, add_special_tokens=False)

            teacher_ids = make_teacher_prefix(rationale_ids)
            was_truncated = len(teacher_ids) > args.max_length
            while len(teacher_ids) > args.max_length and rationale_ids:
                overflow = len(teacher_ids) - args.max_length
                del rationale_ids[max(0, len(rationale_ids) - max(overflow, 1)):]
                teacher_ids = make_teacher_prefix(rationale_ids)
            if len(student_ids) > args.max_length or len(teacher_ids) > args.max_length:
                raise ValueError(f"Question and options exceed max length for {row['id']}")
            self.truncated += int(was_truncated)
            return {
                "student_input_ids": student_ids,
                "teacher_input_ids": teacher_ids,
                "target": row["gold_index"],
                "option_count": len(row["options"]),
            }

    def collate(examples):
        batch = {}
        for name in ("student_input_ids", "teacher_input_ids"):
            width = max(len(item[name]) for item in examples)
            ids, masks = [], []
            for item in examples:
                padding = width - len(item[name])
                ids.append([tokenizer.pad_token_id] * padding + item[name])
                masks.append([0] * padding + [1] * len(item[name]))
            prefix = name.removesuffix("_input_ids")
            batch[f"{prefix}_input_ids"] = torch.tensor(ids, dtype=torch.long)
            batch[f"{prefix}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        batch["target"] = torch.tensor([item["target"] for item in examples], dtype=torch.long)
        batch["option_count"] = torch.tensor([item["option_count"] for item in examples], dtype=torch.long)
        return batch

    rows = read_rows(args.train)
    if args.max_train_examples and len(rows) > args.max_train_examples:
        selection = random.Random(args.seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[index] for index in selection]
    dataset = AnswerSlotDataset(rows)
    encoded = [dataset[index] for index in range(len(dataset))]
    truncated_examples = dataset.truncated

    distributed_config = DistributedConfig(
        tp_size=args.tensor_parallel_size,
        fsdp_size=args.data_parallel_size,
    )

    def capture_answer_hidden_states(model, layers):
        captures = {}
        handles = []
        decoder_layers = [
            module for module in model.modules()
            if "DecoderLayer" in module.__class__.__name__
        ]
        for layer_index in layers:
            def save_answer_position(_module, _inputs, output, index=layer_index - 1):
                hidden_state = output[0] if isinstance(output, (tuple, list)) else output
                captures[index + 1] = hidden_state[:, -1, :].clone()

            handles.append(decoder_layers[layer_index - 1].register_forward_hook(save_answer_position))
        return captures, handles

    def clip_tensor_parallel_grad_norm(parameters, max_norm):
        squared_norm = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float64)
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if isinstance(gradient, DTensor):
                if any(isinstance(placement, Shard) for placement in gradient.placements):
                    squared_norm += gradient.to_local().double().square().sum()
                elif all(isinstance(placement, Replicate) for placement in gradient.placements):
                    if rank == 0:
                        squared_norm += gradient.to_local().double().square().sum()
                else:
                    raise RuntimeError(f"Unsupported TP gradient placements: {gradient.placements}")
            elif rank == 0:
                squared_norm += gradient.double().square().sum()
        dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
        total_norm = squared_norm.sqrt()
        clip_factor = min(max_norm / (total_norm.item() + 1e-6), 1.0)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_factor)
        return total_norm

    use_teacher = args.logit_loss_weight > 0 or args.hidden_loss_weight > 0
    use_hidden = args.hidden_loss_weight > 0
    hidden_layer_count = None
    hidden_layers = None
    teacher = None
    if use_teacher:
        teacher_base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
            distributed_config=distributed_config,
        )
        teacher_base.config.use_cache = False
        teacher_base.eval()
        hidden_layer_count = teacher_base.config.num_hidden_layers
        hidden_layers = (
            list(range(1, hidden_layer_count + 1))
            if args.hidden_layer_selection == "all"
            else sorted({
                round(hidden_layer_count * fraction / 12)
                for fraction in range(1, 13)
            }) if args.hidden_layer_selection == "twelve"
            else sorted({
                max(1, round(hidden_layer_count * fraction))
                for fraction in (0.25, 0.5, 0.75, 1.0)
            })
        )
        if use_hidden:
            teacher_hidden_capture, teacher_hook_handles = capture_answer_hidden_states(
                teacher_base, hidden_layers,
            )
        teacher = teacher_base
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    student_base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False,
        distributed_config=distributed_config,
    )
    student_base.config.use_cache = False
    student_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden_layer_count = student_base.config.num_hidden_layers
    if hidden_layers is None:
        hidden_layers = (
            list(range(1, hidden_layer_count + 1))
            if args.hidden_layer_selection == "all"
            else sorted({
                round(hidden_layer_count * fraction / 12)
                for fraction in range(1, 13)
            }) if args.hidden_layer_selection == "twelve"
            else sorted({
                max(1, round(hidden_layer_count * fraction))
                for fraction in (0.25, 0.5, 0.75, 1.0)
            })
        )
    if use_hidden:
        student_hidden_capture, student_hook_handles = capture_answer_hidden_states(
            student_base, hidden_layers,
        )
    student = student_base

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=0.01, foreach=False,
    )
    samples_per_rank = len(encoded) // args.data_parallel_size
    batches_per_rank = math.ceil(samples_per_rank / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_rank / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    update_step = 0
    completed_epochs = 0
    optimizer.zero_grad(set_to_none=True)
    temperature = args.temperature

    class EncodedDataset(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    for epoch in range(args.epochs):
        sampler = DistributedSampler(
            EncodedDataset(), num_replicas=args.data_parallel_size,
            rank=rank // args.tensor_parallel_size,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            EncodedDataset(), batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
        )
        epoch_sums = torch.zeros(4, device=local_rank)
        progress = tqdm(
            loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch",
            dynamic_ncols=True, disable=rank != 0,
        )
        for batch_index, batch in enumerate(progress):
            targets = batch.pop("target").to(local_rank)
            option_counts = batch.pop("option_count").to(local_rank)
            batch = {key: value.to(local_rank) for key, value in batch.items()}
            teacher_batch = {
                "input_ids": batch.pop("teacher_input_ids"),
                "attention_mask": batch.pop("teacher_attention_mask"),
            }
            student_batch = {
                "input_ids": batch.pop("student_input_ids"),
                "attention_mask": batch.pop("student_attention_mask"),
            }
            accumulation_remainder = len(loader) % args.gradient_accumulation_steps
            group_size = (
                accumulation_remainder
                if accumulation_remainder and batch_index >= len(loader) - accumulation_remainder
                else args.gradient_accumulation_steps
            )
            sync_now = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            teacher_logits = None
            teacher_hidden = []
            if use_teacher:
                if use_hidden:
                    teacher_hidden_capture.clear()
                with torch.no_grad():
                    teacher_outputs = teacher(
                        **teacher_batch, logits_to_keep=1, output_hidden_states=False,
                    )
                    if args.logit_loss_weight > 0:
                        teacher_logits = teacher_outputs.logits[:, -1, :].float()
                    if use_hidden:
                        teacher_hidden = [
                            teacher_hidden_capture[layer].float()
                            for layer in hidden_layers
                        ]
                    del teacher_outputs

            if use_hidden:
                student_hidden_capture.clear()
            student_outputs = student(
                **student_batch, logits_to_keep=1, output_hidden_states=False,
            )
            student_logits = student_outputs.logits[:, -1, :].float()
            option_logits = student_logits[:, candidate_ids]
            valid_options = (
                torch.arange(len(label_chars), device=local_rank)[None, :]
                < option_counts[:, None]
            )
            option_logits = option_logits.masked_fill(
                ~valid_options, torch.finfo(option_logits.dtype).min,
            )
            answer_loss = F.cross_entropy(option_logits, targets)
            logit_loss = answer_loss.new_zeros(())
            if args.logit_loss_weight > 0:
                logit_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)
            hidden_losses = []
            for layer, target_hidden in zip(hidden_layers, teacher_hidden):
                student_hidden = student_hidden_capture[layer].float()
                hidden_losses.append(
                    1.0 - F.cosine_similarity(student_hidden, target_hidden, dim=-1).mean()
                )
            hidden_loss = (
                torch.stack(hidden_losses).mean() if hidden_losses else answer_loss.new_zeros(())
            )
            loss = (
                answer_loss + args.logit_loss_weight * logit_loss
                + args.hidden_loss_weight * hidden_loss
            )
            scaled_loss = loss / group_size
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
            scaled_loss.backward()
            epoch_sums += torch.stack([
                answer_loss.detach(), logit_loss.detach(), hidden_loss.detach(), loss.detach(),
            ])
            del student_outputs, teacher_logits, teacher_hidden

            if sync_now:
                update_step += 1
                grad_norm = clip_tensor_parallel_grad_norm(student.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at update {update_step}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    progress.set_postfix(
                        step=f"{update_step}/{total_steps}",
                        ce=f"{answer_loss.item():.5f}",
                        kl=f"{logit_loss.item():.5f}",
                        hidden=f"{hidden_loss.item():.5f}",
                        total=f"{loss.item():.5f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.8g}",
                    )

        progress.close()
        dist.all_reduce(epoch_sums, op=dist.ReduceOp.SUM)
        mean_losses = (epoch_sums / (len(loader) * world_size)).tolist()
        completed_epochs = epoch + 1
        if rank == 0:
            print(
                f"epoch={epoch + 1} mean_ce={mean_losses[0]:.6f} "
                f"mean_kl={mean_losses[1]:.6f} mean_hidden={mean_losses[2]:.6f} "
                f"mean_total={mean_losses[3]:.6f}", flush=True,
            )
    args.output.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(
        args.output, safe_serialization=True, max_shard_size="5GB",
    )
    if rank == 0:
        tokenizer.save_pretrained(args.output)
        metadata = {
            "model": args.model,
            "teacher": args.model if use_teacher else None,
            "train": str(args.train),
            "epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "optimizer_steps": update_step,
            "lr_schedule_steps": total_steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": "constant",
            "final_learning_rate": optimizer.param_groups[0]["lr"],
            "world_size": world_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "global_batch_size": (
                args.batch_size * args.gradient_accumulation_steps * args.data_parallel_size
            ),
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "examples": len(rows),
            "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "max_train_examples": args.max_train_examples,
            "max_length": args.max_length,
            "student_parameter_dtype": "float32",
            "teacher_parameter_dtype": "bfloat16" if use_teacher else None,
            "logit_distillation": "temperature-scaled KL over the full vocabulary at the next-answer-token position",
            "temperature": temperature,
            "logit_loss_weight": args.logit_loss_weight,
            "answer_loss": "cross-entropy over valid answer-letter logits",
            "student_input": "question and options only, followed by Answer: (",
            "teacher_input": "question, options, and full training rationale, followed by Answer: (",
            "hidden_distillation": "cross-context cosine alignment at the next-answer-token position for evenly spaced transformer layers",
            "hidden_layers_1_indexed": hidden_layers,
            "hidden_layer_selection": args.hidden_layer_selection,
            "hidden_loss_weight": args.hidden_loss_weight,
            "truncated_examples": truncated_examples,
            "max_grad_norm": 1.0,
            "status": "complete",
        }
        (args.output / "experiment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

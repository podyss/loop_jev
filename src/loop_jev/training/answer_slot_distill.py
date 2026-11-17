#!/usr/bin/env python3
"""Distill full-trace answer-slot logits and hidden states into answer-only inputs."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

from loop_jev.training.process_anchors import (
    delta_alignment_loss, make_process_prefixes, process_alignment_loss, split_reasoning_steps,
)


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
    parser = argparse.ArgumentParser(
        epilog="Precision defaults: FP32 master parameters/Adam states, BF16 compute, "
               "FP32 gradient reduction and losses; BF16 frozen teacher.",
    )
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
    parser.add_argument("--logit-loss-type", choices=("kl", "ce"), default="kl",
                        help="Use KL divergence or soft-target cross-entropy for full-vocabulary logits")
    parser.add_argument("--logit-loss-weight", type=float, default=1.0)
    parser.add_argument("--process-loss-weight", type=float, default=0.0,
                        help="Loop-wise cosine alignment to frozen teacher CoT-prefix readouts")
    parser.add_argument("--process-loss-discount", type=float, default=1.0,
                        help="Geometric discount across loop anchors, in (0, 1]")
    parser.add_argument("--hidden-loss-weight", type=float, default=0.1)
    parser.add_argument("--hidden-layer-selection", choices=("quarter", "twelve", "all"), default="twelve",
                        help="Align four, twelve, or every transformer layer")
    parser.add_argument("--loop-start-layer", type=int, default=9,
                        help="First zero-based Qwen3 decoder layer in the recurrent block")
    parser.add_argument("--loop-end-layer", type=int, default=19,
                        help="Exclusive end of the recurrent block")
    parser.add_argument("--loop-count", type=int, default=3,
                        help="Number of shared-parameter passes through the recurrent block")
    parser.add_argument("--loop-damping", action=argparse.BooleanOptionalAction, default=True,
                        help="Interpolate each recurrent update by 1 / loop count")
    parser.add_argument("--transition-rank", type=int, default=0,
                        help="Shared residual transition bottleneck width; 0 disables it")
    parser.add_argument("--transition-type", choices=("mlp", "linear-delta"), default="mlp")
    parser.add_argument("--delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--transition-scope", choices=("all-tokens", "answer-slot"),
                        default="all-tokens", help="Apply the head to every token or only the final answer slot")
    parser.add_argument("--transition-learning-rate", type=float, default=1e-4,
                        help="Learning rate for the newly initialized transition head")
    parser.add_argument("--answer-supervision", choices=("every-loop", "final-loop"),
                        default="every-loop",
                        help="Apply answer CE and teacher-logit KL at every loop readout or only the final one")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="FSDP2 data-parallel size; TP size times DP size must equal torchrun world size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if min(args.logit_loss_weight, args.hidden_loss_weight, args.process_loss_weight) < 0:
        parser.error("Distillation weights must be nonnegative")
    if not 0 < args.process_loss_discount <= 1:
        parser.error("--process-loss-discount must be in (0, 1]")
    if args.loop_count < 1:
        parser.error("--loop-count must be positive")
    if args.delta_loss_weight < 0:
        parser.error("--delta-loss-weight must be nonnegative")
    if args.delta_loss_weight > 0 and (args.transition_type != "linear-delta" or args.process_loss_weight > 0):
        parser.error("Delta supervision requires linear-delta and process-loss-weight=0")
    if args.transition_type == "linear-delta" and (args.transition_rank <= 0 or args.transition_scope != "answer-slot"):
        parser.error("linear-delta requires transition-rank>0 (enable flag) and answer-slot scope")
    if args.transition_rank < 0 or args.transition_learning_rate <= 0:
        parser.error("Transition rank must be nonnegative and its learning rate positive")
    if args.transition_rank and (args.loop_count < 2 or args.tensor_parallel_size != 1):
        parser.error("Transition training requires at least two loops and TP size 1")
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
    from torch.distributed.tensor import DTensor
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig
    from tqdm.auto import tqdm

    @torch.no_grad()
    def clip_grad_norm_mixed(parameters, max_norm):
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not grads:
            return torch.zeros((), device=local_rank)

        # DTensor reductions account for TP shards; ordinary parameters are TP-replicated.
        norm_type = 2.0
        total = 0.0
        for grad in grads:
            grad_norm = torch.linalg.vector_norm(grad.float(), ord=norm_type)
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()
            total += float(grad_norm.item()) ** norm_type
        total_norm = total ** (1.0 / norm_type)
        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
        if clip_coef < 1.0:
            for grad in grads:
                grad.mul_(clip_coef)
        return torch.tensor(total_norm, device=local_rank)

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

    use_delta = args.delta_loss_weight > 0
    use_process = args.process_loss_weight > 0 or use_delta

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
            rationale = remove_answer_disclosure(row.get("rationale", ""))
            rationale_ids = tokenizer.encode(rationale, add_special_tokens=False)

            student_ids = prompt_ids + tokenizer.encode("Answer: (", add_special_tokens=False)

            def make_teacher_prefix(reason_ids):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                continuation = f"Reasoning: {reason}\nAnswer: (" if reason else "Answer: ("
                return prompt_ids + tokenizer.encode(continuation, add_special_tokens=False)

            # Process targets use complete semantic steps, including when capped.
            steps = split_reasoning_steps(rationale) if use_process else []
            process_truncated = False
            if use_process:
                while steps:
                    rationale_ids = tokenizer.encode("\n".join(steps), add_special_tokens=False)
                    if len(make_teacher_prefix(rationale_ids)) <= args.max_length:
                        break
                    steps.pop()
                    process_truncated = True
                if not steps:
                    rationale_ids = []
            teacher_ids = make_teacher_prefix(rationale_ids)
            was_truncated = process_truncated or len(teacher_ids) > args.max_length
            while len(teacher_ids) > args.max_length and rationale_ids:
                overflow = len(teacher_ids) - args.max_length
                del rationale_ids[max(0, len(rationale_ids) - max(overflow, 1)):]
                teacher_ids = make_teacher_prefix(rationale_ids)
            if len(student_ids) > args.max_length or len(teacher_ids) > args.max_length:
                raise ValueError(f"Question and options exceed max length for {row['id']}")
            self.truncated += int(was_truncated)
            item = {
                "student_input_ids": student_ids,
                "teacher_input_ids": teacher_ids,
                "target": row["gold_index"],
                "option_count": len(row["options"]),
            }
            if use_process:
                prefixes = make_process_prefixes(steps, args.loop_count)
                # The final prefix is exactly the full teacher input and is reused.
                for k, prefix in enumerate(prefixes[:-1]):
                    item[f"process_{k}_input_ids"] = make_teacher_prefix(
                        tokenizer.encode(prefix, add_special_tokens=False),
                    )
                item["process_valid"] = bool(steps)
                item["process_step_count"] = len(steps)
            return item

    def collate(examples):
        batch = {}
        names = ["student_input_ids", "teacher_input_ids"]
        if use_process:
            names += [f"process_{k}_input_ids" for k in range(args.loop_count - 1)]
        for name in names:
            width = max(len(item[name]) for item in examples)
            ids, masks = [], []
            for item in examples:
                padding = width - len(item[name])
                ids.append([tokenizer.pad_token_id] * padding + item[name])
                masks.append([0] * padding + [1] * len(item[name]))
            prefix = name.removesuffix("_input_ids")
            batch[f"{prefix}_input_ids"] = torch.tensor(ids, dtype=torch.long)
            batch[f"{prefix}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        if use_process:
            batch["process_valid"] = torch.tensor([item["process_valid"] for item in examples])
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
    process_statistics = None
    if use_process:
        step_counts = [item["process_step_count"] for item in encoded]
        process_statistics = {
            "empty_rationales": sum(n == 0 for n in step_counts),
            "fewer_steps_than_loops": sum(0 < n < args.loop_count for n in step_counts),
            "mean_steps": sum(step_counts) / max(1, len(step_counts)),
            "max_steps": max(step_counts, default=0),
        }
        if not any(step_counts):
            raise ValueError("Process supervision requires at least one nonempty rationale")
        if rank == 0:
            print(json.dumps({"process_anchors": process_statistics,
                              "truncated_examples": truncated_examples}), flush=True)

    distributed_config = DistributedConfig(
        tp_size=args.tensor_parallel_size,
        fsdp_size=args.data_parallel_size,
        # Match verl's FSDP2 policy: BF16 compute with FP32 gradient reduction.
        fsdp_mixed_precision=True,
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

    use_teacher = args.logit_loss_weight > 0 or args.hidden_loss_weight > 0 or use_process
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
        teacher_process_capture = {}
        if use_process:
            def capture_teacher_readout(_module, _inputs, output):
                teacher_process_capture["state"] = output[:, -1, :].detach().clone()

            teacher_process_module = (teacher_base.model.layers[args.loop_end_layer - 1]
                                      if use_delta else teacher_base.model.norm)
            teacher_process_hook = teacher_process_module.register_forward_hook(capture_teacher_readout)
        teacher = teacher_base
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    from transformers import AutoConfig
    from loop_jev.transition import load_loop_model
    student_config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    if getattr(student_config, "loop_transition_rank", 0) != 0:
        raise ValueError("Start this experiment from a base checkpoint without a transition head")
    student_config.loop_transition_rank = args.transition_rank
    student_config.loop_transition_scope = args.transition_scope
    student_config.loop_transition_type = args.transition_type
    student_base = load_loop_model(
        args.model, config=student_config, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False,
        distributed_config=distributed_config,
    )
    from loop_jev.looping import enable_looping
    enable_looping(
        student_base, args.loop_start_layer, args.loop_end_layer, args.loop_count,
        damping=args.loop_damping,
        collect_all_readouts=args.answer_supervision == "every-loop" or args.process_loss_weight > 0,
        collect_updates=use_delta,
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
    # from_pretrained() starts in eval mode; checkpointing requires training mode.
    student.train()

    transition_parameters = [p for name, p in student.named_parameters()
                             if name.startswith("loop_transition.")]
    base_parameters = [p for name, p in student.named_parameters()
                       if not name.startswith("loop_transition.")]
    parameter_groups = [{"params": base_parameters, "lr": args.learning_rate}]
    if transition_parameters:
        parameter_groups.append({"params": transition_parameters,
                                 "lr": args.transition_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01, foreach=False)
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

    training_history = []
    torch.cuda.reset_peak_memory_stats()
    training_started = time.monotonic()
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
        epoch_sums = torch.zeros(6, device=local_rank, dtype=torch.float32)
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
            teacher_process_states = []
            if use_teacher:
                if use_hidden:
                    teacher_hidden_capture.clear()
                with torch.no_grad():
                    if use_delta:
                        initial_outputs = teacher(**student_batch, logits_to_keep=1, output_hidden_states=False)
                        teacher_process_states.append(teacher_process_capture.pop("state"))
                        del initial_outputs
                    if use_process:
                        # Call the root model so FSDP unshards root parameters.
                        # Prefix passes are separate: earlier anchors cannot see
                        # later reasoning. Only answer-slot states are retained.
                        for k in range(args.loop_count - 1):
                            prefix_outputs = teacher(
                                input_ids=batch[f"process_{k}_input_ids"],
                                attention_mask=batch[f"process_{k}_attention_mask"],
                                logits_to_keep=1, output_hidden_states=False,
                            )
                            teacher_process_states.append(teacher_process_capture.pop("state"))
                            del prefix_outputs
                    teacher_outputs = teacher(
                        **teacher_batch, logits_to_keep=1, output_hidden_states=False,
                    )
                    if use_process:
                        teacher_process_states.append(teacher_process_capture.pop("state"))
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_outputs = student(
                    **student_batch, logits_to_keep=1, output_hidden_states=False,
                )
            valid_options = (
                torch.arange(len(label_chars), device=local_rank)[None, :]
                < option_counts[:, None]
            )
            answer_losses = []
            logit_losses = []
            teacher_probs = (
                F.softmax(teacher_logits / temperature, dim=-1)
                if args.logit_loss_weight > 0 else None
            )
            loop_readouts = student_base.model.loop_hidden_states
            if args.answer_supervision == "final-loop":
                loop_readouts = loop_readouts[-1:]
            for loop_hidden in loop_readouts:
                # Reuse the root forward's final logits. Re-entering a tied
                # FSDP LM head alone skips its paired embedding forward.
                student_logits = (
                    student_outputs.logits[:, -1, :].float()
                    if args.answer_supervision == "final-loop"
                    else student_base.lm_head(loop_hidden[:, -1, :]).float()
                )
                option_logits = student_logits[:, candidate_ids]
                option_logits = option_logits.masked_fill(
                    ~valid_options, torch.finfo(option_logits.dtype).min,
                )
                answer_losses.append(F.cross_entropy(option_logits, targets))
                if teacher_probs is not None:
                    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                    if args.logit_loss_type == "kl":
                        loop_logit_loss = F.kl_div(
                            student_log_probs, teacher_probs, reduction="batchmean",
                        )
                    else:
                        loop_logit_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
                    logit_losses.append(loop_logit_loss * (temperature ** 2))
            answer_loss = torch.stack(answer_losses).mean()
            logit_loss = (
                torch.stack(logit_losses).mean()
                if logit_losses else answer_loss.new_zeros(())
            )
            hidden_losses = []
            for layer, target_hidden in zip(hidden_layers, teacher_hidden):
                student_hidden = student_hidden_capture[layer].float()
                hidden_losses.append(
                    1.0 - F.cosine_similarity(student_hidden, target_hidden, dim=-1).mean()
                )
            hidden_loss = (
                torch.stack(hidden_losses).mean() if hidden_losses else answer_loss.new_zeros(())
            )
            process_loss = (
                process_alignment_loss(
                    [state[:, -1, :] for state in student_base.model.loop_hidden_states],
                    teacher_process_states, batch["process_valid"], args.process_loss_discount,
                ) if args.process_loss_weight > 0 else answer_loss.new_zeros(())
            )
            delta_loss = (delta_alignment_loss(student_base.model.loop_updates,
                          teacher_process_states, batch["process_valid"])
                          if use_delta else answer_loss.new_zeros(()))
            loss = (
                answer_loss + args.logit_loss_weight * logit_loss
                + args.hidden_loss_weight * hidden_loss
                + args.process_loss_weight * process_loss
                + args.delta_loss_weight * delta_loss
            )
            scaled_loss = loss / group_size
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
            scaled_loss.backward()
            student_base.model.loop_hidden_states = []
            student_base.model.loop_updates = []
            epoch_sums += torch.stack([
                answer_loss.detach(), logit_loss.detach(), hidden_loss.detach(), loss.detach(),
                process_loss.detach(), delta_loss.detach(),
            ])
            del student_outputs, teacher_logits, teacher_hidden, teacher_process_states

            if sync_now:
                update_step += 1
                grad_norm = clip_grad_norm_mixed(student.parameters(), 1.0)
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
                        process=f"{process_loss.item():.5f}",
                        delta=f"{delta_loss.item():.5f}",
                        total=f"{loss.item():.5f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.8g}",
                    )

        progress.close()
        dist.all_reduce(epoch_sums, op=dist.ReduceOp.SUM)
        mean_losses = (epoch_sums / (len(loader) * world_size)).tolist()
        completed_epochs = epoch + 1
        peak_memory = torch.tensor([
            torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.max_memory_reserved() / 2**30,
        ], device=local_rank, dtype=torch.float64)
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        training_history.append(dict(zip(
            ("ce", "kl", "hidden", "total", "process", "delta"), mean_losses,
        ), epoch=completed_epochs,
            peak_allocated_gib=peak_memory[0].item(), peak_reserved_gib=peak_memory[1].item()))
        if rank == 0:
            print(
                f"epoch={epoch + 1} mean_ce={mean_losses[0]:.6f} "
                f"mean_kl={mean_losses[1]:.6f} mean_hidden={mean_losses[2]:.6f} "
                f"mean_total={mean_losses[3]:.6f} mean_process={mean_losses[4]:.6f} "
                f"mean_delta={mean_losses[5]:.6f} "
                f"peak_allocated_gib={peak_memory[0].item():.3f} "
                f"peak_reserved_gib={peak_memory[1].item():.3f}", flush=True,
            )
    training_seconds = time.monotonic() - training_started
    args.output.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(
        args.output, safe_serialization=True, max_shard_size="5GB",
    )
    if rank == 0:
        tokenizer.save_pretrained(args.output)
        metadata = {
            "model": args.model,
            "model_revision": getattr(student_base.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "training_seconds": training_seconds,
            "training_history": training_history,
            "loop_start_layer": args.loop_start_layer,
            "loop_end_layer": args.loop_end_layer,
            "loop_count": args.loop_count,
            "loop_parameter_sharing": True,
            "loop_damping": args.loop_damping,
            "transition_type": args.transition_type,
            "delta_loss_weight": args.delta_loss_weight,
            "delta_target": "consecutive teacher prefix answer slots at loop_end_layer output" if use_delta else None,
            "transition_rank": args.transition_rank if args.transition_type == "mlp" else None,
            "transition_scope": args.transition_scope if args.transition_rank else None,
            "transition_learning_rate": args.transition_learning_rate if args.transition_rank else None,
            "transition_placement": ("inside every update including final; W applied to pre-block answer slot"
                if args.transition_type == "linear-delta" else
                "between loops, after damped update; shared; zero-initialized output")
                if args.transition_rank else None,
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
            "student_compute_dtype": "bfloat16",
            "gradient_reduce_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "loss_dtype": "float32",
            "fsdp_mixed_precision": args.data_parallel_size > 1,
            "teacher_parameter_dtype": "bfloat16" if use_teacher else None,
            "logit_distillation": (
                f"temperature-scaled {args.logit_loss_type.upper()} over the full vocabulary "
                + ("at the final loop readout's next-answer-token position"
                   if args.answer_supervision == "final-loop" else
                   "at every loop readout's next-answer-token position, averaged across loops")
            ),
            "logit_loss_type": args.logit_loss_type,
            "temperature": temperature,
            "logit_loss_weight": args.logit_loss_weight,
            "answer_loss": (
                "cross-entropy over valid answer-letter logits at the final loop readout"
                if args.answer_supervision == "final-loop"
                else "mean cross-entropy over valid answer-letter logits at every loop readout"
            ),
            "answer_supervision": args.answer_supervision,
            "student_input": "question and options only, followed by Answer: (",
            "teacher_input": "question, options, and full training rationale, followed by Answer: (",
            "hidden_distillation": "cross-context cosine alignment at the next-answer-token position for evenly spaced transformer layers",
            "hidden_layers_1_indexed": hidden_layers,
            "hidden_layer_selection": args.hidden_layer_selection,
            "hidden_loss_weight": args.hidden_loss_weight,
            "process_loss_weight": args.process_loss_weight,
            "process_loss_discount": args.process_loss_discount,
            "process_distillation": (
                "cosine alignment of each loop's final-normalized answer-slot state to "
                "a frozen teacher reading ceil(k * N / K) complete rationale steps; "
                "line boundaries with sentence fallback; final anchor reuses full teacher; "
                "empty rationales masked; weighted mean over loops and valid examples"
                if use_process else None
            ),
            "process_statistics": process_statistics,
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

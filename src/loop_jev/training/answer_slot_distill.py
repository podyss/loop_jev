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

from loop_jev.training.loop_cot import add_workspace_readout, round_targets, token_loss
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
    parser.add_argument("--loop-teacher-model", default=None,
                        help="Use a trained looped checkpoint as teacher and keep the student non-looped")
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
                        help="Loop-wise cosine alignment to teacher CoT states, or loop-teacher states when configured")
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
    parser.add_argument("--loop-carry-workspace", action="store_true",
                        help="Carry only the latent workspace between recurrent rounds")
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
    parser.add_argument("--suffix-loss-weight", type=float, default=0.0,
                        help="Single-pass layer-state suffix prediction with a frozen training-only decoder")
    parser.add_argument("--suffix-max-tokens", type=int, default=128)
    parser.add_argument("--progressive-text-weight", type=float, default=0.0,
                        help="Explicit suffix LM loss with four update-based stages; direct CE+KL always active")
    parser.add_argument("--nonloop-loss-weight", type=float, default=0.0,
                        help="Mix a single-pass CE+KL branch into final-loop CE+KL; 0 recovers baseline")
    parser.add_argument("--latent-slots", type=int, default=0,
                        help="Number of question-only latent workspace positions before the answer slot")
    parser.add_argument("--kv-loss-weight", type=float, default=0.0,
                        help="Align final-loop workspace K/V to offline rationale targets")
    parser.add_argument("--kv-layer", type=int, default=17,
                        help="Zero-based decoder layer for compressed rationale KV supervision")
    parser.add_argument("--kv-cache", type=Path, default=None,
                        help="Offline targets from loop_jev.training.latent_kv")
    parser.add_argument("--loop-cot-weight", type=float, default=0.0)
    parser.add_argument("--loop-cot-order", choices=("ordered", "shuffled"), default="ordered")
    parser.add_argument("--loop-cot-tokens", type=int, default=8,
                        help="Maximum supervised CoT tokens per round, read from the last latent slots")
    parser.add_argument("--workspace-query-weight", type=float, default=0.0,
                        help="Weight for training-only readout of intermediate quantities from workspace slots")
    parser.add_argument("--text-max-targets", type=int, default=128,
                        help="Maximum sampled LM target positions per example; does not truncate input text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    use_kv = args.kv_loss_weight > 0
    use_loop_cot = args.loop_cot_weight > 0
    use_query = args.workspace_query_weight > 0
    if not math.isfinite(args.workspace_query_weight) or args.workspace_query_weight < 0:
        parser.error("Workspace query weight must be finite and nonnegative")
    if use_query and (not args.latent_slots or args.answer_supervision != "final-loop"
                      or args.tensor_parallel_size != 1 or use_kv or use_loop_cot
                      or args.loop_carry_workspace):
        parser.error("Workspace query requires latent slots, final-loop, TP=1 and no other workspace mode")
    if args.loop_carry_workspace and (args.latent_slots < 1 or args.loop_count < 2
                                     or args.loop_teacher_model or args.transition_rank):
        parser.error("Workspace carry requires latent slots, multiple rounds, and no loop teacher or transition")
    if not math.isfinite(args.loop_cot_weight) or args.loop_cot_weight < 0:
        parser.error("CoT weight must be finite and nonnegative")
    if use_loop_cot and (not 0 < args.loop_cot_tokens <= args.latent_slots or args.loop_count != 3
                         or args.answer_supervision != "final-loop" or use_kv
                         or args.tensor_parallel_size != 1):
        parser.error("Loop CoT needs usable latent slots, three loops, final-loop, TP=1 and no KV loss")
    if args.latent_slots < 0 or args.kv_loss_weight < 0:
        parser.error("Latent slots and KV loss weight must be nonnegative")
    if use_kv and (args.latent_slots < 1 or args.kv_cache is None or args.answer_supervision != "final-loop"
                   or args.loop_teacher_model or args.process_loss_weight or args.hidden_loss_weight
                   or args.delta_loss_weight or args.transition_rank or args.tensor_parallel_size != 1):
        parser.error("KV supervision requires slots, cache, final-loop, TP=1 and no other state losses or transition")
    if args.kv_cache is not None and not use_kv:
        parser.error("--kv-cache requires --kv-loss-weight > 0")
    if args.latent_slots and (args.loop_teacher_model or args.suffix_loss_weight
                              or args.progressive_text_weight or args.nonloop_loss_weight
                              or args.process_loss_weight or args.delta_loss_weight
                              or args.transition_rank or args.hidden_loss_weight):
        parser.error("Latent slots support direct CE/KL with optional KV or loop CoT supervision")
    use_nonloop = args.nonloop_loss_weight > 0
    if not 0 <= args.nonloop_loss_weight < 1:
        parser.error("--nonloop-loss-weight must be in [0, 1)")
    if use_nonloop and (args.loop_count < 2 or args.answer_supervision != "final-loop"
                        or args.loop_teacher_model or args.transition_rank
                        or args.process_loss_weight or args.delta_loss_weight
                        or args.hidden_loss_weight or args.suffix_loss_weight
                        or args.progressive_text_weight or args.tensor_parallel_size != 1):
        parser.error("Nonloop branch requires loop-count>=2, final-loop, TP=1 and no other auxiliary modes")
    use_text = args.progressive_text_weight > 0
    if args.progressive_text_weight < 0 or args.text_max_targets < 1:
        parser.error("Text weight must be nonnegative and max targets positive")
    if use_text and (args.loop_count != 1 or args.answer_supervision != "final-loop"
                     or args.loop_teacher_model or args.transition_rank or args.process_loss_weight
                     or args.delta_loss_weight or args.hidden_loss_weight or args.suffix_loss_weight
                     or args.tensor_parallel_size != 1):
        parser.error("Progressive text requires loop-count=1, final-loop, TP=1 and no other auxiliary mode")
    if min(args.batch_size, args.gradient_accumulation_steps, args.max_length) < 1:
        parser.error("Batch size, accumulation and max length must be positive")
    use_suffix = args.suffix_loss_weight > 0
    if args.suffix_loss_weight < 0 or args.suffix_max_tokens < 1:
        parser.error("Suffix weight must be nonnegative and token cap positive")
    if use_suffix and (args.loop_count != 1 or args.answer_supervision != "final-loop"
                       or args.loop_teacher_model or args.transition_rank
                       or args.process_loss_weight or args.delta_loss_weight or args.hidden_loss_weight):
        parser.error("Suffix prediction requires loop-count=1, final-loop, no loop teacher, transition or hidden/process/delta loss")
    if use_suffix and args.tensor_parallel_size != 1:
        parser.error("Suffix prediction currently supports TP=1 only")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if min(args.logit_loss_weight, args.hidden_loss_weight, args.process_loss_weight) < 0:
        parser.error("Distillation weights must be nonnegative")
    if not 0 < args.process_loss_discount <= 1:
        parser.error("--process-loss-discount must be in (0, 1]")
    if args.loop_count < 1:
        parser.error("--loop-count must be positive")
    if args.loop_teacher_model and args.transition_rank:
        parser.error("Loop-teacher distillation trains a plain non-looped student; transition heads are disabled")
    if args.loop_teacher_model and args.answer_supervision != "final-loop":
        parser.error("Loop-teacher distillation requires --answer-supervision final-loop")
    if args.loop_teacher_model and args.delta_loss_weight > 0:
        parser.error("Loop-teacher distillation uses recurrent-state alignment, not teacher-delta targets")
    if args.loop_teacher_model and args.loop_count < 2:
        parser.error("Loop-teacher distillation requires at least two teacher loops")
    if args.loop_teacher_model and not any((args.logit_loss_weight, args.hidden_loss_weight,
                                            args.process_loss_weight)):
        parser.error("Loop-teacher distillation needs a logit, hidden, or process loss")
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

    use_loop_teacher = args.loop_teacher_model is not None
    use_delta = args.delta_loss_weight > 0
    use_process = not use_loop_teacher and (args.process_loss_weight > 0 or use_delta)

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

            if args.latent_slots:
                from loop_jev.training.latent_kv import append_workspace
                student_ids = append_workspace(prompt_ids, tokenizer, args.latent_slots)
            else:
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
            teacher_ids = student_ids if use_loop_teacher else make_teacher_prefix(rationale_ids)
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
            if use_kv:
                item["id"] = row["id"]
            if use_loop_cot:
                item["loop_cot_targets"] = round_targets(
                    rationale_ids, args.loop_cot_tokens, order=args.loop_cot_order,
                    row_id=row["id"], seed=args.seed,
                )
            if use_query:
                from loop_jev.training.workspace_query import encode_queries
                item["query_ids"], item["query_targets"] = encode_queries(tokenizer, rationale)
            if use_text:
                from loop_jev.training.progressive_text import encode_trajectories
                item["trajectories"] = encode_trajectories(
                    tokenizer, prompt_ids, rationale, label_chars[row["gold_index"]], args.max_length,
                )
            if use_suffix:
                from loop_jev.training.suffix_prediction import suffix_texts
                item["suffixes"] = [tokenizer.encode(text, add_special_tokens=False)[:args.suffix_max_tokens]
                                    for text in suffix_texts(split_reasoning_steps(rationale))]
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
        if use_text:
            batch["trajectories"] = [item["trajectories"] for item in examples]
        if use_suffix:
            batch["suffixes"] = [item["suffixes"][suffix_stage] for item in examples]
        if use_kv:
            batch["kv_ids"] = [item["id"] for item in examples]
        if use_loop_cot:
            batch["loop_cot_targets"] = torch.tensor([item["loop_cot_targets"] for item in examples])
        if use_query:
            for key in ("query_ids", "query_targets"):
                batch[key] = torch.tensor([item[key] for item in examples], dtype=torch.long)
        batch["target"] = torch.tensor([item["target"] for item in examples], dtype=torch.long)
        batch["option_count"] = torch.tensor([item["option_count"] for item in examples], dtype=torch.long)
        return batch

    rows = read_rows(args.train)
    if args.max_train_examples and len(rows) > args.max_train_examples:
        selection = random.Random(args.seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[index] for index in selection]
    dataset = AnswerSlotDataset(rows)
    encoded = [dataset[index] for index in range(len(dataset))]
    kv_targets = None
    if use_kv:
        from loop_jev.training.latent_kv import load_cache
        kv_targets = load_cache(
            args.kv_cache, train=args.train, model=args.model, slots=args.latent_slots,
            layer=args.kv_layer, max_length=args.max_length, seed=args.seed,
            selected_ids=[item["id"] for item in encoded],
        )
    truncated_examples = dataset.truncated
    query_coverage = None
    if use_query:
        query_coverage = sum(any(any(ids) for ids in item["query_ids"]) for item in encoded)
        if not query_coverage:
            raise ValueError("Workspace query supervision requires annotated intermediate quantities")
        if rank == 0:
            print(f"workspace_query_coverage={query_coverage}/{len(encoded)}", flush=True)
    if use_loop_cot and not any(token >= 0 for item in encoded
                                for segment in item["loop_cot_targets"] for token in segment):
        raise ValueError("Loop CoT supervision requires nonempty rationale targets")
    if use_suffix and not any(item["suffixes"][0] for item in encoded):
        raise ValueError("Suffix prediction requires nonempty training rationales")
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

    use_teacher = (args.logit_loss_weight > 0 or args.hidden_loss_weight > 0
                   or use_process or use_loop_teacher or use_suffix)
    use_hidden = args.hidden_loss_weight > 0
    hidden_layer_count = None
    hidden_layers = None
    teacher = None
    teacher_loop_states = []
    teacher_loop_hook = None
    if use_teacher:
        teacher_path = args.loop_teacher_model if use_loop_teacher else args.model
        teacher_base = AutoModelForCausalLM.from_pretrained(
            teacher_path, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
            distributed_config=distributed_config,
        )
        teacher_base.config.use_cache = False
        teacher_base.eval()
        if use_loop_teacher:
            if getattr(teacher_base.config, "loop_transition_rank", 0) != 0:
                raise ValueError("Loop teacher checkpoint must not contain a transition head")
            from loop_jev.looping import enable_looping
            enable_looping(
                teacher_base, args.loop_start_layer, args.loop_end_layer, args.loop_count,
                damping=args.loop_damping, collect_all_readouts=False,
            )
            if args.process_loss_weight > 0:
                def capture_loop_state(_module, inputs, output):
                    previous = inputs[0][:, -1, :]
                    block_output = output[0] if isinstance(output, (tuple, list)) else output
                    block_output = block_output[:, -1, :]
                    state = (previous + (block_output - previous) / args.loop_count
                             if args.loop_damping else block_output)
                    teacher_loop_states.append(state.detach().clone())

                teacher_loop_hook = teacher_base.model.layers[
                    args.loop_end_layer - 1
                ].register_forward_hook(capture_loop_state)
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
    if use_loop_teacher and (
        teacher_base.config.num_hidden_layers != student_base.config.num_hidden_layers
        or teacher_base.config.hidden_size != student_base.config.hidden_size
    ):
        raise ValueError("Loop teacher and student must have matching depth and hidden size")
    if not use_loop_teacher:
        from loop_jev.looping import enable_looping
        workspace_slice = None
        if use_loop_cot:
            workspace_slice, _cot_hook = add_workspace_readout(
                student_base, args.loop_cot_tokens,
                len(tokenizer.encode("Answer: (", add_special_tokens=False)),
            )
        if use_query:
            suffix = len(tokenizer.encode("Answer: (", add_special_tokens=False))
            workspace_slice = slice(-args.latent_slots - suffix, -suffix)
        enable_looping(
            student_base, args.loop_start_layer, args.loop_end_layer, args.loop_count,
            damping=args.loop_damping,
            collect_all_readouts=args.answer_supervision == "every-loop" or args.process_loss_weight > 0,
            collect_updates=use_delta,
            workspace_slice=workspace_slice,
            carry_workspace=(args.latent_slots, len(tokenizer.encode("Answer: (", add_special_tokens=False)))
            if args.loop_carry_workspace else None,
        )
    student_base.config.use_cache = False
    student_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden_layer_count = student_base.config.num_hidden_layers
    if args.latent_slots:
        from loop_jev.training.latent_kv import marker_token_id
        student_base.config.latent_slots = args.latent_slots
        student_base.config.latent_marker_token_id = marker_token_id(tokenizer)
    if use_kv and not 0 <= args.kv_layer < hidden_layer_count:
        raise ValueError("kv-layer is outside the student decoder")
    kv_capture, kv_hooks = {}, []
    if use_kv:
        from loop_jev.training.latent_kv import capture_workspace_kv
        kv_capture, kv_hooks = capture_workspace_kv(
            student_base, args.kv_layer, args.latent_slots,
            len(tokenizer.encode("Answer: (", add_special_tokens=False)),
        )
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
    student_loop_capture = {}
    student_loop_hooks = []
    student_loop_layers = []
    if use_loop_teacher and args.process_loss_weight > 0:
        first = args.loop_end_layer - 1
        remaining = hidden_layer_count - 1 - first
        if remaining < args.loop_count - 1:
            raise ValueError("Not enough student layers after the first loop boundary to map all teacher steps")
        student_loop_layers = [
            first + round(k * remaining / (args.loop_count - 1))
            for k in range(args.loop_count)
        ]
        for layer_index in student_loop_layers:
            def capture_student_step(_module, _inputs, output, index=layer_index):
                hidden_state = output[0] if isinstance(output, (tuple, list)) else output
                student_loop_capture[index] = hidden_state[:, -1, :].clone()

            student_loop_hooks.append(
                student_base.model.layers[layer_index].register_forward_hook(capture_student_step)
            )
    suffix_capture, suffix_hooks = {}, []
    suffix_layers = []
    if use_suffix:
        suffix_layers = sorted({round(hidden_layer_count * f) for f in (1/3, 2/3, 1)})
        if len(suffix_layers) != 3:
            raise ValueError("Suffix curriculum requires at least three student layers")
        suffix_capture, suffix_hooks = capture_answer_hidden_states(student_base, suffix_layers)
    student = student_base
    # from_pretrained() starts in eval mode; checkpointing requires training mode.
    student.train()

    transition_parameters = [p for name, p in student.named_parameters()
                             if name.startswith("loop_transition.")]
    base_parameters = [p for name, p in student.named_parameters()
                       if not name.startswith("loop_transition.")]
    parameter_groups = [{"params": base_parameters, "lr": args.learning_rate}]
    if use_query:
        from torch.nn.parallel import DistributedDataParallel
        from loop_jev.training.workspace_query import WorkspaceQueryReadout, query_loss
        query_readout = DistributedDataParallel(
            WorkspaceQueryReadout(student_base.config.hidden_size).to(local_rank), device_ids=[local_rank],
        )
        parameter_groups.append({"params": list(query_readout.parameters()), "lr": 3e-4})
    if transition_parameters:
        parameter_groups.append({"params": transition_parameters,
                                 "lr": args.transition_learning_rate})
    parameters_for_clipping = [parameter for group in parameter_groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01, foreach=False)
    samples_per_rank = len(encoded) // args.data_parallel_size
    batches_per_rank = math.ceil(samples_per_rank / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_rank / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    if total_steps < (4 if use_text else 1):
        raise ValueError("Not enough optimizer updates for the selected training mode")
    update_step = 0
    text_stage_history = []
    previous_text_stage = None
    text_generator = torch.Generator().manual_seed(args.seed + 1000 + rank)
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
        suffix_stage = min(2, epoch * 3 // args.epochs)
        if use_suffix and rank == 0:
            print(f"suffix_stage={suffix_stage + 1} latent_layers={suffix_layers[:suffix_stage + 1]}", flush=True)
        sampler = DistributedSampler(
            EncodedDataset(), num_replicas=args.data_parallel_size,
            rank=rank // args.tensor_parallel_size,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            EncodedDataset(), batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
        )
        epoch_nonloop_sums = torch.zeros(2, device=local_rank, dtype=torch.float32)
        epoch_text_sum = torch.zeros((), device=local_rank)
        epoch_suffix_sum = torch.zeros((), device=local_rank)
        epoch_kv_sum = torch.zeros((), device=local_rank)
        epoch_cot_sum = torch.zeros((), device=local_rank)
        if use_query:
            epoch_query_sum = torch.zeros((), device=local_rank)
            epoch_query_stats = torch.zeros(args.loop_count + 2, dtype=torch.long, device=local_rank)
        epoch_sums = torch.zeros(6, device=local_rank, dtype=torch.float32)
        progress = tqdm(
            loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch",
            dynamic_ncols=True, disable=rank != 0,
        )
        for batch_index, batch in enumerate(progress):
            targets = batch.pop("target").to(local_rank)
            option_counts = batch.pop("option_count").to(local_rank)
            trajectories = batch.pop("trajectories", None)
            text_stage = None
            if use_text:
                from loop_jev.training.progressive_text import curriculum_stage
                text_stage = curriculum_stage(update_step, total_steps)
                if text_stage != previous_text_stage:
                    text_stage_history.append({"stage": text_stage, "first_update": update_step + 1})
                    if rank == 0:
                        print(f"progressive_text_stage={text_stage} first_update={update_step + 1}", flush=True)
                    previous_text_stage = text_stage
            suffix_sequences = batch.pop("suffixes", None)
            kv_ids = batch.pop("kv_ids", None)
            batch = {key: value.to(local_rank) for key, value in batch.items()}
            cot_targets = batch.pop("loop_cot_targets", None)
            if use_query:
                query_ids = batch.pop("query_ids")
                query_targets = batch.pop("query_targets")
            teacher_batch = {
                "input_ids": batch.pop("teacher_input_ids"),
                "attention_mask": batch.pop("teacher_attention_mask"),
            }
            student_batch = {
                "input_ids": batch.pop("student_input_ids"),
                "attention_mask": batch.pop("student_attention_mask"),
            }
            if use_loop_teacher:
                teacher_batch = student_batch
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
                    if use_loop_teacher and args.process_loss_weight > 0:
                        teacher_loop_states.clear()
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
                    if use_loop_teacher and args.process_loss_weight > 0:
                        teacher_process_states = list(teacher_loop_states)
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

            suffix_capture.clear()
            kv_capture.clear()
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
            loop_readouts = ([None] if use_loop_teacher
                             else student_base.model.loop_hidden_states)
            if args.answer_supervision == "final-loop" and not use_loop_teacher:
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
                    ([student_loop_capture[index] for index in student_loop_layers]
                     if use_loop_teacher else
                     [state[:, -1, :] for state in student_base.model.loop_hidden_states]),
                    teacher_process_states,
                    (torch.ones(targets.shape[0], dtype=torch.bool, device=local_rank)
                     if use_loop_teacher else batch["process_valid"]),
                    args.process_loss_discount,
                ) if args.process_loss_weight > 0 else answer_loss.new_zeros(())
            )
            delta_loss = (delta_alignment_loss(student_base.model.loop_updates,
                          teacher_process_states, batch["process_valid"])
                          if use_delta else answer_loss.new_zeros(()))
            suffix_loss = answer_loss.new_zeros(())
            kv_loss = answer_loss.new_zeros(())
            cot_loss = answer_loss.new_zeros(())
            workspace_query_loss = answer_loss.new_zeros(())
            if use_query:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    query_logits, masked_logits = query_readout(
                        torch.stack(student_base.model.loop_workspace_states), query_ids,
                    )
                workspace_query_loss, query_stats = query_loss(query_logits, masked_logits, query_targets)
                epoch_query_sum += workspace_query_loss.detach()
                epoch_query_stats += query_stats
            if use_loop_cot:
                from torch.utils.checkpoint import checkpoint
                cot_loss = checkpoint(token_loss, student_outputs.logits[:, :-1, :],
                                      cot_targets, use_reentrant=False)
            if use_kv:
                from loop_jev.training.latent_kv import kv_alignment_loss
                target_keys = torch.stack([kv_targets[row_id][0] for row_id in kv_ids]).to(local_rank)
                target_values = torch.stack([kv_targets[row_id][1] for row_id in kv_ids]).to(local_rank)
                kv_loss = kv_alignment_loss(
                    kv_capture["key"], kv_capture["value"], target_keys, target_values,
                )
                del target_keys, target_values
            if use_suffix:
                from loop_jev.training.suffix_prediction import suffix_prediction_loss
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    suffix_loss = suffix_prediction_loss(
                        teacher, [suffix_capture[layer] for layer in suffix_layers[:suffix_stage + 1]],
                        suffix_sequences, tokenizer.pad_token_id,
                        embedding_scale=teacher.config.initializer_range,
                    )
            loss = (
                args.kv_loss_weight * kv_loss + args.suffix_loss_weight * suffix_loss
                + args.loop_cot_weight * cot_loss
                + args.workspace_query_weight * workspace_query_loss
                + answer_loss + args.logit_loss_weight * logit_loss
                + args.hidden_loss_weight * hidden_loss
                + args.process_loss_weight * process_loss
                + args.delta_loss_weight * delta_loss
            )
            scaled_loss = ((1 - args.nonloop_loss_weight) if use_nonloop else 1) * loss / group_size
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
            scaled_loss.backward()
            if not use_loop_teacher:
                student_base.model.loop_hidden_states = []
                student_base.model.loop_updates = []
            student_loop_capture.clear()
            suffix_capture.clear()
            kv_capture.clear()
            if use_loop_cot or use_query:
                student_base.model.loop_workspace_states = []
            epoch_suffix_sum += suffix_loss.detach()
            epoch_kv_sum += kv_loss.detach()
            epoch_cot_sum += cot_loss.detach()
            nonloop_answer_loss = answer_loss.new_zeros(())
            nonloop_logit_loss = answer_loss.new_zeros(())
            if use_nonloop:
                from loop_jev.looping import without_looping
                # Release loop activations before the second forward. Both
                # branches use the same student parameters and teacher target.
                del student_outputs
                with without_looping(student_base), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    nonloop_outputs = student(
                        **student_batch, logits_to_keep=1, output_hidden_states=False,
                    )
                nonloop_logits = nonloop_outputs.logits[:, -1, :].float()
                nonloop_options = nonloop_logits[:, candidate_ids].masked_fill(
                    ~valid_options, torch.finfo(nonloop_logits.dtype).min,
                )
                nonloop_answer_loss = F.cross_entropy(nonloop_options, targets)
                if teacher_probs is not None:
                    nonloop_log_probs = F.log_softmax(nonloop_logits / temperature, dim=-1)
                    if args.logit_loss_type == "kl":
                        nonloop_logit_loss = F.kl_div(
                            nonloop_log_probs, teacher_probs, reduction="batchmean",
                        )
                    else:
                        nonloop_logit_loss = -(teacher_probs * nonloop_log_probs).sum(dim=-1).mean()
                    nonloop_logit_loss = nonloop_logit_loss * (temperature ** 2)
                nonloop_loss = nonloop_answer_loss + args.logit_loss_weight * nonloop_logit_loss
                if not torch.isfinite(nonloop_loss):
                    raise FloatingPointError(f"Non-finite nonloop loss at epoch={epoch + 1}, batch={batch_index + 1}")
                (args.nonloop_loss_weight * nonloop_loss / group_size).backward()
                del nonloop_outputs, nonloop_logits, nonloop_options
            epoch_nonloop_sums += torch.stack([
                nonloop_answer_loss.detach(), nonloop_logit_loss.detach(),
            ])
            # Finish direct-answer backward before allocating text-branch activations.
            text_loss = answer_loss.new_zeros(())
            if not use_nonloop:
                del student_outputs
            del teacher_logits, teacher_hidden, teacher_process_states
            if use_text and text_stage < 3:
                from loop_jev.training.progressive_text import collate_trajectories, trajectory_loss
                text_batch, text_targets, text_owners = collate_trajectories(
                    [item[text_stage] for item in trajectories], tokenizer.pad_token_id,
                    args.text_max_targets, local_rank, text_generator,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    text_outputs = student(**text_batch, output_hidden_states=False)
                text_loss = trajectory_loss(text_outputs.logits, text_targets, text_owners, len(trajectories))
                if not torch.isfinite(text_loss):
                    raise FloatingPointError(f"Non-finite text loss at update {update_step + 1}")
                (args.progressive_text_weight * text_loss / group_size).backward()
                student_base.model.loop_hidden_states = []
                student_base.model.loop_updates = []
                del text_outputs, text_batch, text_targets, text_owners
            epoch_text_sum += text_loss.detach()
            loss = ((1 - args.nonloop_loss_weight) * loss.detach()
                    + args.nonloop_loss_weight * (nonloop_answer_loss.detach()
                    + args.logit_loss_weight * nonloop_logit_loss.detach())
                    + args.progressive_text_weight * text_loss.detach())
            epoch_sums += torch.stack([
                answer_loss.detach(), logit_loss.detach(), hidden_loss.detach(), loss.detach(),
                process_loss.detach(), delta_loss.detach(),
            ])
            if sync_now:
                update_step += 1
                grad_norm = clip_grad_norm_mixed(parameters_for_clipping, 1.0)
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
                        suffix=f"{suffix_loss.item():.5f}",
                        kv=f"{kv_loss.item():.5f}",
                        cot=f"{cot_loss.item():.5f}",
                        text=f"{text_loss.item():.5f}",
                        direct_ce=f"{nonloop_answer_loss.item():.5f}" if use_nonloop else "-",
                        direct_kl=f"{nonloop_logit_loss.item():.5f}" if use_nonloop else "-",
                        total=f"{loss.item():.5f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.8g}",
                    )

        progress.close()
        dist.all_reduce(epoch_nonloop_sums)
        dist.all_reduce(epoch_text_sum)
        dist.all_reduce(epoch_suffix_sum)
        dist.all_reduce(epoch_kv_sum)
        dist.all_reduce(epoch_cot_sum)
        dist.all_reduce(epoch_sums, op=dist.ReduceOp.SUM)
        mean_losses = (epoch_sums / (len(loader) * world_size)).tolist()
        mean_suffix_loss = (epoch_suffix_sum / (len(loader) * world_size)).item()
        mean_kv_loss = (epoch_kv_sum / (len(loader) * world_size)).item()
        mean_cot_loss = (epoch_cot_sum / (len(loader) * world_size)).item()
        mean_text_loss = (epoch_text_sum / (len(loader) * world_size)).item()
        mean_nonloop = (epoch_nonloop_sums / (len(loader) * world_size)).tolist()
        completed_epochs = epoch + 1
        peak_memory = torch.tensor([
            torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.max_memory_reserved() / 2**30,
        ], device=local_rank, dtype=torch.float64)
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        training_history.append(dict(zip(
            ("ce", "kl", "hidden", "total", "process", "delta"), mean_losses,
        ), epoch=completed_epochs, suffix=mean_suffix_loss, kv=mean_kv_loss, text=mean_text_loss,
            loop_cot=mean_cot_loss,
            nonloop_ce=mean_nonloop[0], nonloop_kl=mean_nonloop[1],
            peak_allocated_gib=peak_memory[0].item(), peak_reserved_gib=peak_memory[1].item()))
        if use_query:
            dist.all_reduce(epoch_query_sum)
            dist.all_reduce(epoch_query_stats)
            *per_round, masked, count = epoch_query_stats.tolist()
            query_metrics = dict(
                loss=(epoch_query_sum / (len(loader) * world_size)).item(),
                exact_by_round=[correct / count if count else None for correct in per_round],
                exact_without_workspace=masked / count if count else None,
                queries=count,
            )
            training_history[-1]["workspace_query"] = query_metrics
            if rank == 0:
                print(f"workspace_query epoch={epoch + 1} {json.dumps(query_metrics)}", flush=True)
        if rank == 0:
            print(
                f"text={mean_text_loss:.6f} suffix={mean_suffix_loss:.6f} "
                f"kv={mean_kv_loss:.6f} loop_cot={mean_cot_loss:.6f} epoch={epoch + 1} "
                f"mean_nonloop_ce={mean_nonloop[0]:.6f} mean_nonloop_kl={mean_nonloop[1]:.6f} "
                f"mean_ce={mean_losses[0]:.6f} "
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
        if use_query:
            torch.save(query_readout.module.state_dict(), args.output / "workspace_query.pt")
        metadata = {
            "model": args.model,
            "model_revision": getattr(student_base.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "training_seconds": training_seconds,
            "training_history": training_history,
            "loop_start_layer": args.loop_start_layer,
            "loop_end_layer": args.loop_end_layer,
            "loop_count": args.loop_count,
            "loop_parameter_sharing": True if not use_loop_teacher else None,
            "loop_damping": args.loop_damping,
            "loop_carry_workspace": args.loop_carry_workspace,
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
            "teacher": (args.loop_teacher_model if use_loop_teacher else args.model)
            if use_teacher else None,
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
            "nonloop_loss_weight": args.nonloop_loss_weight,
            "latent_slots": args.latent_slots,
            "latent_marker_token_id": getattr(student_base.config, "latent_marker_token_id", None),
            "kv_loss_weight": args.kv_loss_weight,
            "loop_cot_weight": args.loop_cot_weight,
            "loop_cot_order": args.loop_cot_order if use_loop_cot else None,
            "loop_cot_tokens": args.loop_cot_tokens if use_loop_cot else None,
            "workspace_query_weight": args.workspace_query_weight,
            "workspace_query_coverage": query_coverage,
            "kv_layer_0_indexed": args.kv_layer if use_kv else None,
            "kv_cache": str(args.kv_cache) if use_kv else None,
            "kv_objective": "cosine alignment of pre-RoPE normalized K and V at latent workspace positions, final loop only" if use_kv else None,
            "nonloop_objective": "same question-only input and full-rationale teacher target as final-loop; weighted CE+KL mixture, sequential backward" if use_nonloop else None,
            "student_input": (f"question and options, {args.latent_slots} latent marker tokens, then Answer: ("
                              if args.latent_slots else "question and options only, followed by Answer: ("),
            "teacher_input": ("question and options only, followed by Answer: (; recurrent loop teacher"
                              if use_loop_teacher else
                              "question, options, and full training rationale, followed by Answer: ("),
            "hidden_distillation": "cross-context cosine alignment at the next-answer-token position for evenly spaced transformer layers",
            "hidden_layers_1_indexed": hidden_layers,
            "hidden_layer_selection": args.hidden_layer_selection,
            "hidden_loss_weight": args.hidden_loss_weight,
            "process_loss_weight": args.process_loss_weight,
            "process_loss_discount": args.process_loss_discount,
            "process_distillation": (
                "cosine alignment of each damped recurrent-block answer-slot state to "
                "student answer-slot outputs at evenly spaced deeper layers; weighted mean"
                if use_loop_teacher and args.process_loss_weight > 0 else
                "cosine alignment of each loop's final-normalized answer-slot state to "
                "a frozen teacher reading ceil(k * N / K) complete rationale steps; "
                "line boundaries with sentence fallback; final anchor reuses full teacher; "
                "empty rationales masked; weighted mean over loops and valid examples"
                if use_process else None
            ),
            "progressive_text_weight": args.progressive_text_weight,
            "text_max_targets": args.text_max_targets,
            "text_stage_history": text_stage_history,
            "text_objective": "Per-example token-mean CE on explicit rationale suffix and answer, uniform target subsampling; full/last 2 thirds/last third/none in four update quarters; direct CE+KL always active; sequential backward" if use_text else None,
            "suffix_loss_weight": args.suffix_loss_weight,
            "suffix_max_tokens": args.suffix_max_tokens,
            "suffix_layers_1_indexed": suffix_layers,
            "suffix_curriculum": "1/2/3 depth states predict full/final two thirds/final third of steps; no optimizer reset" if use_suffix else None,
            "student_inference": "single non-looped forward pass" if use_loop_teacher or use_suffix or use_text else None,
            "loop_teacher_student_layers_0_indexed": student_loop_layers if use_loop_teacher else None,
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
